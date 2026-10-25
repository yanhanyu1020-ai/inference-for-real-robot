"""Cached telemetry and durable asynchronous console operations; no HTTP dependencies."""
from argparse import Namespace
import copy
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .cli import make_controller
from .manifest import ManifestStore
from .operation_lock import operation_busy
from .postprocess import pending_episodes
from .remote import json_objects
from .telemetry import read_resources, safe_json


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def tail(path, count=8000):
    try:
        with Path(path).open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - count))
            return handle.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


class ConsoleState:
    def __init__(self, config_path, root, *, monotonic=time.monotonic, runner=None):
        self.config_path = Path(config_path).resolve()
        self.root = Path(root).resolve()
        self.store = ManifestStore(self.root)
        self.config = read_json(self.config_path)
        if self.config is None:
            raise ValueError('invalid console configuration')
        self.controller = make_controller(Namespace(config=self.config_path, manifest_root=self.root))
        # Respect the same SSH budget as the CLI; slow handshakes are not proof
        # that the recorder is offline. Probes are serialized per device below.
        self.monotonic = monotonic
        self.runner = runner or self._run_cli
        self.lock = threading.RLock()
        self.quit = threading.Event()
        self.threads = []
        self.worker = None
        self.progress = {}
        self._fault_stops = set()
        self.cache = {h: {k: {'value': None, 'success_at': None, 'success_mono': None,
                              'error': None} for k in ('status', 'resources')}
                      for h in ('p450', 'unitree')}
        self.jobs_root = self.root / '.console' / 'jobs'
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        jobs = sorted(self.jobs_root.glob('*/job.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        self.job = read_json(jobs[0]) if jobs else None
        if self.job and self.job.get('state') == 'running':
            self.job['state'] = 'interrupted'
            self.job['error'] = '后台曾中断；先核对远端状态，不会自动重放操作。'

    def _record(self, host, category, value=None, error=None):
        with self.lock:
            item = self.cache[host][category]
            item['error'] = error
            if error is None:
                item.update(value=value, success_at=time.time(), success_mono=self.monotonic())

    def sample_status(self, host, probe=None):
        began = self.monotonic()
        try:
            status = (probe or self.controller.remotes[host].status)()
            if not status.reachable or status.last_error:
                raise ConnectionError(status.last_error or '设备连接失败')
            value = status.to_dict()
            value['query_ms'] = (self.monotonic() - began) * 1000
            active, episode = self._episode()
            if (host == 'unitree' and active and episode and
                    status.episode_id == episode['episode_id'] and status.format_version == 2):
                for obj in json_objects(status.message):
                    if obj.get('format_version') == 2:
                        # Separate observation file avoids racing CLI manifest updates.
                        self.store._atomic_json(self.store.episode_dir(status.episode_id) / 'raw_health.json',
                            {**obj, 'raw_episode_id': status.episode_id, 'observed_at': time.time()})
            with self.lock:
                previous = self.progress.get(host)
                identity = (status.episode_id, status.progress_name)
                if (not previous or previous['identity'] != identity or
                        previous['value'] != status.progress_value or not status.active):
                    self.progress[host] = {'identity': identity, 'value': status.progress_value,
                                           'changed': self.monotonic()}
                value['progress_unchanged_s'] = self.monotonic() - self.progress[host]['changed']
            self._record(host, 'status', value)
            return True
        except Exception as exc:
            self._record(host, 'status', error=str(exc))
            return False

    def sample_resources(self, host):
        try:
            active, episode = self._episode()
            default = '/home/amov/p450_data' if host == 'p450' else '/home/unitree/heterovla-data'
            directory = (episode or {}).get('remote_directories', {}).get(host, default)
            value = read_resources(self.controller.remotes[host].host, directory)
            self._record(host, 'resources', value)
            return True
        except Exception as exc:
            self._record(host, 'resources', error=str(exc))
            return False

    def start_polling(self):
        if self.threads:
            return
        def loop(host):
            delay = 2
            next_resources = 0
            while not self.quit.is_set():
                ok = self.sample_status(host)
                self.check_capture_fault()
                # Avoid resource SSH competing with initialization/export and
                # back off disconnected hosts instead of opening more sessions.
                if ok and self.monotonic() >= next_resources and not operation_busy(self.root):
                    self.sample_resources(host)
                    next_resources = self.monotonic() + 10
                delay = 2 if ok else min(10, delay * 2)
                self.quit.wait(delay)
        for host in self.cache:
            thread = threading.Thread(target=loop, args=(host,), daemon=True)
            self.threads.append(thread)
            thread.start()

    def check_capture_fault(self):
        """Polling worker only: an explicit owned recorder fault triggers one Stop."""
        with self.lock:
            active, episode = self._episode()
            if not active or not episode or episode.get('state') != 'recording':
                return False
            episode_id = episode['episode_id']
            if episode_id in self._fault_stops:
                return False
            snap = self.snapshot()
            status = snap['hosts']['unitree']['status']
            if status['stale'] or status.get('episode_id') != episode_id or not status.get('fault'):
                return False
            if not snap['allowed']['stop']:
                return False
            self.submit('stop', {}, expected_episode=episode_id)
            self._fault_stops.add(episode_id)
            return True

    def close(self):
        self.quit.set()
        for thread in self.threads:
            thread.join(timeout=4)
        if self.worker:
            self.worker.join(timeout=4)

    def _episode(self):
        try:
            active = self.store.active()
            if active is not None:
                return True, active.to_dict()
        except (OSError, ValueError, TypeError):
            # Corrupt active pointers must not silently grant permission to start.
            return True, None
        episodes = [value for path in self.root.glob('*/manifest.json')
                    if (value := read_json(path)) and
                    type(value.get('created_desktop_ns')) in (int, str) and
                    str(value['created_desktop_ns']).isdigit()]
        return False, (max(episodes, key=lambda item: (int(item['created_desktop_ns']),
                                                    str(item.get('episode_id', ''))))
                       if episodes else None)

    def snapshot(self):
        with self.lock:
            cache = copy.deepcopy(self.cache)
            job = copy.deepcopy(self.job)
            running = self.worker is not None and self.worker.is_alive()
        active, episode = self._episode()
        starter_alive = None
        if active and episode and episode.get('state') == 'starting':
            try:
                starter_pid = int((episode.get('metadata') or {}).get('starter_pid', 0))
                starter_alive = starter_pid > 0 and self.controller.pid_is_running(starter_pid)
            except (TypeError, ValueError, OSError):
                starter_alive = False
        now = self.monotonic()
        hosts = {}
        for host, categories in cache.items():
            item = {}
            for category, record in categories.items():
                age = None if record['success_mono'] is None else max(0, now-record['success_mono'])
                # A successful ROS status command can itself take 3–5 seconds.
                # Allow two measured query durations without hiding old data
                # indefinitely; failed probes invalidate their cache immediately.
                freshness = (min(15, max(6, (record['value'] or {}).get('query_ms', 0) / 500 + 3))
                             if category == 'status' else 30)
                item[category] = {**(record['value'] or {}), 'age_s': age,
                    'updated_at': record['success_at'], 'stale': age is None or age > freshness or bool(record['error']),
                    'error': record['error']}
            status = item['status']
            raw_message = status.pop('message', '')
            if status['stale']:
                status.update(state='unknown', active=None, reachable=False)
            else:
                status['reachable'] = True
            status.setdefault('state', 'unknown')
            status.setdefault('active', None)
            item['write_stalled'] = (not status['stale'] and status.get('active') is True
                                     and bool(status.get('progress_name'))
                                     and status.get('progress_unchanged_s', 0) >= 15)
            item['path'] = (episode or {}).get('remote_directories', {}).get(host)
            item['path_is_current'] = active
            item['frames_dropped'] = None
            item['save_errors'] = None
            for obj in json_objects(raw_message):
                if host == 'unitree' and obj.get('format_version') == 2:
                    item['raw_capture'] = {key: obj.get(key) for key in ('format_version', 'streams',
                        'recording_state', 'quality_ok', 'fault', 'durable_complete', 'disk_available_bytes', 'remaining_minutes')}
                    item['raw_capture']['stale'] = status['stale']
                if host == 'p450' and 'components' in obj:
                    item['components'] = obj['components']
                    item['vehicle'] = obj.get('vehicle')
                if host == 'unitree' and 'gamepad' in obj:
                    item['gamepad'] = obj['gamepad']
                    item['arm_enabled'] = obj.get('arm_enabled')
                    item['arm_connected'] = obj.get('arm_connected')
                    item['motor_enabled'] = obj.get('motor_enabled') if not status['stale'] else None
                    item['feedback_error'] = obj.get('feedback_error') if not status['stale'] else None
                    item['command_inhibited'] = obj.get('command_inhibited') if not status['stale'] else None
                    item['teleop_error'] = obj.get('teleop_error') if not status['stale'] else None
                    item['teleop_mode'] = obj.get('up_level_mode') if not status['stale'] else None
                    for key in ('speed_factor', 'movement_speed'):
                        value = obj.get(key)
                        item[key] = (value if not status['stale'] and
                                     type(value) in (int, float) and
                                     0 < value <= (5 if key == 'speed_factor' else 100)
                                     else None)
                if host == 'unitree' and 'session_prepared' in obj:
                    item['session_prepared'] = obj['session_prepared'] if not status['stale'] else None
                    item['camera_health'] = obj.get('camera_health', {})
                    item['session_error'] = obj.get('session_error')
                for key in ('frames_dropped', 'save_errors'):
                    if type(obj.get(key)) is int:
                        item[key] = obj[key]
            if host == 'unitree' and episode and not active:
                saved = ((episode.get('metadata') or {}).get('raw_capture') or
                         read_json(self.store.episode_dir(episode['episode_id']) / 'raw_health.json'))
                if saved and saved.get('raw_episode_id') == episode['episode_id']:
                    item['raw_capture'] = {**saved, 'historical': True, 'stale': False}
                outputs = episode.get('stop_results', {}).get(host, {}).get('stdout', '')
                for obj in json_objects(outputs):
                    if 'warnings' in obj:
                        item['validation_warnings'] = obj['warnings']
            hosts[host] = item
        busy = running or operation_busy(self.root)
        own_id = (episode or {}).get('episode_id') if active else None
        foreign = any(not h['status']['stale'] and (
            h['status'].get('episode_id') not in (None, own_id)
            or (h['status'].get('active') and not h['status'].get('episode_id')))
            for h in hosts.values()) if active else False
        all_idle = all(not h['status']['stale'] and h['status'].get('active') is False
                       and not h['status'].get('episode_id')
                       for h in hosts.values())
        try:
            pending = pending_episodes(self.store)
        except (OSError, ValueError, TypeError):
            pending = []
        camera_blocked = (bool(hosts['unitree'].get('session_error')) or
                          any(c.get('ready') is False for c in
                              hosts['unitree'].get('camera_health', {}).values()))
        allowed = {'start': not busy and not active and all_idle and not camera_blocked,
                   'prepare': not busy and not active and all_idle,
                   'stop': not busy and active and episode is not None and not foreign,
                   'recover': not busy and (active or any(h['status'].get('active') or h['status'].get('episode_id') for h in hosts.values())),
                   'align': not busy and not active and episode is not None
                            and (episode.get('metadata') or {}).get('postprocess') not in {'pending', 'failed'},
                   'finalize': not busy and not active and all_idle and bool(pending)}
        validation = {'state': 'unavailable', 'report': None, 'path': None}
        if episode and not active:
            postprocess_state = (episode.get('metadata') or {}).get('postprocess')
            if postprocess_state in {'pending', 'failed'}:
                validation['state'] = postprocess_state
                validation['error'] = (episode.get('metadata') or {}).get('postprocess_error')
        clock = {'degraded': True, 'reasons': ['尚无活动采集的时钟探测'], 'estimated_error_ms': None}
        elapsed = None
        remaining = None
        if episode:
            t0, t1 = episode.get('t0_desktop_ns'), episode.get('t1_desktop_ns')
            if t0 is not None:
                elapsed = max(0, ((t1 if t1 is not None else time.time_ns())-t0)/1e9)
            if active and episode.get('start_results', {}).get('p450'):
                # Creation precedes the start request, so this is conservative.
                remaining = max(0, 1800-(time.time_ns()-episode['created_desktop_ns'])/1e9)
            report_path = self.store.episode_dir(episode['episode_id']) / 'alignment.json'
            report = read_json(report_path)
            if report and not active and postprocess_state not in {'pending', 'failed'}:
                quality = report.get('quality', {})
                validated = quality.get('data_validated') is True and quality.get('degraded') is False and bool(report.get('valid_interval'))
                validation = {'state': 'passed' if validated else 'failed', 'report': report, 'path': str(report_path)}
                clock = {'degraded': quality.get('degraded', True), 'reasons': quality.get('degradation_reasons', []),
                         'estimated_error_ms': quality.get('estimated_error_ns')/1e6 if quality.get('estimated_error_ns') is not None else None}
            if active:
                try:
                    manifest = self.store.load(episode['episode_id'])
                    estimates, health, reasons = self.controller._clock_health(manifest)
                    clock = {'degraded': bool(reasons), 'reasons': reasons, 'health': health,
                             'estimated_error_ms': sum(e.estimated_error_ns for e in estimates)/1e6 if len(estimates)==2 else None}
                except Exception as exc:
                    clock = {'degraded': True, 'reasons': [str(exc)], 'estimated_error_ms': None}
        if job:
            directory = self.jobs_root / job['id']
            job['log'] = '\n'.join(filter(None, [tail(directory/'stdout.log'), tail(directory/'stderr.log')]))
            if running and job['action'] in ('align','finalize') and episode and episode.get('state') == 'complete':
                job['phase'] = 'validating'
                validation['state'] = 'running'
            else:
                job['phase'] = job.get('state')
            if job.get('action') in ('align','finalize') and job.get('state') == 'failed':
                validation['state'] = 'failed'
            if job.get('action') in ('stop','align') and job.get('state') == 'interrupted':
                # The detached CLI may survive the server, but its completion
                # was not observed. Old reports and a released lock prove no
                # outcome: require an explicit fresh align operation.
                validation = {'state': 'unavailable', 'report': None, 'path': None}
                clock = {'degraded': True, 'estimated_error_ms': None,
                         'reasons': ['校验任务曾中断，结果未确认；等待操作结束后重新验收。']}
        result = {'service': 'joint-capture-console', 'updated_at': time.time(), 'active': active,
                  'hosts': hosts, 'episode': self._public_episode(episode), 'job': job, 'allowed': allowed,
                  'postprocess': {'pending_count': len(pending),
                                  'next_episode_id': pending[0].episode_id if pending else None},
                  'clock': clock, 'validation': validation, 'elapsed_s': elapsed, 'remaining_s': remaining,
                  'disk_warning_bytes': int(self.config.get('disk_warning_bytes', 10*1024**3)),
                  'operation_busy': busy, 'starter_alive': starter_alive,
                  'manifest_root': str(self.root)}
        return safe_json(result)

    @staticmethod
    def _public_episode(episode):
        if not episode:
            return None
        return {k: episode.get(k) for k in ('episode_id','label','mode','state','created_desktop_ns',
                't0_desktop_ns','t1_desktop_ns','remote_directories','diagnostics','clock_monitor_closed_cleanly','metadata')}

    def submit(self, action, payload, *, expected_episode=None):
        if action not in {'start','stop','recover','align','prepare','finalize'} or not isinstance(payload, dict):
            raise ValueError('unsupported action')
        fields = {'instruction','task'} if action == 'start' else set()
        if set(payload)-fields:
            raise ValueError('unexpected action fields')
        if action == 'start':
            payload = dict(payload)
            for key in fields:
                value = payload.get(key, '')
                if not isinstance(value, str) or len(value)>1000 or '\0' in value:
                    raise ValueError(f'invalid {key}')
                payload[key] = value.strip()
            if payload['task'] and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', payload['task']):
                raise ValueError('任务名只支持英文字母、数字、下划线、点和横线；中文备注请填采集说明。')
        with self.lock:
            if not self.snapshot()['allowed'][action]:
                raise RuntimeError('当前状态不允许此操作；请等待正在执行的任务，或检查设备状态。')
            job_id = f'{time.time_ns()}-{uuid.uuid4().hex[:8]}'
            directory = self.jobs_root/job_id
            directory.mkdir()
            self.job = {'id': job_id, 'action': action, 'state': 'running',
                        'started_at': time.time(), 'finished_at': None, 'exit_code': None, 'error': None}
            self._save_job()
            argv = [sys.executable, '-m', 'jointctl', '--config', str(self.config_path),
                    '--manifest-root', str(self.root), action]
            if action == 'start':
                if payload['instruction']:
                    argv += ['--instruction', payload['instruction']]
                if payload['task']:
                    argv += ['--task', payload['task']]
            if action == 'align':
                argv.append(self._episode()[1]['episode_id'])
            if action == 'stop' and expected_episode is not None:
                argv += ['--expected-episode', expected_episode]
            self.worker = threading.Thread(target=self._execute, args=(argv,directory), daemon=True)
            self.worker.start()
            return copy.deepcopy(self.job)

    def _save_job(self):
        ManifestStore._atomic_json(self.jobs_root/self.job['id']/'job.json', self.job)

    def _execute(self, argv, directory):
        error = None
        try:
            code = self.runner(argv, directory)
        except Exception as exc:
            code, error = -1, str(exc)
        with self.lock:
            self.job.update(state='succeeded' if code == 0 else 'failed', exit_code=code,
                            finished_at=time.time(), error=error)
            self._save_job()

    @staticmethod
    def _run_cli(argv, directory):
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH','')
        env['PYTHONIOENCODING'] = 'utf-8'
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (directory/'stdout.log').open('wb') as out, (directory/'stderr.log').open('wb') as err:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       env=env, creationflags=flags)
            # No outer timeout kills a still-finalizing operation. Each SSH
            # operation already has its own finite controller timeout.
            return process.wait()
