"""Owned processes, warm devices and retryable per-episode recording."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from camera_session import request
from session_support import (atomic_json, episode_path, owned_process, read_json,
                             save_pid, stop_owned, sync_episode_csv,
                             sync_episode_parents, clear_active_marker, sync_directory)


class Collection:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.root = self.config_path.parent.parent
        self.config = read_json(self.config_path)
        self.run = self.root / 'run'
        self.session = self.run / 'session'
        self.onboard = self.root / 'onboard'
        self.run.mkdir(exist_ok=True)
        self.session.mkdir(exist_ok=True)
        (self.session / 'raw').mkdir(exist_ok=True)
        self.children = []

    def active(self):
        marker = self.run / 'active_episode'
        return episode_path(marker.read_text().strip(), self.config['data_root']) if marker.exists() else None

    @contextmanager
    def lock(self):
        import fcntl
        with (self.run / 'operation.lock').open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('another device operation is running; wait and retry')
            yield

    def spawn(self, name, argv, log, env=None, warm=False):
        pid_file = (self.session if warm else self.run) / (name + '.pid')
        with Path(log).open('ab') as output:
            child = subprocess.Popen([str(a) for a in argv], stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=output, env=env, start_new_session=True)
        self.children.append(child)
        save_pid(pid_file, child)
        return child

    def camera_status(self):
        if not owned_process(self.session / 'camera.pid', self.onboard / 'camera_session.py'):
            return None
        return request(self.session / 'camera.sock', 'status', timeout=3)

    def gamepad_alive(self):
        return owned_process(self.session / 'gamepad.pid', self.onboard / 'piper_gamepad_teleop.py', self.session)

    def gamepad_snapshot(self):
        value = read_json(self.session / 'raw/piper_gamepad_snapshot.json', {})
        age = (time.monotonic_ns() - value.get('monotonic_ns', 0)) / 1e9
        if not self.gamepad_alive() or not 0 <= age < 3:
            return {'gamepad': {'connected': False}, 'arm_enabled': None,
                    'snapshot_stale': True}
        return value

    def prepare(self):
        if self.active():
            raise RuntimeError('finish the active episode before initialization')
        subprocess.run(['sudo', '-n', '/bin/systemctl', 'start', 'joint-can0.service'], check=True)
        # Refuse foreign CAN writers; only our registered warm gamepad is allowed.
        expected = self.gamepad_alive()
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit():
                continue
            try:
                args = (folder / 'cmdline').read_bytes().decode().split('\0')
            except (OSError, UnicodeError):
                continue
            if any(Path(a).name in ('piper_gamepad_teleop.py', 'piper_chunk_loop',
                                   'piper_stream_loop', 'hetero_teleop_loop', 'hetero_teach_loop')
                   or ('Gamepad_PiPER' in a and a.endswith('/main.py')) for a in args):
                if not expected or int(folder.name) != expected['pid']:
                    raise RuntimeError('another Piper controller is running: PID ' + folder.name)
        if not owned_process(self.session / 'camera.pid', self.onboard / 'camera_session.py'):
            self.spawn('camera', [sys.executable, self.onboard / 'camera_session.py',
                                 '--config', self.config_path, '--session-dir', self.session],
                       self.session / 'camera.log', warm=True)
        deadline = time.monotonic() + 20
        health = None
        while time.monotonic() < deadline:
            try:
                health = self.camera_status()
                if health and health['prepared']:
                    break
                if health and any(c['error'] for c in health['cameras'].values()):
                    atomic_json(self.session / 'prepare_failure.json', health)
                    # Exited camera readers can be retried by next initialization.
                    if not health['running']:
                        stop_owned(self.session / 'camera.pid', self.onboard / 'camera_session.py')
                    raise RuntimeError('camera initialization failed: ' + json.dumps(health['cameras']))
            except (FileNotFoundError, ConnectionRefusedError):
                pass
            time.sleep(.2)
        else:
            if health:
                atomic_json(self.session / 'prepare_failure.json', health)
                if not health.get('running'):
                    stop_owned(self.session / 'camera.pid', self.onboard / 'camera_session.py')
            raise RuntimeError('cameras not ready; check USB connections: ' + str(health))
        if not expected:
            atomic_json(self.session / 'segment.json', {'episode': None})
            env = {**os.environ, 'SDL_VIDEODRIVER': 'dummy',
                   'PYTHONPATH': self.config['piper_gamepad']['runtime']}
            self.spawn('gamepad', [self.config['piper_gamepad']['python'],
                       self.onboard / 'piper_gamepad_teleop.py', '--config', self.config_path,
                       '--session-dir', self.session], self.session / 'gamepad.log', env, warm=True)
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline:
            if not self.gamepad_alive():
                raise RuntimeError('gamepad process exited; inspect run/session/gamepad.log')
            if not self.gamepad_snapshot().get('snapshot_stale'):
                (self.session / 'prepare_failure.json').unlink(missing_ok=True)
                return {'prepared': True, 'cameras': health['cameras'],
                        'gamepad': self.gamepad_snapshot()['gamepad']}
            time.sleep(.2)
        raise RuntimeError('gamepad initialization timeout; retry initialization')

    def select_gamepad(self, episode):
        selected = str(episode) if episode else None
        atomic_json(self.session / 'segment.json', {'episode': selected})
        if not self.gamepad_alive():
            if selected:
                raise RuntimeError('gamepad process is unavailable')
            return
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            value = self.gamepad_snapshot()
            if not value.get('snapshot_stale') and value.get('recording_episode') == selected:
                return
            time.sleep(.05)
        raise RuntimeError('gamepad log switch not acknowledged; retry stop')

    def start(self, episode_id, instruction, task):
        for value in (episode_id, task):
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
                raise ValueError('invalid episode or task name')
        if self.active():
            raise RuntimeError('an episode still needs stop confirmation')
        self.prepare()
        directory = episode_path(Path(self.config['data_root']) / task / episode_id, self.config['data_root'])
        directory.mkdir(parents=True, exist_ok=False)
        for name in ('raw', 'frames', 'logs'):
            (directory / name).mkdir()
        format_meta = {}
        if self.config.get('format_version', 1) == 2:
            format_meta = {'format_version': 2, 'image_storage': 'mcap',
                'codec': self.config.get('raw_codec', 'lz4'), 'pixel_format': 'bgr8',
                'boot_id': self.config.get('boot_id') or Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'software': 'heterovla-raw/2 mcap-python/1.3.0', 'durable_complete': False}
        atomic_json(directory / 'meta.json', {**format_meta, 'episode_id': episode_id, 'task': task,
                    'instruction': instruction, 'start_wall_time_ns': time.time_ns(),
                    'host': platform.node(), 'collection_config': self.config, 'warm_session': True})
        (self.run / 'active_episode').write_text(str(directory) + '\n')
        try:
            self.select_gamepad(directory)
            self.spawn('go2', [self.root / 'build/go2_state_bridge',
                       self.config['network_interface'], directory / 'raw'],
                       directory / 'logs/go2_state_bridge.log',
                       {'HOME': os.environ['HOME'], 'PATH': '/usr/local/bin:/usr/bin:/bin'})
            request(self.session / 'camera.sock', 'start', str(directory))
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                state = read_json(directory / 'status.json', {})
                health = self.camera_status()
                if not health or not health['running']:
                    raise RuntimeError('recorder exited: ' + str(health))
                ready = (state.get('ready', False) if self.config.get('format_version', 1) == 2
                         else state.get('frames_saved', 0) > 0)
                if ready:
                    print('collection started: ' + str(directory))
                    print('Piper gamepad remains available; HOME is required to enable')
                    return
                time.sleep(.2)
            raise RuntimeError('no complete frames; check camera, CAN and Go2 state')
        except Exception as exc:
            atomic_json(directory / 'start_error.json', {'error': str(exc)})
            try:
                self.stop()
            except Exception as cleanup:
                raise RuntimeError(str(exc) + '; cleanup pending: ' + str(cleanup)) from exc
            raise

    def stop(self):
        directory = self.active()
        if directory is None:
            print('no collection is active')
            return
        meta = read_json(directory / 'meta.json')
        if not isinstance(meta, dict):
            raise RuntimeError('missing episode metadata; ownership needs inspection')
        errors = []
        # A warm session keeps control/cameras alive; stop only the selected writer.
        if meta.get('warm_session'):
            try:
                health = self.camera_status()
                if health is not None:
                    request(self.session / 'camera.sock', 'stop', str(directory))
            except Exception as exc:
                errors.append(str(exc))
            # If camera flush is pending, retain state sources until the retry.
            if not errors:
                try:
                    self.select_gamepad(None)
                except Exception as exc:
                    errors.append(str(exc))
        else:
            for name, script, suffix, wait in (
                ('recorder', 'hetero_pkl_recorder.py', directory, 60),
                ('gamepad', 'piper_gamepad_teleop.py', directory, 15)):
                try:
                    stop_owned(self.run / (name + '.pid'), self.onboard / script, suffix, wait)
                except Exception as exc:
                    errors.append(str(exc))
        if not errors:
            try:
                stop_owned(self.run / 'go2.pid', self.root / 'build/go2_state_bridge', directory / 'raw')
            except Exception as exc:
                errors.append(str(exc))
        if not errors and meta.get('format_version', 1) == 1:
            try:
                # Legacy writers close/rename files but do not fsync. Do this
                # only after the writer acknowledged close, before releasing ownership.
                frames = directory / 'frames'
                if frames.is_dir():
                    for path in frames.glob('*.pkl'):
                        with path.open('rb') as handle:
                            os.fsync(handle.fileno())
                    sync_directory(frames)
                if (directory / 'raw').is_dir():
                    meta['synced_state_csv'] = sync_episode_csv(directory)
                sync_episode_parents(directory, self.config['data_root'])
            except Exception as exc:
                errors.append(str(exc))
        if not errors and meta.get('format_version') == 2:
            try:
                summary = read_json(directory / 'summary.json', {})
                if not summary.get('camera_durable_complete'):
                    raise RuntimeError('camera durability unconfirmed; recover before next episode')
                meta['synced_state_csv'] = sync_episode_csv(directory)
                sync_episode_parents(directory, self.config['data_root'])
                meta['durable_complete'] = True
                atomic_json(directory / 'summary.json', {**summary, 'durable_complete': True,
                    'synced_state_csv': meta['synced_state_csv']}, durable=True)
                manifest = read_json(directory / 'capture_manifest.json', {})
                atomic_json(directory / 'capture_manifest.json', {**manifest, 'durable_complete': True,
                    'synced_state_csv': meta['synced_state_csv']}, durable=True)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            atomic_json(directory / 'stop_error.json', {'errors': errors, 'at_ns': time.time_ns()})
            raise RuntimeError('; '.join(errors))
        meta.setdefault('stop_wall_time_ns', time.time_ns())
        meta['postprocess'] = 'pending'
        if not (directory / 'summary.json').exists():
            meta['incomplete_shutdown'] = True
        atomic_json(directory / 'meta.json', meta, durable=True)
        prior = read_json(directory / 'status.json', {})
        atomic_json(directory / 'status.json', {**prior, 'running': False,
                    **({'durable_complete': True} if meta.get('durable_complete') else {})},
                    durable=True)
        clear_active_marker(self.run / 'active_episode')
        if meta.get('format_version') == 2:
            print(json.dumps({**read_json(directory / 'summary.json', {}),
                              'raw_episode_id': directory.name}))
        print('collection stopped; raw data preserved for later validation: ' + str(directory))

    def status(self):
        directory = self.active()
        try:
            health = self.camera_status()
        except Exception as exc:
            health = {'prepared': False, 'error': str(exc)}
        if health is None:
            previous = read_json(self.session / 'prepare_failure.json')
            if previous:
                health = {**previous, 'prepared': False, 'running': False,
                          'error': 'last initialization failed; check cameras and retry initialization',
                          'cameras': {name: {**value, 'ready': False, 'age_s': None}
                                      for name, value in previous.get('cameras', {}).items()}}
        if directory:
            print('episode=' + str(directory))
            status = read_json(directory / 'status.json', {})
            warm = read_json(directory / 'meta.json', {}).get('warm_session')
            running = (bool(health and health.get('running') and health.get('episode') == str(directory))
                       if warm else bool(owned_process(self.run / 'recorder.pid',
                                         self.onboard / 'hetero_pkl_recorder.py', directory)))
            print(json.dumps({**status, 'running': running, 'cleanup_pending': not running}))
        else:
            print(json.dumps({'running': False}))
        if self.gamepad_alive():
            print(json.dumps(self.gamepad_snapshot()))
        else:
            print(json.dumps({'gamepad': {'connected': False}, 'arm_enabled': None,
                              'snapshot_stale': True}))
        print(json.dumps({'session_prepared': bool(health and health.get('prepared') and self.gamepad_alive()),
                          'camera_health': (health or {}).get('cameras', {}),
                          'session_error': ((health or {}).get('error')
                                            if not health or not health.get('prepared') else None)}))

    def finalize(self, directory):
        if self.active():
            raise RuntimeError('cannot validate while recording')
        subprocess.run([sys.executable, str(self.onboard / 'validate_episode.py'),
                        '--episode', str(episode_path(directory, self.config['data_root'])),
                        '--config', str(self.config_path), '--write'], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('command', choices=['prepare', 'start', 'stop', 'status', 'finalize'])
    parser.add_argument('values', nargs='*')
    args = parser.parse_args()
    ctl = Collection(args.config)
    try:
        if args.command == 'status':
            ctl.status()
        else:
            with ctl.lock():
                if args.command == 'start':
                    if not 2 <= len(args.values) <= 3:
                        raise ValueError('start requires episode, instruction, optional task')
                    ctl.start(*args.values, *(['default'] if len(args.values) == 2 else []))
                elif args.command == 'prepare':
                    print(json.dumps(ctl.prepare()))
                elif args.command == 'stop':
                    ctl.stop()
                else:
                    ctl.finalize(*args.values)
        return 0
    except Exception as exc:
        print('collection error: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
