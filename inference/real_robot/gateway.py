"""Authenticated device-local observation, prepare/commit, and deadman service."""
import argparse
import hmac
import json
import math
import os
import signal
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .core import DT, action_plan, angle_delta, finite


class Gateway:
    def __init__(self, hardware, execute=False, velocity_scale=.3, max_action_age=.75):
        if not 0 < velocity_scale <= 1 or not 0 < max_action_age <= 1:
            raise ValueError('Velocity scale must be (0,1]; action age must be (0,1] seconds')
        self.hardware = hardware
        self.kind = hardware.sensors.kind
        self.execute = execute
        self.velocity_scale = velocity_scale
        self.max_action_age = max_action_age
        self.session = None
        self.lock = threading.RLock()
        self.snapshots = OrderedDict()
        self.plan = None
        self.fault = None
        self.cancel = threading.Event()
        self.closed = threading.Event()
        self.active = False
        self.worker = None
        self.last_tick = time.monotonic()
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        self.watchdog.start()

    def status(self):
        with self.lock:
            return dict(kind=self.kind, monotonic=time.monotonic(), session=self.session,
                        capabilities=['validate_actions_v1'],
                        execution_enabled=self.execute, fault=self.fault, active=self.active,
                        sensor_error=self.hardware.sensors.error,
                        execution=None if self.plan is None else {
                            k: v for k, v in self.plan.items() if k not in ('rows', 'snapshot')})

    def reset(self):
        with self.lock:
            if self.active or self.fault:
                raise RuntimeError('Active/faulted gateway cannot reset; restart after inspecting the fault')
            self.hardware.sensors.reset()
            self.session = uuid.uuid4().hex
            self.snapshots.clear()
            self.plan = None
            return self.status()

    def observe(self):
        with self.lock:
            if self.fault or self.session is None:
                raise RuntimeError(self.fault or 'Reset the episode first')
            snapshot = self.hardware.sensors.snapshot()
            snapshot.update(observation_id=uuid.uuid4().hex, session=self.session)
            self.snapshots[snapshot['observation_id']] = {
                k: v for k, v in snapshot.items() if k != 'images'}
            while len(self.snapshots) > 32:
                self.snapshots.popitem(last=False)
            return snapshot

    def prepare(self, request):
        with self.lock:
            if not self.execute or self.fault or self.active:
                raise RuntimeError('Gateway is observe-only, faulted, or executing')
            if request.get('session') != self.session or self.session is None:
                raise ValueError('Wrong episode session')
            if self.plan and self.plan['status'] in ('prepared', 'scheduled', 'running'):
                raise RuntimeError('A plan is already pending')
            snapshot = self.snapshots.get(request.get('observation_id'))
            if snapshot is None:
                raise ValueError('Unknown/consumed observation id')
            if not 0 <= time.monotonic()-snapshot['stamp'] <= self.max_action_age:
                raise RuntimeError('Model actions expired; inference result will not execute')
            self.hardware.check_execution()
            current = self.hardware.sensors.snapshot()
            old_state, state = snapshot['state'], current['state']
            if sum((a-b)**2 for a, b in zip(state[:2], old_state[:2])) > .1**2:
                raise RuntimeError('Robot moved while inference was running')
            yaw_change = angle_delta(math.atan2(old_state[5], old_state[6]), math.atan2(state[5], state[6]))
            if abs(yaw_change) > .15 or (self.kind == 'p450' and abs(state[2]-old_state[2]) > .08):
                raise RuntimeError('Robot orientation/altitude changed while inference was running')
            rows = action_plan(request['actions'], self.kind,
                               current_arm=state[10:17] if self.kind == 'unitree' else None,
                               velocity_scale=self.velocity_scale)
            self.snapshots.pop(request['observation_id'])
            self.plan = dict(plan_id=uuid.uuid4().hex, status='prepared', rows=rows,
                             snapshot=snapshot, prepared_at=time.monotonic(), executed_steps=0)
            return dict(plan_id=self.plan['plan_id'], status='prepared', applied_actions=rows.tolist())

    def diagnose(self):
        with self.lock:
            if self.execute or self.active:
                raise RuntimeError('Diagnostic observations require an observe-only gateway')
            # No session/observation nonce is issued or stored: this data cannot prepare actions.
            return self.hardware.sensors.snapshot(diagnostic=True)

    def commit(self, request):
        with self.lock:
            if self.fault or self.active or self.plan is None or self.plan['status'] != 'prepared':
                raise RuntimeError('No committable plan')
            if request.get('plan_id') != self.plan['plan_id']:
                raise ValueError('Wrong plan id')
            start = float(request['start_at'])
            now = time.monotonic()
            if not math.isfinite(start) or not .025 <= start-now <= .4:
                raise ValueError('Start time must be 25..400 ms in the future')
            if start-self.plan['snapshot']['stamp'] > self.max_action_age:
                raise RuntimeError('Actions will be stale at scheduled execution time')
            self.hardware.check_execution()
            self.plan.update(status='scheduled', scheduled_start=start)
            self.cancel.clear()
            self.active = True
            self.last_tick = now
            self.worker = threading.Thread(target=self._run, args=(self.plan,), daemon=True)
            self.worker.start()
            return dict(plan_id=self.plan['plan_id'], status='scheduled')

    def validate_actions(self, request):
        """Receive and check a real action payload without creating an execution plan."""
        with self.lock:
            if self.execute or self.active:
                raise RuntimeError('Action validation requires an observe-only gateway')
            request_id = request.get('request_id')
            if not isinstance(request_id, str) or len(request_id) != 32:
                raise ValueError('Expected a 32-character request id')
            values = request.get('actions')
            if not isinstance(values, list) or not 1 <= len(values) <= 15:
                raise ValueError('Expected 1..15 action rows')
            rows = finite(values, (len(values), 10 if self.kind == 'unitree' else 4))
            result = dict(kind=self.kind, request_id=request_id, received=True,
                          received_actions=rows.tolist(), received_at=time.monotonic(),
                          executed=False, executable=False, validation_only=True,
                          validation_passed=False, candidate_actions=None,
                          execution_readiness_checked=False, error=None)
            try:
                # Diagnostic snapshot retains grounded/unarmed state. No actuator methods,
                # observation credential, plan id, or worker are created by this endpoint.
                snapshot = self.hardware.sensors.snapshot(diagnostic=True)
                result['readiness_issues'] = snapshot.get('readiness_issues', [])
                result['feedback_stamp'] = snapshot['stamp']
                candidate = action_plan(rows, self.kind,
                    current_arm=snapshot['state'][10:17] if self.kind == 'unitree' else None,
                    velocity_scale=self.velocity_scale)
                result.update(validation_passed=True, candidate_actions=candidate.tolist())
            except (ValueError, RuntimeError) as exc:
                result['error'] = str(exc)
            return result

    def _run(self, plan):
        try:
            start = plan['scheduled_start']
            while time.monotonic() < start:
                self.last_tick = time.monotonic()
                if self.cancel.wait(.005):
                    return
            if time.monotonic()-start > .03:
                raise RuntimeError('Missed execution start deadline')
            self.hardware.check_execution()
            plan.update(status='running', actual_start=time.monotonic())
            previous = -1
            while not self.cancel.is_set():
                now = time.monotonic()
                index = int((now-start)/DT)
                if index >= len(plan['rows']):
                    break
                if index > previous+1:
                    raise RuntimeError('Control loop missed an action interval')
                self.hardware.send(plan['rows'][index])
                self.last_tick = time.monotonic()
                plan['executed_steps'] = index+1
                previous = index
                self.cancel.wait(.005)
            if self.cancel.is_set():
                return
            self.hardware.hold()
            with self.lock:
                if not self.fault:
                    plan.update(status='done', finished_at=time.monotonic())
        except Exception as exc:
            self.stop(str(exc))
        finally:
            with self.lock:
                self.active = False

    def _watch(self):
        while not self.closed.wait(.02):
            if self.active and time.monotonic()-self.last_tick > .2:
                self.stop('Gateway control loop watchdog expired')

    def stop(self, reason='Operator/controller stop'):
        with self.lock:
            self.fault = self.fault or str(reason)
            self.cancel.set()
            if self.plan and self.plan['status'] != 'done':
                self.plan.update(status='fault', error=self.fault)
        try:
            if self.execute:
                self.hardware.halt()
        except Exception as exc:
            with self.lock:
                self.fault += '; stop error: ' + str(exc)
        return self.status()

    def close(self):
        self.stop('Gateway shutdown')
        self.closed.set()
        if self.worker is not None and self.worker is not threading.current_thread():
            self.worker.join(timeout=2)
        self.hardware.close()


def handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def dispatch(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+token):
                self.send_error(403)
                return
            parsed = urlparse(self.path)
            try:
                if self.command == 'GET' and parsed.path == '/status':
                    result = service.status()
                elif self.command == 'GET' and parsed.path == '/observation':
                    result = service.observe()
                elif self.command == 'GET' and parsed.path == '/preview':
                    result = service.hardware.sensors.preview()
                    result['hardware_fault'] = getattr(service.hardware, 'fault', None)
                    result['mode'] = getattr(service.hardware, 'mode', None)
                    result['callback_errors'] = getattr(service.hardware, 'callback_errors', {})
                elif self.command == 'GET' and parsed.path == '/diagnostic':
                    result = service.diagnose()
                elif self.command == 'POST':
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 65536:
                        raise ValueError('Invalid request size')
                    body = json.loads(self.rfile.read(size))
                    if parsed.path == '/reset':
                        result = service.reset()
                    elif parsed.path == '/prepare':
                        result = service.prepare(body)
                    elif parsed.path == '/commit':
                        result = service.commit(body)
                    elif parsed.path == '/validate_actions':
                        result = service.validate_actions(body)
                    elif parsed.path == '/stop':
                        result = service.stop()
                    else:
                        raise ValueError('Unknown endpoint')
                else:
                    raise ValueError('Unknown endpoint')
                encoded = json.dumps(result, allow_nan=False).encode()
                self.send_response(200)
            except Exception as exc:
                if parsed.path in ('/prepare', '/commit'):
                    service.stop(str(exc))
                encoded = json.dumps(dict(error=str(exc))).encode()
                self.send_response(409)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch()

        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--token-file', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    token = args.token_file.expanduser().read_text().strip() if args.token_file else os.environ.get('HETEROVLA_ROBOT_TOKEN', '')
    if len(token) < 24:
        parser.error('Set HETEROVLA_ROBOT_TOKEN to a shared random token of at least 24 characters')
    if config['kind'] == 'unitree':
        from .unitree import UnitreeHardware
        hardware = UnitreeHardware(config, args.execute)
    elif config['kind'] == 'p450':
        from .p450 import P450Hardware
        hardware = P450Hardware(config, args.execute)
    else:
        parser.error('Unknown hardware kind')
    service = Gateway(hardware, args.execute, config.get('velocity_scale', .3),
                      config.get('max_action_age_s', .75))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', config['port']), handler(service, token))
    except Exception:
        service.close()
        raise
    def terminate(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    print(json.dumps(dict(listening='127.0.0.1:%s' % config['port'], kind=config['kind'],
                          execute=args.execute)), flush=True)
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        service.close()
        server.server_close()


if __name__ == '__main__':
    main()
