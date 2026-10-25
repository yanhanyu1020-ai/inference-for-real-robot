"""Run the three-camera client on the field PC; only Unitree can execute."""
import argparse
import json
from pathlib import Path
import sys
import time

from .controller import Device, Policy, observer_status


def check(config, token):
    """Read-only gateway clock and remote model connectivity check; no reset or motion."""
    devices = {name: Device(config[name], token) for name in ('unitree', 'p450')}
    report = dict(passed=False, execution_targets=['unitree'], observation_only=['p450'],
                  robot_commands_sent=False, gateways={}, errors=[])
    for name, device in devices.items():
        try:
            status = device.calibrate()
            report['gateways'][name] = dict(status=status,
                clock_uncertainty_ms=device.uncertainty*1000)
        except Exception as exc:
            report['gateways'][name] = dict(error=str(exc))
            report['errors'].append(name+': '+str(exc))
    try:
        observer_status(devices, {'unitree': devices['unitree']})
    except Exception as exc:
        report['errors'].append(str(exc))
    started = time.monotonic()
    try:
        policy = Policy(config['policy'])
        try:
            report['policy'] = dict(url=config['policy'], metadata=policy.metadata,
                                    connect_s=time.monotonic()-started)
        finally:
            policy.close()
    except Exception as exc:
        report['errors'].append('policy: '+str(exc))
    status = report['gateways'].get('unitree', {}).get('status', {})
    report['unitree_sensor_error'] = status.get('sensor_error')
    report['passed'] = not report['errors']
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'infer', 'dry-run', 'execute', 'stop'))
    parser.add_argument('--config', type=Path, default=Path('controller.json'))
    parser.add_argument('--token-file', type=Path, default=Path('token'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--cycles', type=int, default=1)
    args = parser.parse_args(argv)
    if args.cycles < 1:
        parser.error('--cycles must be positive')
    config = json.loads(args.config.read_text())
    token = args.token_file.read_text().strip()
    if args.mode == 'check':
        report = check(config, token)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print('Check tests clock/connectivity only; use infer to measure the complete '
              'image-upload/model/response path. It does not certify motion readiness.', flush=True)
        if not report['passed']:
            raise SystemExit(1)
        return
    if args.mode == 'stop':
        result = Device(config['unitree'], token).call('/stop', {})
        print(json.dumps(dict(execution_targets=['unitree'], stopped=result),
                         ensure_ascii=False), flush=True)
        return
    arguments = ['--config', str(args.config), '--token-file', str(args.token_file),
                 '--output', str(args.output), '--cycles', str(args.cycles)]
    if args.mode == 'infer':
        from . import diagnostic
        entrypoint = diagnostic.main
    else:
        from . import controller
        entrypoint = controller.main
        arguments += ['--execution-target', 'unitree']
        if args.mode == 'execute':
            arguments.append('--execute')
    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        entrypoint()
    finally:
        sys.argv = original


if __name__ == '__main__':
    main()
