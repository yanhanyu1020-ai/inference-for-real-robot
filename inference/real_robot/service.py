"""Device-local lifecycle for the gateway process started by this package."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        directory = Path('/proc')/str(pid)
        stat = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        command = (directory/'cmdline').read_bytes().split(b'\0')
        return dict(pid=pid, start_ticks=stat[19],
                    command=[part.decode(errors='replace') for part in command if part],
                    cwd=str((directory/'cwd').resolve()))
    except (OSError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('start', 'stop', 'status', 'assert-stopped'))
    parser.add_argument('--kind', choices=('unitree', 'p450'), required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    directory = Path.home()/'.config/heterovla-live'
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory/('service_'+args.kind+'.json')
    old = json.loads(record_path.read_text()) if record_path.exists() else None
    alive = old is not None and identity(old['pid']) == old['identity']
    if args.operation == 'assert-stopped':
        if alive:
            raise RuntimeError('Stop the existing gateway before replacing its source files')
        return
    if args.operation == 'status':
        print(json.dumps(dict(running=alive, record=old, log=str(root/'gateway.log'))))
        return
    if args.operation == 'stop':
        if alive:
            if 'real_robot.gateway' not in old['identity']['command'] or old['identity']['cwd'] != str(root):
                raise RuntimeError('Refusing to stop a process outside this deployment')
            os.kill(old['pid'], signal.SIGTERM)
            deadline = time.monotonic()+8
            while identity(old['pid']) == old['identity'] and time.monotonic() < deadline:
                time.sleep(.1)
            if identity(old['pid']) == old['identity']:
                raise RuntimeError('Gateway still stopping; inspect gateway.log and hardware state')
        print(json.dumps(dict(stopped=True, was_running=alive)))
        return
    if alive:
        raise RuntimeError('Gateway already running; stop it before changing execution mode')
    token = directory/'token'
    if not token.is_file():
        raise FileNotFoundError(token)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if args.kind == 'unitree':
        environment.pop('LD_LIBRARY_PATH', None)
        environment.pop('CYCLONEDDS_URI', None)
        environment['PYTHONPATH'] = str(root)
    else:
        environment['PYTHONPATH'] = str(root)+os.pathsep+environment.get('PYTHONPATH', '')
    command = [sys.executable, '-m', 'real_robot.gateway', '--config',
               str(root/'real_robot/config'/(args.kind+'.json')), '--token-file', str(token)]
    if args.execute:
        command.append('--execute')
    with (root/'gateway.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    time.sleep(.2)
    ident = identity(process.pid)
    if ident is None or process.poll() is not None:
        raise RuntimeError('Gateway exited during startup; inspect '+str(root/'gateway.log'))
    record_path.write_text(json.dumps(dict(pid=process.pid, identity=ident, execute=args.execute), indent=2)+'\n')
    print(json.dumps(dict(started=True, pid=process.pid, execute=args.execute, log=str(root/'gateway.log'))))


if __name__ == '__main__':
    main()
