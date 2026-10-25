"""Deploy the live gateway bundle and open local forwards using the configured SSH hosts."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('probe', 'pack', 'deploy', 'start', 'stop', 'status', 'tunnel',
                                              'sensors-build', 'sensors-start', 'sensors-stop', 'sensors-status'))
    parser.add_argument('--ssh-config', type=Path, default=ROOT/'.cache/real_robot/ssh_config')
    parser.add_argument('--kind', choices=('unitree', 'p450', 'both'), default='both')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--unitree-python', default='/usr/bin/python3')
    parser.add_argument('--p450-python', default='/usr/bin/python3')
    parser.add_argument('--sdk-root', default='/home/unitree/unitree_sdk2-main')
    parser.add_argument('--p450-workspace', default='/home/amov/p450_experiment/devel/setup.bash')
    args = parser.parse_args()
    if args.operation.startswith('sensors-') and args.kind != 'p450':
        parser.error('P450 sensor operations require --kind p450')
    ssh = ['ssh', '-F', str(args.ssh_config)]
    kinds = ('unitree', 'p450') if args.kind == 'both' else (args.kind,)
    def remote(kind, command):
        subprocess.run([*ssh, 'v3-'+kind, command], check=True)
    def service_command(kind, operation, execute=False):
        python = args.unitree_python if kind == 'unitree' else args.p450_python
        command = ('cd "$HOME/heterovla-live" && ' if kind == 'unitree' else
                   'source /opt/ros/noetic/setup.bash && source '+shlex.quote(args.p450_workspace)+' && cd "$HOME/heterovla-live" && ')
        command += 'PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" '+shlex.quote(python)
        if operation.startswith('sensors-'):
            command += ' -m real_robot.p450_sensors '+operation[len('sensors-'):]
        else:
            command += ' -m real_robot.service '+operation+' --kind '+kind
        if execute and operation == 'start': command += ' --execute'
        return 'bash -lc '+shlex.quote(command)
    if args.operation == 'pack':
        destination = ROOT/'.cache/real_robot/live-bundle.tar.gz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, 'w:gz') as archive:
            base = ROOT/'inference/real_robot'
            for file in sorted(base.rglob('*')):
                if file.is_file() and not set(file.relative_to(base).parts) & {'__pycache__', 'build'}:
                    archive.add(file, arcname=str(Path('real_robot')/file.relative_to(base)), recursive=False)
        print(destination)
        return
    started = []
    for kind in kinds:
        if args.operation == 'probe':
            remote(kind, "hostname; uname -m; command -v python3; " +
                   ("test -d /home/unitree/unitree_sdk2-main && echo SDK_PRESENT; "
                    "ls -l "+shlex.quote(args.unitree_python)+"; ip -brief link show can0"
                    if kind == 'unitree' else
                    "test -f /opt/ros/noetic/setup.bash && echo ROS_PRESENT; "
                    "find /home/amov -maxdepth 4 -path '*/devel/setup.bash' -print 2>/dev/null"))
        elif args.operation == 'deploy':
            bundle = ROOT/'.cache/real_robot/live-bundle.tar.gz'
            token = ROOT/'.cache/real_robot/token'
            if not bundle.exists() or not token.exists():
                raise RuntimeError('Run pack first and create .cache/real_robot/token')
            python = args.unitree_python if kind == 'unitree' else args.p450_python
            remote(kind, 'if [ -f "$HOME/heterovla-live/real_robot/service.py" ]; then '
                   'cd "$HOME/heterovla-live" && PYTHONPATH="$PWD" '+shlex.quote(python)+
                   ' -m real_robot.service assert-stopped --kind '+kind+'; fi')
            remote(kind, 'mkdir -p "$HOME/heterovla-live" "$HOME/.config/heterovla-live"; chmod 700 "$HOME/.config/heterovla-live"')
            subprocess.run(['scp', '-F', str(args.ssh_config), str(bundle), 'v3-'+kind+':heterovla-live/live-bundle.tar.gz'], check=True)
            subprocess.run(['scp', '-F', str(args.ssh_config), str(token), 'v3-'+kind+':.config/heterovla-live/token'], check=True)
            remote(kind, 'set -e; chmod 600 "$HOME/.config/heterovla-live/token"; '
                   'cd "$HOME/heterovla-live" && '
                   'if [ -d real_robot ]; then cp -a real_robot "real_robot.backup.$(date +%s)"; fi; '
                   'tar -xmzf live-bundle.tar.gz')
            if kind == 'unitree':
                remote(kind, 'cd "$HOME/heterovla-live" && cmake -S real_robot -B real_robot/build '
                       '-DUNITREE_SDK2_ROOT='+shlex.quote(args.sdk_root)+' && cmake --build real_robot/build -j2')
        elif args.operation == 'tunnel':
            if args.kind == 'both':
                raise ValueError('Open each tunnel in a separate terminal with --kind unitree / --kind p450')
            local, remote_port = (18701, 8701) if kind == 'unitree' else (18702, 8702)
            os.execvp('ssh', [*ssh, '-NT', '-o', 'ExitOnForwardFailure=yes',
                             '-L', '127.0.0.1:%s:127.0.0.1:%s' % (local, remote_port), 'v3-'+kind])
        else:
            try:
                remote(kind, service_command(kind,args.operation,args.execute))
            except Exception:
                for previous in started:
                    try:remote(previous, service_command(previous,'stop'))
                    except Exception as exc:print('Startup rollback failed:',previous,str(exc))
                raise
            if args.operation == 'start':started.append(kind)


if __name__ == '__main__':
    main()
