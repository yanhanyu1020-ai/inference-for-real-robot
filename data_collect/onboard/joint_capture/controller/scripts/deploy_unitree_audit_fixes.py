"""Install reviewed modules only while all production capture owners are stopped.

Run on Unitree after staging the files. Keeps the production storage format unchanged.
"""
import fcntl
import json
from pathlib import Path
import shutil
import sys
import hashlib

root = Path('/home/unitree/heterovla-collection')
stage = root / '.release-20260927-fixes'
backup = root / 'backups/20260927-audit-fixes'
sys.path.insert(0, str(root / 'onboard'))
from session_support import owned_process

with (root / 'run/operation.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'run/active_episode').exists():
        raise RuntimeError('active episode: refusing deployment')
    for name, script in [('camera', 'camera_session.py'), ('gamepad', 'piper_gamepad_teleop.py')]:
        if owned_process(root / ('run/session/' + name + '.pid'), root / ('onboard/' + script)):
            raise RuntimeError('warm process active: ' + name)
    if backup.exists():
        raise RuntimeError('backup already exists; inspect previous deployment first')
    backup.mkdir(parents=True)
    shutil.copytree(root / 'onboard', backup / 'onboard')
    shutil.copytree(root / 'config', backup / 'config')
    result = {}
    for source in sorted(stage.glob('*.py')):
        if source.name == 'deploy_audit.py':
            continue
        destination = root / 'onboard' / source.name
        temporary = destination.with_suffix('.py.installing')
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        result[source.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(json.dumps({'backup': str(backup), 'sha256': result, 'format_changed': False}))
