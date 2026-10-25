"""Explicit on-device v2 trial activation; preserve the running gamepad owner."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/home/unitree/heterovla-collection')
sys.path.insert(0, str(ROOT / 'onboard'))
from collection_manager import Collection
from session_support import atomic_json, stop_owned

collection = Collection(ROOT / 'config/collection.json')
with collection.lock():
    if collection.active():
        raise RuntimeError('active capture; activation refused')
    health = collection.camera_status()
    if health and (health.get('running') or health.get('episode')):
        raise RuntimeError('camera episode still open')
    gamepad = collection.gamepad_alive()
    if not gamepad:
        raise RuntimeError('existing gamepad owner required; will not start a controller')
    source = ROOT / '.raw-capture-test-20260926/mcap'
    sys.path.insert(0, str(source.parent))
    import mcap
    import lz4.frame
    from mcap.writer import Writer
    if mcap.__version__ != '1.3.0':
        raise RuntimeError('unexpected MCAP version')
    sys.path.remove(str(source.parent))
    backup = ROOT / 'backups/raw-trial-20260927'
    backup.mkdir(exist_ok=False)
    shutil.copy2(collection.config_path, backup / 'collection.json')
    shutil.copytree(ROOT / 'onboard', backup / 'onboard')
    destination = ROOT / 'onboard/mcap'
    if destination.exists():
        raise RuntimeError('MCAP target already exists; inspect before retrying')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__'))
    stop_owned(collection.session / 'camera.pid', ROOT / 'onboard/camera_session.py', timeout=20)
    config = dict(collection.config, format_version=2, raw_codec='lz4', raw_queue_bytes=64*1024**2)
    atomic_json(collection.config_path, config)
    try:
        updated = Collection(collection.config_path)
        updated.spawn('camera', [sys.executable, updated.onboard / 'camera_session.py',
                      '--config', updated.config_path, '--session-dir', updated.session],
                      updated.session / 'camera.log', warm=True)
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            try:
                health = updated.camera_status()
            except (FileNotFoundError, ConnectionRefusedError):
                health = None
            if health and health.get('prepared'):
                break
            time.sleep(.2)
        else:
            raise RuntimeError('new camera owner not ready')
        if updated.gamepad_alive() != gamepad:
            raise RuntimeError('gamepad identity changed externally')
        print(json.dumps({'format_version': 2, 'backup': str(backup),
                          'camera': health, 'gamepad_pid_unchanged': gamepad['pid']}))
    except Exception:
        stop_owned(collection.session / 'camera.pid', ROOT / 'onboard/camera_session.py', timeout=20)
        atomic_json(collection.config_path, collection.config)
        collection.spawn('camera', [sys.executable, collection.onboard / 'camera_session.py',
                         '--config', collection.config_path, '--session-dir', collection.session],
                         collection.session / 'camera.log', warm=True)
        raise
