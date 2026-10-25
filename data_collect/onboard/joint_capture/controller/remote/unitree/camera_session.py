"""Resident dual-camera owner. Local requests select independent recordings."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time
from types import SimpleNamespace

from session_support import atomic_json, episode_path, read_json


class CameraSession:
    def __init__(self, config, cameras, recorder):
        self.config, self.cameras, self.recorder = config, cameras, recorder
        self.thread = None
        self.stop_event = threading.Event()
        self.episode = None
        self.error = None
        self._operation = threading.RLock()
        self._recording_status = {}

    def status(self):
        cameras = {}
        for name, camera in self.cameras.items():
            snapshot = camera.snapshot()
            age = ((time.monotonic_ns() - snapshot['monotonic_ns']) / 1e9
                   if snapshot else None)
            cameras[name] = {'serial': camera.serial, 'device': camera.device,
                             'age_s': age, 'error': camera.error,
                             'ready': age is not None and 0 <= age < 1 and not camera.error}
        running = bool(self.thread and self.thread.is_alive())
        return {**self._recording_status, 'running': running, 'episode': self.episode,
                'prepared': all(c['ready'] for c in cameras.values()),
                'cameras': cameras, 'error': self.error}

    def start(self, directory):
        with self._operation:
            return self._start(directory)

    def _start(self, directory):
        directory = episode_path(directory, self.config['data_root'])
        if self.thread and self.thread.is_alive():
            raise RuntimeError('a recording is already active')
        if self.episode is not None:
            raise RuntimeError('previous recording needs stop acknowledgement')
        health = self.status()
        if not health['prepared']:
            raise RuntimeError('camera not ready: ' + json.dumps(health['cameras']))
        if (directory / 'status.json').exists():
            raise RuntimeError('refusing to reuse an existing recording')
        self.episode = str(directory)
        self.stop_event = threading.Event()
        self.error = None
        self._recording_status = {}

        def record():
            try:
                if self.config.get('format_version', 1) == 2:
                    from raw_episode import run
                    def publish(value):
                        self._recording_status = value
                    code = run(SimpleNamespace(episode_dir=directory), self.config,
                               cameras=self.cameras, stop_event=self.stop_event, publish=publish)
                else:
                    code = self.recorder(SimpleNamespace(episode_dir=directory), self.config,
                                         cameras=self.cameras, stop_event=self.stop_event)
                if code:
                    self.error = self._recording_status.get('recording_error') or 'recorder failed; inspect summary.json'
            except Exception as exc:
                self.error = str(exc)
            finally:
                previous = read_json(directory / 'status.json', {})
                atomic_json(directory / 'status.json', {**previous, 'running': False,
                                                       'recording_error': self.error})
        self.thread = threading.Thread(target=record, daemon=True)
        self.thread.start()
        return self.status()

    def stop(self, directory):
        with self._operation:
            return self._stop(directory)

    def _stop(self, directory):
        directory = str(episode_path(directory, self.config['data_root']))
        if self.episode is not None and self.episode != directory:
            raise RuntimeError('cannot stop another episode')
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.config.get('camera_stop_timeout_s', 55))
            if self.thread.is_alive():
                raise RuntimeError('still saving raw data; retry stop')
        if self.config.get('format_version', 1) == 2 and self.episode is not None:
            if not self._recording_status.get('camera_durable_complete'):
                raise RuntimeError('camera/CSV durability unconfirmed; retry stop or recover')
        self.episode = None
        return self.status()


def request(socket_path, action, episode=None, timeout=65):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall((json.dumps({'action': action, 'episode': episode}) + '\n').encode())
        response = client.makefile('r').readline(1024 * 1024)
    value = json.loads(response)
    if not value.get('ok'):
        raise RuntimeError(value.get('error', 'camera service failed'))
    return value['status']


class CameraRequestPool:
    """Bounded local request handlers; status never joins the operation queue."""
    def __init__(self, session, capacity=8):
        self.session = session
        self.slots = threading.BoundedSemaphore(capacity)

    def submit(self, client):
        if not self.slots.acquire(blocking=False):
            client.close()
            return
        threading.Thread(target=self._handle, args=(client,), daemon=True).start()

    def _handle(self, client):
        try:
            with client:
                client.settimeout(65)
                try:
                    with client.makefile('r') as reader:
                        command = json.loads(reader.readline(8192))
                    action = command.get('action')
                    if action == 'start':
                        result = self.session.start(command['episode'])
                    elif action == 'stop':
                        result = self.session.stop(command['episode'])
                    elif action == 'status':
                        result = self.session.status()
                    else:
                        raise ValueError('unknown camera command')
                    reply = {'ok': True, 'status': result}
                except Exception as exc:
                    reply = {'ok': False, 'error': str(exc)}
                try:
                    client.sendall((json.dumps(reply) + '\n').encode())
                except OSError:
                    pass
        finally:
            self.slots.release()


def serve(config_path, session_dir):
    import fcntl
    from hetero_pkl_recorder import CameraReader, run
    config = read_json(config_path)
    session_dir.mkdir(parents=True, exist_ok=True)
    # Hold throughout the lifetime: an interrupted launcher cannot open cameras twice.
    lock = (session_dir / 'camera.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cameras = {name: CameraReader(name, serial, config['width'], config['height'], config['fps'])
               for name, serial in config['cameras'].items()}
    session = CameraSession(config, cameras, run)
    requests = CameraRequestPool(session)
    stopped = threading.Event()
    def terminate(*_):
        stopped.set()
        session.stop_event.set()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    socket_path = session_dir / 'camera.sock'
    socket_path.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(4)
        server.settimeout(.5)
        for camera in cameras.values():
            camera.start()
        try:
            while not stopped.is_set():
                try:
                    client, _ = server.accept()
                except socket.timeout:
                    continue
                requests.submit(client)
        finally:
            session.stop_event.set()
            if session.thread:
                session.thread.join(55)
            for camera in cameras.values():
                camera.stop()
            socket_path.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--session-dir', type=Path, required=True)
    args = parser.parse_args()
    serve(args.config, args.session_dir)
