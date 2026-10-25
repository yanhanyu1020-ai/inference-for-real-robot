"""Standalone field bundle and HTTP/WebSocket isolation checks; no real hardware."""
from contextlib import contextmanager
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import numpy as np
import pytest

from inference.real_robot.gateway import Gateway, handler
from inference.scripts.package_field_client import build
from inference.tests.test_real_robot import ARM, FakeHardware


@pytest.fixture
def bundle(tmp_path):
    archive = tmp_path/'client.zip'
    build(archive)
    with zipfile.ZipFile(archive) as package:
        package.extractall(tmp_path)
    return tmp_path/'field-client'


def run_client(bundle, *arguments):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    return subprocess.run([sys.executable, '-m', 'real_robot.field_client', *arguments],
                          cwd=bundle, env=environment, capture_output=True, text=True, timeout=30)


def test_bundle_is_standalone_and_contains_only_manifested_client_files(bundle):
    manifest = json.loads((bundle/'MANIFEST.json').read_text())
    files = {str(path.relative_to(bundle)) for path in bundle.rglob('*') if path.is_file()}
    assert files == set(manifest) | {'MANIFEST.json'}
    for name, digest in manifest.items():
        assert hashlib.sha256((bundle/name).read_bytes()).hexdigest() == digest
    assert not (bundle/'token').exists()
    assert not any('jax' in name or 'gateway.py' in name or 'h100_hph' in name for name in files)
    result = run_client(bundle, '--help')
    assert result.returncode == 0, result.stderr
    assert 'execute' in result.stdout and 'check' in result.stdout
    assert json.loads((bundle/'controller.json').read_text())['policy'] == 'ws://127.0.0.1:48080'


@contextmanager
def field_servers(execute):
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve
    gateways = {}; servers = []; calls = {'unitree': [], 'p450': []}; observations = []
    metadata = dict(physical_action_dim=14, action_horizon=15, fps=15,
                    vit_lora_rank=8, vit_lora_alpha=16)
    def policy(connection):
        connection.send(msgpack_numpy.packb(metadata))
        for message in connection:
            observation = msgpack_numpy.unpackb(message)
            observations.append(observation)
            actions = np.zeros((15,14)); actions[:,3:10] = ARM
            connection.send(msgpack_numpy.packb({'actions': actions}))
    websocket = serve(policy, '127.0.0.1', 0, compression=None, max_size=8*1024*1024)
    policy_thread = threading.Thread(target=websocket.serve_forever, daemon=True)
    policy_thread.start()
    config = dict(policy='ws://127.0.0.1:'+str(websocket.socket.getsockname()[1]),
                  execute_steps=3, prompt='test task')
    token = 'test-field-client-token-32-characters'
    try:
        for kind in ('unitree', 'p450'):
            gateway = Gateway(FakeHardware(kind), execute=execute and kind == 'unitree')
            gateways[kind] = gateway
            base = handler(gateway, token)
            def audited(base, kind):
                class Handler(base):
                    def dispatch(self):
                        calls[kind].append((self.command, self.path))
                        super().dispatch()
                return Handler
            server = ThreadingHTTPServer(('127.0.0.1', 0), audited(base, kind))
            servers.append(server)
            threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True).start()
            config[kind] = 'http://127.0.0.1:'+str(server.server_port)
        yield config, token, gateways, calls, observations
    finally:
        for server in servers:
            server.shutdown(); server.server_close()
        for gateway in gateways.values():
            gateway.close()
        websocket.shutdown(); policy_thread.join(timeout=2)


loopback = pytest.mark.skipif(os.environ.get('HETEROVLA_LOOPBACK_TEST') != '1',
                              reason='Explicit local socket permission required')


@loopback
def test_packaged_check_and_infer_use_wire_protocol_without_actuation(bundle):
    with field_servers(execute=False) as (config, token, gateways, calls, observations):
        (bundle/'controller.json').write_text(json.dumps(config))
        (bundle/'token').write_text(token)
        result = run_client(bundle, 'check')
        assert result.returncode == 0, result.stdout+result.stderr
        assert '"passed": true' in result.stdout
        assert all(method == 'GET' for entries in calls.values() for method, path in entries)
        result = run_client(bundle, 'infer')
        assert result.returncode == 0, result.stdout+result.stderr
        assert '"executed": false' in result.stdout
        assert len(observations) == 1
        assert observations[0]['state'].shape == (30,)
        assert len(observations[0]['image']) == 3
        assert all(not gateway.hardware.sent for gateway in gateways.values())
        assert all(path not in ('/prepare', '/commit', '/stop')
                   for entries in calls.values() for method, path in entries)


@loopback
def test_packaged_execute_controls_unitree_and_preserves_p450_as_observer(bundle):
    with field_servers(execute=True) as (config, token, gateways, calls, observations):
        (bundle/'controller.json').write_text(json.dumps(config))
        (bundle/'token').write_text(token)
        result = run_client(bundle, 'execute')
        assert result.returncode == 0, result.stdout+result.stderr
        assert '"executed": true' in result.stdout
        assert gateways['unitree'].hardware.sent
        assert not gateways['p450'].hardware.sent
        assert not gateways['p450'].hardware.stops
        assert gateways['p450'].plan is None
        assert len(observations) == 2  # Warm-up is discarded, then fresh inference.
        assert all(path not in ('/prepare', '/commit', '/stop') for method, path in calls['p450'])
        assert ('POST', '/stop') in calls['unitree']


@loopback
def test_packaged_stop_does_not_need_policy_or_p450(bundle):
    with field_servers(execute=True) as (config, token, gateways, calls, observations):
        config.update(p450='http://127.0.0.1:1', policy='ws://127.0.0.1:1')
        (bundle/'controller.json').write_text(json.dumps(config))
        (bundle/'token').write_text(token)
        result = run_client(bundle, 'stop')
        assert result.returncode == 0, result.stdout+result.stderr
        assert calls['unitree'] == [('POST', '/stop')]
        assert not calls['p450']
        assert not observations
