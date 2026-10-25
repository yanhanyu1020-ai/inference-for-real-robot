"""Scheduler tests use synthetic GPU inventory; never launch the real trainer."""
import fcntl
import json
import subprocess
import sys

import pytest

from . import watch_gpu as w


IDLE = dict(uuid='GPU-test', total_mib=81920, free_mib=81000, has_compute_process=False)
BUSY = {**IDLE, 'has_compute_process': True}


def test_requires_memory_and_no_compute_process():
    assert w.available(IDLE)
    assert not w.available(BUSY)
    assert not w.available({**IDLE, 'free_mib': 70000})
    assert not w.available({**IDLE, 'total_mib': 40000, 'free_mib': 40000})


@pytest.mark.parametrize('returncode', [0, 7])
def test_one_attempt_after_busy_then_two_idle_checks(tmp_path, monkeypatch, returncode):
    scans = iter([[BUSY], [IDLE], [IDLE], [IDLE]])
    calls = []
    monkeypatch.setattr(w.time, 'sleep', lambda _: None)

    def launch(argv, **kwargs):
        calls.append(argv)
        return subprocess.Popen([sys.executable, '-c', f'print("test trainer"); exit({returncode})'], **kwargs)

    w.watch(30, queue=tmp_path, probe=lambda: next(scans), verify=lambda: None, launch=launch)
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['status'] == ('complete' if returncode == 0 else 'failed')
    assert state['returncode'] == returncode
    assert calls == [['bash', 'train/V3/relative14/train.sh', '--gpu', 'GPU-test']]
    assert 'test trainer' in (tmp_path / 'training.log').read_text()
    with pytest.raises(RuntimeError, match='attempt already exists'):
        w.watch(30, queue=tmp_path, probe=lambda: pytest.fail('must not probe'))
    assert len(calls) == 1


def test_lost_gpu_and_probe_failure_do_not_launch(tmp_path, monkeypatch):
    scans = iter([[IDLE], [IDLE], [BUSY], subprocess.TimeoutExpired('nvidia-smi', 20), [BUSY]])
    count = 0

    def probe():
        nonlocal count
        count += 1
        result = next(scans)
        if count == 5:
            (tmp_path / 'stop_requested').touch()
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(w.time, 'sleep', lambda _: None)
    w.watch(30, queue=tmp_path, probe=probe, verify=lambda: None,
            launch=lambda *a, **kw: pytest.fail('busy GPU must not launch'))
    assert json.loads((tmp_path / 'state.json').read_text())['status'] == 'stopped'
    assert not (tmp_path / 'attempt.json').exists()


def test_launch_failure_keeps_permanent_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(w.time, 'sleep', lambda _: None)

    def fail(*a, **kw):
        raise OSError('simulated launch failure')

    with pytest.raises(OSError, match='simulated'):
        w.watch(30, queue=tmp_path, probe=lambda: [IDLE], verify=lambda: None, launch=fail)
    assert (tmp_path / 'attempt.json').exists()
    with pytest.raises(FileExistsError):
        w.claim_attempt(tmp_path, IDLE)
    assert json.loads((tmp_path / 'state.json').read_text())['status'] == 'failed'


def test_singleton_lock(tmp_path):
    with (tmp_path / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already active'):
            w.watch(30, queue=tmp_path, probe=lambda: pytest.fail('must not probe'))


def test_preflight_rejects_changed_sources_and_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(w.c, 'MODELS', tmp_path)
    monkeypatch.setattr(w.c, 'PROJECT', tmp_path)
    monkeypatch.setattr(w, 'RUN', tmp_path / 'run')
    monkeypatch.setattr(w, 'CHECKPOINT', tmp_path / 'checkpoint')
    source = tmp_path / 'example.py'
    source.write_text('original')
    proof = tmp_path / 'V3/relative14/preflight.json'
    w.atomic_json(proof, dict(passed=True, gpu_used=False, source_hashes={'example.py': w.sha256(source)}))
    w.validate()
    source.write_text('changed')
    with pytest.raises(RuntimeError, match='Source changed'):
        w.validate()
    source.write_text('original')
    w.RUN.mkdir()
    with pytest.raises(FileExistsError, match='already exists'):
        w.validate()
