import os
import subprocess
from types import SimpleNamespace

import pytest

from jointctl.remote import run_process
from jointctl.clock_sync import ClockProbe
from jointctl.models import EpisodeManifest, EpisodeState, RemoteStatus
from tests.test_start_controller import _Harness, _status


@pytest.mark.skipif(os.name != 'nt', reason='Windows process contract')
def test_background_commands_do_not_create_console(monkeypatch):
    seen = []
    def run(argv, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(returncode=0, stdout='ok', stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    assert run_process(['ssh', 'p450', 'status'], 1).stdout == 'ok'
    assert seen[0].get('creationflags', 0) & subprocess.CREATE_NO_WINDOW
    assert seen[0].get('stdin') == subprocess.DEVNULL


@pytest.mark.skipif(os.name != 'nt', reason='Windows process contract')
def test_clock_ssh_is_hidden_and_noninteractive(monkeypatch):
    seen = {}
    def popen(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return 'process'
    monkeypatch.setattr(subprocess, 'Popen', popen)
    assert ClockProbe('unitree')._start_process() == 'process'
    assert seen.get('creationflags', 0) & subprocess.CREATE_NO_WINDOW
    assert 'BatchMode=yes' in seen['argv']


def test_readiness_recovers_from_transient_query_failure(tmp_path, monkeypatch):
    h = _Harness(tmp_path)
    rows = iter([
        {'p450': _status('p450', active=True, episode_id='ep', progress=1),
         'unitree': RemoteStatus('unitree', False, 'unknown', last_error='timeout')},
        *[ {host: _status(host, active=True, episode_id='ep', progress=n)
             for host in ('p450', 'unitree')} for n in (2, 3, 4)]])
    monkeypatch.setattr(h.controller, '_statuses', lambda: next(rows))
    assert h.controller._wait_until_ready('ep')['unitree'].progress_value == 4


def test_dead_starter_is_not_reported_as_still_starting(tmp_path, monkeypatch):
    h = _Harness(tmp_path)
    h.store.create(EpisodeManifest(episode_id='ep', label='test', mode='test',
                                  created_desktop_ns=1, metadata={'starter_pid': 999}))
    monkeypatch.setattr(h.controller, 'pid_is_running', lambda pid: False)
    report = h.controller.status()
    assert report.state == EpisodeState.PARTIAL
    assert 'interrupted' in report.message
    assert report.timing_degraded
    assert h.store.active().t0_desktop_ns is None


def test_readiness_retry_is_bounded(tmp_path, monkeypatch):
    h = _Harness(tmp_path)
    calls = []
    def statuses():
        calls.append(1)
        return {host: RemoteStatus(host, False, 'unknown', last_error='timeout')
                for host in ('p450', 'unitree')}
    monkeypatch.setattr(h.controller, '_statuses', statuses)
    with pytest.raises(RuntimeError, match='timeout'):
        h.controller._wait_until_ready('ep')
    assert len(calls) == 11


def test_readiness_never_retries_foreign_episode(tmp_path, monkeypatch):
    h = _Harness(tmp_path)
    monkeypatch.setattr(h.controller, '_statuses', lambda: {
        host: _status(host, active=True, episode_id='foreign')
        for host in ('p450', 'unitree')})
    with pytest.raises(RuntimeError, match='another episode'):
        h.controller._wait_until_ready('ep')
    assert h.sleep_calls == 0


def test_preflight_transient_failure_retries_before_single_start(tmp_path, monkeypatch):
    h = _Harness(tmp_path)
    original = h.controller._statuses
    calls = []
    def statuses():
        calls.append(1)
        if len(calls) == 1:
            return {'p450': _status('p450', active=False, episode_id=None),
                    'unitree': RemoteStatus('unitree', False, 'unknown', last_error='timeout')}
        return original()
    monkeypatch.setattr(h.controller, '_statuses', statuses)
    manifest = h.controller.start('test', 'test')
    assert manifest.state == EpisodeState.RECORDING
    assert h.p450.start_calls == h.unitree.start_calls == 1


def test_preflight_persistent_failure_does_not_start(tmp_path, monkeypatch):
    h = _Harness(tmp_path)
    calls = []
    def statuses():
        calls.append(1)
        return {host: RemoteStatus(host, False, 'unknown', last_error='timeout')
                for host in ('p450', 'unitree')}
    monkeypatch.setattr(h.controller, '_statuses', statuses)
    with pytest.raises(RuntimeError, match='preflight failed'):
        h.controller.start('test', 'test')
    assert len(calls) == 3
    assert h.p450.start_calls == h.unitree.start_calls == 0
    assert h.store.active() is None


def test_rollback_retries_transient_status_before_stopping_owned_episode(tmp_path, monkeypatch):
    h = _Harness(tmp_path)
    episode_id = "joint_retry"
    rows = iter([
        {
            "p450": _status("p450", active=True, episode_id=episode_id),
            "unitree": RemoteStatus("unitree", False, "unknown", last_error="timeout"),
        },
        {
            host: _status(host, active=True, episode_id=episode_id)
            for host in ("p450", "unitree")
        },
    ])
    monkeypatch.setattr(h.controller, "_statuses", lambda: next(rows))

    statuses, results, diagnostics = h.controller._rollback(episode_id)

    assert statuses["unitree"].episode_id == episode_id
    assert results["p450"].ok and results["unitree"].ok
    assert h.p450.stop_calls == h.unitree.stop_calls == 1
    assert diagnostics == []


def test_start_emits_progress_before_remote_initialization(tmp_path):
    h = _Harness(tmp_path)
    events = []
    h.controller.progress = lambda text: events.append((text, h.p450.start_calls))
    h.controller.start('test', 'test')
    assert any('连接' in text and starts == 0 for text, starts in events)
    assert any('准备' in text and starts == 0 for text, starts in events)
    assert any('对时' in text for text, _ in events)
