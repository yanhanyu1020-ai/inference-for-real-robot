"""Regressions for request-time stopping and continuous clock quality."""
import json
from dataclasses import replace

import pytest

from jointctl.cli import build_alignment_report, main, make_controller, _parser
from jointctl.manifest import ActiveEpisodeConflict
from jointctl.models import CommandResult, EpisodeState
from jointctl.inspectors import StreamSummary
from test_stop_recovery import _Harness
from test_report import _fixture, _sample


def test_stop_request_time_precedes_slow_ownership_check(tmp_path):
    h = _Harness(tmp_path)
    now = [100]
    h.controller.time_ns = lambda: now[0]
    original = h.controller._statuses
    def slow_statuses():
        now[0] = 900
        return original()
    h.controller._statuses = slow_statuses
    assert h.controller.stop().t1_desktop_ns == 100


def test_stop_retry_preserves_first_t1(tmp_path):
    h = _Harness(tmp_path)
    h.store.update('joint_active', state=EpisodeState.STOPPING, t1_desktop_ns=42)
    assert h.controller.stop().t1_desktop_ns == 42


def test_start_local_conflict_precedes_all_remote_work(tmp_path):
    h = _Harness(tmp_path)
    h.controller._statuses = lambda: pytest.fail('local conflict must precede SSH')
    with pytest.raises(ActiveEpisodeConflict):
        h.controller.start('demo', 'joint')
    assert len(list(tmp_path.glob('*/manifest.json'))) == 1


@pytest.mark.parametrize('command', ['start', 'recover'])
def test_cli_classifies_active_pointer_race(monkeypatch, capsys, command):
    class Controller:
        def start(self, *_): raise ActiveEpisodeConflict('another start won')
        def recover(self): raise ActiveEpisodeConflict('another start won')
    monkeypatch.setattr('jointctl.cli.make_controller', lambda _: Controller())
    args = ['start', '--instruction', 'demo', '--task', 'joint'] if command == 'start' else ['recover']
    assert main(args) == 4
    assert 'conflict' in capsys.readouterr().err


@pytest.mark.parametrize('code, closed, expected', [(0, False, 5), (255, True, 3), (-1, True, 3), (7, True, 6)])
def test_stop_cli_reports_partial_and_failure_provenance(tmp_path, monkeypatch, capsys, code, closed, expected):
    h = _Harness(tmp_path)
    h.controller._stops = lambda hosts=None: {'p450': CommandResult('stop', code), 'unitree': CommandResult('stop', 0)}
    h.controller._stop_monitor = lambda _: closed
    monkeypatch.setattr('jointctl.cli.make_controller', lambda _: h.controller)
    assert main(['stop']) == expected
    assert capsys.readouterr().err


@pytest.mark.parametrize('condition', ['fresh', 'stale', 'dead', 'error', 'rtt'])
def test_status_uses_latest_samples_and_monitor_health(tmp_path, monkeypatch, capsys, condition):
    h = _Harness(tmp_path)
    now = 100_000_000_000
    h.controller.time_ns = lambda: now
    h.controller.pid_is_running = lambda _: condition != 'dead'
    initial = {host: _sample(host, 1_000, offset_ns=100) for host in ('p450', 'unitree')}
    h.store.update('joint_active', clock_samples=initial, clock_estimates=h.controller._clock_estimates(initial))
    for host in initial:
        sample = replace(_sample(host, now - (30_000_000_000 if condition == 'stale' else 100_000_000), offset_ns=500),
                         rtt_ns=30_000_000 if condition == 'rtt' else 2_000_000)
        rows = [sample.to_dict()]
        if condition == 'error':
            rows.append({'kind': 'probe_error', 'host': host, 'message': 'timeout', 'at_wall_ns': now})
        (h.store.episode_dir('joint_active') / f'clock_{host}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    monkeypatch.setattr('jointctl.cli.make_controller', lambda _: h.controller)
    assert main(['status']) == (0 if condition == 'fresh' else 5)
    report = json.loads(capsys.readouterr().out)
    assert report['clock_estimates'][0]['offset_ns'] == 500
    assert report['clock_health']['p450']['sample_age_ns'] is not None
    assert report['timing_degraded'] == (condition != 'fresh')


def test_alignment_rejects_long_internal_gap_and_retains_errors(tmp_path):
    episode = _fixture(tmp_path)
    payload = json.loads((episode / 'manifest.json').read_text())
    payload.update(t0_desktop_ns=100_000_000_000, t1_desktop_ns=500_000_000_000)
    (episode / 'manifest.json').write_text(json.dumps(payload), encoding='utf-8')
    before = {}
    for host in ('p450', 'unitree'):
        rows = [replace(_sample(host, t), rtt_ns=2_000_000).to_dict() for t in (1_100, 600_000_001_100)]
        rows.insert(1, {'kind': 'probe_error', 'host': host, 'message': 'disconnected', 'at_wall_ns': 300_000_000_000})
        path = episode / f'clock_{host}.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        before[host] = path.read_bytes()
    report = build_alignment_report(episode, stream_summaries={
        host: [StreamSummary('stream', 10, 100_000_000_100, 500_000_000_100)] for host in before
    })
    assert report['quality']['degraded'] is True
    error = report['quality']['estimated_error_ns']
    assert error is None or error > 10_000_000
    assert report['clock_diagnostics']['p450']['errors'][0]['message'] == 'disconnected'
    assert any('gap' in reason for reason in report['quality']['degradation_reasons'])
    assert all((episode / f'clock_{host}.jsonl').read_bytes() == raw for host, raw in before.items())


def test_configured_aliases_keep_logical_probe_identity_and_monitor_settings(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'manifest_root': str(tmp_path / 'episodes'),
        'p450': {'host': 'capture-p450'}, 'unitree': {'host': 'capture-go2'},
        'clock_interval_s': 0.7, 'ssh_timeout_s': 3.0}), encoding='utf-8')
    c = make_controller(_parser().parse_args(['--config', str(config), 'status']))
    assert c.clock_probes['p450'].host == 'p450'
    assert c.clock_probes['p450'].destination == 'capture-p450'
    from jointctl.models import EpisodeManifest
    c.store.create(EpisodeManifest('joint_alias', 'demo', 'joint', 1))
    c.monitor_launcher = lambda *_: 123
    c._launch_monitor('joint_alias')
    settings = json.loads((c.store.episode_dir('joint_alias') / 'clock_monitor_config.json').read_text())
    assert settings['hosts'] == ['p450', 'unitree']
    assert settings['destinations'] == {'p450': 'capture-p450', 'unitree': 'capture-go2'}
    assert settings['clock_interval_s'] == 0.7
    assert settings['ssh_timeout_s'] == 3.0


@pytest.mark.parametrize('t0, t1', [(100_000_000_000, 110_000_000_000), (-110_000_000_000, -100_000_000_000)])
def test_requested_interval_far_outside_clock_coverage_withdraws_uncertainty(tmp_path, t0, t1):
    episode = _fixture(tmp_path)
    payload = json.loads((episode / 'manifest.json').read_text())
    payload.update(t0_desktop_ns=t0, t1_desktop_ns=t1)
    (episode / 'manifest.json').write_text(json.dumps(payload), encoding='utf-8')
    for host in ('p450', 'unitree'):
        with (episode / f'clock_{host}.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(_sample(host, 6_000_001_100).to_dict()) + '\n')
    report = build_alignment_report(episode, stream_summaries={
        host: [StreamSummary('stream', 10, t0 + 100, t1 + 100)] for host in ('p450', 'unitree')
    })
    assert report['quality']['degraded']
    assert report['quality']['estimated_error_ns'] is None
    assert any('coverage' in reason for reason in report['quality']['degradation_reasons'])


def test_custom_gap_threshold_controls_internal_gap_support(tmp_path):
    episode = _fixture(tmp_path)
    for host in ('p450', 'unitree'):
        with (episode / f'clock_{host}.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(_sample(host, 6_100).to_dict()) + '\n')
    streams = {host: [StreamSummary('stream', 10, 2_100, 5_100)] for host in ('p450', 'unitree')}
    supported = build_alignment_report(episode, stream_summaries=streams, alignment_window_ns=500, max_clock_gap_ns=10_000)
    unsupported = build_alignment_report(episode, stream_summaries=streams, alignment_window_ns=500, max_clock_gap_ns=1_000)
    # These generic streams test clock coverage, not the required camera set.
    assert supported['quality']['estimated_error_ns'] is not None
    assert not any('unsupported clock' in r for r in supported['quality']['degradation_reasons'])
    assert unsupported['quality']['estimated_error_ns'] is None


@pytest.mark.parametrize('unreachable, expected', [(False, 4), (True, 3)])
def test_status_ownership_and_connectivity_take_priority_over_dead_monitor(tmp_path, monkeypatch, unreachable, expected):
    h = _Harness(tmp_path)
    h.unitree.status_value = replace(h.unitree.status_value, episode_id='foreign', reachable=not unreachable)
    monkeypatch.setattr('jointctl.cli.make_controller', lambda _: h.controller)
    assert main(['status']) == expected


def test_monitor_uses_alias_transport_but_writes_logical_rows(tmp_path, monkeypatch):
    from jointctl import clock_sync
    from tests.fake_ssh import FakePersistentProcess
    seen = []
    stop_file = tmp_path / 'clock.stop'
    def fake_process(probe):
        seen.append((probe.host, probe.destination, probe.timeout_s))
        return FakePersistentProcess()
    original_append = clock_sync._append_line
    def append(path, row):
        original_append(path, row)
        stop_file.touch()
    monkeypatch.setattr(clock_sync.ClockProbe, '_start_process', fake_process)
    monkeypatch.setattr(clock_sync, '_append_line', append)
    clock_sync.run_monitor(tmp_path, ('p450',), 0.7, stop_file,
                           destinations={'p450': 'capture-p450'}, timeout_s=3.0)
    row = json.loads((tmp_path / 'clock_p450.jsonl').read_text().splitlines()[0])
    assert row['host'] == 'p450'
    assert seen == [('p450', 'capture-p450', 3.0)]
    assert not (tmp_path / 'clock_capture-p450.jsonl').exists()


def test_stderr_drain_reads_bounded_chunks():
    import queue
    from jointctl.clock_sync import ClockProbe
    class LongStderr:
        def __init__(self): self.remaining = 100_000
        def readline(self, *args): pytest.fail('unbounded lines can exhaust memory')
        def read(self, limit):
            assert 0 < limit <= 4096
            count = min(limit, self.remaining)
            self.remaining -= count
            return 'x' * count
    stream = LongStderr()
    ClockProbe._drain(stream, queue.Queue())
    assert stream.remaining == 0
