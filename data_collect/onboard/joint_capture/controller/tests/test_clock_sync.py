import json
import io
import os
import queue
import subprocess
import time
from pathlib import Path

import pytest

from jointctl.clock_sync import ClockProbe, ClockProbeError, launch_detached_monitor, run_monitor_for_test, run_monitor
from jointctl.models import ClockSample
from tests.fake_ssh import FakePersistentProcess


def _sample(host, sequence):
    now = time.time_ns()
    mono = time.monotonic_ns()
    return ClockSample.from_exchange(host=host, sequence=sequence,
        local_send_wall_ns=now, local_send_mono_ns=mono,
        remote_receive_wall_ns=now, remote_send_wall_ns=now,
        remote_monotonic_ns=mono, local_receive_wall_ns=now,
        local_receive_mono_ns=mono)


@pytest.fixture
def fake_probe(monkeypatch):
    probe = ClockProbe("p450")
    starts = []
    def start():
        starts.append(None)
        return FakePersistentProcess(offset_ns=2_000_000, delay_ns=400_000)
    monkeypatch.setattr(probe, "_start_process", start)
    yield probe


def test_probe_reuses_one_process_for_multiple_samples(fake_probe):
    with fake_probe as probe:
        first = probe.sample()
        second = probe.sample()
    assert first.sequence == 0
    assert second.sequence == 1
    # Fake transport delays only the return leg. Its midpoint estimate can
    # therefore differ by up to half the measured wall-clock exchange duration;
    # a fixed 1 ms bound also tested Windows scheduling, not process reuse.
    span = first.local_receive_wall_ns - first.local_send_wall_ns
    assert abs(first.offset_ns - 2_000_000) <= span / 2 + 1
    assert probe.process_start_count == 1


def test_probe_uses_high_resolution_elapsed_clock_on_windows(fake_probe, monkeypatch):
    # Windows Python 3.12 monotonic can remain constant throughout a short RTT.
    monkeypatch.setattr(time, 'monotonic_ns', lambda: 500)
    ticks = iter((1_000_000, 3_000_000))
    monkeypatch.setattr(time, 'perf_counter_ns', lambda: next(ticks))
    with fake_probe as probe:
        sample = probe.sample()
    assert sample.rtt_ns == 2_000_000


def test_probe_malformed_response_is_reported_and_reconnects(monkeypatch):
    responses = [FakePersistentProcess(), FakePersistentProcess(offset_ns=9)]
    malformed = responses[0]
    malformed.write = lambda _line: malformed._responses.put("not a response\n") or 1
    probe = ClockProbe("p450")
    monkeypatch.setattr(probe, "_start_process", lambda: responses.pop(0))
    with pytest.raises(RuntimeError, match="malformed"):
        probe.sample()
    sample = probe.sample()
    # The fake responder is scheduled on another thread, so only its intended
    # offset (not an exact nanosecond exchange midpoint) is deterministic.
    assert abs(sample.offset_ns - 9) < 1_000_000
    assert probe.drain_errors()[0]["kind"] == "probe_error"


@pytest.mark.parametrize("returncode, response_timeout", [(255, False), (None, True)])
def test_probe_classifies_ssh_transport_exit_and_timeout_as_connectivity(monkeypatch, returncode, response_timeout):
    process = FakePersistentProcess()
    process.returncode = returncode
    if response_timeout:
        process.write = lambda _line: 1
    probe = ClockProbe("p450", timeout_s=0.01)
    monkeypatch.setattr(probe, "_start_process", lambda: process)
    with pytest.raises(ClockProbeError) as error:
        probe.sample()
    assert error.value.failure_kind == "connectivity"


@pytest.mark.parametrize("returncode, expected", [(255, "connectivity"), (7, "remote_command"),
                                                  (None, "connectivity")])
def test_probe_eof_waits_for_asynchronous_ssh_exit_code(monkeypatch, returncode, expected):
    class EofProcess:
        def __init__(self):
            self.stdin = self
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self.poll_calls = 0
            self.returncode = None
        def write(self, _line): return 1
        def flush(self): pass
        def poll(self):
            self.poll_calls += 1
            return self.returncode
        def wait(self, timeout=None):
            if returncode is None and self.returncode is None:
                raise subprocess.TimeoutExpired("ssh", timeout)
            self.returncode = returncode
            return returncode
        def terminate(self): self.returncode = -1
        def kill(self): self.returncode = -1
    process = EofProcess()
    probe = ClockProbe("p450")
    monkeypatch.setattr(probe, "_start_process", lambda: process)
    with pytest.raises(ClockProbeError) as error:
        probe.sample()
    assert error.value.failure_kind == expected
    assert process.poll_calls >= 2


def test_monitor_appends_valid_jsonl_and_stops_on_sample_limit(tmp_path):
    def fake_probe_factory(host):
        probe = ClockProbe(host)
        probe._start_process = lambda: FakePersistentProcess(offset_ns=100)  # type: ignore[method-assign]
        return probe

    result = run_monitor_for_test(tmp_path, tmp_path / "stop", sample_count=3,
                                  probes=fake_probe_factory)
    rows = [json.loads(line) for line in (tmp_path / "clock_p450.jsonl").read_text().splitlines()]
    assert result == 0
    assert len(rows) == 3
    assert all(row["host"] == "p450" for row in rows)


def test_monitor_stops_when_stop_file_exists(tmp_path):
    stop_file = tmp_path / "stop"
    stop_file.touch()
    assert run_monitor_for_test(tmp_path, stop_file, sample_count=3,
                                probes=lambda host: ClockProbe(host)) == 0


def test_fake_responder_cli_emits_configured_offset_and_outlier(tmp_path):
    import subprocess
    import sys
    script = Path(__file__).with_name("fake_ssh.py")
    proc = subprocess.Popen([sys.executable, str(script), "--offset-ns", "1234",
                             "--delay-ns", "0", "--outliers", "1"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True)
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write("x\nx\n")
    proc.stdin.close()
    rows = [list(map(int, line.split())) for line in proc.stdout.read().splitlines()]
    proc.wait(timeout=2)
    assert len(rows) == 2 and all(len(row) == 3 for row in rows)
    assert rows[0][0] - rows[0][1] == 0
    assert rows[1][1] - rows[1][0] == 100_000_000


def test_monitor_stops_on_terminal_manifest(tmp_path):
    manifest = tmp_path / "episode.json"
    manifest.write_text(json.dumps({"state": "complete"}))
    assert run_monitor_for_test(tmp_path, tmp_path / "never", sample_count=None,
                                probes=lambda host: pytest.fail("must not probe")) == 0


def test_monitor_does_not_treat_recovered_manifest_as_terminal(tmp_path):
    manifest = tmp_path / "episode.json"
    manifest.write_text(json.dumps({"state": "recovered"}))

    class Probe:
        def __init__(self, host):
            self.host = host

        def sample(self):
            return _sample(self.host, 0)

        def close(self):
            pass

        def drain_errors(self):
            return []

    assert run_monitor_for_test(
        tmp_path,
        tmp_path / "never",
        sample_count=1,
        probes=Probe,
    ) == 0
    assert (tmp_path / "clock_p450.jsonl").is_file()
    assert (tmp_path / "clock_unitree.jsonl").is_file()


def test_launch_detached_monitor_uses_config_and_persists_pid(tmp_path, monkeypatch):
    manifest = tmp_path / "episode.json"
    manifest.write_text(json.dumps({"episode_id": "e1", "state": "recording"}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"hosts": ["p450"]}))
    class Dummy:
        pid = 4321
    calls = []
    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Dummy()
    monkeypatch.setattr("jointctl.clock_sync.subprocess.Popen", fake_popen)
    assert launch_detached_monitor(tmp_path, config) == 4321
    assert calls and "--config" in calls[0][0]
    expected_src = str(Path(__file__).resolve().parents[1] / "src")
    assert calls[0][1]["env"]["PYTHONPATH"].split(os.pathsep)[0] == expected_src
    assert json.loads(manifest.read_text())["clock_monitor_pid"] == 4321


def test_launch_rejects_invalid_config_or_missing_manifest_without_spawn(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("jointctl.clock_sync.subprocess.Popen", lambda *a, **k: calls.append(1))
    config = tmp_path / "config.json"
    config.write_text("not-json")
    with pytest.raises(ValueError):
        launch_detached_monitor(tmp_path, config)
    config.write_text(json.dumps({"hosts": []}))
    with pytest.raises(ValueError):
        launch_detached_monitor(tmp_path, config)
    assert calls == []


def test_launch_terminates_child_if_pid_persistence_fails(tmp_path, monkeypatch):
    manifest = tmp_path / "episode.json"
    manifest.write_text(json.dumps({"state": "recording"}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"hosts": ["p450"], "manifest_file": "episode.json"}))
    class Dummy:
        pid = 77
        def terminate(self): self.terminated = True
    child = Dummy()
    monkeypatch.setattr("jointctl.clock_sync.subprocess.Popen", lambda *a, **k: child)
    monkeypatch.setattr("jointctl.clock_sync.os.replace", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        launch_detached_monitor(tmp_path, config)
    assert getattr(child, "terminated", False)


def test_monitor_slow_host_does_not_block_healthy_host(tmp_path):
    class Probe:
        def __init__(self, host): self.host, self.count = host, 0
        def sample(self):
            self.count += 1
            if self.host == "unitree": time.sleep(0.25)
            return _sample(self.host, self.count)
        def close(self): pass
        def drain_errors(self): return []
    probes = {}
    def factory(host): probes[host] = Probe(host); return probes[host]
    stop = tmp_path / "stop"
    import threading
    timer = threading.Timer(0.55, stop.touch); timer.start()
    run_monitor(tmp_path, ("p450", "unitree"), 0.1, stop) if False else __import__("jointctl.clock_sync", fromlist=["_run_monitor"])._run_monitor(tmp_path, ("p450", "unitree"), 0.1, stop, factory)
    assert probes["p450"].count >= 4


def test_probe_reader_bounds_oversized_and_chatty_output():
    responses = queue.Queue(maxsize=2)
    ClockProbe._reader(io.StringIO("x" * 10000 + "\n1 2 3\n1 2 3\n"), responses)
    assert responses.maxsize == 2
    first = responses.get_nowait()
    assert isinstance(first, RuntimeError)


def test_probe_reader_requests_finite_limit_and_invalidates_on_queue_overflow():
    class Stream:
        def __init__(self): self.limits = []; self.lines = iter(["1 2 3\n", "4 5 6\n"])
        def readline(self, limit=-1):
            self.limits.append(limit)
            return next(self.lines, "")
    stream = Stream()
    responses = queue.Queue(maxsize=1)
    ClockProbe._reader(stream, responses)
    assert stream.limits and stream.limits[0] == 4097
    assert any(isinstance(item, RuntimeError) and "overflow" in str(item)
               for item in list(responses.queue))
