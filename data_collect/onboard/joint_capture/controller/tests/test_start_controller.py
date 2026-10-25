from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading

import pytest

from jointctl.controller import (
    JointController,
    JointStartFailed,
    RemoteAlreadyActive,
    generate_episode_id,
)
from jointctl.clock_sync import ClockProbeError
from jointctl.manifest import ManifestStore
from jointctl.models import ClockSample, CommandResult, EpisodeManifest, EpisodeState, RemoteStatus


FIXED_NOW = datetime(2026, 9, 2, 9, 8, 7, tzinfo=timezone.utc)


def _status(kind: str, *, active: bool, episode_id: str | None,
            progress: int = 0, message: str = "") -> RemoteStatus:
    progress_name = "session_bytes" if kind == "p450" else "frames_saved"
    return RemoteStatus(kind, True, "recording" if active else "idle", message=message,
                        active=active, episode_id=episode_id,
                        progress_name=progress_name, progress_value=progress)


class _Concurrency:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_status = 0
        self.max_status = 0
        self.active_start = 0
        self.max_start = 0
        self.status_barrier = threading.Barrier(2)
        self.start_barrier = threading.Barrier(2)

    def enter(self, operation: str, barrier: threading.Barrier | None = None) -> None:
        active_name = f"active_{operation}"
        maximum_name = f"max_{operation}"
        with self.lock:
            setattr(self, active_name, getattr(self, active_name) + 1)
            setattr(self, maximum_name, max(getattr(self, maximum_name),
                                            getattr(self, active_name)))
        if barrier is not None:
            barrier.wait(timeout=1)

    def leave(self, operation: str) -> None:
        active_name = f"active_{operation}"
        with self.lock:
            setattr(self, active_name, getattr(self, active_name) - 1)


class _Remote:
    def __init__(self, kind: str, concurrency: _Concurrency,
                 progress: tuple[int, ...] = (0, 1, 2)) -> None:
        self.host = kind
        self.kind = kind
        self._concurrency = concurrency
        self._progress = list(progress)
        self.status_value: RemoteStatus | None = None
        self.start_failure: CommandResult | None = None
        self.started_episode: str | None = None
        self.start_calls = 0
        self.stop_calls = 0
        self.status_calls = 0
        self.store: ManifestStore | None = None

    def status(self) -> RemoteStatus:
        self.status_calls += 1
        first = self.status_calls == 1
        self._concurrency.enter("status", self._concurrency.status_barrier if first else None)
        try:
            if self.status_value is not None:
                return self.status_value
            if self.started_episode is None:
                return _status(self.kind, active=False, episode_id=None)
            progress = self._progress.pop(0) if self._progress else 999
            directory = f"/capture/{self.started_episode}"
            message = (json.dumps({"recorder": {"session_dir": directory}})
                       if self.kind == "p450" else f"episode={directory}\n")
            return _status(self.kind, active=True, episode_id=self.started_episode,
                           progress=progress, message=message)
        finally:
            self._concurrency.leave("status")

    def start(self, episode_id: str, instruction: str = "", task: str = "") -> CommandResult:
        self.start_calls += 1
        assert self.store is not None
        persisted = self.store.load(episode_id)
        assert persisted.clock_monitor_pid == 4321
        self._concurrency.enter("start", self._concurrency.start_barrier)
        try:
            if self.start_failure is not None:
                return self.start_failure
            self.started_episode = episode_id
            directory = f"/capture/{episode_id}"
            stdout = (json.dumps({"session_dir": directory}) if self.kind == "p450"
                      else f"episode={directory}\n")
            return CommandResult(f"start {episode_id}", 0, stdout=stdout)
        finally:
            self._concurrency.leave("start")

    def stop(self) -> CommandResult:
        self.stop_calls += 1
        self.started_episode = None
        return CommandResult("stop", 0, stdout="stopped")


class _Probe:
    def __init__(self, host: str) -> None:
        self.host = host
        self.calls = 0

    def sample(self) -> ClockSample:
        self.calls += 1
        return ClockSample(
            host=self.host, sequence=3,
            local_send_wall_ns=10_000, local_send_mono_ns=1_000,
            remote_receive_wall_ns=10_100, remote_send_wall_ns=10_100,
            remote_monotonic_ns=2_000,
            local_receive_wall_ns=10_200, local_receive_mono_ns=1_200,
            rtt_ns=200, offset_ns=0,
        )


class _Harness:
    def __init__(self, tmp_path: Path, *, p450_progress: tuple[int, ...] = (0, 1, 2),
                 unitree_progress: tuple[int, ...] = (0, 1, 2)) -> None:
        self.concurrency = _Concurrency()
        self.store = ManifestStore(tmp_path)
        self.p450 = _Remote("p450", self.concurrency, p450_progress)
        self.unitree = _Remote("unitree", self.concurrency, unitree_progress)
        self.p450.store = self.store
        self.unitree.store = self.store
        self.p450_probe = _Probe("p450")
        self.unitree_probe = _Probe("unitree")
        self.wall_time_calls = 0
        self.sleep_calls = 0

        def time_ns() -> int:
            self.wall_time_calls += 1
            return self.wall_time_calls * 1_000_000_000

        def launcher(manifest_dir: Path, config_path: Path) -> int:
            manifest = EpisodeManifest.from_dict(
                json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
            )
            assert manifest.state == EpisodeState.STARTING
            assert config_path.is_file()
            return 4321

        self.controller = JointController(
            p450=self.p450,
            unitree=self.unitree,
            store=self.store,
            clock_probes={"p450": self.p450_probe, "unitree": self.unitree_probe},
            monitor_launcher=launcher,
            now=lambda: FIXED_NOW,
            random_bytes=lambda count: bytes.fromhex("ab12"),
            time_ns=time_ns,
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: setattr(self, "sleep_calls", self.sleep_calls + 1),
            readiness_timeout_s=2.0,
        )


def _only_manifest(store: ManifestStore) -> EpisodeManifest:
    manifests = list(store.root.glob("joint_*/manifest.json"))
    assert len(manifests) == 1
    return store.load(manifests[0].parent.name)


def test_generate_episode_id_uses_supplied_timezone_and_four_hex_digits():
    value = generate_episode_id(
        datetime(2026, 9, 2, 17, 8, 7, tzinfo=timezone.utc), bytes.fromhex("09af")
    )
    assert value == "joint_20260902_170807_UTCp0000_09af"


def test_generate_episode_id_uses_desktop_local_time_and_explicit_offset():
    value = generate_episode_id(
        datetime(2026, 9, 26, 14, 30, 15, tzinfo=timezone(timedelta(hours=8))),
        bytes.fromhex("ab12"),
    )
    assert value == "joint_20260926_143015_UTCp0800_ab12"


def test_start_without_user_task_uses_same_timestamp_id_as_remote_group(tmp_path):
    harness = _Harness(tmp_path)
    result = harness.controller.start("", "")
    assert result.mode == result.episode_id
    assert result.label == result.episode_id


def test_start_runs_status_and_start_on_both_hosts_concurrently_and_persists_outputs(tmp_path):
    harness = _Harness(tmp_path)

    result = harness.controller.start("pick box", "joint")

    assert result.state == EpisodeState.RECORDING
    assert result.t0_desktop_ns == 2_500_000_000
    assert result.clock_monitor_pid == 4321
    assert result.remote_directories == {
        "p450": "/capture/joint_20260902_090807_UTCp0000_ab12",
        "unitree": "/capture/joint_20260902_090807_UTCp0000_ab12",
    }
    assert result.start_results["p450"].ok
    assert result.start_results["unitree"].ok
    assert set(result.clock_samples) == {"p450", "unitree"}
    assert set(result.clock_estimates) == {"p450", "unitree"}
    assert harness.concurrency.max_status == 2
    assert harness.concurrency.max_start == 2
    assert harness.p450_probe.calls == harness.unitree_probe.calls == 3


def test_start_rejects_persistently_poor_clock_quality_and_rolls_back(tmp_path):
    from dataclasses import replace
    harness = _Harness(tmp_path)
    original = harness.p450_probe.sample
    harness.p450_probe.sample = lambda: replace(original(), rtt_ns=100_000_000)
    with pytest.raises(JointStartFailed, match='clock quality'):
        harness.controller.start('test', 'joint')
    assert harness.p450.stop_calls == 1
    assert harness.unitree.stop_calls == 1


def test_start_refuses_when_remote_episode_is_active_without_creating_manifest(tmp_path):
    harness = _Harness(tmp_path)
    harness.p450.status_value = _status(
        "p450", active=True, episode_id="foreign_episode", progress=8
    )

    with pytest.raises(RemoteAlreadyActive, match="foreign_episode"):
        harness.controller.start("demo", "joint")

    assert harness.p450.start_calls == 0
    assert harness.unitree.start_calls == 0
    assert not list(tmp_path.glob("joint_*"))


def test_preflight_failure_marks_transport_and_remote_command_distinctly(tmp_path):
    transport = _Harness(tmp_path / "transport")
    transport.p450.status_value = RemoteStatus("p450", False, "unknown", last_error="ssh timeout")
    with pytest.raises(JointStartFailed) as transport_error:
        transport.controller.start("demo", "joint")
    assert transport_error.value.failure_kind == "connectivity"

    command = _Harness(tmp_path / "command")
    command.p450.status_value = RemoteStatus("p450", True, "unknown", last_error="status script failed")
    with pytest.raises(JointStartFailed) as command_error:
        command.controller.start("demo", "joint")
    assert command_error.value.failure_kind == "remote_command"


@pytest.mark.parametrize("failure_kind", ["connectivity", "remote_command"])
def test_start_propagates_clock_probe_failure_kind(tmp_path, failure_kind):
    harness = _Harness(tmp_path)
    def fail_sample():
        raise ClockProbeError("clock probe failed", failure_kind=failure_kind)
    harness.p450_probe.sample = fail_sample
    with pytest.raises(JointStartFailed) as error:
        harness.controller.start("demo", "joint")
    assert error.value.failure_kind == failure_kind


def test_readiness_requires_two_consecutive_increases_on_each_host(tmp_path):
    harness = _Harness(tmp_path, p450_progress=(0, 1, 1, 2, 3),
                       unitree_progress=(0, 1, 1, 2, 3))

    harness.controller.start("demo", "joint")

    # Preflight plus all five readiness polls proves the equal-value poll reset the streak.
    assert harness.p450.status_calls == 6
    assert harness.unitree.status_calls == 6


def test_partial_start_rolls_back_only_matching_episode_and_persists_diagnostics(tmp_path):
    harness = _Harness(tmp_path)
    harness.unitree.start_failure = CommandResult("start unitree", 1, stderr="camera failed")

    with pytest.raises(JointStartFailed, match="camera failed"):
        harness.controller.start("demo", "joint")

    result = _only_manifest(harness.store)
    assert harness.p450.stop_calls == 1
    assert harness.unitree.stop_calls == 0
    assert result.state == EpisodeState.START_FAILED
    assert result.start_results["unitree"].stderr == "camera failed"
    assert result.rollback_statuses["p450"].episode_id == result.episode_id
    assert result.rollback_results["p450"].ok
    assert any(item.phase == "start" and "camera failed" in item.message
               for item in result.diagnostics)


def test_rollback_does_not_stop_remote_with_different_episode_id(tmp_path):
    harness = _Harness(tmp_path)
    harness.unitree.start_failure = CommandResult("start unitree", 1, stderr="camera failed")
    harness.p450.status_value = _status("p450", active=False, episode_id=None)

    # The preflight must be idle; switch to a foreign active status after start failure.
    original_start = harness.unitree.start

    def fail_and_report_foreign(*args, **kwargs):
        result = original_start(*args, **kwargs)
        harness.p450.status_value = _status(
            "p450", active=True, episode_id="foreign_episode", progress=99
        )
        return result

    harness.unitree.start = fail_and_report_foreign

    with pytest.raises(JointStartFailed):
        harness.controller.start("demo", "joint")

    assert harness.p450.stop_calls == 0
    result = _only_manifest(harness.store)
    assert result.rollback_statuses["p450"].episode_id == "foreign_episode"


def test_failed_start_with_unreachable_cleanup_keeps_active_owner(tmp_path):
    harness = _Harness(tmp_path)
    original = harness.unitree.start
    def disconnect(*args, **kwargs):
        original(*args, **kwargs)
        harness.unitree.status_value = RemoteStatus('unitree', False, last_error='offline')
        return CommandResult('start', -1, stderr='transport timeout')
    harness.unitree.start = disconnect
    with pytest.raises(JointStartFailed):
        harness.controller.start('demo', 'joint')
    assert harness.store.active().state == EpisodeState.PARTIAL
    assert harness.p450.stop_calls == 1
    assert harness.unitree.stop_calls == 0


def test_old_manifest_without_start_fields_remains_loadable():
    restored = EpisodeManifest.from_dict({
        "episode_id": "joint_old",
        "label": "demo",
        "mode": "joint",
        "created_desktop_ns": 123,
        "state": "starting",
        "metadata": {},
    })

    assert restored.clock_monitor_pid is None
    assert restored.t0_desktop_ns is None
    assert restored.start_results == {}
    assert restored.diagnostics == []
