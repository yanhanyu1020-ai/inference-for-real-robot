import threading
from datetime import datetime, timezone

import pytest

from jointctl.controller import EpisodeMismatch, JointController, RecoveryConflict
from jointctl.manifest import ManifestStore
from jointctl.models import (
    CommandResult,
    EpisodeManifest,
    EpisodeState,
    JointStatusReport,
    RemoteStatus,
)


def active_status(host: str, episode_id: str) -> RemoteStatus:
    progress_name = "session_bytes" if host == "p450" else "frames_saved"
    return RemoteStatus(
        host=host,
        reachable=True,
        state="recording",
        active=True,
        episode_id=episode_id,
        progress_name=progress_name,
        progress_value=12,
        message=(f'{{"session_dir":"/capture/{episode_id}"}}'
                 if host == "p450" else f"episode=/capture/{episode_id}\n"),
    )


class _Remote:
    def __init__(self, host: str, store: ManifestStore, barrier: threading.Barrier) -> None:
        self.host = host
        self.store = store
        self.barrier = barrier
        self.status_value = active_status(host, "joint_active")
        self.stop_calls = 0
        self.stop_entry_state: EpisodeState | None = None
        self.stop_entry_t1: int | None = None
        self._lock = threading.Lock()
        self._active_stops = 0
        self.max_stop_overlap = 0

    def status(self) -> RemoteStatus:
        return self.status_value

    def stop(self) -> CommandResult:
        self.stop_calls += 1
        persisted = self.store.active()
        assert persisted is not None
        self.stop_entry_state = persisted.state
        self.stop_entry_t1 = persisted.t1_desktop_ns
        with self._lock:
            self._active_stops += 1
            self.max_stop_overlap = max(self.max_stop_overlap, self._active_stops)
        try:
            self.barrier.wait(timeout=2)
            return CommandResult(f"stop {self.host}", 0, stdout="stopped")
        finally:
            with self._lock:
                self._active_stops -= 1


class _Harness:
    def __init__(self, tmp_path) -> None:
        self.store = ManifestStore(tmp_path)
        self.store.create(EpisodeManifest(
            episode_id="joint_active",
            label="demo",
            mode="joint",
            created_desktop_ns=1,
            state=EpisodeState.RECORDING,
            clock_monitor_pid=9876,
        ))
        barrier = threading.Barrier(2)
        self.p450 = _Remote("p450", self.store, barrier)
        self.unitree = _Remote("unitree", self.store, barrier)
        self.controller = JointController(
            p450=self.p450,
            unitree=self.unitree,
            store=self.store,
            clock_probes={"p450": object(), "unitree": object()},
            now=lambda: datetime(2026, 9, 2, tzinfo=timezone.utc),
            time_ns=lambda: 123_456_789,
            pid_is_running=lambda _pid: False,
        )

    @property
    def max_stop_overlap(self) -> int:
        return min(self.p450.max_stop_overlap, self.unitree.max_stop_overlap)


@pytest.fixture
def harness(tmp_path):
    return _Harness(tmp_path)


def test_stop_records_t1_and_stopping_before_concurrent_remote_stops(harness):
    result = harness.controller.stop()

    assert result.t1_desktop_ns == 123_456_789
    assert harness.p450.stop_entry_state == EpisodeState.STOPPING
    assert harness.unitree.stop_entry_state == EpisodeState.STOPPING
    assert harness.p450.stop_entry_t1 == result.t1_desktop_ns
    assert harness.unitree.stop_entry_t1 == result.t1_desktop_ns
    assert harness.max_stop_overlap == 1
    assert result.state == EpisodeState.COMPLETE
    assert result.metadata["postprocess"] == "pending"
    assert result.clock_monitor_closed_cleanly is True
    assert (harness.store.episode_dir(result.episode_id) / "clock.stop").is_file()
    persisted = harness.store.load(result.episode_id)
    assert persisted.stop_results["p450"].ok
    assert persisted.stop_results["unitree"].ok


def test_stop_records_monitor_timeout_after_remote_stops(tmp_path):
    harness = _Harness(tmp_path)
    monotonic_values = iter((0.0, 0.0, 5.1))
    harness.controller.monotonic = lambda: next(monotonic_values)
    harness.controller.pid_is_running = lambda _pid: True
    harness.controller.sleep = lambda _seconds: None

    result = harness.controller.stop()

    assert result.state == EpisodeState.PARTIAL
    assert result.clock_monitor_closed_cleanly is False
    assert any(item.phase == "clock_monitor" for item in result.diagnostics)


def test_stop_refuses_mismatched_remote_episode(harness):
    harness.unitree.status_value = active_status("unitree", "foreign_episode")

    with pytest.raises(EpisodeMismatch) as caught:
        harness.controller.stop()

    assert caught.value.report.episode_id == "joint_active"
    assert [status.episode_id for status in caught.value.report.remotes] == [
        "joint_active", "foreign_episode"
    ]
    assert harness.p450.stop_calls == 0
    assert harness.unitree.stop_calls == 0
    assert harness.store.active().state == EpisodeState.RECORDING


def test_status_reports_local_manifest_and_both_remote_statuses(harness):
    report = harness.controller.status()

    assert isinstance(report, JointStatusReport)
    assert report.episode_id == "joint_active"
    assert report.state == EpisodeState.RECORDING
    assert [status.host for status in report.remotes] == ["p450", "unitree"]


def test_recover_rebuilds_active_manifest_when_both_remote_ids_match(harness):
    harness.store.clear_active("joint_active")
    harness.p450.status_value = active_status("p450", "joint_recover")
    harness.unitree.status_value = active_status("unitree", "joint_recover")

    result = harness.controller.recover()

    assert result.episode_id == "joint_recover"
    assert result.state == EpisodeState.RECOVERED
    assert result.remote_directories == {
        "p450": "/capture/joint_recover",
        "unitree": "/capture/joint_recover",
    }
    assert harness.store.active().episode_id == "joint_recover"


def test_recover_rebuilds_existing_manifest_when_only_pointer_was_lost(harness):
    harness.store.clear_active("joint_active")

    result = harness.controller.recover()

    assert result.episode_id == "joint_active"
    assert result.state == EpisodeState.RECOVERED
    assert result.label == "demo"
    assert result.clock_monitor_pid == 9876
    assert harness.store.active().state == EpisodeState.RECOVERED


@pytest.mark.parametrize(
    ("p450", "unitree"),
    [
        (active_status("p450", "joint_a"), active_status("unitree", "joint_b")),
        (active_status("p450", "joint_a"), RemoteStatus("unitree", True, "idle")),
    ],
)
def test_recovery_conflict_contains_both_statuses_and_stops_nothing(
    harness, p450, unitree
):
    harness.store.clear_active("joint_active")
    harness.p450.status_value = p450
    harness.unitree.status_value = unitree

    with pytest.raises(RecoveryConflict) as caught:
        harness.controller.recover()

    assert caught.value.report.state == EpisodeState.PARTIAL
    assert caught.value.report.remotes == [p450, unitree]
    assert harness.p450.stop_calls == 0
    assert harness.unitree.stop_calls == 0
    assert harness.store.active() is None


def test_recovered_episode_can_be_stopped(harness):
    harness.store.clear_active("joint_active")
    recovered = EpisodeManifest(
        episode_id="joint_recovered",
        label="recovered",
        mode="joint",
        created_desktop_ns=2,
        state=EpisodeState.RECOVERED,
    )
    harness.store.create(recovered)
    harness.p450.status_value = active_status("p450", "joint_recovered")
    harness.unitree.status_value = active_status("unitree", "joint_recovered")

    result = harness.controller.stop()

    assert result.state == EpisodeState.COMPLETE
    assert result.t1_desktop_ns == 123_456_789
