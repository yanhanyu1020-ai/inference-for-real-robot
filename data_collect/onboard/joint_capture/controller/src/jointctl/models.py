"""Serializable, dependency-free models shared by controller components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any, Mapping


class EpisodeState(str, Enum):
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"
    COMPLETE = "complete"
    START_FAILED = "start_failed"
    STOP_FAILED = "stop_failed"
    PARTIAL = "partial"
    RECOVERED = "recovered"


def _json_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return asdict(value)


def _round_div_two(value: int) -> int:
    """Round an integer divided by two using Python's ties-to-even rule."""
    quotient, remainder = divmod(value, 2)
    if remainder == 0 or quotient % 2 == 0:
        return quotient
    return quotient + 1


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_ns: int = 0

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def exit_code(self) -> int:
        """Compatibility alias for callers that call the process result an exit code."""
        return self.returncode

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CommandResult":
        return cls(**dict(data))


@dataclass(frozen=True)
class RemoteStatus:
    host: str
    reachable: bool
    state: str = "unknown"
    message: str = ""
    last_error: str | None = None
    active: bool = False
    episode_id: str | None = None
    progress_name: str | None = None
    progress_value: int = 0
    format_version: int = 1
    streams: dict[str, Any] = field(default_factory=dict)
    ready: bool | None = None
    quality_ok: bool | None = None
    fault: list[str] = field(default_factory=list)
    raw_closed: bool | None = None
    state_gap_ticks: int = 0
    missing_states: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RemoteStatus":
        return cls(**dict(data))


@dataclass(frozen=True)
class ClockSample:
    host: str
    sequence: int
    local_send_wall_ns: int
    local_send_mono_ns: int
    remote_receive_wall_ns: int
    remote_send_wall_ns: int
    remote_monotonic_ns: int
    local_receive_wall_ns: int
    local_receive_mono_ns: int
    rtt_ns: int
    offset_ns: int

    def __post_init__(self) -> None:
        if self.rtt_ns < 0:
            raise ValueError("rtt_ns must not be negative")

    @classmethod
    def from_exchange(cls, *, host: str, sequence: int,
                      local_send_wall_ns: int, local_send_mono_ns: int,
                      remote_receive_wall_ns: int, remote_send_wall_ns: int,
                      remote_monotonic_ns: int, local_receive_wall_ns: int,
                      local_receive_mono_ns: int) -> "ClockSample":
        rtt_ns = (local_receive_mono_ns - local_send_mono_ns) - (
            remote_send_wall_ns - remote_receive_wall_ns
        )
        if rtt_ns < 0:
            raise ValueError("clock exchange produced negative RTT")
        offset_sum_ns = ((remote_receive_wall_ns - local_send_wall_ns)
                         + (remote_send_wall_ns - local_receive_wall_ns))
        offset_ns = _round_div_two(offset_sum_ns)
        return cls(host, sequence, local_send_wall_ns, local_send_mono_ns,
                   remote_receive_wall_ns, remote_send_wall_ns,
                   remote_monotonic_ns, local_receive_wall_ns,
                   local_receive_mono_ns, rtt_ns, offset_ns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClockSample":
        return cls(**dict(data))


@dataclass(frozen=True)
class ClockEstimate:
    host: str
    offset_ns: int
    jitter_ns: int = 0
    sample_count: int = 0
    selected_sequence: int | None = None
    # Mapping fields are optional so persisted pre-alignment estimates remain valid.
    desktop_time_ns: int | None = None
    estimated_error_ns: int = 0
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClockEstimate":
        return cls(**dict(data))


@dataclass(frozen=True)
class JointStatusReport:
    episode_id: str | None
    state: EpisodeState
    remotes: list[RemoteStatus] = field(default_factory=list)
    clock_estimates: list[ClockEstimate] = field(default_factory=list)
    message: str = ""
    clock_health: dict[str, Any] = field(default_factory=dict)
    timing_degraded: bool = False
    timing_degradation_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "state", EpisodeState(self.state))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid episode state: {self.state!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id, "state": self.state.value,
                "remotes": [_json_dict(x) for x in self.remotes],
                "clock_estimates": [_json_dict(x) for x in self.clock_estimates],
                "message": self.message, "clock_health": self.clock_health,
                "timing_degraded": self.timing_degraded,
                "timing_degradation_reasons": self.timing_degradation_reasons}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JointStatusReport":
        value = dict(data)
        value["remotes"] = [RemoteStatus.from_dict(x) for x in value.get("remotes", [])]
        value["clock_estimates"] = [ClockEstimate.from_dict(x) for x in value.get("clock_estimates", [])]
        return cls(**value)


@dataclass(frozen=True)
class DiagnosticRecord:
    """A durable, structured explanation of an orchestration event."""

    phase: str
    message: str
    at_desktop_ns: int
    host: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiagnosticRecord":
        return cls(**dict(data))


@dataclass(frozen=True)
class EpisodeManifest:
    episode_id: str
    label: str
    mode: str
    created_desktop_ns: int
    state: EpisodeState = EpisodeState.STARTING
    metadata: dict[str, Any] = field(default_factory=dict)
    start_results: dict[str, CommandResult] = field(default_factory=dict)
    remote_directories: dict[str, str] = field(default_factory=dict)
    clock_monitor_pid: int | None = None
    t0_desktop_ns: int | None = None
    t1_desktop_ns: int | None = None
    clock_samples: dict[str, ClockSample] = field(default_factory=dict)
    clock_estimates: dict[str, ClockEstimate] = field(default_factory=dict)
    stop_results: dict[str, CommandResult] = field(default_factory=dict)
    clock_monitor_closed_cleanly: bool | None = None
    rollback_statuses: dict[str, RemoteStatus] = field(default_factory=dict)
    rollback_results: dict[str, CommandResult] = field(default_factory=dict)
    diagnostics: list[DiagnosticRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "state", EpisodeState(self.state))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid episode state: {self.state!r}") from exc

    @classmethod
    def new(cls, episode_id: str, label: str, mode: str) -> "EpisodeManifest":
        return cls(episode_id, label, mode, time.time_ns())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeManifest":
        value = dict(data)
        value["start_results"] = {
            host: CommandResult.from_dict(result)
            for host, result in value.get("start_results", {}).items()
        }
        value["clock_samples"] = {
            host: ClockSample.from_dict(sample)
            for host, sample in value.get("clock_samples", {}).items()
        }
        value["clock_estimates"] = {
            host: ClockEstimate.from_dict(estimate)
            for host, estimate in value.get("clock_estimates", {}).items()
        }
        value["stop_results"] = {
            host: CommandResult.from_dict(result)
            for host, result in value.get("stop_results", {}).items()
        }
        value["rollback_statuses"] = {
            host: RemoteStatus.from_dict(status)
            for host, status in value.get("rollback_statuses", {}).items()
        }
        value["rollback_results"] = {
            host: CommandResult.from_dict(result)
            for host, result in value.get("rollback_results", {}).items()
        }
        value["diagnostics"] = [
            DiagnosticRecord.from_dict(item) for item in value.get("diagnostics", [])
        ]
        return cls(**value)
