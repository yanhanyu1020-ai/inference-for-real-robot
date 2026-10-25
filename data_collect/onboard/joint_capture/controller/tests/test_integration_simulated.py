from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from jointctl.cli import build_alignment_report
from jointctl.controller import JointController
from jointctl.inspectors import StreamSummary
from jointctl.manifest import ManifestStore
from jointctl.models import ClockSample, CommandResult, EpisodeState, RemoteStatus


class SimulatedClock:
    def __init__(self):
        self.wall_ns = 1_800_000_000_000_000_000
        self.monotonic_s = 0.0

    def advance(self, seconds: float):
        self.wall_ns += round(seconds * 1_000_000_000)
        self.monotonic_s += seconds

    def sleep(self, seconds: float):
        self.advance(seconds)


class SimulatedRemote:
    def __init__(self, host: str):
        self.host = host
        self.episode_id = None
        self.progress = 0

    def status(self):
        if self.episode_id is None:
            return RemoteStatus(self.host, True, "idle")
        self.progress += 1
        progress_name = "session_bytes" if self.host == "p450" else "frames_saved"
        directory = f"/capture/{self.episode_id}"
        message = (json.dumps({"recorder": {"session_dir": directory}})
                   if self.host == "p450" else f"episode={directory}\n")
        return RemoteStatus(self.host, True, "recording", message=message, active=True,
                            episode_id=self.episode_id, progress_name=progress_name,
                            progress_value=self.progress)

    def start(self, episode_id, instruction, task):
        self.episode_id = episode_id
        directory = f"/capture/{episode_id}"
        output = (json.dumps({"session_dir": directory}) if self.host == "p450"
                  else f"episode={directory}\n")
        return CommandResult("simulated start", 0, stdout=output)

    def stop(self):
        self.episode_id = None
        return CommandResult("simulated stop", 0)


class InitialProbe:
    def __init__(self, simulation, host):
        self.simulation = simulation
        self.host = host

    def sample(self):
        return self.simulation.sample(self.host, self.simulation.clock.wall_ns, 1_000_000)


class SimulatedSession:
    base_offsets = {"p450": 800_000_000, "unitree": -350_000_000}

    def __init__(self, root: Path):
        self.clock = SimulatedClock()
        self.origin_ns = self.clock.wall_ns
        self.drifts = {"p450": 0, "unitree": 0}
        self.sequence = {"p450": 0, "unitree": 0}
        self.store = ManifestStore(root)
        self.remotes = {host: SimulatedRemote(host) for host in ("p450", "unitree")}
        self.controller = JointController(
            self.remotes["p450"], self.remotes["unitree"], self.store,
            {host: InitialProbe(self, host) for host in self.remotes},
            monitor_launcher=lambda _episode, _config: 4242,
            now=lambda: datetime(2026, 9, 2, tzinfo=timezone.utc),
            random_bytes=lambda _count: b"\x12\x34",
            time_ns=lambda: self.clock.wall_ns,
            monotonic=lambda: self.clock.monotonic_s,
            sleep=self.clock.sleep,
            pid_is_running=lambda _pid: False,
            readiness_timeout_s=2,
        )
        self.actual_points = {host: [] for host in self.remotes}

    @property
    def episode_dir(self):
        return next(self.store.root.glob('joint_*/manifest.json')).parent

    def remote_time(self, host, desktop_ns):
        elapsed_ns = desktop_ns - self.origin_ns
        return desktop_ns + self.base_offsets[host] + round(elapsed_ns * self.drifts[host] / 1_000_000)

    def sample(self, host, desktop_ns, rtt_ns):
        remote_ns = self.remote_time(host, desktop_ns)
        sequence = self.sequence[host]
        self.sequence[host] += 1
        return ClockSample(
            host, sequence,
            desktop_ns - rtt_ns // 2, desktop_ns - rtt_ns // 2,
            remote_ns, remote_ns, remote_ns,
            desktop_ns + rtt_ns // 2, desktop_ns + rtt_ns // 2,
            rtt_ns, remote_ns - desktop_ns,
        )

    def advance(self, seconds, p450_drift_ppm, unitree_drift_ppm):
        self.drifts.update(p450=p450_drift_ppm, unitree=unitree_drift_ppm)
        rows = {host: [] for host in self.remotes}
        for index in range(int(seconds) * 5 + 1):
            desktop_ns = self.clock.wall_ns + index * 200_000_000
            for host in self.remotes:
                jitter = (index * (1_300_003 if host == "p450" else 1_700_009)) % 2_000_000
                rtt_ns = 1_000_000 + jitter
                if index and index % 37 == 0:
                    rtt_ns += 40_000_000
                rows[host].append(self.sample(host, desktop_ns, rtt_ns))
                self.actual_points[host].append((self.remote_time(host, desktop_ns), desktop_ns))
        for host, samples in rows.items():
            (self.episode_dir / f"clock_{host}.jsonl").write_text(
                "".join(json.dumps(sample.to_dict()) + "\n" for sample in samples),
                encoding="utf-8",
            )
        self.clock.advance(seconds)


def test_simulated_session_survives_offset_drift_jitter_and_produces_valid_interval(tmp_path):
    simulation = SimulatedSession(tmp_path)
    started = simulation.controller.start("demo", "joint")
    simulation.advance(seconds=30, p450_drift_ppm=-500, unitree_drift_ppm=42)
    stopped = simulation.controller.stop()
    streams = {
        host: [StreamSummary("simulated", len(points), points[2][0], points[-2][0])]
        for host, points in simulation.actual_points.items()
    }
    report = build_alignment_report(simulation.episode_dir, stream_summaries=streams)
    assert started.t0_desktop_ns < stopped.t1_desktop_ns
    assert stopped.state == EpisodeState.COMPLETE
    assert report["state"] == "complete"
    # Synthetic telemetry proves mapping coverage, not camera completeness.
    assert report["coverage_interval"]["start_inclusive_ns"] < report["coverage_interval"]["end_exclusive_ns"]
    assert report['valid_interval'] is None
    assert report["quality"]["estimated_error_ns"] < 10_000_000
    for host, points in simulation.actual_points.items():
        # Verify actual mapped error, independently from the reported bound.
        from jointctl.alignment import build_clock_timeline
        samples = [ClockSample.from_dict(json.loads(line)) for line in
                   (simulation.episode_dir / f"clock_{host}.jsonl").read_text(encoding="utf-8").splitlines()]
        timeline = build_clock_timeline(samples, 5_000_000_000)
        assert max(abs(timeline.remote_to_desktop(remote).desktop_time_ns - actual)
                   for remote, actual in points[2:-2]) < 10_000_000
