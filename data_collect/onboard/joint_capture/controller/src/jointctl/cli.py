"""The dependency-free ``jointctl`` operator command line."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from .alignment import build_clock_timeline, DEFAULT_MAX_CLOCK_GAP_NS
from .clock_sync import ClockProbe, ClockProbeError, run_monitor, read_clock_records, DEFAULT_CLOCK_FRESHNESS_NS
from .controller import (
    EpisodeMismatch,
    JointController,
    JointStartFailed,
    RecoveryConflict,
    RemoteAlreadyActive,
)
from .inspectors import InspectorConnectionError, InspectorError, P450Inspector, StreamSummary, UnitreeInspector
from .manifest import ManifestStore, ActiveEpisodeConflict
from .models import ClockSample, EpisodeManifest, EpisodeState, JointStatusReport
from .remote import RemoteClient
from .operation_lock import operation_lock, OperationBusy
from .postprocess import finalize_pending


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONNECTIVITY = 3
EXIT_CONFLICT = 4
EXIT_TIMING = 5
EXIT_REMOTE_COMMAND = 6


def _default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default.json"


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def _manifest_root(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    configured = args.manifest_root or config.get("manifest_root", "episodes")
    return Path(configured).expanduser()


def make_controller(args: argparse.Namespace) -> JointController:
    config = _load_config(Path(args.config))
    root = _manifest_root(args, config)
    p450_config = config.get("p450", {})
    unitree_config = config.get("unitree", {})
    if not isinstance(p450_config, dict) or not isinstance(unitree_config, dict):
        raise ValueError("p450 and unitree configuration must be objects")
    p450_host = str(p450_config.get("host", "p450"))
    unitree_host = str(unitree_config.get("host", "unitree"))
    timeout = float(config.get("ssh_timeout_s", 10.0))
    operation_timeouts = {"start_timeout_s": float(config.get("start_timeout_s", 120.0)),
                          "prepare_before_start": True,
                          "stop_timeout_s": float(config.get("stop_timeout_s", 300.0))}
    return JointController(
        p450=RemoteClient(p450_host, kind="p450", timeout_s=timeout, **operation_timeouts),
        unitree=RemoteClient(unitree_host, kind="unitree", timeout_s=timeout, **operation_timeouts),
        store=ManifestStore(root),
        clock_probes={"p450": ClockProbe("p450", destination=p450_host, timeout_s=timeout),
                      "unitree": ClockProbe("unitree", destination=unitree_host, timeout_s=timeout)},
        readiness_timeout_s=float(config.get("readiness_timeout_s", 10.0)),
        poll_interval_s=float(config.get("poll_interval_s", 0.2)),
        clock_interval_s=float(config.get("clock_interval_s", 0.2)),
        ssh_timeout_s=timeout,
        clock_freshness_ns=int(config.get("clock_freshness_ns", DEFAULT_CLOCK_FRESHNESS_NS)),
        alignment_window_ns=int(config.get("alignment_window_ns", 5_000_000_000)),
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )


def _read_clock_samples(episode_dir: Path, manifest: EpisodeManifest) -> dict[str, list[ClockSample]]:
    return read_clock_records(episode_dir, manifest)[0]


def _percentile(values: Iterable[int], percentile: float) -> int | None:
    """Return the nearest-rank percentile from integer nanosecond values."""
    ordered = sorted(values)
    if not ordered:
        return None
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between zero and one")
    index = max(0, (len(ordered) * percentile).__ceil__() - 1)
    return ordered[index]


def _summary_stats(values: Iterable[int]) -> dict[str, int] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2
              else (ordered[middle - 1] + ordered[middle]) // 2)
    return {
        "min": ordered[0], "median": median,
        "p95": _percentile(ordered, 0.95), "max": ordered[-1],
    }


def _clock_diagnostics(samples: Iterable[ClockSample], timeline) -> dict[str, Any]:
    values = list(samples)
    rtt_stats = _summary_stats(sample.rtt_ns for sample in values)
    residual_stats = _summary_stats(
        abs(sample.offset_ns - timeline.remote_to_desktop(
            sample.remote_send_wall_ns, allow_single_anchor=True
        ).offset_ns)
        for sample in values
    )
    slopes: list[int] = []
    for left, right in zip(timeline.anchors, timeline.anchors[1:]):
        span = right.remote_wall_ns - left.remote_wall_ns
        if span:
            slopes.append(round((right.offset_ns - left.offset_ns) * 1_000_000 / span))
    drift_stats = _summary_stats(slopes)
    return {
        "rtt_ns": (None if rtt_stats is None else {
            "min": rtt_stats["min"], "median": rtt_stats["median"], "p95": rtt_stats["p95"],
        }),
        "fit_residual_ns": (None if residual_stats is None else {
            "max": residual_stats["max"], "median": residual_stats["median"],
            "p95": residual_stats["p95"],
        }),
        "drift_ppm": (None if drift_stats is None else {
            "min": drift_stats["min"], "median": drift_stats["median"],
            "max": drift_stats["max"], "representative": drift_stats["median"],
        }),
        "uncertainty_components": {"network": "rtt_ns", "fit": "fit_residual_ns"},
    }


def run_probe_session(
    hosts: tuple[str, ...],
    duration_s: float,
    interval_s: float,
    *,
    probe_factory: Callable[[str], ClockProbe] = ClockProbe,
) -> tuple[dict[str, list[ClockSample]], dict[str, list[dict[str, Any]]]]:
    """Sample each host on its own worker so one slow link cannot stall another."""
    if duration_s <= 0:
        raise ValueError("--duration must be positive")
    if interval_s <= 0:
        raise ValueError("probe interval must be positive")
    if not hosts or len(set(hosts)) != len(hosts):
        raise ValueError("--hosts must contain unique host names")
    samples: dict[str, list[ClockSample]] = {host: [] for host in hosts}
    errors: dict[str, list[dict[str, Any]]] = {host: [] for host in hosts}
    deadline = time.monotonic() + duration_s

    def worker(host: str) -> None:
        probe = probe_factory(host)
        try:
            probe.open()
            next_at = time.monotonic()
            while time.monotonic() < deadline:
                try:
                    samples[host].append(probe.sample())
                except Exception as exc:
                    errors[host].append({
                        "kind": "probe_error",
                        "failure_kind": getattr(exc, "failure_kind", "remote_command"),
                        "message": str(exc),
                        "at_wall_ns": time.time_ns(),
                    })
                next_at += interval_s
                wait_s = min(max(0.0, next_at - time.monotonic()),
                             max(0.0, deadline - time.monotonic()))
                if wait_s:
                    time.sleep(wait_s)
        except Exception as exc:
            errors[host].append({
                "kind": "probe_error",
                "failure_kind": getattr(exc, "failure_kind", "remote_command"),
                "message": str(exc),
                "at_wall_ns": time.time_ns(),
            })
        finally:
            probe.close()

    workers = [threading.Thread(target=worker, args=(host,), daemon=True) for host in hosts]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    return samples, errors


def build_probe_report(
    samples_by_host: Mapping[str, Iterable[ClockSample]],
    errors_by_host: Mapping[str, Iterable[Mapping[str, Any]]],
    duration_s: float,
) -> dict[str, Any]:
    """Summarize a read-only multi-host probe, including relative uncertainty."""
    host_reports: dict[str, dict[str, Any]] = {}
    uncertainties: list[int] = []
    for host, source in samples_by_host.items():
        samples = sorted(source, key=lambda sample: sample.remote_send_wall_ns)
        if not samples:
            host_reports[host] = {
                "sample_count": 0, "rtt_ns": None, "offset_drift_ns": None,
                "drift_ppm": None, "fit_residual_ns": None,
                "outlier_percentage": None, "estimated_uncertainty_ns": None,
                "errors": list(errors_by_host.get(host, ())),
            }
            continue
        timeline = build_clock_timeline(samples, 5_000_000_000)
        diagnostics = _clock_diagnostics(samples, timeline)
        span_ns = samples[-1].remote_send_wall_ns - samples[0].remote_send_wall_ns
        offset_drift_ns = samples[-1].offset_ns - samples[0].offset_ns
        drift_ppm = round(offset_drift_ns * 1_000_000 / span_ns) if span_ns else 0
        min_rtt = min(sample.rtt_ns for sample in samples)
        outlier_count = sum(sample.rtt_ns > min_rtt + 2_000_000 for sample in samples)
        estimate = timeline.remote_to_desktop(
            samples[-1].remote_send_wall_ns, allow_single_anchor=True
        )
        uncertainties.append(estimate.estimated_error_ns)
        rtt = _summary_stats(sample.rtt_ns for sample in samples)
        host_reports[host] = {
            "sample_count": len(samples),
            "rtt_ns": {key: rtt[key] for key in ("min", "median", "p95")},
            "offset_start_ns": samples[0].offset_ns,
            "offset_end_ns": samples[-1].offset_ns,
            "offset_drift_ns": offset_drift_ns,
            "drift_ppm": drift_ppm,
            "fit_residual_ns": diagnostics["fit_residual_ns"],
            "outlier_percentage": outlier_count * 100.0 / len(samples),
            "estimated_uncertainty_ns": estimate.estimated_error_ns,
            "errors": list(errors_by_host.get(host, ())),
        }
    return {
        "duration_s": float(duration_s),
        "hosts": host_reports,
        "estimated_cross_host_uncertainty_ns": (
            sum(uncertainties) if len(uncertainties) == len(host_reports) else None
        ),
        "uncertainty_scope": "sum of independent per-host desktop-clock mapping bounds",
    }


def _mapping_uncertainty_over_interval(timeline, interval: Mapping[str, int],
                                       endpoint_error_ns: int) -> int:
    """Bound interval uncertainty using its covering streams and interior segments.

    Stream endpoint errors bound the extrapolated portions (including a single
    anchor's constant-offset mapping). They may cover more than the requested
    interval, which is deliberately conservative. Interior segments must also
    be checked: their anchor errors can exceed both endpoint errors.
    """
    start_ns = interval["start_inclusive_ns"]
    end_ns = interval["end_exclusive_ns"]
    uncertainties = [endpoint_error_ns]
    anchors = timeline.anchors
    for anchor in anchors:
        desktop_ns = anchor.remote_wall_ns - anchor.offset_ns
        if start_ns <= desktop_ns < end_ns:
            uncertainties.append(anchor.uncertainty_ns)
    for left, right in zip(anchors, anchors[1:]):
        left_desktop = left.remote_wall_ns - left.offset_ns
        right_desktop = right.remote_wall_ns - right.offset_ns
        segment_start, segment_end = sorted((left_desktop, right_desktop))
        if segment_start < end_ns and start_ns <= segment_end:
            uncertainties.append(max(left.uncertainty_ns, right.uncertainty_ns) + 1)
    return max(uncertainties)


def _mapped_stream(summary: StreamSummary, timeline) -> dict[str, Any]:
    first = timeline.remote_to_desktop(summary.first_ns, allow_single_anchor=True)
    last = timeline.remote_to_desktop(summary.last_ns, allow_single_anchor=True)
    return {
        **summary.to_dict(),
        "first_desktop_ns": first.desktop_time_ns,
        "last_desktop_ns": last.desktop_time_ns,
        "estimated_error_ns": max(first.estimated_error_ns, last.estimated_error_ns),
        "degraded": first.degraded or last.degraded,
    }


def build_alignment_report(
    episode_dir: str | Path,
    stream_summaries: Mapping[str, Iterable[StreamSummary]] | None = None,
    *,
    alignment_window_ns: int = 5_000_000_000,
    t0_guard_ns: int = 0,
    max_clock_gap_ns: int = DEFAULT_MAX_CLOCK_GAP_NS,
    max_data_gap_ns: int = 250_000_000,
) -> dict[str, Any]:
    """Build a JSON-serializable report from an episode without touching remotes."""
    episode_path = Path(episode_dir)
    try:
        manifest = EpisodeManifest.from_dict(json.loads(
            (episode_path / "manifest.json").read_text(encoding="utf-8")
        ))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid episode manifest: {exc}") from exc
    if alignment_window_ns <= 0:
        raise ValueError("alignment_window_ns must be positive")
    if max_clock_gap_ns <= 0:
        raise ValueError("max_clock_gap_ns must be positive")
    if max_data_gap_ns <= 0:
        raise ValueError('max_data_gap_ns must be positive')
    if t0_guard_ns < 0:
        raise ValueError("t0_guard_ns must not be negative")
    samples, clock_errors = read_clock_records(episode_path, manifest)
    normalized_summaries = {
        host: list(entries) for host, entries in (stream_summaries or {}).items()
    }
    timelines = {
        host: build_clock_timeline(values, alignment_window_ns, max_gap_ns=max_clock_gap_ns)
        for host, values in samples.items() if values
    }
    estimates = {
        host: timeline.remote_to_desktop(values[-1].remote_send_wall_ns, allow_single_anchor=True)
        for host, (timeline, values) in ((h, (timelines[h], samples[h])) for h in timelines)
    }
    clock_diagnostics = {
        host: _clock_diagnostics(samples[host], timeline)
        for host, timeline in timelines.items()
    }
    first_times: list[int] = []
    last_times: list[int] = []
    mapped_streams: dict[str, list[dict[str, Any]]] = {"p450": [], "unitree": []}
    if normalized_summaries:
        for host, entries in normalized_summaries.items():
            timeline = timelines.get(host)
            if timeline is None:
                continue
            for entry in entries:
                try:
                    mapped = _mapped_stream(entry, timeline)
                except (ValueError, TypeError):
                    continue
                mapped_streams.setdefault(host, []).append(mapped)
                first_times.append(mapped["first_desktop_ns"])
                last_times.append(mapped["last_desktop_ns"])
    reasons: list[str] = []
    missing_timelines = [host for host in ("p450", "unitree") if host not in timelines]
    if missing_timelines:
        reasons.append("missing clock samples for " + ", ".join(missing_timelines))
    requested_interval = {
        "start_inclusive_ns": manifest.t0_desktop_ns,
        "end_exclusive_ns": manifest.t1_desktop_ns,
    }
    valid_interval: dict[str, int] | None = None
    if not normalized_summaries:
        reasons.append("no stream summaries")
    elif not all(mapped_streams[host] for host in ("p450", "unitree")):
        reasons.append("missing mapped stream summaries for one or more hosts")
    elif missing_timelines:
        # The mapping data itself is incomplete even if an injected summary is
        # present, so it cannot establish a joint interval.
        pass
    else:
        requested_stop = manifest.t1_desktop_ns if manifest.t1_desktop_ns is not None else min(last_times)
        common_start = max(first_times) + t0_guard_ns
        start_ns = max(common_start, manifest.t0_desktop_ns if manifest.t0_desktop_ns is not None else common_start)
        end_ns = min(requested_stop, min(last_times))
        if start_ns < end_ns:
            valid_interval = {"start_inclusive_ns": start_ns, "end_exclusive_ns": end_ns}
        else:
            reasons.append("stream coverage has no common interval")
    host_errors: dict[str, dict[str, int | None]] = {}
    gap_hosts = set()
    for host in ("p450", "unitree"):
        if clock_errors[host]:
            reasons.append(f"{host}: clock probe outage or invalid records retained in diagnostics")
        if host in timelines:
            gaps = timelines[host].unsupported_gaps
            coverage_start = min(sample.remote_send_wall_ns - sample.offset_ns for sample in samples[host])
            coverage_end = max(sample.remote_send_wall_ns - sample.offset_ns for sample in samples[host])
            outside_coverage = False
            for interval in (requested_interval, valid_interval):
                if interval is None or any(value is None for value in interval.values()):
                    continue
                if any(min(start, end) < interval["end_exclusive_ns"]
                       and max(start, end) > interval["start_inclusive_ns"]
                       for _, _, start, end in gaps):
                    gap_hosts.add(host)
                if (interval["start_inclusive_ns"] < coverage_start - max_clock_gap_ns
                        or interval["end_exclusive_ns"] > coverage_end + max_clock_gap_ns):
                    outside_coverage = True
            if host in gap_hosts:
                reasons.append(f"{host}: unsupported clock sample gap affects requested or valid interval")
            if outside_coverage:
                gap_hosts.add(host)
                reasons.append(f"{host}: requested or valid interval exceeds clock coverage freshness limit")
        endpoint_error = max((entry["estimated_error_ns"] for entry in mapped_streams[host]), default=None)
        interval_error = (
            _mapping_uncertainty_over_interval(timelines[host], valid_interval, endpoint_error)
            if valid_interval is not None and host in timelines and endpoint_error is not None else None
        )
        clock_error = (max(estimates[host].estimated_error_ns, interval_error or 0)
                       if host in estimates else None)
        host_errors[host] = {
            "clock_estimated_error_ns": clock_error,
            "data_endpoint_estimated_error_ns": endpoint_error,
            "interval_mapping_estimated_error_ns": interval_error,
            "estimated_error_ns": (max(clock_error, endpoint_error)
                                   if clock_error is not None and endpoint_error is not None else None),
        }
        if host in gap_hosts:
            # No evidence supports a bounded offset during this outage. Keep
            # converted times for inspection while withdrawing precision claims.
            host_errors[host] = {key: None for key in host_errors[host]}

    def pair_bound(component: str) -> int | None:
        values = [host_errors[host][component] for host in ("p450", "unitree")]
        # Relative alignment subtracts two independently estimated host clocks.
        # Without evidence about correlated errors, absolute bounds must add.
        return sum(values) if all(value is not None for value in values) else None

    clock_error_ns = pair_bound("clock_estimated_error_ns")
    data_endpoint_error_ns = pair_bound("data_endpoint_estimated_error_ns")
    interval_mapping_error_ns = pair_bound("interval_mapping_estimated_error_ns")
    estimated_error_ns = pair_bound("estimated_error_ns")
    single_anchor_hosts = [host for host, timeline in timelines.items() if len(timeline.anchors) == 1]
    if single_anchor_hosts:
        reasons.append("constant-offset mapping has no drift evidence")
    # Any non-single-anchor degraded estimate is bounded extrapolation beyond
    # the timeline domain; make that provenance visible to downstream users.
    if (any(estimate.degraded for host, estimate in estimates.items() if host not in single_anchor_hosts)
            or any(entry["degraded"] for host, values in mapped_streams.items()
                   if host not in single_anchor_hosts for entry in values)):
        reasons.append("timestamp extrapolates beyond mapping domain")
    if interval_mapping_error_ns is not None and interval_mapping_error_ns > 10_000_000:
        reasons.append("valid interval mapping uncertainty exceeds 10 ms")
    if any(error is not None and error > 10_000_000
           for error in (estimated_error_ns, clock_error_ns, data_endpoint_error_ns)):
        reasons.append("estimated error exceeds 10 ms")
    data_errors = []
    required = {
        'p450': [('/uav1/camera/color/image_raw/compressed', '/uav1/camera/color/image_raw'),
                 ('/Odometry',), ('/uav1/mavros/imu/data',)],
        'unitree': [('pkl:front','camera:front'), ('pkl:wrist','camera:wrist'), ('raw/low_state.csv',),
                    ('raw/sport_mode_state.csv',), ('raw/piper_state.csv',),
                    ('raw/piper_status.csv',), ('raw/piper_gamepad.csv',)],
    }
    for host, groups in required.items():
        entries = {s.name: s for s in normalized_summaries.get(host, [])}
        reported_drop_counts = set()
        for alternatives in groups:
            found = [entries[n] for n in alternatives if n in entries]
            if not found:
                data_errors.append(f'{host}: missing required stream {alternatives[0]}')
            for stream in found:
                label = f'{host}:{stream.name}'
                if stream.count < 2:
                    data_errors.append(f'{label}: fewer than two samples')
                if stream.producer_dropped is not None and stream.producer_dropped > 0:
                    reported_drop_counts.add(stream.producer_dropped)
                if stream.max_gap_ns is None or stream.nonmonotonic_count is None:
                    data_errors.append(f'{label}: missing gap/continuity evidence')
                elif stream.max_gap_ns > max_data_gap_ns or stream.nonmonotonic_count:
                    data_errors.append(f'{label}: data gap or non-increasing timestamps')
                elif stream.last_ns-stream.first_ns > max(stream.count-1, 0)*stream.max_gap_ns:
                    data_errors.append(f'{label}: inconsistent count/gap evidence')
                expected_source = ('ros_header' if host == 'p450' else
                                   'camera.wall_time_ns' if stream.name.startswith(('pkl:','camera:')) else 'wall_time_ns')
                if stream.timestamp_source != expected_source:
                    data_errors.append(f'{label}: unverified timestamp source {stream.timestamp_source}')
        for dropped in sorted(reported_drop_counts):
            data_errors.append(f'{host}: recorder dropped {dropped} synchronized records')
    if valid_interval is None:
        data_errors.append('no common data coverage')
    if any(value is None for value in requested_interval.values()):
        data_errors.append('joint start/stop boundaries are unavailable')
    elif valid_interval and valid_interval != requested_interval:
        data_errors.append('data does not cover the complete requested interval')
    reasons.extend(data_errors)
    degraded = bool(reasons)
    report: dict[str, Any] = {
        "episode_id": manifest.episode_id,
        "state": manifest.state.value,
        "label": manifest.label,
        "mode": manifest.mode,
        "requested_interval": requested_interval,
        "coverage_interval": valid_interval,
        "valid_interval": valid_interval if not degraded else None,
        "timestamp_source_rules": {
            "p450": {"preferred_timestamp": "ros_header", "fallback": "bag_timestamp"},
            "unitree": {"frame_timestamp": "camera.wall_time_ns", "pkl_timestamp": "per-camera metadata"},
        },
        "p450": {
            "preferred_timestamp": "ros_header",
            "timestamp_source": "ros_header",
            "fallback_timestamp": "bag_timestamp",
            "streams": mapped_streams["p450"],
        },
        "unitree": {
            "frame_timestamp": "camera.wall_time_ns",
            "timestamp_source": "camera.wall_time_ns",
            "pkl_timestamp": "per-camera metadata",
            "streams": mapped_streams["unitree"],
        },
        "mappings": {
            host: ({**timelines[host].to_mapping_dict(), "reference_estimate": estimates[host].to_dict()}
                   if host in timelines else {"host": host, "unavailable": True})
            for host in ("p450", "unitree")
        },
        "clock_diagnostics": {
            host: {**clock_diagnostics.get(host, {
                "rtt_ns": None, "fit_residual_ns": None, "drift_ppm": None,
                "uncertainty_components": {"network": "rtt_ns", "fit": "fit_residual_ns"},
            }), "errors": clock_errors[host]}
            for host in ("p450", "unitree")
        },
        "quality": {
            "data_validated": not data_errors,
            "data_errors": data_errors,
            "max_data_gap_ns": max_data_gap_ns,
            "estimated_error_ns": estimated_error_ns,
            "clock_estimated_error_ns": clock_error_ns,
            "data_endpoint_estimated_error_ns": data_endpoint_error_ns,
            "interval_mapping_estimated_error_ns": interval_mapping_error_ns,
            "per_host_estimated_error_ns": {
                host: values["estimated_error_ns"] for host, values in host_errors.items()
            },
            "per_host_uncertainty": host_errors,
            "uncertainty_scope": "p450-to-unitree relative alignment; sum of per-host bounds",
            "uncertainty_limitations": [
                "RTT/2 bounds measured network asymmetry under the four-timestamp model; "
                "it does not identify or remove directional network bias.",
                "Between-probe drift and extrapolation use observed clock behavior; "
                "unobserved clock jumps or changes in network asymmetry are not guaranteed bounded.",
            ],
            "degraded": degraded,
            "degradation_reasons": reasons,
            "sample_count": sum(len(values) for values in samples.values()),
        },
    }
    return report


def write_alignment_report(episode_dir: str | Path, **kwargs: Any) -> Path:
    episode_path = Path(episode_dir)
    try:
        report = build_alignment_report(episode_path, **kwargs)
    except ValueError as exc:
        if not any(word in str(exc).lower() for word in ('clock model', 'clock discontinuity')):
            raise
        # Never leave an earlier success report in place after model rejection.
        report = {
            'episode_id': episode_path.name, 'valid_interval': None,
            'coverage_interval': None,
            'quality': {'degraded': True, 'data_validated': False,
                        'estimated_error_ns': None,
                        'degradation_reasons': [str(exc)]},
        }
    destination = episode_path / "alignment.json"
    temporary = destination.with_name(destination.name + ".tmp")
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jointctl")
    parser.add_argument("--config", type=Path, default=_default_config())
    parser.add_argument("--manifest-root", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--instruction")
    start.add_argument("--task")
    for name in ("stop", "status", "recover", "prepare"):
        command = commands.add_parser(name)
        if name == 'stop':
            command.add_argument('--expected-episode', help='refuse automatic Stop if active episode changed')
    monitor = commands.add_parser("monitor")
    monitor.add_argument("--episode-dir", type=Path)
    probe = commands.add_parser("probe")
    probe.add_argument("--host")
    probe.add_argument("--hosts", nargs="+")
    probe.add_argument("--count", type=int)
    probe.add_argument("--duration", type=float)
    align = commands.add_parser("align")
    align.add_argument("episode_id", nargs="?")
    align.add_argument("--episode-dir", type=Path)
    align.add_argument("--no-remote-inspect", action="store_true")
    finalize = commands.add_parser("finalize", help="validate the oldest stopped episode; never export video")
    finalize.add_argument("episode_id", nargs="?")
    return parser


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _status_exit(report: JointStatusReport) -> int:
    if any(not item.reachable for item in report.remotes):
        return EXIT_CONNECTIVITY
    if any(item.last_error for item in report.remotes):
        return EXIT_REMOTE_COMMAND
    if report.message and (report.episode_id is not None or any(item.active for item in report.remotes)):
        return EXIT_CONFLICT
    if report.timing_degraded:
        return EXIT_TIMING
    return EXIT_OK


def _run_alignment_for_finalization(args: argparse.Namespace, episode: EpisodeManifest) -> int:
    return _run(argparse.Namespace(**{
        **vars(args), 'command': 'align', 'episode_id': episode.episode_id,
        'episode_dir': None, 'no_remote_inspect': False,
    }))


def _run(args: argparse.Namespace) -> int:
    if args.command == 'finalize':
        config = _load_config(Path(args.config))
        store = ManifestStore(_manifest_root(args, config))
        controller = make_controller(args)
        print('正在处理待办原始数据与对时记录；不生成视频。', file=sys.stderr, flush=True)
        try:
            completed = finalize_pending(
                store, controller.remotes,
                lambda episode: _run_alignment_for_finalization(args, episode),
                args.episode_id,
            )
        except RuntimeError as exc:
            print(f'后处理失败，原始数据已保留，可稍后重试：{exc}', file=sys.stderr, flush=True)
            return EXIT_REMOTE_COMMAND
        _emit(completed.to_dict())
        print(f'后处理验收通过：{completed.episode_id}', file=sys.stderr, flush=True)
        return EXIT_OK
    if args.command == 'prepare':
        from concurrent.futures import ThreadPoolExecutor
        controller = make_controller(args)
        if controller.store.active() is not None:
            raise ValueError('已有活动会话，请先停止；初始化不会强制清理任何进程。')
        statuses = controller._statuses()
        if any(not s.reachable or s.last_error or s.active or s.episode_id for s in statuses.values()):
            raise ValueError('两端必须连接且无遗留会话，才能初始化。')
        print('正在准备 P450 定位/相机链路与 Unitree CAN；不会解锁或使能机械臂。', file=sys.stderr, flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {host: pool.submit(remote.prepare) for host, remote in controller.remotes.items()}
            results = {host: future.result() for host, future in pending.items()}
        _emit({host: result.to_dict() for host, result in results.items()})
        failed = [host + ': ' + (r.stderr or r.stdout) for host, r in results.items() if not r.ok]
        if failed:
            print('初始化未完成：' + '; '.join(failed), file=sys.stderr, flush=True)
            return EXIT_REMOTE_COMMAND
        print('传感器链路 / CAN 初始化完成。手柄识别与机械臂使能请单独确认；尚未开始录制。', file=sys.stderr, flush=True)
        return EXIT_OK
    if args.command == "start":
        print('联合采集启动程序已打开。将按本机时间自动命名；看到“联合采集已开始”才表示启动成功。',
              file=sys.stderr, flush=True)
        task = args.task or ''
        instruction = args.instruction or task
        manifest = make_controller(args).start(instruction, task)
        _emit(manifest.to_dict())
        if manifest.state == EpisodeState.RECORDING and manifest.t0_desktop_ns is not None:
            time.sleep(max(0, min(1.0, (manifest.t0_desktop_ns - time.time_ns()) / 1e9)))
            print('\n========== 联合采集已开始 ==========\n'
                  f'会话：{manifest.episode_id}\n'
                  '两端数据写入及启动对时检查已通过。\n'
                  '结束时双击 JointStop.bat；停止确认后可开始下一条，数据验收稍后执行。',
                  file=sys.stderr, flush=True)
        return EXIT_OK
    if args.command == "stop":
        expected = getattr(args, 'expected_episode', None)
        if expected is not None:
            config = _load_config(Path(args.config))
            active = ManifestStore(_manifest_root(args, config)).active()
            if active is None or active.episode_id != expected:
                raise ActiveEpisodeConflict('automatic Stop target no longer active: ' + expected)
        manifest = make_controller(args).stop()
        _emit(manifest.to_dict())
        failures = [item for item in manifest.stop_results.values() if not item.ok]
        if any(item.returncode in {-1, 255} for item in failures):
            print("connectivity failure: one or more remote stop commands lost transport", file=sys.stderr)
            return EXIT_CONNECTIVITY
        if failures:
            print("remote command failed: one or more stop commands failed", file=sys.stderr)
            return EXIT_REMOTE_COMMAND
        if (manifest.state != EpisodeState.COMPLETE
                or manifest.clock_monitor_closed_cleanly is False
                or len(manifest.stop_results) != 2):
            print("timing/monitoring degradation: stop is incomplete or clock monitor did not close cleanly", file=sys.stderr)
            return EXIT_TIMING
        print('\n========== 两端录制已停止 · 待后处理 ==========\n'
              '原始数据和对时记录已保留，可开始下一轮采集。'
              '请在设备空闲时点击“处理待办”完成数据验收；停止成功不代表验收通过。',
              file=sys.stderr, flush=True)
        for host, directory in manifest.remote_directories.items():
            print(f'{host} 数据目录：{directory}', file=sys.stderr, flush=True)
        return EXIT_OK
    if args.command == "status":
        report = make_controller(args).status(); _emit(report.to_dict())
        status_code = _status_exit(report)
        if status_code == EXIT_CONFLICT:
            print(f"episode mismatch: {report.message or 'remote episode conflict'}", file=sys.stderr)
        elif status_code == EXIT_CONNECTIVITY:
            print("connectivity failure: one or more remotes are unreachable", file=sys.stderr)
        elif status_code == EXIT_REMOTE_COMMAND:
            print("remote command failed: status command failed", file=sys.stderr)
        elif status_code == EXIT_TIMING:
            print("timing degradation: " + "; ".join(report.timing_degradation_reasons), file=sys.stderr)
        return status_code
    if args.command == "recover":
        manifest = make_controller(args).recover(); _emit(manifest.to_dict()); return EXIT_OK
    if args.command == "probe":
        if args.host and args.hosts:
            raise ValueError("use either --host or --hosts, not both")
        hosts = tuple(args.hosts or ([args.host] if args.host else ()))
        if not hosts:
            raise ValueError("probe requires --host or --hosts")
        if args.duration is not None:
            if args.count is not None:
                raise ValueError("use either --duration or --count, not both")
            config = _load_config(Path(args.config))
            samples, errors = run_probe_session(
                hosts, args.duration, float(config.get("clock_interval_s", 0.2))
            )
            _emit(build_probe_report(samples, errors, args.duration))
            if any(not values for values in samples.values()):
                failure_kinds = {
                    error.get("failure_kind")
                    for values in errors.values() for error in values
                }
                return EXIT_CONNECTIVITY if "connectivity" in failure_kinds else EXIT_REMOTE_COMMAND
            return EXIT_OK
        count = 1 if args.count is None else args.count
        if count <= 0:
            raise ValueError("--count must be positive")
        if len(hosts) != 1:
            raise ValueError("--hosts requires --duration")
        with ClockProbe(hosts[0]) as probe:
            for _ in range(count):
                print(json.dumps(probe.sample().to_dict(), sort_keys=True))
        return EXIT_OK
    if args.command == "monitor":
        config = _load_config(Path(args.config)); root = _manifest_root(args, config)
        episode = args.episode_dir or (ManifestStore(root).active() and ManifestStore(root).episode_dir(ManifestStore(root).active().episode_id))
        if episode is None: raise ValueError("monitor requires an active episode or --episode-dir")
        hosts = ("p450", "unitree")
        destinations = {kind: str(config.get(kind, {}).get("host", kind)) for kind in hosts}
        interval = float(config.get("clock_interval_s", 0.2))
        return run_monitor(Path(episode), hosts, interval, Path(episode) / "clock.stop",
                           destinations=destinations, timeout_s=float(config.get("ssh_timeout_s", 10.0)))
    if args.command == "align":
        config = _load_config(Path(args.config)); root = _manifest_root(args, config)
        store = ManifestStore(root)
        episode = args.episode_dir
        if episode is None:
            if args.episode_id:
                store.load(args.episode_id)
                episode = store.episode_dir(args.episode_id)
            else:
                active = store.active()
                if active is None: raise ValueError("align requires an active episode, episode ID, or --episode-dir")
                episode = store.episode_dir(active.episode_id)
        stream_summaries: dict[str, list[StreamSummary]] = {}
        if not args.no_remote_inspect:
            manifest = EpisodeManifest.from_dict(json.loads((Path(episode) / "manifest.json").read_text(encoding="utf-8")))
            timeout = float(config.get("inspect_timeout_s", 300.0))
            p450_dir = manifest.remote_directories.get("p450")
            unitree_dir = manifest.remote_directories.get("unitree")
            if p450_dir: stream_summaries["p450"] = P450Inspector(str(config.get("p450", {}).get("host", "p450")), timeout_s=timeout).summarize(p450_dir)
            if unitree_dir: stream_summaries["unitree"] = UnitreeInspector(str(config.get("unitree", {}).get("host", "unitree")), timeout_s=timeout).summarize(unitree_dir)
        try:
            output = write_alignment_report(
                episode, stream_summaries=stream_summaries,
                alignment_window_ns=int(config.get("alignment_window_ns", 5_000_000_000)),
                t0_guard_ns=int(config.get("t0_guard_ns", 0)),
                max_clock_gap_ns=int(config.get("max_clock_gap_ns", DEFAULT_MAX_CLOCK_GAP_NS)),
            )
        except ValueError as exc:
            if any(word in str(exc).lower() for word in ('interval', 'stop request', 'clock')):
                print(f"timing degradation: {exc}", file=sys.stderr)
                return EXIT_TIMING
            raise
        report = json.loads(output.read_text(encoding="utf-8")); _emit(report)
        return EXIT_TIMING if report["quality"].get("degraded") else EXIT_OK
    raise ValueError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        try:
            if args.command in {'start', 'stop', 'recover', 'align', 'prepare', 'finalize'}:
                config = _load_config(Path(args.config))
                with operation_lock(_manifest_root(args, config)):
                    return _run(args)
            return _run(args)
        except OperationBusy as exc:
            print(f'operation busy: {exc}', file=sys.stderr); return EXIT_CONFLICT
        except (RemoteAlreadyActive, ActiveEpisodeConflict) as exc:
            print(f"active-state conflict: {exc}", file=sys.stderr); return EXIT_CONFLICT
        except (EpisodeMismatch, RecoveryConflict) as exc:
            statuses = getattr(exc, "statuses", ())
            if any(not item.reachable for item in statuses):
                print(f"connectivity failure: {exc}", file=sys.stderr); return EXIT_CONNECTIVITY
            if any(item.last_error for item in statuses):
                print(f"remote command failed: {exc}", file=sys.stderr); return EXIT_REMOTE_COMMAND
            print(f"episode mismatch: {exc}", file=sys.stderr); return EXIT_CONFLICT
        except JointStartFailed as exc:
            code = (EXIT_CONNECTIVITY if exc.failure_kind == "connectivity"
                    else EXIT_REMOTE_COMMAND)
            print(f"remote operation failed: {exc}", file=sys.stderr); return code
        except InspectorConnectionError as exc:
            print(f"connectivity failure: {exc}", file=sys.stderr); return EXIT_CONNECTIVITY
        except InspectorError as exc:
            print(f"remote command failed: {exc}", file=sys.stderr); return EXIT_REMOTE_COMMAND
        except ClockProbeError as exc:
            code = (EXIT_CONNECTIVITY if exc.failure_kind == "connectivity"
                    else EXIT_REMOTE_COMMAND)
            print(f"clock probe failed: {exc}", file=sys.stderr); return code
        except (ConnectionError, TimeoutError) as exc:
            print(f"connectivity failure: {exc}", file=sys.stderr); return EXIT_CONNECTIVITY
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"configuration/usage error: {exc}", file=sys.stderr); return EXIT_USAGE
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
