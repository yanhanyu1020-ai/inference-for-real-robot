"""Recoverable orchestration for starting a joint P450/Unitree episode."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import secrets
import time
from typing import Callable, Mapping

from .alignment import build_clock_timeline
from .clock_sync import ClockProbe, launch_detached_monitor, read_clock_records, DEFAULT_CLOCK_FRESHNESS_NS
from .manifest import ManifestStore, ActiveEpisodeConflict
from .models import (
    ClockEstimate,
    ClockSample,
    CommandResult,
    DiagnosticRecord,
    EpisodeManifest,
    EpisodeState,
    JointStatusReport,
    RemoteStatus,
)
from .remote import RemoteClient, json_objects


class RemoteAlreadyActive(RuntimeError):
    """Raised when preflight finds an existing remote capture episode."""


class JointStartFailed(RuntimeError):
    """Raised after a failed joint start has been durably diagnosed and rolled back."""

    def __init__(self, message: str, episode_id: str | None = None,
                 *, failure_kind: str = "remote_command") -> None:
        if failure_kind not in {"connectivity", "remote_command"}:
            raise ValueError("failure_kind must be connectivity or remote_command")
        super().__init__(message)
        self.episode_id = episode_id
        self.failure_kind = failure_kind


class _StartOperationError(RuntimeError):
    """Internal failure with transport/remote-command provenance."""

    def __init__(self, message: str, failure_kind: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


class EpisodeMismatch(RuntimeError):
    """Raised before stop when remote ownership does not match the manifest."""

    def __init__(self, message: str, report: JointStatusReport) -> None:
        super().__init__(message)
        self.report = report
        self.statuses = report.remotes


class RecoveryConflict(RuntimeError):
    """Raised when remote state cannot identify one recoverable episode."""

    def __init__(self, message: str, report: JointStatusReport) -> None:
        super().__init__(message)
        self.report = report
        self.statuses = report.remotes


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def generate_episode_id(now: datetime, random_bytes: bytes) -> str:
    """Return a desktop-local timestamp, UTC offset, and collision suffix."""
    if len(random_bytes) < 2:
        raise ValueError("at least two random bytes are required")
    local_now = now if now.tzinfo is not None else now.astimezone()
    offset = local_now.strftime('%z')
    direction = 'p' if offset.startswith('+') else 'm'
    return f"joint_{local_now:%Y%m%d_%H%M%S}_UTC{direction}{offset[1:]}_{random_bytes[:2].hex()}"


class JointController:
    """Coordinate two remote capture adapters through durable state transitions."""

    def __init__(
        self,
        p450: RemoteClient,
        unitree: RemoteClient,
        store: ManifestStore,
        clock_probes: Mapping[str, ClockProbe],
        *,
        monitor_launcher: Callable[[Path, Path], int] = launch_detached_monitor,
        now: Callable[[], datetime] | None = None,
        random_bytes: Callable[[int], bytes] = secrets.token_bytes,
        time_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        pid_is_running: Callable[[int], bool] = _process_is_running,
        readiness_timeout_s: float = 10.0,
        poll_interval_s: float = 0.2,
        monitor_stop_timeout_s: float = 5.0,
        clock_interval_s: float = 0.2,
        ssh_timeout_s: float = 10.0,
        clock_freshness_ns: int = DEFAULT_CLOCK_FRESHNESS_NS,
        alignment_window_ns: int = 5_000_000_000,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if readiness_timeout_s <= 0:
            raise ValueError("readiness_timeout_s must be positive")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if monitor_stop_timeout_s <= 0:
            raise ValueError("monitor_stop_timeout_s must be positive")
        if min(clock_interval_s, ssh_timeout_s, clock_freshness_ns, alignment_window_ns) <= 0:
            raise ValueError("clock cadence, timeout, freshness and window must be positive")
        if set(clock_probes) != {"p450", "unitree"}:
            raise ValueError("clock_probes must contain p450 and unitree")
        self.remotes = {"p450": p450, "unitree": unitree}
        self.store = store
        self.clock_probes = dict(clock_probes)
        self.monitor_launcher = monitor_launcher
        self.now = now or (lambda: datetime.now().astimezone())
        self.random_bytes = random_bytes
        self.time_ns = time_ns
        self.monotonic = monotonic
        self.sleep = sleep
        self.pid_is_running = pid_is_running
        self.readiness_timeout_s = readiness_timeout_s
        self.poll_interval_s = poll_interval_s
        self.monitor_stop_timeout_s = monitor_stop_timeout_s
        self.clock_interval_s = clock_interval_s
        self.ssh_timeout_s = ssh_timeout_s
        self.clock_freshness_ns = clock_freshness_ns
        self.alignment_window_ns = alignment_window_ns
        self.progress = progress or (lambda _message: None)

    def start(self, instruction: str, task: str) -> EpisodeManifest:
        existing = self.store.active()
        if existing is not None:
            raise ActiveEpisodeConflict(f"active episode already exists: {existing.episode_id}")
        self.progress('[1/4] 正在检查 P450 和 Unitree 连接，请等待...')
        preflight = self._statuses()
        for attempt in range(2):
            # Only retry read-only checks. Never replay start commands, and
            # never overlook an already-active remote while retrying a peer.
            if any(status.active for status in preflight.values()):
                break
            if all(status.reachable and not status.last_error for status in preflight.values()):
                break
            self.sleep(self.poll_interval_s)
            self.progress(f'连接检查未通过，正在重试 ({attempt + 2}/3)...')
            preflight = self._statuses()
        active = [status for status in preflight.values() if status.active]
        if active:
            details = ", ".join(
                f"{status.host}={status.episode_id or '<unknown>'}" for status in active
            )
            raise RemoteAlreadyActive(f"remote episode already active: {details}")
        failures = [status for status in preflight.values()
                    if not status.reachable or status.last_error]
        if failures:
            details = "; ".join(
                f"{status.host}: {status.last_error or 'unreachable'}" for status in failures
            )
            kind = "connectivity" if any(not status.reachable for status in failures) else "remote_command"
            raise JointStartFailed(f"remote preflight failed: {details}", failure_kind=kind)

        episode_id = generate_episode_id(self.now(), self.random_bytes(2))
        task = task or episode_id
        instruction = instruction or task
        manifest = EpisodeManifest(
            episode_id=episode_id,
            label=instruction,
            mode=task,
            created_desktop_ns=self.time_ns(),
            metadata={'starter_pid': os.getpid()},
        )
        self.store.create(manifest)

        start_results: dict[str, CommandResult] = {}
        remote_directories: dict[str, str] = {}
        diagnostics: list[DiagnosticRecord] = []
        try:
            monitor_pid = self._launch_monitor(episode_id)
            self.store.update(episode_id, clock_monitor_pid=monitor_pid)

            self.progress('[2/4] 正在准备 P450 传感器链路、Unitree CAN 并启动采集；冷启动可能需要数分钟，请勿关闭窗口。')
            start_results = self._starts(episode_id, instruction, task)
            remote_directories.update(self._directories_from_results(start_results))
            self.store.update(
                episode_id,
                start_results=start_results,
                remote_directories=remote_directories,
            )
            failed_starts = {host: result for host, result in start_results.items()
                             if not result.ok}
            if failed_starts:
                detail = "; ".join(
                    f"{host}: {result.stderr or result.stdout or f'exit {result.returncode}'}"
                    for host, result in failed_starts.items()
                )
                kind = ("connectivity" if any(result.returncode in {-1, 255}
                                                for result in failed_starts.values())
                        else "remote_command")
                raise _StartOperationError(detail, kind)

            self.progress('[3/4] 启动命令已返回，正在确认两端数据持续写入...')
            ready_statuses = self._wait_until_ready(episode_id)
            for host, status in ready_statuses.items():
                if status.state_gap_ticks:
                    note = f'{host}: 启动预录存在状态缺口，当前已恢复；原始缺口保留，整段完整性仍须离线验收'
                    diagnostics.append(DiagnosticRecord(phase='readiness', host=host,
                        message=note, at_desktop_ns=self.time_ns()))
                    self.progress(note)
            remote_directories.update(self._directories_from_statuses(ready_statuses))
            self.progress('[4/4] 正在检查对时质量并建立共同起点...')
            clock_samples = self._ready_clock_samples()
            clock_estimates = self._clock_estimates(clock_samples)
            t0_desktop_ns = self.time_ns() + 500_000_000
            return self.store.update(
                episode_id,
                state=EpisodeState.RECORDING,
                start_results=start_results,
                remote_directories=remote_directories,
                t0_desktop_ns=t0_desktop_ns,
                clock_samples=clock_samples,
                clock_estimates=clock_estimates,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            self.progress(f'启动未完成：{exc}。正在清理本次已启动的采集，请等待...')
            diagnostics.append(DiagnosticRecord(
                phase="start", message=str(exc), at_desktop_ns=self.time_ns()
            ))
            rollback_statuses, rollback_results, rollback_diagnostics = self._rollback(
                episode_id
            )
            diagnostics.extend(rollback_diagnostics)
            cleanup_complete = all(
                (host in rollback_results and rollback_results[host].ok)
                or (status.reachable and not status.last_error and not status.active
                    and status.episode_id is None)
                for host, status in rollback_statuses.items()
            ) and len(rollback_statuses) == len(self.remotes)
            if not cleanup_complete:
                self.progress('清理尚未确认：保留本次会话。恢复连接后请点击停止并校验，勿重复开始。')
            self.store.update(
                episode_id,
                state=EpisodeState.START_FAILED if cleanup_complete else EpisodeState.PARTIAL,
                start_results=start_results,
                remote_directories=remote_directories,
                rollback_statuses=rollback_statuses,
                rollback_results=rollback_results,
                diagnostics=diagnostics,
            )
            kind = getattr(exc, "failure_kind", "remote_command")
            raise JointStartFailed(str(exc), episode_id, failure_kind=kind) from exc

    def status(self) -> JointStatusReport:
        manifest = self.store.active()
        statuses = self._statuses()
        remotes = [statuses[host] for host in ("p450", "unitree")]
        if manifest is not None:
            mismatched = [
                status for status in remotes
                if (not status.reachable or status.last_error or not status.active
                    or status.episode_id != manifest.episode_id)
            ]
            message = ""
            if mismatched:
                message = "remote ownership does not exactly match the active manifest"
            estimates, health, reasons = self._clock_health(manifest)
            reported_state = manifest.state
            starter_pid = manifest.metadata.get('starter_pid')
            if (manifest.state == EpisodeState.STARTING and isinstance(starter_pid, int)
                    and not self.pid_is_running(starter_pid)):
                reported_state = EpisodeState.PARTIAL
                message = 'startup interrupted before common T0; stop this episode before starting again'
                reasons.append(message)
            return JointStatusReport(
                episode_id=manifest.episode_id,
                state=reported_state,
                remotes=remotes,
                clock_estimates=estimates,
                message=message,
                clock_health=health,
                timing_degraded=bool(reasons),
                timing_degradation_reasons=reasons,
            )

        active_ids = [status.episode_id for status in remotes if status.active]
        if (len(active_ids) == 2 and active_ids[0] is not None
                and active_ids[0] == active_ids[1]
                and all(status.reachable and not status.last_error for status in remotes)):
            return JointStatusReport(
                episode_id=active_ids[0],
                state=EpisodeState.RECOVERED,
                remotes=remotes,
                message="matching remote episode can be recovered",
            )
        if active_ids or any(not status.reachable or status.last_error for status in remotes):
            return JointStatusReport(
                episode_id=None,
                state=EpisodeState.PARTIAL,
                remotes=remotes,
                message="remote states do not identify one recoverable episode",
            )
        return JointStatusReport(
            episode_id=None,
            state=EpisodeState.COMPLETE,
            remotes=remotes,
            message="no active episode",
        )

    def stop(self) -> EpisodeManifest:
        requested_t1 = self.time_ns()
        self.progress('[1/4] 已收到停止请求，正在核对两端采集会话...')
        manifest = self.store.active()
        statuses = self._statuses()
        remotes = [statuses[host] for host in ("p450", "unitree")]
        if manifest is None:
            report = JointStatusReport(
                episode_id=None,
                state=EpisodeState.PARTIAL,
                remotes=remotes,
                message="no active local episode",
            )
            raise EpisodeMismatch(report.message, report)
        starter_pid = manifest.metadata.get('starter_pid')
        if (manifest.state == EpisodeState.STARTING and isinstance(starter_pid, int)
                and self.pid_is_running(starter_pid)):
            report = JointStatusReport(manifest.episode_id, manifest.state, remotes,
                message='startup is still running; wait for it to finish before cleanup')
            raise EpisodeMismatch(report.message, report)
        if any(status.reachable and not status.last_error and (
            (status.episode_id is not None and status.episode_id != manifest.episode_id)
            or (status.active and status.episode_id is None)) for status in remotes):
            report = JointStatusReport(
                episode_id=manifest.episode_id,
                state=manifest.state,
                remotes=remotes,
                clock_estimates=list(manifest.clock_estimates.values()),
                message="remote ownership does not exactly match the active manifest",
            )
            raise EpisodeMismatch(report.message, report)

        t1_desktop_ns = manifest.t1_desktop_ns if manifest.t1_desktop_ns is not None else requested_t1
        stopping = self.store.update(
            manifest.episode_id,
            state=EpisodeState.STOPPING,
            t1_desktop_ns=t1_desktop_ns,
        )
        targets = []
        stop_results = dict(manifest.stop_results)
        diagnostics = list(stopping.diagnostics)
        for host, status in statuses.items():
            if not status.reachable or status.last_error:
                previous = manifest.stop_results.get(host)
                stop_results[host] = previous if previous and previous.ok else CommandResult(
                    f'{host} stop deferred', -1,
                    stderr=status.last_error or 'host unreachable; retry stop when connected')
            elif status.episode_id == manifest.episode_id:
                # A stopped P450 recorder may still need finish/export. Its
                # capture marker proves ownership even though active is false.
                targets.append(host)
                if not status.active and host not in manifest.stop_results:
                    diagnostics.append(DiagnosticRecord(phase='early_exit', host=host,
                        message='recorder exited before joint stop; validate data coverage',
                        at_desktop_ns=requested_t1))
            else:
                # Verified idle with no episode: never send an unowned stop.
                previous = manifest.stop_results.get(host)
                stop_results[host] = previous if previous and previous.ok else CommandResult(
                    f'{host} verified idle', 0, stdout='no active episode; no stop command sent')
                if previous is None or not previous.ok:
                    diagnostics.append(DiagnosticRecord(phase='early_exit', host=host,
                        message='already idle; collection finalization is not verified',
                        at_desktop_ns=requested_t1))
        self.progress('[2/3] 正在停止两端录制并保存原始数据；视频导出和完整校验留待后处理。')
        stop_results.update(self._stops(targets))
        raw_capture = manifest.metadata.get('raw_capture')
        observed = self.store.episode_dir(manifest.episode_id) / 'raw_health.json'
        try:
            saved = json.loads(observed.read_text(encoding='utf-8'))
            if isinstance(saved, dict) and saved.get('raw_episode_id') == manifest.episode_id:
                raw_capture = saved
        except (OSError, ValueError):
            pass
        unitree_status = statuses['unitree']
        if unitree_status.episode_id == manifest.episode_id and unitree_status.format_version == 2:
            for obj in json_objects(unitree_status.message):
                if obj.get('format_version') == 2:
                    raw_capture = {**obj, 'raw_episode_id': manifest.episode_id}
        result = stop_results.get('unitree')
        prior_faults = list((raw_capture or {}).get('fault') or [])
        if result:
            for obj in json_objects(result.stdout):
                if obj.get('format_version') == 2 and obj.get('raw_episode_id') == manifest.episode_id:
                    raw_capture = {**obj, 'fault': list(dict.fromkeys(prior_faults + (obj.get('fault') or [])))}
        if raw_capture:
            for message in raw_capture.get('fault') or []:
                if not any(d.phase == 'raw_capture' and d.message == message for d in diagnostics):
                    diagnostics.append(DiagnosticRecord(phase='raw_capture', host='unitree',
                        message=message, at_desktop_ns=requested_t1))
        self.progress('[3/3] 远端停止命令已返回，正在收尾对时记录并检查停止结果...')
        monitor_closed = self._stop_monitor(stopping)
        for host, result in stop_results.items():
            if not result.ok:
                diagnostics.append(DiagnosticRecord(
                    phase="stop",
                    host=host,
                    message=result.stderr or result.stdout or f"exit {result.returncode}",
                    at_desktop_ns=self.time_ns(),
                ))
        if not monitor_closed:
            diagnostics.append(DiagnosticRecord(
                phase="clock_monitor",
                message=("clock monitor did not exit within "
                         f"{self.monitor_stop_timeout_s:g} seconds"),
                at_desktop_ns=self.time_ns(),
            ))
        successes = sum(result.ok for result in stop_results.values())
        if successes == len(self.remotes) and monitor_closed:
            final_state = EpisodeState.COMPLETE
        elif successes == 0:
            final_state = EpisodeState.STOP_FAILED
        else:
            final_state = EpisodeState.PARTIAL
        return self.store.update(
            manifest.episode_id,
            state=final_state,
            metadata={**manifest.metadata, **({'raw_capture': raw_capture} if raw_capture else {}),
                      **({'postprocess': 'pending'} if final_state == EpisodeState.COMPLETE else {})},
            stop_results=stop_results,
            clock_monitor_closed_cleanly=monitor_closed,
            diagnostics=diagnostics,
        )

    def recover(self) -> EpisodeManifest:
        existing = self.store.active()
        statuses = self._statuses()
        remotes = [statuses[host] for host in ("p450", "unitree")]
        episode_ids = [status.episode_id for status in remotes]
        if existing is not None:
            if all(not s.reachable or s.last_error or (
                s.episode_id in (None, existing.episode_id) and not (s.active and s.episode_id is None)
            ) for s in remotes):
                directories = dict(existing.remote_directories)
                directories.update(self._directories_from_statuses({
                    h: s for h, s in statuses.items() if s.reachable and not s.last_error
                    and s.episode_id == existing.episode_id}))
                return self.store.update(existing.episode_id, remote_directories=directories)
        else:
            identified = {s.episode_id for s in remotes if s.reachable and not s.last_error and s.episode_id}
            if len(identified) == 1:
                candidate_id = next(iter(identified))
                try:
                    known = self.store.load(candidate_id)
                except FileNotFoundError:
                    pass
                else:
                    if all(not s.reachable or s.last_error or (
                        s.episode_id in (None, candidate_id) and not (s.active and s.episode_id is None)
                    ) for s in remotes):
                        directories = dict(known.remote_directories)
                        directories.update(self._directories_from_statuses({
                            h: s for h, s in statuses.items() if s.reachable and not s.last_error
                            and s.episode_id == candidate_id}))
                        return self.store.restore(replace(known, state=EpisodeState.RECOVERED,
                                                          remote_directories=directories))
        matches = (
            existing is None
            and all(status.reachable and not status.last_error and status.active
                    for status in remotes)
            and episode_ids[0] is not None
            and episode_ids[0] == episode_ids[1]
        )
        if not matches:
            report = JointStatusReport(
                episode_id=existing.episode_id if existing is not None else None,
                state=existing.state if existing is not None else EpisodeState.PARTIAL,
                remotes=remotes,
                clock_estimates=(list(existing.clock_estimates.values())
                                 if existing is not None else []),
                message="remote states do not identify one recoverable episode",
            )
            raise RecoveryConflict(report.message, report)
        episode_id = episode_ids[0]
        assert episode_id is not None
        observed_directories = self._directories_from_statuses(statuses)
        try:
            orphaned = self.store.load(episode_id)
        except FileNotFoundError:
            recovered = EpisodeManifest(
                episode_id=episode_id,
                label="recovered",
                mode="joint",
                created_desktop_ns=self.time_ns(),
                state=EpisodeState.RECOVERED,
                remote_directories=observed_directories,
            )
        else:
            recovered = replace(
                orphaned,
                state=EpisodeState.RECOVERED,
                remote_directories={**orphaned.remote_directories, **observed_directories},
            )
        return self.store.restore(recovered)

    def _statuses(self) -> dict[str, RemoteStatus]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {host: executor.submit(remote.status)
                       for host, remote in self.remotes.items()}
            statuses: dict[str, RemoteStatus] = {}
            for host, future in futures.items():
                try:
                    statuses[host] = future.result()
                except Exception as exc:
                    statuses[host] = RemoteStatus(
                        host, False, last_error=f"status raised: {exc}"
                    )
            return statuses

    def _clock_health(self, manifest: EpisodeManifest):
        samples, errors = read_clock_records(self.store.episode_dir(manifest.episode_id), manifest)
        now_ns = self.time_ns()
        monitor_alive = (manifest.clock_monitor_pid is not None
                         and self.pid_is_running(manifest.clock_monitor_pid))
        reasons = [] if monitor_alive else ["clock monitor is missing or no longer running"]
        estimates = []
        health = {}
        for host, values in samples.items():
            latest = max(values, key=lambda sample: sample.local_receive_wall_ns) if values else None
            age = now_ns - latest.local_receive_wall_ns if latest is not None else None
            host_reasons = []
            if age is None or age < 0 or age > self.clock_freshness_ns:
                host_reasons.append("clock samples are missing, stale, or ahead of desktop time")
            # Use the newest observation's sliding window even when stale, so
            # operators see the last measured offset and its actual age.
            recent = [sample for sample in values if latest is not None
                      and sample.local_receive_wall_ns >= latest.local_receive_wall_ns - self.alignment_window_ns]
            recent_errors = [error for error in errors[host]
                             if not isinstance(error.get("at_wall_ns"), int)
                             or error["at_wall_ns"] >= now_ns - self.clock_freshness_ns]
            if recent_errors:
                host_reasons.append("clock probe errors in freshness window")
            min_rtt = min((sample.rtt_ns for sample in recent), default=None)
            initial = manifest.clock_samples.get(host)
            if min_rtt is not None and initial is not None and min_rtt > initial.rtt_ns + 2_000_000:
                host_reasons.append("minimum clock RTT worsened by more than 2 ms")
            if recent:
                try:
                    timeline = build_clock_timeline(recent, self.alignment_window_ns)
                except ValueError as exc:
                    timeline = None
                    host_reasons.append(f'clock model unavailable: {exc}')
            else:
                timeline = None
            if timeline is not None:
                estimate = timeline.remote_to_desktop(latest.remote_send_wall_ns, allow_single_anchor=True)
                # Estimate current offset from the sliding low-RTT window. A
                # single window is sufficient for live offset health, not drift.
                anchor = timeline.anchors[-1]
                estimate = replace(estimate, estimated_error_ns=anchor.uncertainty_ns,
                                   jitter_ns=anchor.uncertainty_ns, degraded=False)
                if estimate.estimated_error_ns > 10_000_000:
                    host_reasons.append("clock uncertainty exceeds 10 ms")
                estimates.append(replace(estimate, degraded=bool(host_reasons) or not monitor_alive))
            health[host] = {"sample_age_ns": age, "monitor_alive": monitor_alive,
                            "minimum_rtt_ns": min_rtt, "errors": recent_errors,
                            "freshness_limit_ns": self.clock_freshness_ns,
                            "degradation_reasons": host_reasons}
            reasons.extend(f"{host}: {reason}" for reason in host_reasons)
        if len(estimates) == 2 and sum(item.estimated_error_ns for item in estimates) > 10_000_000:
            reasons.append("relative clock uncertainty exceeds 10 ms")
        return estimates, health, reasons

    def _starts(self, episode_id: str, instruction: str,
                task: str) -> dict[str, CommandResult]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                host: executor.submit(remote.start, episode_id, instruction, task)
                for host, remote in self.remotes.items()
            }
            results: dict[str, CommandResult] = {}
            for host, future in futures.items():
                try:
                    results[host] = future.result()
                except Exception as exc:
                    results[host] = CommandResult(
                        f"{host} start {episode_id}", -1, stderr=f"start raised: {exc}"
                    )
            return results

    def _stops(self, hosts=None) -> dict[str, CommandResult]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {host: executor.submit(remote.stop)
                       for host, remote in self.remotes.items() if hosts is None or host in hosts}
            results: dict[str, CommandResult] = {}
            for host, future in futures.items():
                try:
                    results[host] = future.result()
                except Exception as exc:
                    results[host] = CommandResult(
                        f"{host} stop", -1, stderr=f"stop raised: {exc}"
                    )
            return results

    def _stop_monitor(self, manifest: EpisodeManifest) -> bool:
        stop_file = self.store.episode_dir(manifest.episode_id) / "clock.stop"
        stop_file.touch(exist_ok=True)
        if manifest.clock_monitor_pid is None:
            return True
        deadline = self.monotonic() + self.monitor_stop_timeout_s
        while self.pid_is_running(manifest.clock_monitor_pid):
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                return False
            self.sleep(min(self.poll_interval_s, remaining))
        return True

    def _launch_monitor(self, episode_id: str) -> int:
        manifest_dir = self.store.episode_dir(episode_id)
        config_path = manifest_dir / "clock_monitor_config.json"
        temporary = config_path.with_suffix(config_path.suffix + ".tmp")
        payload = {
            "hosts": ["p450", "unitree"],
            "destinations": {host: remote.host for host, remote in self.remotes.items()},
            "manifest_file": self.store.manifest_path(episode_id).name,
            "clock_interval_s": self.clock_interval_s,
            "ssh_timeout_s": self.ssh_timeout_s,
            "clock_stop_file": "clock.stop",
        }
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, config_path)
        pid = self.monitor_launcher(manifest_dir, config_path)
        if not isinstance(pid, int) or pid <= 0:
            raise RuntimeError("clock monitor launcher returned an invalid PID")
        return pid

    def _wait_until_ready(self, episode_id: str) -> dict[str, RemoteStatus]:
        previous: dict[str, int | tuple[int, int]] = {}
        streaks = {"p450": 0, "unitree": 0}
        deadline = self.monotonic() + self.readiness_timeout_s
        max_polls = math.ceil(self.readiness_timeout_s / self.poll_interval_s) + 1
        expected_progress = {"p450": "session_bytes", "unitree": "frames_saved"}
        last_failure = None
        for poll in range(max_polls):
            statuses = self._statuses()
            for host, status in statuses.items():
                if status.active and status.episode_id not in (None, episode_id):
                    raise _StartOperationError(f'{host} belongs to another episode', 'remote_command')
                if not status.reachable:
                    last_failure = _StartOperationError(
                        f"{host} readiness status does not match {episode_id}: "
                        f"{status.last_error or status.episode_id or status.state}", "connectivity"
                    )
                    streaks[host] = 0
                    previous.pop(host, None)
                    continue
                if status.last_error:
                    last_failure = _StartOperationError(
                        f"{host} readiness status does not match {episode_id}: "
                        f"{status.last_error}", "remote_command"
                    )
                    streaks[host] = 0
                    previous.pop(host, None)
                    continue
                if not status.active or status.episode_id != episode_id:
                    raise _StartOperationError(
                        f"{host} readiness status does not match {episode_id}: "
                        f"{status.episode_id or status.state}", "remote_command"
                    )
                if status.format_version == 2 and host == 'unitree':
                    if status.fault:
                        raise _StartOperationError('; '.join(status.fault), 'remote_command')
                    if any(s.get('rejected', 0) or s.get('write_errors', 0) for s in status.streams.values()):
                        raise _StartOperationError('unitree: camera rejected records or write errors; 原始数据不完整', 'remote_command')
                    if status.quality_ok is not True and not status.state_gap_ticks:
                        raise _StartOperationError('unitree: 数据质量异常，无法确认启动；请检查原始状态', 'remote_command')
                    # Lifetime quality remains false after a pre-roll state gap.
                    # Current readiness must recover, but never erase that history.
                    if status.ready is not True:
                        last_failure = _StartOperationError('unitree: 等待状态恢复: ' +
                            ', '.join(status.missing_states or ['状态未就绪']), 'remote_command')
                        streaks[host] = 0
                        previous.pop(host, None)
                        continue
                    current = tuple(status.streams[n]['written'] for n in ('front', 'wrist'))
                    if host in previous:
                        streaks[host] = streaks[host] + 1 if all(a > b for a,b in zip(current,previous[host])) else 0
                    previous[host] = current
                    continue
                if status.progress_name != expected_progress[host]:
                    raise _StartOperationError(
                        f"{host} readiness status lacks {expected_progress[host]}", "remote_command"
                    )
                if host in previous:
                    streaks[host] = streaks[host] + 1 if (
                        status.progress_value > previous[host]
                    ) else 0
                previous[host] = status.progress_value
            if all(streak >= 2 for streak in streaks.values()):
                return statuses
            if poll + 1 >= max_polls or self.monotonic() >= deadline:
                break
            self.sleep(self.poll_interval_s)
        if last_failure is not None:
            raise last_failure
        raise RuntimeError("capture readiness timed out before two consecutive increases")

    def _clock_samples(self) -> dict[str, ClockSample]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {host: executor.submit(probe.sample)
                       for host, probe in self.clock_probes.items()}
            return {host: future.result() for host, future in futures.items()}

    def _ready_clock_samples(self) -> dict[str, ClockSample]:
        history = {host: [] for host in self.clock_probes}
        deadline = self.monotonic() + 10.0
        for attempt in range(20):
            for host, sample in self._clock_samples().items():
                history[host].append(sample)
            if attempt >= 2:
                # First exchanges include SSH establishment; never let one
                # handshake sample be the only startup quality evidence.
                for values in history.values():
                    build_clock_timeline(sorted(values, key=lambda s: s.remote_send_wall_ns),
                                         self.alignment_window_ns)
                selected = {host: min(values, key=lambda s: s.rtt_ns) for host, values in history.items()}
                if sum(e.estimated_error_ns for e in self._clock_estimates(selected).values()) <= 10_000_000:
                    return selected
            if self.monotonic() >= deadline:
                break
            self.sleep(self.poll_interval_s)
        raise RuntimeError('clock quality did not reach 10 ms relative uncertainty; capture rolled back')

    @staticmethod
    def _clock_estimates(samples: Mapping[str, ClockSample]) -> dict[str, ClockEstimate]:
        return {
            host: build_clock_timeline((sample,), window_ns=5_000_000_000)
            .remote_to_desktop(sample.remote_send_wall_ns)
            for host, sample in samples.items()
        }

    def _rollback(
        self, episode_id: str
    ) -> tuple[dict[str, RemoteStatus], dict[str, CommandResult], list[DiagnosticRecord]]:
        statuses = self._statuses()
        for _attempt in range(2):
            if any(status.active and status.episode_id not in (None, episode_id)
                   for status in statuses.values()):
                break
            if all(status.reachable and not status.last_error
                   for status in statuses.values()):
                break
            self.sleep(self.poll_interval_s)
            statuses = self._statuses()
        matching = {
            host: self.remotes[host]
            for host, status in statuses.items()
            if status.reachable and not status.last_error and status.episode_id == episode_id
        }
        results: dict[str, CommandResult] = {}
        diagnostics: list[DiagnosticRecord] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {host: executor.submit(remote.stop)
                       for host, remote in matching.items()}
            for host, future in futures.items():
                try:
                    results[host] = future.result()
                except Exception as exc:
                    results[host] = CommandResult(
                        f"{host} rollback stop", -1, stderr=f"stop raised: {exc}"
                    )
        for host, status in statuses.items():
            if status.active and status.episode_id != episode_id:
                diagnostics.append(DiagnosticRecord(
                    phase="rollback",
                    host=host,
                    message=(f"refused to stop mismatched episode "
                             f"{status.episode_id or '<unknown>'}"),
                    at_desktop_ns=self.time_ns(),
                ))
        for host, result in results.items():
            if not result.ok:
                diagnostics.append(DiagnosticRecord(
                    phase="rollback", host=host,
                    message=result.stderr or result.stdout or f"exit {result.returncode}",
                    at_desktop_ns=self.time_ns(),
                ))
        return statuses, results, diagnostics

    @staticmethod
    def _directories_from_results(results: Mapping[str, CommandResult]) -> dict[str, str]:
        return {
            host: directory
            for host, result in results.items()
            if (directory := JointController._directory_from_text(host, result.stdout))
        }

    @staticmethod
    def _directories_from_statuses(statuses: Mapping[str, RemoteStatus]) -> dict[str, str]:
        return {
            host: directory
            for host, status in statuses.items()
            if (directory := JointController._directory_from_text(host, status.message))
        }

    @staticmethod
    def _directory_from_text(host: str, text: str) -> str | None:
        if not text:
            return None
        if host == "p450":
            objects = json_objects(text)
            if len(objects) != 1:
                return text.strip() if text.strip().startswith("/") else None
            payload = objects[0]
            for container in (payload, payload.get("recorder", {}), payload.get("capture", {})):
                if isinstance(container, dict):
                    for key in ("session_dir", "episode_dir", "directory"):
                        value = container.get(key)
                        if isinstance(value, str) and value:
                            return value
            return None
        for line in text.splitlines():
            if line.startswith("episode="):
                value = line.split("=", 1)[1].strip()
                return value or None
        stripped = text.strip()
        return stripped if stripped.startswith("/") else None
