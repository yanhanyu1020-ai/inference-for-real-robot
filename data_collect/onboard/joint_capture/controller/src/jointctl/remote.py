"""SSH command adapter and status parsers for the two capture computers."""

from __future__ import annotations

import json
from dataclasses import replace
import os
import shlex
import subprocess
import time
from pathlib import PurePosixPath
from typing import Sequence

from .models import CommandResult, RemoteStatus

P450_STATUS = "/home/amov/bin/p450_capture status"
P450_START = "/home/amov/bin/p450_capture start {episode_id}"
P450_STOP = "/home/amov/bin/p450_capture stop-fast"
UNITREE_STATUS = "~/heterovla-collection/onboard/collection_ctl.sh status"
UNITREE_START = "~/heterovla-collection/onboard/collection_ctl.sh start {episode_id} {instruction_q} {task_q}"
UNITREE_STOP = "~/heterovla-collection/onboard/collection_ctl.sh stop"


def run_process(argv: Sequence[str], timeout_s: float) -> CommandResult:
    """Run a local process, retaining stdout, stderr, and return code separately."""
    command = shlex.join([str(part) for part in argv])
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout_s, check=False,
            encoding='utf-8', errors='replace',
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        return CommandResult(command, completed.returncode, completed.stdout, completed.stderr,
                             time.perf_counter_ns() - started)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return CommandResult(command, -1, stdout, stderr or f"timed out after {timeout_s}s",
                             time.perf_counter_ns() - started)


def run_ssh(host: str, command: str, timeout_s: float) -> CommandResult:
    """Execute command on *host* through the system ssh client."""
    connect_timeout_s = min(20, max(5, int(timeout_s - 1)))
    return run_process(("ssh", "-n", "-T", "-o", "BatchMode=yes", "-o",
                        f"ConnectTimeout={connect_timeout_s}", host, command), timeout_s)


def json_objects(text: str) -> list[dict]:
    """Read complete JSON objects surrounded by command diagnostics."""
    decoder = json.JSONDecoder()
    objects = []
    position = 0
    while (position := text.find('{', position)) >= 0:
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            position += 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        position = end
    return objects


def parse_p450_status(text: str) -> RemoteStatus:
    try:
        candidates = [obj for obj in json_objects(text) if 'recorder' in obj and 'capture' in obj]
        if len(candidates) != 1:
            raise ValueError('expected exactly one P450 status')
        data = candidates[0]
        capture = data["capture"] or {}
        recorder = data["recorder"]
        active = bool(recorder["active"])
        episode = capture.get("session_name") or PurePosixPath(recorder.get("session_dir", "")).name or None
        progress = int(recorder.get("session_bytes", 0))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid P450 status JSON") from exc
    return RemoteStatus("p450", True, "recording" if active else "idle", text,
                        None, active, episode, "session_bytes", progress)


def parse_unitree_status(text: str) -> RemoteStatus:
    if text.strip() == 'stopped':
        return RemoteStatus('unitree', True, 'idle', text, active=False)
    episode_path = None
    for line in text.splitlines():
        if line.startswith("episode="):
            episode_path = line.split("=", 1)[1].strip()
    candidates = [obj for obj in json_objects(text) if 'running' in obj]
    if len(candidates) != 1 or not isinstance(candidates[0]['running'], bool):
        raise ValueError("invalid Unitree status JSON")
    payload = candidates[0]
    cleanup_pending = payload.get("cleanup_pending") is True
    active = bool(payload["running"]) or cleanup_pending
    episode = PurePosixPath(episode_path).name if episode_path else None
    if payload.get('format_version') == 2:
        streams = payload.get('streams', {})
        if not isinstance(streams, dict) or set(streams) != {'front', 'wrist'} or any(
                not isinstance(streams[n], dict) or type(streams[n].get('written')) is not int
                or streams[n]['written'] < 0 for n in streams):
            raise ValueError('invalid v2 camera counters')
        fault = payload.get('fault') or []
        if not isinstance(fault, list) or any(not isinstance(item, str) for item in fault):
            raise ValueError('invalid v2 fault report')
        gaps = payload.get('state_gap_ticks', 0)
        missing = payload.get('missing_states', [])
        if type(gaps) is not int or gaps < 0 or not isinstance(missing, list) or any(not isinstance(x, str) for x in missing):
            raise ValueError('invalid v2 state health')
        return RemoteStatus('unitree', True, 'fault' if fault else 'cleanup_pending' if cleanup_pending else 'recording' if active else 'idle',
            text, None, active, episode, 'camera_records', sum(s['written'] for s in streams.values()),
            format_version=2, streams=streams, ready=payload.get('ready') is True,
            quality_ok=payload.get('quality_ok') is True, fault=fault, raw_closed=payload.get('durable_complete') is True,
            state_gap_ticks=gaps, missing_states=missing)
    progress = int(payload.get("frames_saved", 0))
    return RemoteStatus("unitree", True, "cleanup_pending" if cleanup_pending else "recording" if active else "idle", text,
                        None, active, episode, "frames_saved", progress)


class RemoteClient:
    """Small host-specific adapter used by orchestration code."""

    def __init__(self, host: str, kind: str | None = None, timeout_s: float = 10.0, platform: str | None = None,
                 *, start_timeout_s: float | None = None, stop_timeout_s: float | None = None,
                 prepare_before_start: bool = False):
        self.host = host
        self.kind = kind or platform or host
        self.timeout_s = timeout_s
        self.prepare_before_start = prepare_before_start
        self.start_timeout_s = timeout_s if start_timeout_s is None else start_timeout_s
        self.stop_timeout_s = timeout_s if stop_timeout_s is None else stop_timeout_s
        if self.kind not in {"p450", "unitree"}:
            raise ValueError("kind must be 'p450' or 'unitree'")

    def status(self) -> RemoteStatus:
        result = run_ssh(self.host, P450_STATUS if self.kind == "p450" else UNITREE_STATUS, self.timeout_s)
        if not result.ok:
            # ``-1`` is our synthetic timeout result; OpenSSH reserves 255 for
            # transport failure. Other codes mean ssh connected and the remote
            # status command itself failed, so the host remains reachable.
            reachable = result.returncode not in {-1, 255}
            detail = result.stderr or result.stdout or "status command failed"
            error = f"remote status command failed (exit code {result.returncode}): {detail}"
            return RemoteStatus(self.host, reachable, "unknown", result.stdout, error)
        try:
            status = (parse_p450_status(result.stdout) if self.kind == "p450"
                      else parse_unitree_status(result.stdout))
        except ValueError as exc:
            return RemoteStatus(
                self.host, True, "unknown", result.stdout,
                f"remote status protocol error: {exc}",
            )
        return replace(status, host=self.host)

    def start(self, episode_id: str, instruction: str = "", task: str = "") -> CommandResult:
        if self.prepare_before_start:
            prepared = self.prepare()
            if not prepared.ok:
                return CommandResult(prepared.command, prepared.returncode, prepared.stdout,
                                     'preparation failed: ' + prepared.stderr, prepared.duration_ns)
        if self.kind == "p450":
            command = P450_START.format(episode_id=shlex.quote(episode_id))
        else:
            command = UNITREE_START.format(episode_id=shlex.quote(episode_id),
                                           instruction_q=shlex.quote(instruction), task_q=shlex.quote(task))
        return run_ssh(self.host, command, self.start_timeout_s)

    def prepare(self) -> CommandResult:
        command = ('/home/amov/bin/p450_capture prepare' if self.kind == 'p450'
                   else '~/heterovla-collection/onboard/collection_ctl.sh prepare')
        return run_ssh(self.host, command, self.start_timeout_s)

    def stop(self) -> CommandResult:
        return run_ssh(self.host, P450_STOP if self.kind == "p450" else UNITREE_STOP, self.stop_timeout_s)

    def finalize_raw(self, directory: str) -> CommandResult:
        if self.kind == "p450":
            command = "/home/amov/bin/p450_capture finalize-raw " + shlex.quote(directory)
        else:
            command = "~/heterovla-collection/onboard/collection_ctl.sh finalize " + shlex.quote(directory)
        return run_ssh(self.host, command, max(self.stop_timeout_s, 3600.0))
