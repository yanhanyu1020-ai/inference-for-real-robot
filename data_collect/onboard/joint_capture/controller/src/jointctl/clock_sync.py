"""Persistent, four-timestamp SSH clock probes for capture hosts."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Callable, Mapping

from .models import ClockSample, EpisodeManifest


REMOTE_RESPONDER = """import sys,time
for _ in sys.stdin:
    recv=time.time_ns(); mono=time.monotonic_ns(); send=time.time_ns()
    print(recv, send, mono, flush=True)
"""

TERMINAL_STATES = {"complete", "start_failed", "stop_failed", "partial"}
MAX_RESPONSE_LINE = 4096
RESPONSE_QUEUE_SIZE = 32
DEFAULT_CLOCK_FRESHNESS_NS = 10_000_000_000


def read_clock_records(episode_dir: Path, manifest: EpisodeManifest):
    """Read append-only evidence, keeping outages and malformed-row diagnostics."""
    samples = {host: [] for host in ("p450", "unitree")}
    errors = {host: [] for host in samples}
    for host in samples:
        path = episode_dir / f"clock_{host}.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        for number, line in enumerate(lines, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("clock record must be an object")
                if "remote_send_wall_ns" not in row:
                    errors[host].append(row)
                    continue
                sample = ClockSample.from_dict(row)
                if sample.host != host:
                    raise ValueError(f"clock sample host mismatch: {sample.host}")
                samples[host].append(sample)
            except (ValueError, TypeError, KeyError) as exc:
                errors[host].append({"kind": "invalid_record", "host": host,
                                     "line": number, "message": str(exc)})
        initial = manifest.clock_samples.get(host)
        if initial is not None and initial not in samples[host]:
            samples[host].append(initial)
        samples[host].sort(key=lambda sample: sample.remote_send_wall_ns)
    return samples, errors


class ClockProbeError(RuntimeError):
    """Clock-probe failure with transport versus remote-protocol provenance."""

    def __init__(self, message: str, *, failure_kind: str = "remote_command") -> None:
        if failure_kind not in {"connectivity", "remote_command"}:
            raise ValueError("failure_kind must be connectivity or remote_command")
        super().__init__(message)
        self.failure_kind = failure_kind


class ClockProbe:
    """One persistent SSH process that emits a four-timestamp response per line."""

    def __init__(self, host: str, ssh_executable: str = "ssh", timeout_s: float = 2.0,
                 *, destination: str | None = None) -> None:
        self.host = host
        self.destination = destination or host
        self.ssh_executable = ssh_executable
        self.timeout_s = timeout_s
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | BaseException] | None = None
        self._sequence = 0
        self.process_start_count = 0
        self._errors: list[dict[str, object]] = []

    @staticmethod
    def remote_command() -> str:
        encoded = base64.b64encode(REMOTE_RESPONDER.encode("utf-8")).decode("ascii")
        return "python3 -u -c \"exec(__import__('base64').b64decode('" + encoded + "'))\""

    def _start_process(self):
        return subprocess.Popen(
            [self.ssh_executable, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
             self.destination, self.remote_command()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )

    @staticmethod
    def _reader(stream, responses: queue.Queue[str | BaseException], *, max_line: int = MAX_RESPONSE_LINE) -> None:
        def invalidate(message: str) -> None:
            try:
                while True:
                    responses.get_nowait()
            except queue.Empty:
                pass
            responses.put_nowait(ClockProbeError(message))
        try:
            for line in iter(lambda: stream.readline(max_line + 1), ""):
                if len(line) > max_line:
                    invalidate("clock responder response exceeds limit")
                    return
                try:
                    responses.put_nowait(line)
                except queue.Full:
                    invalidate("clock responder response queue overflow")
                    return
            try:
                responses.put_nowait(EOFError("clock responder closed stdout"))
            except queue.Full:
                invalidate("clock responder response queue overflow")
        except BaseException as exc:  # pipe I/O errors need reconnect too
            invalidate(str(exc))

    @staticmethod
    def _drain(stream, responses: queue.Queue[str | BaseException]) -> None:
        try:
            for _line in iter(lambda: stream.read(MAX_RESPONSE_LINE), ""):
                continue
        except BaseException:
            pass

    def open(self) -> "ClockProbe":
        if self._process is not None:
            return self
        process = self._start_process()
        if process.stdin is None or process.stdout is None:
            raise ClockProbeError("clock responder did not expose standard streams")
        self._process = process
        self._responses = queue.Queue(maxsize=RESPONSE_QUEUE_SIZE)
        threading.Thread(target=self._reader, args=(process.stdout, self._responses), daemon=True).start()
        if process.stderr is not None and process.stderr is not process.stdout:
            threading.Thread(target=self._drain, args=(process.stderr, self._responses), daemon=True).start()
        self.process_start_count += 1
        return self

    def _record_error(self, message: str) -> None:
        self._errors.append({"kind": "probe_error", "host": self.host,
                             "message": message, "at_wall_ns": time.time_ns()})

    def drain_errors(self) -> list[dict[str, object]]:
        errors, self._errors = self._errors, []
        return errors

    def _closed_process_failure_kind(self) -> str:
        """Classify EOF only after giving SSH a chance to publish its exit code."""
        process = self._process
        if process is None:
            return "remote_command"
        code = process.poll()
        if code is None:
            try:
                code = process.wait(timeout=min(max(self.timeout_s, 0.01), 0.2))
            except subprocess.TimeoutExpired:
                code = process.poll()
        # An exit status that is still unavailable after the bounded wait is
        # a lost/timed-out responder, consistent with the sample timeout path.
        return "connectivity" if code in {None, -1, 255} else "remote_command"

    def sample(self) -> ClockSample:
        self.open()
        assert self._process is not None and self._responses is not None
        try:
            if self._process.poll() is not None:
                code = self._process.returncode
                kind = "connectivity" if code in {-1, 255} else "remote_command"
                raise ClockProbeError("clock responder exited", failure_kind=kind)
            local_send_wall = time.time_ns()
            # perf_counter uses QueryPerformanceCounter on Windows Python 3.12;
            # monotonic_ns can use the much coarser GetTickCount64 there.
            local_send_mono = time.perf_counter_ns()
            assert self._process.stdin is not None
            self._process.stdin.write("sample\n")
            self._process.stdin.flush()
            try:
                response = self._responses.get(timeout=self.timeout_s)
            except queue.Empty as exc:
                raise ClockProbeError(
                    f"clock responder timed out after {self.timeout_s}s", failure_kind="connectivity"
                ) from exc
            local_receive_wall = time.time_ns()
            local_receive_mono = time.perf_counter_ns()
            if isinstance(response, EOFError):
                raise ClockProbeError(
                    str(response), failure_kind=self._closed_process_failure_kind()
                )
            if isinstance(response, BaseException):
                raise ClockProbeError(str(response))
            fields = response.strip().split()
            if len(fields) != 3:
                raise ClockProbeError("malformed clock responder response")
            try:
                remote_receive, remote_send, remote_mono = (int(value) for value in fields)
            except ValueError as exc:
                raise ClockProbeError("malformed clock responder response") from exc
            sample = ClockSample.from_exchange(
                host=self.host, sequence=self._sequence,
                local_send_wall_ns=local_send_wall, local_send_mono_ns=local_send_mono,
                remote_receive_wall_ns=remote_receive, remote_send_wall_ns=remote_send,
                remote_monotonic_ns=remote_mono, local_receive_wall_ns=local_receive_wall,
                local_receive_mono_ns=local_receive_mono,
            )
            self._sequence += 1
            return sample
        except (OSError, ValueError, ClockProbeError) as exc:
            self._record_error(str(exc))
            self.close()
            if isinstance(exc, ClockProbeError):
                raise
            raise ClockProbeError(str(exc), failure_kind="connectivity") from exc

    def close(self) -> None:
        process, self._process = self._process, None
        self._responses = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)

    def __enter__(self) -> "ClockProbe":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()


def _terminal_manifest_present(manifest_dir: Path) -> bool:
    for path in manifest_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and str(payload.get("state", "")).lower() in TERMINAL_STATES:
            return True
    return False


def _append_line(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, separators=(",", ":")) + "\n")
        output.flush()


def _run_monitor(manifest_dir: Path, hosts: tuple[str, ...], interval_s: float,
                 stop_file: Path, probe_factory: Callable[[str], ClockProbe],
                 sample_count: int | None = None) -> int:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    # Avoid even opening SSH processes after an episode has already finished.
    if stop_file.exists() or _terminal_manifest_present(manifest_dir):
        return 0
    if not hosts:
        return 0
    probes = {host: probe_factory(host) for host in hosts}
    stop_event = threading.Event()
    completed: set[str] = set()
    completed_lock = threading.Lock()

    def worker(host: str) -> None:
        probe = probes[host]
        next_at = time.monotonic()
        count = 0
        while not stop_event.is_set() and not stop_file.exists() and not _terminal_manifest_present(manifest_dir):
            wait_s = next_at - time.monotonic()
            if wait_s > 0 and stop_event.wait(wait_s):
                return
            try:
                sample = probe.sample()
                _append_line(manifest_dir / f"clock_{host}.jsonl", sample.to_dict())
            except Exception as exc:
                errors = probe.drain_errors() or [{"kind": "probe_error", "host": host,
                                                   "message": str(exc), "at_wall_ns": time.time_ns()}]
                for error in errors:
                    _append_line(manifest_dir / f"clock_{host}.jsonl", error)
            count += 1
            if sample_count is not None and count >= sample_count:
                with completed_lock:
                    completed.add(host)
                    if len(completed) == len(probes):
                        stop_event.set()
                return
            next_at = max(next_at + max(0.0, interval_s), time.monotonic())

    threads = [threading.Thread(target=worker, args=(host,), daemon=True) for host in hosts]
    try:
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            if stop_file.exists() or _terminal_manifest_present(manifest_dir):
                stop_event.set()
            time.sleep(0.01)
        for thread in threads:
            thread.join()
    finally:
        for probe in probes.values():
            probe.close()
    return 0


def run_monitor(manifest_dir: Path, hosts: tuple[str, ...], interval_s: float,
                stop_file: Path, *, destinations: Mapping[str, str] | None = None,
                timeout_s: float = 2.0) -> int:
    return _run_monitor(manifest_dir, hosts, interval_s, stop_file,
                        lambda host: ClockProbe(host, timeout_s=timeout_s,
                                                destination=(destinations or {}).get(host, host)))


def run_monitor_for_test(manifest_dir: Path, stop_file: Path, sample_count: int,
                         probes: Callable[[str], ClockProbe]) -> int:
    return _run_monitor(manifest_dir, ("p450", "unitree"), 0.0, stop_file, probes, sample_count)


def launch_detached_monitor(manifest_dir: Path, config_path: Path) -> int:
    """Launch this module detached and save its PID in an existing JSON manifest."""
    manifest_dir = Path(manifest_dir)
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid monitor config") from exc
    hosts = config.get("hosts", ("p450", "unitree"))
    if not isinstance(hosts, (list, tuple)) or not hosts or not all(isinstance(h, str) and h for h in hosts):
        raise ValueError("config hosts must be a non-empty sequence")
    manifest_name = config.get("manifest_file") or config.get("manifest_path")
    if manifest_name:
        candidate = Path(manifest_name)
        manifest = candidate if candidate.is_absolute() else manifest_dir / candidate
        manifests = [manifest]
    else:
        manifests = [p for p in manifest_dir.glob("*.json") if p.resolve() != Path(config_path).resolve()]
    if len(manifests) != 1 or not manifests[0].is_file():
        raise ValueError("exactly one owning episode manifest is required")
    manifest = manifests[0]
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid owning episode manifest") from exc
    if not isinstance(payload, dict):
        raise ValueError("owning episode manifest must be an object")
    command = [sys.executable, "-m", "jointctl.clock_sync", "--manifest-dir", str(manifest_dir),
               "--config", str(config_path)]
    child_environment = os.environ.copy()
    package_src = str(Path(__file__).resolve().parents[1])
    inherited_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        package_src + (os.pathsep + inherited_pythonpath if inherited_pythonpath else "")
    )
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    stdout = (manifest_dir / "clock_monitor.stdout.log").open("a", encoding="utf-8")
    stderr = (manifest_dir / "clock_monitor.stderr.log").open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, creationflags=flags,
            env=child_environment,
        )
    finally:
        # The child inherits the descriptors; the parent must not retain them.
        stdout.close()
        stderr.close()
    try:
        payload["clock_monitor_pid"] = process.pid
        temporary = manifest.with_suffix(manifest.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest)
    except Exception:
        try:
            process.terminate()
        finally:
            raise
    return process.pid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    hosts = tuple(config.get("hosts", ("p450", "unitree")))
    return run_monitor(args.manifest_dir, hosts, float(config.get("clock_interval_s", 0.2)),
                       args.manifest_dir / config.get("clock_stop_file", "clock.stop"),
                       destinations=config.get("destinations"),
                       timeout_s=float(config.get("ssh_timeout_s", 2.0)))


if __name__ == "__main__":
    raise SystemExit(main())
