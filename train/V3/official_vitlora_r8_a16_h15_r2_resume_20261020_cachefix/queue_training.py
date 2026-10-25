#!/usr/bin/env python3
"""One-shot, non-polling GPU queue for the official V3 training entry point.

Run this script on 16.7. Each check observes GPUs once, rechecks before launch,
and either leaves the queue waiting or starts exactly one real training attempt.
An attempt claim is permanent: errors, crashes, or PID reuse never trigger a
retry. The supervisor and training process inherit the flock descriptor, so
exiting the short-lived checker does not release the active queue lock.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
from typing import Any

RUN_NAME = "vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_20261019_r2"
MODEL_ROOT = Path("/meta_eon_cfs/home/hph/heterovla-model/V3")
PYTHON = "/meta_eon_cfs/home/hph/heterovla/.venvs/openpi-pi05/bin/python"
MIN_TOTAL_MIB = 80000
MIN_FREE_RATIO = 0.95
COMPLETE_STATUSES = {"completed", "complete", "finished", "success", "succeeded"}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_preflight(code_dir: Path) -> dict[str, str]:
    """Bind a passed CPU preflight to the exact deployed manifest and sources."""
    manifest_path = code_dir / "manifest.json"
    preflight_path = code_dir / "preflight_report.json"
    manifest = read_json(manifest_path)
    preflight = read_json(preflight_path)
    if manifest is None or preflight is None:
        raise ValueError("manifest.json and preflight_report.json must both exist")
    if preflight.get("passed") is not True:
        raise ValueError("CPU preflight did not pass")
    manifest_hash = sha256(manifest_path)
    if preflight.get("manifest_sha256") != manifest_hash:
        raise ValueError("CPU preflight is not bound to this manifest SHA256")
    files = manifest.get("files")
    if not isinstance(files, dict) or "run_training.py" not in files or "queue_training.py" not in files:
        raise ValueError("manifest.files must hash run_training.py and queue_training.py")
    for relative, expected in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        path = code_dir / relative_path
        if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
            raise ValueError(f"Manifest SHA256 mismatch: {relative}")
    for name, expected in manifest.get("runtime_files", {}).items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Runtime data SHA256 mismatch: {name}")
    if "dataset_root" in manifest:
        alias = Path("/meta_eon_cfs/home/hph/heterovla-data/V3")
        if str(alias.resolve()) != manifest["dataset_root"]:
            raise ValueError("V3 data alias changed after CPU preflight")
    return {"manifest_sha256": manifest_hash, "preflight_sha256": sha256(preflight_path)}


def query_gpus(binary: str = "nvidia-smi") -> list[dict[str, Any]]:
    def query(fields: str, kind: str) -> list[list[str]]:
        result = subprocess.run(
            [binary, f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=20,
        )
        return [[part.strip() for part in row] for row in csv.reader(io.StringIO(result.stdout)) if row]

    devices = query("uuid,memory.total,memory.free", "gpu")
    processes = query("gpu_uuid,pid", "compute-apps")
    busy: set[str] = set()
    for row in processes:
        if len(row) == 1 and "No running processes found" in row[0]:
            continue
        if len(row) != 2 or not row[1].isdigit():
            raise ValueError(f"Unparseable GPU process row: {row!r}")
        busy.add(row[0])
    result = []
    for row in devices:
        if len(row) != 3:
            raise ValueError(f"Unparseable GPU memory row: {row!r}")
        uuid, total, free = row
        total_mib, free_mib = int(total), int(free)
        if not uuid.startswith("GPU-") or total_mib <= 0 or not 0 <= free_mib <= total_mib:
            raise ValueError(f"Invalid GPU inventory: {row!r}")
        result.append({"uuid": uuid, "total_mib": total_mib, "free_mib": free_mib,
                       "has_compute_process": uuid in busy})
    # Refuse ambiguous MIG or changing-inventory UUIDs rather than treating a
    # compute process that cannot be mapped to a physical GPU as an idle card.
    unknown_busy = busy - {gpu["uuid"] for gpu in result}
    if unknown_busy:
        raise ValueError(f"Compute process UUIDs are absent from GPU inventory: {sorted(unknown_busy)}")
    return result


def available(gpu: dict[str, Any]) -> bool:
    return (gpu["total_mib"] >= MIN_TOTAL_MIB
            and gpu["free_mib"] >= gpu["total_mib"] * MIN_FREE_RATIO
            and not gpu["has_compute_process"])


def process_identity(pid: int) -> dict[str, Any] | None:
    """Linux identity including birth ticks and the full argv; never trust PID alone."""
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text()
        tail = stat[stat.rfind(")") + 2:].split()
        status = (proc / "status").read_text().splitlines()
        uid = int(next(line for line in status if line.startswith("Uid:")).split()[1])
        command = [item.decode(errors="replace") for item in (proc / "cmdline").read_bytes().split(b"\0") if item]
        if tail[0] == "Z" or not command:
            return None
        return {"pid": pid, "uid": uid, "start_ticks": tail[19], "command": command}
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity_matches(saved: dict[str, Any] | None) -> bool:
    if not saved:
        return False
    actual = process_identity(saved["pid"])
    return bool(actual and all(actual.get(key) == saved.get(key)
                               for key in ("pid", "uid", "start_ticks", "command")))


def snapshot(args: argparse.Namespace, *, lock_busy: bool = False) -> dict[str, Any]:
    state = read_json(args.queue_dir / "state.json") or {"status": "waiting", "run_name": args.run_name}
    claim = read_json(args.queue_dir / "attempt.json")
    result = dict(state, checked_utc=utcnow(), lock_busy=lock_busy, attempt_exists=claim is not None)
    if state.get("training_identity"):
        result["training_identity_matches"] = identity_matches(state["training_identity"])
    if state.get("supervisor_identity"):
        result["supervisor_identity_matches"] = identity_matches(state["supervisor_identity"])
    runtime = read_json(args.run_dir / "runtime_status.json")
    if runtime is not None:
        result["runtime_status"] = runtime
    if claim is not None and "attempt_utc" not in result:
        result["attempt_utc"] = claim.get("attempt_utc")
    if claim is not None and result.get("status") in {"waiting", "claimed", "starting", "running"}:
        if not result.get("training_identity_matches") and not result.get("supervisor_identity_matches"):
            result["status"] = "attempt_needs_inspection"
            result["detail"] = "An attempt already exists but its processes are not verified; automatic retry is disabled"
    return result


def claim_attempt(path: Path, value: dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def child_arguments(args: argparse.Namespace, lock_fd: int, ack_fd: int) -> list[str]:
    return [args.python, str(args.code_dir / "queue_training.py"), "_supervise",
            "--code-dir", str(args.code_dir), "--queue-dir", str(args.queue_dir),
            "--run-dir", str(args.run_dir), "--run-name", args.run_name,
            "--python", args.python, "--nvidia-smi", args.nvidia_smi,
            "--lock-fd", str(lock_fd), "--ack-fd", str(ack_fd)]


def check_and_start(args: argparse.Namespace) -> dict[str, Any]:
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(args.queue_dir / "queue.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return snapshot(args, lock_busy=True)
        if (args.queue_dir / "attempt.json").exists():
            return snapshot(args)
        try:
            proof = verify_preflight(args.code_dir)
            observed = query_gpus(args.nvidia_smi)
            preferred = getattr(args, "preferred_gpu", None)
            candidates = sorted((gpu for gpu in observed if available(gpu)
                                 and (preferred is None or gpu["uuid"] == preferred)), key=lambda gpu: gpu["uuid"])
        except Exception as error:
            state = {"status": "blocked", "run_name": args.run_name, "checked_utc": utcnow(),
                     "detail": f"{type(error).__name__}: {error}", "attempt_exists": False}
            atomic_json(args.queue_dir / "state.json", state)
            return state
        if not candidates:
            state = {"status": "waiting", "run_name": args.run_name, "checked_utc": utcnow(),
                     "detail": "No eligible idle GPU with at least 95% free memory" + (f" (required GPU: {preferred})" if preferred else ""),
                     "gpus": observed, "attempt_exists": False}
            atomic_json(args.queue_dir / "state.json", state)
            return state
        selected = candidates[0]
        # A change between snapshots is a normal wait, not a consumed attempt.
        try:
            rechecked = next((gpu for gpu in query_gpus(args.nvidia_smi) if gpu["uuid"] == selected["uuid"]), None)
        except Exception as error:
            return {"status": "waiting", "run_name": args.run_name, "detail": f"GPU recheck unavailable: {error}", "attempt_exists": False}
        if rechecked is None or not available(rechecked):
            return {"status": "waiting", "run_name": args.run_name, "detail": "Selected GPU became busy before launch", "attempt_exists": False}
        claim = {"status": "claimed", "run_name": args.run_name, "attempt_utc": utcnow(),
                 "gpu": selected, "uid": os.getuid(), "proof": proof,
                 "no_automatic_retry": True}
        claim_attempt(args.queue_dir / "attempt.json", claim)
        atomic_json(args.queue_dir / "state.json", claim)
        read_fd, write_fd = os.pipe()
        try:
            with (args.queue_dir / "supervisor.log").open("ab", buffering=0) as log:
                subprocess.Popen(child_arguments(args, lock_fd, write_fd), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, cwd=args.code_dir,
                                 start_new_session=True, close_fds=True, pass_fds=(lock_fd, write_fd))
            os.close(write_fd)
            write_fd = -1
            if select.select([read_fd], [], [], 10)[0]:
                ack = os.read(read_fd, 65536)
                if ack:
                    return json.loads(ack)
            result = snapshot(args, lock_busy=True)
            result["detail"] = "Supervisor acknowledgement pending; the one-shot attempt is retained"
            return result
        except Exception as error:
            state = dict(claim, status="failed_to_launch_supervisor", detail=f"{type(error).__name__}: {error}")
            atomic_json(args.queue_dir / "state.json", state)
            return state
        finally:
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
    finally:
        # Do not LOCK_UN: the same open-file description is inherited by the
        # supervisor and child. Closing only this descriptor preserves their lock.
        os.close(lock_fd)


def supervise(args: argparse.Namespace) -> int:
    state = read_json(args.queue_dir / "attempt.json")
    assert state is not None
    ack_sent = False

    def save_and_ack(value: dict[str, Any]) -> None:
        nonlocal ack_sent
        atomic_json(args.queue_dir / "state.json", value)
        if not ack_sent:
            try:
                os.write(args.ack_fd, json.dumps(value).encode())
            except BrokenPipeError:
                pass
            os.close(args.ack_fd)
            ack_sent = True

    try:
        if os.fstat(args.lock_fd).st_ino != (args.queue_dir / "queue.lock").stat().st_ino:
            raise ValueError("Inherited queue lock does not match this queue")
        fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor_identity = process_identity(os.getpid())
        if supervisor_identity is None or supervisor_identity["uid"] != state["uid"]:
            raise ValueError("Supervisor UID/start_ticks/command identity could not be verified")
        if verify_preflight(args.code_dir) != state["proof"]:
            raise ValueError("Deployment changed after the launch claim")
        gpu = next((gpu for gpu in query_gpus(args.nvidia_smi) if gpu["uuid"] == state["gpu"]["uuid"]), None)
        if gpu is None or not available(gpu):
            raise ValueError("Selected GPU is no longer idle at the supervisor launch gate")
        command = [args.python, str(args.code_dir / "run_training.py"),
                   "--gpu", gpu["uuid"], "--run-name", args.run_name]
        log_path = args.queue_dir / "training.log"
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, cwd=args.code_dir,
                                       close_fds=True, pass_fds=(args.lock_fd,))
        training_identity = process_identity(process.pid)
        if training_identity is None:
            # wait() still collects the exit; do not ever launch a replacement.
            returncode = process.wait()
            raise RuntimeError(f"Training exited before process identity verification (returncode={returncode})")
        if training_identity["uid"] != state["uid"] or training_identity["command"] != command:
            raise ValueError("Training process UID/command differs from the launch contract")
        state = dict(state, status="running", started_utc=utcnow(), gpu=gpu,
                     supervisor_identity=supervisor_identity, training_identity=training_identity,
                     command=command, log=str(log_path), runtime_status_path=str(args.run_dir / "runtime_status.json"))
        save_and_ack(state)
        returncode = process.wait()
        runtime = read_json(args.run_dir / "runtime_status.json") or {}
        runtime_phase = runtime.get("status", runtime.get("phase"))
        if returncode != 0:
            outcome = "failed"
        elif runtime_phase in COMPLETE_STATUSES:
            outcome = "completed"
        else:
            outcome = "finished_unverified"
        state = dict(state, status=outcome, returncode=returncode,
                     finished_utc=utcnow(), runtime_status=runtime)
        save_and_ack(state)
        return returncode
    except Exception as error:
        state = dict(state, status="failed", finished_utc=utcnow(),
                     detail=f"{type(error).__name__}: {error}")
        save_and_ack(state)
        return 1
    finally:
        os.close(args.lock_fd)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("check-and-start", "status", "_supervise"))
    result.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent)
    result.add_argument("--run-name", default=RUN_NAME)
    result.add_argument("--queue-dir", type=Path)
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--python", default=PYTHON)
    result.add_argument("--nvidia-smi", default="nvidia-smi")
    result.add_argument("--preferred-gpu", help="Wait for this UUID only; never select another GPU")
    result.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    result.add_argument("--ack-fd", type=int, help=argparse.SUPPRESS)
    return result


def main() -> int:
    args = parser().parse_args()
    args.code_dir = args.code_dir.resolve()
    args.queue_dir = (args.queue_dir or MODEL_ROOT / "queues" / args.run_name).resolve()
    args.run_dir = (args.run_dir or MODEL_ROOT / "runs" / args.run_name).resolve()
    if args.action == "_supervise":
        if args.lock_fd is None or args.ack_fd is None:
            raise SystemExit("_supervise requires inherited file descriptors")
        return supervise(args)
    result = check_and_start(args) if args.action == "check-and-start" else snapshot(args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
