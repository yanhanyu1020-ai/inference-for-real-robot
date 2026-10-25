#!/usr/bin/env python3
"""CPU controller for one sequential GPU-5 evaluation of three saved EMA models.

This controller never imports JAX, edits a checkpoint, stops a process, or
chooses another GPU. A permanent attempt claim precedes each evaluator launch;
an interrupted/failed attempt is reported for human inspection, never retried.
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
import subprocess
import tempfile
import time
from typing import Any


GPU_UUID = "GPU-e66640c0-8e1e-3f09-8f3f-2d10f877ec04"
GPU_INDEX = 5
STEPS = [5000, 10000, 15000]
WINDOWS = 2426
MIN_FREE_MIB = 28672
POLL_SECONDS = 30
MAX_WAIT_SECONDS = 24 * 60 * 60


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def claim(path: Path, value: dict[str, Any]) -> None:
    """O_EXCL makes even a crash between claim and launch a non-retryable attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_identity(pid: int) -> dict[str, Any] | None:
    """Identify a Linux process by PID, UID, start ticks, boot ID and full argv."""
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        uid_line = next(line for line in (proc / "status").read_text().splitlines()
                        if line.startswith("Uid:"))
        command = [part.decode(errors="replace")
                   for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        return {"pid": pid, "uid": int(uid_line.split()[1]),
                "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "command": command}
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        return None


def query_gpu(binary: str) -> dict[str, Any]:
    result = subprocess.run(
        [binary, "-i", GPU_UUID,
         "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError("Unexpected nvidia-smi GPU inventory")
    index, uuid, total, free, utilization = [part.strip() for part in rows[0]]
    if int(index) != GPU_INDEX or uuid != GPU_UUID:
        raise ValueError(f"GPU UUID no longer maps to requested GPU {GPU_INDEX}")
    total_mib, free_mib = int(total), int(free)
    if not (total_mib >= 80000 and 0 <= free_mib <= total_mib):
        raise ValueError("Unexpected GPU capacity or memory values")
    return {"index": int(index), "uuid": uuid, "total_mib": total_mib,
            "free_mib": free_mib, "utilization_percent": int(utilization),
            "checked_utc": utcnow()}


def child_env(*, gpu: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": GPU_UUID if gpu else "",
        "JAX_PLATFORMS": "cuda" if gpu else "cpu",
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
    })
    # The waiting controller/build step do not create any CUDA context.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if gpu else "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"
    return env


def summary_is_complete(summary: dict[str, Any] | None, *, step: int | None) -> bool:
    if not summary or summary.get("complete") is not True:
        return False
    if summary.get("windows") != WINDOWS:
        return False
    if step is not None and summary.get("checkpoint_step", summary.get("step")) != step:
        return False
    return True


class Controller:
    def __init__(self, plan_path: Path, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.plan_path = plan_path
        self.output_root = Path(plan["output_root"])
        self.cache_dir = Path(plan["cache_dir"])
        self.code_dir = Path(plan["code_dir"])
        self.eval_dir = Path(plan["eval_dir"])
        self.checkpoint_root = Path(plan["checkpoint_root"])
        self.python = str(plan["python"])
        self.cache_summary = Path(plan.get("cache_summary_path", str(self.cache_dir / "summary.json")))
        self.state_path = self.output_root / "state.json"
        self.state = read_json(self.state_path) or {}
        self.identity = process_identity(os.getpid())
        if self.identity is None:
            raise RuntimeError("Cannot establish Linux controller process identity")
        # Preserve the original total waiting deadline if a CPU-only wait was
        # interrupted and an operator re-invokes this controller.
        now = time.time()
        duration = int(plan.get("max_wait_seconds", MAX_WAIT_SECONDS))
        if not 1 <= duration <= MAX_WAIT_SECONDS:
            raise ValueError("max_wait_seconds must be between 1 and 86400")
        self.deadline = float(self.state.get("deadline_unix", now + duration))
        self.state.update({
            "schema_version": 1, "plan_path": str(plan_path),
            "plan_sha256": sha256(plan_path),
            "controller_identity": self.identity,
            "started_utc": self.state.get("started_utc", utcnow()),
            "deadline_unix": self.deadline,
            "gpu_uuid": GPU_UUID, "gpu_index": GPU_INDEX,
            "checkpoint_steps": STEPS, "expected_windows": WINDOWS,
            "minimum_free_mib": MIN_FREE_MIB,
            "automatic_retry": False, "changes_training": False,
        })
        atomic_json(self.output_root / "controller_identity.json", self.identity)

    def update(self, status: str, **details: Any) -> None:
        self.state.update(details)
        self.state.update(status=status, updated_utc=utcnow())
        atomic_json(self.state_path, self.state)

    def wait(self, reason: str, **details: Any) -> bool:
        if time.time() >= self.deadline:
            self.update("blocked", reason="waiting_deadline_exceeded",
                        waiting_for=reason, **details)
            return False
        self.update("waiting", reason=reason, **details)
        time.sleep(min(POLL_SECONDS, max(0.0, self.deadline - time.time())))
        return True

    def run_child(self, command: list[str], directory: Path, *, gpu: bool,
                  phase: str) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "process.log").open("a", buffering=1) as logfile:
            process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                       env=child_env(gpu=gpu), close_fds=True,
                                       cwd=self.eval_dir)
            identity = process_identity(process.pid)
            # A very short-lived failure still retains its PID/command and
            # return code, without claiming a PID identity match.
            launch = {"started_utc": utcnow(), "command": command,
                      "process_identity": identity, "pid": process.pid,
                      "gpu_uuid": GPU_UUID if gpu else None,
                      "environment": {key: child_env(gpu=gpu)[key] for key in (
                          "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS",
                          "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                          "OMP_NUM_THREADS", "WANDB_MODE", "HF_HUB_OFFLINE")}}
            atomic_json(directory / "launch.json", launch)
            self.update(phase, active_child=launch)
            while process.poll() is None:
                self.state["child_identity_current"] = process_identity(process.pid)
                self.update(phase)
                try:
                    process.wait(timeout=POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
            completion = dict(launch, finished_utc=utcnow(), returncode=process.returncode)
            atomic_json(directory / "exit.json", completion)
            self.state.pop("active_child", None)
            self.state.pop("child_identity_current", None)
            return completion

    def build_cache(self) -> bool:
        if summary_is_complete(read_json(self.cache_summary), step=None):
            self.update("cache_ready", cache_summary_path=str(self.cache_summary))
            return True
        # A CPU preparation job may already have been launched by the operator.
        # Wait for its existing lock; never launch a competing builder.
        cache_lock = self.cache_dir / 'build.lock'
        if cache_lock.exists():
            while True:
                if summary_is_complete(read_json(self.cache_summary), step=None):
                    self.update('cache_ready', cache_summary_path=str(self.cache_summary))
                    return True
                error = read_json(self.cache_dir / 'build_error.json')
                if error:
                    self.update('failed', reason='existing_cpu_cache_builder_failed', cache_error=error)
                    return False
                with cache_lock.open('a') as handle:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        active = True
                    else:
                        active = False
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                if not active:
                    if summary_is_complete(read_json(self.cache_summary), step=None):
                        return True
                    # A manually interrupted CPU preparation may leave a valid
                    # partial cache. The builder verifies its contract and can
                    # continue it when the operator explicitly executes us.
                    break
                if not self.wait('existing_cpu_cache_builder_in_progress'):
                    return False
        directory = self.output_root / "cache_build"
        attempt = directory / "attempt.json"
        if attempt.exists():
            self.update("blocked", reason="cache_attempt_exists_without_complete_cache",
                        attempt_path=str(attempt))
            return False
        command = [self.python, "-B", str(self.eval_dir / "build_cache.py"),
                   "--code-dir", str(self.code_dir), "--output-dir", str(self.cache_dir)]
        claim(attempt, {"claimed_utc": utcnow(), "controller_identity": self.identity,
                        "command": command, "automatic_retry": False})
        completion = self.run_child(command, directory, gpu=False, phase="building_cache_cpu")
        if completion["returncode"] != 0:
            self.update("failed", reason="cache_builder_exited_nonzero",
                        returncode=completion["returncode"], log=str(directory / "process.log"))
            return False
        if not summary_is_complete(read_json(self.cache_summary), step=None):
            self.update("failed", reason="cache_summary_missing_or_incomplete",
                        cache_summary_path=str(self.cache_summary))
            return False
        self.update("cache_ready", cache_summary_path=str(self.cache_summary))
        return True

    def source_status(self) -> dict[str, Any] | None:
        path = self.plan.get("training_status_path")
        return read_json(Path(path)) if path else None

    def checkpoint_ready(self, step: int) -> bool:
        checkpoint = self.checkpoint_root / str(step)
        return ((checkpoint / "_CHECKPOINT_METADATA").is_file()
                and (checkpoint / "params" / "_METADATA").is_file())

    def evaluate(self, step: int) -> dict[str, Any] | None:
        directory = self.output_root / "results" / str(step)
        attempt = directory / "attempt.json"
        existing = read_json(directory / "summary.json")
        previous_exit = read_json(directory / "exit.json")
        if attempt.exists():
            if previous_exit and previous_exit.get("returncode") == 0 and summary_is_complete(existing, step=step):
                return existing
            self.update("blocked", reason="permanent_evaluation_attempt_already_exists",
                        step=step, attempt_path=str(attempt), previous_exit=previous_exit)
            return None
        while not self.checkpoint_ready(step):
            if not self.wait("source_not_yet_available", step=step,
                             checkpoint=str(self.checkpoint_root / str(step)),
                             source_training_status=self.source_status()):
                return None
        preflight_directory = self.output_root / 'cpu_preflight' / str(step)
        preflight_command = [self.python, '-B', str(self.eval_dir / 'evaluate_checkpoint.py'),
                             '--code-dir', str(self.code_dir),
                             '--checkpoint', str(self.checkpoint_root / str(step)),
                             '--cache-dir', str(self.cache_dir), '--output-dir', str(preflight_directory),
                             '--cpu-preflight']
        completion = self.run_child(preflight_command, preflight_directory, gpu=False,
                                    phase='checking_checkpoint_on_cpu')
        preflight = read_json(preflight_directory / 'contract_proof.json')
        if (completion['returncode'] != 0 or not preflight or preflight.get('passed') is not True
                or preflight.get('cpu_only') is not True or preflight.get('checkpoint_step') != step):
            self.update('failed', reason='checkpoint_cpu_contract_failed', step=step,
                        preflight_log=str(preflight_directory / 'process.log'))
            return None
        while True:
            try:
                gpu = query_gpu(str(self.plan.get("nvidia_smi", "nvidia-smi")))
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                if not self.wait("gpu_query_unavailable", step=step, gpu_query_error=str(exc)):
                    return None
                continue
            if gpu["free_mib"] < MIN_FREE_MIB:
                if not self.wait("gpu_memory_below_threshold", step=step, gpu=gpu):
                    return None
                continue
            # Recheck both filesystem commitment and GPU immediately before the
            # permanent claim; this minimizes the unavoidable NVML/start race.
            if not self.checkpoint_ready(step):
                self.update("blocked", reason="checkpoint_disappeared_before_launch", step=step)
                return None
            try:
                gpu = query_gpu(str(self.plan.get("nvidia_smi", "nvidia-smi")))
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                if not self.wait("gpu_query_unavailable", step=step, gpu_query_error=str(exc)):
                    return None
                continue
            if gpu["free_mib"] >= MIN_FREE_MIB:
                break
            if not self.wait("gpu_memory_below_threshold", step=step, gpu=gpu):
                return None
        command = [self.python, "-B", str(self.eval_dir / "evaluate_checkpoint.py"),
                   "--code-dir", str(self.code_dir),
                   "--checkpoint", str(self.checkpoint_root / str(step)),
                   "--cache-dir", str(self.cache_dir), "--output-dir", str(directory),
                   "--gpu", GPU_UUID, "--batch-size", "16", "--seed", "42"]
        claim(attempt, {"claimed_utc": utcnow(), "controller_identity": self.identity,
                        "step": step, "command": command, "gpu": gpu,
                        "automatic_retry": False})
        self.update("launching_evaluator", step=step, gpu=gpu)
        completion = self.run_child(command, directory, gpu=True, phase="evaluating")
        if completion["returncode"] != 0:
            self.update("failed", reason="evaluator_exited_nonzero", step=step,
                        returncode=completion["returncode"], log=str(directory / "process.log"))
            return None
        summary = read_json(directory / "summary.json")
        if not summary_is_complete(summary, step=step):
            self.update("failed", reason="evaluator_summary_missing_or_incomplete",
                        step=step, summary_path=str(directory / "summary.json"))
            return None
        self.update("checkpoint_completed", step=step, summary_path=str(directory / "summary.json"))
        return summary

    def run(self) -> int:
        if self.state.get("status") == "completed":
            return 0
        if self.state.get("status") in {"failed", "blocked"}:
            # No automatic recovery even if the permanent attempt failed before
            # CUDA initialization; the operator must inspect the preserved logs.
            return 2
        self.update("starting")
        if not self.build_cache():
            return 2
        summaries = []
        for step in STEPS:
            summary = self.evaluate(step)
            if summary is None:
                return 2
            summaries.append(summary)
        comparison = {
            "complete": True, "created_utc": utcnow(), "checkpoint_steps": STEPS,
            "windows": WINDOWS, "gpu_uuid": GPU_UUID, "gpu_index": GPU_INDEX,
            "batch_size": 16, "seed": 42, "weights": "EMA",
            "plan_sha256": self.state["plan_sha256"],
            "cache_summary": read_json(self.cache_summary), "results": summaries,
        }
        atomic_json(self.output_root / "comparison.json", comparison)
        self.update("completed", completed_utc=utcnow(),
                    comparison_path=str(self.output_root / "comparison.json"),
                    completed_steps=STEPS)
        return 0


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("gpu_uuid") != GPU_UUID or int(plan.get("gpu_index", GPU_INDEX)) != GPU_INDEX:
        raise ValueError("This evaluation is restricted to the authorized GPU 5 UUID")
    if plan.get("steps") != STEPS:
        raise ValueError("steps must be exactly [5000, 10000, 15000]")
    for key in ("checkpoint_root", "output_root", "code_dir", "eval_dir", "python", "cache_dir"):
        if not isinstance(plan.get(key), str) or not Path(plan[key]).is_absolute():
            raise ValueError(f"plan.{key} must be an absolute path")
    checkpoint_root = Path(plan["checkpoint_root"]).resolve()
    code_dir = Path(plan["code_dir"]).resolve()
    for key in ("output_root", "cache_dir", "eval_dir"):
        destination = Path(plan[key]).resolve()
        if destination == checkpoint_root or checkpoint_root in destination.parents:
            raise ValueError(f"plan.{key} must not write inside checkpoints")
        if destination == code_dir or code_dir in destination.parents:
            raise ValueError(f"plan.{key} must not write inside original training code")
    for filename in ("build_cache.py", "evaluate_checkpoint.py"):
        if not (Path(plan["eval_dir"]) / filename).is_file():
            raise ValueError(f"Missing evaluation source: {filename}")
    for filename, expected in plan.get('eval_sources_sha256', {}).items():
        if sha256(Path(plan['eval_dir']) / filename) != expected:
            raise ValueError(f'Evaluation source SHA changed: {filename}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument('--execute', action='store_true',
                        help='Explicitly start the prepared CPU/GPU evaluation queue; otherwise inspect only')
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = read_json(plan_path)
    if plan is None:
        parser.error("plan file does not exist")
    validate_plan(plan)
    if not args.execute:
        print(json.dumps({'status': 'prepared_only', 'plan_path': str(plan_path),
                          'gpu_index': GPU_INDEX, 'checkpoint_steps': STEPS,
                          'message': 'No jobs started. Add --execute only when evaluation is wanted.'}))
        return 0
    output_root = Path(plan["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "already_running", "state_path": str(output_root / "state.json")}))
            return 0
        controller = Controller(plan_path, plan)
        old_hash = (read_json(output_root / "state.json") or {}).get("plan_sha256")
        if old_hash and old_hash != sha256(plan_path):
            raise ValueError("Plan SHA256 changed after controller initialization")
        try:
            result = controller.run()
        except BaseException as exc:
            controller.update("failed", reason="controller_exception",
                              error=f"{type(exc).__name__}: {exc}")
            raise
        print(json.dumps({"status": controller.state.get("status"),
                          "state_path": str(controller.state_path)}, ensure_ascii=False))
        return result


if __name__ == "__main__":
    raise SystemExit(main())
