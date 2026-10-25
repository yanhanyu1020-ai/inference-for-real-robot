#!/usr/bin/env python3
"""Manually launch one fresh V3 experiment; never resume or overwrite a run.

This entry imports only the standard library and the CPU-only queue. --check
performs the same read-only readiness checks without creating a queue or claim.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from typing import Any

HERE = Path(__file__).resolve().parent
RUN_NAME = "vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_data20261021"
MODEL_ROOT = Path("/meta_eon_cfs/home/hph/heterovla-model/V3")
CONFIG_NAME = "pi05_heterovla_v3_vitlora"
PYTHON = "/meta_eon_cfs/home/hph/heterovla/.venvs/openpi-pi05/bin/python"


def load_queue():
    spec = importlib.util.spec_from_file_location("fresh_v3_queue", HERE / "queue_training.py")
    if spec is None or spec.loader is None:
        raise ValueError("Could not load the CPU-only queue module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def queue_arguments(nvidia_smi: str) -> argparse.Namespace:
    return argparse.Namespace(
        code_dir=HERE,
        queue_dir=MODEL_ROOT / "queues" / RUN_NAME,
        run_dir=MODEL_ROOT / "runs" / RUN_NAME,
        run_name=RUN_NAME,
        python=PYTHON,
        nvidia_smi=nvidia_smi,
        preferred_gpu=None,
    )


def verify_fresh_configuration(code_dir: Path) -> None:
    """Read configuration without importing OpenPI, JAX, or model weights."""
    tree = ast.parse((code_dir / "configuration.py").read_text())
    configs = [node for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "TrainConfig"]
    if len(configs) != 1:
        raise ValueError("Expected exactly one explicit TrainConfig in configuration.py")
    values = {keyword.arg: keyword.value for keyword in configs[0].keywords}
    for name in ("resume", "overwrite"):
        value = values.get(name)
        if not isinstance(value, ast.Constant) or value.value is not False:
            raise ValueError(f"Fresh training requires literal {name}=False")
    names = {node.targets[0].id: node.value.value for node in tree.body
             if isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)}
    if names.get("RUN_NAME") != RUN_NAME or names.get("CONFIG_NAME") != CONFIG_NAME:
        raise ValueError("Configuration run name or checkpoint namespace differs from this launcher")


def verify_empty_destination(args: argparse.Namespace) -> None:
    checkpoint_root = MODEL_ROOT / "checkpoints" / CONFIG_NAME / RUN_NAME
    if checkpoint_root.exists() and any(path.name.isdigit() for path in checkpoint_root.iterdir()):
        raise ValueError("This run already has checkpoint directories; use a separate resume entry, never fresh overwrite")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("This run directory is nonempty; inspect it and use a separate resume entry if appropriate")


def query_indices(binary: str) -> dict[str, str]:
    result = subprocess.run(
        [binary, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=20,
    )
    mapping = {}
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row:
            continue
        parts = [item.strip() for item in row]
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].startswith("GPU-"):
            raise ValueError(f"Unparseable index/UUID row: {parts!r}")
        if parts[0] in mapping or parts[1] in mapping.values():
            raise ValueError("Duplicate GPU index or UUID")
        mapping[parts[0]] = parts[1]
    return mapping


def parse_gpu(value: str) -> str:
    if value == "auto" or value.isdigit() or value.startswith("GPU-"):
        return value
    raise argparse.ArgumentTypeError("--gpu must be auto, a nonnegative card index, or a GPU UUID")


def observed_progress(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") == "running":
        result = dict(result)
        result["detail"] = "Training process started; initialization may still be in progress. Running does not prove optimizer updates."
    return result


def execute(options: argparse.Namespace, queue=None) -> dict[str, Any]:
    q = queue if queue is not None else load_queue()
    args = queue_arguments(options.nvidia_smi)
    # A permanent claim always wins, including after failures or a reboot. Do
    # not silently reuse a different GPU, start again, or reset an experiment.
    if (args.queue_dir / "attempt.json").exists():
        return observed_progress(q.snapshot(args))
    try:
        proof = q.verify_preflight(args.code_dir)
        verify_fresh_configuration(args.code_dir)
        verify_empty_destination(args)
        gpus = q.query_gpus(options.nvidia_smi)
        indices = query_indices(options.nvidia_smi)
        if set(indices.values()) != {gpu["uuid"] for gpu in gpus}:
            raise ValueError("GPU inventory changed between index and memory/process queries; retry the command")
        if options.gpu != "auto":
            args.preferred_gpu = indices.get(options.gpu) if options.gpu.isdigit() else options.gpu
            if args.preferred_gpu not in indices.values():
                raise ValueError(f"Requested GPU was not found: {options.gpu}")
        candidates = sorted(
            (gpu for gpu in gpus if q.available(gpu)
             and (args.preferred_gpu is None or gpu["uuid"] == args.preferred_gpu)),
            key=lambda gpu: gpu["uuid"],
        )
        if not candidates:
            return {"status": "waiting", "run_name": RUN_NAME, "requested_gpu": options.gpu,
                    "check_only": options.check, "attempt_exists": False,
                    "detail": "No eligible idle 80GB GPU with at least 95% free memory; no attempt created"}
        if options.check:
            return {"status": "ready", "run_name": RUN_NAME, "requested_gpu": options.gpu,
                    "gpu": candidates[0], "check_only": True, "attempt_exists": False,
                    "proof": proof, "resume": False, "overwrite": False,
                    "detail": "Read-only checks passed; no process, queue, or attempt created"}
        return observed_progress(q.check_and_start(args))
    except Exception as error:
        return {"status": "blocked", "run_name": RUN_NAME, "requested_gpu": options.gpu,
                "check_only": options.check, "attempt_exists": False,
                "detail": f"{type(error).__name__}: {error}"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--gpu", type=parse_gpu, default="auto",
                        help="auto (default), physical GPU index, or exact GPU UUID; explicit choices never fall back")
    result.add_argument("--check", action="store_true", help="Read-only readiness check; do not start training")
    result.add_argument("--nvidia-smi", default="nvidia-smi", help=argparse.SUPPRESS)
    return result


def main() -> int:
    result = execute(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 1 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
