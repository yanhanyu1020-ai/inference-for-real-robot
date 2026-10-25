#!/usr/bin/env python3
"""Validate a stopped raw episode and write a transfer manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Any


REQUIRED_RAW_FILES = (
    "wireless_controller.csv",
    "sport_mode_state.csv",
    "low_state.csv",
    "piper_state.csv",
    "piper_status.csv",
    "piper_gamepad.csv",
)
ALLOW_EMPTY_RAW_FILES = {"wireless_controller.csv"}
CONTINUOUS_STATE_FILES = {"sport_mode_state.csv", "low_state.csv", "piper_state.csv"}
# State subscribers may start before cameras; only their coverage of the
# captured frame interval matters. Allow modest endpoint scheduling jitter.
FRAME_COVERAGE_TOLERANCE_NS = 250_000_000


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def csv_stats(path: Path) -> dict[str, Any]:
    rows = 0
    first = None
    last = None
    monotonic = True
    previous = None
    with path.open(newline="") as handle:
        for item in csv.DictReader(handle):
            timestamp = int(item["wall_time_ns"])
            if first is None:
                first = timestamp
            if previous is not None and timestamp < previous:
                monotonic = False
            previous = timestamp
            last = timestamp
            rows += 1
    duration_s = 0.0 if first is None or last is None else max(0.0, (last - first) / 1e9)
    rate_hz = (rows - 1) / duration_s if duration_s > 0 and rows > 1 else 0.0
    return {
        "rows": rows,
        "first_wall_time_ns": first,
        "last_wall_time_ns": last,
        "duration_s": duration_s,
        "rate_hz": rate_hz,
        "monotonic": monotonic,
    }


def inspect_frames(frames_dir: Path, expected_cameras: dict[str, str]) -> dict[str, Any]:
    paths = sorted(frames_dir.glob("*.pkl"))
    errors = []
    first = None
    last = None
    previous = None
    max_camera_skew = 0
    for path in paths:
        try:
            with path.open("rb") as handle:
                record = pickle.load(handle)
            timestamp = int(record["timestamp_ns"])
            if previous is not None and timestamp <= previous:
                errors.append(f"non-increasing timestamp: {path.name}")
            previous = timestamp
            first = timestamp if first is None else first
            last = timestamp
            for name, serial in expected_cameras.items():
                camera = record["camera"][name]
                if camera["serial"] != serial:
                    errors.append(f"{path.name}: {name} serial mismatch")
                rgb = camera.get("rgb")
                if not isinstance(rgb, bytes) or not rgb.startswith(b"\x89PNG\r\n\x1a\n"):
                    errors.append(f"{path.name}: {name} is not a PNG byte stream")
                else:
                    try:
                        # Offline only: verify chunk checksums AND fully decode IDAT.
                        from PIL import Image
                        with Image.open(io.BytesIO(rgb)) as image:
                            if image.format != 'PNG':
                                raise ValueError('unexpected image format')
                            image.verify()
                        with Image.open(io.BytesIO(rgb)) as image:
                            image.load()
                    except Exception as exc:
                        errors.append(f"{path.name}: {name} invalid PNG: {exc}")
            skew = int(record.get("diagnostics", {}).get("camera_skew_ns", 0))
            max_camera_skew = max(max_camera_skew, skew)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    duration_s = 0.0 if first is None or last is None else max(0.0, (last - first) / 1e9)
    frame_rate = (len(paths) - 1) / duration_s if duration_s > 0 and len(paths) > 1 else 0.0
    return {
        "frames": len(paths),
        "first_timestamp_ns": first,
        "last_timestamp_ns": last,
        "duration_s": duration_s,
        "effective_fps": frame_rate,
        "max_camera_skew_ns": max_camera_skew,
        "errors": errors,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def build_manifest(episode: Path) -> dict[str, Any]:
    excluded = {"manifest.json", "validation_report.json", "status.json"}
    files = []
    for path in sorted(item for item in episode.rglob("*") if item.is_file()):
        relative = path.relative_to(episode).as_posix()
        if relative in excluded or Path(relative).name.startswith("."):
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    return {"version": 1, "file_count": len(files), "files": files}


def validate(episode: Path, config: dict[str, Any], write: bool) -> dict[str, Any]:
    meta_path = episode / 'meta.json'
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if metadata.get('format_version', 1) != 1:
            from episode_io import validate_raw_episode
            report = validate_raw_episode(episode, config)
            if write:
                atomic_json(episode / 'validation_report.json', report)
                if report['ok']:
                    atomic_json(episode / 'manifest.json', build_manifest(episode))
            return report
    errors = []
    warnings = []
    raw_stats = {}
    raw_dir = episode / "raw"
    for name in REQUIRED_RAW_FILES:
        path = raw_dir / name
        if not path.is_file():
            errors.append(f"missing raw/{name}")
            continue
        stats = csv_stats(path)
        raw_stats[name] = stats
        if stats["rows"] == 0:
            message = f"raw/{name} has no rows"
            if name in ALLOW_EMPTY_RAW_FILES:
                warnings.append(message)
            else:
                errors.append(message)
        if not stats["monotonic"]:
            errors.append(f"raw/{name} timestamps are not monotonic")

    frame_stats = inspect_frames(episode / "frames", config["cameras"])
    errors.extend(frame_stats["errors"])
    if frame_stats["frames"] < 2:
        errors.append("frames directory must contain at least two PKL files")
    if frame_stats["effective_fps"] < config["fps"] * 0.8:
        errors.append(
            f"effective PKL rate too low: {frame_stats['effective_fps']:.2f} Hz"
        )
    if frame_stats["max_camera_skew_ns"] > int(config["max_camera_skew_ms"] * 1e6):
        errors.append("camera skew exceeds configured maximum")

    try:
        summary = json.loads((episode / "summary.json").read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("expected an object")
        if type(summary.get("save_errors")) is not int or summary["save_errors"] != 0:
            errors.append("summary.json save_errors must be zero")
        if "frames_dropped" in summary and (
                type(summary["frames_dropped"]) is not int or summary["frames_dropped"] != 0):
            errors.append("summary.json frames_dropped must be zero")
        if "piper_error" not in summary or summary["piper_error"] is not None:
            errors.append("summary.json piper_error must be null")
        if (type(summary.get("frames_saved")) is not int
                or summary["frames_saved"] != frame_stats["frames"]):
            errors.append("summary.json frames_saved does not match inspected PKL count")
    except (OSError, ValueError) as exc:
        errors.append(f"missing or invalid summary.json: {exc}")

    first_frame = frame_stats["first_timestamp_ns"]
    last_frame = frame_stats["last_timestamp_ns"]
    if first_frame is not None and last_frame is not None:
        for name in sorted(CONTINUOUS_STATE_FILES):
            stats = raw_stats.get(name)
            if stats is None or stats["rows"] == 0:
                continue  # Already reported as missing or empty above.
            if (stats["first_wall_time_ns"] > first_frame + FRAME_COVERAGE_TOLERANCE_NS
                    or stats["last_wall_time_ns"] < last_frame - FRAME_COVERAGE_TOLERANCE_NS):
                errors.append(f"raw/{name} does not cover frame interval within 250 ms")

    report = {
        "ok": not errors,
        "episode": str(episode),
        "frames": frame_stats,
        "raw": raw_stats,
        "errors": errors,
        "warnings": warnings,
    }
    if write:
        atomic_json(episode / "validation_report.json", report)
        if report["ok"]:
            atomic_json(episode / "manifest.json", build_manifest(episode))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = validate(args.episode.resolve(), config, args.write)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
