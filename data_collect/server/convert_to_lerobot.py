#!/usr/bin/env python3
"""Convert trusted HeteroVLA raw episodes to LeRobot Parquet + videos."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PIPER_NAMES = tuple(f"joint_{i}.pos" for i in range(1, 7)) + ("gripper.pos",)
STATE_NAMES = (
    "base.position_x",
    "base.position_y",
    "base.yaw",
    "base.velocity_x",
    "base.velocity_y",
    "base.yaw_speed",
    *PIPER_NAMES,
)
ACTION_NAMES = (
    "base.vx",
    "base.vy",
    "base.vyaw",
    *PIPER_NAMES,
)
CAMERA_NAMES = ("front", "wrist")


@dataclass
class TimeSeries:
    timestamps: np.ndarray
    values: dict[str, np.ndarray]

    @property
    def start(self) -> int:
        return int(self.timestamps[0])

    @property
    def end(self) -> int:
        return int(self.timestamps[-1])

    def interpolate(self, timestamp_ns: int, fields: Iterable[str]) -> list[float]:
        target = float(timestamp_ns)
        return [
            float(np.interp(target, self.timestamps, self.values[field]))
            for field in fields
        ]


def load_csv_series(path: Path, fields: Iterable[str]) -> TimeSeries:
    field_names = tuple(fields)
    timestamps = []
    values = {field: [] for field in field_names}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamps.append(int(row["wall_time_ns"]))
            for field in field_names:
                values[field].append(float(row[field]))
    if not timestamps:
        raise ValueError(f"no rows in {path}")
    order = np.argsort(np.asarray(timestamps, dtype=np.int64))
    return TimeSeries(
        timestamps=np.asarray(timestamps, dtype=np.int64)[order],
        values={
            field: np.asarray(items, dtype=np.float64)[order]
            for field, items in values.items()
        },
    )


def discover_episodes(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if (path / "frames").is_dir():
        return [path]
    return sorted(
        item.parent
        for item in path.rglob("frames")
        if item.is_dir() and any(item.glob("*.pkl"))
    )


def decode_png(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode PNG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_record(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_features(height: int, width: int) -> dict:
    features: dict = {
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": list(ACTION_NAMES),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
    }
    for camera in CAMERA_NAMES:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
        }
    return features


def episode_frames(
    episode: Path,
    action_offset_ns: int,
) -> Iterable[dict]:
    sport_fields = (
        "position_x",
        "position_y",
        "yaw",
        "velocity_x",
        "velocity_y",
        "yaw_speed",
    )
    sport = load_csv_series(episode / "raw" / "sport_mode_state.csv", sport_fields)
    piper = load_csv_series(episode / "raw" / "piper_state.csv", PIPER_NAMES)
    support_start = max(sport.start, piper.start)
    support_end = min(sport.end, piper.end)

    for path in sorted((episode / "frames").glob("*.pkl")):
        record = load_record(path)
        timestamp = int(record["timestamp_ns"])
        action_timestamp = timestamp + action_offset_ns
        if timestamp < support_start or action_timestamp > support_end:
            continue
        state = sport.interpolate(timestamp, sport_fields)
        state.extend(piper.interpolate(timestamp, PIPER_NAMES))
        action = sport.interpolate(
            action_timestamp, ("velocity_x", "velocity_y", "yaw_speed")
        )
        action.extend(piper.interpolate(action_timestamp, PIPER_NAMES))
        frame = {
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }
        for camera in CAMERA_NAMES:
            frame[f"observation.images.{camera}"] = decode_png(
                record["camera"][camera]["rgb"]
            )
        yield frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/heterovla_collection")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action-offset-ms", type=float, default=50.0)
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "LeRobot is not installed in this Python environment. "
            "Create the pinned A800 conversion environment first."
        ) from exc

    episodes = discover_episodes(args.input)
    if not episodes:
        raise SystemExit(f"no episodes found under {args.input}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    first_path = next(iter(sorted((episodes[0] / "frames").glob("*.pkl"))), None)
    if first_path is None:
        raise SystemExit(f"first episode has no PKL files: {episodes[0]}")
    first = load_record(first_path)
    sample = decode_png(first["camera"][CAMERA_NAMES[0]]["rgb"])
    height, width = sample.shape[:2]

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=build_features(height, width),
        root=output,
        robot_type="go2_piper",
        use_videos=True,
        image_writer_threads=4,
    )
    action_offset_ns = int(round(args.action_offset_ms * 1e6))
    report = []
    for episode in episodes:
        meta_path = episode / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        task = meta.get("instruction") or meta.get("task") or "HeteroVLA task"
        count = 0
        for frame in episode_frames(episode, action_offset_ns):
            dataset.add_frame(frame, task=task)
            count += 1
        if count == 0:
            raise RuntimeError(f"episode has no convertible frames: {episode}")
        dataset.save_episode()
        report.append({"episode": str(episode), "frames": count, "task": task})
        print(f"converted {episode.name}: {count} frames")

    (output / "conversion_report.json").write_text(
        json.dumps(
            {
                "source": str(args.input.expanduser().resolve()),
                "fps": args.fps,
                "action_offset_ms": args.action_offset_ms,
                "episodes": report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"LeRobot dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
