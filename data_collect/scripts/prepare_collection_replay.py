#!/usr/bin/env python3
"""Build replay JSON files from a transferred heterovla-collection episode.

Collection episodes store high-rate CSVs under ``raw/``.  This script writes
``go2_replay.json`` and ``piper_replay.json`` next to them so
``send_hetero_demo.py`` can replay the recorded motion on hardware.

    python3 data_collect/scripts/prepare_collection_replay.py \\
      ~/heterovla-data/raw/go2_remote_test/episode_001
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys


REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(
    0,
    os.path.join(REPO_ROOT, "inference", "onboard", "unitree_go2", "control"),
)
from episode_log import (  # noqa: E402
    GO2_REPLAY_DT,
    build_go2_replay_from_sport,
    downsample_rows,
    write_replay_json,
)

JOINT_FIELDS = tuple(f"joint_{i}.pos" for i in range(1, 7))


def _find_raw(episode_dir: str, name: str) -> str | None:
    for candidate in (
        os.path.join(episode_dir, "raw", name),
        os.path.join(episode_dir, name),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _piper_actions(csv_path: str) -> list[list[float]]:
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            try:
                joints = [math.degrees(float(item[name])) for name in JOINT_FIELDS]
                gripper = float(item["gripper.pos"]) * 1000.0
                rows.append(joints + [gripper])
            except (TypeError, ValueError, KeyError):
                continue
    return downsample_rows(rows)


def prepare(episode_dir: str, dt: float = GO2_REPLAY_DT) -> dict[str, str | None]:
    episode_dir = os.path.abspath(episode_dir)
    outputs: dict[str, str | None] = {"go2_replay.json": None, "piper_replay.json": None}

    sport_path = _find_raw(episode_dir, "sport_mode_state.csv")
    if sport_path:
        actions, pose_events = build_go2_replay_from_sport(sport_path, dt=dt)
        if actions:
            out = os.path.join(episode_dir, "go2_replay.json")
            write_replay_json(
                out, actions, source="sport_mode_state", pose_events=pose_events)
            outputs["go2_replay.json"] = out

    piper_path = _find_raw(episode_dir, "piper_state.csv")
    if piper_path:
        actions = _piper_actions(piper_path)
        if actions:
            out = os.path.join(episode_dir, "piper_replay.json")
            write_replay_json(out, actions, source="piper_state.csv")
            outputs["piper_replay.json"] = out

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="transferred episode directory on A800")
    parser.add_argument("--dt", type=float, default=GO2_REPLAY_DT)
    args = parser.parse_args()
    outputs = prepare(args.episode, dt=args.dt)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    if not any(outputs.values()):
        raise SystemExit("no replay files were generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
