#!/usr/bin/env python3
"""Build go2_replay.json from measured sport_mode_state.csv.

Joystick command logs stop when the stick returns to center, but the dog may
keep walking.  Replay is built from pose deltas so sum(vx,vy,vyaw)*dt tracks
the recorded path and heading, not clipped IMU yaw_speed.

    python3 data_collect/scripts/sport_to_go2_replay.py path/to/sport_mode_state.csv
    python3 data_collect/scripts/sport_to_go2_replay.py path/to/episode_dir
"""

from __future__ import annotations

import argparse
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
    write_go2_replay,
    write_replay_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="sport_mode_state.csv or an episode directory")
    parser.add_argument("-o", "--output", help="go2_replay.json path")
    parser.add_argument("--dt", type=float, default=GO2_REPLAY_DT)
    args = parser.parse_args()
    path = os.path.abspath(args.path)
    if os.path.isdir(path):
        out = write_go2_replay(path, dt=args.dt)
        if not out:
            raise SystemExit("no usable sport_mode_state.csv in %s" % path)
        print("wrote %s" % out)
        return
    actions, pose_events = build_go2_replay_from_sport(path, dt=args.dt)
    if not actions:
        raise SystemExit("no samples in %s" % path)
    out = args.output or os.path.join(os.path.dirname(path), "go2_replay.json")
    write_replay_json(
        out, actions, source="sport_mode_state", pose_events=pose_events)
    print("wrote %s (%d rows, %.1fs, %d pose events)" % (
        out, len(actions), len(actions) * args.dt, len(pose_events)))


if __name__ == "__main__":
    main()
