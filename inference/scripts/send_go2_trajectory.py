#!/usr/bin/env python3
"""Send a Go2 velocity trajectory from the GPU server through the control channel.

Mirrors the Piper path (robotctl + action_chunk), but actions are 3-D:
    [vx, vy, vyaw]  in m/s and rad/s

Prerequisites on the robot side (see README_zh.md):
  - control broker + agent online with --execute
  - remote closed; lease free
  - bash inference/scripts/robotctl.sh system.go2_control_start   # C++ bridge + chunk loop

Examples:
  python3 inference/scripts/send_go2_trajectory.py
  python3 inference/scripts/send_go2_trajectory.py --file dog_traj.json
  python3 inference/scripts/send_go2_trajectory.py --vx 0.3 --seconds 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


INFERENCE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INFERENCE_ROOT.parent
VX_LIMIT = 0.5
VY_LIMIT = 0.5
VYAW_LIMIT = 1.0


def smooth_step_waypoints(waypoints, steps_per_segment):
    chunks = []
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        for s in range(steps_per_segment):
            t = (s + 1) / float(steps_per_segment)
            eased = 0.5 - 0.5 * math.cos(math.pi * t)
            chunks.append([a[j] + (b[j] - a[j]) * eased for j in range(3)])
    return chunks


def default_trajectory(steps_per_segment=40):
    """Clear demo: hold each phase so motion is obvious on the dog.

    Phases (default 40 steps × 50 ms ≈ 2 s each hold/transition):
      forward hold -> brake -> yaw hold -> brake -> backward hold -> brake.
    Duplicate waypoints keep velocity constant for a full segment (no blend).
    """
    return smooth_step_waypoints(
        [
            [0.3, 0.0, 0.0],
            [0.3, 0.0, 0.0],   # ~2 s forward
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],   # ~2 s stop
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],   # ~2 s yaw in place
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],   # ~2 s stop
            [-0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0], # ~2 s backward
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],   # ~2 s stop
        ],
        steps_per_segment,
    )


def constant_velocity_trajectory(vx, vy, vyaw, seconds, dt=0.05):
    steps = max(1, int(round(seconds / dt)))
    rows = [[vx, vy, vyaw] for _ in range(steps)]
    rows.append([0.0, 0.0, 0.0])
    return rows


def clamp_row(row):
    vx, vy, vyaw = (float(v) for v in row)
    return [
        max(-VX_LIMIT, min(VX_LIMIT, vx)),
        max(-VY_LIMIT, min(VY_LIMIT, vy)),
        max(-VYAW_LIMIT, min(VYAW_LIMIT, vyaw)),
    ]


def load_actions(path):
    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        actions = payload.get("actions")
    else:
        actions = payload
    if not isinstance(actions, list) or not actions:
        raise SystemExit("trajectory file must contain a non-empty actions list")
    return [clamp_row(row) for row in actions]


def build_actions(args):
    if args.file:
        return load_actions(args.file)
    if args.vx is not None or args.vy is not None or args.vyaw is not None:
        return [
            clamp_row(row)
            for row in constant_velocity_trajectory(
                args.vx or 0.0,
                args.vy or 0.0,
                args.vyaw or 0.0,
                args.seconds,
                args.dt,
            )
        ]
    return [clamp_row(row) for row in default_trajectory(args.steps_per_segment)]


def robotctl(command, data=None):
    argv = [str(INFERENCE_ROOT / "scripts" / "robotctl.sh"), command]
    payload = None
    if data is not None:
        argv.append("-")
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    env = os.environ.copy()
    for key in (
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    ):
        env.pop(key, None)
    completed = subprocess.run(
        argv, cwd=REPO_ROOT, env=env, text=True, input=payload)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="JSON with {\"actions\": [[vx,vy,vyaw], ...]}")
    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--vy", type=float, default=None)
    parser.add_argument("--vyaw", type=float, default=None)
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="duration when using --vx/--vy/--vyaw")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="seconds per row for constant-velocity mode")
    parser.add_argument("--steps-per-segment", type=int, default=40,
                        help="interpolation steps for the built-in demo trajectory")
    parser.add_argument("--ensure-control", action="store_true",
                        help="call system.go2_control_start before sending")
    parser.add_argument("--dump", metavar="PATH",
                        help="write the actions JSON and exit (do not send)")
    args = parser.parse_args()

    actions = build_actions(args)
    if not 1 <= len(actions) <= 5000:
        raise SystemExit("actions length must be 1..5000, got %d" % len(actions))
    payload = {"actions": actions}

    if args.dump:
        with open(args.dump, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        print("wrote %d rows to %s" % (len(actions), args.dump), file=sys.stderr)
        return

    if args.ensure_control:
        robotctl("system.go2_control_start")

    print("sending go2.action_chunk: %d rows" % len(actions), file=sys.stderr)
    robotctl("go2.action_chunk", payload)


if __name__ == "__main__":
    main()
