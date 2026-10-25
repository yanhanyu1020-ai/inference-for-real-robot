#!/usr/bin/env python3
"""Coordinated Go2 + Piper demo from the GPU server.

Default scene (open-loop, scripted; not vision grasp):
  1. Go2 forward 2 s
  2. Go2 yaw right ~90 deg
  3. Go2 lie down (STAND_DOWN through the lease-holding bridge)
  4. Piper scripted reach → close gripper → lift

Dog and arm use independent executors, so they can overlap; this script waits
until the dog has finished walking AND the lie-down animation before the arm
starts. Do not treat the chunk-received log ``after=stand_down`` as done.

Prerequisites (see README_zh.md):
  - remote closed; control broker + agent online with --execute
  - ``bash inference/scripts/robotctl.sh system.go2_control_start``
  - Jetson ``go2_chunk_loop.py`` that honors before/after pose fields
    (stand_up/stand_down go through the lease bridge; do NOT use
    ``robotctl go2.stand_up`` while the bridge is up)

Examples:
  python3 inference/scripts/send_hetero_demo.py --ensure-control
  python3 inference/scripts/send_hetero_demo.py --ensure-control --record
  python3 inference/scripts/send_hetero_demo.py --no-stand-up --skip-arm
  python3 inference/scripts/send_hetero_demo.py --arm-file my_grasp.json
  python3 inference/scripts/send_hetero_demo.py --ensure-control --together --no-stand-down \\
    --dog-file go2_replay.json --arm-file piper_replay.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


INFERENCE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INFERENCE_ROOT.parent
sys.path.insert(
    0,
    str(INFERENCE_ROOT / "onboard" / "unitree_go2" / "control"),
)
from episode_log import GO2_REPLAY_DT  # noqa: E402

VX_LIMIT = 1.0
VY_LIMIT = 1.0
VYAW_LIMIT = 3.5
GO2_DT = 0.05
PIPER_DT = 0.005
MAX_CHUNK_ROWS = 5000
MAX_TOTAL_ROWS = 60000
STAND_UP_HOLD_S = 3.0
STAND_DOWN_HOLD_S = 3.5
POST_STAND_UP_SETTLE_S = 0.3
POST_STAND_DOWN_SETTLE_S = 0.0
CHUNK_LOOP_HINT = """
机载 go2_chunk_loop.py 还不支持 before/after。
A800 解析不了 hostname unitree，必须在狗上把文件拉下来（不要用 robotctl go2.stand_up，
那会另起 go2_command_executor 抢桥的租约，卡 12 秒、狗也不动）。

在 Jetson 上执行：
  scp -F ~/heterovla-control/ssh_config \\
    heterovla-a800:~/heterovla/inference/onboard/unitree_go2/control/go2_chunk_loop.py \\
    ~/heterovla-control/go2_chunk_loop.py

然后在 A800：
  bash inference/scripts/robotctl.sh system.go2_control_stop
  bash inference/scripts/robotctl.sh system.go2_control_start
  python3 inference/scripts/send_hetero_demo.py --ensure-control
""".strip()
ARM_HINT = """
机械臂 chunk 执行器没起来。常见原因：agent 用虚拟环境 Python 起 piper_chunk_loop.py，
里面没有 piper_sdk（piper.go_zero 用的是 /usr/bin/python3，所以还能动）。

在 Jetson 上拉最新 agent 并重启（改的是 robot_command_agent.py，必须重启）：

  bash ~/heterovla/inference/onboard/scripts/pull_control_on_go2.sh
  bash ~/heterovla-control/control_up.sh

然后在 A800：
  bash inference/scripts/robotctl.sh system.control_stop
  bash inference/scripts/robotctl.sh system.control_start
  bash inference/scripts/robotctl.sh system.control_status
  # 期望 pid=数字，且没有 ModuleNotFoundError
  python3 inference/scripts/send_hetero_demo.py --skip-dog

若暂时只想看夹爪动，不必等 chunk：
  python3 inference/scripts/send_hetero_demo.py --skip-dog --arm-basic
""".strip()

JOINT_LIMITS_DEG = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)
GRIPPER_RANGE_MM = (1.0, 65.0)


def clamp_go2(row):
    vx, vy, vyaw = (float(v) for v in row)
    return [
        max(-VX_LIMIT, min(VX_LIMIT, vx)),
        max(-VY_LIMIT, min(VY_LIMIT, vy)),
        max(-VYAW_LIMIT, min(VYAW_LIMIT, vyaw)),
    ]


def clamp_piper(row):
    joints = [float(v) for v in row[:6]]
    gripper = float(row[6])
    out = [
        max(lo, min(hi, joints[i]))
        for i, (lo, hi) in enumerate(JOINT_LIMITS_DEG)
    ]
    out.append(max(GRIPPER_RANGE_MM[0], min(GRIPPER_RANGE_MM[1], gripper)))
    return [round(v, 3) for v in out]


def hold_rows(row, seconds, dt):
    n = max(1, int(round(seconds / dt)))
    return [list(row) for _ in range(n)]


def dog_approach_trajectory(forward_s, vx, yaw_deg, yaw_rate, dt, pause_s=0.4):
    """Open-loop: forward, pause, yaw in place, stop.

    yaw_deg > 0 is left (positive vyaw); yaw_deg < 0 is right.
    """
    rows = []
    rows.extend(hold_rows([vx, 0.0, 0.0], forward_s, dt))
    rows.extend(hold_rows([0.0, 0.0, 0.0], pause_s, dt))
    angle = math.radians(yaw_deg)
    if angle != 0.0:
        rate = math.copysign(abs(yaw_rate), angle)
        duration = abs(angle / rate)
        rows.extend(hold_rows([0.0, 0.0, rate], duration, dt))
    rows.append([0.0, 0.0, 0.0])
    return [clamp_go2(row) for row in rows]


def smooth_step_waypoints(waypoints, steps_per_segment):
    chunks = []
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        dim = len(a)
        for s in range(steps_per_segment):
            t = (s + 1) / float(steps_per_segment)
            eased = 0.5 - 0.5 * math.cos(math.pi * t)
            chunks.append([a[j] + (b[j] - a[j]) * eased for j in range(dim)])
    return chunks


def default_grasp_trajectory(steps_per_segment=200):
    """Reach first, open at hover, then descend/close/lift.

    Do not open the gripper on the first waypoint: a jump to home with
    gripper 50 mm looks like the arm dropped and then the gripper opened.
    """
    waypoints = [
        [0.0, 10.0, -20.0, 0.0, 15.0, 0.0, 2.0],   # compact, gripper closed
        [0.0, 30.0, -50.0, 0.0, 25.0, 0.0, 2.0],   # hover, still closed
        [0.0, 30.0, -50.0, 0.0, 25.0, 0.0, 50.0],  # open at hover
        [0.0, 45.0, -75.0, 0.0, 35.0, 0.0, 50.0],  # descend
        [0.0, 45.0, -75.0, 0.0, 35.0, 0.0, 2.0],   # close
        [0.0, 25.0, -45.0, 0.0, 20.0, 0.0, 2.0],   # lift
    ]
    return [clamp_piper(row) for row in smooth_step_waypoints(
        waypoints, steps_per_segment)]


def load_actions(path, dim):
    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        actions = payload.get("actions")
    else:
        actions = payload
    if not isinstance(actions, list) or not actions:
        raise SystemExit("%s must contain a non-empty actions list" % path)
    rows = []
    for row in actions:
        if not isinstance(row, (list, tuple)) or len(row) != dim:
            raise SystemExit("%s rows must be %d numbers" % (path, dim))
        rows.append(list(row))
    return rows


def load_go2_replay(path):
    with open(path) as handle:
        payload = json.load(handle)
    actions = [clamp_go2(row) for row in load_actions_from_payload(payload, 3)]
    pose_events = payload.get("pose_events") if isinstance(payload, dict) else None
    if pose_events is None:
        pose_events = []
    if not isinstance(pose_events, list):
        raise SystemExit("%s pose_events must be a list" % path)
    return actions, pose_events


def pose_hold_duration(pose_events, before=None):
    total = 0.0
    if str(before or "").strip().lower() == "stand_up":
        total += STAND_UP_HOLD_S + POST_STAND_UP_SETTLE_S
    for item in pose_events:
        if item.get("pose") == "stand_up":
            total += STAND_UP_HOLD_S + POST_STAND_UP_SETTLE_S
        elif item.get("pose") == "stand_down":
            total += STAND_DOWN_HOLD_S + POST_STAND_DOWN_SETTLE_S
    return total


def split_opening_stand_up(actions, pose_events):
    """Run a leading stand_up via before=, keep later squats as mid-chunk poses.

    The first stand_up is peeled off so it does not sit inside the velocity
    clock.  Later stand_down / stand_up pairs stay in pose_events; the chunk
    loop skips Move while prone and resumes after the next StandUp.
    """
    if not actions or not pose_events:
        return actions, pose_events, False
    events = sorted(
        pose_events,
        key=lambda item: int(item["index"]),
    )
    first = events[0]
    if str(first.get("pose", "")).strip().lower() != "stand_up":
        return actions, pose_events, False
    start = max(0, int(first["index"]))
    if start >= len(actions):
        return actions, pose_events, False
    rest = []
    for item in events[1:]:
        idx = int(item["index"]) - start
        if 0 <= idx < len(actions) - start:
            rest.append({"index": idx, "pose": item["pose"]})
    return [list(row) for row in actions[start:]], rest, True


def load_actions_from_payload(payload, dim):
    if isinstance(payload, dict):
        actions = payload.get("actions")
    else:
        actions = payload
    if not isinstance(actions, list) or not actions:
        raise SystemExit("payload must contain a non-empty actions list")
    rows = []
    for row in actions:
        if not isinstance(row, (list, tuple)) or len(row) != dim:
            raise SystemExit("rows must be %d numbers" % dim)
        rows.append(list(row))
    return rows


def robotctl(command, data=None, quiet=False):
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
        argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        input=payload)
    if not quiet:
        if completed.stdout:
            sys.stdout.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if completed.stderr:
            sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        if quiet and completed.stdout:
            sys.stdout.write(completed.stdout)
        raise SystemExit(completed.returncode)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"ok": True, "output": completed.stdout}


def wait(seconds, label):
    seconds = max(0.0, float(seconds))
    print("waiting %.1fs (%s)" % (seconds, label), file=sys.stderr)
    time.sleep(seconds)


def chunk_duration(n_rows, dt):
    return n_rows * dt


def split_rows(rows, size=MAX_CHUNK_ROWS):
    if not rows:
        return []
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def send_go2_chunk(actions, before=None, after=None, pose_events=None):
    payload = {"actions": actions}
    if before:
        payload["before"] = before
    if after:
        payload["after"] = after
    if pose_events:
        payload["pose_events"] = pose_events
    return robotctl("go2.action_chunk", payload)


def send_go2_chunks(actions, before=None, after=None, pose_events=None):
    """Deliver dog rows, splitting at the protocol 5000-row limit."""
    pieces = split_rows(actions)
    offset = 0
    for index, piece in enumerate(pieces):
        payload_before = before if index == 0 else None
        payload_after = after if index == len(pieces) - 1 else None
        chunk_events = []
        if pose_events:
            end = offset + len(piece)
            chunk_events = [
                {"index": ev["index"] - offset, "pose": ev["pose"]}
                for ev in pose_events
                if offset <= ev["index"] < end
            ]
        send_go2_chunk(
            piece,
            before=payload_before,
            after=payload_after,
            pose_events=chunk_events or None,
        )
        offset += len(piece)
        if index < len(pieces) - 1:
            wait(chunk_duration(len(piece), GO2_DT), "dog chunk %d/%d" % (
                index + 1, len(pieces)))


def send_arm_chunks(actions):
    pieces = split_rows(actions)
    for index, piece in enumerate(pieces):
        robotctl("piper.action_chunk", {"actions": piece})
        wait(chunk_duration(len(piece), PIPER_DT) + 0.5,
             "arm chunk %d/%d" % (index + 1, len(pieces)))
    return True


def go2_status_output():
    return str(robotctl("system.go2_control_status", quiet=True).get("output", ""))


def new_log_text(current, baseline=""):
    """Return log text that appeared after ``baseline``.

    Status only tails a few lines, so a previous run's ``after: STAND_DOWN``
    can still be visible. The chunk-received line also contains
    ``after=stand_down`` the moment the walk starts — that is not done.
    """
    current = current or ""
    baseline = baseline or ""
    if not baseline:
        return current
    if current.startswith(baseline):
        return current[len(baseline):]
    base_lines = set(baseline.splitlines())
    return "\n".join(line for line in current.splitlines() if line not in base_lines)


def log_has_any(text, tokens):
    return any(token in (text or "") for token in tokens)


def wait_go2_log(tokens, timeout, label, since=""):
    print("waiting up to %.1fs (%s)" % (timeout, label), file=sys.stderr)
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        last = go2_status_output()
        if log_has_any(new_log_text(last, since), tokens):
            return True, last
        time.sleep(0.4)
    return False, last


# Receipt log is ``chunk received: N rows before=... after=stand_down``.
# That appears when walking starts. Lie-down itself logs ``after: STAND_DOWN``.
LIE_DOWN_STARTED = ("after: STAND_DOWN",)
LIE_DOWN_DONE = ("after pose complete: STAND_DOWN", "chunk done; STOP issued")
STAND_UP_STARTED = ("before: STAND_UP",)
STAND_UP_DONE = ("before pose complete: STAND_UP", "chunk done; STOP issued")


def piper_executor_healthy():
    result = robotctl("system.control_status", quiet=True)
    output = str(result.get("output", ""))
    if "pid=none" in output:
        return False, output
    bad = (
        "No such file", "ModuleNotFoundError", "Traceback",
        "not healthy", "ImportError",
    )
    if any(token in output for token in bad):
        return False, output
    return True, output


def run_arm_basic():
    print("arm basic: go_zero -> open gripper -> close gripper", file=sys.stderr)
    robotctl("piper.go_zero")
    robotctl("piper.open_gripper")
    robotctl("piper.close_gripper")


def run_arm_chunk(arm_actions):
    print("ensuring Piper chunk executor", file=sys.stderr)
    result = robotctl("system.control_start")
    output = str(result.get("output", ""))
    if "already running" in output:
        wait(0.3, "piper chunk already up")
    else:
        wait(2.5, "piper enable + chunk executor")
    healthy, status = piper_executor_healthy()
    if not healthy:
        print(status, file=sys.stderr)
        return False
    arm_s = chunk_duration(len(arm_actions), PIPER_DT)
    print("phase 4: piper.action_chunk %d rows (%.1fs)"
          % (len(arm_actions), arm_s), file=sys.stderr)
    send_arm_chunks(arm_actions)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure-control", action="store_true",
                        help="start Go2 bridge/chunk and Piper chunk executor")
    parser.add_argument("--stand-up", dest="stand_up", action="store_true",
                        default=True,
                        help="stand via the bridge before walking (default)")
    parser.add_argument("--no-stand-up", dest="stand_up", action="store_false",
                        help="skip stand-up (dog must already be standing)")
    parser.add_argument("--skip-dog", action="store_true")
    parser.add_argument("--skip-arm", action="store_true")
    parser.add_argument("--arm-basic", action="store_true",
                        help="use piper.go_zero/open/close instead of action_chunk "
                             "(works without piper_chunk_loop.py)")
    parser.add_argument("--forward-seconds", type=float, default=2.0)
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--yaw-deg", type=float, default=-90.0,
                        help="in-place yaw; negative is right (default -90)")
    parser.add_argument("--yaw-rate", type=float, default=0.5,
                        help="abs yaw rate rad/s (sign comes from --yaw-deg)")
    parser.add_argument("--stand-down-wait", type=float, default=4.0,
                        help="extra seconds to settle after lie-down before the arm")
    parser.add_argument("--dog-file", help="JSON {\"actions\": [[vx,vy,vyaw], ...]}")
    parser.add_argument("--arm-file", help="JSON {\"actions\": [[j1..j6,grip], ...]}")
    parser.add_argument("--together", action="store_true",
                        help="run dog and arm chunks at the same time (teleop replay); "
                             "implies no lie-down between them")
    parser.add_argument("--stand-down", dest="stand_down", action="store_true",
                        default=True,
                        help="lie down after the dog chunk (default, scripted demo)")
    parser.add_argument("--no-stand-down", dest="stand_down", action="store_false",
                        help="do not lie down after walking (use for teleop replay)")
    parser.add_argument("--arm-steps", type=int, default=200,
                        help="interpolation steps per grasp waypoint")
    parser.add_argument("--dump-dog", metavar="PATH",
                        help="write the dog actions JSON and exit")
    parser.add_argument("--dump-arm", metavar="PATH",
                        help="write the arm actions JSON and exit")
    parser.add_argument(
        "--record", nargs="?", const="", metavar="EPISODE_ID",
        help="record dog+arm into ~/heterovla-data/raw/<id> on the Jetson "
             "(default id demo_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--record-instruction", default="hetero demo",
        help="instruction.txt for --record",
    )
    args = parser.parse_args()

    dog_pose_events = []
    dog_before = None
    if args.dog_file:
        dog_actions, dog_pose_events = load_go2_replay(args.dog_file)
        dog_actions, dog_pose_events, opening_stand = split_opening_stand_up(
            dog_actions, dog_pose_events)
        if opening_stand:
            dog_before = "stand_up"
            args.stand_up = False
            leftover = ", ".join(
                "%s@%.1fs" % (ev["pose"], ev["index"] * GO2_DT)
                for ev in dog_pose_events
            ) or "none"
            print("before=stand_up on chunk, then walk (%.1fs); mid-chunk pose: %s"
                  % (len(dog_actions) * GO2_DT, leftover), file=sys.stderr)
        elif dog_pose_events and args.stand_up:
            print("dog replay includes body_height pose_events; "
                  "skipping default stand-up", file=sys.stderr)
            args.stand_up = False
        if dog_pose_events and not opening_stand:
            first = min(ev["index"] for ev in dog_pose_events) * GO2_DT
            print("dog pose replay: first event around %.1fs "
                  "(%d events)" % (first, len(dog_pose_events)), file=sys.stderr)
    else:
        dog_actions = dog_approach_trajectory(
            args.forward_seconds, args.vx, args.yaw_deg, args.yaw_rate, GO2_DT)
    if args.arm_file:
        arm_actions = [clamp_piper(row) for row in load_actions(args.arm_file, 7)]
    else:
        arm_actions = default_grasp_trajectory(args.arm_steps)

    if not 1 <= len(dog_actions) <= MAX_TOTAL_ROWS:
        raise SystemExit("dog actions length must be 1..%d, got %d"
                         % (MAX_TOTAL_ROWS, len(dog_actions)))
    if not 1 <= len(arm_actions) <= MAX_TOTAL_ROWS:
        raise SystemExit("arm actions length must be 1..%d, got %d"
                         % (MAX_TOTAL_ROWS, len(arm_actions)))

    if args.record is not None and (args.dump_dog or args.dump_arm):
        raise SystemExit("--record cannot be combined with --dump-dog/--dump-arm")

    if args.dump_dog:
        with open(args.dump_dog, "w") as handle:
            json.dump({"actions": dog_actions}, handle,
                      ensure_ascii=False, separators=(",", ":"))
        print("wrote %d dog rows (%.1fs) to %s"
              % (len(dog_actions), chunk_duration(len(dog_actions), GO2_DT),
                 args.dump_dog), file=sys.stderr)
    if args.dump_arm:
        with open(args.dump_arm, "w") as handle:
            json.dump({"actions": arm_actions}, handle,
                      ensure_ascii=False, separators=(",", ":"))
        print("wrote %d arm rows (%.1fs) to %s"
              % (len(arm_actions), chunk_duration(len(arm_actions), PIPER_DT),
                 args.dump_arm), file=sys.stderr)
    if args.dump_dog or args.dump_arm:
        return

    if args.ensure_control or args.record is not None:
        if not args.skip_dog:
            if args.together:
                print("restarting Go2 control stack for clean replay", file=sys.stderr)
                robotctl("system.go2_control_stop")
                wait(1.0, "go2 stack stop")
            print("ensuring Go2 control stack", file=sys.stderr)
            robotctl("system.go2_control_start")
        if not args.skip_arm and not args.arm_basic:
            if args.together:
                print("restarting Piper chunk executor for clean replay", file=sys.stderr)
                robotctl("system.control_stop")
                wait(1.0, "piper chunk stop")
            print("ensuring Piper chunk executor", file=sys.stderr)
            robotctl("system.control_start")
            wait(2.5 if args.together else 0.3, "piper chunk")

    episode_id = None
    if args.record is not None:
        episode_id = args.record.strip() or time.strftime("demo_%Y%m%d_%H%M%S")
        print("recording episode %s" % episode_id, file=sys.stderr)
        robotctl("system.record_start", {
            "episode_id": episode_id,
            "instruction": args.record_instruction,
        })
        wait(0.4, "chunk loops attach to episode")

    try:
        _run_demo(args, dog_actions, arm_actions, dog_pose_events, dog_before)
    finally:
        if episode_id is not None:
            print("stopping recording %s" % episode_id, file=sys.stderr)
            robotctl("system.record_stop")
            robotctl("system.record_status")

    print("demo done", file=sys.stderr)


def _run_demo(args, dog_actions, arm_actions, dog_pose_events=None, dog_before=None):
    together = bool(args.together)
    lie_down = bool(args.stand_down) and not together
    dog_pose_events = dog_pose_events or []
    if together and args.stand_down:
        print("together replay: skipping lie-down so dog and arm stay in sync",
              file=sys.stderr)

    if not args.skip_dog:
        if args.stand_up:
            print("phase 0: stand up via go2.action_chunk before=stand_up",
                  file=sys.stderr)
            baseline = go2_status_output()
            send_go2_chunk([[0.0, 0.0, 0.0]], before="stand_up")
            ok, log = wait_go2_log(
                STAND_UP_DONE + STAND_UP_STARTED,
                STAND_UP_HOLD_S + 4.0,
                "stand-up animation",
                since=baseline,
            )
            if not ok:
                raise SystemExit(CHUNK_LOOP_HINT)
            fresh = new_log_text(log, baseline)
            if not log_has_any(fresh, STAND_UP_DONE):
                wait(STAND_UP_HOLD_S + POST_STAND_UP_SETTLE_S, "stand-up hold + settle")

        dog_s = chunk_duration(len(dog_actions), GO2_DT)
        dog_s += pose_hold_duration(dog_pose_events, before=dog_before)
        after = "stand_down" if lie_down else None
        if together:
            print("together: go2.action_chunk %d rows (%.1fs) overlapping arm"
                  % (len(dog_actions), dog_s), file=sys.stderr)
        elif lie_down:
            print("phase 1-3: go2.action_chunk %d rows (%.1fs) then after=stand_down"
                  % (len(dog_actions), dog_s), file=sys.stderr)
            print("arm waits until walk finishes and lie-down hold completes",
                  file=sys.stderr)
        else:
            print("phase 1-3: go2.action_chunk %d rows (%.1fs), no lie-down"
                  % (len(dog_actions), dog_s), file=sys.stderr)

        if together and not args.skip_arm and not args.arm_basic:
            print("ensuring Piper chunk executor before overlapping replay",
                  file=sys.stderr)
            healthy, status = piper_executor_healthy()
            if not healthy:
                print(status, file=sys.stderr)
                print(ARM_HINT, file=sys.stderr)
                raise SystemExit("together replay needs a healthy piper chunk executor")
            errors = []

            def _dog():
                try:
                    send_go2_chunks(
                        dog_actions, before=dog_before, after=after,
                        pose_events=dog_pose_events)
                    if errors:
                        return
                    wait(dog_s + 0.3, "dog trajectory")
                except Exception as exc:
                    errors.append(exc)

            def _arm():
                try:
                    send_arm_chunks(arm_actions)
                except Exception as exc:
                    errors.append(exc)

            dog_thread = threading.Thread(target=_dog)
            arm_thread = threading.Thread(target=_arm)
            dog_thread.start()
            arm_thread.start()
            dog_thread.join()
            arm_thread.join()
            if errors:
                raise SystemExit(str(errors[0]))
            return

        baseline = go2_status_output()
        t_send = time.monotonic()
        send_go2_chunks(
            dog_actions, before=dog_before, after=after,
            pose_events=dog_pose_events)
        if lie_down:
            timeout = max(40.0, dog_s * 4.0 + STAND_DOWN_HOLD_S + args.stand_down_wait)
            ok, log = wait_go2_log(
                LIE_DOWN_DONE + LIE_DOWN_STARTED,
                timeout,
                "dog chunk + lie down",
                since=baseline,
            )
            fresh = new_log_text(log, baseline)
            elapsed = time.monotonic() - t_send
            if log_has_any(fresh, LIE_DOWN_DONE):
                settle = args.stand_down_wait
                if settle > 0:
                    wait(settle, "settle after lie-down")
            elif log_has_any(fresh, LIE_DOWN_STARTED):
                wait(STAND_DOWN_HOLD_S + args.stand_down_wait,
                     "lie-down animation + settle")
            else:
                remaining = max(
                    0.0,
                    dog_s + STAND_DOWN_HOLD_S + args.stand_down_wait - elapsed)
                print("warning: lie-down not visible in log tail; waiting %.1fs more"
                      % remaining, file=sys.stderr)
                if remaining > 0:
                    wait(remaining, "walk + lie-down fallback")
        else:
            wait(dog_s + 0.3, "dog trajectory")

    if not args.skip_arm:
        if args.arm_basic:
            run_arm_basic()
        elif not run_arm_chunk(arm_actions):
            print(ARM_HINT, file=sys.stderr)
            print("falling back to piper.go_zero / gripper so the arm still moves",
                  file=sys.stderr)
            run_arm_basic()


if __name__ == "__main__":
    main()
