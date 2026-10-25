#!/usr/bin/env python3
"""Record a drag-teach episode (dog joystick + Piper drag-teach).

On the Jetson this starts ``hetero_teach_loop.py``: left stick → Go2
velocity bridge; Piper is enabled once then ``MotionCtrl_1`` drag-teach
opens the joint brakes (standby/ctrl_mode=0 otherwise holds).  Both
streams share the active episode clock.  ``piper_replay.json`` is built
from measured ``piper_state.csv``, not from a later chunk replay.

Typical session (remote OFF until the bridge holds the lease):

    python3 data_collect/scripts/record_teach.py --ensure-control --stand-up --record teach_001
    # turn the remote ON, drag the arm (press the Piper teach button if
    # it is still stiff), walk
    # press Enter (or wait --seconds) to stop

Replay (after copying JSON back to A800; exit teach on the arm first):

    python3 scripts/send_hetero_demo.py --ensure-control --together --no-stand-down \\
      --dog-file go2_replay.json --arm-file piper_replay.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = str(Path(__file__).resolve().parents[2])
AGENT_HINT = """
机载 agent 还不认识 system.teach_start（A800 白名单已经更新，狗上的进程没有）。

先关掉遥控器，然后在 Unitree 上：

  bash ~/heterovla/scripts/pull_control_on_go2.sh
  bash ~/heterovla-control/control_up.sh

再在 A800 重启 broker 后重跑本脚本：

  bash scripts/control_server_ctl.sh restart

站起来之后再开遥控器，再拖机械臂（还发硬就按示教键）。
""".strip()
TEACH_DEAD_HINT = """
示教采集进程立刻退出了，不会写出 go2_replay.json / piper_replay.json。

在 Unitree 上看 ~/heterovla-control/teach.log。常见原因：
  - 缺 hetero_teach_loop.py：再跑 pull_control_on_go2.sh
  - Piper CAN 不健康
  - 没有 piper_sdk（应用系统 Python，和 piper.go_zero 相同）

换一个新 episode id 重录（已有目录不能覆盖，例如 teach_002）。
""".strip()


def robotctl(command, data=None, quiet=False):
    argv = [os.path.join(REPO_ROOT, "inference", "scripts", "robotctl.sh"), command]
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
    if completed.stderr and not quiet:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        if completed.stdout:
            sys.stdout.write(completed.stdout)
        raise SystemExit(completed.stdout.strip() or str(completed.returncode))
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {"ok": True, "output": completed.stdout}
    if not quiet:
        print(result.get("output", completed.stdout), file=sys.stderr)
    return result


def wait(seconds, label):
    print("waiting %.1fs (%s)" % (seconds, label), file=sys.stderr)
    time.sleep(max(0.0, float(seconds)))


def wait_enter_or_seconds(seconds):
    if seconds is not None and seconds > 0:
        print("recording for %.1fs; Ctrl-C stops early" % seconds, file=sys.stderr)
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                time.sleep(min(0.5, deadline - time.monotonic()))
        except KeyboardInterrupt:
            print("stopped by Ctrl-C", file=sys.stderr)
        return
    if not sys.stdin.isatty():
        raise SystemExit("stdin is not a TTY; pass --seconds N")
    print(
        "left stick = dog vx/yaw; select = dog STOP; "
        "hold L2, tap A = toggle stand_down/stand_up; "
        "START = stand_up while prone. "
        "Right stick is ignored. Drag the arm (press the Piper teach button "
        "if it is still stiff). Walk the dog with the left stick.",
        file=sys.stderr,
    )
    print(
        "turn the remote ON now (bridge must already hold the lease)",
        file=sys.stderr,
    )
    try:
        input("press Enter to stop recording and teach capture... ")
    except EOFError:
        raise SystemExit("EOF on stdin; pass --seconds N")
    except KeyboardInterrupt:
        print("stopped by Ctrl-C", file=sys.stderr)


def wait_bridge_gone(timeout=3.0):
    """Wait until the velocity bridge pid is gone so the remote can squat."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = robotctl("system.go2_control_status", quiet=True)
        text = status.get("output") or ""
        first = text.splitlines()[0] if text.strip() else ""
        if "bridge_pid=none" in first:
            print("Go2 lease released; remote can stand_down", file=sys.stderr)
            return True
        time.sleep(0.2)
    print(
        "Go2 bridge still listed after %.1fs; open the remote and wait a few "
        "seconds for the sport lease to drop"
        % timeout,
        file=sys.stderr,
    )
    return False


def release_go2_lease(stand_down_first=True):
    """Stop the velocity bridge so the Unitree remote can stand_down."""
    if stand_down_first:
        status = robotctl("system.go2_control_status", quiet=True)
        first = (status.get("output") or "").splitlines()
        first = first[0] if first else ""
        if first.startswith("bridge_pid=") and "bridge_pid=none" not in first:
            print("lying dog down via bridge before releasing lease", file=sys.stderr)
            try:
                robotctl("go2.stand_down")
                wait(3.5, "stand-down animation")
            except SystemExit as exc:
                print("go2.stand_down: %s" % exc, file=sys.stderr)
    print(
        "releasing Go2 lease (bridge STOP was blocking stand_down / remote)",
        file=sys.stderr,
    )
    try:
        robotctl("system.go2_control_stop")
    except SystemExit as exc:
        print("go2_control_stop: %s" % exc, file=sys.stderr)
    wait_bridge_gone()


def status_pid(status):
    text = status.get("output") or ""
    first = text.splitlines()[0] if text.strip() else ""
    if first.startswith("pid=") and first != "pid=none":
        return first[4:]
    return None


POSE_BRIDGE_HINT = """
狗上的 hetero_teleop_loop.py 还是旧版，录制时 L2+A 站/趴不会转发到速度桥。

在 Unitree 狗上执行（A800 改完不会自动同步）：

  bash ~/heterovla-control/pull_control_on_go2.sh
  bash ~/heterovla-control/control_up.sh

然后在 A800 重启 Go2 控制栈后重录：

  bash scripts/robotctl.sh system.go2_control_stop
  bash scripts/robotctl.sh system.go2_control_start
""".strip()


def teach_has_pose_bridge(status):
    text = status.get("output") or ""
    return "pose-bridge-v2" in text or "pose-bridge-v3" in text or "hold L2 + tap A" in text


def hint_for_teach_failure(text):
    blob = text or ""
    if "not supported" in blob or "not in the allowlist" in blob:
        return AGENT_HINT
    return TEACH_DEAD_HINT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure-control", action="store_true",
                        help="start the Go2 velocity bridge before teach capture")
    parser.add_argument("--stand-up", action="store_true",
                        help="stand via the bridge before teach capture")
    parser.add_argument(
        "--record", nargs="?", const="", metavar="EPISODE_ID",
        help="episode id on the Jetson (default teach_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--record-instruction", default="drag teach + joystick dog")
    parser.add_argument("--seconds", type=float,
                        help="stop after N seconds instead of waiting for Enter")
    parser.add_argument("--no-teach-stop", action="store_true",
                        help="leave teach capture running after record_stop")
    parser.add_argument(
        "--keep-bridge", action="store_true",
        help="leave the Go2 velocity bridge running (remote cannot lie down "
             "while the bridge holds the sport lease)",
    )
    parser.add_argument(
        "--no-stand-down-on-exit", action="store_true",
        help="do not send stand_down before releasing the Go2 bridge",
    )
    args = parser.parse_args()
    stop_output = ""
    episode_id = None
    try:
        if args.ensure_control:
            print("ensuring Go2 control stack (bridge + chunk)", file=sys.stderr)
            robotctl("system.go2_control_start")
            wait(0.4, "go2 bridge")

        if args.stand_up:
            print("standing up via go2.action_chunk", file=sys.stderr)
            try:
                robotctl("go2.action_chunk", {
                    "actions": [[0.0, 0.0, 0.0]],
                    "before": "stand_up",
                })
                wait(3.5, "stand-up animation")
            except SystemExit:
                print("go2 chunk not running; trying robotctl go2.stand_up",
                      file=sys.stderr)
                robotctl("go2.stand_up")
                wait(3.5, "stand-up animation")

        print(
            "starting teach capture (stops piper/go2 chunk writers; keeps the bridge)",
            file=sys.stderr,
        )
        try:
            robotctl("system.teach_start")
        except SystemExit as exc:
            print(hint_for_teach_failure(str(exc)), file=sys.stderr)
            raise
        wait(0.8, "teach attach")
        status = robotctl("system.teach_status")
        if not status_pid(status):
            print(hint_for_teach_failure(status.get("output")), file=sys.stderr)
            raise SystemExit("teach capture is not running; not starting a recorder")
        if not teach_has_pose_bridge(status):
            print(POSE_BRIDGE_HINT, file=sys.stderr)

        if args.record is not None:
            episode_id = args.record.strip() or time.strftime("teach_%Y%m%d_%H%M%S")
            print("recording episode %s" % episode_id, file=sys.stderr)
            try:
                robotctl("system.record_start", {
                    "episode_id": episode_id,
                    "instruction": args.record_instruction,
                })
            except SystemExit:
                print(
                    "record_start failed; stopping teach capture. "
                    "Episode id already exists — use a new one "
                    "(teach_002 and teach_003 are taken → teach_004).",
                    file=sys.stderr,
                )
                robotctl("system.teach_stop")
                raise
            wait(0.4, "loggers attach to episode")
            robotctl("system.record_status")

        try:
            wait_enter_or_seconds(args.seconds)
        finally:
            if not args.no_teach_stop:
                print("stopping teach capture", file=sys.stderr)
                robotctl("system.teach_stop")
                wait(0.6, "teach flush replay json")
            if episode_id is not None:
                print("stopping recording %s" % episode_id, file=sys.stderr)
                try:
                    stopped = robotctl("system.record_stop")
                    stop_output = stopped.get("output") or ""
                    robotctl("system.record_status")
                except SystemExit as exc:
                    print("record_stop: %s" % exc, file=sys.stderr)
    finally:
        if not args.keep_bridge:
            release_go2_lease(stand_down_first=not args.no_stand_down_on_exit)

    print("done", file=sys.stderr)
    if episode_id:
        has_replay = (
            "go2_replay.json" in stop_output
            and "piper_replay.json" in stop_output
        )
        if not has_replay:
            print(
                "episode %s has no go2_replay.json / piper_replay.json "
                "(teach capture did not log). Check ~/heterovla-control/teach.log "
                "on the Jetson; do not scp empty paths."
                % episode_id,
                file=sys.stderr,
            )
            raise SystemExit(1)
        print("""
Press the Piper teach button again to EXIT teach (ctrl_mode must be 1)
before replay, or the arm will ignore JointCtrl.

On Unitree, copy replay JSON to A800:

  scp -F ~/heterovla-control/ssh_config \\
    ~/heterovla-data/raw/%(id)s/go2_replay.json \\
    ~/heterovla-data/raw/%(id)s/piper_replay.json \\
    heterovla-a800:~/heterovla/

On A800, replay dog and arm together (remote OFF):

  python3 scripts/send_hetero_demo.py --ensure-control --together --no-stand-down \\
    --dog-file go2_replay.json --arm-file piper_replay.json
""" % {"id": episode_id}, file=sys.stderr)


if __name__ == "__main__":
    main()
