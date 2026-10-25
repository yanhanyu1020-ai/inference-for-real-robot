#!/usr/bin/env python3
"""Record a joystick teleop episode, then print how to replay it.

On the Jetson this starts ``hetero_teleop_loop.py`` (Unitree remote → dog
velocity + Piper joints) and attaches both loggers to the active episode.
On A800, use this script via robotctl.

Typical session (remote OFF until the bridge holds the lease):

    python3 data_collect/scripts/record_teleop.py --ensure-control --stand-up --record teleop_001
    # then turn the remote ON and drive
    # press Enter (or wait --seconds) to stop

Replay (after copying JSON back to A800):

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
机载 agent 还不认识 system.teleop_start（A800 白名单已经更新，狗上的进程没有）。
遥控器现在能直接赶狗，是固件占了租约，不是摇操记录在跑，所以不会写出
go2_replay.json / piper_replay.json。

先关掉遥控器，然后在 Unitree 上：

  bash ~/heterovla/scripts/pull_control_on_go2.sh
  bash ~/heterovla-control/control_up.sh

再在 A800 重跑本脚本。站起来之后再开遥控器。
""".strip()
TELEOP_DEAD_HINT = """
摇操进程立刻退出了，DDS 遥测目录里不会有 go2_replay.json / piper_replay.json。

在 Unitree 上看 ~/heterovla-control/teleop.log。若是 No module named
'unitree_sdk2py'：更新后的 hetero_teleop_loop 会改读 episode 里的
stick.json / wireless_controller.csv，先：

  bash ~/heterovla/scripts/pull_control_on_go2.sh
  bash ~/heterovla-control/control_up.sh

再换一个新 episode id 重录（已有目录不能覆盖，例如 teleop_002）。
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
    print("remote mapping: left stick = dog vx/yaw; right stick = arm J1/J2; "
          "L1=J3/J4 L2=J5/J6 R1=open R2=close select=dog STOP; "
          "L2+A=toggle stand_down/stand_up START=stand_up (while prone)",
          file=sys.stderr)
    print("turn the remote ON now (bridge must already hold the lease), then drive",
          file=sys.stderr)
    try:
        input("press Enter to stop recording and teleop... ")
    except EOFError:
        raise SystemExit("EOF on stdin; pass --seconds N")
    except KeyboardInterrupt:
        print("stopped by Ctrl-C", file=sys.stderr)


def teleop_pid(status):
    text = status.get("output") or ""
    first = text.splitlines()[0] if text.strip() else ""
    if first.startswith("pid=") and first != "pid=none":
        return first[4:]
    return None


def hint_for_teleop_failure(text):
    blob = text or ""
    if "not supported" in blob:
        return AGENT_HINT
    return TELEOP_DEAD_HINT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure-control", action="store_true",
                        help="start the Go2 velocity bridge before teleop")
    parser.add_argument("--stand-up", action="store_true",
                        help="stand via the bridge before teleop")
    parser.add_argument(
        "--record", nargs="?", const="", metavar="EPISODE_ID",
        help="episode id on the Jetson (default teleop_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument("--record-instruction", default="joystick teleop")
    parser.add_argument("--seconds", type=float,
                        help="stop after N seconds instead of waiting for Enter")
    parser.add_argument("--no-teleop-stop", action="store_true",
                        help="leave teleop running after record_stop")
    args = parser.parse_args()

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

    print("starting teleop (stops piper/go2 chunk writers; keeps the bridge)",
          file=sys.stderr)
    try:
        robotctl("system.teleop_start")
    except SystemExit as exc:
        print(hint_for_teleop_failure(str(exc)), file=sys.stderr)
        raise
    wait(0.8, "teleop attach")
    status = robotctl("system.teleop_status")
    if not teleop_pid(status):
        print(hint_for_teleop_failure(status.get("output")), file=sys.stderr)
        raise SystemExit("teleop is not running; not starting a recorder")

    episode_id = None
    if args.record is not None:
        episode_id = args.record.strip() or time.strftime("teleop_%Y%m%d_%H%M%S")
        print("recording episode %s" % episode_id, file=sys.stderr)
        robotctl("system.record_start", {
            "episode_id": episode_id,
            "instruction": args.record_instruction,
        })
        wait(0.4, "loggers attach to episode")
        robotctl("system.record_status")

    stop_output = ""
    try:
        wait_enter_or_seconds(args.seconds)
    finally:
        if not args.no_teleop_stop:
            print("stopping teleop", file=sys.stderr)
            robotctl("system.teleop_stop")
            wait(0.6, "teleop flush replay json")
        if episode_id is not None:
            print("stopping recording %s" % episode_id, file=sys.stderr)
            stopped = robotctl("system.record_stop")
            stop_output = stopped.get("output") or ""
            robotctl("system.record_status")

    print("done", file=sys.stderr)
    if episode_id:
        has_replay = (
            "go2_replay.json" in stop_output
            and "piper_replay.json" in stop_output
        )
        if not has_replay:
            print(
                "episode %s has no go2_replay.json / piper_replay.json "
                "(teleop did not log). Check ~/heterovla-control/teleop.log "
                "on the Jetson; do not scp empty paths."
                % episode_id,
                file=sys.stderr,
            )
            raise SystemExit(1)
        print("""
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
