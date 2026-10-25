#!/usr/bin/env python3
"""Persistent Piper chunk executor.

Started once via system.control_start; waits on a unix socket for JSON
action chunks ({"actions": [[j1..j6 deg, gripper mm], ...]}) and executes
each chunk at 200 Hz.  Between chunks it keeps streaming the last pose so
the arm holds.  SIGTERM exits gracefully (arm keeps its pose).

This mirrors the real VLA loop: the policy emits one action chunk per
inference; here each robotctl piper.action_chunk is one inference output.
"""

import argparse
import json
import logging
import math
import os
import signal
import socket
import sys
import time

from piper_sdk import C_PiperInterface_V2

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
_CONTROL = os.path.normpath(os.path.join(
    _HERE, "..", "unitree_go2", "control"))
if os.path.isdir(_CONTROL):
    sys.path.insert(0, _CONTROL)
from episode_log import DEFAULT_POINTER, EpisodeLogger  # noqa: E402

JOINT_LIMITS_DEG = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--socket", default=os.path.expanduser(
        "~/heterovla-control/chunk.sock"))
    parser.add_argument("--dt", type=float, default=0.005,
                        help="seconds per action step (200 Hz default)")
    parser.add_argument("--mode-speed", type=int, default=100)
    parser.add_argument(
        "--episode-pointer", default=DEFAULT_POINTER,
        help="path written by go2_capture_ctl / system.record_start",
    )
    return parser.parse_args()


def send_row(piper, row, args):
    joints_ctl = []
    for i, (lo, hi) in enumerate(JOINT_LIMITS_DEG):
        value = max(lo, min(hi, float(row[i])))
        joints_ctl.append(int(round(value * 1000)))
    gripper_ctl = int(round(max(0.5, min(65.0, float(row[6]))) * 1000))
    piper.ModeCtrl(0x01, 0x01, args.mode_speed, 0x00)
    piper.JointCtrl(*joints_ctl)
    piper.GripperCtrl(gripper_ctl, 3000, 0x01, 0)
    return joints_ctl, gripper_ctl


def motor_enable_flags(piper):
    info = piper.GetArmLowSpdInfoMsgs()
    return [
        bool(getattr(info, "motor_%d" % i).foc_status.driver_enable_status)
        for i in range(1, 7)
    ]


def ensure_enabled(piper, speed=100):
    flags = motor_enable_flags(piper)
    ctrl = int(getattr(piper.GetArmStatus().arm_status, "ctrl_mode", -1))
    if ctrl in (2, 0x02):
        logging.info("exiting teach mode (ctrl=%s) before chunk replay", ctrl)
        piper.MotionCtrl_1(0x00, 0x00, 0x02)
        time.sleep(0.2)
        ctrl = int(getattr(piper.GetArmStatus().arm_status, "ctrl_mode", -1))
    if all(flags) and ctrl in (1, 0x01):
        piper.ModeCtrl(0x01, 0x01, speed, 0x00)
        logging.info("already enabled (ctrl=%s); skip re-enable", ctrl)
        return
    logging.info("motors off %s ctrl=%s; enabling", flags, ctrl)
    piper.EmergencyStop(0x02)
    time.sleep(0.1)
    while not piper.EnablePiper():
        time.sleep(0.01)
    piper.ModeCtrl(0x01, 0x01, speed, 0x00)


def read_pose_deg(piper):
    state = piper.GetArmJointMsgs().joint_state
    joints = [
        state.joint_1, state.joint_2, state.joint_3,
        state.joint_4, state.joint_5, state.joint_6,
    ]
    grip = piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0
    return [v / 1000.0 for v in joints] + [max(0.5, grip)]


def blend_rows(start, end, steps):
    rows = []
    for s in range(steps):
        t = (s + 1) / float(steps)
        eased = 0.5 - 0.5 * math.cos(math.pi * t)
        rows.append([start[j] + (end[j] - start[j]) * eased for j in range(7)])
    return rows


def approach_then(current, rows, dt):
    """Fly to the first row's joints while keeping the current gripper.

    Opening the gripper is deferred to the chunk itself (after the arm has
    arrived).  A jump from the live pose to a far first waypoint with gripper
    50 mm is what made the arm drop and then look like the gripper opened.
    """
    first = rows[0]
    joint_err = max(abs(current[i] - first[i]) for i in range(6))
    if joint_err < 2.0:
        return rows
    fly_to = first[:6] + [current[6]]
    steps = min(400, max(80, int(round(joint_err / max(dt * 40.0, 0.05)))))
    logging.info("blending %d steps from current pose (joint err %.1f deg), gripper held at %.1f mm",
                 steps, joint_err, current[6])
    return blend_rows(current, fly_to, steps) + rows


def _piper_logger(pointer_path):
    fields = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper_mm")
    return EpisodeLogger(
        pointer_path,
        cmd_filename="piper_cmd.csv",
        cmd_fields=fields,
        state_filename="piper_state.csv",
        state_fields=fields,
        replay_filename="piper_replay.json",
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logger = _piper_logger(args.episode_pointer)

    piper = C_PiperInterface_V2(args.can, start_sdk_joint_limit=True,
                                start_sdk_gripper_limit=True)
    piper.ConnectPort()
    time.sleep(0.5)
    if not piper.isOk():
        raise RuntimeError("Piper CAN receive thread is not healthy")

    ensure_enabled(piper, speed=args.mode_speed)
    current = read_pose_deg(piper)
    hold_ctl, hold_grip = send_row(piper, current, args)
    logging.info("holding current pose %s", [round(v, 2) for v in current])

    def release(signum, frame):
        logging.warning("signal %d: exiting; arm holds last pose", signum)
        logger.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, release)
    signal.signal(signal.SIGINT, release)

    sock_path = args.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(4)
    server.settimeout(0.01)
    logging.info("listening on %s", sock_path)
    logging.info("chunk executor ready; episode pointer %s", args.episode_pointer)
    while True:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            conn = None
        if conn is not None:
            raw = b""
            while True:
                part = conn.recv(1 << 20)
                if not part:
                    break
                raw += part
            conn.close()
            try:
                msg = json.loads(raw.decode())
                raw_rows = msg["actions"]
                rows = []
                for item in raw_rows:
                    if not isinstance(item, (list, tuple)) or len(item) != 7:
                        raise ValueError("each action row must have 7 numbers")
                    rows.append([float(v) for v in item])
                if not rows:
                    raise ValueError("empty actions")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                logging.error("bad chunk: %s", exc)
                continue
            logging.info("chunk received: %d rows", len(rows))
            current = read_pose_deg(piper)
            rows = approach_then(current, rows, args.dt)
            t0 = time.monotonic()
            for idx, row in enumerate(rows):
                hold_ctl, hold_grip = send_row(piper, row, args)
                logger.log(row, read_pose_deg(piper))
                next_time = t0 + (idx + 1) * args.dt
                sleep_for = next_time - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    logging.warning("chunk overran by %.0f ms at row %d",
                                    -sleep_for * 1000, idx)
            logging.info("chunk done (%d rows); holding last pose", len(rows))
            logger.flush()
        if hold_ctl is not None:
            piper.ModeCtrl(0x01, 0x01, args.mode_speed, 0x00)
            piper.JointCtrl(*hold_ctl)
            piper.GripperCtrl(hold_grip, 3000, 0x01, 0)
            hold_row = [
                hold_ctl[0] / 1000.0, hold_ctl[1] / 1000.0, hold_ctl[2] / 1000.0,
                hold_ctl[3] / 1000.0, hold_ctl[4] / 1000.0, hold_ctl[5] / 1000.0,
                hold_grip / 1000.0,
            ]
            logger.log(hold_row, read_pose_deg(piper))
        time.sleep(args.dt)


if __name__ == "__main__":
    main()
