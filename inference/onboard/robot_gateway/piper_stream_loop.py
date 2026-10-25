#!/usr/bin/env python3
"""Stream an arm action trajectory from a (mock) policy server and execute it.

OpenPI-compatible client on the robot side.  Each infer() call returns an
action chunk (H, 7) in [j1..j6 deg, gripper mm]; this loop converts to Piper
SDK units and executes one step every dt seconds.

Every frame re-issues ModeCtrl(0x01, 0x01, 20, 0x00): the gripper (and the
arm controller pipeline) only honors position commands while the arm is in
motion mode 0x01.

On exit the arm is left ENABLED holding its last position; the operator
decides when to disable (support the arm when disabling).
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

import numpy as np
from piper_sdk import C_PiperInterface_V2
from openpi_client import websocket_client_policy

JOINT_LIMITS_DEG = np.array([
    [-150.0, 150.0],
    [0.0, 180.0],
    [-170.0, 0.0],
    [-100.0, 100.0],
    [-70.0, 70.0],
    [-120.0, 120.0],
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--prompt", default="scripted arm motion")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="seconds per action step; joint+gripper streaming needs 200 Hz")
    parser.add_argument("--mode-speed", type=int, default=100,
                        help="ModeCtrl speed percent; low speeds fail to track changing targets")
    parser.add_argument("--speed-scale", type=float, default=0.5,
                        help="scales the model action before execution (safety)")
    parser.add_argument("--can", default="can0")
    return parser.parse_args()


def run(args):
    piper = C_PiperInterface_V2(args.can, start_sdk_joint_limit=True,
                                start_sdk_gripper_limit=True)
    piper.ConnectPort()
    time.sleep(0.5)
    if not piper.isOk():
        raise RuntimeError("Piper CAN receive thread is not healthy")

    stop_requested = False

    def graceful_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        logging.warning("stopping trajectory; arm keeps last pose (physical e-stop for emergencies)")

    signal.signal(signal.SIGTERM, graceful_stop)
    signal.signal(signal.SIGINT, graceful_stop)

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("server metadata: %s", policy.get_server_metadata())

    # State-aware re-initialization.  The controller's enable bits and
    # ctrl_mode can be stale after crashes, so decide on motor enable bits
    # first:
    # - motors off                    -> enable (known-good recovery path)
    # - motors on + CAN_CTRL          -> stream directly (enable-once)
    # - motors on + not CAN_CTRL      -> stale zombie; cycle disable->enable
    motors_on = all(
        getattr(piper.GetArmLowSpdInfoMsgs(), "motor_%d" % i).foc_status.driver_enable_status
        for i in range(1, 7))
    ctrl = piper.GetArmStatus().arm_status.ctrl_mode
    if motors_on:
        logging.info("already enabled (ctrl=%s); skip re-enable", ctrl)
        piper.ModeCtrl(0x01, 0x01, args.mode_speed, 0x00)
    else:
        logging.info("motors off; enabling")
        piper.EmergencyStop(0x02)
        time.sleep(0.1)
        while not piper.EnablePiper():
            time.sleep(0.01)
        piper.ModeCtrl(0x01, 0x01, args.mode_speed, 0x00)
    logging.info("starting streaming")

    observation = {
        "prompt": args.prompt,
        "observation/state": np.zeros(7, dtype=np.float32),
    }
    response = policy.infer(observation)
    actions = np.asarray(response["actions"], dtype=np.float32).reshape(-1, 7).copy()
    actions[:, :6] *= args.speed_scale
    logging.info("received action chunk: shape=%s prompt=%r",
                 actions.shape, response.get("echo_prompt"))

    start = time.monotonic()
    first_feedback = None
    for i, row in enumerate(actions):
        if stop_requested:
            logging.info("trajectory stopped by Ctrl-C at step %d", i + 1)
            break
        target = np.clip(row[:6], JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])
        joints_ctl = np.round(target * 1000).astype(int).tolist()
        gripper_ctl = int(round(np.clip(row[6], 0.5, 65.0) * 1000))
        piper.ModeCtrl(0x01, 0x01, args.mode_speed, 0x00)
        piper.JointCtrl(*joints_ctl)
        piper.GripperCtrl(gripper_ctl, 3000, 0x01, 0)
        if i == 0:
            state = piper.GetArmJointMsgs().joint_state
            first_feedback = [state.joint_1, state.joint_2, state.joint_3,
                              state.joint_4, state.joint_5, state.joint_6]
        if i == 30 and first_feedback is not None:
            state = piper.GetArmJointMsgs().joint_state
            now = [state.joint_1, state.joint_2, state.joint_3,
                   state.joint_4, state.joint_5, state.joint_6]
            moved = max(abs(a - b) for a, b in zip(first_feedback, now))
            target_moved = max(abs(a - b) for a, b in zip(first_feedback, joints_ctl))
            if moved < 200 and target_moved > 1000:
                raise RuntimeError(
                    "arm not responding (joints static after 30 steps, max delta %d "
                    "while target moved %d); aborting - check controller state"
                    % (moved, target_moved))
        if i % 10 == 0 or i == len(actions) - 1:
            state = piper.GetArmJointMsgs().joint_state
            feedback = [state.joint_1, state.joint_2, state.joint_3,
                        state.joint_4, state.joint_5, state.joint_6]
            logging.info("step %d/%d target=%s feedback=%s",
                         i + 1, len(actions), joints_ctl, feedback)
        next_time = start + (i + 1) * args.dt
        sleep_for = next_time - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            logging.warning("step %d overran by %.0f ms", i + 1, -sleep_for * 1000)

    logging.info("trajectory done; arm stays enabled holding last pose (physical e-stop for emergencies)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(parse_args())
    except KeyboardInterrupt:
        logging.warning("interrupted; arm left ENABLED holding position")
        sys.exit(130)


if __name__ == "__main__":
    main()
