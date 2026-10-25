#!/usr/bin/env python3
"""Small allowlisted Piper command adapter.

Commands are deliberately one-shot and contain no user-provided joint values.
"""

import argparse
import time

from piper_sdk import C_PiperInterface_V2


def wait_until(action, expected, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(action()) is expected:
            return
        time.sleep(0.05)
    raise RuntimeError("Piper enable-state change timed out")


def motor_enable_flags(piper):
    info = piper.GetArmLowSpdInfoMsgs()
    return [
        bool(getattr(info, "motor_%d" % i).foc_status.driver_enable_status)
        for i in range(1, 7)
    ]


def ctrl_mode(piper):
    return piper.GetArmStatus().arm_status.ctrl_mode


def ensure_enabled(piper, speed=100):
    """Enable only when motors are off. Never EmergencyStop a live arm.

    Re-enable / EmergencyStop(0x02) on an already-holding arm drops it; the
    gripper command then runs after the fall and looks like "open after drop".
    """
    flags = motor_enable_flags(piper)
    ctrl = ctrl_mode(piper)
    if all(flags):
        piper.ModeCtrl(0x01, 0x01, speed, 0x00)
        return "already enabled (ctrl=%s); skip re-enable" % ctrl
    piper.EmergencyStop(0x02)
    time.sleep(0.1)
    wait_until(piper.EnablePiper, True)
    piper.ModeCtrl(0x01, 0x01, speed, 0x00)
    return "enabled (motors were %s, ctrl=%s)" % (flags, ctrl)


def joint_feedback(piper):
    state = piper.GetArmJointMsgs().joint_state
    return [
        state.joint_1,
        state.joint_2,
        state.joint_3,
        state.joint_4,
        state.joint_5,
        state.joint_6,
    ]


# Exact zero pose sits ON the J2 (min 0 deg) and J3 (max 0 deg) joint limits;
# commanding it triggers REACH_TARGET_POS_FAILED and the arm auto-disables and
# drops.  Use a small-offset safe home instead.
SAFE_HOME = (0, 500, -500, 0, 500, 0)


def go_zero(piper, target=SAFE_HOME, timeout=8.0, tolerance=3000):
    """Continuously command the safe home pose until feedback converges.

    Joint values and tolerance use the Piper SDK unit of 0.001 degrees.
    """
    deadline = time.monotonic() + timeout
    last_feedback = joint_feedback(piper)
    while time.monotonic() < deadline:
        piper.ModeCtrl(0x01, 0x01, 20, 0x00)
        piper.JointCtrl(*target)
        time.sleep(0.05)
        last_feedback = joint_feedback(piper)
        if max(abs(a - b) for a, b in zip(last_feedback, target)) <= tolerance:
            return last_feedback
    raise RuntimeError(
        "Piper did not reach safe home within %.1fs; feedback=%s" %
        (timeout, last_feedback)
    )


def open_gripper(piper, target=50000, timeout=5.0, tolerance=2000):
    """Open the gripper to the vendor-demo position of 50 mm.

    The gripper only responds while the arm is in motion mode 0x01;
    ModeCtrl must be re-issued every frame alongside the gripper stream.
    """
    deadline = time.monotonic() + timeout
    last_feedback = piper.GetArmGripperMsgs().gripper_state.grippers_angle
    while time.monotonic() < deadline:
        piper.ModeCtrl(0x01, 0x01, 20, 0x00)
        piper.GripperCtrl(target, 3000, 0x01, 0)
        time.sleep(0.005)
        last_feedback = piper.GetArmGripperMsgs().gripper_state.grippers_angle
        if abs(last_feedback - target) <= tolerance:
            return last_feedback
    raise RuntimeError(
        "Piper gripper did not reach 50mm within %.1fs; feedback=%d" %
        (timeout, last_feedback)
    )


def close_gripper(piper, target=1000, timeout=5.0, tolerance=2000):
    """Close the gripper to a safe closed position (1 mm).

    Never command 0 mm: the physical closed stop reads ~0.2 mm, so a 0 mm
    target makes the driver hunt against the stop (shake + noise).
    """
    deadline = time.monotonic() + timeout
    last_feedback = piper.GetArmGripperMsgs().gripper_state.grippers_angle
    while time.monotonic() < deadline:
        piper.ModeCtrl(0x01, 0x01, 20, 0x00)
        piper.GripperCtrl(target, 3000, 0x01, 0)
        time.sleep(0.005)
        last_feedback = piper.GetArmGripperMsgs().gripper_state.grippers_angle
        if abs(last_feedback - target) <= tolerance:
            return last_feedback
    raise RuntimeError(
        "Piper gripper did not close within %.1fs; feedback=%d" %
        (timeout, last_feedback)
    )


def run(can_name, command):
    piper = C_PiperInterface_V2(
        can_name, start_sdk_joint_limit=True, start_sdk_gripper_limit=True
    )
    piper.ConnectPort()
    time.sleep(0.5)
    if not piper.isOk():
        raise RuntimeError("Piper CAN receive thread is not healthy")

    if command == "enable":
        print("Piper %s" % ensure_enabled(piper))
    elif command == "disable":
        wait_until(piper.DisablePiper, False)
    elif command == "stop":
        piper.EmergencyStop(0x01)
    elif command == "go_zero":
        print("Piper %s" % ensure_enabled(piper, speed=20))
        feedback = go_zero(piper)
        print("Piper reached safe home; feedback=%s" % feedback)
    elif command == "open_gripper":
        print("Piper %s" % ensure_enabled(piper, speed=20))
        feedback = open_gripper(piper)
        print("Piper gripper opened to %.3fmm" % (feedback / 1000.0))
    elif command == "close_gripper":
        print("Piper %s" % ensure_enabled(piper, speed=20))
        feedback = close_gripper(piper)
        print("Piper gripper closed to %.3fmm" % (feedback / 1000.0))
    else:
        raise ValueError("unsupported Piper command: %s" % command)
    time.sleep(0.5)
    if command not in ("go_zero", "open_gripper", "close_gripper"):
        print("Piper command accepted: %s" % command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument(
        "command",
        choices=("enable", "disable", "stop", "go_zero", "open_gripper", "close_gripper"),
    )
    args = parser.parse_args()
    run(args.can, args.command)


if __name__ == "__main__":
    main()
