#!/usr/bin/env python3
"""Joystick teleop for Go2 + Piper, logged into the active episode.

Reads the Unitree remote on ``rt/wirelesscontroller`` (sticks still publish
when the velocity bridge already holds the sport lease).  If
``unitree_sdk2py`` is missing, falls back to ``stick.json`` /
``wireless_controller.csv`` written by the C++ recorder in the active
episode.  Left stick drives the dog through ``go2.sock``; right stick
rate-controls the arm over CAN unless ``--teach-arm`` is set.

``--teach-arm`` is the drag-teach capture path.  After a one-shot enable it
sends ``MotionCtrl_1(..., grag_teach_ctrl=0x01)`` so the joint brakes open
(standby/ctrl_mode=0 holds with brakes; a receive-only CAN open does not).
The loop then stops sending ``ModeCtrl`` / ``JointCtrl`` / ``GripperCtrl``.
The hardware teach button can still raise ``ctrl_mode==2``.  Arm CSV rows
are measured joint/gripper feedback on the same clock as the dog.  Replay
JSON is built from ``piper_state.csv``.

Both command streams are written with EpisodeLogger so ``record_stop`` can
emit ``go2_replay.json`` / ``piper_replay.json``.

Must be the only Piper CAN user: stop ``piper_chunk_loop`` / streamer
before starting this process.  Stop ``go2_chunk_loop`` too (keep the
bridge) so idle zero-velocity logs do not pollute the dog CSV.

Remote mapping (Unitree key bits):

    left ly/lx     dog vx / vyaw
    right rx/ry    arm J1 / J2          (ignored with --teach-arm)
    L1 + right     arm J3 / J4          (ignored with --teach-arm)
    L2 + right     arm J5 / J6          (ignored with --teach-arm)
    R1 / R2        gripper open / close (ignored with --teach-arm)
    select         local stop (dog STOP, arm holds / stays in teach)
    L2 hold + A    tap A again while holding L2 to toggle stand_down / stand_up
    start          stand_up while prone (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from episode_log import DEFAULT_POINTER, EpisodeLogger, read_pointer  # noqa: E402
from go2_send import send_line  # noqa: E402


LOG = logging.getLogger("heterovla.teleop")

TEACH_CTRL_MODE = 2
LINKAGE_TEACH_CTRL_MODE = 6
TEACH_STATUS_RECORDING = 1
ARM_STATUS_TEACHING_RECORD = 0x0B
ARM_STATUS_ESTOP = 0x01
ARM_STATUS_BRAKE = 0x06
CTRL_MODE_NAMES = {
    0: "standby",
    1: "can",
    2: "teach",
    3: "ethernet",
    4: "wifi",
    5: "remote",
    6: "linkage_teach",
    7: "offline_traj",
}
ARM_STATUS_NAMES = {
    0: "normal",
    1: "estop",
    2: "no_solution",
    3: "singularity",
    4: "limit",
    5: "joint_comm",
    6: "brake",
    7: "collision",
    8: "teach_overspeed",
    9: "joint_err",
    10: "other",
    11: "teach_record",
    12: "teach_exec",
    13: "teach_pause",
}
TEACH_STATUS_NAMES = {
    0: "off",
    1: "recording",
    2: "stop_record",
    3: "execute",
    4: "pause",
    5: "resume",
    6: "terminate",
    7: "to_start",
}

KEY_R1 = 0x0001
KEY_L1 = 0x0002
KEY_START = 0x0004
KEY_SELECT = 0x0008
KEY_R2 = 0x0010
KEY_L2 = 0x0020
KEY_F1 = 0x0040
KEY_F2 = 0x0080
KEY_A = 0x0100
KEY_B = 0x0200
KEY_X = 0x0400
KEY_Y = 0x0800
KEY_UP = 0x1000
KEY_RIGHT = 0x2000
KEY_DOWN = 0x4000
KEY_LEFT = 0x8000

REMOTE_STAND_DOWN = KEY_L2 | KEY_A
REMOTE_STAND_UP = KEY_START

POSE_HOLD_S = 3.5
POSE_DEBOUNCE_S = 0.8


class RemotePoseTracker(object):
    """Detect L2+A taps even when stick samples arrive slowly (CSV ~1 Hz).

    Arms on the first L2+A sample after both keys were released, then waits
    for release before accepting the next tap.
    """

    def __init__(self):
        self.dog_prone = False
        self._combo_armed = True
        self._start_armed = True

    def feed(self, keys, now, debounce_s=POSE_DEBOUNCE_S, last_cmd=0.0):
        keys = int(keys)
        combo = bool(keys & KEY_L2) and bool(keys & KEY_A)
        action = None
        if combo and self._combo_armed and now - last_cmd >= debounce_s:
            action = "stand_up" if self.dog_prone else "stand_down"
            self.dog_prone = action == "stand_down"
            self._combo_armed = False
        elif not combo:
            self._combo_armed = True

        if (
            self.dog_prone
            and bool(keys & KEY_START)
            and self._start_armed
            and now - last_cmd >= debounce_s
        ):
            action = "stand_up"
            self.dog_prone = False
            self._start_armed = False
        elif not (keys & KEY_START):
            self._start_armed = True
        return action

JOINT_LIMITS_DEG = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)
GRIPPER_RANGE_MM = (1.0, 65.0)

VX_LIMIT = 0.5
VY_LIMIT = 0.5
VYAW_LIMIT = 1.0


class StickState(object):
    __slots__ = ("lx", "ly", "rx", "ry", "keys", "ok", "source")

    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.keys = 0
        self.ok = False
        self.source = "none"

    def as_tuple(self):
        return (self.lx, self.ly, self.rx, self.ry, int(self.keys), self.ok)


def key_combo_active(keys, mask):
    return (int(keys) & int(mask)) == int(mask)


def key_combo_rising(prev_keys, keys, mask):
    return key_combo_active(keys, mask) and not key_combo_active(prev_keys, mask)


def key_bit_rising(prev_keys, keys, bit):
    bit = int(bit)
    return bool(int(keys) & bit) and not bool(int(prev_keys) & bit)


def send_bridge_pose(bridge_sock, action):
    """STOP first, then STAND_UP / STAND_DOWN so Move does not fight the pose."""
    send_line(bridge_sock, "STOP", timeout=1.0)
    line = "STAND_UP" if action == "stand_up" else "STAND_DOWN"
    send_line(bridge_sock, line, timeout=1.0)


def apply_deadzone(value, deadzone):
    value = float(value)
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    span = 1.0 - deadzone
    if span <= 0:
        return 0.0
    scaled = (abs(value) - deadzone) / span
    return sign * max(-1.0, min(1.0, scaled))


def clamp_joint(index, value):
    lo, hi = JOINT_LIMITS_DEG[index]
    return max(lo, min(hi, float(value)))


def clamp_gripper(value):
    lo, hi = GRIPPER_RANGE_MM
    return max(lo, min(hi, float(value)))


def _status_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def ctrl_mode_name(mode):
    mode = _status_int(mode)
    if mode < 0:
        return "unknown"
    return CTRL_MODE_NAMES.get(mode, "unknown_%s" % mode)


def arm_status_name(status):
    status = _status_int(status)
    if status < 0:
        return "unknown"
    return ARM_STATUS_NAMES.get(status, "unknown_%s" % status)


def teach_status_name(status):
    status = _status_int(status)
    if status < 0:
        return "unknown"
    return TEACH_STATUS_NAMES.get(status, "unknown_%s" % status)


def is_teach_ctrl_mode(mode):
    return _status_int(mode) == TEACH_CTRL_MODE


def is_drag_teach_active(ctrl_mode, teach_status=0, arm_status=0):
    """True when the arm is gravity-compensated / recording a drag."""
    if _status_int(arm_status) == ARM_STATUS_ESTOP:
        return False
    ctrl = _status_int(ctrl_mode)
    if ctrl in (TEACH_CTRL_MODE, LINKAGE_TEACH_CTRL_MODE):
        return True
    if _status_int(teach_status) == TEACH_STATUS_RECORDING:
        return True
    return _status_int(arm_status) == ARM_STATUS_TEACHING_RECORD


def clamp_dog(vx, vy, vyaw):
    return [
        max(-VX_LIMIT, min(VX_LIMIT, float(vx))),
        max(-VY_LIMIT, min(VY_LIMIT, float(vy))),
        max(-VYAW_LIMIT, min(VYAW_LIMIT, float(vyaw))),
    ]


def stick_to_dog(lx, ly, deadzone, vx_max, vyaw_max):
    """Left stick: ly → forward, lx → yaw (no strafe)."""
    fwd = apply_deadzone(ly, deadzone)
    yaw = apply_deadzone(lx, deadzone)
    return clamp_dog(fwd * vx_max, 0.0, -yaw * vyaw_max)


def integrate_arm(pose, rx, ry, keys, dt, deadzone, joint_rate, grip_rate):
    """Rate-control joints from the right stick. Returns a new 7-vector."""
    pose = [float(v) for v in pose]
    dx = apply_deadzone(rx, deadzone) * joint_rate * dt
    dy = apply_deadzone(ry, deadzone) * joint_rate * dt
    if keys & KEY_L2:
        pair = (4, 5)  # J5, J6
    elif keys & KEY_L1:
        pair = (2, 3)  # J3, J4
    else:
        pair = (0, 1)  # J1, J2
    pose[pair[0]] = clamp_joint(pair[0], pose[pair[0]] + dx)
    pose[pair[1]] = clamp_joint(pair[1], pose[pair[1]] + dy)
    if keys & KEY_R1:
        pose[6] = clamp_gripper(pose[6] + grip_rate * dt)
    if keys & KEY_R2:
        pose[6] = clamp_gripper(pose[6] - grip_rate * dt)
    return pose


def parse_args():
    here = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--bridge-sock", default=os.path.join(here, "go2.sock"))
    parser.add_argument("--episode-pointer", default=DEFAULT_POINTER)
    parser.add_argument("--dt", type=float, default=0.005, help="arm period (200 Hz)")
    parser.add_argument("--dog-hz", type=float, default=20.0)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--vx-max", type=float, default=0.35)
    parser.add_argument("--vyaw-max", type=float, default=0.7)
    parser.add_argument("--joint-rate", type=float, default=40.0,
                        help="deg/s at full stick")
    parser.add_argument("--grip-rate", type=float, default=40.0,
                        help="mm/s at full trigger")
    parser.add_argument("--mode-speed", type=int, default=100)
    parser.add_argument("--arm-only", action="store_true",
                        help="ignore left stick; dog stays stopped")
    parser.add_argument("--skip-arm", action="store_true",
                        help="do not open Piper CAN (dog-only teleop)")
    parser.add_argument(
        "--teach-arm", action="store_true",
        help="drag-teach capture: one-shot enable + MotionCtrl_1 drag-teach, "
             "then no ModeCtrl/JointCtrl; left stick still drives the dog",
    )
    parser.add_argument("--stick-file",
                        help="read {lx,ly,rx,ry,keys} JSON instead of DDS")
    parser.add_argument("--dry-run", action="store_true",
                        help="print mapped commands; do not touch hardware")
    return parser.parse_args()


def apply_stick_payload(state, payload, source):
    state.lx = float(payload.get("lx", 0.0))
    state.ly = float(payload.get("ly", 0.0))
    state.rx = float(payload.get("rx", 0.0))
    state.ry = float(payload.get("ry", 0.0))
    state.keys = int(payload.get("keys", 0))
    state.ok = True
    state.source = source
    return state


def parse_wireless_csv_tail(path):
    """Last complete wireless_controller.csv row as a stick dict, or None."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 4096), os.SEEK_SET)
            chunk = handle.read().decode("utf-8", "replace")
    except OSError:
        return None
    for raw in reversed(chunk.splitlines()):
        line = raw.strip()
        if not line or line.startswith("monotonic_ns"):
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            return {
                "lx": float(parts[3]),
                "ly": float(parts[4]),
                "rx": float(parts[5]),
                "ry": float(parts[6]),
                "keys": int(float(parts[7])),
            }
        except ValueError:
            continue
    return None


def parse_wireless_csv_since(path, last_seq):
    """Return new wireless_controller.csv rows with seq > last_seq."""
    rows = []
    try:
        with open(path) as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("monotonic_ns"):
                    continue
                parts = line.split(",")
                if len(parts) < 8:
                    continue
                try:
                    seq = int(float(parts[2]))
                except ValueError:
                    continue
                if seq <= last_seq:
                    continue
                rows.append((
                    seq,
                    {
                        "lx": float(parts[3]),
                        "ly": float(parts[4]),
                        "rx": float(parts[5]),
                        "ry": float(parts[6]),
                        "keys": int(float(parts[7])),
                    },
                ))
    except OSError:
        return rows, last_seq
    if not rows:
        return rows, last_seq
    return rows, rows[-1][0]


class StickFileReader(object):
    def __init__(self, path):
        self.path = path
        self.state = StickState()
        self.state.source = "file:%s" % path

    def start(self):
        return self

    def read(self):
        try:
            with open(self.path) as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return self.state
        return apply_stick_payload(self.state, payload, "file:%s" % self.path)

    def close(self):
        return


class EpisodeStickReader(object):
    """Read sticks from the C++ recorder in the active episode directory.

    Prefers atomic ``stick.json`` (written on every wireless sample by a
    current recorder).  Falls back to the last complete
    ``wireless_controller.csv`` row, which older binaries only flush ~1 Hz.
    """

    def __init__(self, pointer_path):
        self.pointer_path = pointer_path
        self.state = StickState()
        self.state.source = "episode"
        self._last_csv_seq = -1
        self._last_keys = -1
        self._key_updates = []

    def start(self):
        return self

    def drain_key_updates(self):
        pending = self._key_updates
        self._key_updates = []
        return pending

    def read(self):
        episode = read_pointer(self.pointer_path)
        if not episode:
            return self.state
        json_path = os.path.join(episode, "stick.json")
        try:
            with open(json_path) as handle:
                payload = json.load(handle)
            state = apply_stick_payload(self.state, payload, "json:%s" % json_path)
            if state.keys != self._last_keys:
                self._key_updates = [state.keys]
                self._last_keys = state.keys
            else:
                self._key_updates = []
            return state
        except (OSError, ValueError):
            pass
        csv_path = os.path.join(episode, "wireless_controller.csv")
        rows, self._last_csv_seq = parse_wireless_csv_since(
            csv_path, self._last_csv_seq)
        if rows:
            self._key_updates = [payload["keys"] for _, payload in rows]
            self._last_keys = self._key_updates[-1]
            latest = rows[-1][1]
            return apply_stick_payload(self.state, latest, "csv:%s" % episode)
        payload = parse_wireless_csv_tail(csv_path)
        if payload is None:
            self._key_updates = []
            return self.state
        keys = payload["keys"]
        if keys != self._last_keys:
            self._key_updates = [keys]
            self._last_keys = keys
        else:
            self._key_updates = []
        return apply_stick_payload(
            self.state, payload, "csv:%s" % episode)

    def close(self):
        return


class DdsStickReader(object):
    """Subscribe to rt/wirelesscontroller via unitree_sdk2py."""

    def __init__(self, network_interface):
        self.network_interface = network_interface
        self.state = StickState()
        self.state.source = "dds"
        self._lock = threading.Lock()
        self._last_keys = -1
        self._key_updates = []

    def start(self):
        ChannelFactoryInitialize, WirelessController_, ChannelSubscriber = (
            _import_unitree_dds())
        ChannelFactoryInitialize(0, self.network_interface)

        def _on_msg(msg):
            with self._lock:
                self.state.lx = float(getattr(msg, "lx"))
                self.state.ly = float(getattr(msg, "ly"))
                self.state.rx = float(getattr(msg, "rx"))
                self.state.ry = float(getattr(msg, "ry"))
                self.state.keys = int(getattr(msg, "keys"))
                self.state.ok = True

        sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        sub.Init(_on_msg, 16)
        self._sub = sub
        LOG.info("subscribed rt/wirelesscontroller on %s", self.network_interface)
        return self

    def drain_key_updates(self):
        pending = self._key_updates
        self._key_updates = []
        return pending

    def read(self):
        with self._lock:
            snap = StickState()
            snap.lx = self.state.lx
            snap.ly = self.state.ly
            snap.rx = self.state.rx
            snap.ry = self.state.ry
            snap.keys = self.state.keys
            snap.ok = self.state.ok
            snap.source = self.state.source
            if snap.ok and snap.keys != self._last_keys:
                self._key_updates = [snap.keys]
                self._last_keys = snap.keys
            else:
                self._key_updates = []
            return snap

    def close(self):
        return


def _import_unitree_dds():
    extra_roots = [
        os.path.expanduser("~/unitree_sdk2_python"),
        "/home/unitree/unitree_sdk2_python",
        os.path.expanduser("~/unitree_sdk2-main/python"),
        "/home/unitree/unitree_sdk2-main/python",
        os.path.expanduser("~/unitree_ros2/cyclonedds_ws/src/unitree"),
    ]
    try:
        import glob
        extra_roots.extend(glob.glob(
            "/home/unitree/.local/lib/python3.*/site-packages"))
        extra_roots.extend(glob.glob(
            "/usr/local/lib/python3.*/site-packages"))
    except Exception:
        pass
    for root in extra_roots:
        if os.path.isdir(root) and root not in sys.path:
            sys.path.append(root)
    errors = []
    try:
        from unitree_sdk2py.core.channel import (  # type: ignore
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # type: ignore
            WirelessController_,
        )
        return ChannelFactoryInitialize, WirelessController_, ChannelSubscriber
    except Exception as exc:
        errors.append("unitree_sdk2py: %s" % exc)
    raise RuntimeError(
        "cannot import Unitree DDS Python bindings (%s). "
        "Install unitree_sdk2py on the Jetson, or pass --stick-file"
        % "; ".join(errors)
    )


def _open_stick_reader(args):
    if args.stick_file:
        return StickFileReader(args.stick_file).start()
    try:
        return DdsStickReader(args.network_interface).start()
    except RuntimeError as exc:
        LOG.warning(
            "%s; falling back to episode stick.json / wireless_controller.csv "
            "(pose keys are slower on csv; update the Go2 recorder for stick.json)",
            exc,
        )
        return EpisodeStickReader(args.episode_pointer).start()


def _piper_helpers():
    from piper_sdk import C_PiperInterface_V2  # type: ignore

    def motor_enable_flags(piper):
        info = piper.GetArmLowSpdInfoMsgs()
        return [
            bool(getattr(info, "motor_%d" % i).foc_status.driver_enable_status)
            for i in range(1, 7)
        ]

    def ensure_enabled(piper, speed=100):
        flags = motor_enable_flags(piper)
        if all(flags):
            piper.ModeCtrl(0x01, 0x01, speed, 0x00)
            LOG.info("piper already enabled; skip re-enable")
            return
        LOG.info("enabling piper motors %s", flags)
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
        return [v / 1000.0 for v in joints] + [max(GRIPPER_RANGE_MM[0], grip)]

    def send_row(piper, row, speed):
        joints_ctl = []
        for i, (lo, hi) in enumerate(JOINT_LIMITS_DEG):
            value = max(lo, min(hi, float(row[i])))
            joints_ctl.append(int(round(value * 1000)))
        gripper_ctl = int(round(max(0.5, min(65.0, float(row[6]))) * 1000))
        piper.ModeCtrl(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*joints_ctl)
        piper.GripperCtrl(gripper_ctl, 3000, 0x01, 0)

    def read_ctrl_mode(piper):
        try:
            return int(piper.GetArmStatus().arm_status.ctrl_mode)
        except Exception:
            return -1

    def read_arm_feedback(piper):
        empty = {
            "ctrl_mode": -1,
            "arm_status": -1,
            "teach_status": -1,
        }
        try:
            status = piper.GetArmStatus().arm_status
        except Exception:
            return empty
        return {
            "ctrl_mode": _status_int(getattr(status, "ctrl_mode", -1)),
            "arm_status": _status_int(getattr(status, "arm_status", -1)),
            "teach_status": _status_int(getattr(status, "teach_status", -1)),
        }

    def enter_drag_teach(piper, speed=100):
        """Leave standby, then open drag-teach so the arm can be moved by hand.

        Standby (ctrl_mode=0) holds with joint brakes.  Opening CAN read-only
        does not release them, so the hardware teach button often never
        reaches ctrl_mode=2.  Enable + one ModeCtrl/JointCtrl gets the arm
        into CAN mode; MotionCtrl_1(grag_teach_ctrl=0x01) starts firmware
        drag-teach.  Do not keep streaming ModeCtrl after this returns.
        """
        feedback = read_arm_feedback(piper)
        flags = motor_enable_flags(piper)
        LOG.info(
            "enter drag-teach: ctrl=%s(%s) arm=%s(%s) teach=%s(%s) motors=%s",
            feedback["ctrl_mode"], ctrl_mode_name(feedback["ctrl_mode"]),
            feedback["arm_status"], arm_status_name(feedback["arm_status"]),
            feedback["teach_status"], teach_status_name(feedback["teach_status"]),
            flags,
        )
        if feedback["arm_status"] == ARM_STATUS_ESTOP:
            LOG.warning("piper in e-stop; sending resume before drag-teach")
            piper.EmergencyStop(0x02)
            time.sleep(0.2)
            flags = motor_enable_flags(piper)
        if not all(flags) or feedback["arm_status"] == ARM_STATUS_BRAKE:
            LOG.info("enabling piper motors %s", flags)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if piper.EnablePiper():
                    break
                time.sleep(0.01)
        pose = read_pose_deg(piper)
        joints_ctl = []
        for i, (lo, hi) in enumerate(JOINT_LIMITS_DEG):
            value = max(lo, min(hi, float(pose[i])))
            joints_ctl.append(int(round(value * 1000)))
        piper.ModeCtrl(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*joints_ctl)
        time.sleep(0.15)
        piper.MotionCtrl_1(0x00, 0x00, 0x01)
        time.sleep(0.2)
        feedback = read_arm_feedback(piper)
        LOG.info(
            "after MotionCtrl_1 drag-teach: ctrl=%s(%s) arm=%s(%s) teach=%s(%s) "
            "pose=%s",
            feedback["ctrl_mode"], ctrl_mode_name(feedback["ctrl_mode"]),
            feedback["arm_status"], arm_status_name(feedback["arm_status"]),
            feedback["teach_status"], teach_status_name(feedback["teach_status"]),
            [round(v, 2) for v in pose],
        )
        return feedback

    def exit_drag_teach(piper):
        try:
            piper.MotionCtrl_1(0x00, 0x00, 0x02)
            LOG.info("sent MotionCtrl_1 exit drag-teach")
        except Exception as exc:
            LOG.error("exit drag-teach failed: %s", exc)

    return (
        C_PiperInterface_V2, ensure_enabled, read_pose_deg, send_row,
        read_ctrl_mode, read_arm_feedback, enter_drag_teach, exit_drag_teach,
    )


def _make_loggers(pointer, replay_from_state=False):
    go2_log = EpisodeLogger(
        pointer,
        cmd_filename="go2_cmd.csv",
        cmd_fields=("vx", "vy", "vyaw"),
        replay_filename="go2_replay.json",
    )
    piper_log = EpisodeLogger(
        pointer,
        cmd_filename="piper_cmd.csv",
        cmd_fields=("j1", "j2", "j3", "j4", "j5", "j6", "gripper_mm"),
        state_filename="piper_state.csv",
        state_fields=("j1", "j2", "j3", "j4", "j5", "j6", "gripper_mm"),
        replay_filename="piper_replay.json",
        replay_from_state=replay_from_state,
    )
    return go2_log, piper_log


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    teach_arm = bool(args.teach_arm)
    if teach_arm:
        LOG.info("teach-arm capture: dog from left stick, Piper read-only")
    stop = {"flag": False}

    def _request_stop(signum, frame):
        LOG.warning("signal %d: stopping %s", signum,
                    "teach capture" if teach_arm else "teleop")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    reader = _open_stick_reader(args)
    go2_log, piper_log = _make_loggers(
        args.episode_pointer, replay_from_state=teach_arm)

    piper = None
    send_row = None
    read_pose_deg = None
    read_arm_feedback = None
    enter_drag_teach = None
    exit_drag_teach = None
    pose = [0.0, 0.5, -0.5, 0.0, 0.5, 0.0, 2.0]
    feedback = {"ctrl_mode": -1, "arm_status": -1, "teach_status": -1}
    ctrl = -1
    if not args.skip_arm and not args.dry_run:
        (C_PiperInterface_V2, ensure_enabled, read_pose_deg, send_row,
         _read_ctrl_mode, read_arm_feedback, enter_drag_teach,
         exit_drag_teach) = _piper_helpers()
        piper = C_PiperInterface_V2(
            args.can, start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
        piper.ConnectPort()
        time.sleep(0.5)
        if not piper.isOk():
            raise RuntimeError("Piper CAN receive thread is not healthy")
        pose = read_pose_deg(piper)
        feedback = read_arm_feedback(piper)
        ctrl = feedback["ctrl_mode"]
        if teach_arm:
            if is_drag_teach_active(
                    feedback["ctrl_mode"], feedback["teach_status"],
                    feedback["arm_status"]):
                LOG.info(
                    "already in drag-teach; skip ModeCtrl so brakes stay open"
                )
            else:
                feedback = enter_drag_teach(piper, speed=args.mode_speed)
            pose = read_pose_deg(piper)
            ctrl = feedback["ctrl_mode"]
            LOG.info(
                "teach-arm: no streaming ModeCtrl/JointCtrl after enter; "
                "ctrl=%s(%s) arm=%s(%s) teach=%s(%s) pose=%s",
                ctrl, ctrl_mode_name(ctrl),
                feedback["arm_status"], arm_status_name(feedback["arm_status"]),
                feedback["teach_status"],
                teach_status_name(feedback["teach_status"]),
                [round(v, 2) for v in pose],
            )
            LOG.info(
                "drag the arm when teach_status=recording or ctrl_mode=2; "
                "press the Piper teach button if it is still stiff. "
                "Left stick still drives the dog. Do not send piper.go_zero "
                "or start chunk/teleop while recording"
            )
        else:
            ensure_enabled(piper, speed=args.mode_speed)
            pose = read_pose_deg(piper)
            send_row(piper, pose, args.mode_speed)
            LOG.info("holding piper pose %s", [round(v, 2) for v in pose])
    elif args.skip_arm:
        LOG.info("arm skipped (--skip-arm)")
    else:
        LOG.info("dry-run: hardware writes disabled")

    dog_period = 1.0 / max(args.dog_hz, 1.0)
    next_dog = time.monotonic()
    last_status = time.monotonic()
    last_dog = [0.0, 0.0, 0.0]
    warned_no_stick = False
    warned_not_teach = False
    teach_rows = 0
    teach_enter_tries = 1 if teach_arm else 0
    last_teach_enter = time.monotonic()
    ticks = 0

    LOG.info(
        "ready source=%s teach_arm=%s arm_only=%s skip_arm=%s pointer=%s "
        "(pose-bridge-v3: hold L2, tap A to lie down / stand up; no STOP spam "
        "during pose hold)",
        reader.read().source, teach_arm, args.arm_only, args.skip_arm,
        args.episode_pointer,
    )
    if teach_arm:
        LOG.info(
            "mapping: left ly/lx = dog vx/vyaw; select=dog STOP; "
            "hold L2 + tap A = toggle stand_down/stand_up; "
            "START = stand_up while prone; "
            "right stick / R1 / R2 ignored (arm is drag-teach)"
        )
    else:
        LOG.info(
            "mapping: left ly/lx = dog vx/vyaw; right = J1/J2; "
            "L1=J3/J4 L2=J5/J6 R1=open R2=close select=stop; "
            "hold L2 + tap A = toggle stand_down/stand_up; "
            "START = stand_up while prone"
        )

    prev_keys = 0
    prev_keys_logged = -1
    pose_tracker = RemotePoseTracker()
    pose_hold_until = 0.0
    last_pose_cmd = 0.0
    dog_streaming = False

    try:
        t0 = time.monotonic()
        while not stop["flag"]:
            now = time.monotonic()
            stick = reader.read()
            if not stick.ok and not warned_no_stick and now - t0 > 2.0:
                LOG.warning(
                    "no wirelesscontroller samples yet; keep the remote ON "
                    "after the velocity bridge has the lease"
                )
                warned_no_stick = True

            if not args.dry_run:
                key_updates = []
                if hasattr(reader, "drain_key_updates"):
                    key_updates = reader.drain_key_updates()
                elif stick.keys != prev_keys:
                    key_updates = [stick.keys]
                for keys in key_updates:
                    pose_action = pose_tracker.feed(
                        keys, now, last_cmd=last_pose_cmd)
                    if not pose_action:
                        continue
                    try:
                        send_bridge_pose(args.bridge_sock, pose_action)
                        pose_hold_until = now + POSE_HOLD_S
                        last_pose_cmd = now
                        dog_streaming = False
                        LOG.info(
                            "remote pose -> %s (bridge); hold %.1fs keys=0x%x",
                            pose_action.upper(), POSE_HOLD_S, keys,
                        )
                    except Exception as exc:
                        LOG.error("remote %s failed: %s", pose_action, exc)
            prev_keys = stick.keys
            if stick.keys != prev_keys_logged:
                LOG.info(
                    "remote keys=%u (0x%x) source=%s",
                    stick.keys, stick.keys, stick.source,
                )
                prev_keys_logged = stick.keys

            local_stop = bool(stick.keys & KEY_SELECT)
            pose_locked = now < pose_hold_until
            dog_prone = pose_tracker.dog_prone
            if args.arm_only or local_stop or dog_prone or pose_locked:
                dog_cmd = [0.0, 0.0, 0.0]
            else:
                dog_cmd = stick_to_dog(
                    stick.lx, stick.ly, args.deadzone, args.vx_max, args.vyaw_max)

            if teach_arm:
                if piper is not None:
                    pose = read_pose_deg(piper)
                    feedback = read_arm_feedback(piper)
                    ctrl = feedback["ctrl_mode"]
                    piper_log.log(pose, pose)
                    if is_drag_teach_active(
                            ctrl, feedback["teach_status"],
                            feedback["arm_status"]):
                        teach_rows += 1
                        warned_not_teach = False
                    else:
                        if (teach_enter_tries < 6
                                and now - last_teach_enter >= 2.0):
                            try:
                                piper.MotionCtrl_1(0x00, 0x00, 0x01)
                                teach_enter_tries += 1
                                last_teach_enter = now
                                LOG.info(
                                    "retry MotionCtrl_1 drag-teach (%d/6)",
                                    teach_enter_tries,
                                )
                            except Exception as exc:
                                LOG.error(
                                    "retry drag-teach failed: %s", exc)
                        if now - last_status >= 2.0 or not warned_not_teach:
                            LOG.warning(
                                "arm not in drag-teach yet "
                                "(ctrl=%s %s arm=%s %s teach=%s %s). "
                                "Press the Piper teach button if still stiff; "
                                "CSV still logs measured pose",
                                ctrl, ctrl_mode_name(ctrl),
                                feedback["arm_status"],
                                arm_status_name(feedback["arm_status"]),
                                feedback["teach_status"],
                                teach_status_name(feedback["teach_status"]),
                            )
                            warned_not_teach = True
                elif not args.skip_arm:
                    piper_log.log(pose, pose)
            else:
                pose = integrate_arm(
                    pose, stick.rx, stick.ry, stick.keys, args.dt,
                    args.deadzone, args.joint_rate, args.grip_rate,
                )
                if piper is not None:
                    send_row(piper, pose, args.mode_speed)
                    state = read_pose_deg(piper)
                    piper_log.log(pose, state)
                elif not args.skip_arm:
                    piper_log.log(pose, pose)

            if now >= next_dog:
                last_dog = dog_cmd
                if not args.dry_run and not pose_locked:
                    want_move = not local_stop and not (
                        dog_cmd[0] == 0.0 and dog_cmd[2] == 0.0)
                    try:
                        if want_move:
                            line = "MOVE %.6f %.6f %.6f" % tuple(dog_cmd)
                            send_line(args.bridge_sock, line, timeout=0.2)
                            dog_streaming = True
                        elif dog_streaming:
                            send_line(args.bridge_sock, "STOP", timeout=0.2)
                            dog_streaming = False
                    except Exception as exc:
                        LOG.error("dog bridge command failed: %s", exc)
                go2_log.log(dog_cmd)
                next_dog = now + dog_period

            ticks += 1
            if now - last_status >= 2.0:
                extra = ""
                if teach_arm:
                    extra = (
                        " ctrl=%s(%s) arm=%s(%s) teach=%s(%s) teach_rows=%d"
                        % (
                            ctrl, ctrl_mode_name(ctrl),
                            feedback["arm_status"],
                            arm_status_name(feedback["arm_status"]),
                            feedback["teach_status"],
                            teach_status_name(feedback["teach_status"]),
                            teach_rows,
                        )
                    )
                LOG.info(
                    "stick lx=%.2f ly=%.2f rx=%.2f ry=%.2f keys=%u ok=%s "
                    "dog=%s arm=%s%s",
                    stick.lx, stick.ly, stick.rx, stick.ry, stick.keys,
                    stick.ok, [round(v, 3) for v in last_dog],
                    [round(v, 2) for v in pose], extra,
                )
                last_status = now
                go2_log.flush()
                piper_log.flush()

            sleep_for = (t0 + ticks * args.dt) - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        if not args.dry_run:
            try:
                send_line(args.bridge_sock, "STOP", timeout=1.0)
            except Exception as exc:
                LOG.error("final dog STOP failed: %s", exc)
        if teach_arm and piper is not None and exit_drag_teach is not None:
            exit_drag_teach(piper)
        go2_log.close()
        piper_log.close()
        reader.close()
        if teach_arm:
            LOG.info(
                "teach capture stopped; teach_rows=%d ctrl=%s arm=%s "
                "teach=%s; exit teach on the arm before replay",
                teach_rows, ctrl_mode_name(ctrl),
                arm_status_name(feedback["arm_status"]),
                teach_status_name(feedback["teach_status"]),
            )
        else:
            LOG.info("teleop stopped; arm holds last pose")


if __name__ == "__main__":
    main()
