#!/usr/bin/env python3
import argparse
import dataclasses
import math
import os
import pickle
import socket
import subprocess
import sys
import time
from pathlib import Path


JOINT_NAMES = tuple("joint_%d.pos" % index for index in range(1, 7))
GRIPPER_NAME = "gripper.pos"
JOINT_LIMITS = (
    (-2.6179, 2.6179),
    (0.0, 3.14),
    (-2.967, 0.0),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-2.09439, 2.09439),
)
CONFLICT_NAMES = (
    "go2_velocity_bridge",
    "go2_command_executor",
    "go2_stream_loop.py",
    "go2_chunk_loop.py",
    "go2_state_bridge",
    "go2_replay_bridge",
    "piper_stream_loop.py",
    "piper_chunk_loop.py",
    "hetero_teleop_loop.py",
    "hetero_teach_loop.py",
    "hetero_pkl_recorder.py",
)


@dataclasses.dataclass
class Sample:
    time_s: float
    position: tuple
    yaw: float
    velocity: tuple
    yaw_speed: float
    body_height: float
    joints: tuple
    gripper: float
    posture: str = "unknown"


@dataclasses.dataclass
class Go2State:
    valid: bool
    monotonic_ns: int
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    yaw_speed: float
    body_height: float


def positive(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def unit_interval(value):
    number = positive(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must not exceed 1.0")
    return number


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a synchronized PKL episode on Go2 and Piper. The default is dry-run."
    )
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="send commands to real hardware")
    parser.add_argument(
        "--confirm",
        help="real hardware mode requires the exact value REPLAY",
    )
    parser.add_argument("--devices", choices=("both", "go2", "piper"), default="both")
    parser.add_argument("--speed", type=unit_interval, default=1.0, help="replay speed, (0, 1]")
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--can-interface", default="can0")
    parser.add_argument(
        "--go2-bridge",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build" / "go2_replay_bridge",
    )
    parser.add_argument("--kp-xy", type=positive, default=1.0)
    parser.add_argument("--kp-yaw", type=positive, default=1.5)
    parser.add_argument("--feedforward", type=float, default=1.0)
    parser.add_argument("--max-vx", type=positive, help="default: auto from recorded vx")
    parser.add_argument("--max-vy", type=positive, help="default: auto from recorded vy")
    parser.add_argument("--max-vyaw", type=positive, help="default: auto from recorded vyaw")
    parser.add_argument("--max-position-error", type=positive, default=0.5)
    parser.add_argument("--max-yaw-error", type=positive, default=0.8)
    parser.add_argument("--watchdog-ms", type=int, default=400)
    parser.add_argument("--piper-mode-speed", type=int, default=100)
    parser.add_argument("--piper-start-seconds", type=positive, default=5.0)
    parser.add_argument(
        "--max-joint-speed",
        type=positive,
        help="rad/s; default: auto from recorded Piper trajectory",
    )
    parser.add_argument("--gripper-effort", type=int, default=3000)
    parser.add_argument("--down-height", type=float, default=0.15)
    parser.add_argument("--up-height", type=float, default=0.25)
    parser.add_argument("--log-hz", type=positive, default=1.0)
    return parser.parse_args()


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def frame_timestamp(path):
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError("frame filename is not a timestamp: %s" % path.name) from exc


def assign_postures(samples, down_height, up_height):
    stable = []
    for sample in samples:
        if sample.body_height <= down_height:
            stable.append("down")
        elif sample.body_height >= up_height:
            stable.append("up")
        else:
            stable.append(None)
    next_state = None
    future = [None] * len(samples)
    for index in range(len(samples) - 1, -1, -1):
        if stable[index] is not None:
            next_state = stable[index]
        future[index] = next_state
    previous = next((value for value in stable if value is not None), "down")
    for index, sample in enumerate(samples):
        if stable[index] is not None:
            previous = stable[index]
        sample.posture = stable[index] or future[index] or previous


def load_trajectory(episode, down_height=0.15, up_height=0.25):
    if down_height >= up_height:
        raise ValueError("down-height must be less than up-height")
    frames_dir = episode.expanduser().resolve() / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError("frames directory does not exist: %s" % frames_dir)
    paths = sorted(frames_dir.glob("*.pkl"), key=frame_timestamp)
    if not paths:
        raise FileNotFoundError("no PKL frames found in: %s" % frames_dir)

    samples = []
    first_timestamp = None
    last_timestamp = None
    for path in paths:
        with path.open("rb") as handle:
            record = pickle.load(handle)
        try:
            timestamp = int(record["timestamp_ns"])
            sport = record["go2"]["sport_mode_state"]
            piper = record["piper"]["state"]
            position = tuple(float(value) for value in sport["position"][:3])
            velocity = tuple(float(value) for value in sport["velocity"][:2])
            yaw = float(sport["rpy"][2])
            joints = tuple(float(piper[name]) for name in JOINT_NAMES)
            gripper = float(piper[GRIPPER_NAME])
            body_height = float(sport["body_height"])
            yaw_speed = float(sport["yaw_speed"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("%s: invalid replay state: %s" % (path, exc)) from exc
        values = position + velocity + (yaw, yaw_speed, body_height) + joints + (gripper,)
        if len(position) != 3 or len(velocity) != 2 or not all(math.isfinite(v) for v in values):
            raise ValueError("%s: replay state contains invalid numeric values" % path)
        if last_timestamp is not None and timestamp <= last_timestamp:
            raise ValueError("%s: timestamps are not strictly increasing" % path)
        first_timestamp = timestamp if first_timestamp is None else first_timestamp
        samples.append(
            Sample(
                time_s=(timestamp - first_timestamp) / 1e9,
                position=position,
                yaw=yaw,
                velocity=velocity,
                yaw_speed=yaw_speed,
                body_height=body_height,
                joints=joints,
                gripper=gripper,
            )
        )
        last_timestamp = timestamp
    assign_postures(samples, down_height, up_height)
    return samples


def summarize(samples):
    path_length = 0.0
    max_joint_speed = 0.0
    for previous, current in zip(samples, samples[1:]):
        path_length += math.hypot(
            current.position[0] - previous.position[0],
            current.position[1] - previous.position[1],
        )
        dt = current.time_s - previous.time_s
        if dt > 0:
            max_joint_speed = max(
                max_joint_speed,
                max(abs(a - b) for a, b in zip(current.joints, previous.joints)) / dt,
            )
    events = []
    previous = None
    for sample in samples:
        if sample.posture != previous:
            events.append((sample.time_s, sample.posture))
            previous = sample.posture
    return {
        "frames": len(samples),
        "duration_s": samples[-1].time_s,
        "path_length_m": path_length,
        "max_recorded_vx": max(abs(s.velocity[0]) for s in samples),
        "max_recorded_vy": max(abs(s.velocity[1]) for s in samples),
        "max_recorded_vyaw": max(abs(s.yaw_speed) for s in samples),
        "max_joint_speed": max_joint_speed,
        "posture_events": events,
    }


def resolve_limits(args, summary):
    requirements = {
        "max_vx": summary["max_recorded_vx"] * abs(args.feedforward) * args.speed,
        "max_vy": summary["max_recorded_vy"] * abs(args.feedforward) * args.speed,
        "max_vyaw": summary["max_recorded_vyaw"] * abs(args.feedforward) * args.speed,
        "max_joint_speed": summary["max_joint_speed"] * args.speed,
    }
    baselines = {
        "max_vx": 0.5,
        "max_vy": 0.5,
        "max_vyaw": 1.0,
        "max_joint_speed": 1.0,
    }
    warnings = []
    for name, required in requirements.items():
        configured = getattr(args, name)
        if configured is None:
            setattr(args, name, max(baselines[name], required * 1.05))
        elif configured + 1e-9 < required:
            warnings.append(
                "%s=%.3f is below the recorded requirement %.3f and will limit replay"
                % (name.replace("_", "-"), configured, required)
            )
    return warnings


def relative_pose(origin, sample):
    dx = sample.position[0] - origin.position[0]
    dy = sample.position[1] - origin.position[1]
    cosine = math.cos(origin.yaw)
    sine = math.sin(origin.yaw)
    return (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        wrap_angle(sample.yaw - origin.yaw),
    )


def replay_target(anchor, relative):
    cosine = math.cos(anchor.yaw)
    sine = math.sin(anchor.yaw)
    return (
        anchor.x + cosine * relative[0] - sine * relative[1],
        anchor.y + sine * relative[0] + cosine * relative[1],
        wrap_angle(anchor.yaw + relative[2]),
    )


def compute_go2_command(sample, target, state, args):
    error_x = target[0] - state.x
    error_y = target[1] - state.y
    position_error = math.hypot(error_x, error_y)
    yaw_error = wrap_angle(target[2] - state.yaw)
    if position_error > args.max_position_error:
        raise RuntimeError("Go2 position tracking error %.3f m exceeds limit" % position_error)
    if abs(yaw_error) > args.max_yaw_error:
        raise RuntimeError("Go2 yaw tracking error %.3f rad exceeds limit" % yaw_error)
    cosine = math.cos(state.yaw)
    sine = math.sin(state.yaw)
    body_error_x = cosine * error_x + sine * error_y
    body_error_y = -sine * error_x + cosine * error_y
    feedforward = args.feedforward * args.speed
    vx = feedforward * sample.velocity[0] + args.kp_xy * body_error_x
    vy = feedforward * sample.velocity[1] + args.kp_xy * body_error_y
    vyaw = feedforward * sample.yaw_speed + args.kp_yaw * yaw_error
    return (
        max(-args.max_vx, min(args.max_vx, vx)),
        max(-args.max_vy, min(args.max_vy, vy)),
        max(-args.max_vyaw, min(args.max_vyaw, vyaw)),
        position_error,
        yaw_error,
    )


def conflicting_processes():
    conflicts = []
    own_pid = os.getpid()
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
            if pid == own_pid:
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError):
            continue
        if any(name in command for name in CONFLICT_NAMES):
            conflicts.append((pid, command.strip()))
    return conflicts


class Go2Controller:
    def __init__(self, args):
        bridge = args.go2_bridge.expanduser().resolve()
        if not bridge.is_file():
            raise FileNotFoundError("Go2 replay bridge is missing: %s" % bridge)
        self.socket_path = Path("/tmp/heterovla_go2_replay_%d.sock" % os.getpid())
        # Interactive Unitree shells source ROS/CycloneDDS libraries that are
        # ABI-incompatible with the DDS bundled in Unitree SDK2.
        clean_environment = {
            "HOME": str(Path.home()),
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
        self.process = subprocess.Popen(
            [
                str(bridge),
                args.network_interface,
                str(self.socket_path),
                str(args.max_vx),
                str(args.max_vy),
                str(args.max_vyaw),
                str(args.watchdog_ms),
                "10",
            ],
            env=clean_environment,
        )
        deadline = time.monotonic() + 20.0
        while not self.socket_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError("Go2 replay bridge exited with code %s" % self.process.returncode)
            if time.monotonic() >= deadline:
                self.process.terminate()
                raise RuntimeError("timed out waiting for Go2 replay bridge")
            time.sleep(0.05)
        state = self.state()
        if not state.valid:
            deadline = time.monotonic() + 3.0
            while not state.valid and time.monotonic() < deadline:
                time.sleep(0.05)
                state = self.state()
        if not state.valid:
            self.close()
            raise RuntimeError("Go2 sport state is unavailable")

    def request(self, command):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2.0)
            connection.connect(str(self.socket_path))
            connection.sendall((command + "\n").encode())
            connection.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        fields = b"".join(chunks).decode().strip().split()
        if len(fields) != 12 or fields[0] != "OK":
            raise RuntimeError("invalid Go2 bridge response: %r" % fields)
        rc = int(fields[1])
        if rc != 0:
            raise RuntimeError("Go2 command %s failed with rc=%d" % (command.split()[0], rc))
        return Go2State(
            valid=bool(int(fields[2])),
            monotonic_ns=int(fields[3]),
            x=float(fields[4]),
            y=float(fields[5]),
            z=float(fields[6]),
            yaw=float(fields[7]),
            vx=float(fields[8]),
            vy=float(fields[9]),
            yaw_speed=float(fields[10]),
            body_height=float(fields[11]),
        )

    def state(self):
        return self.request("STATE")

    def move(self, vx, vy, vyaw):
        return self.request("MOVE %.9f %.9f %.9f" % (vx, vy, vyaw))

    def stop(self):
        return self.request("STOP")

    def set_posture(self, posture):
        return self.request("STAND_UP" if posture == "up" else "STAND_DOWN")

    def close(self):
        if getattr(self, "process", None) is None:
            return
        if self.process.poll() is None:
            try:
                self.request("STOP")
                self.request("QUIT")
            except Exception:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.process = None


class PiperController:
    def __init__(self, args):
        from piper_sdk import C_PiperInterface_V2

        self.args = args
        self.piper = None
        try:
            self.piper = C_PiperInterface_V2(
                args.can_interface,
                start_sdk_joint_limit=True,
                start_sdk_gripper_limit=True,
            )
            self.piper.ConnectPort()
            time.sleep(0.5)
            if not self.piper.isOk():
                raise RuntimeError("Piper CAN receive thread is not healthy")
            status = self.piper.GetArmStatus().arm_status
            if int(getattr(status, "arm_status", -1)) != 0:
                raise RuntimeError("Piper arm_status is not normal: %s" % status.arm_status)
            if int(getattr(status, "ctrl_mode", -1)) == 2:
                self.piper.MotionCtrl_1(0x00, 0x00, 0x02)
                time.sleep(0.2)
            deadline = time.monotonic() + 5.0
            while not all(self.piper.GetArmEnableStatus()):
                self.piper.EnableArm(7)
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out enabling Piper motors")
                time.sleep(0.02)
            self.last_joints, self.last_gripper = self.read_state()
            self.piper.ModeCtrl(0x01, 0x01, args.piper_mode_speed, 0x00)
        except Exception:
            if self.piper is not None:
                self.piper.DisconnectPort()
                self.piper = None
            raise

    def read_state(self):
        state = self.piper.GetArmJointMsgs().joint_state
        raw = (state.joint_1, state.joint_2, state.joint_3, state.joint_4, state.joint_5, state.joint_6)
        joints = tuple(math.radians(float(value) / 1000.0) for value in raw)
        gripper = float(self.piper.GetArmGripperMsgs().gripper_state.grippers_angle) / 1_000_000.0
        return joints, gripper

    def send(self, joints, gripper, dt=None):
        values = []
        for index, (value, limits) in enumerate(zip(joints, JOINT_LIMITS)):
            value = max(limits[0], min(limits[1], float(value)))
            if dt is not None:
                step = self.args.max_joint_speed * dt
                value = max(self.last_joints[index] - step, min(self.last_joints[index] + step, value))
            values.append(value)
        gripper = max(0.0005, min(0.065, float(gripper)))
        commands = [int(round(math.degrees(value) * 1000.0)) for value in values]
        self.piper.ModeCtrl(0x01, 0x01, self.args.piper_mode_speed, 0x00)
        self.piper.JointCtrl(*commands)
        self.piper.GripperCtrl(int(round(gripper * 1_000_000.0)), self.args.gripper_effort, 0x01, 0)
        self.last_joints = tuple(values)
        self.last_gripper = gripper

    def move_to_start(self, sample):
        start_joints, start_gripper = self.read_state()
        duration = self.args.piper_start_seconds
        required_speed = max(abs(a - b) for a, b in zip(sample.joints, start_joints)) / duration
        if required_speed > self.args.max_joint_speed:
            raise RuntimeError(
                "Piper start motion requires %.3f rad/s, above %.3f rad/s limit; increase --piper-start-seconds"
                % (required_speed, self.args.max_joint_speed)
            )
        steps = max(1, round(duration * 30))
        started = time.monotonic()
        for index in range(1, steps + 1):
            ratio = index / steps
            joints = tuple(a + ratio * (b - a) for a, b in zip(start_joints, sample.joints))
            gripper = start_gripper + ratio * (sample.gripper - start_gripper)
            self.send(joints, gripper)
            deadline = started + index * duration / steps
            time.sleep(max(0.0, deadline - time.monotonic()))

    def hold(self):
        joints, gripper = self.read_state()
        self.send(joints, gripper)

    def close(self):
        if getattr(self, "piper", None) is None:
            return
        try:
            self.hold()
        finally:
            self.piper.DisconnectPort()
            self.piper = None


def align_initial_posture(go2, posture, args):
    state = go2.state()
    reached = state.body_height >= args.up_height if posture == "up" else state.body_height <= args.down_height
    if reached:
        return state
    print("aligning Go2 initial posture: %s" % posture)
    go2.set_posture(posture)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        time.sleep(0.1)
        state = go2.state()
        reached = state.body_height >= args.up_height if posture == "up" else state.body_height <= args.down_height
        if reached:
            return state
    raise RuntimeError("Go2 did not reach initial %s posture" % posture)


def execute(samples, args):
    if args.confirm != "REPLAY":
        raise RuntimeError("real hardware replay requires --execute --confirm REPLAY")
    active_episode = Path(__file__).resolve().parents[1] / "run" / "active_episode"
    if active_episode.exists():
        raise RuntimeError("collection is active: %s" % active_episode.read_text().strip())
    conflicts = conflicting_processes()
    if conflicts:
        details = "; ".join("pid=%d %s" % item for item in conflicts)
        raise RuntimeError("conflicting control/collection process detected: %s" % details)

    use_go2 = args.devices in ("both", "go2")
    use_piper = args.devices in ("both", "piper")
    go2 = None
    piper = None
    try:
        if use_go2:
            go2 = Go2Controller(args)
            anchor = align_initial_posture(go2, samples[0].posture, args)
        else:
            anchor = None
        if use_piper:
            piper = PiperController(args)
            print("moving Piper to the first recorded joint pose")
            piper.move_to_start(samples[0])

        origin = samples[0]
        current_state = anchor
        current_posture = samples[0].posture
        posture_pending = None
        started = time.monotonic()
        next_log = started
        previous_time = 0.0
        for index, sample in enumerate(samples):
            target_time = started + sample.time_s / args.speed
            time.sleep(max(0.0, target_time - time.monotonic()))
            dt = (sample.time_s - previous_time) / args.speed if index else 1.0 / 30.0
            previous_time = sample.time_s

            if piper is not None:
                piper.send(sample.joints, sample.gripper, dt=max(0.001, dt))

            command = (0.0, 0.0, 0.0, 0.0, 0.0)
            if go2 is not None:
                if sample.posture != current_posture:
                    current_state = go2.set_posture(sample.posture)
                    current_posture = sample.posture
                    posture_pending = sample.posture
                if posture_pending is not None:
                    current_state = go2.state()
                    reached = (
                        current_state.body_height >= args.up_height
                        if posture_pending == "up"
                        else current_state.body_height <= args.down_height
                    )
                    if reached:
                        posture_pending = None
                standing = (
                    posture_pending is None
                    and current_state.body_height >= args.up_height
                    and sample.posture == "up"
                )
                if standing:
                    target = replay_target(anchor, relative_pose(origin, sample))
                    command = compute_go2_command(sample, target, current_state, args)
                    current_state = go2.move(*command[:3])

            now = time.monotonic()
            if now >= next_log:
                print(
                    "t=%7.3f frame=%d posture=%s vx=%+.3f vy=%+.3f vyaw=%+.3f pos_err=%.3f yaw_err=%+.3f"
                    % (sample.time_s, index, sample.posture, *command)
                )
                next_log = now + 1.0 / args.log_hz
        if go2 is not None:
            go2.stop()
        if piper is not None:
            piper.hold()
        print("replay completed; Go2 stopped and Piper is holding its final pose")
    finally:
        if go2 is not None:
            go2.close()
        if piper is not None:
            piper.close()


def print_summary(summary, args, warnings):
    print("frames=%d duration_s=%.3f path_length_m=%.3f" % (
        summary["frames"], summary["duration_s"], summary["path_length_m"]
    ))
    print("recorded max abs velocity: vx=%.3f vy=%.3f vyaw=%.3f" % (
        summary["max_recorded_vx"], summary["max_recorded_vy"], summary["max_recorded_vyaw"]
    ))
    print("recorded max Piper joint speed: %.3f rad/s" % summary["max_joint_speed"])
    print("posture events: %s" % ", ".join(
        "%.3fs=%s" % event for event in summary["posture_events"]
    ))
    print("resolved command limits: vx=%.3f vy=%.3f vyaw=%.3f joint=%.3f rad/s" % (
        args.max_vx, args.max_vy, args.max_vyaw, args.max_joint_speed
    ))
    for warning in warnings:
        print("warning: %s" % warning)


def main():
    args = parse_args()
    try:
        if args.watchdog_ms <= 0:
            raise ValueError("watchdog-ms must be positive")
        if not 1 <= args.piper_mode_speed <= 100:
            raise ValueError("piper-mode-speed must be in [1, 100]")
        if not 0 <= args.gripper_effort <= 5000:
            raise ValueError("gripper-effort must be in [0, 5000]")
        samples = load_trajectory(args.episode, args.down_height, args.up_height)
        summary = summarize(samples)
        warnings = resolve_limits(args, summary)
        print_summary(summary, args, warnings)
        if args.execute:
            execute(samples, args)
        else:
            print("dry-run only; no hardware command was sent")
            print("real hardware requires --execute --confirm REPLAY")
    except (FileNotFoundError, ImportError, OSError, ValueError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
