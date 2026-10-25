#!/usr/bin/env python3
"""Run the existing Gamepad_PiPER controller as a collection data source."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


JOINT_NAMES = tuple(f"joint_{index}.pos" for index in range(1, 7))
BUTTON_NAMES = ("a", "b", "x", "y", "lb", "rb", "back", "start", "home", "l3", "r3")
AXIS_NAMES = (
    "left_x",
    "left_y",
    "right_x",
    "right_y",
    "left_trigger",
    "right_trigger",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_controller(runtime: Path):
    sys.path.insert(0, str(runtime))
    from main import Teleop  # type: ignore

    return Teleop


def controller_paths(runtime: Path) -> tuple[Path, str, str]:
    urdf = runtime / "piper/piper.urdf"
    return urdf, "/base_link", "link6"


def check(config: dict[str, Any]) -> None:
    gamepad = config["piper_gamepad"]
    runtime = Path(gamepad["runtime"])
    urdf, _root, _target = controller_paths(runtime)
    if not runtime.is_dir():
        raise RuntimeError(f"Gamepad_PiPER runtime not found: {runtime}")
    if not urdf.is_file():
        raise RuntimeError(f"Piper URDF not found: {urdf}")
    Controller = load_controller(runtime)
    import pygame  # type: ignore
    from piper_sdk import C_PiperInterface_V2  # noqa: F401

    pygame.init()
    pygame.joystick.init()
    # mesh_path=None disables Viser while still initializing cuRobo FK/IK.
    Controller(None, str(urdf), None, _root, _target)
    count = pygame.joystick.get_count()
    names = [pygame.joystick.Joystick(index).get_name() for index in range(count)]
    print(f"Gamepad_PiPER runtime={runtime}")
    print(f"pygame={pygame.version.ver} gamepads={count} names={names}")
    if count == 0:
        print("warning: no gamepad is currently connected", file=sys.stderr)
    pygame.quit()


def input_snapshot(controller) -> dict[str, Any]:
    if controller.joystick is None:
        return {
            "connected": False,
            "name": None,
            "axes": {name: 0.0 for name in AXIS_NAMES},
            "buttons": {name: 0 for name in BUTTON_NAMES},
            "dpad": [0, 0],
        }
    return {
        "connected": True,
        "name": controller.joystick.get_name(),
        "axes": {name: float(controller._get_axis_value(name)) for name in AXIS_NAMES},
        "buttons": {
            name: (
                int(controller.joystick.get_button(controller.button_map[name]))
                if name in controller.button_map
                and controller.button_map[name] < controller.joystick.get_numbuttons()
                else 0
            )
            for name in BUTTON_NAMES
        },
        "dpad": list(controller._get_hat_value("dpad")),
    }


def send_command(piper, controller, state: dict[str, Any]) -> bool:
    if not state["arm_connected"] or not state["arm_enabled"]:
        return False
    import numpy as np  # type: ignore

    if state["low_level_mode"] == "joint":
        joints = np.round(np.degrees(state["joints"][:6]) * 1000).astype(int).tolist()
        piper.ModeCtrl(0x01, 0x01, state["movement_speed"], state["command_mode"])
        piper.JointCtrl(*joints)
    else:
        pose = state["xyz_rpy"].copy()
        pose[:3] = np.round(pose[:3] * 1_000_000)
        pose[3:] = np.round(pose[3:] * 1000)
        piper.ModeCtrl(0x01, 0x00, state["movement_speed"], state["command_mode"])
        piper.EndPoseCtrl(*pose.astype(int).tolist())
    gripper = int(controller.gripper_max_width * state["gripper"] * 10_000)
    piper.GripperCtrl(gripper, 3000, 0x01, 0)
    return True


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    gamepad_config = config["piper_gamepad"]
    runtime = Path(gamepad_config["runtime"])
    urdf, root_name, target_link_name = controller_paths(runtime)
    Controller = load_controller(runtime)
    from piper_sdk import C_PiperInterface_V2  # type: ignore
    import pygame  # type: ignore

    raw_dir = args.episode_dir.resolve() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "piper_gamepad.csv"
    snapshot_path = raw_dir / "piper_gamepad_snapshot.json"
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    piper = C_PiperInterface_V2(
        config["can_interface"],
        start_sdk_joint_limit=True,
        start_sdk_gripper_limit=True,
    )
    controller = Controller(piper, str(urdf), None, root_name, target_link_name)
    period = 1.0 / float(gamepad_config.get("control_hz", 200))
    next_tick = time.monotonic()
    sequence = 0
    fields = [
        "monotonic_ns",
        "wall_time_ns",
        "seq",
        "gamepad_connected",
        "gamepad_name",
        *AXIS_NAMES,
        "dpad_x",
        "dpad_y",
        *BUTTON_NAMES,
        "arm_connected",
        "arm_enabled",
        "up_level_mode",
        "low_level_mode",
        "command_mode",
        "movement_speed",
        "speed_factor",
        "command_sent",
        *JOINT_NAMES,
        "gripper.pos",
    ]

    try:
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            while not stop:
                controller.update()
                state = controller.get_state()
                sent = send_command(piper, controller, state)
                gamepad = input_snapshot(controller)
                monotonic_ns = time.monotonic_ns()
                wall_time_ns = time.time_ns()
                sequence += 1
                targets = {
                    name: float(value) for name, value in zip(JOINT_NAMES, controller.joint_angles)
                }
                targets["gripper.pos"] = controller.gripper_max_width * float(state["gripper"]) / 100.0
                snapshot = {
                    "monotonic_ns": monotonic_ns,
                    "wall_time_ns": wall_time_ns,
                    "seq": sequence,
                    "valid": bool(state["arm_connected"] and state["arm_enabled"]),
                    "arm_connected": bool(state["arm_connected"]),
                    "arm_enabled": bool(state["arm_enabled"]),
                    "up_level_mode": state["up_level_mode"],
                    "low_level_mode": state["low_level_mode"],
                    "command_mode": int(state["command_mode"]),
                    "movement_speed": int(state["movement_speed"]),
                    "speed_factor": float(state["speed_factor"]),
                    "command_sent": sent,
                    "target": targets,
                    "target_pose": [float(value) for value in state["xyz_rpy"]],
                    "gamepad": gamepad,
                }
                atomic_json(snapshot_path, snapshot)
                writer.writerow(
                    {
                        "monotonic_ns": monotonic_ns,
                        "wall_time_ns": wall_time_ns,
                        "seq": sequence,
                        "gamepad_connected": int(gamepad["connected"]),
                        "gamepad_name": gamepad["name"] or "",
                        **gamepad["axes"],
                        "dpad_x": gamepad["dpad"][0],
                        "dpad_y": gamepad["dpad"][1],
                        **gamepad["buttons"],
                        "arm_connected": int(state["arm_connected"]),
                        "arm_enabled": int(state["arm_enabled"]),
                        "up_level_mode": state["up_level_mode"],
                        "low_level_mode": state["low_level_mode"],
                        "command_mode": state["command_mode"],
                        "movement_speed": state["movement_speed"],
                        "speed_factor": state["speed_factor"],
                        "command_sent": int(sent),
                        **targets,
                    }
                )
                if sequence % 200 == 0:
                    csv_file.flush()
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
    finally:
        pygame.quit()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.check:
        check(config)
        return 0
    if args.episode_dir is None:
        parser.error("--episode-dir is required unless --check is used")
    return run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
