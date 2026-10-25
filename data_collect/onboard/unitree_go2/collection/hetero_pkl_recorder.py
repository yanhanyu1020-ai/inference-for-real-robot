#!/usr/bin/env python3
"""Read-only Piper + dual RealSense recorder with atomic per-frame PKL output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cv2


JOINT_NAMES = tuple(f"joint_{i}.pos" for i in range(1, 7))
GRIPPER_NAME = "gripper.pos"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def udev_properties(device: Path) -> dict[str, str]:
    completed = subprocess.run(
        ["udevadm", "info", "--query=property", f"--name={device}"],
        check=False,
        capture_output=True,
        text=True,
    )
    properties: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    return properties


def camera_candidates(serial: str) -> list[Path]:
    candidates = []
    for device in sorted(Path("/dev").glob("video*")):
        properties = udev_properties(device)
        if serial not in properties.get("ID_SERIAL", ""):
            continue
        # RealSense RGB is interface 1.3; interface 1.0 is depth/infrared.
        if not properties.get("ID_PATH", "").endswith(":1.3"):
            continue
        candidates.append(device)
    return candidates


def open_color_camera(serial: str, width: int, height: int, fps: int):
    errors = []
    for device in camera_candidates(serial):
        cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            errors.append(f"{device}: open failed")
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None and frame.shape[:2] == (height, width):
            return cap, device, frame
        errors.append(f"{device}: first frame failed or shape mismatch")
        cap.release()
    detail = "; ".join(errors) if errors else "no matching video nodes"
    raise RuntimeError(f"RealSense {serial} RGB unavailable: {detail}")


class CameraReader:
    def __init__(self, name: str, serial: str, width: int, height: int, fps: int):
        self.name = name
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] | None = None
        self.frames = 0
        self.failures = 0
        self.device = ""
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.name}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cap = None
        try:
            cap, device, first = open_color_camera(
                self.serial, self.width, self.height, self.fps
            )
            self.device = str(device)
            self._publish(first)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.failures += 1
                    time.sleep(0.01)
                    continue
                self._publish(frame)
        except Exception as exc:
            self.failures += 1
            self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()

    def _publish(self, frame) -> None:
        now_mono = time.monotonic_ns()
        now_wall = time.time_ns()
        self.frames += 1
        with self._lock:
            self._latest = {
                "seq": self.frames,
                "monotonic_ns": now_mono,
                "wall_time_ns": now_wall,
                "image": frame,
            }

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest is None:
                return None
            value = dict(self._latest)
            value["image"] = self._latest["image"].copy()
            return value

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


class PiperReader:
    def __init__(self, can_name: str, raw_dir: Path, period_s: float = 0.005):
        self.can_name = can_name
        self.raw_dir = raw_dir
        self.period_s = period_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] | None = None
        self.rows = 0
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="piper-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        state_path = self.raw_dir / "piper_state.csv"
        status_path = self.raw_dir / "piper_status.csv"
        try:
            from piper_sdk import C_PiperInterface_V2  # type: ignore

            piper = C_PiperInterface_V2(
                self.can_name,
                start_sdk_joint_limit=True,
                start_sdk_gripper_limit=True,
            )
            piper.ConnectPort()
            time.sleep(0.5)
            if not piper.isOk():
                raise RuntimeError("Piper CAN receive thread is not healthy")

            with state_path.open("w", newline="") as state_file, status_path.open(
                "w", newline=""
            ) as status_file:
                state_fields = [
                    "monotonic_ns",
                    "wall_time_ns",
                    "seq",
                    *JOINT_NAMES,
                    GRIPPER_NAME,
                ]
                status_fields = [
                    "monotonic_ns",
                    "wall_time_ns",
                    "seq",
                    "ctrl_mode",
                    "arm_status",
                    "teach_status",
                ]
                state_writer = csv.DictWriter(state_file, fieldnames=state_fields)
                status_writer = csv.DictWriter(status_file, fieldnames=status_fields)
                state_writer.writeheader()
                status_writer.writeheader()
                next_tick = time.monotonic()
                while not self._stop.is_set():
                    monotonic = time.monotonic_ns()
                    wall = time.time_ns()
                    joints_msg = piper.GetArmJointMsgs().joint_state
                    joints_raw = [
                        joints_msg.joint_1,
                        joints_msg.joint_2,
                        joints_msg.joint_3,
                        joints_msg.joint_4,
                        joints_msg.joint_5,
                        joints_msg.joint_6,
                    ]
                    joints = [math.radians(float(value) / 1000.0) for value in joints_raw]
                    gripper_raw = piper.GetArmGripperMsgs().gripper_state.grippers_angle
                    gripper = float(gripper_raw) / 1_000_000.0
                    arm_status = piper.GetArmStatus().arm_status
                    status = {
                        "ctrl_mode": int(getattr(arm_status, "ctrl_mode", -1)),
                        "arm_status": int(getattr(arm_status, "arm_status", -1)),
                        "teach_status": int(getattr(arm_status, "teach_status", -1)),
                    }
                    self.rows += 1
                    state = dict(zip(JOINT_NAMES, joints))
                    state[GRIPPER_NAME] = gripper
                    state_writer.writerow(
                        {
                            "monotonic_ns": monotonic,
                            "wall_time_ns": wall,
                            "seq": self.rows,
                            **state,
                        }
                    )
                    status_writer.writerow(
                        {
                            "monotonic_ns": monotonic,
                            "wall_time_ns": wall,
                            "seq": self.rows,
                            **status,
                        }
                    )
                    if self.rows % 200 == 0:
                        state_file.flush()
                        status_file.flush()
                    with self._lock:
                        self._latest = {
                            "monotonic_ns": monotonic,
                            "wall_time_ns": wall,
                            "seq": self.rows,
                            "state": state,
                            "status": status,
                        }
                    next_tick += self.period_s
                    delay = next_tick - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_tick = time.monotonic()
        except Exception as exc:
            self.error = str(exc)

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


class EpisodeWriter:
    def __init__(self, frames_dir: Path, workers: int, queue_size: int, png_level: int):
        self.frames_dir = frames_dir
        self.png_level = png_level
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.threads = [
            threading.Thread(target=self._worker, name=f"pkl-writer-{i}", daemon=True)
            for i in range(workers)
        ]
        self._lock = threading.Lock()
        self.saved = 0
        self.errors = 0
        self.dropped = 0
        for thread in self.threads:
            thread.start()

    def put(self, record: dict[str, Any]) -> bool:
        try:
            self.queue.put_nowait(record)
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            return False

    def _worker(self) -> None:
        while True:
            record = self.queue.get()
            if record is None:
                self.queue.task_done()
                return
            try:
                for camera in record["camera"].values():
                    image = camera.pop("_image")
                    ok, encoded = cv2.imencode(
                        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, self.png_level]
                    )
                    if not ok:
                        raise RuntimeError("PNG encoding failed")
                    camera["rgb"] = encoded.tobytes()
                filename = f"{record['timestamp_ns']}.pkl"
                destination = self.frames_dir / filename
                temporary = self.frames_dir / f".{filename}.tmp"
                with temporary.open("wb") as handle:
                    pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.flush()
                os.replace(temporary, destination)
                with self._lock:
                    self.saved += 1
            except Exception:
                with self._lock:
                    self.errors += 1
            finally:
                self.queue.task_done()

    def close(self) -> None:
        self.queue.join()
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join(timeout=10)


def load_json_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def check_cameras(config: dict[str, Any]) -> None:
    opened = []
    try:
        for name, serial in config["cameras"].items():
            cap, device, frame = open_color_camera(
                serial, config["width"], config["height"], config["fps"]
            )
            opened.append(cap)
            print(f"{name}: serial={serial} device={device} shape={frame.shape}")
    finally:
        for cap in opened:
            cap.release()


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    episode_dir = args.episode_dir.resolve()
    frames_dir = episode_dir / "frames"
    raw_dir = episode_dir / "raw"
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def request_stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cameras = {
        name: CameraReader(
            name, serial, config["width"], config["height"], config["fps"]
        )
        for name, serial in config["cameras"].items()
    }
    piper = PiperReader(config["can_interface"], raw_dir)
    writer = EpisodeWriter(
        frames_dir,
        config["save_workers"],
        config["save_queue_size"],
        config["png_compression"],
    )
    for camera in cameras.values():
        camera.start()
    piper.start()

    started_wall = time.time_ns()
    started_mono = time.monotonic_ns()
    frame_index = 0
    skipped = 0
    last_sequences = {name: 0 for name in cameras}
    # Poll at twice the camera rate; a frame is emitted only after both
    # cameras advance, which avoids phase-dependent drops without duplicates.
    period = 1.0 / (float(config["fps"]) * 2.0)
    next_tick = time.monotonic()
    last_status = 0.0
    go2_path = raw_dir / "go2_snapshot.json"
    gamepad_path = raw_dir / "piper_gamepad_snapshot.json"

    try:
        while not stop.is_set():
            now = time.monotonic()
            snapshots = {name: camera.snapshot() for name, camera in cameras.items()}
            piper_snapshot = piper.snapshot()
            go2_snapshot = load_json_snapshot(go2_path)
            gamepad_snapshot = load_json_snapshot(gamepad_path)
            camera_errors = {
                name: camera.error for name, camera in cameras.items() if camera.error
            }
            if camera_errors:
                raise RuntimeError(f"camera reader stopped: {camera_errors}")
            complete = (
                all(snapshots.values())
                and piper_snapshot
                and go2_snapshot
                and gamepad_snapshot
            )
            if piper.error:
                raise RuntimeError(f"Piper reader stopped: {piper.error}")
            if complete:
                sequences = {name: value["seq"] for name, value in snapshots.items()}
                all_new = all(sequences[name] != last_sequences[name] for name in cameras)
                camera_times = [value["monotonic_ns"] for value in snapshots.values()]
                camera_skew = max(camera_times) - min(camera_times)
                now_mono_ns = time.monotonic_ns()
                camera_stale = max(now_mono_ns - value["monotonic_ns"] for value in snapshots.values())
                piper_stale = now_mono_ns - piper_snapshot["monotonic_ns"]
                gamepad_stale = now_mono_ns - int(gamepad_snapshot["monotonic_ns"])
                go2_states = (
                    go2_snapshot.get("sport_mode_state", {}),
                    go2_snapshot.get("low_state", {}),
                )
                go2_valid = all(state.get("valid") for state in go2_states)
                go2_stale = max(
                    (now_mono_ns - int(state.get("monotonic_ns", 0)))
                    for state in go2_states
                )
                valid = (
                    all_new
                    and camera_skew <= int(config["max_camera_skew_ms"] * 1e6)
                    and camera_stale <= int(config["camera_stale_ms"] * 1e6)
                    and go2_valid
                    and go2_stale <= int(config["go2_stale_ms"] * 1e6)
                    and piper_stale <= int(config["piper_stale_ms"] * 1e6)
                    and gamepad_stale <= int(config["piper_gamepad_stale_ms"] * 1e6)
                )
                if valid:
                    timestamp = max(value["wall_time_ns"] for value in snapshots.values())
                    record = {
                        "timestamp_ns": timestamp,
                        "frame_index": frame_index,
                        "camera": {
                            name: {
                                "serial": camera.serial,
                                "monotonic_ns": snapshots[name]["monotonic_ns"],
                                "wall_time_ns": snapshots[name]["wall_time_ns"],
                                "_image": snapshots[name]["image"],
                            }
                            for name, camera in cameras.items()
                        },
                        "go2": go2_snapshot,
                        "piper": {
                            "state": piper_snapshot["state"],
                            "status": piper_snapshot["status"],
                            "monotonic_ns": piper_snapshot["monotonic_ns"],
                            "wall_time_ns": piper_snapshot["wall_time_ns"],
                            "command": gamepad_snapshot,
                        },
                        "diagnostics": {
                            "camera_skew_ns": camera_skew,
                            "camera_stale_ns": camera_stale,
                            "go2_stale_ns": go2_stale,
                            "piper_stale_ns": piper_stale,
                            "piper_gamepad_stale_ns": gamepad_stale,
                        },
                    }
                    writer.put(record)
                    frame_index += 1
                    last_sequences = sequences
                else:
                    skipped += 1
            else:
                skipped += 1

            if now - last_status >= 1.0:
                atomic_json(
                    episode_dir / "status.json",
                    {
                        "running": True,
                        "elapsed_s": round((time.monotonic_ns() - started_mono) / 1e9, 3),
                        "frames_enqueued": frame_index,
                        "frames_saved": writer.saved,
                        "frames_dropped": writer.dropped,
                        "save_errors": writer.errors,
                        "queue_size": writer.queue.qsize(),
                        "skipped_sync_ticks": skipped,
                        "piper_rows": piper.rows,
                        "cameras": {
                            name: {
                                "serial": camera.serial,
                                "device": camera.device,
                                "frames": camera.frames,
                                "failures": camera.failures,
                            }
                            for name, camera in cameras.items()
                        },
                    },
                )
                last_status = now
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    finally:
        for camera in cameras.values():
            camera.stop()
        piper.stop()
        writer.close()

    ended_wall = time.time_ns()
    elapsed_s = max((ended_wall - started_wall) / 1e9, 1e-9)
    summary = {
        "start_wall_time_ns": started_wall,
        "end_wall_time_ns": ended_wall,
        "duration_s": elapsed_s,
        "frames_enqueued": frame_index,
        "frames_saved": writer.saved,
        "effective_fps": writer.saved / elapsed_s,
        "frames_dropped": writer.dropped,
        "save_errors": writer.errors,
        "skipped_sync_ticks": skipped,
        "piper_rows": piper.rows,
        "piper_error": piper.error,
        "cameras": {
            name: {
                "serial": camera.serial,
                "device": camera.device,
                "frames": camera.frames,
                "failures": camera.failures,
            }
            for name, camera in cameras.items()
        },
    }
    atomic_json(episode_dir / "summary.json", summary)
    atomic_json(episode_dir / "status.json", {"running": False, **summary})
    return 0 if writer.errors == 0 and piper.error is None else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--check-cameras", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.check_cameras:
        check_cameras(config)
        return 0
    if args.episode_dir is None:
        raise SystemExit("--episode-dir is required unless --check-cameras is used")
    return run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
