import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

from .common import OPTIONAL_TOPICS, capture_topics, validate_session_name
from .process_tracking import process_exists as _linux_process_exists


DEFAULT_DATA_ROOT = Path("/home/amov/p450_data")
MIN_FREE_BYTES = 20 * 1024**3
DEFAULT_MAX_MINUTES = 30.0
RAW_MAX_MINUTES = 2.0
DESKTOP_JOINT_ID = re.compile(r'joint_\d{8}_\d{6}_UTC[pm]\d{4}_[0-9a-f]{4}')


def _default_topics():
    import rosgraph

    published = rosgraph.Master("/p450_record").getPublishedTopics("/")
    return {name: topic_type for name, topic_type in published}


def _default_active_topics(topic_names, timeout_seconds=2.0):
    import rospy

    if not rospy.core.is_initialized():
        rospy.init_node(
            "p450_record_check", anonymous=True, disable_signals=True
        )
    received = set()
    lock = threading.Lock()
    ready = threading.Event()
    expected = set(topic_names)

    def callback(_, topic_name):
        with lock:
            received.add(topic_name)
            if received == expected:
                ready.set()

    subscribers = [
        rospy.Subscriber(
            topic_name,
            rospy.AnyMsg,
            callback,
            callback_args=topic_name,
            queue_size=1,
        )
        for topic_name in expected
    ]
    try:
        ready.wait(timeout_seconds)
        return received
    finally:
        for subscriber in subscribers:
            subscriber.unregister()


def _process_exists(pid):
    return _linux_process_exists(pid)


class RecorderManager:
    def __init__(
        self,
        data_root=DEFAULT_DATA_ROOT,
        topic_provider=None,
        activity_provider=None,
        disk_free_provider=None,
        popen_factory=None,
        now_provider=None,
        hostname_provider=None,
        proc_cmdline_reader=None,
        killpg_func=None,
        process_exists=None,
        sleep_func=None,
    ):
        self.data_root = Path(data_root)
        self.state_path = self.data_root / ".recording_state.json"
        self.topic_provider = topic_provider or _default_topics
        self.activity_provider = activity_provider or (
            _default_active_topics
            if topic_provider is None
            else lambda topic_names: set(topic_names)
        )
        self.disk_free_provider = disk_free_provider or (lambda path: shutil.disk_usage(path).free)
        self.popen_factory = popen_factory or subprocess.Popen
        self.now_provider = now_provider or (lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.hostname_provider = hostname_provider or socket.gethostname
        self.proc_cmdline_reader = proc_cmdline_reader or (
            lambda pid: Path(f"/proc/{pid}/cmdline").read_bytes()
        )
        self.killpg_func = killpg_func or getattr(os, "killpg", os.kill)
        self.process_exists = process_exists or _process_exists
        self.sleep_func = sleep_func or time.sleep

    def _read_state(self):
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def check(self, raw_rgb=False):
        self.data_root.mkdir(parents=True, exist_ok=True)
        errors = []
        try:
            topics = dict(self.topic_provider())
        except Exception as exc:
            topics = {}
            errors.append(f"cannot query ROS master: {exc}")

        required_topics = capture_topics(raw_rgb)
        missing_required = [
            name for name, expected in required_topics.items() if topics.get(name) != expected
        ]
        missing_optional = [
            name for name, expected in OPTIONAL_TOPICS.items() if topics.get(name) != expected
        ]
        if missing_required:
            errors.append("missing required topics: " + ", ".join(missing_required))

        inactive_required = []
        queryable_required = [
            name for name in required_topics if name not in missing_required
        ]
        if queryable_required:
            try:
                active_topics = set(self.activity_provider(queryable_required))
                inactive_required = [
                    name for name in queryable_required if name not in active_topics
                ]
            except Exception as exc:
                errors.append(f"cannot verify required topic activity: {exc}")
            if inactive_required:
                errors.append(
                    "required topics are not publishing messages: "
                    + ", ".join(inactive_required)
                )

        free_bytes = int(self.disk_free_provider(self.data_root))
        if free_bytes < MIN_FREE_BYTES:
            errors.append(
                f"disk space below 20 GiB: {free_bytes / 1024**3:.2f} GiB available"
            )

        state = self._read_state()
        active = bool(state and self.process_exists(int(state["pid"])))
        if active:
            errors.append(f"recorder already active with PID {state['pid']}")

        return {
            "ok": not errors,
            "errors": errors,
            "missing_required": missing_required,
            "inactive_required": inactive_required,
            "missing_optional": missing_optional,
            "available_topics": sorted(topics),
            "free_bytes": free_bytes,
            "active": active,
        }

    def start(self, session_name, raw_rgb=False, max_minutes=None):
        session_name = validate_session_name(session_name)
        if max_minutes is None:
            max_minutes = RAW_MAX_MINUTES if raw_rgb else DEFAULT_MAX_MINUTES
        max_minutes = float(max_minutes)
        if max_minutes <= 0:
            raise ValueError("max_minutes must be positive")
        if raw_rgb and max_minutes > RAW_MAX_MINUTES:
            raise ValueError("raw RGB capture is limited to 2 minutes")
        existing = self._read_state()
        if existing and self.process_exists(int(existing["pid"])):
            raise RuntimeError(f"recorder already active with PID {existing['pid']}")
        if existing:
            raise RuntimeError(
                f"stale recorder state exists at {self.state_path}; inspect it before removing"
            )

        report = self.check(raw_rgb=raw_rgb)
        if not report["ok"]:
            raise RuntimeError("; ".join(report["errors"]))

        stamp = self.now_provider()
        directory_name = (session_name if DESKTOP_JOINT_ID.fullmatch(session_name)
                          else f"{stamp}_{session_name}")
        session_dir = self.data_root / directory_name
        raw_dir = session_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=False)

        available = set(report["available_topics"])
        required_topics = capture_topics(raw_rgb)
        selected_topics = [
            topic
            for topic in list(required_topics) + list(OPTIONAL_TOPICS)
            if topic in available
        ]
        metadata = {
            "session_name": session_name,
            "session_directory": str(session_dir),
            "host": self.hostname_provider(),
            "start_time_local": stamp,
            "capture_profile": "raw_rgb" if raw_rgb else "compressed_rgb",
            "max_minutes": max_minutes,
            "required_topics": list(required_topics),
            "selected_topics": selected_topics,
            "missing_optional_topics": report["missing_optional"],
            "coordinate_frame": "FAST-LIO /Odometry in its camera_init local frame",
        }
        (session_dir / "metadata.yaml").write_text(
            yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        command = [
            "timeout",
            "--signal=INT",
            "--kill-after=20s",
            f"{max_minutes * 60.0:g}s",
            "rosbag",
            "record",
            "--lz4",
            "--split",
            "--size=1024",
            "-O",
            str(raw_dir / "flight"),
            *selected_topics,
        ]
        log_path = session_dir / "record.log"
        with log_path.open("ab", buffering=0) as log_file:
            process = self.popen_factory(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.sleep_func(0.5)
        poll = getattr(process, "poll", None)
        return_code = poll() if poll is not None else None
        if return_code is not None:
            raise RuntimeError(
                f"rosbag record exited during startup with code {return_code}; "
                f"inspect {log_path}"
            )

        state = {
            "pid": int(process.pid),
            "pgid": int(process.pid),
            "session_dir": str(session_dir),
            "command": command,
            "start_time_local": stamp,
            "max_minutes": max_minutes,
        }
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return session_dir

    def status(self):
        state = self._read_state()
        if not state:
            return {"active": False, "stale": False}
        active = self.process_exists(int(state["pid"]))
        session_dir = Path(state["session_dir"])
        session_bytes = sum(
            path.stat().st_size for path in session_dir.rglob("*") if path.is_file()
        ) if session_dir.exists() else 0
        free_bytes = int(self.disk_free_provider(self.data_root))
        return {
            "active": active,
            "stale": not active,
            "session_bytes": session_bytes,
            "free_bytes": free_bytes,
            **state,
        }

    def stop(self, timeout_seconds=20.0):
        state = self._read_state()
        if not state:
            return {"stopped": False, "message": "no active recording state"}

        pid = int(state["pid"])
        pgid = int(state["pgid"])
        if not self.process_exists(pid):
            self.state_path.unlink()
            return {
                "stopped": True,
                "already_exited": True,
                "session_dir": state["session_dir"],
                "pid": pid,
            }
        cmdline = self.proc_cmdline_reader(pid)
        if b"rosbag" not in cmdline or b"record" not in cmdline:
            raise RuntimeError(
                f"refusing to signal PID {pid}: process command is not owned rosbag record"
            )

        self.killpg_func(pgid, signal.SIGINT)
        deadline = time.monotonic() + timeout_seconds
        while self.process_exists(pid) and time.monotonic() < deadline:
            self.sleep_func(0.1)
        if self.process_exists(pid):
            raise RuntimeError(
                f"rosbag PID {pid} did not stop within {timeout_seconds:.1f} seconds"
            )

        self.state_path.unlink()
        return {
            "stopped": True,
            "session_dir": state["session_dir"],
            "pid": pid,
        }
