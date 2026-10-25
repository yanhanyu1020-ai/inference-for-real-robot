import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from p450_recording.common import OPTIONAL_TOPICS, REQUIRED_TOPICS
from p450_recording.manager import RecorderManager


class FakeProcess:
    def __init__(self, pid=4321):
        self.pid = pid


class ManagerTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1] / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.temp_dir.name)
        self.topics = {**REQUIRED_TOPICS, **OPTIONAL_TOPICS}
        self.popen = mock.Mock(return_value=FakeProcess())
        self.manager = RecorderManager(
            data_root=self.root,
            topic_provider=lambda: self.topics,
            disk_free_provider=lambda _: 20 * 1024**3,
            popen_factory=self.popen,
            now_provider=lambda: "20260725_120000",
            hostname_provider=lambda: "amov",
            proc_cmdline_reader=lambda _: b"python3\x00rosbag\x00record\x00",
            killpg_func=mock.Mock(),
            process_exists=lambda _: True,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_rejects_low_disk_space(self):
        manager = RecorderManager(
            data_root=self.root,
            topic_provider=lambda: self.topics,
            disk_free_provider=lambda _: 9 * 1024**3,
        )
        result = manager.check()
        self.assertFalse(result["ok"])
        self.assertIn("disk space", result["errors"][0])

    def test_check_rejects_missing_required_topics(self):
        topics = dict(self.topics)
        topics.pop("/uav1/camera/color/image_raw/compressed")
        self.manager.topic_provider = lambda: topics
        result = self.manager.check()
        self.assertFalse(result["ok"])
        self.assertIn("/uav1/camera/color/image_raw/compressed", result["missing_required"])

    def test_check_rejects_required_topic_without_messages(self):
        active = set(REQUIRED_TOPICS)
        active.remove("/Odometry")
        self.manager.activity_provider = lambda _: active
        result = self.manager.check()
        self.assertFalse(result["ok"])
        self.assertIn("/Odometry", result["inactive_required"])
        self.assertTrue(any("not publishing messages" in error for error in result["errors"]))

    def test_start_allows_missing_optional_topics_and_records_metadata(self):
        self.manager.topic_provider = lambda: dict(REQUIRED_TOPICS)
        session = self.manager.start("bench_static")
        metadata = (session / "metadata.yaml").read_text(encoding="utf-8")
        self.assertIn("/uav1/mavros/imu/data", metadata)
        command = self.popen.call_args.args[0]
        self.assertIn("--lz4", command)
        self.assertIn("--split", command)
        self.assertIn("/uav1/camera/color/image_raw/compressed", command)
        self.assertIn("/Odometry", command)
        self.assertNotIn("/uav1/mavros/imu/data", command)

    def test_joint_session_uses_desktop_id_without_stale_robot_clock_prefix(self):
        session_id = 'joint_20260926_143015_UTCp0800_ab12'
        session = self.manager.start(session_id)
        self.assertEqual(session.name, session_id)
        self.assertEqual(session.parent, self.root)

    def test_start_refuses_second_active_recorder(self):
        self.manager.start("first")
        with self.assertRaises(RuntimeError):
            self.manager.start("second")

    def test_stop_rejects_pid_not_owned_by_rosbag(self):
        self.manager.start("bench")
        self.manager.proc_cmdline_reader = lambda _: b"python3\x00other_process.py\x00"
        with self.assertRaises(RuntimeError):
            self.manager.stop()
        self.manager.killpg_func.assert_not_called()

    def test_stop_sends_sigint_to_owned_process_group(self):
        session = self.manager.start("bench")
        state = json.loads((self.root / ".recording_state.json").read_text())
        self.manager.process_exists = mock.Mock(side_effect=[True, False, False])
        result = self.manager.stop()
        self.manager.killpg_func.assert_called_once_with(state["pgid"], signal.SIGINT)
        self.assertEqual(result["session_dir"], str(session))
        self.assertFalse((self.root / ".recording_state.json").exists())

    def test_status_marks_stale_state(self):
        self.manager.start("bench")
        self.manager.process_exists = lambda _: False
        result = self.manager.status()
        self.assertFalse(result["active"])
        self.assertTrue(result["stale"])


if __name__ == "__main__":
    unittest.main()
