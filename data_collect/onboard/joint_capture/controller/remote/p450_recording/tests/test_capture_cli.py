import io
import json
import sys
import unittest
from unittest import mock

sys.modules.setdefault("cv2", mock.Mock())
sys.modules.setdefault("rosbag", mock.Mock())
sys.modules.setdefault("cv_bridge", mock.Mock(CvBridge=mock.Mock))

from p450_recording import capture_cli


class CaptureCliTests(unittest.TestCase):
    def invoke(self, arguments, orchestrator):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            capture_cli, "CaptureOrchestrator", return_value=orchestrator
        ):
            code = capture_cli.main(arguments, stdout=stdout, stderr=stderr)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_start_returns_session_json(self):
        orchestrator = mock.Mock()
        orchestrator.start.return_value = {
            "started": True,
            "session_dir": "/data/demo",
        }
        code, stdout, stderr = self.invoke(["start", "demo"], orchestrator)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["session_dir"], "/data/demo")
        self.assertEqual(stderr, "")

    def test_prepare_returns_stack_json(self):
        orchestrator = mock.Mock()
        orchestrator.prepare.return_value = {
            "prepared": True,
            "started_components": ["mid360_driver"],
        }
        code, stdout, stderr = self.invoke(["prepare"], orchestrator)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout)["prepared"])
        self.assertEqual(stderr, "")

    def test_finish_returns_export_json(self):
        orchestrator = mock.Mock()
        orchestrator.finish.return_value = {
            "finished": True,
            "frame_count": 42,
            "valid_alignment_percent": 100.0,
        }
        code, stdout, _ = self.invoke(["finish"], orchestrator)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["frame_count"], 42)

    def test_stop_fast_does_not_export(self):
        orchestrator = mock.Mock()
        orchestrator.stop_fast.return_value = {"stopped": True, "postprocess": "pending"}
        code, stdout, _ = self.invoke(["stop-fast"], orchestrator)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout)["stopped"])
        orchestrator.export.assert_not_called()

    def test_manual_export_takes_session_directory(self):
        orchestrator = mock.Mock()
        orchestrator.export.return_value = {"video_path": "/data/demo/export/rgb.mp4"}
        code, stdout, _ = self.invoke(["export", "/data/demo"], orchestrator)
        self.assertEqual(code, 0)
        orchestrator.export.assert_called_once_with("/data/demo")
        self.assertEqual(json.loads(stdout)["video_path"], "/data/demo/export/rgb.mp4")

    def test_status_returns_orchestrator_status(self):
        orchestrator = mock.Mock()
        orchestrator.status.return_value = {"recorder": {"active": False}}
        code, stdout, _ = self.invoke(["status"], orchestrator)
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(stdout)["recorder"]["active"])

    def test_shutdown_returns_stopped_components(self):
        orchestrator = mock.Mock()
        orchestrator.shutdown.return_value = {"stopped_components": ["d435i"]}
        code, stdout, _ = self.invoke(["shutdown"], orchestrator)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["stopped_components"], ["d435i"])

    def test_expected_error_has_no_traceback(self):
        orchestrator = mock.Mock()
        orchestrator.start.side_effect = RuntimeError("Odometry is inactive")
        code, stdout, stderr = self.invoke(["start", "demo"], orchestrator)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "error: Odometry is inactive\n")


if __name__ == "__main__":
    unittest.main()
