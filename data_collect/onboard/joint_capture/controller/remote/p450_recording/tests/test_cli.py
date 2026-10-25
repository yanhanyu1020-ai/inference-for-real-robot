import io
import json
import unittest
from unittest import mock

from p450_recording import cli


class CliTests(unittest.TestCase):
    def invoke(self, arguments, manager):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(cli, "RecorderManager", return_value=manager):
            code = cli.main(arguments, stdout=stdout, stderr=stderr)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_check_returns_zero_and_json_when_ready(self):
        manager = mock.Mock()
        manager.check.return_value = {
            "ok": True,
            "errors": [],
            "missing_required": [],
            "missing_optional": [],
        }
        code, stdout, stderr = self.invoke(["check"], manager)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout)["ok"])
        self.assertEqual(stderr, "")

    def test_check_returns_two_when_not_ready(self):
        manager = mock.Mock()
        manager.check.return_value = {
            "ok": False,
            "errors": ["missing required topics"],
        }
        code, stdout, _ = self.invoke(["check"], manager)
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(stdout)["ok"])

    def test_start_prints_session_directory(self):
        manager = mock.Mock()
        manager.start.return_value = "/home/amov/p450_data/session"
        code, stdout, _ = self.invoke(["start", "demo_01"], manager)
        self.assertEqual(code, 0)
        manager.start.assert_called_once_with("demo_01")
        self.assertEqual(
            json.loads(stdout)["session_dir"], "/home/amov/p450_data/session"
        )

    def test_stop_outputs_manager_result(self):
        manager = mock.Mock()
        manager.stop.return_value = {"stopped": True, "pid": 123}
        code, stdout, _ = self.invoke(["stop"], manager)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout)["stopped"])

    def test_export_outputs_summary(self):
        manager = mock.Mock()
        summary = {
            "frame_count": 10,
            "pose_count": 50,
            "video_path": "/data/export/rgb.mp4",
        }
        with mock.patch.object(cli, "export_session", return_value=summary) as export:
            code, stdout, _ = self.invoke(
                [
                    "export",
                    "/data/session",
                    "--max-delta-ms",
                    "25",
                    "--max-interp-gap-ms",
                    "175",
                ],
                manager,
            )
        export.assert_called_once_with(
            "/data/session", max_delta_ms=25.0, max_interp_gap_ms=175.0
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["frame_count"], 10)

    def test_export_passes_default_interpolation_gap(self):
        manager = mock.Mock()
        with mock.patch.object(cli, "export_session", return_value={}) as export:
            code, _, _ = self.invoke(["export", "/data/session"], manager)
        export.assert_called_once_with(
            "/data/session", max_delta_ms=50.0, max_interp_gap_ms=200.0
        )
        self.assertEqual(code, 0)

    def test_runtime_error_is_reported_without_traceback(self):
        manager = mock.Mock()
        manager.start.side_effect = RuntimeError("camera topic is missing")
        code, stdout, stderr = self.invoke(["start", "demo"], manager)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("camera topic is missing", stderr)


if __name__ == "__main__":
    unittest.main()
