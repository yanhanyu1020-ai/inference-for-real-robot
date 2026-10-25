import math
import os
import sys
import unittest
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFERENCE_ROOT))

from heterovla.control_protocol import validate_command

sys.path.insert(0, str(INFERENCE_ROOT / "scripts"))
import send_hetero_demo as demo  # noqa: E402


class HeteroDemoTrajectoryTest(unittest.TestCase):
    def test_dog_right_turn_open_loop(self):
        rows = demo.dog_approach_trajectory(
            forward_s=2.0, vx=0.3, yaw_deg=-90.0, yaw_rate=0.5, dt=0.05)
        self.assertTrue(1 <= len(rows) <= 5000)
        self.assertEqual(rows[0], [0.3, 0.0, 0.0])
        self.assertEqual(rows[-1], [0.0, 0.0, 0.0])
        yaw_rows = [row for row in rows if row[2] != 0.0]
        self.assertTrue(yaw_rows)
        self.assertTrue(all(row[2] < 0.0 for row in yaw_rows))
        yaw_integral = sum(row[2] for row in rows) * 0.05
        self.assertAlmostEqual(yaw_integral, -math.pi / 2, places=2)
        validate_command({
            "type": "command", "id": "1", "robot_id": "r1",
            "command": "go2.action_chunk",
            "arguments": {"actions": rows, "after": "stand_down"},
        })

    def test_scripted_grasp_fits_protocol(self):
        rows = demo.default_grasp_trajectory(200)
        self.assertEqual(len(rows), 1000)
        self.assertTrue(all(len(row) == 7 for row in rows))
        self.assertGreaterEqual(min(row[6] for row in rows), 1.0)
        # Gripper stays closed until the arm has reached hover.
        self.assertLess(rows[50][6], 10.0)
        self.assertGreater(rows[550][6], 40.0)
        validate_command({
            "type": "command", "id": "1", "robot_id": "r1",
            "command": "piper.action_chunk", "arguments": {"actions": rows},
        })

    def test_chunk_receipt_is_not_lie_down(self):
        receipt = (
            "bridge_pid=1 chunk_pid=2\n"
            "--- go2_chunk.log ---\n"
            "2026-08-13 13:00:00 INFO chunk received: 80 rows "
            "before=None after=stand_down\n"
        )
        fresh = demo.new_log_text(receipt, baseline="")
        self.assertTrue("after=stand_down" in fresh)
        self.assertFalse(demo.log_has_any(fresh, demo.LIE_DOWN_STARTED))
        self.assertFalse(demo.log_has_any(fresh, demo.LIE_DOWN_DONE))

    def test_lie_down_waits_for_new_after_pose(self):
        baseline = (
            "bridge_pid=1 chunk_pid=2\n"
            "--- go2_chunk.log ---\n"
            "2026-08-13 12:00:00 INFO after: STAND_DOWN\n"
            "2026-08-13 12:00:04 INFO chunk done; STOP issued\n"
        )
        walking = (
            "bridge_pid=1 chunk_pid=2\n"
            "--- go2_chunk.log ---\n"
            "2026-08-13 13:00:00 INFO chunk received: 80 rows "
            "before=None after=stand_down\n"
        )
        started = walking + "2026-08-13 13:00:08 INFO after: STAND_DOWN\n"
        done = started + "2026-08-13 13:00:12 INFO after pose complete: STAND_DOWN\n"
        self.assertFalse(demo.log_has_any(
            demo.new_log_text(walking, baseline), demo.LIE_DOWN_STARTED))
        self.assertTrue(demo.log_has_any(
            demo.new_log_text(started, baseline), demo.LIE_DOWN_STARTED))
        self.assertTrue(demo.log_has_any(
            demo.new_log_text(done, baseline), demo.LIE_DOWN_DONE))


class RobotctlPayloadTest(unittest.TestCase):
    def test_teleop_arm_replay_exceeds_linux_arg_strlen(self):
        path = str(INFERENCE_ROOT.parent / "piper_replay.json")
        if not os.path.isfile(path):
            self.skipTest("piper_replay.json not in repo root")
        self.assertGreater(os.path.getsize(path), 131072)


if __name__ == "__main__":
    unittest.main()
