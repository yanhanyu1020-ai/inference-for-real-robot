import os
import sys
import tempfile
import unittest
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFERENCE_ROOT))

sys.path.insert(
    0,
    str(INFERENCE_ROOT / "onboard" / "unitree_go2" / "control"),
)
import hetero_teleop_loop as teleop  # noqa: E402

sys.path.insert(0, str(INFERENCE_ROOT / "scripts"))
import send_hetero_demo as demo  # noqa: E402

from heterovla.control_protocol import ProtocolError, validate_command  # noqa: E402


class TeleopMappingTest(unittest.TestCase):
    def test_deadzone_zeros_small_stick(self):
        self.assertEqual(teleop.apply_deadzone(0.05, 0.08), 0.0)
        self.assertGreater(teleop.apply_deadzone(1.0, 0.08), 0.9)

    def test_left_stick_forward_and_yaw(self):
        vx, vy, vyaw = teleop.stick_to_dog(0.0, 1.0, 0.08, 0.35, 0.7)
        self.assertGreater(vx, 0.3)
        self.assertEqual(vy, 0.0)
        self.assertEqual(vyaw, 0.0)
        vx, vy, vyaw = teleop.stick_to_dog(1.0, 0.0, 0.08, 0.35, 0.7)
        self.assertEqual(vx, 0.0)
        self.assertLess(vyaw, -0.6)

    def test_right_stick_default_moves_j1_j2(self):
        start = [0.0, 10.0, -20.0, 0.0, 0.0, 0.0, 2.0]
        moved = teleop.integrate_arm(
            start, rx=1.0, ry=1.0, keys=0, dt=0.1,
            deadzone=0.08, joint_rate=40.0, grip_rate=40.0,
        )
        self.assertGreater(moved[0], start[0])
        self.assertGreater(moved[1], start[1])
        self.assertEqual(moved[2], start[2])

    def test_l1_bank_moves_j3_j4(self):
        start = [0.0, 10.0, -20.0, 0.0, 0.0, 0.0, 2.0]
        moved = teleop.integrate_arm(
            start, rx=1.0, ry=-1.0, keys=teleop.KEY_L1, dt=0.1,
            deadzone=0.08, joint_rate=40.0, grip_rate=40.0,
        )
        self.assertEqual(moved[0], start[0])
        self.assertGreater(moved[2], start[2])
        self.assertLess(moved[3], start[3])

    def test_remote_stand_down_combo(self):
        self.assertTrue(teleop.key_combo_active(
            teleop.KEY_L2 | teleop.KEY_A, teleop.REMOTE_STAND_DOWN))
        self.assertFalse(teleop.key_combo_active(teleop.KEY_L2, teleop.REMOTE_STAND_DOWN))
        self.assertTrue(teleop.key_combo_rising(
            teleop.KEY_L2, teleop.KEY_L2 | teleop.KEY_A, teleop.REMOTE_STAND_DOWN))
        self.assertFalse(teleop.key_combo_rising(
            teleop.KEY_L2 | teleop.KEY_A,
            teleop.KEY_L2 | teleop.KEY_A,
            teleop.REMOTE_STAND_DOWN))

    def test_remote_pose_action_toggles(self):
        tracker = teleop.RemotePoseTracker()
        down = tracker.feed(teleop.KEY_L2 | teleop.KEY_A, 1.0, last_cmd=0.0)
        self.assertEqual(down, "stand_down")
        self.assertTrue(tracker.dog_prone)
        self.assertIsNone(tracker.feed(teleop.KEY_L2 | teleop.KEY_A, 1.1, last_cmd=0.0))
        self.assertIsNone(tracker.feed(teleop.KEY_L2, 1.2, last_cmd=0.0))
        up = tracker.feed(teleop.KEY_L2 | teleop.KEY_A, 2.0, last_cmd=0.0)
        self.assertEqual(up, "stand_up")
        self.assertFalse(tracker.dog_prone)
        tracker.dog_prone = True
        up = tracker.feed(teleop.KEY_START, 3.0, last_cmd=0.0)
        self.assertEqual(up, "stand_up")

    def test_parse_wireless_csv_since(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wireless_controller.csv")
            with open(path, "w") as handle:
                handle.write("monotonic_ns,wall_time_ns,seq,lx,ly,rx,ry,keys\n")
                handle.write("1,2,0,0,0,0,0,32\n")
                handle.write("3,4,1,0,0,0,0,288\n")
                handle.write("5,6,2,0,0,0,0,32\n")
            rows, seq = teleop.parse_wireless_csv_since(path, -1)
            self.assertEqual(len(rows), 3)
            self.assertEqual(seq, 2)
            more, seq2 = teleop.parse_wireless_csv_since(path, seq)
            self.assertEqual(more, [])
            self.assertEqual(seq2, 2)

    def test_gripper_triggers(self):
        start = [0.0, 10.0, -20.0, 0.0, 0.0, 0.0, 2.0]
        opened = teleop.integrate_arm(
            start, 0.0, 0.0, teleop.KEY_R1, 0.5, 0.08, 40.0, 40.0)
        closed = teleop.integrate_arm(
            opened, 0.0, 0.0, teleop.KEY_R2, 0.5, 0.08, 40.0, 40.0)
        self.assertGreater(opened[6], start[6])
        self.assertLess(closed[6], opened[6])

    def test_joint_limits(self):
        start = [0.0, 179.0, -1.0, 0.0, 0.0, 0.0, 2.0]
        moved = teleop.integrate_arm(
            start, 0.0, 1.0, 0, 1.0, 0.0, 40.0, 40.0)
        self.assertLessEqual(moved[1], 180.0)

    def test_stick_file_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stick.json")
            with open(path, "w") as handle:
                handle.write('{"lx":0.2,"ly":0.9,"rx":-0.1,"ry":0.0,"keys":1}')
            reader = teleop.StickFileReader(path).start()
            state = reader.read()
            self.assertTrue(state.ok)
            self.assertAlmostEqual(state.ly, 0.9)
            self.assertEqual(state.keys, 1)

    def test_wireless_csv_tail_skips_header_and_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wireless_controller.csv")
            with open(path, "w") as handle:
                handle.write("monotonic_ns,wall_time_ns,seq,lx,ly,rx,ry,keys\n")
                handle.write("1,2,0,0.1,0.2,-0.3,0.4,8\n")
                handle.write("3,4,1,0.5,0.9,0.0,-0.1,2\n")
                handle.write("partial")
            payload = teleop.parse_wireless_csv_tail(path)
            self.assertAlmostEqual(payload["ly"], 0.9)
            self.assertEqual(payload["keys"], 2)

    def test_episode_stick_prefers_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = os.path.join(tmp, "ep")
            os.mkdir(episode)
            pointer = os.path.join(tmp, "active_episode")
            with open(pointer, "w") as handle:
                handle.write(episode + "\n")
            with open(os.path.join(episode, "stick.json"), "w") as handle:
                handle.write('{"lx":0.0,"ly":1.0,"rx":0.0,"ry":0.0,"keys":0}')
            with open(os.path.join(episode, "wireless_controller.csv"), "w") as handle:
                handle.write("monotonic_ns,wall_time_ns,seq,lx,ly,rx,ry,keys\n")
                handle.write("1,2,0,9,9,9,9,9\n")
            state = teleop.EpisodeStickReader(pointer).start().read()
            self.assertTrue(state.ok)
            self.assertAlmostEqual(state.ly, 1.0)
            self.assertTrue(state.source.startswith("json:"))


class TeleopReplayHelpersTest(unittest.TestCase):
    def test_split_rows_respects_protocol_limit(self):
        rows = [[float(i), 0.0, 0.0] for i in range(5001)]
        pieces = demo.split_rows(rows)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(len(pieces[0]), 5000)
        self.assertEqual(len(pieces[1]), 1)
        validate_command({
            "type": "command", "id": "1", "robot_id": "r1",
            "command": "go2.action_chunk",
            "arguments": {"actions": pieces[0]},
        })

    def test_teach_ctrl_mode(self):
        self.assertTrue(teleop.is_teach_ctrl_mode(2))
        self.assertTrue(teleop.is_teach_ctrl_mode("2"))
        self.assertFalse(teleop.is_teach_ctrl_mode(1))
        self.assertFalse(teleop.is_teach_ctrl_mode(None))
        self.assertEqual(teleop.ctrl_mode_name(2), "teach")
        self.assertEqual(teleop.ctrl_mode_name(1), "can")
        self.assertTrue(teleop.ctrl_mode_name(99).startswith("unknown"))

    def test_drag_teach_active(self):
        self.assertTrue(teleop.is_drag_teach_active(2, 0, 0))
        self.assertTrue(teleop.is_drag_teach_active(6, 0, 0))
        self.assertTrue(teleop.is_drag_teach_active(1, 1, 0))
        self.assertTrue(teleop.is_drag_teach_active(0, 0, 0x0B))
        self.assertFalse(teleop.is_drag_teach_active(0, 0, 0))
        self.assertFalse(teleop.is_drag_teach_active(1, 0, 0))
        self.assertFalse(teleop.is_drag_teach_active(0, 1, 1))
        self.assertEqual(teleop.arm_status_name(6), "brake")
        self.assertEqual(teleop.teach_status_name(1), "recording")

    def test_teleop_commands_are_allowlisted(self):
        for command in (
            "system.teleop_start",
            "system.teleop_stop",
            "system.teleop_status",
            "system.teach_start",
            "system.teach_stop",
            "system.teach_status",
        ):
            validate_command({
                "type": "command", "id": "1", "robot_id": "r1",
                "command": command, "arguments": {},
            })
        with self.assertRaises(ProtocolError):
            validate_command({
                "type": "command", "id": "1", "robot_id": "r1",
                "command": "system.teleop_start",
                "arguments": {"speed": 1},
            })


if __name__ == "__main__":
    unittest.main()
