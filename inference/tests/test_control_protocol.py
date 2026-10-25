import json
import unittest
from pathlib import Path
import sys

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFERENCE_ROOT))

from heterovla.control_protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode,
    encode,
    validate_command,
    validate_registration,
)


class ProtocolTest(unittest.TestCase):
    def test_round_trip_unicode(self):
        message = {"type": "result", "message": "已站立"}
        self.assertEqual(decode(encode(message)), message)

    def test_robot_registration(self):
        message = {"type": "register", "protocol": PROTOCOL_VERSION,
                   "role": "robot", "robot_id": "unitree-go2-01"}
        self.assertEqual(validate_registration(message), message)

    def test_forbids_unknown_command(self):
        with self.assertRaises(ProtocolError):
            validate_command({"type": "command", "id": "1", "robot_id": "r1",
                              "command": "go2.raw_torque", "arguments": {}})

    def test_forbids_arguments_in_v1(self):
        with self.assertRaises(ProtocolError):
            validate_command({"type": "command", "id": "1", "robot_id": "r1",
                              "command": "go2.stand_up", "arguments": {"speed": 99}})

    def test_rejects_unvalidated_raw_command(self):
        with self.assertRaises(ProtocolError):
            validate_command({"type": "command", "id": "1", "robot_id": "r1",
                              "command": "piper.raw_torque", "arguments": {}})

    def test_accepts_gripper_commands(self):
        for command in ("piper.open_gripper", "piper.close_gripper"):
            validate_command({"type": "command", "id": "1", "robot_id": "r1",
                              "command": command, "arguments": {}})

    def test_accepts_action_chunk(self):
        validate_command({"type": "command", "id": "1", "robot_id": "r1",
                          "command": "piper.action_chunk",
                          "arguments": {"actions": [[0, 5, -10, 0, 15, 0, 2]]}})

    def test_rejects_bad_action_chunk(self):
        with self.assertRaises(ProtocolError):
            validate_command({"type": "command", "id": "1", "robot_id": "r1",
                              "command": "piper.action_chunk",
                              "arguments": {"actions": [[0, 5, -10, 0, 15]]}})
        with self.assertRaises(ProtocolError):
            validate_command({"type": "command", "id": "1", "robot_id": "r1",
                              "command": "piper.action_chunk",
                              "arguments": {"actions": [["a", 5, -10, 0, 15, 0, 2]]}})

    def test_accepts_go2_pose_fields(self):
        validate_command({
            "type": "command", "id": "1", "robot_id": "r1",
            "command": "go2.action_chunk",
            "arguments": {
                "actions": [[0.0, 0.0, 0.0]],
                "before": "stand_up",
                "after": "stand_down",
            },
        })

    def test_accepts_go2_pose_events(self):
        validate_command({
            "type": "command", "id": "1", "robot_id": "r1",
            "command": "go2.action_chunk",
            "arguments": {
                "actions": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                "pose_events": [{"index": 1, "pose": "stand_up"}],
            },
        })

    def test_rejects_bad_go2_pose_fields(self):
        with self.assertRaises(ProtocolError):
            validate_command({
                "type": "command", "id": "1", "robot_id": "r1",
                "command": "go2.action_chunk",
                "arguments": {"actions": [[0.0, 0.0, 0.0]], "before": "dance"},
            })
        validate_command({"type": "command", "id": "1", "robot_id": "r1",
                          "command": "go2.action_chunk",
                          "arguments": {"actions": [[0.3, 0.0, 0.0], [0.0, 0.0, 0.5]]}})

    def test_rejects_bad_go2_action_chunk(self):
        with self.assertRaises(ProtocolError):
            validate_command({"type": "command", "id": "1", "robot_id": "r1",
                              "command": "go2.action_chunk",
                              "arguments": {"actions": [[0.3, 0.0]]}})
        with self.assertRaises(ProtocolError):
            validate_command({"type": "command", "id": "1", "robot_id": "r1",
                              "command": "go2.action_chunk",
                              "arguments": {"actions": [[0.3, 0.0, "x"]]}})

    def test_accepts_record_start(self):
        validate_command({
            "type": "command", "id": "1", "robot_id": "r1",
            "command": "system.record_start",
            "arguments": {
                "episode_id": "demo_001",
                "instruction": "hetero demo",
            },
        })
        validate_command({
            "type": "command", "id": "1", "robot_id": "r1",
            "command": "system.record_stop",
            "arguments": {},
        })

    def test_accepts_teleop_commands(self):
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

    def test_rejects_bad_record_start(self):
        with self.assertRaises(ProtocolError):
            validate_command({
                "type": "command", "id": "1", "robot_id": "r1",
                "command": "system.record_start",
                "arguments": {"episode_id": "../etc"},
            })
        with self.assertRaises(ProtocolError):
            validate_command({
                "type": "command", "id": "1", "robot_id": "r1",
                "command": "system.record_start",
                "arguments": {},
            })

    def test_rejects_non_object_json(self):
        with self.assertRaises(ProtocolError):
            decode(json.dumps(["command"]))


if __name__ == "__main__":
    unittest.main()
