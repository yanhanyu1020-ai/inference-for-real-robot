import json
import math
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
from episode_log import (  # noqa: E402
    EpisodeLogger,
    downsample_rows,
    replay_from_csv,
    replay_from_sport_state,
    replay_pose_events_from_body_height,
    write_go2_replay,
)


class EpisodeLogTest(unittest.TestCase):
    def test_downsample_keeps_ends(self):
        rows = [[float(i), 0.0, 0.0] for i in range(10)]
        picked = downsample_rows(rows, max_rows=4)
        self.assertEqual(picked[0], [0.0, 0.0, 0.0])
        self.assertEqual(picked[-1], [9.0, 0.0, 0.0])
        self.assertLessEqual(len(picked), 4)

    def test_attach_write_replay_detach(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = os.path.join(tmp, "ep1")
            os.mkdir(episode)
            pointer = os.path.join(tmp, "active_episode")
            logger = EpisodeLogger(
                pointer,
                cmd_filename="go2_cmd.csv",
                cmd_fields=("vx", "vy", "vyaw"),
                replay_filename="go2_replay.json",
            )
            logger.log([0.3, 0.0, 0.0])
            self.assertFalse(os.path.isfile(os.path.join(episode, "go2_cmd.csv")))

            with open(pointer, "w") as handle:
                handle.write(episode + "\n")
            logger.log([0.3, 0.0, 0.0])
            logger.log([0.0, 0.0, -0.5])
            os.remove(pointer)
            logger.log([0.0, 0.0, 0.0])

            cmd_path = os.path.join(episode, "go2_cmd.csv")
            replay_path = os.path.join(episode, "go2_replay.json")
            self.assertTrue(os.path.isfile(cmd_path))
            self.assertTrue(os.path.isfile(replay_path))
            actions = replay_from_csv(cmd_path, ("vx", "vy", "vyaw"))
            self.assertEqual(len(actions), 2)
            self.assertEqual(actions[0], [0.3, 0.0, 0.0])
            self.assertEqual(actions[1], [0.0, 0.0, -0.5])

    def test_piper_cmd_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = os.path.join(tmp, "ep1")
            os.mkdir(episode)
            pointer = os.path.join(tmp, "active_episode")
            with open(pointer, "w") as handle:
                handle.write(episode + "\n")
            fields = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper_mm")
            logger = EpisodeLogger(
                pointer,
                cmd_filename="piper_cmd.csv",
                cmd_fields=fields,
                state_filename="piper_state.csv",
                state_fields=fields,
                replay_filename="piper_replay.json",
            )
            cmd = [0.0, 10.0, -20.0, 0.0, 15.0, 0.0, 2.0]
            state = [0.1, 9.8, -19.7, 0.0, 14.9, 0.0, 2.1]
            logger.log(cmd, state)
            logger.close()
            self.assertTrue(os.path.isfile(os.path.join(episode, "piper_cmd.csv")))
            self.assertTrue(os.path.isfile(os.path.join(episode, "piper_state.csv")))
            self.assertTrue(os.path.isfile(os.path.join(episode, "piper_replay.json")))
            actions = replay_from_csv(
                os.path.join(episode, "piper_cmd.csv"), fields)
            self.assertEqual(actions[0], cmd)

    def test_replay_from_csv_skips_nul_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "go2_cmd.csv")
            with open(path, "wb") as handle:
                handle.write(b"monotonic_ns,wall_time_ns,vx,vy,vyaw\n")
                handle.write(b"1,2,0.3,0.0,0.0\n")
                handle.write(b"3,4,0.0\x00,0.0,-0.5\n")
            rows = replay_from_csv(path, ("vx", "vy", "vyaw"))
            self.assertEqual(rows[0], [0.3, 0.0, 0.0])
            self.assertTrue(any(row[2] < 0 for row in rows))

    def test_replay_from_state_uses_measured_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = os.path.join(tmp, "ep1")
            os.mkdir(episode)
            pointer = os.path.join(tmp, "active_episode")
            with open(pointer, "w") as handle:
                handle.write(episode + "\n")
            fields = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper_mm")
            logger = EpisodeLogger(
                pointer,
                cmd_filename="piper_cmd.csv",
                cmd_fields=fields,
                state_filename="piper_state.csv",
                state_fields=fields,
                replay_filename="piper_replay.json",
                replay_from_state=True,
            )
            cmd = [0.0, 10.0, -20.0, 0.0, 15.0, 0.0, 2.0]
            state = [1.5, 12.0, -18.0, 0.2, 14.0, -1.0, 40.0]
            logger.log(cmd, state)
            logger.close()
            replay_path = os.path.join(episode, "piper_replay.json")
            with open(replay_path) as handle:
                payload = json.load(handle)
            self.assertEqual(payload["source"], "piper_state.csv")
            self.assertEqual(payload["actions"][0], state)


def _write_sport_csv(path, rows, include_body_height=False):
    fields = (
        "monotonic_ns,velocity_x,velocity_y,yaw_speed,"
        "position_x,position_y,yaw"
    )
    if include_body_height:
        fields += ",body_height"
    with open(path, "w") as handle:
        handle.write(fields + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


class SportReplayTest(unittest.TestCase):
    def test_uses_measured_velocity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sport_mode_state.csv")
            rows = []
            dt_ns = int(0.02 * 1e9)
            for i in range(100):
                t = i * dt_ns
                rows.append((t, 0.4, 0.0, 0.0, 0.4 * i * 0.02, 0.0, 0.0))
            _write_sport_csv(path, rows)
            actions = replay_from_sport_state(path, dt=0.05)
            self.assertGreaterEqual(len(actions), 35)
            self.assertLessEqual(len(actions), 45)
            moving = [row for row in actions if row[0] > 0.2]
            self.assertGreater(len(moving), 20)

    def test_position_fallback_when_velocity_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sport_mode_state.csv")
            rows = []
            dt_ns = int(0.02 * 1e9)
            for i in range(100):
                t = i * dt_ns
                rows.append((t, 0.0, 0.0, 0.0, 0.5 * i * 0.02, 0.0, 0.0))
            _write_sport_csv(path, rows)
            actions = replay_from_sport_state(path, dt=0.05)
            moving = [row for row in actions if row[0] > 0.2]
            self.assertGreater(len(moving), 10)

    def test_write_go2_replay_overwrites_cmd_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "go2_replay.json"), "w") as handle:
                handle.write('{"actions":[[0.35,0,0]],"source":"go2_cmd.csv"}')
            rows = []
            dt_ns = int(0.05 * 1e9)
            for i in range(40):
                t = i * dt_ns
                rows.append((t, 0.6, 0.0, 0.0, 0.6 * i * 0.05, 0.0, 0.0))
            _write_sport_csv(os.path.join(tmp, "sport_mode_state.csv"), rows)
            out = write_go2_replay(tmp)
            self.assertTrue(out)
            with open(out) as handle:
                payload = json.load(handle)
            self.assertEqual(payload["source"], "sport_mode_state")
            self.assertGreater(len(payload["actions"]), 10)
            self.assertGreater(max(row[0] for row in payload["actions"]), 0.4)

    def test_inplace_yaw_integral_matches_heading(self):
        """180° in place must survive even if yaw_speed is 0 or would clamp."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sport_mode_state.csv")
            rows = []
            dt = 0.02
            dt_ns = int(dt * 1e9)
            yaw_target = -math.pi
            n = 60  # ~1.2 s → ~2.6 rad/s, above the old ±1.5 clamp
            for i in range(n):
                t = i * dt_ns
                yaw = yaw_target * i / float(n - 1)
                rows.append((t, 0.0, 0.0, 0.0, 1.0, 2.0, yaw))
            _write_sport_csv(path, rows)
            actions = replay_from_sport_state(path, dt=0.05)
            yaw_int = sum(row[2] for row in actions) * 0.05
            self.assertAlmostEqual(yaw_int, yaw_target, delta=math.radians(5))
            self.assertTrue(any(row[2] < -1.5 for row in actions))

    def test_yaw_unwrap_across_pi(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sport_mode_state.csv")
            rows = []
            dt_ns = int(0.05 * 1e9)
            yaws = [3.0, 3.1, -3.1, -3.0]
            for i, yaw in enumerate(yaws):
                rows.append((i * dt_ns, 0.0, 0.0, 2.5, 0.0, 0.0, yaw))
            _write_sport_csv(path, rows)
            actions = replay_from_sport_state(path, dt=0.05)
            yaw_int = sum(row[2] for row in actions) * 0.05
            self.assertGreater(yaw_int, 0.0)
            self.assertLess(abs(yaw_int), 0.5)

    def test_body_height_pose_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sport_mode_state.csv")
            rows = []
            dt_ns = int(0.05 * 1e9)
            heights = [0.07] * 100 + [0.30] * 120 + [0.07] * 100
            for i, height in enumerate(heights):
                rows.append((i * dt_ns, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, height))
            _write_sport_csv(path, rows, include_body_height=True)
            actions = replay_from_sport_state(path, dt=0.05)
            events = replay_pose_events_from_body_height(path, len(actions), dt=0.05)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["pose"], "stand_up")
            self.assertEqual(events[1]["pose"], "stand_down")
            self.assertLess(events[0]["index"], events[1]["index"])

    def test_write_go2_replay_includes_pose_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            dt_ns = int(0.05 * 1e9)
            for i in range(120):
                t = i * dt_ns
                height = 0.30 if i >= 40 else 0.07
                px = 0.4 * max(0, i - 40) * 0.05
                rows.append((t, 0.4, 0.0, 0.0, px, 0.0, 0.0, height))
            sport_path = os.path.join(tmp, "sport_mode_state.csv")
            _write_sport_csv(sport_path, rows, include_body_height=True)
            out = write_go2_replay(tmp)
            with open(out) as handle:
                payload = json.load(handle)
            self.assertIn("pose_events", payload)
            self.assertEqual(len(payload["pose_events"]), 1)
            self.assertEqual(payload["pose_events"][0]["pose"], "stand_up")
            self.assertEqual(payload["pose_events"][0]["index"], 0)
            self.assertLess(len(payload["actions"]), 120)

    def test_gate_zeros_velocity_while_prone(self):
        from episode_log import gate_go2_actions_for_pose_events

        actions = [[0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0]]
        events = [{"index": 1, "pose": "stand_up"}, {"index": 3, "pose": "stand_down"}]
        gated = gate_go2_actions_for_pose_events(actions, events)
        self.assertEqual(gated[0], [0.0, 0.0, 0.0])
        self.assertEqual(gated[1], [0.3, 0.0, 0.0])
        self.assertEqual(gated[2], [0.3, 0.0, 0.0])
        self.assertEqual(gated[3], [0.0, 0.0, 0.0])

    def test_compact_pose_events_keeps_main_motion_cluster(self):
        from episode_log import compact_pose_events_for_replay

        actions = [[0.0, 0.0, 0.0]] * 220
        actions[10] = [0.2, 0.0, 0.0]
        actions[11] = [0.2, 0.0, 0.0]
        actions[150] = [0.3, 0.0, 0.5]
        for i in range(151, 220):
            actions[i] = [0.3, 0.0, 0.5]
        events = compact_pose_events_for_replay(actions)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["pose"], "stand_up")
        self.assertEqual(events[0]["index"], 150)

    def test_height_and_pose_keep_stand_down_after_walk(self):
        from episode_log import compact_pose_events_for_replay

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sport_mode_state.csv")
            rows = []
            dt_ns = int(0.05 * 1e9)
            # In-place test squat, then stand + walk, then sit.
            for i in range(400):
                t = i * dt_ns
                if i < 80:
                    height, vx, px = 0.07, 0.0, 0.0
                elif i < 140:
                    height, vx, px = 0.30, 0.0, 0.0
                elif i < 200:
                    height, vx, px = 0.07, 0.0, 0.0
                elif i < 320:
                    height, vx, px = 0.30, 0.4, 0.4 * (i - 200) * 0.05
                else:
                    height, vx, px = 0.07, 0.0, 0.4 * 120 * 0.05
                rows.append((t, vx, 0.0, 0.0, px, 0.0, 0.0, height))
            _write_sport_csv(path, rows, include_body_height=True)
            from episode_log import replay_from_sport_state

            actions = replay_from_sport_state(path, dt=0.05)
            events = compact_pose_events_for_replay(actions, sport_path=path)
            poses = [item["pose"] for item in events]
            self.assertEqual(
                poses, ["stand_up", "stand_down", "stand_up", "stand_down"])
            self.assertLess(events[0]["index"], 120)
            self.assertGreater(events[2]["index"], events[1]["index"])
            self.assertGreater(events[3]["index"], events[2]["index"])

    def test_trim_drops_leading_prone_and_trailing_idle(self):
        from episode_log import trim_go2_replay_idle

        actions = [[0.0, 0.0, 0.0]] * 80
        for i in range(50, 60):
            actions[i] = [0.3, 0.0, 0.0]
        events = [{"index": 50, "pose": "stand_up"}]
        trimmed, shifted = trim_go2_replay_idle(actions, events, trailing_rows=4)
        self.assertEqual(shifted, [{"index": 0, "pose": "stand_up"}])
        self.assertEqual(trimmed[0], [0.3, 0.0, 0.0])
        self.assertEqual(len(trimmed), 14)

    def test_build_keeps_both_squat_cycles(self):
        from episode_log import build_go2_replay_from_sport

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sport_mode_state.csv")
            rows = []
            dt_ns = int(0.05 * 1e9)
            for i in range(400):
                t = i * dt_ns
                if i < 80:
                    height, vx, px = 0.07, 0.0, 0.0
                elif i < 140:
                    height, vx, px = 0.30, 0.0, 0.0
                elif i < 200:
                    height, vx, px = 0.07, 0.0, 0.0
                elif i < 320:
                    height, vx, px = 0.30, 0.4, 0.4 * (i - 200) * 0.05
                else:
                    height, vx, px = 0.07, 0.0, 0.4 * 120 * 0.05
                rows.append((t, vx, 0.0, 0.0, px, 0.0, 0.0, height))
            _write_sport_csv(path, rows, include_body_height=True)
            actions, events = build_go2_replay_from_sport(path, dt=0.05)
            poses = [item["pose"] for item in events]
            self.assertEqual(
                poses, ["stand_up", "stand_down", "stand_up", "stand_down"])
            self.assertEqual(events[0]["index"], 0)
            self.assertGreater(events[2]["index"], events[1]["index"])
            moving = [
                i for i, row in enumerate(actions)
                if abs(row[0]) > 0.05 or abs(row[1]) > 0.05 or abs(row[2]) > 0.05
            ]
            self.assertTrue(moving)
            self.assertGreater(min(moving), events[1]["index"])
            self.assertLessEqual(max(moving), events[3]["index"])


if __name__ == "__main__":
    unittest.main()
