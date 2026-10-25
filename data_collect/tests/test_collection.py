from __future__ import annotations

import csv
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = load_module("episode_validator", ROOT / "onboard" / "unitree_go2" / "collection" / "validate_episode.py")
converter = load_module("lerobot_converter", ROOT / "server" / "convert_to_lerobot.py")
transfer_validator = load_module(
    "transfer_validator", ROOT / "server" / "validate_transferred_episode.py"
)
replay = load_module("episode_replay", ROOT / "onboard" / "unitree_go2" / "collection" / "replay_episode.py")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class CollectionTests(unittest.TestCase):
    def test_replay_loads_synchronized_pkl_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            (episode / "frames").mkdir()
            for index, height in enumerate((0.1, 0.2, 0.3)):
                timestamp = 1_000_000_000 + index * 100_000_000
                record = {
                    "timestamp_ns": timestamp,
                    "go2": {
                        "sport_mode_state": {
                            "position": [index * 0.1, 0.0, height],
                            "velocity": [0.1, 0.0, 0.0],
                            "rpy": [0.0, 0.0, 0.0],
                            "yaw_speed": 0.0,
                            "body_height": height,
                        }
                    },
                    "piper": {
                        "state": {
                            **{name: index * 0.01 for name in replay.JOINT_NAMES},
                            replay.GRIPPER_NAME: 0.01,
                        }
                    },
                }
                with (episode / "frames" / f"{timestamp}.pkl").open("wb") as handle:
                    pickle.dump(record, handle)

            samples = replay.load_trajectory(episode)
            self.assertEqual([sample.posture for sample in samples], ["down", "up", "up"])
            self.assertAlmostEqual(samples[-1].time_s, 0.2)
            self.assertAlmostEqual(replay.summarize(samples)["path_length_m"], 0.2)

    def test_interpolation_and_frame_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            (episode / "frames").mkdir()
            (episode / "raw").mkdir()
            sport_fields = [
                "wall_time_ns",
                "position_x",
                "position_y",
                "yaw",
                "velocity_x",
                "velocity_y",
                "yaw_speed",
            ]
            write_csv(
                episode / "raw" / "sport_mode_state.csv",
                sport_fields,
                [
                    dict(zip(sport_fields, [1_000_000_000, 0, 0, 0, 0, 0, 0])),
                    dict(zip(sport_fields, [2_000_000_000, 1, 2, 3, 4, 5, 6])),
                ],
            )
            piper_fields = ["wall_time_ns", *converter.PIPER_NAMES]
            write_csv(
                episode / "raw" / "piper_state.csv",
                piper_fields,
                [
                    dict(zip(piper_fields, [1_000_000_000, *([0] * 7)])),
                    dict(zip(piper_fields, [2_000_000_000, *([1] * 7)])),
                ],
            )
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            ok, png = cv2.imencode(".png", image)
            self.assertTrue(ok)
            record = {
                "timestamp_ns": 1_500_000_000,
                "camera": {
                    "front": {"rgb": png.tobytes()},
                    "wrist": {"rgb": png.tobytes()},
                },
            }
            with (episode / "frames" / "1500000000.pkl").open("wb") as handle:
                pickle.dump(record, handle)
            frames = list(converter.episode_frames(episode, 100_000_000))
            self.assertEqual(len(frames), 1)
            np.testing.assert_allclose(frames[0]["observation.state"][:6], [0.5, 1, 1.5, 2, 2.5, 3])
            np.testing.assert_allclose(frames[0]["action"][:3], [2.4, 3, 3.6])
            self.assertEqual(frames[0]["observation.images.front"].shape, (4, 5, 3))

    def test_episode_validator_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            (episode / "frames").mkdir()
            (episode / "raw").mkdir()
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            ok, png = cv2.imencode(".png", image)
            self.assertTrue(ok)
            for index in range(2):
                timestamp = 1_000_000_000 + index * 33_333_333
                record = {
                    "timestamp_ns": timestamp,
                    "camera": {
                        "front": {"serial": "front-serial", "rgb": png.tobytes()},
                        "wrist": {"serial": "wrist-serial", "rgb": png.tobytes()},
                    },
                    "diagnostics": {"camera_skew_ns": 1_000_000},
                }
                with (episode / "frames" / f"{timestamp}.pkl").open("wb") as handle:
                    pickle.dump(record, handle)
            for name in validator.REQUIRED_RAW_FILES:
                rows = [] if name == "wireless_controller.csv" else [
                    {"wall_time_ns": 1_000_000_000, "value": 1},
                    {"wall_time_ns": 1_033_333_333, "value": 2},
                ]
                write_csv(
                    episode / "raw" / name,
                    ["wall_time_ns", "value"],
                    rows,
                )
            config = {
                "fps": 30,
                "max_camera_skew_ms": 50,
                "cameras": {"front": "front-serial", "wrist": "wrist-serial"},
            }
            report = validator.validate(episode, config, write=True)
            self.assertTrue(report["ok"], report)
            self.assertEqual(
                report["warnings"],
                ["raw/wireless_controller.csv has no rows"],
            )
            self.assertEqual(transfer_validator.validate(episode), [])

    def test_transfer_validator_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            data = episode / "data.bin"
            data.write_bytes(b"original")
            manifest = validator.build_manifest(episode)
            (episode / "manifest.json").write_text(json.dumps(manifest))
            data.write_bytes(b"changed")
            errors = transfer_validator.validate(episode)
            self.assertTrue(any("size mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
