import math
import unittest

from p450_recording.common import (
    interpolate_pose,
    nearest_pose,
    quaternion_to_euler,
    stamp_to_ns,
    validate_session_name,
)


def pose(x, yaw=0.0, frame_id="map", quaternion=None):
    if quaternion is None:
        quaternion = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    return {
        "frame_id": frame_id,
        "x": x, "y": 2.0 * x, "z": -x,
        **dict(zip(("qx", "qy", "qz", "qw"), quaternion)),
    }


class FakeStamp:
    def __init__(self, secs, nsecs):
        self.secs = secs
        self.nsecs = nsecs


class CommonTests(unittest.TestCase):
    def test_validate_session_name_accepts_safe_names(self):
        self.assertEqual(validate_session_name("table-01"), "table-01")
        self.assertEqual(validate_session_name("bench_static"), "bench_static")

    def test_validate_session_name_rejects_paths_and_empty_names(self):
        for value in ("", "../bad", "bad name", "a/b", "."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_session_name(value)

    def test_stamp_to_ns_preserves_integer_nanoseconds(self):
        self.assertEqual(stamp_to_ns(FakeStamp(12, 345)), 12_000_000_345)

    def test_quaternion_to_euler_returns_expected_yaw(self):
        half = math.pi / 4.0
        roll, pitch, yaw = quaternion_to_euler(0.0, 0.0, math.sin(half), math.cos(half))
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, math.pi / 2.0, places=6)

    def test_nearest_pose_selects_nearest_sample(self):
        poses = [(1_000_000_000, "a"), (1_100_000_000, "b")]
        stamp, value, delta_ms, valid = nearest_pose(poses, 1_091_000_000)
        self.assertEqual((stamp, value), (1_100_000_000, "b"))
        self.assertAlmostEqual(delta_ms, 9.0)
        self.assertTrue(valid)

    def test_nearest_pose_handles_boundaries_and_invalid_threshold(self):
        poses = [(1_000_000_000, "a"), (1_100_000_000, "b")]
        self.assertEqual(nearest_pose(poses, 900_000_000, 50.0)[:2], poses[0])
        self.assertFalse(nearest_pose(poses, 900_000_000, 50.0)[3])
        self.assertEqual(nearest_pose(poses, 1_200_000_000, 50.0)[:2], poses[-1])
        self.assertFalse(nearest_pose(poses, 1_200_000_000, 50.0)[3])

    def test_nearest_pose_rejects_empty_sequence(self):
        with self.assertRaises(ValueError):
            nearest_pose([], 1_000_000_000)

    def test_interpolate_pose_linearly_interpolates_position_and_metadata(self):
        result = interpolate_pose(
            [(10_000_000_000, pose(10.0)), (10_200_000_000, pose(20.0))],
            10_102_000_000,
        )
        self.assertAlmostEqual(result["pose"]["x"], 15.1)
        self.assertAlmostEqual(result["pose"]["y"], 30.2)
        self.assertAlmostEqual(result["pose"]["z"], -15.1)
        self.assertEqual(result["pose_stamp_ns"], 10_200_000_000)
        self.assertEqual(result["aligned_pose_stamp_ns"], 10_102_000_000)
        self.assertAlmostEqual(result["delta_ms"], 98.0)
        self.assertEqual(result["alignment_method"], "interpolated")
        self.assertTrue(result["valid"])
        self.assertTrue(result["interp_valid"])
        self.assertEqual(result["prev_pose_stamp_ns"], 10_000_000_000)
        self.assertEqual(result["next_pose_stamp_ns"], 10_200_000_000)
        self.assertAlmostEqual(result["prev_delta_ms"], 102.0)
        self.assertAlmostEqual(result["next_delta_ms"], 98.0)
        self.assertAlmostEqual(result["alpha"], 0.51)

    def test_interpolate_pose_slerps_shortest_path_and_normalizes(self):
        q0 = (0.0, 0.0, 0.0, 2.0)
        q1 = (0.0, 0.0, -2.0, 0.0)
        result = interpolate_pose(
            [(0, pose(0.0, quaternion=q0)), (100_000_000, pose(1.0, quaternion=q1))],
            50_000_000,
        )
        q = result["pose"]
        self.assertAlmostEqual(sum(q[k] ** 2 for k in ("qx", "qy", "qz", "qw")), 1.0)
        self.assertAlmostEqual(abs(q["qz"]), math.sqrt(0.5), places=6)
        self.assertAlmostEqual(abs(q["qw"]), math.sqrt(0.5), places=6)

    def test_interpolate_pose_exact_normalizes_pose(self):
        result = interpolate_pose([(10, pose(1.0, quaternion=(0, 0, 0, 2)))], 10)
        self.assertEqual(result["alignment_method"], "exact")
        self.assertTrue(result["valid"])
        self.assertFalse(result["interp_valid"])
        self.assertEqual(result["alpha"], 0.0)
        self.assertAlmostEqual(result["pose"]["qw"], 1.0)

    def test_interpolate_pose_boundary_near_and_far(self):
        poses = [(100_000_000, pose(1.0)), (200_000_000, pose(2.0))]
        near = interpolate_pose(poses, 60_000_000)
        self.assertEqual(near["alignment_method"], "nearest_boundary")
        self.assertTrue(near["valid"])
        self.assertFalse(near["interp_valid"])
        self.assertEqual(near["pose"], pose(1.0))
        far = interpolate_pose(poses, 40_000_000)
        self.assertEqual(far["alignment_method"], "invalid")
        self.assertFalse(far["valid"])
        self.assertIsNone(far["pose"])

    def test_interpolate_pose_rejects_large_gap_frame_mismatch_and_bad_samples(self):
        cases = [
            ([(0, pose(0)), (300_000_000, pose(1))], 150_000_000),
            ([(0, pose(0, frame_id="a")), (100_000_000, pose(1, frame_id="b"))], 50_000_000),
            ([(0, pose(0)), (100_000_000, pose(1, quaternion=(0, 0, 0, 0)))], 50_000_000),
            ([(0, pose(float("nan"))), (100_000_000, pose(1))], 50_000_000),
        ]
        for poses, target in cases:
            with self.subTest(poses=poses):
                result = interpolate_pose(poses, target)
                self.assertEqual(result["alignment_method"], "invalid")
                self.assertFalse(result["valid"])
                self.assertIsNone(result["pose"])
                self.assertEqual(result["prev_pose_stamp_ns"], poses[0][0])
                self.assertEqual(result["next_pose_stamp_ns"], poses[1][0])
                self.assertAlmostEqual(
                    result["prev_delta_ms"],
                    (target - poses[0][0]) / 1_000_000.0,
                )
                self.assertAlmostEqual(
                    result["next_delta_ms"],
                    (poses[1][0] - target) / 1_000_000.0,
                )
                self.assertAlmostEqual(
                    result["alpha"],
                    (target - poses[0][0]) / float(poses[1][0] - poses[0][0]),
                )

    def test_interpolate_pose_rejects_empty_and_negative_thresholds(self):
        with self.assertRaises(ValueError):
            interpolate_pose([], 0)
        for kwargs in ({"max_delta_ms": -1}, {"max_interp_gap_ms": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                interpolate_pose([(0, pose(0))], 0, **kwargs)

    def test_interpolate_pose_handles_duplicate_timestamps_without_division(self):
        result = interpolate_pose([(0, pose(0)), (0, pose(1)), (100, pose(2))], 0)
        self.assertEqual(result["alignment_method"], "exact")
        self.assertTrue(result["valid"])

    def test_interpolate_pose_uses_valid_duplicate_exact_sample(self):
        invalid = pose(0, quaternion=(0, 0, 0, 0))
        valid = pose(1)
        result = interpolate_pose([(0, invalid), (0, valid), (100, pose(2))], 0)
        self.assertEqual(result["alignment_method"], "exact")
        self.assertTrue(result["valid"])
        self.assertEqual(result["pose"]["x"], 1.0)

        all_invalid = interpolate_pose([(0, invalid), (0, invalid)], 0)
        self.assertEqual(all_invalid["alignment_method"], "invalid")
        self.assertFalse(all_invalid["valid"])
        self.assertIsNone(all_invalid["pose"])

    def test_interpolate_pose_normalizes_extreme_quaternion_magnitudes(self):
        for magnitude in (1e308, 1e-300):
            with self.subTest(magnitude=magnitude):
                result = interpolate_pose(
                    [(0, pose(0, quaternion=(magnitude, 0, 0, magnitude)))],
                    0,
                )
                self.assertTrue(result["valid"])
                self.assertAlmostEqual(result["pose"]["qx"], math.sqrt(0.5))
                self.assertAlmostEqual(result["pose"]["qw"], math.sqrt(0.5))


if __name__ == "__main__":
    unittest.main()
