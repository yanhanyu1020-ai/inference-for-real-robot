import csv
import tempfile
import unittest
from pathlib import Path

import rosbag
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from p450_recording.exporter import export_session


POSE_TOPIC = "/Odometry"
IMAGE_TOPIC = "/uav1/camera/color/image_raw"


def make_pose(stamp_seconds, x):
    message = Odometry()
    message.header.stamp = rospy.Time.from_sec(stamp_seconds)
    message.header.frame_id = "camera_init"
    message.child_frame_id = "body"
    message.pose.pose.position.x = x
    message.pose.pose.orientation.w = 1.0
    return message


def make_image(stamp_seconds, value):
    message = Image()
    message.header.stamp = rospy.Time.from_sec(stamp_seconds)
    message.height = 2
    message.width = 2
    message.encoding = "bgr8"
    message.is_bigendian = False
    message.step = 6
    message.data = bytes([value, value, value] * 4)
    return message


class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session = Path(self.temp_dir.name)
        raw_dir = self.session / "raw"
        raw_dir.mkdir()
        bag_path = raw_dir / "flight_0.bag"
        with rosbag.Bag(str(bag_path), "w") as bag:
            for stamp, x in [(1.0, 10.0), (1.1, 11.0)]:
                message = make_pose(stamp, x)
                bag.write(POSE_TOPIC, message, message.header.stamp)
            for stamp, value in [(1.02, 30), (1.09, 60)]:
                message = make_image(stamp, value)
                bag.write(IMAGE_TOPIC, message, message.header.stamp)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_writes_video_pose_and_interpolated_pose_alignment(self):
        result = export_session(self.session, max_delta_ms=50.0)
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["pose_count"], 2)
        self.assertEqual(result["max_interp_gap_ms"], 200.0)
        self.assertTrue(Path(result["video_path"]).stat().st_size > 0)

        with Path(result["alignment_csv"]).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["valid"] for row in rows], ["1", "1"])
        self.assertEqual([row["interp_valid"] for row in rows], ["1", "1"])
        self.assertEqual(
            [row["alignment_method"] for row in rows],
            ["interpolated", "interpolated"],
        )
        self.assertEqual([float(row["x"]) for row in rows], [10.2, 10.9])
        self.assertEqual([row["frame_id"] for row in rows], ["camera_init"] * 2)
        self.assertEqual(
            [row["aligned_pose_stamp_ns"] for row in rows],
            ["1020000000", "1090000000"],
        )
        self.assertEqual(
            [row["prev_pose_stamp_ns"] for row in rows],
            ["1000000000", "1000000000"],
        )
        self.assertEqual(
            [row["next_pose_stamp_ns"] for row in rows],
            ["1100000000", "1100000000"],
        )
        self.assertEqual([row["alpha"] for row in rows], ["0.200000", "0.900000"])
        self.assertEqual(
            [row["prev_delta_ms"] for row in rows], ["20.000000", "90.000000"]
        )
        self.assertEqual(
            [row["next_delta_ms"] for row in rows], ["80.000000", "10.000000"]
        )
        self.assertAlmostEqual(float(rows[0]["delta_ms"]), 20.0, places=3)
        self.assertAlmostEqual(float(rows[1]["delta_ms"]), 10.0, places=3)
        for row in rows:
            self.assertAlmostEqual(
                sum(float(row[key]) ** 2 for key in ("qx", "qy", "qz", "qw")),
                1.0,
            )
            self.assertEqual(
                [float(row[key]) for key in ("roll", "pitch", "yaw")],
                [0.0, 0.0, 0.0],
            )

    def test_export_marks_large_interpolation_gap_invalid_and_retains_provenance(self):
        session = self.session / "large_gap"
        raw_dir = session / "raw"
        raw_dir.mkdir(parents=True)
        bag_path = raw_dir / "flight_0.bag"
        with rosbag.Bag(str(bag_path), "w") as bag:
            for stamp, x in [(1.0, 10.0), (1.3, 13.0)]:
                message = make_pose(stamp, x)
                bag.write(POSE_TOPIC, message, message.header.stamp)
            message = make_image(1.1, 30)
            bag.write(IMAGE_TOPIC, message, message.header.stamp)

        result = export_session(session, max_interp_gap_ms=200.0)
        with Path(result["alignment_csv"]).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["alignment_method"], "invalid")
        self.assertEqual(row["valid"], "0")
        self.assertEqual(row["interp_valid"], "0")
        self.assertEqual(row["pose_stamp_ns"], "1000000000")
        self.assertEqual(row["aligned_pose_stamp_ns"], "")
        self.assertEqual(row["prev_pose_stamp_ns"], "1000000000")
        self.assertEqual(row["next_pose_stamp_ns"], "1300000000")
        self.assertEqual(row["alpha"], "0.333333")
        self.assertEqual(row["prev_delta_ms"], "100.000000")
        self.assertEqual(row["next_delta_ms"], "200.000000")
        for field in (
            "frame_id", "x", "y", "z", "qx", "qy", "qz", "qw",
            "roll", "pitch", "yaw",
        ):
            self.assertEqual(row[field], "")

    def test_export_rejects_negative_alignment_thresholds(self):
        for kwargs in ({"max_delta_ms": -1}, {"max_interp_gap_ms": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                export_session(self.session, **kwargs)

    def test_alignment_rows_follow_video_order_across_bags(self):
        session = self.session / "multiple_bags"
        raw_dir = session / "raw"
        raw_dir.mkdir(parents=True)
        cases = [
            ("flight_0.bag", 2.0, 20.0, 30),
            ("flight_1.bag", 1.0, 10.0, 60),
        ]
        for name, stamp, x, value in cases:
            with rosbag.Bag(str(raw_dir / name), "w") as bag:
                pose = make_pose(stamp, x)
                bag.write(POSE_TOPIC, pose, pose.header.stamp)
                image = make_image(stamp, value)
                bag.write(IMAGE_TOPIC, image, image.header.stamp)

        result = export_session(session)
        with Path(result["alignment_csv"]).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))

        self.assertEqual(
            [row["image_stamp_ns"] for row in rows],
            ["2000000000", "1000000000"],
        )
        self.assertEqual([row["frame_index"] for row in rows], ["0", "1"])

    def test_export_rejects_session_without_bags(self):
        empty_session = self.session / "empty"
        empty_session.mkdir()
        with self.assertRaises(ValueError):
            export_session(empty_session)


if __name__ == "__main__":
    unittest.main()
