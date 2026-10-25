import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import rosbag
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage

from p450_recording.common import COMPRESSED_IMAGE_TOPIC
from p450_recording.exporter import export_session


class CompressedExporterTests(unittest.TestCase):
    def test_exports_jpeg_topic_and_reports_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            raw = session / "raw"
            raw.mkdir()
            bag_path = raw / "flight_0.bag"
            stamp = rospy.Time.from_sec(1.0)
            image = np.full((8, 12, 3), 73, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            compressed = CompressedImage()
            compressed.header.stamp = stamp
            compressed.format = "bgr8; jpeg compressed bgr8"
            compressed.data = encoded.tobytes()
            pose = Odometry()
            pose.header.stamp = stamp
            pose.header.frame_id = "camera_init"
            pose.pose.pose.orientation.w = 1.0
            with rosbag.Bag(str(bag_path), "w") as bag:
                bag.write("/Odometry", pose, stamp)
                bag.write(COMPRESSED_IMAGE_TOPIC, compressed, stamp)

            result = export_session(session)

            self.assertEqual(result["image_topic"], COMPRESSED_IMAGE_TOPIC)
            self.assertEqual(result["frame_count"], 1)
            self.assertGreater(Path(result["video_path"]).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
