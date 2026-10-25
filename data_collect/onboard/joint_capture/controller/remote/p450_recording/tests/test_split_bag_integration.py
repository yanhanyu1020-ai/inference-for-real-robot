"""ROS bag integration on generated data only; no ROS master or robot is used."""
import csv
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import rosbag
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from p450_recording.exporter import export_session, RAW_IMAGE_TOPIC, POSE_TOPIC
from p450_recording.orchestrator import CaptureOrchestrator, _bag_is_ready


def messages(index):
    stamp=rospy.Time.from_sec(float(index+1))
    pose=Odometry()
    pose.header.stamp=stamp
    pose.header.frame_id='test'
    pose.pose.pose.position.x=float(index)
    pose.pose.pose.orientation.w=1.
    image=Image()
    image.header.stamp=stamp
    image.height=image.width=16
    image.encoding='bgr8'
    image.step=48
    image.data=bytes([index*15]*16*16*3)
    return pose,image


def write_bag(path,index):
    pose,image=messages(index)
    with rosbag.Bag(str(path),'w') as bag:
        bag.write(POSE_TOPIC,pose,pose.header.stamp)
        bag.write(RAW_IMAGE_TOPIC,image,image.header.stamp)


class SplitBagIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='p450-generated-test-')
        self.session=Path(self.temporary.name)
        self.raw=self.session/'raw'
        self.raw.mkdir()

    def tearDown(self): self.temporary.cleanup()

    def test_twelve_shards_have_monotonic_csv_and_matching_video(self):
        for i in range(12): write_bag(self.raw/f'flight_{i}.bag',i)
        result=export_session(self.session)
        self.assertEqual(result['frame_count'],12)
        with Path(result['alignment_csv']).open() as f: rows=list(csv.DictReader(f))
        self.assertEqual([int(row['image_stamp_ns']) for row in rows],
                         [i*1_000_000_000 for i in range(1,13)])
        cap=cv2.VideoCapture(result['video_path'])
        brightness=[]
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                brightness.append(float(frame.mean()))
        finally: cap.release()
        self.assertEqual(len(brightness),12)
        self.assertTrue(all(a<b for a,b in zip(brightness,brightness[1:])),brightness)

    def test_real_reindex_recovers_interrupted_tail_and_preserves_original(self):
        write_bag(self.raw/'flight_0.bag',0)
        source=self.raw/'flight_1.bag.active'
        # Terminate a subprocess without closing its bag, leaving no final index.
        script='''import os,sys
import rosbag,rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
bag=rosbag.Bag(sys.argv[1],'w')
stamp=rospy.Time.from_sec(2.)
pose=Odometry(); pose.header.stamp=stamp; pose.header.frame_id='test'; pose.pose.pose.orientation.w=1.
image=Image(); image.header.stamp=stamp; image.height=image.width=16; image.encoding='bgr8'; image.step=48; image.data=bytes([100]*768)
bag.write('/Odometry',pose,stamp)
bag.write('/uav1/camera/color/image_raw',image,stamp)
bag.flush(); bag._file.flush(); os.fsync(bag._file.fileno())
os._exit(0)
'''
        subprocess.run([sys.executable,'-c',script,str(source)],check=True)
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'unfinished'): export_session(self.session)
        recorder=mock.Mock()
        recorder.status.return_value={'active':False}
        ctl=CaptureOrchestrator(data_root=self.session/'runtime',recorder=recorder,bag_finalize_timeout=0)
        finalized=ctl.finalize_raw(self.session)
        self.assertEqual(finalized['recovered_bags'],[str(self.raw/'flight_000001.bag')])
        self.assertTrue(_bag_is_ready(self.session))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)
        result=export_session(self.session)
        self.assertEqual(result['frame_count'],2)
        with Path(result['alignment_csv']).open() as f: rows=list(csv.DictReader(f))
        self.assertEqual([int(row['image_stamp_ns']) for row in rows],[1_000_000_000,2_000_000_000])
        self.assertEqual(ctl.finalize_raw(self.session)['recovered_bags'],[])


if __name__=='__main__': unittest.main()
