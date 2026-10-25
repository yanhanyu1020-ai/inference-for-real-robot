import unittest
from unittest import mock
from p450_recording import process_tracking as tracking


class ProcessTrackingTests(unittest.TestCase):
    def test_exact_launch_tokens(self):
        self.assertTrue(tracking.command_matches(b'/usr/bin/python3\0/opt/ros/noetic/bin/roslaunch\0pkg\0rs_camera_d435i.launch\0', 'rs_camera_d435i.launch'))
        self.assertFalse(tracking.command_matches(b'bash\0-c\0roslaunch pkg rs_camera_d435i.launch\0', 'rs_camera_d435i.launch'))
        self.assertFalse(tracking.command_matches(b'roslaunch\0pkg\0old_rs_camera_d435i.launch\0', 'rs_camera_d435i.launch'))

    def test_zombie_is_not_alive(self):
        with mock.patch.object(tracking, 'process_stat', return_value=['Z']):
            self.assertFalse(tracking.process_exists(123))

    def test_missing_process_is_not_alive(self):
        with mock.patch.object(tracking, 'process_stat', side_effect=FileNotFoundError):
            self.assertFalse(tracking.process_exists(123))

    def test_recorder_does_not_wait_for_zombie(self):
        from p450_recording.manager import _process_exists
        with mock.patch.object(tracking, 'process_stat', return_value=['Z']), mock.patch('os.kill'):
            self.assertFalse(_process_exists(123))
