"""Synthetic regressions for causality, irregular sampling and H15 boundaries."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from prepare import DT, Series, causal_video, kinematic_action, segments, video_timeline, dump


class AlignmentTests(unittest.TestCase):
    def test_nonfuture_selection_and_staleness(self):
        s = Series(pd.DataFrame({'x': [1., 2., 3.]}), [0.01, 0.07, 0.2], ['x'])
        self.assertEqual(s.past(1/15)[1], 0)
        self.assertIsNone(s.past(0))
        self.assertIsNone(s.past(0.19, 0.075))
        mapping = dict(times=np.array([0.01, 0.07]), frames=np.array([10, 20]),
                       common_rows=np.array([0, 1]), warnings=np.array([False, False]))
        self.assertEqual(causal_video(mapping, 1/15)[0], 10)
        self.assertIsNone(causal_video(mapping, 0))

    def test_motion_label_uses_one_fifteenth_and_heading(self):
        # Irregular samples of constant world-y velocity; yaw=pi/2 means body-forward.
        t = np.array([0., 0.031, 0.079, 0.12, 0.19, 0.24])
        s = Series(pd.DataFrame(dict(x=0*t, y=2*t, z=0*t, yaw=0*t+np.pi/2)),
                   t, ['x', 'y', 'z', 'yaw'])
        action, _ = kinematic_action(s, 0.08, 3, 2, False)
        np.testing.assert_allclose(action, [2., 0., 0.], atol=1e-12)
        # A 0.1 s interval would give a different target for acceleration x=t^2.
        t = np.array([0., 0.1, 0.1+DT, 0.2, 0.3])
        s = Series(pd.DataFrame(dict(x=t*t, y=0*t, z=0*t, yaw=0*t)), t, ['x', 'y', 'z', 'yaw'])
        self.assertAlmostEqual(kinematic_action(s, 0.1, 3, 2, False)[0][0], 0.2+DT)

    def test_invalid_latest_command_is_not_replaced_by_older_command(self):
        s = Series(pd.DataFrame({'q': [1., 2.]}), [0.01, 0.06], ['q'], [False, True])
        self.assertIsNone(s.past(1/15))

    def test_no_interpolation_across_pose_gap(self):
        t = np.array([0., 0.02, 0.5, 0.52])
        s = Series(pd.DataFrame(dict(x=t, y=t*0, z=t*0, yaw=t*0)), t, ['x', 'y', 'z', 'yaw'])
        self.assertIsNone(kinematic_action(s, 0.01, 3, 2, False))

    def test_segments_do_not_cross_rejections_or_missing_video_rows(self):
        def row(i, valid=True):
            return dict(source_frame_index=i, source_task_time_s=i/15, eligible=valid)
        rows = [row(i) for i in range(15)] + [row(15, False)] + [row(i) for i in range(16,30)]
        self.assertEqual([len(x) for x in segments(rows)], [15])
        rows = [row(i) for i in range(14)] + [row(i) for i in range(15,30)]
        self.assertEqual([len(x) for x in segments(rows)], [15])

    def test_timeline_comes_from_video_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'timing').mkdir()
            dump(p/'episode.json', {'duration_s': 0.19})
            pd.DataFrame({'frame': [0,1,2], 'task_time_s': [0., round(1/15,9), round(2/15,9)]}).to_csv(p/'timing/frame_mapping.csv',index=False)
            self.assertEqual(video_timeline(p)[1][1], round(1/15,9))
            pd.DataFrame({'frame': [0,1,2], 'task_time_s': [0., 0.1, 0.2]}).to_csv(p/'timing/frame_mapping.csv',index=False)
            with self.assertRaises(ValueError): video_timeline(p)


if __name__ == '__main__':
    unittest.main()
