import copy
import unittest

from replace_short_sources import fill_rows


class PreviousValueTests(unittest.TestCase):
    def test_no_future_fill_and_recovery_preserves_measured_values(self):
        rows=[]
        for i in range(24):
            valid=i not in (0,2,3)
            rows.append(dict(source_task_time_s=i/15, state=[float(i)]*30, action=[float(i)]*14,
                state_valid=[valid]*30, action_valid=[valid]*14, camera_valid=[True]*3,
                eligible=valid, rejection_reasons=[] if valid else ['action_target_missing_or_invalid']))
        original=copy.deepcopy(rows)
        filled=fill_rows(rows)
        self.assertEqual(filled[0]['source_task_time_s'],1/15)
        self.assertEqual(filled[1]['action'],[1.0]*14)
        self.assertEqual(filled[2]['action'],[1.0]*14)
        self.assertEqual(filled[2]['action_value_source_grid_time_s'],[1/15]*14)
        self.assertTrue(all(filled[2]['action_imputed']))
        self.assertFalse(any(filled[2]['action_valid']))
        self.assertEqual(filled[3]['action'],[4.0]*14)
        self.assertFalse(any(filled[3]['action_imputed']))
        self.assertEqual(rows,original)


if __name__=='__main__':
    unittest.main()
