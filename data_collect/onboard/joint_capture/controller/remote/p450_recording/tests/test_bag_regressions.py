"""Regression checks for split naming, ordering and interrupted tail handling."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault('rosbag',mock.Mock())
sys.modules.setdefault('cv_bridge',mock.Mock(CvBridge=mock.Mock))
from p450_recording import exporter
from p450_recording.common import pending_bags, normalize_bag_names
from p450_recording.orchestrator import CaptureOrchestrator, _bag_is_ready


class SplitBagRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.session=Path(self.temporary.name)
        self.raw=self.session/'raw'
        self.raw.mkdir()

    def tearDown(self): self.temporary.cleanup()

    def test_twelve_legacy_shards_are_sorted_numerically(self):
        for i in range(12): (self.raw/f'flight_{i}.bag').write_bytes(b'bag')
        self.assertEqual([p.name for p in exporter._find_bags(self.session)],
                         [f'flight_{i}.bag' for i in range(12)])

    def test_manual_export_refuses_an_unrecovered_tail(self):
        (self.raw/'flight_0.bag').write_bytes(b'complete')
        (self.raw/'flight_1.bag.active').write_bytes(b'tail')
        with self.assertRaisesRegex(ValueError,'unfinished'):
            exporter.export_session(self.session)
        self.assertFalse((self.session/'export').exists())
        self.assertFalse(_bag_is_ready(self.session))

    def test_finalize_recovers_tail_even_when_an_earlier_shard_is_ready(self):
        (self.raw/'flight_0.bag').write_bytes(b'complete')
        source=self.raw/'flight_1.bag.active'
        source.write_bytes(b'original-tail')
        calls=[]
        def reindex(active,output):
            calls.append(active)
            output.mkdir()
            (output/active.name).write_bytes(b'recovered-tail')
        recorder=mock.Mock()
        recorder.status.return_value={'active':False}
        ctl=CaptureOrchestrator(data_root=self.session/'runtime',recorder=recorder,
            bag_ready_provider=lambda _:True,bag_reindexer=reindex,bag_finalize_timeout=0,killpg_func=mock.Mock())
        result=ctl.finalize_raw(self.session)
        self.assertEqual(calls,[source])
        self.assertEqual((self.raw/'flight_000000.bag').read_bytes(),b'complete')
        self.assertEqual((self.raw/'flight_000001.bag').read_bytes(),b'recovered-tail')
        self.assertEqual(source.read_bytes(),b'original-tail')
        self.assertEqual(result['recovered_bags'],[str(self.raw/'flight_000001.bag')])
        self.assertTrue(_bag_is_ready(self.session))
        self.assertFalse(pending_bags(self.raw))
        ctl.finalize_raw(self.session)
        self.assertEqual(len(calls),1)

    def test_failed_recovery_cannot_report_success(self):
        (self.raw/'flight_0.bag').write_bytes(b'complete')
        (self.raw/'flight_1.bag.active').write_bytes(b'tail')
        recorder=mock.Mock()
        recorder.status.return_value={'active':False}
        ctl=CaptureOrchestrator(data_root=self.session/'runtime',recorder=recorder,
            bag_ready_provider=lambda _:True,bag_reindexer=lambda *_:None,bag_finalize_timeout=0,killpg_func=mock.Mock())
        with self.assertRaisesRegex(RuntimeError,'no usable output'):
            ctl.finalize_raw(self.session)
        self.assertTrue((self.raw/'flight_1.bag.active').exists())

    def test_normalized_names_are_sorted_and_do_not_overwrite_conflicts(self):
        for i in (0,1,10,11,2): (self.raw/f'flight_{i}.bag').write_bytes(bytes([i]))
        normalize_bag_names(self.raw)
        self.assertEqual([p.name for p in sorted(self.raw.glob('*.bag'))],
            ['flight_000000.bag','flight_000001.bag','flight_000002.bag','flight_000010.bag','flight_000011.bag'])
        (self.raw/'flight_1.bag').write_bytes(b'different')
        with self.assertRaisesRegex(RuntimeError,'duplicate'):
            normalize_bag_names(self.raw)
        self.assertEqual((self.raw/'flight_000001.bag').read_bytes(),bytes([1]))

    def test_empty_final_copy_does_not_hide_an_interrupted_shard(self):
        (self.raw/'flight_1.bag').write_bytes(b'')
        active=self.raw/'flight_1.bag.active'
        active.write_bytes(b'tail')
        self.assertEqual(pending_bags(self.raw),[active])

    def test_failed_partial_reindex_is_retried(self):
        from p450_recording.orchestrator import recover_interrupted_bags
        source=self.raw/'flight_1.bag.active'
        source.write_bytes(b'original')
        calls=[]
        def reindex(active,output):
            calls.append(active)
            output.mkdir(parents=True)
            (output/active.name).write_bytes(b'partial' if len(calls)==1 else b'complete')
            if len(calls)==1: raise RuntimeError('interrupted reindex')
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            recover_interrupted_bags(self.session,reindex)
        self.assertFalse((self.raw/'flight_1.bag').exists())
        recover_interrupted_bags(self.session,reindex)
        self.assertEqual(len(calls),2)
        self.assertEqual((self.raw/'flight_1.bag').read_bytes(),b'complete')
        self.assertEqual(source.read_bytes(),b'original')

    def test_manual_export_rejects_duplicate_legacy_and_padded_shards(self):
        (self.raw/'flight_1.bag').write_bytes(b'legacy')
        (self.raw/'flight_000001.bag').write_bytes(b'copy')
        with self.assertRaisesRegex(ValueError,'duplicate'):
            exporter._find_bags(self.session)


if __name__=='__main__': unittest.main()
