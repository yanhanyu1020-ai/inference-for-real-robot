"""CPU-only resource gates; no NVIDIA/JAX/model calls."""
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import queue_eval as q


class QueueContract(unittest.TestCase):
    def test_exact_summary_coverage(self):
        self.assertTrue(q.summary_is_complete(dict(complete=True, windows=2426, checkpoint_step=5000), step=5000))
        self.assertFalse(q.summary_is_complete(dict(complete=True, windows=2400, checkpoint_step=5000), step=5000))
        self.assertFalse(q.summary_is_complete(dict(complete=True, windows=2426, checkpoint_step=10000), step=5000))

    def test_gpu_uuid_and_index(self):
        with patch.object(q.subprocess, 'run', return_value=SimpleNamespace(stdout='4, GPU-wrong,81920,70000,100')):
            with self.assertRaises(ValueError):
                q.query_gpu('nvidia-smi')
        with patch.object(q.subprocess, 'run', return_value=SimpleNamespace(stdout=f'5, {q.GPU_UUID},81920,9317,100')):
            self.assertEqual(q.query_gpu('nvidia-smi')['free_mib'], 9317)

    def test_permanent_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'attempt.json'
            q.claim(path, {'first': True})
            with self.assertRaises(FileExistsError):
                q.claim(path, {'first': False})
            self.assertEqual(q.read_json(path), {'first': True})

    def test_resource_environment(self):
        env = q.child_env(gpu=True)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], q.GPU_UUID)
        self.assertEqual(env['XLA_PYTHON_CLIENT_MEM_FRACTION'], '0.30')
        self.assertEqual(q.child_env(gpu=False)['CUDA_VISIBLE_DEVICES'], '')

    def test_low_memory_cannot_launch_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = q.Controller.__new__(q.Controller)
            controller.output_root = root
            controller.code_dir = root / 'original_code'
            controller.eval_dir = root / 'eval_code'
            controller.checkpoint_root = root / 'checkpoints'
            controller.cache_dir = root / 'cache'
            controller.python = '/not/executed/python'
            controller.plan = {}
            controller.identity = {'pid': 1}
            controller.state = {}
            controller.checkpoint_ready = lambda step: True
            controller.update = lambda *args, **kwargs: None
            controller.wait = lambda *args, **kwargs: False
            child_types = []

            def child(command, destination, *, gpu, phase):
                child_types.append(gpu)
                destination.mkdir(parents=True)
                (destination / 'contract_proof.json').write_text(json.dumps(
                    dict(passed=True, cpu_only=True, checkpoint_step=5000)))
                return {'returncode': 0}

            controller.run_child = child
            with patch.object(q, 'query_gpu', return_value={'free_mib': 9317}):
                self.assertIsNone(controller.evaluate(5000))
            self.assertEqual(child_types, [False])
            self.assertFalse((root / 'results/5000/attempt.json').exists())

    def test_default_mode_does_not_start_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(gpu_uuid=q.GPU_UUID, gpu_index=5, steps=q.STEPS,
                        checkpoint_root=str(root/'checkpoints'), output_root=str(root/'outputs'),
                        cache_dir=str(root/'cache'), code_dir=str(root/'original'),
                        eval_dir=str(Path(__file__).resolve().parent), python=sys.executable)
            path = root/'plan.json'
            path.write_text(json.dumps(plan))
            with patch.object(sys, 'argv', ['queue_eval.py', '--plan', str(path)]):
                with patch.object(q, 'Controller', side_effect=AssertionError('No worker may start')):
                    self.assertEqual(q.main(), 0)
            self.assertFalse((root/'outputs').exists())


if __name__ == '__main__':
    unittest.main()
