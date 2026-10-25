#!/usr/bin/env python3
"""CPU-only launch contract tests: fake inventory, temporary files, no GPU/JAX."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("fresh_launch", Path(__file__).with_name("launch_training.py"))
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LaunchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.code = self.root / "code"
        self.code.mkdir()
        self.model = self.root / "models"
        self.options = argparse.Namespace(gpu="auto", check=False, nvidia_smi="FAKE-NVIDIA-SMI")
        self.configuration()
        self.queue = types.SimpleNamespace(
            verify_preflight=mock.Mock(return_value={"manifest_sha256": "test-manifest"}),
            available=lambda g: (g["total_mib"] >= 80000 and g["free_mib"] >= g["total_mib"] * .95
                                 and not g["has_compute_process"]),
            query_gpus=mock.Mock(), check_and_start=mock.Mock(return_value={"status": "running"}),
            snapshot=mock.Mock(return_value={"status": "failed", "attempt_exists": True}),
        )
        self.gpu0 = {"uuid": "GPU-0", "total_mib": 81920, "free_mib": 81000,
                     "has_compute_process": False}
        self.gpu5 = dict(self.gpu0, uuid="GPU-5")
        self.queue.query_gpus.return_value = [self.gpu0, self.gpu5]
        self.indices = {"0": "GPU-0", "5": "GPU-5"}
        for name, value in (("HERE", self.code), ("MODEL_ROOT", self.model)):
            patcher = mock.patch.object(launcher, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def configuration(self, resume="False", overwrite="False"):
        (self.code / "configuration.py").write_text(
            f"RUN_NAME = {launcher.RUN_NAME!r}\nCONFIG_NAME = {launcher.CONFIG_NAME!r}\n"
            f"def make_config():\n return cfg.TrainConfig(resume={resume}, overwrite={overwrite})\n")

    def execute(self):
        with mock.patch.object(launcher, "query_indices", return_value=self.indices):
            return launcher.execute(self.options, self.queue)

    def test_auto_passes_fixed_fresh_run_to_one_shot_queue(self):
        result = self.execute()
        self.assertEqual(result["status"], "running")
        self.assertIn("does not prove", result["detail"])
        args = self.queue.check_and_start.call_args.args[0]
        self.assertEqual(args.run_name, launcher.RUN_NAME)
        self.assertIsNone(args.preferred_gpu)
        self.assertEqual(args.python, launcher.PYTHON)
        self.assertEqual(args.run_dir, self.model / "runs" / launcher.RUN_NAME)

    def test_explicit_index_resolves_uuid_and_does_not_fallback(self):
        self.options.gpu = "5"
        result = self.execute()
        self.assertEqual(result["status"], "running")
        self.assertEqual(self.queue.check_and_start.call_args.args[0].preferred_gpu, "GPU-5")
        self.queue.check_and_start.reset_mock()
        self.queue.query_gpus.return_value = [self.gpu0, dict(self.gpu5, has_compute_process=True)]
        result = self.execute()
        self.assertEqual(result["status"], "waiting")
        self.queue.check_and_start.assert_not_called()

    def test_explicit_uuid_passes_exact_device(self):
        self.options.gpu = "GPU-5"
        self.execute()
        self.assertEqual(self.queue.check_and_start.call_args.args[0].preferred_gpu, "GPU-5")

    def test_check_ready_creates_no_directories_or_attempt(self):
        self.options.check = True
        result = self.execute()
        self.assertEqual(result["status"], "ready")
        self.assertFalse(self.model.exists())
        self.queue.check_and_start.assert_not_called()

    def test_check_busy_creates_no_directories_or_attempt(self):
        self.options.check = True
        self.queue.query_gpus.return_value = [dict(self.gpu0, free_mib=77823),
                                               dict(self.gpu5, has_compute_process=True)]
        result = self.execute()
        self.assertEqual(result["status"], "waiting")
        self.assertFalse(self.model.exists())
        self.queue.check_and_start.assert_not_called()

    def test_existing_claim_returns_status_even_with_run_and_checkpoint(self):
        args = launcher.queue_arguments(self.options.nvidia_smi)
        args.queue_dir.mkdir(parents=True)
        (args.queue_dir / "attempt.json").write_text("{}")
        args.run_dir.mkdir(parents=True)
        (args.run_dir / "runtime_status.json").write_text("{}")
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.queue.snapshot.assert_called_once()
        self.queue.verify_preflight.assert_not_called()
        self.queue.query_gpus.assert_not_called()
        self.queue.check_and_start.assert_not_called()

    def test_checkpoint_refused_before_gpu_query_or_start(self):
        path = self.model / "checkpoints" / launcher.CONFIG_NAME / launcher.RUN_NAME / "1000"
        path.mkdir(parents=True)
        result = self.execute()
        self.assertEqual(result["status"], "blocked")
        self.assertIn("separate resume entry", result["detail"])
        self.queue.query_gpus.assert_not_called()
        self.queue.check_and_start.assert_not_called()

    def test_nonempty_run_refused(self):
        args = launcher.queue_arguments(self.options.nvidia_smi)
        args.run_dir.mkdir(parents=True)
        (args.run_dir / "any.json").write_text("{}")
        self.assertEqual(self.execute()["status"], "blocked")
        self.queue.check_and_start.assert_not_called()

    def test_resume_or_overwrite_configuration_refused(self):
        for resume, overwrite in (("True", "False"), ("False", "True"), ("unknown", "False")):
            with self.subTest(resume=resume, overwrite=overwrite):
                self.configuration(resume, overwrite)
                result = self.execute()
                self.assertEqual(result["status"], "blocked")
                self.assertIn("literal", result["detail"])
        self.queue.check_and_start.assert_not_called()

    def test_preflight_failure_blocks_before_gpu_query(self):
        self.queue.verify_preflight.side_effect = ValueError("Manifest SHA256 mismatch")
        result = self.execute()
        self.assertEqual(result["status"], "blocked")
        self.queue.query_gpus.assert_not_called()
        self.queue.check_and_start.assert_not_called()

    def test_inventory_changed_or_unknown_card_blocks(self):
        self.indices["6"] = "GPU-6"
        self.assertEqual(self.execute()["status"], "blocked")
        self.indices.pop("6")
        self.options.gpu = "6"
        self.assertEqual(self.execute()["status"], "blocked")
        self.queue.check_and_start.assert_not_called()

    def test_index_query_uses_explicit_index_uuid_fields(self):
        completed = types.SimpleNamespace(stdout="5, GPU-5\n0, GPU-0\n")
        with mock.patch.object(launcher.subprocess, "run", return_value=completed) as run:
            result = launcher.query_indices("FAKE")
        self.assertEqual(result, {"5": "GPU-5", "0": "GPU-0"})
        self.assertIn("--query-gpu=index,uuid", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
