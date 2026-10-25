#!/usr/bin/env python3
"""CPU-only queue tests: fake nvidia-smi and a tiny fake training process."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("queue_training", Path(__file__).with_name("queue_training.py"))
assert SPEC and SPEC.loader
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.code = self.root / "code"
        self.code.mkdir()
        shutil.copy(Path(__file__).with_name("queue_training.py"), self.code / "queue_training.py")
        (self.code / "run_training.py").write_text("# CPU test placeholder\n")
        self.queue_dir = self.root / "queue"
        self.run_dir = self.root / "run"
        self.args = argparse.Namespace(code_dir=self.code, queue_dir=self.queue_dir,
                                       run_dir=self.run_dir, run_name="test-run", python=sys.executable,
                                       nvidia_smi="FAKE-NVIDIA-SMI")
        self.good = {"uuid": "GPU-test", "total_mib": 81920, "free_mib": 81920,
                     "has_compute_process": False}
        self.bind_manifest()

    def bind_manifest(self):
        manifest = {"files": {name: queue.sha256(self.code / name)
                              for name in ("queue_training.py", "run_training.py")}}
        queue.atomic_json(self.code / "manifest.json", manifest)
        queue.atomic_json(self.code / "preflight_report.json", {
            "passed": True, "manifest_sha256": queue.sha256(self.code / "manifest.json")})

    def test_free_threshold_and_80gb_gate(self):
        self.assertTrue(queue.available(self.good))
        self.assertFalse(queue.available(dict(self.good, total_mib=79999, free_mib=79999)))
        self.assertFalse(queue.available(dict(self.good, free_mib=77823)))
        self.assertTrue(queue.available(dict(self.good, free_mib=77824)))
        self.assertFalse(queue.available(dict(self.good, has_compute_process=True)))

    def test_waiting_does_not_consume_attempt(self):
        with mock.patch.object(queue, "query_gpus", return_value=[dict(self.good, has_compute_process=True)]):
            result = queue.check_and_start(self.args)
        self.assertEqual(result["status"], "waiting")
        self.assertFalse((self.queue_dir / "attempt.json").exists())

    def test_second_gpu_check_prevents_start(self):
        with mock.patch.object(queue, "query_gpus", side_effect=[[self.good], [dict(self.good, free_mib=1)]]):
            result = queue.check_and_start(self.args)
        self.assertEqual(result["status"], "waiting")
        self.assertFalse((self.queue_dir / "attempt.json").exists())

    def test_preflight_manifest_mismatch_blocks(self):
        (self.code / "run_training.py").write_text("# changed after preflight\n")
        with mock.patch.object(queue, "query_gpus") as query:
            result = queue.check_and_start(self.args)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("SHA256 mismatch", result["detail"])
        query.assert_not_called()

    def test_preflight_requires_manifest_binding(self):
        queue.atomic_json(self.code / "preflight_report.json", {"passed": True})
        with self.assertRaisesRegex(ValueError, "not bound"):
            queue.verify_preflight(self.code)

    def test_claim_is_exclusive_and_permanent(self):
        self.queue_dir.mkdir()
        claim = {"attempt_utc": "test", "run_name": "test-run"}
        queue.claim_attempt(self.queue_dir / "attempt.json", claim)
        with self.assertRaises(FileExistsError):
            queue.claim_attempt(self.queue_dir / "attempt.json", claim)
        for outcome in ("failed", "completed", "finished_unverified", "claimed"):
            queue.atomic_json(self.queue_dir / "state.json", {"status": outcome})
            with mock.patch.object(queue, "query_gpus") as query:
                result = queue.check_and_start(self.args)
            self.assertTrue(result["attempt_exists"])
            query.assert_not_called()

    def test_launcher_failure_retains_claim_no_retry(self):
        with mock.patch.object(queue, "query_gpus", return_value=[self.good]), \
                mock.patch.object(queue.subprocess, "Popen", side_effect=OSError("fake spawn failure")):
            result = queue.check_and_start(self.args)
        self.assertEqual(result["status"], "failed_to_launch_supervisor")
        with mock.patch.object(queue, "query_gpus") as query:
            again = queue.check_and_start(self.args)
        query.assert_not_called()
        self.assertEqual(again["status"], "failed_to_launch_supervisor")

    def test_pid_reuse_fails_identity_match(self):
        saved = {"pid": 123, "uid": 1037, "start_ticks": "100", "command": ["python", "run.py"]}
        with mock.patch.object(queue, "process_identity", return_value=dict(saved, start_ticks="101")):
            self.assertFalse(queue.identity_matches(saved))
        with mock.patch.object(queue, "process_identity", return_value=saved):
            self.assertTrue(queue.identity_matches(saved))

    def test_gpu_query_fails_closed_on_unknown_process(self):
        outputs = ["GPU-test, 81920, 81920\n", "GPU-test, [N/A]\n"]
        with mock.patch.object(queue.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0, out, "") for out in outputs]):
            with self.assertRaisesRegex(ValueError, "Unparseable GPU process"):
                queue.query_gpus("FAKE")

    def test_gpu_query_process_detection(self):
        outputs = ["GPU-test, 81920, 81920\n", "GPU-test, 123\n"]
        with mock.patch.object(queue.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0, out, "") for out in outputs]):
            self.assertFalse(queue.available(queue.query_gpus("FAKE")[0]))

    def test_mig_or_unknown_process_uuid_fails_closed(self):
        outputs = ["GPU-test, 81920, 81920\n", "MIG-GPU-test/1/0, 123\n"]
        with mock.patch.object(queue.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0, out, "") for out in outputs]):
            with self.assertRaisesRegex(ValueError, "absent from GPU inventory"):
                queue.query_gpus("FAKE")

    def configure_fake_integration(self, returncode=0):
        self.fake_smi = self.root / "fake-nvidia-smi"
        self.fake_smi.write_text(
            f"#!{sys.executable}\nimport sys\n"
            "if any(arg.startswith('--query-gpu=') for arg in sys.argv):\n"
            "    print('GPU-test, 81920, 81920')\n")
        self.fake_smi.chmod(0o755)
        self.marker = self.root / "actual-starts.txt"
        script = f'''import argparse, json, pathlib, time, os
p = argparse.ArgumentParser()
p.add_argument('--gpu')
p.add_argument('--run-name')
a = p.parse_args()
assert a.gpu == 'GPU-test'
with open({str(self.marker)!r}, 'a') as f:
    f.write(str(os.getpid()) + '\\n')
path = pathlib.Path({str(self.run_dir)!r})
path.mkdir(parents=True, exist_ok=True)
(path / 'runtime_status.json').write_text(json.dumps({{'status':'training'}}))
time.sleep(1.5)
(path / 'runtime_status.json').write_text(json.dumps({{'status':{'completed' if returncode == 0 else 'failed'!r}}}))
raise SystemExit({returncode})
'''
        (self.code / "run_training.py").write_text(script)
        self.bind_manifest()
        self.args.nvidia_smi = str(self.fake_smi)

    def cli(self, action):
        return [sys.executable, str(self.code / "queue_training.py"), action,
                "--code-dir", str(self.code), "--queue-dir", str(self.queue_dir),
                "--run-dir", str(self.run_dir), "--run-name", "test-run",
                "--python", sys.executable, "--nvidia-smi", str(self.fake_smi)]

    def call(self, action):
        return json.loads(subprocess.check_output(self.cli(action), text=True, timeout=15))

    def wait_for_terminal(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            current = self.call("status")
            if current["status"] in {"completed", "failed", "finished_unverified"}:
                return current
            time.sleep(0.05)
        self.fail("CPU fake training did not finish")

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux /proc required for real PID identity integration")
    def test_concurrent_checks_inherit_lock_and_start_once(self):
        self.configure_fake_integration()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda _: self.call("check-and-start"), range(5)))
        self.assertTrue(any(item["status"] == "running" for item in results), results)
        running = self.call("check-and-start")
        self.assertTrue(running["lock_busy"])
        self.assertTrue(running["training_identity_matches"])
        self.assertTrue(running["supervisor_identity_matches"])
        terminal = self.wait_for_terminal()
        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(len(self.marker.read_text().splitlines()), 1)
        self.assertEqual(self.call("check-and-start")["status"], "completed")
        self.assertEqual(len(self.marker.read_text().splitlines()), 1)

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux /proc required for real PID identity integration")
    def test_error_exit_never_retries(self):
        self.configure_fake_integration(returncode=7)
        result = self.call("check-and-start")
        self.assertEqual(result["status"], "running")
        terminal = self.wait_for_terminal()
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["returncode"], 7)
        self.assertEqual(self.call("check-and-start")["status"], "failed")
        self.assertEqual(len(self.marker.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
