"""CPU/stdlib-only regression checks for JSON-safe numerical failure reporting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import runtime_reporting as report


def strict_loads(encoded):
    def reject(token):
        raise AssertionError(f"Invalid JSON numeric extension: {token}")
    return json.loads(encoded, parse_constant=reject)


class ForeignScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class ForeignArray:
    def tolist(self):
        return [[1.0, float("inf")], [float("nan"), -2]]


class RuntimeReportingTests(unittest.TestCase):
    def test_finite_values_are_unchanged(self):
        record = {"step": 200, "loss": 0.123, "negative_zero": -0.0,
                  "nested": {"values": [True, False, None, 4, 5.0]}, "name": "训练"}
        original = json.dumps(record, ensure_ascii=False, allow_nan=False)
        safe = report.sanitize_record(record)
        self.assertEqual(json.dumps(safe, ensure_ascii=False, allow_nan=False), original)
        self.assertEqual(math.copysign(1, safe["negative_zero"]), -1)
        self.assertNotIn("nonfinite_detected", safe)

    def test_recursive_nonfinite_evidence_and_no_mutation(self):
        record = {"loss": float("nan"), "metrics": [float("inf"), {"a/b~c": -float("inf")}]}
        safe = report.sanitize_record(record)
        self.assertEqual(safe["loss"], None)
        self.assertEqual(safe["metrics"], [None, {"a/b~c": None}])
        self.assertEqual(safe["nonfinite_values"], [
            {"path": "/loss", "value": "NaN"},
            {"path": "/metrics/0", "value": "+Infinity"},
            {"path": "/metrics/1/a~1b~0c", "value": "-Infinity"},
        ])
        self.assertTrue(math.isnan(record["loss"]))
        strict_loads(report.dumps_record(record, allow_nan=True))
        self.assertEqual(report.sanitize_record(safe), safe)

    def test_optional_scalar_array_protocols(self):
        safe = report.sanitize_record({"scalar": ForeignScalar(float("nan")), "array": ForeignArray()})
        self.assertEqual(safe["scalar"], None)
        self.assertEqual(safe["array"], [[1.0, None], [None, -2]])
        self.assertEqual(len(safe["nonfinite_values"]), 3)
        strict_loads(report.dumps_record(safe))

    def test_nonfinite_metric_is_logged_before_training_raises(self):
        for number, label in ((float("nan"), "NaN"), (float("inf"), "+Infinity"), (-float("inf"), "-Infinity")):
            with self.subTest(number=label), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "metrics.jsonl"
                with self.assertRaises(FloatingPointError) as raised:
                    report.append_metrics(path, {"step": 100, "loss": number, "grad_norm": 0.4})
                self.assertIsInstance(raised.exception, report.NonFiniteMetricError)
                row = strict_loads(path.read_text())
                self.assertEqual(row["loss"], None)
                self.assertEqual(row["grad_norm"], 0.4)
                self.assertEqual(row["nonfinite_metrics"], [{"path": "/loss", "value": label}])
                self.assertEqual(raised.exception.record, row)
                self.assertIn(label, str(raised.exception))

    def test_finite_metrics_passthrough_and_multiline_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            rows = [{"step": 100, "loss": 0.3}, {"step": 200, "loss": 0.2}]
            for row in rows:
                self.assertEqual(report.append_metrics(path, row), row)
            self.assertEqual([strict_loads(line) for line in path.read_text().splitlines()], rows)

    def test_all_metrics_checked_not_just_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            with self.assertRaises(report.NonFiniteMetricError):
                report.append_metrics(path, {"step": 100, "loss": 0.3, "grad_norm": float("nan")})
            row = strict_loads(path.read_text())
            self.assertEqual(row["nonfinite_metrics"][0]["path"], "/grad_norm")

    def test_failure_status_and_nested_queue_snapshot_are_strict_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "runtime_status.json"
            metrics = root / "metrics.jsonl"
            status = {"status": "training", "step": 100, "train_loss": float("nan")}
            try:
                report.append_metrics(metrics, {"step": 100, "loss": float("nan")})
            except report.NonFiniteMetricError as error:
                status.update(status="failed", error=str(error), nonfinite_metrics=error.nonfinite_metrics)
            else:
                self.fail("Non-finite metrics must stop actual training")
            safe = report.atomic_json(path, status)
            self.assertEqual(strict_loads(path.read_text()), safe)
            self.assertEqual(safe["status"], "failed")
            self.assertEqual(safe["train_loss"], None)
            self.assertTrue(safe["nonfinite_detected"])
            # This is the serialization performed by queue snapshot/status:
            # embedding the runtime record must never reintroduce NaN tokens.
            snapshot = {"status": "failed", "runtime_status": strict_loads(path.read_text())}
            self.assertEqual(strict_loads(json.dumps(snapshot, allow_nan=False)), snapshot)
            self.assertFalse(list(root.glob(".runtime_status.json.*.tmp")))

    def test_current_queue_snapshot_and_atomic_status_accept_failure(self):
        # Exercise the real queue read/snapshot path when co-located. It does
        # not query GPUs or launch anything; no identities are supplied.
        import queue_training
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue"
            run = root / "run"
            queue_training.atomic_json(queue / "state.json", {"status": "failed"})
            report.atomic_json(run / "runtime_status.json", {"status": "failed", "train_loss": float("inf")})
            args = SimpleNamespace(queue_dir=queue, run_dir=run, run_name="cpu-report-test")
            snapshot = queue_training.snapshot(args)
            self.assertEqual(snapshot["runtime_status"]["train_loss"], None)
            self.assertEqual(snapshot["runtime_status"]["nonfinite_values"][0]["value"], "+Infinity")
            queue_training.atomic_json(queue / "snapshot.json", snapshot)
            self.assertEqual(strict_loads((queue / "snapshot.json").read_text()), snapshot)

    def test_unsupported_values_fail_explicitly(self):
        with self.assertRaises(TypeError):
            report.sanitize_record({"unsafe": object()})
        with self.assertRaises(TypeError):
            report.sanitize_record({1: "non-string-key"})


if __name__ == "__main__":
    unittest.main()
