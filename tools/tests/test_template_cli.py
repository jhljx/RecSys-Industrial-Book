"""Exercise every dependency-free teaching template through its public CLI."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


def run_template(name: str, *args: str) -> tuple[int, dict]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "templates" / name / "app.py"), *args],
        cwd=ROOT, capture_output=True, text=True, timeout=10,
    )
    if completed.returncode not in (0, 2):
        raise AssertionError(f"{name} exited {completed.returncode}: {completed.stderr}")
    return completed.returncode, json.loads(completed.stdout)


class TemplateCliTest(unittest.TestCase):
    def test_pipeline_normal_timeout_and_overload(self) -> None:
        code, result = run_template("recommendation-pipeline", "--user", "u1", "--limit", "2")
        self.assertEqual(0, code)
        self.assertEqual(["item-1", "item-2"], result["items"])
        self.assertFalse(result["degraded"])
        self.assertTrue({"retrieval", "merge", "ranking", "total"} <= result["stage_latency_ms"].keys())
        code, overloaded = run_template("recommendation-pipeline", "--user", "u1", "--limit", "1", "--simulate-overload")
        self.assertEqual((0, ["hot-1"], "overload"), (code, overloaded["items"], overloaded["reason"]))
        # Deadline validation is deterministic; no scheduling-sensitive sleep or timeout threshold.
        failed = subprocess.run([sys.executable, str(ROOT / "templates/recommendation-pipeline/app.py"),
                                 "--user", "u1", "--timeout-ms", "0"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(0, failed.returncode)
        self.assertIn("timeout_ms must be positive", failed.stderr)

    def test_two_tower_version_contract(self) -> None:
        code, result = run_template("two-tower-ann", "--user", "u1")
        self.assertEqual((0, "ok", "v1", "i1"), (code, result["status"], result["version"], result["positive"]))
        self.assertEqual("i1", result["neighbors"][0][0])
        self.assertNotIn("i1", result["negatives"])
        for args, message in [(("--inject-version-mismatch",), "model/index version mismatch"),
                              (("--version", "v0"), "request/index version mismatch")]:
            with self.subTest(args=args):
                code, rejected = run_template("two-tower-ann", *args)
                self.assertEqual((2, "rejected", message), (code, rejected["status"], rejected["error"]))

    def test_feature_point_in_time_and_leakage(self) -> None:
        code, result = run_template("feature-consistency")
        self.assertEqual((0, "ok", 2, 50), (code, result["status"], len(result["rows"]), result["max_feature_age"]))
        self.assertTrue(all(row["feature_time"] <= row["label_time"] for row in result["rows"]))
        code, rejected = run_template("feature-consistency", "--inject-leakage")
        self.assertEqual((2, "rejected"), (code, rejected["status"]))
        self.assertIn("future features detected", rejected["error"])

    def test_serving_capacity_timeout_and_bad_input(self) -> None:
        code, result = run_template("serving-reliability", "--requests", "100", "--capacity", "80")
        self.assertEqual(0, code)
        self.assertEqual((80, 20, 100, 0.2), (result["personalized"], result["fallback"], result["non_empty"], result["fallback_rate"]))
        self.assertEqual(20, result["p99_ms"])
        code, slow = run_template("serving-reliability", "--requests", "100", "--capacity", "80", "--dependency-latency-ms", "100")
        self.assertEqual((0, 0, 100, 100, 1.0),
                         (code, slow["personalized"], slow["fallback"], slow["non_empty"], slow["fallback_rate"]))
        failed = subprocess.run([sys.executable, str(ROOT / "templates/serving-reliability/app.py"),
                                 "--capacity", "0"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(0, failed.returncode)
        self.assertIn("invalid simulation parameters", failed.stderr)


if __name__ == "__main__":
    unittest.main()
