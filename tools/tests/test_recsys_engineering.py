"""Check domain contracts on supplied observations, including renamed target files."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recsys_engineering as engineering


class EngineeringEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "observations.json"

    def check(self, domain, records):
        self.source.write_text(json.dumps({"domain": domain, "records": records}), encoding="utf-8")
        return engineering.check_evidence(self.source, domain)

    def test_feature_future_row_fails_even_with_unrelated_fields(self):
        result = self.check("feature-time", [
            {"phase": "baseline", "label_time": 100, "feature_time": 100, "selected": True},
            {"phase": "fault", "label_time": 100, "feature_time": 101, "selected": True, "noise": "unrelated"},
            {"phase": "recovery", "label_time": 100, "feature_time": 101, "selected": False},
        ])
        self.assertEqual([2], result["failed_records"])
        self.assertTrue(all(result["coverage"].values()))
        self.assertEqual("user_supplied_local_file", result["provenance"])

    def test_serving_timeout_and_empty_fallback_fail(self):
        result = self.check("serving", [
            {"phase": "baseline", "budget_ms": 200, "elapsed_ms": 100, "candidates": 3},
            {"phase": "fault", "budget_ms": 200, "elapsed_ms": 201, "candidates": 3},
            {"phase": "recovery", "budget_ms": 200, "elapsed_ms": 80, "candidates": 0},
        ])
        self.assertEqual([2, 3], result["failed_records"])

    def test_mixed_release_must_be_rejected(self):
        result = self.check("release-version", [
            {"phase": "baseline", "model_release": "v1", "embedding_release": "v1", "index_release": "v1", "accepted": True},
            {"phase": "fault", "model_release": "v2", "embedding_release": "v1", "index_release": "v1", "accepted": True},
            {"phase": "recovery", "model_release": "v2", "embedding_release": "v1", "index_release": "v1", "accepted": False},
        ])
        self.assertEqual([2], result["failed_records"])
        all_rejected = self.check("release-version", [
            {"phase": "baseline", "model_release": "v1", "embedding_release": "v1", "index_release": "v1", "accepted": False},
            {"phase": "fault", "model_release": "v2", "embedding_release": "v1", "index_release": "v1", "accepted": False},
        ])
        self.assertEqual([1], all_rejected["failed_records"])

    def test_evidence_boundary_and_target_discovery(self):
        with self.assertRaises(ValueError):
            self.check("serving", [{"phase": "fault", "budget_ms": 200, "elapsed_ms": True, "candidates": 1}])
        self.source.write_text(json.dumps({"domain": "feature-time", "records": [{}]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            engineering.check_evidence(self.source, "serving")
        project = self.root / "my-service"
        (project / "src").mkdir(parents=True)
        (project / "src/app.py").write_text("pass\n", encoding="utf-8")
        context = engineering.project_context(project)
        self.assertIn("src/app.py", context["entry_candidates"])

    def test_review_cli_target_project_and_sample_violation(self):
        import recsys_agent

        project = self.root / "my-service"
        (project / "src").mkdir(parents=True)
        (project / "src/producer.py").write_text("def join(feature, label):\n    return feature.timestamp <= label.timestamp\n", encoding="utf-8")
        self.source.write_text(json.dumps({"domain": "feature-time", "records": [
            {"phase": "fault", "label_time": 100, "feature_time": 101, "selected": True}]}), encoding="utf-8")
        book_root = Path(recsys_agent.__file__).resolve().parents[1]
        command = [sys.executable, str(Path(recsys_agent.__file__)), "--root", str(book_root), "review", "src/producer.py",
                   "--project", str(project), "--domain", "feature-time", "--evidence", str(self.source)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual([1], report["evidence_check"]["failed_records"])
        self.assertEqual(str((project / "src/producer.py").resolve()), report["target"])
        command[command.index("src/producer.py")] = "../outside.py"
        blocked = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(0, blocked.returncode)
        self.assertIn("必须位于", blocked.stderr)

    def test_verify_commands_real_local_fault_and_recovery(self):
        project = self.root / "my-service"
        project.mkdir()
        (project / "emit.py").write_text(
            "import json, sys\n"
            "phase = sys.argv[1]\n"
            "print(json.dumps({'label_time': 100, 'feature_time': 101 if phase != 'baseline' else 100, "
            "'selected': phase != 'recovery'}))\n", encoding="utf-8")
        result = engineering.verify_commands(project, "feature-time", [
            (phase, f"python3 emit.py {phase}") for phase in ("baseline", "fault", "recovery")])
        self.assertEqual("fault_reproduced_and_recovered", result["status"])
        self.assertEqual([2], result["evidence_check"]["failed_records"])
        self.assertEqual([0, 0, 0], [row["exit_code"] for row in result["runs"]])
        self.assertEqual("local_command_output", result["evidence_check"]["provenance"])
        with self.assertRaises(ValueError):
            engineering.verify_commands(project, "feature-time", [(phase, "python3 -c 'print(1)'") for phase in ("baseline", "fault", "recovery")])

    def test_verify_commands_distinguishes_missing_failure_and_bad_recovery(self):
        project = self.root / "my-service"
        project.mkdir()
        (project / "emit.py").write_text(
            "import json, sys\n"
            "phase = sys.argv[1]\n"
            "if phase == 'exit': sys.exit(2)\n"
            "print(json.dumps({'budget_ms': 200, 'elapsed_ms': 10, 'candidates': 1 if phase == 'ok' else 0}))\n",
            encoding="utf-8")
        command = lambda phase, arg: (phase, f"python3 emit.py {arg}")
        no_fault = engineering.verify_commands(project, "serving", [command("baseline", "ok"),
                   command("fault", "ok"), command("recovery", "ok")])
        self.assertEqual("fault_not_reproduced", no_fault["status"])
        bad_recovery = engineering.verify_commands(project, "serving", [command("baseline", "ok"),
                       command("fault", "bad"), command("recovery", "bad")])
        self.assertEqual("baseline_or_recovery_failed", bad_recovery["status"])
        aborted = engineering.verify_commands(project, "serving", [command("baseline", "ok"),
                  command("fault", "exit"), command("recovery", "ok")])
        self.assertEqual("command_failed", aborted["status"])
        self.assertEqual(2, aborted["runs"][1]["exit_code"])

    def test_verify_repo_template_contracts(self):
        root = Path(__file__).resolve().parents[2] / "templates"
        feature = engineering.verify_commands(root / "feature-consistency", "feature-time", [
            ("baseline", "python3 app.py"), ("fault", "python3 app.py --inject-leakage"),
            ("recovery", "python3 app.py")])
        self.assertEqual("fault_rejected_and_recovered", feature["status"])
        release = engineering.verify_commands(root / "two-tower-ann", "release-version", [
            ("baseline", "python3 app.py --user u1"), ("fault", "python3 app.py --inject-version-mismatch"),
            ("recovery", "python3 app.py --user u1")])
        self.assertEqual("fault_rejected_and_recovered", release["status"])


if __name__ == "__main__":
    unittest.main()
