from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import recsys_agent  # noqa: E402


class RecSysAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        index_dir = self.root / ".recsys-agent"
        index_dir.mkdir()
        chunk = {
            "id": "ch01:p1:1",
            "source": "ch01 推荐系统概述/ch01.pdf",
            "locator": "PDF page 1",
            "title": "推荐系统",
            "text": "推荐系统通过召回和排序为用户选择内容。",
            "chapter": 1,
            "section": "推荐系统概述",
            "page_start": 1,
            "page_end": 1,
            "kind": "prose",
        }
        (index_dir / "index.jsonl").write_text(json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def capture_json(self, function, *args) -> dict[str, object]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            function(*args)
        return json.loads(output.getvalue())

    def test_explain_cli_returns_audience_guidance_and_pdf_evidence(self) -> None:
        script = TOOLS / "recsys_agent.py"
        result = subprocess.run(
            [sys.executable, str(script), "--root", str(self.root), "explain", "推荐系统", "--audience", "new-engineer"],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual("ready", report["status"])
        self.assertEqual("new-engineer", report["audience"])
        self.assertEqual(1, report["evidence"][0]["page"])

    def test_design_rejects_non_numeric_and_non_positive_constraints(self) -> None:
        report = self.capture_json(
            recsys_agent.design,
            self.root,
            "在线推荐",
            ["objective=click", "qps=not-a-number", "p99=0", "items=x", "freshness=5m", "availability=120%", "cost=bounded"],
        )
        self.assertEqual("needs_input", report["status"])
        self.assertEqual(["qps", "p99", "items", "availability"], report["invalid"])
        self.assertIsNone(report["capacity"]["minimum_inflight_at_p99"])

    def test_design_accepts_documented_scientific_notation_and_preserves_budget(self) -> None:
        report = self.capture_json(
            recsys_agent.design,
            self.root,
            "在线推荐",
            ["objective=click", "qps=1e4", "p99=200ms", "items=1e8", "freshness=5m", "availability=99.99%", "cost=bounded"],
        )
        self.assertEqual("complete", report["status"])
        self.assertEqual(10_000, report["capacity"]["offered_qps"])
        self.assertEqual(100_000_000, report["capacity"]["items"])
        self.assertEqual(2_000, report["capacity"]["minimum_inflight_at_p99"])
        self.assertAlmostEqual(200, sum(item["p99_ms"] for item in report["latency_budget"]))

    def test_design_cli_emits_json_for_valid_constraints(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "recsys_agent.py"),
                "--root",
                str(self.root),
                "design",
                "在线推荐",
                "objective=click",
                "qps=10000",
                "p99=200ms",
                "items=1e8",
                "freshness=5m",
                "availability=99.99%",
                "cost=bounded",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual("complete", json.loads(result.stdout)["status"])

    def test_review_does_not_treat_comments_or_strings_as_passes(self) -> None:
        target = self.root / "comment_only.py"
        target.write_text(
            '# feature.timestamp <= label.timestamp\n'
            'message = \'row["feature_time"] > row["label_time"]\'\n',
            encoding="utf-8",
        )
        report = self.capture_json(recsys_agent.review_surface, self.root, target)
        self.assertTrue(all(item["severity"] != "pass" for item in report["findings"]))

    def test_review_recognizes_python_comparison_expressions(self) -> None:
        target = self.root / "checks.py"
        target.write_text(
            "eligible = feature.timestamp <= label.timestamp\n"
            "leaked = row[\"feature_time\"] > row[\"label_time\"]\n",
            encoding="utf-8",
        )
        report = self.capture_json(recsys_agent.review_surface, self.root, target)
        self.assertEqual(["pass", "pass"], [item["severity"] for item in report["findings"]])


if __name__ == "__main__":
    unittest.main()
