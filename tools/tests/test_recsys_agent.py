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
        source = self.root / chunk["source"]
        source.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-1.4 test fixture")
        manifest = {"format": recsys_agent.FORMAT_VERSION, "mode": "public", "sources": recsys_agent.source_records(self.root, "public")}
        (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def capture_json(self, function, *args) -> dict[str, object]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            function(*args)
        return json.loads(output.getvalue())

    def test_eval_enforces_top1_threshold_and_reports_misses(self) -> None:
        dataset = self.root / "eval.jsonl"
        dataset.write_text(json.dumps({"query": "推荐系统", "expected_chapters": [2]}, ensure_ascii=False) + "\n", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as error:
            recsys_agent.evaluate(self.root, dataset)
        self.assertEqual(1, error.exception.code)
        report = json.loads(output.getvalue())
        self.assertEqual(0.0, report["hit_at_1"])
        self.assertEqual(1.0, report["min_hit_at_1"])
        self.assertEqual([{"query": "推荐系统", "expected": [2], "actual": 1}], report["top1_misses"])
        self.assertEqual(1, len(report["failures"]))
        with self.assertRaisesRegex(SystemExit, "有限数值"):
            recsys_agent.evaluate(self.root, dataset, float("nan"))
        dataset.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "评测集为空"):
            recsys_agent.evaluate(self.root, dataset)

    def test_eval_allows_configured_top1_misses_but_not_recall_failures(self) -> None:
        from unittest.mock import patch

        dataset = self.root / "eval.jsonl"
        cases = [{"query": "first", "expected_chapters": [1]}, {"query": "second", "expected_chapters": [2]}]
        dataset.write_text("\n".join(json.dumps(row) for row in cases) + "\n", encoding="utf-8")
        original = recsys_agent.Chunk("first", "ch01/a.pdf", "PDF page 1", "title", "text", 1, "section", 1, 1)
        other = recsys_agent.Chunk("second", "ch02/b.pdf", "PDF page 2", "title", "text", 2, "section", 2, 2)
        with patch.object(recsys_agent, "rank", side_effect=[[(1.0, original)], [(2.0, original), (1.0, other)]]):
            report = self.capture_json(recsys_agent.evaluate, self.root, dataset, 0.5)
        self.assertEqual(0.5, report["hit_at_1"])
        self.assertEqual(1.0, report["recall_at_5"])
        self.assertEqual([], report["failures"])
        self.assertEqual(1, len(report["top1_misses"]))
        with patch.object(recsys_agent, "rank", side_effect=[[(1.0, original)], [(2.0, original), (1.0, other)]]):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                recsys_agent.evaluate(self.root, dataset, 0.75)

    def test_specific_concept_title_beats_generic_alias_chapter_hint(self) -> None:
        from unittest.mock import patch

        cards = self.root / "knowledge/concepts"
        cards.mkdir(parents=True)
        (cards / "collaborative-filtering-recall.md").write_text("title: 协同过滤召回\nchapter: 10\n", encoding="utf-8")
        (cards / "filtering.md").write_text("title: 过滤\nchapter: 4\n", encoding="utf-8")
        (self.root / "knowledge/aliases.json").write_text('{"过滤": ["filtering"]}', encoding="utf-8")
        chunks = [recsys_agent.Chunk(str(ch), f"ch{ch}/book.pdf", "PDF page 1", "title", "过滤 协同过滤召回", ch, "section", 1, 1) for ch in (4, 10)]
        with patch.object(recsys_agent, "load", return_value=chunks):
            ranked = recsys_agent.rank(self.root, "ItemCF 协同过滤召回稀疏交互")
        self.assertEqual(10, ranked[0][1].chapter)

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
    def test_stale_index_and_wrong_mode_are_rejected(self) -> None:
        recsys_agent.require_fresh_index(self.root, "public")
        with self.assertRaisesRegex(SystemExit, "没有可用知识源"):
            recsys_agent.require_fresh_index(self.root, "private")
        (self.root / "ch01 推荐系统概述/ch01.pdf").write_bytes(b"changed")
        with self.assertRaisesRegex(SystemExit, "索引已过期"):
            recsys_agent.require_fresh_index(self.root, "public")

    def test_specific_model_concept_wins_over_generic_alias(self) -> None:
        concepts = self.root / "knowledge/concepts"
        concepts.mkdir(parents=True)
        for identifier, title in (("multi-objective", "多目标建模"), ("multi-task-experts", "多任务专家模型")):
            (concepts / f"{identifier}.md").write_text(f"title: {title}\n", encoding="utf-8")
        (self.root / "knowledge/aliases.json").write_text('{"多目标": ["多任务"]}', encoding="utf-8")
        self.assertEqual("multi-task-experts", recsys_agent.concept_id(self.root, "PLE 多任务专家"))
        self.assertEqual("multi-objective", recsys_agent.concept_id(self.root, "多目标"))

    def test_evidence_rejects_nonempty_stub_and_accepts_structured_log(self) -> None:
        evidence = self.root / "evidence.md"
        evidence.write_text("学完了", encoding="utf-8")
        self.assertTrue(recsys_agent.evidence_gaps(evidence))
        evidence.write_text(
            "## 本书依据\n第十二章 PDF 第 34 页：共享与专属专家。\n"
            "## 我的解释\n共享和专属路径分离。\n"
            "## 实践\n主路径命令：python app.py\n主路径退出码：0\n主路径输出：ok\n"
            "失败路径命令：python app.py --fail\n失败路径退出码：2\n失败路径输出：rejected\n"
            "失败原因：版本不匹配\n恢复条件：版本一致\n"
            "## 问题答案\n共享收益需要同预算消融。\n", encoding="utf-8"
        )
        self.assertEqual([], recsys_agent.evidence_gaps(evidence))

    def test_review_submission_closes_due_task_and_self_eval_count(self) -> None:
        courses = self.root / "learning-paths"
        courses.mkdir()
        (courses / "courses.json").write_text(json.dumps({"courses": {"new-engineer": {"title": "测试课程", "units": [{"id": "ne-01", "title": "第一单元"}, {"id": "ne-02", "title": "第二单元"}]}}}), encoding="utf-8")
        recsys_agent.append_progress(self.root, {"record_type": "course", "action": "submit", "course_id": "new-engineer", "unit_id": "ne-01", "status": "completed", "review_due": "2020-01-01"})
        self.assertEqual(1, len(recsys_agent.due_reviews(self.root)))
        evidence = self.root / "review.md"
        evidence.write_text("## 机制回忆\n机制。\n## 错误设计\n错误。\n## 参数预测\n结果。\n## 迁移取舍\n取舍。\n", encoding="utf-8")
        result = self.capture_json(recsys_agent.course_review, self.root, "new-engineer", "ne-01", evidence, 0.9)
        self.assertEqual("completed", result["status"])
        self.assertEqual([], recsys_agent.due_reviews(self.root))
        status = self.capture_json(recsys_agent.course_command, self.root, "status", "new-engineer", None, None)
        self.assertEqual(1, status["completed_units"])
        with self.assertRaises(SystemExit):
            self.capture_json(recsys_agent.course_submit, self.root, "new-engineer", "ne-02", self.root / "missing.md", float("nan"), True, True)
        status = self.capture_json(recsys_agent.course_command, self.root, "status", "new-engineer", None, None)
        self.assertEqual(1, status["completed_units"])
