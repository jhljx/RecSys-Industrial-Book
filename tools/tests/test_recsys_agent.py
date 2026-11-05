from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        index_path = index_dir / "index.jsonl"
        index_path.write_text(json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8")
        source = self.root / chunk["source"]
        source.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-1.4 test fixture")
        manifest = {"format": recsys_agent.FORMAT_VERSION, "mode": "public", "chunks": 1,
                    "index_sha256": recsys_agent.sha256(index_path), "sources": recsys_agent.source_records(self.root, "public")}
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

    def test_only_explicit_unreleased_pdf_requests_are_rejected(self) -> None:
        root = TOOLS.parent
        self.assertTrue(recsys_agent.requests_unreleased_pdf(root, "第十三章完整原书页码"))
        self.assertTrue(recsys_agent.requests_unreleased_pdf(root, "第22章已发布PDF第1页"))
        self.assertFalse(recsys_agent.requests_unreleased_pdf(root, "第十二章已发布PDF第1页"))
        self.assertFalse(recsys_agent.requests_unreleased_pdf(root, "第22章尚未发布，第5章如何解释链路一致性？"))

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

    def test_page_evidence_eval_rejects_wrong_page(self) -> None:
        from unittest.mock import patch

        dataset = self.root / "page-evidence.jsonl"
        dataset.write_text(json.dumps({"query": "推荐系统", "expected_chapters": [1],
                                       "expected_pages": {"1": [[2, 3]]}}, ensure_ascii=False) + "\n", encoding="utf-8")
        chunk = recsys_agent.Chunk("one", "ch01/book.pdf", "PDF page 1", "title", "text", 1, "section", 1, 1)
        with patch.object(recsys_agent, "rank", return_value=[(1.0, chunk)]):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                recsys_agent.evaluate(self.root, dataset)
        report = json.loads(output.getvalue())
        self.assertEqual(0.0, report["page_hit_at_1"])
        self.assertEqual(1, len(report["page_misses"]))
        dataset.write_text(json.dumps({"query": "推荐系统", "expected_chapters": [1],
                                       "expected_pages": {"1": [[1, 1]]}}, ensure_ascii=False) + "\n", encoding="utf-8")
        with patch.object(recsys_agent, "rank", return_value=[(1.0, chunk)]):
            self.assertEqual(1.0, self.capture_json(recsys_agent.evaluate, self.root, dataset)["page_hit_at_1"])

    def test_overlapping_aliases_preserve_distinct_intents(self) -> None:
        root = TOOLS.parent
        aliases = recsys_agent.load_aliases(root)
        cases = (
            ("ItemCF 协同过滤召回稀疏交互与热门偏置", {"协同过滤召回"}),
            ("UserCF 协同过滤", {"协同过滤召回"}),
            ("协同过滤召回后过滤已看内容", {"协同过滤召回", "过滤"}),
            ("协同过滤和候选过滤", {"协同过滤召回", "过滤"}),
            ("re-ranking", {"重排"}),
            ("ranking", {"精排"}),
            ("pre-ranking", {"粗排"}),
            ("training-serving consistency", {"离在线一致性"}),
            ("overload", {"限流"}),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(expected, set(recsys_agent.matched_alias_families(aliases, query)))
                expanded = recsys_agent.expand_query(root, query)
                if "过滤" not in expected:
                    self.assertNotIn("候选过滤", expanded)
                    self.assertNotIn("过滤限流模块", expanded)
        self.assertEqual([], recsys_agent.matched_alias_families(aliases, "pre-rankingx"))

    def test_chinese_alias_containment_does_not_match_unrelated_family(self) -> None:
        aliases = {"过滤": ["filtering"], "协同过滤": ["ItemCF"]}
        self.assertEqual(["协同过滤"], recsys_agent.matched_alias_families(aliases, "协同过滤"))
        self.assertEqual(["过滤", "协同过滤"], recsys_agent.matched_alias_families(aliases, "协同过滤再做过滤"))

    def test_itemcf_query_has_no_filtering_chapter_hint(self) -> None:
        root = TOOLS.parent
        query = "ItemCF 协同过滤召回稀疏交互与热门偏置"
        self.assertEqual({10}, recsys_agent.concept_chapter_hints(root, query))
        self.assertEqual({10}, recsys_agent.concept_chapter_hints(root, "协同过滤"))
        self.assertEqual({10, 4}, recsys_agent.concept_chapter_hints(root, "协同过滤召回后过滤已看内容"))
        self.assertEqual({11}, recsys_agent.concept_chapter_hints(root, "粗排双塔独立编码与交互损失"))
        self.assertEqual(10, recsys_agent.rank(root, query)[0][1].chapter)

    def test_specific_concept_title_beats_generic_alias_chapter_hint(self) -> None:
        from unittest.mock import patch

        cards = self.root / "knowledge/concepts"
        cards.mkdir(parents=True)
        (cards / "collaborative-filtering-recall.md").write_text("title: 协同过滤召回\nchapter: 10\n", encoding="utf-8")
        (cards / "filtering.md").write_text("title: 过滤\nchapter: 4\n", encoding="utf-8")
        aliases = self.root / "knowledge/aliases.json"
        aliases.write_text('{"过滤": ["filtering"]}', encoding="utf-8")
        chunks = [recsys_agent.Chunk(str(ch), f"ch{ch}/book.pdf", "PDF page 1", "title", "过滤 协同过滤召回", ch, "section", 1, 1) for ch in (4, 10)]
        with patch.object(recsys_agent, "load", return_value=chunks):
            ranked = recsys_agent.rank(self.root, "协同过滤召回稀疏交互")
        self.assertEqual(10, ranked[0][1].chapter)

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
        self.assertEqual("draft", report["status"])
        self.assertEqual("not_run", report["verification"])
        self.assertEqual(10_000, report["capacity"]["offered_qps"])
        self.assertEqual(100_000_000, report["capacity"]["items"])
        self.assertEqual(2_000, report["capacity"]["minimum_inflight_at_p99"])
        self.assertIn("not a mathematical minimum", report["capacity"]["inflight_note"])
        self.assertIn("consistency", report["pending_business_confirmation"])
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
        self.assertEqual("draft", json.loads(result.stdout)["status"])
        self.assertEqual("not_run", json.loads(result.stdout)["verification"])

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
        self.assertEqual(["static_candidate", "static_candidate"], [item["severity"] for item in report["findings"]])

    def test_eval_rejects_irrelevant_query_without_positive_cases(self) -> None:
        dataset = self.root / "negative.jsonl"
        dataset.write_text(json.dumps({"query": "zxqvplm unrelated", "expected_chapters": []}) + "\n", encoding="utf-8")
        report = self.capture_json(recsys_agent.evaluate, self.root, dataset)
        self.assertEqual(1.0, report["negative_rejection_rate"])
        self.assertIsNone(report["hit_at_1"])
        dataset.write_text(json.dumps({"query": "推荐系统", "expected_chapters": []}, ensure_ascii=False) + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            recsys_agent.evaluate(self.root, dataset)

    def test_corrupt_index_never_reports_ready(self) -> None:
        index_path = self.root / ".recsys-agent/index.jsonl"
        manifest_path = self.root / ".recsys-agent/manifest.json"
        for content in ("", "{invalid json\n"):
            with self.subTest(content=content):
                index_path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "索引已过期"):
                    recsys_agent.require_fresh_index(self.root, "public")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    recsys_agent.doctor(self.root, "public")
                self.assertIn("index=stale", output.getvalue())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["index_sha256"] = recsys_agent.sha256(index_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIn("malformed", recsys_agent.stale_reasons(self.root, "public")[0])

    def test_course_submit_cannot_skip_or_undo_completed_unit(self) -> None:
        courses = self.root / "learning-paths"
        courses.mkdir()
        (courses / "courses.json").write_text(json.dumps({"courses": {"new-engineer": {"title": "测试课程", "units": [{"id": "ne-01", "title": "第一单元"}, {"id": "ne-02", "title": "第二单元"}]}}}), encoding="utf-8")
        evidence = self.root / "evidence.md"
        evidence.write_text("## 本书依据\nPDF 第 1 页\n## 我的解释\n机制\n## 实践\n主路径命令：test\n主路径退出码：0\n主路径输出：ok\n失败路径命令：test --fail\n失败路径退出码：2\n失败路径输出：fail\n失败原因：输入错误\n恢复条件：输入正确\n## 问题答案\n结论\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "当前课程单元"):
            recsys_agent.course_submit(self.root, "new-engineer", "ne-02", evidence, 0.9, True, True)
        self.capture_json(recsys_agent.course_submit, self.root, "new-engineer", "ne-01", evidence, 0.9, True, True)
        with self.assertRaisesRegex(SystemExit, "当前课程单元"):
            recsys_agent.course_submit(self.root, "new-engineer", "ne-01", evidence, 0.1, False, False)
        status = self.capture_json(recsys_agent.course_command, self.root, "status", "new-engineer", None, None)
        self.assertEqual(1, status["completed_units"])
        self.assertEqual("ne-02", status["current_unit"])
        self.assertEqual("completed", status["units"][0]["status"])

    def test_course_feedback_and_design_evidence(self) -> None:
        path = self.root / "learning-paths"
        path.mkdir()
        (path / "courses.json").write_text(json.dumps({"courses": {"new-engineer": {
            "title": "测试课程", "default_minutes": 90, "units": [{"id": "ne-01", "title": "方案设计", "concepts": ["a", "b"],
            "chapters": [], "reading": "", "deliverable": "方案", "acceptance": [], "questions": []}]}}}), encoding="utf-8")
        recsys_agent.append_progress(self.root, {"record_type": "course", "action": "start", "course_id": "new-engineer", "status": "started"})
        recsys_agent.append_progress(self.root, {"record_type": "course", "action": "feedback", "course_id": "new-engineer", "unit_id": "ne-01", "feedback": "需要补基础"})
        with patch("recsys_code.search", return_value=[]):
            today = self.capture_json(recsys_agent.course_command, self.root, "today", "new-engineer", None, None)
        self.assertEqual("需要补基础", today["feedback_context"])
        self.assertEqual("design", today["evidence_kind"])
        self.assertEqual(["a", "b"], today["checkpoints"][0]["concepts"])
        self.assertEqual("ne-01-1", today["current_checkpoint"])
        self.capture_json(recsys_agent.course_checkpoint, self.root, "new-engineer", "ne-01-1")
        with patch("recsys_code.search", return_value=[]):
            resumed = self.capture_json(recsys_agent.course_command, self.root, "today", "new-engineer", None, None)
        self.assertIsNone(resumed["current_checkpoint"])
        evidence = self.root / "design.md"
        evidence.write_text("## 本书依据\nPDF 第 1 页\n## 我的解释\n机制\n## 方案验证\n检验方法：对照基线\n反例或失败条件：流量异常\n验收依据：候选覆盖率\n## 问题答案\n结论\n", encoding="utf-8")
        result = self.capture_json(recsys_agent.course_submit, self.root, "new-engineer", "ne-01", evidence, 0.9, False, False)
        self.assertEqual("completed", result["status"])
        overview = self.capture_json(recsys_agent.learning_overview, self.root)
        self.assertEqual(1, overview["courses"]["new-engineer"]["completed_units"])

    def test_checkpoint_feedback_keeps_published_order_and_resumes(self) -> None:
        unit = {"id": "ne-05", "concepts": ["a", "b", "c", "d", "e", "f", "g"]}
        cards = self.root / "knowledge/concepts"
        cards.mkdir(parents=True)
        (cards / "g.md").write_text("title: 最后概念\n", encoding="utf-8")
        checkpoints = recsys_agent.course_checkpoints(unit, "最后概念仍然困难", self.root)
        self.assertEqual(["a", "b", "c", "d", "e"], checkpoints[0]["concepts"])
        self.assertEqual(["f", "g"], checkpoints[1]["concepts"])
        self.assertIn("重点复核", checkpoints[1]["task"])
        self.assertNotIn("重点复核", checkpoints[0]["task"])
        (self.root / "learning-paths").mkdir()
        (self.root / "learning-paths/courses.json").write_text(json.dumps({"courses": {"new-engineer": {
            "title": "测试", "default_minutes": 60, "units": [unit]}}}), encoding="utf-8")
        self.capture_json(recsys_agent.course_checkpoint, self.root, "new-engineer", "ne-05-1")
        with self.assertRaises(SystemExit):
            self.capture_json(recsys_agent.course_checkpoint, self.root, "new-engineer", "ne-05-1")
        with patch("recsys_code.search", return_value=[]):
            today = self.capture_json(recsys_agent.course_command, self.root, "today", "new-engineer", None, None)
        self.assertEqual("ne-05-2", today["current_checkpoint"])

    def test_unrelated_explain_and_course_progress_filter(self) -> None:
        self.assertEqual("needs_source", self.capture_json(recsys_agent.explain, self.root, "zxqvplm unrelated", "new-engineer")["status"])
        recsys_agent.append_progress(self.root, {"record_type": "course", "unit_id": "ne-01", "status": "completed"})
        (self.root / "knowledge").mkdir()
        (self.root / "knowledge/graph.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
        report = self.capture_json(recsys_agent.adaptive_next, self.root)
        self.assertNotIn("None", report["completed"])

    def test_review_and_debug_do_not_claim_live_verification(self) -> None:
        target = self.root / "unreachable.py"
        target.write_text("if False:\n    eligible = feature.timestamp <= label.timestamp\n", encoding="utf-8")
        report = self.capture_json(recsys_agent.review_surface, self.root, target)
        self.assertEqual("needs_runtime_verification", report["status"])
        self.assertEqual("static_candidate", report["findings"][0]["severity"])
        report = self.capture_json(recsys_agent.debug_scenario, self.root, "召回 P99 延迟")
        self.assertEqual("needs_evidence", report["status"])
        self.assertIsNone(report["fixture_evidence"])
        self.assertIn("traffic", report["diagnostic_tree"][0]["hypothesis"])
        self.assertIsNotNone(self.capture_json(recsys_agent.debug_scenario, self.root, "召回 P99 延迟", True)["fixture_evidence"])

    def test_stale_index_and_wrong_mode_are_rejected(self) -> None:
        recsys_agent.require_fresh_index(self.root, "public")
        ready = io.StringIO()
        with contextlib.redirect_stdout(ready):
            recsys_agent.doctor(self.root, "public")
        self.assertIn("index=ready", ready.getvalue())
        with self.assertRaisesRegex(SystemExit, "没有可用知识源"):
            recsys_agent.require_fresh_index(self.root, "private")
        unavailable = io.StringIO()
        with contextlib.redirect_stdout(unavailable):
            recsys_agent.doctor(self.root, "private")
        self.assertIn("index=stale", unavailable.getvalue())
        self.assertIn("没有可用知识源", unavailable.getvalue())
        (self.root / "ch01 推荐系统概述/ch01.pdf").write_bytes(b"changed")
        with self.assertRaisesRegex(SystemExit, "索引已过期"):
            recsys_agent.require_fresh_index(self.root, "public")
        stale = io.StringIO()
        with contextlib.redirect_stdout(stale):
            recsys_agent.doctor(self.root, "public")
        self.assertIn("index=stale", stale.getvalue())

    def test_doctor_detects_mode_mismatch_even_with_same_sources(self) -> None:
        manifest_path = self.root / ".recsys-agent/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["mode"] = "private"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "mode private != public"):
            recsys_agent.require_fresh_index(self.root, "public")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            recsys_agent.doctor(self.root, "public")
        self.assertIn("index=stale", output.getvalue())
        self.assertIn("mode private != public", output.getvalue())

    def test_doctor_reports_malformed_manifest_as_stale(self) -> None:
        (self.root / ".recsys-agent/manifest.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "索引元数据不可用"):
            recsys_agent.require_fresh_index(self.root, "public")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            recsys_agent.doctor(self.root, "public")
        self.assertIn("index=stale", output.getvalue())
        self.assertIn("索引元数据不可用", output.getvalue())

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
