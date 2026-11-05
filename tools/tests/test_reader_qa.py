from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import recsys_agent as agent
import recsys_qa as qa


class ReaderQATest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "ch10/book.pdf").parent.mkdir()
        (self.root / "ch10/book.pdf").write_bytes(b"%PDF fixture")
        self.page = {"id": 1, "chapter": 10, "source": "ch10/book.pdf", "page": 6,
                     "section": "协同过滤", "excerpt": "ItemCF 基于物品的相似性。"}

    def tearDown(self):
        self.temporary.cleanup()

    def test_render_citations_are_bound_to_selected_pages(self):
        answer = {"status": "answered", "conclusion": "ItemCF 从物品相似性出发。",
                  "points": [{"text": "它以物品相似性为线索。", "evidence_ids": [1]}],
                  "boundary": "稀疏时需结合上下文。", "follow_up": "如何评估？"}
        self.assertEqual(answer, qa.validate_answer(answer, [self.page]))
        rendered = qa.render_answer("ItemCF 是什么", answer, [self.page])
        self.assertIn("ch10/book.pdf，PDF 第 6 页", rendered)
        self.assertIn("可以接着问", rendered)
        for ids in ([], [2], [True]):
            with self.subTest(ids=ids), self.assertRaises(RuntimeError):
                qa.validate_answer({**answer, "points": [{"text": "猜测", "evidence_ids": ids}]}, [self.page])

    def test_uncovered_question_never_calls_model(self):
        with patch.object(qa, "evidence_pages", return_value=[]), patch.object(qa, "model_answer") as model:
            text, answer, pages = qa.answer_question(self.root, "量子菠菜引擎")
        model.assert_not_called()
        self.assertEqual("insufficient_evidence", answer["status"])
        self.assertEqual([], pages)
        self.assertIn("不足以可靠回答", text)

    def test_unreleased_chapter_pdf_question_abstains_without_model(self):
        with patch.object(qa, "model_answer") as model:
            _, answer, pages = qa.answer_question(TOOLS.parent, "第十四章正式出版的 PDF 第 1 页是什么？")
        self.assertEqual("insufficient_evidence", answer["status"])
        self.assertEqual([], pages)
        model.assert_not_called()

    def test_concept_navigation_reuses_answer_session(self):
        root = TOOLS.parent
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            with patch.object(agent, "workspace", return_value=workspace), patch.object(
                qa, "answer_question", return_value=("书中回答", {"status": "answered"}, [self.page])
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    agent.reader_ask(root, "双塔召回是什么？", "topic", "claude", 120, False)
            output = io.StringIO()
            with patch.object(agent, "workspace", return_value=workspace), contextlib.redirect_stdout(output):
                agent.reader_navigate(root, "再给我代码", "topic")
            report = json.loads(output.getvalue())
            self.assertEqual("two-tower", report["matches"][0]["id"])
            self.assertTrue(report["used_context"])

    def test_session_remembers_topic_without_pdf_extracts(self):
        with patch.object(qa, "answer_question", return_value=("书中回答", {"status": "answered"}, [self.page])) as ask:

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                agent.reader_ask(self.root, "ItemCF 是什么", "lesson", "claude", 120, False)
                agent.reader_ask(self.root, "它的局限呢", "lesson", "claude", 120, False)
        self.assertEqual("ItemCF 是什么", ask.call_args.kwargs["prior_topic"])
        record = (self.root / ".recsys-workspace/reader-sessions/lesson.json").read_text(encoding="utf-8")
        self.assertNotIn("excerpt", record)
        self.assertEqual("ItemCF 是什么", json.loads(record)["topic"])
        with self.assertRaises(SystemExit):
            agent.reader_session_path(self.root, "../elsewhere")

    def test_followup_receives_short_answer_summary(self):
        first = {"status": "answered", "points": [{"text": "第一种取舍较稳健。", "evidence_ids": [1]}], "boundary": "规模较小时"}
        with patch.object(qa, "answer_question", side_effect=[("回答", first, [self.page]), ("续问", first, [self.page])]) as ask:
            with contextlib.redirect_stdout(io.StringIO()):
                agent.reader_ask(self.root, "两种方案怎么选", "lesson", "claude", 120, False)
                agent.reader_ask(self.root, "第二种呢", "lesson", "claude", 120, False)
        self.assertIn("第一种取舍", ask.call_args.kwargs["context"])
        record = (self.root / ".recsys-workspace/reader-sessions/lesson.json").read_text(encoding="utf-8")
        self.assertNotIn("ItemCF 基于", record)

    def test_explicit_new_concept_resets_session_without_retrieval_guess(self):
        cards = self.root / "knowledge/concepts"
        cards.mkdir(parents=True)
        (cards / "two-tower.md").write_text("title: 双塔召回\n", encoding="utf-8")
        (cards / "negative-sampling.md").write_text("title: 负采样\n", encoding="utf-8")
        reply = {"status": "answered", "points": [], "boundary": ""}
        with patch.object(qa, "answer_question", return_value=("回答", reply, [self.page])) as ask:
            with contextlib.redirect_stdout(io.StringIO()):
                agent.reader_ask(self.root, "双塔召回是什么", "lesson", "claude", 120, False)
                agent.reader_ask(self.root, "它的局限呢", "lesson", "claude", 120, False)
            self.assertEqual("双塔召回是什么", ask.call_args.kwargs["prior_topic"])
            with contextlib.redirect_stdout(io.StringIO()):
                agent.reader_ask(self.root, "负采样是什么", "lesson", "claude", 120, False)
            self.assertEqual("", ask.call_args.kwargs["prior_topic"])
            self.assertEqual("", ask.call_args.kwargs["context"])

    def test_codex_adapter_uses_read_only_schema_and_output_file(self):
        from subprocess import CompletedProcess

        def fake_run(command, **kwargs):
            self.assertIn("read-only", command)
            self.assertIn("--output-schema", command)
            Path(command[command.index("-o") + 1]).write_text('{"status":"insufficient_evidence"}', encoding="utf-8")
            return CompletedProcess(command, 0, "", "")

        with patch.object(qa.shutil, "which", return_value="/usr/local/bin/codex"), patch.object(qa.subprocess, "run", side_effect=fake_run):
            result = qa.invoke_model("question", qa.ANSWER_SCHEMA, cli="codex")
        self.assertEqual("insufficient_evidence", result["status"])


if __name__ == "__main__":
    unittest.main()
