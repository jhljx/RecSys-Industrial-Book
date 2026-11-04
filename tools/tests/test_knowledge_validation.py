"""Regression checks for curated PDF citations, course readings, and dependency graph."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import check_knowledge  # noqa: E402


class KnowledgeValidationTest(unittest.TestCase):
    def run_with(self, original: Path, modified: str) -> str:
        original_read = Path.read_text
        with tempfile.TemporaryDirectory() as directory:
            replacement = Path(directory) / original.name
            replacement.write_text(modified, encoding="utf-8")
            with patch.object(Path, "read_text", autospec=True) as read:
                def content(path, *args, **kwargs):
                    return original_read(replacement if path == original else path, *args, **kwargs)
                read.side_effect = content
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                    check_knowledge.main()
                return output.getvalue()

    def test_unreleased_concept_must_cite_released_pdf(self) -> None:
        path = ROOT / "knowledge/concepts/consistency-modeling.md"
        changed = path.read_text(encoding="utf-8").replace("第五章-粗排模块.pdf", "第二十二章-链路一致性建模.pdf")
        self.assertIn("unreleased PDF citation", self.run_with(path, changed))

    def test_concept_citation_must_stay_within_released_pages(self) -> None:
        path = ROOT / "knowledge/concepts/pre-ranking-chain-consistency.md"
        changed = path.read_text(encoding="utf-8").replace("PDF 第 2–4 页", "PDF 第 200–201 页")
        self.assertIn("invalid PDF page range", self.run_with(path, changed))

    def test_course_reading_out_of_bounds_fails(self) -> None:
        path = ROOT / "learning-paths/courses.json"
        original = path.read_text(encoding="utf-8")
        changed = original.replace("ch01 PDF 5-10", "ch01 PDF 5-28", 1)
        self.assertIn("reading pages 5-28 out of bounds", self.run_with(path, changed))

    def test_course_reading_must_match_unit_chapters(self) -> None:
        path = ROOT / "learning-paths/courses.json"
        original = path.read_text(encoding="utf-8")
        changed = original.replace("ch01 PDF 5-10", "ch22 PDF 5-10", 1)
        self.assertIn("reading references chapter 22", self.run_with(path, changed))

    def test_graph_edge_disagreement_fails(self) -> None:
        path = ROOT / "knowledge/graph.json"
        graph = json.loads(path.read_text(encoding="utf-8"))
        graph["edges"].pop()
        self.assertIn("edges differ from card prerequisites", self.run_with(path, json.dumps(graph)))

    def test_prerequisite_cycle_fails_even_when_graph_agrees(self) -> None:
        path = ROOT / "knowledge/concepts/industrial-recsys-scope.md"
        text = path.read_text(encoding="utf-8")
        self.assertIn("prerequisites: []", text)
        changed = text.replace("prerequisites: []", "prerequisites: [feedback-loop]", 1)
        self.assertIn("prerequisite cycle", self.run_with(path, changed))


if __name__ == "__main__":
    unittest.main()
