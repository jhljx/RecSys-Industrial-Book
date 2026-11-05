from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from recsys_navigation import NavigationIndex, resolve  # noqa: E402


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = NavigationIndex(ROOT)

    def test_alias_resolves_public_card_pdf_and_runs_code_paths(self):
        result = self.index.resolve("双塔召回")
        self.assertEqual(len(result["matches"]), 1)
        entity = result["matches"][0]
        self.assertEqual(entity["id"], "two-tower")
        self.assertTrue((ROOT / entity["concept_card"]).is_file())
        self.assertTrue((ROOT / entity["pdf"]).is_file())
        self.assertGreaterEqual(entity["pdf_page"], 1)
        self.assertTrue(entity["code"])
        for path in entity["code"]:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_published_paper_has_exact_pdf_page_pointer(self):
        result = self.index.resolve("DeepFM")
        self.assertTrue(result["matches"])
        match = result["matches"][0]
        self.assertEqual(match["paper_id"], "deepfm")
        self.assertEqual(match["paper_pages"], [14, 16])
        self.assertTrue((ROOT / match["pdf"]).is_file())

    def test_followup_uses_previous_entity_context(self):
        first = self.index.resolve("双塔召回")
        second = self.index.resolve("再给我代码", previous=first)
        self.assertTrue(second["used_context"])
        self.assertEqual(second["matches"][0]["id"], "two-tower")

    def test_unrelated_query_does_not_inherit_context(self):
        first = self.index.resolve("双塔召回")
        second = self.index.resolve("重排", previous=first)
        self.assertFalse(second["used_context"])
        self.assertEqual(second["matches"][0]["id"], "reranking")

    def test_unknown_query_stays_unresolved(self):
        self.assertEqual(self.index.resolve("未知的专有主题") ["matches"], [])

    def test_one_shot_api(self):
        self.assertTrue(resolve("MMR", root=ROOT)["matches"])

    def test_every_public_concept_resolves_to_released_evidence(self):
        from build_knowledge_map import build
        for concept in build(ROOT)["concepts"]:
            with self.subTest(concept=concept["id"]):
                row = next((item for item in self.index.resolve(concept["id"])["matches"]
                            if item["id"] == concept["id"]), None)
                self.assertIsNotNone(row)
                self.assertTrue((ROOT / row["pdf"]).is_file())
                self.assertTrue((ROOT / row["chapter_map"]).is_file())
        self.assertEqual(22, self.index.resolve("consistency-modeling")["matches"][0]["chapter"])
        self.assertEqual(5, self.index.resolve("consistency-modeling")["matches"][0]["evidence_chapter"])

    def test_code_pages_only_point_to_generated_pages(self):
        for topic in ("two-tower", "fallback", "point-in-time-join"):
            row = self.index.resolve(topic)["matches"][0]
            self.assertTrue(all((ROOT / "knowledge-map" / item["url"]).is_file() for item in row["code_pages"]))
            self.assertTrue(all((ROOT / "knowledge-map" / item["local_url"]).is_file() for item in row["template_guides"]))
            self.assertTrue(all((ROOT / "knowledge-map" / item["url"] / "README.md").is_file() for item in row["template_guides"]))


if __name__ == "__main__":
    unittest.main()
