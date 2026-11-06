"""Reader-facing chapter code discovery is grounded in real published files."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import recsys_code
import recsys_navigation


class ChapterCodeDiscoveryTest(unittest.TestCase):
    def test_across_chapters_with_existing_sources_and_readmes(self):
        for topic, chapter, expected in (("负采样偏差", 10, "two_tower_debias.py"),
                                         ("MMR", 7, "mmr_diversification.py"),
                                         ("HNSW", 18, "hnsw.py")):
            with self.subTest(topic=topic):
                examples = recsys_code.search(ROOT, topic, chapter=chapter)
                self.assertTrue(examples)
                self.assertTrue(examples[0]["path"].endswith(expected))
                self.assertTrue((ROOT / examples[0]["path"]).is_file())
                self.assertTrue((ROOT / examples[0]["readme"]).is_file())

    def test_navigation_connects_concept_to_chapter_example(self):
        result = recsys_navigation.resolve("HNSW", root=ROOT)
        self.assertIn("ch18 向量检索系统/code/ch05_hnsw/hnsw.py", result["matches"][0]["code"])

    def test_unknown_topic_abstains_and_never_executes_source(self):
        self.assertEqual("not_found", recsys_code.report(ROOT, "不存在的算法zzzz")["status"])
        with self.assertRaises(ValueError):
            recsys_code.search(ROOT, "")

    def test_short_acronym_does_not_match_incidental_words(self):
        # "DIN" appears as a substring in 16+ unrelated filenames (e.g. cascading,
        # embedding, sliding); the tokeniser must reject it as an incidental match.
        self.assertEqual([], recsys_code.search(ROOT, "DIN"))
        # "SIM" appears in simhash, simcse, simulation; same guard.
        self.assertEqual([], recsys_code.search(ROOT, "SIM"))
        # "PLE" is the basename of ch12 ple.py — it is a legitimate exact match.
        ple_results = recsys_code.search(ROOT, "PLE")
        self.assertTrue(ple_results)
        self.assertTrue(ple_results[0]["path"].endswith("ple.py"))
        # "DeepFM" is an exact filename stem — legitimate match expected.
        deepfm_results = recsys_code.search(ROOT, "DeepFM")
        self.assertTrue(deepfm_results)
        self.assertTrue(deepfm_results[0]["path"].endswith("deepfm.py"))

    def test_changed_chapter_files_are_discovered_without_curated_hints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "ch24 演示章节/code"
            directory.mkdir(parents=True)
            (directory / "README.md").write_text("├── novel_example.py  新主题演示\n", encoding="utf-8")
            (directory / "novel_example.py").write_text("raise RuntimeError('must not run')\n", encoding="utf-8")
            matches = recsys_code.search(root, "新主题", chapter=24)
            self.assertEqual("ch24 演示章节/code/novel_example.py", matches[0]["path"])
            self.assertEqual([], recsys_code.search(root, "新主题", chapter=10))


if __name__ == "__main__":
    unittest.main()
