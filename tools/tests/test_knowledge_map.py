from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import build_knowledge_map  # noqa: E402


class KnowledgeMapTest(unittest.TestCase):
    def test_chapter_focus_keeps_paper_specific_cards_out_of_overview(self) -> None:
        data = build_knowledge_map.build()
        focus = data["conceptFocus"]["20"]
        chapter_cards = {row["id"] for row in data["concepts"] if row["chapter"] == 20}
        self.assertEqual(11, len(focus))
        self.assertEqual(len(focus), len(set(focus)))
        self.assertTrue(set(focus) <= chapter_cards)
        self.assertNotIn("din-local-activation", focus)
        self.assertIn("din-local-activation", chapter_cards)
        self.assertIn("sequence-attention-overview", focus)
        self.assertTrue(any(a in focus and b in focus for a, b in data["edges"]))

    def test_chapter_focus_rejects_cross_chapter_card(self) -> None:
        data = build_knowledge_map.build()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge").mkdir()
            (root / "knowledge/concept-focus.json").write_text(
                json.dumps({"version": 1, "chapters": {"20": ["semantic-id"]}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid concept focus for chapter 20"):
                build_knowledge_map.load_concept_focus(root, data["chapters"], data["concepts"])

    def test_paper_hierarchy_has_one_parent_per_paper(self) -> None:
        data = build_knowledge_map.build()
        hierarchy = data["paperHierarchy"]
        self.assertEqual({paper["id"] for paper in data["papers"]},
                         {paper_id for groups in hierarchy.values() for group in groups for paper_id in group["papers"]})
        self.assertEqual(7, len(hierarchy["20"]))
        self.assertEqual(28, sum(len(group["papers"]) for group in hierarchy["20"]))
        self.assertEqual(["rq-vae"], next(group["papers"] for group in hierarchy["9"]
                                          if group["concept"] == "semantic-id"))
        self.assertEqual({"rq-kmeans-tokenizer", "pq-semantic-tokenizer", "fsq-tokenizer", "lfq-tokenizer"},
                         set(hierarchy["9"][0]["extraConcepts"]))
        self.assertEqual({"diger", "etegrec", "unisid"}, set(hierarchy["9"][1]["papers"]))
        for paper in data["papers"]:
            if paper["chapter"] == 9 and paper["id"] != "rq-vae":
                self.assertNotIn("semantic-id", paper["concepts"])
                self.assertGreater(len(paper["concepts"]), 1)

    def test_paper_hierarchy_rejects_duplicate_placement(self) -> None:
        data = build_knowledge_map.build()
        catalog = {"version": 1, "chapters": json.loads(json.dumps(data["paperHierarchy"]))}
        catalog["chapters"]["23"][1]["papers"].append("d2q")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge").mkdir()
            (root / "knowledge/paper-hierarchy.json").write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, "exactly once"):
                build_knowledge_map.load_paper_hierarchy(root, data["chapters"], data["concepts"], data["papers"])

    def test_new_chapter_papers_receive_one_temporary_parent(self) -> None:
        data = build_knowledge_map.build()
        catalog = {"version": 1, "chapters": json.loads(json.dumps(data["paperHierarchy"]))}
        del catalog["chapters"]["9"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge").mkdir()
            (root / "knowledge/paper-hierarchy.json").write_text(json.dumps(catalog))
            hierarchy = build_knowledge_map.load_paper_hierarchy(root, data["chapters"], data["concepts"], data["papers"])
            self.assertEqual({"rq-vae", "diger", "etegrec", "unisid"},
                             {id for group in hierarchy["9"] for id in group["papers"]})

    def test_chapter_20_readme_lists_every_subsection(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        chapter = readme.split("- [**第 20 章 用户兴趣与序列建模**]", 1)[1].split("- **第 21 章", 1)[0]
        sections = re.findall(r"(?m)^        - (20\.\d+) ", chapter)
        subsections = re.findall(r"(?m)^            - (20\.\d+\.\d+) ", chapter)
        self.assertEqual([f"20.{n}" for n in range(1, 9)], sections)
        self.assertEqual([6, 6, 2, 7, 3, 4, 2, 0],
                         [sum(part.startswith(f"20.{n}.") for part in subsections) for n in range(1, 9)])
        for n in range(1, 9):
            self.assertEqual([f"20.{n}.{j}" for j in range(1, sum(x.startswith(f"20.{n}.") for x in subsections) + 1)],
                             [x for x in subsections if x.startswith(f"20.{n}.")])

    def test_data_covers_released_book_and_valid_links(self) -> None:
        data = build_knowledge_map.build()
        self.assertEqual(len(list((ROOT / "knowledge/chapters").glob("ch[0-9][0-9].md"))), len(data["chapters"]))
        self.assertEqual(len(list((ROOT / "knowledge/concepts").glob("*.md"))), len(data["concepts"]))
        graph = json.loads((ROOT / "knowledge/graph.json").read_text(encoding="utf-8"))
        self.assertEqual(len(graph["edges"]), len(data["edges"]))
        self.assertEqual({"new-engineer", "graduate"}, {path["id"] for path in data["paths"]})
        for chapter in data["chapters"]:
            self.assertTrue((ROOT / unquote(chapter["pdf"].removeprefix("../"))).is_file())
            self.assertTrue((ROOT / chapter["map"].removeprefix("../")).is_file())
        self.assertEqual(sorted(int(path.stem[2:]) for path in (ROOT / "knowledge/chapters").glob("ch[0-9][0-9].md")),
                         [chapter["number"] for chapter in data["chapters"]])
        unpublished = [c for c in data["concepts"] if c["chapter"] not in {x["number"] for x in data["chapters"]}]
        self.assertEqual(["consistency-modeling"], [c["id"] for c in unpublished])

    def test_reader_facing_counts_match_catalogs(self) -> None:
        data = build_knowledge_map.build()
        homepage = (ROOT / "README.md").read_text(encoding="utf-8")
        retrieval_guide = (ROOT / "evals/README.md").read_text(encoding="utf-8")
        queries = [line for line in (ROOT / "evals/retrieval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertIn(f'{len(data["chapters"])} 章、{len(data["concepts"])} 张概念卡、{len(data["edges"])} 条前置依赖', homepage)
        self.assertIn(f'{len(data["papers"])} 个代表论文方法', homepage)
        self.assertIn(f'当前含 {len(queries)} 条中英文高频查询', retrieval_guide)

    def test_papers_have_verified_book_coordinates(self) -> None:
        data = build_knowledge_map.build()
        chapter12 = [paper for paper in data["papers"] if paper["chapter"] == 12]
        self.assertTrue({"wide-and-deep", "pnn", "deepfm", "xdeepfm", "mmoe", "ple", "deepseekmoe"}
                        <= {paper["id"] for paper in chapter12})
        self.assertEqual({7, 9, 10, 11, 12, 15, 20, 23}, {paper["chapter"] for paper in data["papers"]})
        self.assertEqual(len(data["papers"]), len({paper["id"] for paper in data["papers"]}))
        self.assertTrue(all(paper["pages"][0] >= 1 for paper in data["papers"]))
        self.assertTrue(all(set(paper["concepts"]) <= {concept["id"] for concept in data["concepts"]}
                            for paper in data["papers"]))

    def test_paper_schema_rejects_invalid_metadata(self) -> None:
        data = build_knowledge_map.build()
        original = json.loads((ROOT / "knowledge/paper-map.json").read_text(encoding="utf-8"))
        changes = [
            ({"chapter": 24}, "unknown paper chapter"),
            ({"pages": [1, 70]}, "paper pages exceed PDF bounds"),
            ({"pages": [10, 9]}, "invalid paper pages"),
            ({"concepts": ["not-a-concept"]}, "paper concepts"),
            ({"url": "javascript:alert(1)"}, "invalid paper URL"),
            ({"id": "Bad ID"}, "invalid or duplicate paper id"),
        ]
        for changed, error in changes:
            with self.subTest(changed=changed):
                catalog = json.loads(json.dumps(original))
                catalog["papers"][0].update(changed)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "paper-map.json"
                    path.write_text(json.dumps(catalog), encoding="utf-8")
                    original_read = Path.read_text
                    with patch.object(build_knowledge_map.Path, "read_text", autospec=True) as read:
                        def content(file, *args, **kwargs):
                            if file == ROOT / "knowledge/paper-map.json":
                                return original_read(path, *args, **kwargs)
                            return original_read(file, *args, **kwargs)

                        read.side_effect = content
                        with self.assertRaisesRegex(ValueError, error):
                            build_knowledge_map.load_papers(ROOT, data["chapters"],
                                                            {concept["id"] for concept in data["concepts"]})

    def test_unclassified_released_chapter_is_grouped(self) -> None:
        with patch.object(build_knowledge_map, "GROUPS", [
            (key, title, description, [number for number in numbers if number != 23])
            for key, title, description, numbers in build_knowledge_map.GROUPS
        ]):
            data = build_knowledge_map.build()
            self.assertEqual([23], data["groups"][-1]["chapters"])
            self.assertEqual("additional", next(row["group"] for row in data["chapters"] if row["number"] == 23))
            self.assertEqual([19, 20, 28], data["groups"][-2]["chapters"])

    def test_committed_data_matches_generator(self) -> None:
        result = subprocess.run([sys.executable, str(ROOT / "tools/build_knowledge_map.py")], capture_output=True, text=True, check=True)
        self.assertIn(f'{len(build_knowledge_map.build()["chapters"])} chapters', result.stdout)
        generated = (ROOT / "knowledge-map/data.js").read_text(encoding="utf-8")
        self.assertTrue(generated.startswith("/* Generated by"))
        parsed = json.loads(generated.split("window.KNOWLEDGE_MAP = ", 1)[1].removesuffix(";\n"))
        self.assertEqual(build_knowledge_map.build(), parsed)


if __name__ == "__main__":
    unittest.main()
