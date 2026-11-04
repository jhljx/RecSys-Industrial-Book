"""Ensure generated chapter-code HTML is complete, safely escaped, and navigable."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import build_code_pages
import recsys_code
import recsys_navigation


class CodePageTests(unittest.TestCase):
    def test_every_catalog_source_has_generated_html_page(self):
        data = (ROOT / "knowledge-map/code-data.js").read_text(encoding="utf-8")
        entries = json.loads(data.split("= ", 1)[1].removesuffix(";\n"))
        sources = recsys_code.catalog(ROOT)
        self.assertEqual({item["path"] for item in sources}, {item["path"] for item in entries})
        self.assertEqual({recsys_code.page_name(item["path"]) for item in sources},
                         {page.name for page in (ROOT / "knowledge-map/code").glob("*.html")})
        self.assertEqual(len(sources), len(list((ROOT / "knowledge-map/code").glob("*.html"))))
        for ident in recsys_navigation.NavigationIndex(ROOT).concepts:
            match = recsys_navigation.resolve(ident, root=ROOT)["matches"][0]
            expected = {path for path in match["code"] if path in {item["path"] for item in sources}}
            actual = {item["path"] for item in entries if ident in item["concept_ids"]}
            self.assertEqual(expected, actual, ident)
        for item in sources:
            page = ROOT / "knowledge-map" / item["page"]
            self.assertTrue(page.is_file(), item["path"])
            text = page.read_text(encoding="utf-8")
            self.assertIn('id="source-code"', text)
            self.assertIn('href="../../' + quote(item["path"].split("/")[0], safe=""), text)

    def test_escaped_source_and_line_anchors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "ch24 示例/code"
            directory.mkdir(parents=True)
            (directory / "README.md").write_text("├── demo.py  安全演示\n", encoding="utf-8")
            (directory / "demo.py").write_text('"""<script>alert(1)</script>"""\nvalue = "<img src=x>"\n', encoding="utf-8")
            entries = build_code_pages.build(root)
            self.assertEqual(1, len(entries))
            page = (root / "knowledge-map" / entries[0]["url"]).read_text(encoding="utf-8")
            self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', page)
            self.assertNotIn('<script>alert(1)</script>', page)
            self.assertIn('id="L2"', page)

    def test_agent_and_web_use_same_stable_url(self):
        item = recsys_code.search(ROOT, "负采样偏差", chapter=10)[0]
        self.assertTrue((ROOT / "knowledge-map" / item["page"]).is_file())
        self.assertEqual(item["page"], "./code/" + recsys_code.page_name(item["path"]))


if __name__ == "__main__":
    unittest.main()
