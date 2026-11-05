from __future__ import annotations

import sys
import unittest
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import build_search_pages  # noqa: E402


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for key, value in attrs if key == "href")


class SearchPagesTest(unittest.TestCase):
    def test_static_catalog_has_real_links_and_pdf_evidence(self):
        generated = build_search_pages.render()
        self.assertEqual(generated, (ROOT / "knowledge-map/catalog.html").read_text(encoding="utf-8"))
        self.assertNotIn("<script", generated)
        self.assertIn("双塔召回", generated)
        self.assertIn("PDF 第 60 页", generated)
        self.assertIn("相关教学代码", generated)
        links = Links()
        links.feed(generated)
        self.assertTrue(any(href.endswith("#page=60") and ".pdf" in href for href in links.hrefs))
        for href in links.hrefs:
            split = urlsplit(href)
            if split.scheme or split.path.startswith("/") or split.path in ("", "./", "../"):
                continue
            path = (ROOT / "knowledge-map" / unquote(split.path)).resolve()
            if path.suffix == ".html" and not path.is_file():
                path = path.with_suffix(".md")
            self.assertTrue(path.is_file(), href)

    def test_sitemap_is_deterministic_and_targets_exist(self):
        generated = build_search_pages.sitemap()
        self.assertEqual(generated, (ROOT / "sitemap.xml").read_text(encoding="utf-8"))
        xml = ET.fromstring(generated)
        locations = [node.text for node in xml.iter() if node.tag.endswith("loc")]
        self.assertEqual(len(locations), len(set(locations)))
        self.assertGreaterEqual(len(locations), 400)
        self.assertTrue(any("knowledge-map/catalog.html" in loc for loc in locations))
        self.assertTrue(any("knowledge-map/code/" in loc for loc in locations))
        self.assertTrue(all(loc.startswith(build_search_pages.BASE) for loc in locations))
    def test_concept_layout_links_are_from_published_navigation(self):
        import json
        generated = build_search_pages.concept_links()
        committed = json.loads((ROOT / "_data/concept_links.json").read_text(encoding="utf-8"))
        self.assertEqual(generated, committed)
        self.assertEqual(60, committed["two-tower"]["page"])
        self.assertEqual(2, committed["duration-modeling"]["page"])
        self.assertEqual(19, committed["lsh-retrieval"]["page"])
        self.assertTrue(committed["two-tower"]["chapter_map"].endswith("ch10.html"))
        self.assertTrue((ROOT / committed["two-tower"]["code"].lstrip("/")).is_file())
        self.assertNotIn("consistency-modeling", committed)  # unpublished chapter


if __name__ == "__main__":
    unittest.main()
