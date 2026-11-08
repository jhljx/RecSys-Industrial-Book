"""Static integrity checks for the dependency-free, local-first PDF reader."""
from __future__ import annotations

import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
READER = ROOT / "knowledge-map"


class PdfReaderTest(unittest.TestCase):
    def test_reader_uses_generated_released_chapters_and_relative_pdfs(self) -> None:
        from sys import path
        path.insert(0, str(ROOT / "tools"))
        from build_knowledge_map import build

        chapters = build()["chapters"]
        self.assertEqual(len(list((ROOT / "knowledge/chapters").glob("ch[0-9][0-9].md"))), len(chapters))
        for chapter in chapters:
            href = chapter["pdf"]
            self.assertFalse(urlsplit(href).scheme, href)
            self.assertTrue((READER / unquote(href)).resolve().is_file(), href)
        script = (READER / "reader.js").read_text(encoding="utf-8")
        self.assertIn("window.KNOWLEDGE_MAP", script)
        self.assertIn("chapter.pdf}#page=${page}", script)
        self.assertNotIn("raw.githubusercontent.com", script)

    def test_entrypoint_and_assets(self) -> None:
        page = (READER / "reader.html").read_text(encoding="utf-8")
        for asset in ("reader.js", "reader.css", "data.js"):
            self.assertIn(f'./{asset}', page)
            self.assertTrue((READER / asset).is_file())
        self.assertIn('id="pdf-frame"', page)
        self.assertIn('id="chapter-list"', page)
        self.assertIn('id="page-number"', page)
        self.assertIn('id="direct-pdf"', page)
        self.assertIn('reader.html', (READER / "index.html").read_text(encoding="utf-8"))
        self.assertIn('reader.html#ch${c.number}', (READER / "app.js").read_text(encoding="utf-8"))

    def test_public_entrypoints_share_the_knowledge_map_theme(self) -> None:
        root = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("./knowledge-map/", root)
        map_page = (READER / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="./agent.html"', map_page)
        agent_page = (READER / "agent.html").read_text(encoding="utf-8")
        self.assertIn("./style.css", agent_page)
        self.assertIn("./agent.css", agent_page)
        self.assertTrue((READER / "agent.css").is_file())

    def test_reader_layout_can_use_full_viewport_width(self) -> None:
        css = (READER / "reader.css").read_text(encoding="utf-8")
        self.assertIn(".reader-layout{max-width:none;width:100%", css)
        self.assertIn(".reader-layout .reading{min-width:0;width:100%;max-width:none;margin:0;padding:0}", css)
        script = (READER / "reader.js").read_text(encoding="utf-8")
        self.assertIn("#page=${page}&view=FitH", script)
        self.assertRegex((READER / "reader.html").read_text(encoding="utf-8"),
                         r'<script src="\./reader\.js\?v=[a-z0-9-]+" defer></script>')


if __name__ == "__main__":
    unittest.main()
