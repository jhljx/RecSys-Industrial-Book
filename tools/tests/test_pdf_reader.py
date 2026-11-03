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
        self.assertEqual(20, len(chapters))
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


if __name__ == "__main__":
    unittest.main()
