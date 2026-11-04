"""Static checks for Pages routes and navigation consistency."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SiteRoutesTest(unittest.TestCase):
    def test_markdown_readme_routes_are_not_rewritten_to_missing_html(self) -> None:
        for name in ("app.js", "reader.js"):
            script = (ROOT / "knowledge-map" / name).read_text(encoding="utf-8")
            self.assertIn("/README\\.md", script)
            self.assertIn("return href.replace(/\\.md(?=#|$)/, '.html')", script)
        page = (ROOT / "knowledge-map" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="concept-catalog-link" href="#catalog"', page)

    def test_agent_guide_uses_same_pages_document_shell(self) -> None:
        page = (ROOT / "knowledge-map" / "agent.html").read_text(encoding="utf-8")
        self.assertIn('href="../docs/agent-guide.html"', page)
        layout = (ROOT / "_layouts" / "default.html").read_text(encoding="utf-8")
        self.assertIn("knowledge-map/document.css", layout)

    def test_reader_requests_fit_view(self) -> None:
        script = (ROOT / "knowledge-map" / "reader.js").read_text(encoding="utf-8")
        self.assertIn("#page=${page}&view=FitH", script)

    def test_local_html_targets_exist(self) -> None:
        for path in (ROOT / "knowledge-map").glob("*.html"):
            text = path.read_text(encoding="utf-8")
            for href in re.findall(r'href="([^"#]+\.html)(?:#[^"]*)?"', text):
                if href.startswith(("http://", "https://")):
                    continue
                target = (path.parent / href).resolve()
                if not target.is_file():
                    self.assertTrue(target.with_suffix(".md").is_file(), f"{path}: {href}")


if __name__ == "__main__":
    unittest.main()
