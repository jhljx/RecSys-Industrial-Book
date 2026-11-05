"""Check browser-resolved site links in local file and Pages modes."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://jhljx.github.io/RecSys-Industrial-Book/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        for key in ("href", "src"):
            if attributes.get(key) and (tag in ("a", "script", "link", "iframe") or key == "src"):
                self.targets.append(attributes[key])


def parsed(page):
    links = Links()
    links.feed(page.read_text(encoding="utf-8"))
    return links


def destination(page, href, local):
    origin = page.as_uri() if local else BASE + page.relative_to(ROOT).as_posix()
    absolute = urlsplit(urljoin(origin, href))
    if absolute.scheme not in ("file", "https") or (not local and not absolute.geturl().startswith(BASE)):
        return None
    target = Path(unquote(absolute.path)) if local else ROOT / unquote(absolute.path.removeprefix("/RecSys-Industrial-Book/"))
    if local:
        relative = target.relative_to(ROOT).as_posix()
        if target.is_dir():
            target = target / ("README.md" if relative.startswith(("knowledge/", "learning-paths/", "docs/", "templates/")) else "index.html")
        elif re.match(r"^(README\.en|(?:knowledge|learning-paths|docs|templates)/.*)\.html$", relative):
            target = target.with_suffix(".md")
    else:
        if target.name == "README.md":
            target = target.parent / ("index.html" if target.parent == ROOT / "knowledge-map" else "README.md")
        elif target.suffix == ".md":
            target = target.with_suffix(".html")
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file() and target.suffix == ".html":
            target = target.with_suffix(".md")  # Jekyll's rendered Markdown document
    return target


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

    def test_unpublished_concept_layout_does_not_link_missing_pdf(self) -> None:
        layout = (ROOT / "_layouts" / "default.html").read_text(encoding="utf-8")
        self.assertNotIn("#ch{{ page.chapter }}", layout)

    def test_reader_requests_fit_view(self) -> None:
        script = (ROOT / "knowledge-map" / "reader.js").read_text(encoding="utf-8")
        self.assertIn("#page=${page}&view=FitH", script)

    def test_site_links_resolve_in_file_and_pages_modes(self) -> None:
        pages = [ROOT / "index.html", *(ROOT / "knowledge-map").rglob("*.html")]
        for page in pages:
            links = parsed(page)
            if page.name == "catalog.html" or page.name == "agent.html":
                self.assertIn('src="./site-links.js"', page.read_text(encoding="utf-8"))
            for href in links.targets:
                if href.startswith("#"):
                    if href[1:] and page.parent == ROOT / "knowledge-map":
                        self.assertIn(href[1:], links.ids, f"{page}: {href}")
                    continue
                if href.startswith("https://"):
                    continue
                for local in (True, False):
                    with self.subTest(page=page.relative_to(ROOT), href=href, local=local):
                        target = destination(page, href, local)
                        if target is not None:
                            self.assertTrue(target.is_file(), f"{page}: {href} -> {target}")

    def test_generated_dynamic_links(self) -> None:
        data = (ROOT / "knowledge-map/data.js").read_text(encoding="utf-8")
        map_data = json.loads(data.split("window.KNOWLEDGE_MAP = ", 1)[1].rstrip().removesuffix(";"))
        code_data = (ROOT / "knowledge-map/code-data.js").read_text(encoding="utf-8")
        code_rows = json.loads(code_data.split("window.CHAPTER_CODE = ", 1)[1].rstrip().removesuffix(";"))
        page = ROOT / "knowledge-map/index.html"
        for href in [*(ch["pdf"] for ch in map_data["chapters"]),
                     *(ch["map"] for ch in map_data["chapters"]),
                     *(c["url"] for c in map_data["concepts"]),
                     *(path["url"] for path in map_data["paths"]),
                     *(code["url"] for code in code_rows)]:
            for local in (True, False):
                with self.subTest(href=href, local=local):
                    self.assertTrue(destination(page, href, local).is_file(), href)


if __name__ == "__main__":
    unittest.main()
