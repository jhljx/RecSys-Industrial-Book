"""Real-Chrome navigation smoke tests (required with a pinned browser in CI).

Local runs use an installed Chrome or RECSYS_TEST_CHROME. The HTTP fixture maps
Markdown to Pages-style routes; file:// and repository-subpath URLs are checked.
PDF viewer pixels/page rendering are browser-owned, not asserted via iframe DOM.
"""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "/RecSys-Industrial-Book/"
CHROME = os.environ.get("RECSYS_TEST_CHROME") or next((path for path in (
    shutil.which("google-chrome"), shutil.which("chromium"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
) if path and Path(path).is_file()), None)
# GitHub-hosted Ubuntu runners disable unprivileged user namespaces, so Chrome's sandbox cannot start in CI.
SANDBOX_FLAGS = ["--no-sandbox"] if os.environ.get("RECSYS_REQUIRE_BROWSER") == "1" else []

# Run inside the actual reader DOM after its deferred scripts finish. The result
# is serialized in a marker that headless Chrome --dump-dom can return to Python.
PROBE = r"""<script>
window.addEventListener('load', () => {
  const checks = [];
  const check = (name, ok, detail) => checks.push({name, ok: !!ok, detail: String(detail || '')});
  const root = location.protocol === 'file:' ? 'file:' : '/RecSys-Industrial-Book/';
  try {
    check('rendered catalog', document.querySelectorAll('#items .item').length > 0, document.querySelector('#result-count').textContent);
    const select = document.querySelector('#graph-chapter');
    select.value = '20'; select.dispatchEvent(new Event('change', {bubbles: true}));
    check('chapter 20 summary graph', document.querySelectorAll('#graph-view .graph-node.local').length === 11 &&
      !document.querySelector('#graph-view').textContent.includes('DIN 目标局部激活'), document.querySelector('#graph-count').textContent);
    select.value = '4'; select.dispatchEvent(new Event('change', {bubbles: true}));
    check('chapter graph', document.querySelectorAll('#graph-view .graph-node').length > 0 &&
      document.querySelector('#graph-count').textContent.includes('前置关系'), document.querySelector('#graph-count').textContent);
    document.querySelector('#graph-view .graph-node').click();
    check('graph to concept detail', document.querySelector('#detail .concept-definition') !== null &&
      document.querySelector('[data-kind="concepts"]').getAttribute('aria-pressed') === 'true', document.querySelector('#detail').textContent.slice(0, 80));
    const concept = document.querySelector('#detail .detail-links a').href;
    check('concept URL', concept.includes(root) && /knowledge\/concepts\/[^/]+\.(md|html)$/.test(concept), concept);
    check('local Agent command', document.querySelector('#detail .agent-command code')?.textContent.includes('recsys_agent.py ask'), document.querySelector('#detail .agent-command')?.textContent.slice(0, 80));
    document.querySelector('[data-course="graduate"]').click();
    check('course selection', document.querySelector('[data-course="graduate"]').getAttribute('aria-pressed') === 'true' &&
      document.querySelectorAll('#course-list .course-unit').length > 0, document.querySelector('#course-list').textContent.slice(0, 80));
    const course = document.querySelector('#course-list .course-intro a').href;
    check('course URL', course.includes(root) && /learning-paths\/graduate\.(md|html)$/.test(course), course);
    document.querySelector('#course-list .course-unit button').click();
    check('course to chapter PDF', document.querySelector('#detail .detail-links a[href*="reader.html#ch"]') !== null &&
      document.querySelector('[data-kind="chapters"]').getAttribute('aria-pressed') === 'true', document.querySelector('#detail').textContent.slice(0, 80));
    const map = [...document.querySelectorAll('#detail .detail-links a')].find(a => a.textContent.includes('章节阅读地图')).href;
    check('chapter map URL', map.includes(root) && /knowledge\/chapters\/ch\d+\.(md|html)$/.test(map), map);
    document.querySelector('[data-kind="concepts"]').click();
    const item = [...document.querySelectorAll('#items .item')].find(button => button.textContent.includes('链路一致性建模'));
    item.click();
    const conceptLinks = [...document.querySelectorAll('#detail .detail-links a')];
    const cited = conceptLinks.find(a => a.textContent.includes('站内阅读所引 PDF'));
    check('unpublished concept cites released PDF', cited && /reader\.html#ch5$/.test(cited.href) &&
      !conceptLinks.some(a => /reader\.html#ch22$/.test(a.href)), conceptLinks.map(a => a.href).join(' | '));
    const negative = [...document.querySelectorAll('#items .item')].find(button => button.textContent.includes('负采样'));
    negative.click();
    const codeLink = [...document.querySelectorAll('#detail a')].find(a => a.textContent.includes('two_tower_debias.py'));
    check('concept to code page', codeLink && codeLink.href.includes(root) && /code\/[0-9a-f]+\.html$/.test(codeLink.href), codeLink?.href);
  } catch (error) { check('browser script', false, error.stack || error); }
  setTimeout(() => {
    const map = document.querySelector('#paper-map-view');
    const expected = 1 + window.KNOWLEDGE_MAP.paperHierarchy['20'].length +
      window.KNOWLEDGE_MAP.papers.filter(paper => paper.chapter === 20).length +
      window.KNOWLEDGE_MAP.paperHierarchy['20'].reduce((count, group) => count + (group.extraConcepts || []).length, 0);
    check('paper methods expanded', map.querySelectorAll('g.markmap-node').length === expected,
      `${map.querySelectorAll('g.markmap-node').length}/${expected} visible nodes`);
    const marker = document.createElement('output'); marker.id = 'browser-test-result';
    marker.textContent = JSON.stringify(checks); document.body.append(marker);
  }, 700);
});
</script>"""


READER_PROBE = r"""<script>
window.addEventListener('DOMContentLoaded', async () => {
  const checks = [];
  const check = (name, ok, detail) => checks.push({name, ok: !!ok, detail: String(detail || '')});
  const pause = () => new Promise(resolve => setTimeout(resolve, 250));
  const inspect = (number, page) => {
    const chapter = window.KNOWLEDGE_MAP.chapters.find(row => row.number === number);
    const expected = new URL(chapter.pdf, location.href).href + '#page=' + page + '&view=FitH';
    const frame = document.querySelector('#pdf-frame');
    check('iframe source ' + number + '/' + page, frame.src === expected, frame.src);
    check('direct PDF fallback ' + number + '/' + page, document.querySelector('#direct-pdf').href === expected, expected);
    check('page control ' + page, document.querySelector('#page-number').value === String(page), location.hash);
    check('active chapter ' + number, document.querySelector('#chapter-list [aria-current="page"]').dataset.chapter === String(number));
    check('reader status ' + page, document.querySelector('#reader-status').textContent.includes('第 ' + page + ' 页'));
    const map = document.querySelector('#chapter-map').href;
    check('reader map route', location.protocol === 'file:' ? map.endsWith('.md') : map.includes('/RecSys-Industrial-Book/') && map.endsWith('.html'), map);
  };
  try {
    inspect(4, 2);
    document.querySelector('#page-number').value = '3';
    document.querySelector('#page-form').requestSubmit();
    await pause();
    check('page form updates hash', location.hash === '#ch4&page=3', location.hash);
    inspect(4, 3);
    document.querySelector('#next').click();
    await pause();
    check('next chapter updates hash', location.hash === '#ch5&page=1', location.hash);
    inspect(5, 1);
    document.querySelector('#chapter-list a[data-chapter="4"]').click();
    await pause();
    inspect(4, 1);
  } catch (error) { check('reader script', false, error.stack || error); }
  const marker = document.createElement('output'); marker.id = 'browser-test-result';
  marker.textContent = JSON.stringify(checks); document.body.append(marker);
});
</script>"""


CODE_PROBE = r"""<script>
window.addEventListener('load', () => {
  const checks = [];
  const check = (name, ok, detail) => checks.push({name, ok: !!ok, detail: String(detail || '')});
  try {
    const search = document.querySelector('#code-search');
    check('code catalog', document.querySelectorAll('#code-results a.code-item').length === window.CHAPTER_CODE.length && window.CHAPTER_CODE.length > 0, document.querySelector('#code-count').textContent);
    search.value = 'two_tower_debias'; search.dispatchEvent(new Event('input', {bubbles: true}));
    const links = [...document.querySelectorAll('#code-results a.code-item')];
    check('code search', links.length === 1 && links[0].href.endsWith('/code/' + window.CHAPTER_CODE.find(row => row.path.endsWith('/two_tower_debias.py')).url.split('/').at(-1)), links.map(a => a.href).join(' | '));
    search.value = 'no_such_example_zz'; search.dispatchEvent(new Event('input', {bubbles: true}));
    check('empty code search', document.querySelector('#code-results .empty') !== null);
  } catch (error) { check('code catalog script', false, error.stack || error); }
  const marker = document.createElement('output'); marker.id = 'browser-test-result';
  marker.textContent = JSON.stringify(checks); document.body.append(marker);
});
</script>"""

SOURCE_PROBE = r"""<script>
window.addEventListener('load', () => {
  const checks = [];
  const check = (name, ok, detail) => checks.push({name, ok: !!ok, detail: String(detail || '')});
  try {
    const code = document.querySelector('#source-code');
    check('source and line anchors', code.querySelectorAll('.source-line').length > 0 && code.querySelector('#L27') !== null);
    check('source resources', [...document.styleSheets].some(sheet => sheet.href?.endsWith('/code.css')));
    const links = [...document.querySelectorAll('.detail-links a')];
    check('source links', links.some(a => a.textContent.includes('原始源码')) && links.some(a => a.textContent.includes('阅读第 10 章 PDF')));
    check('source hrefs', links.filter(a => !a.href.includes('reader.html')).length === 2, links.filter(a => !a.href.includes('reader.html')).map(a => a.href).join(' | '));
  } catch (error) { check('code source script', false, error.stack || error); }
  const marker = document.createElement('output'); marker.id = 'browser-test-result';
  marker.textContent = JSON.stringify(checks); document.body.append(marker);
});
</script>"""

CATALOG_PROBE = r"""<script>
window.addEventListener('load', () => {
  const checks = [];
  const check = (name, ok, detail) => checks.push({name, ok: !!ok, detail: String(detail || '')});
  const nav = document.querySelector('main nav[aria-label="章节跳转"]');
  const heading = nav.querySelector('h2').getBoundingClientRect();
  const list = nav.querySelector('ul').getBoundingClientRect();
  check('chapter heading precedes links', getComputedStyle(nav).display === 'block' &&
    heading.bottom <= list.top, `${heading.bottom} <= ${list.top}`);
  const first = document.querySelector('#ch0');
  const second = document.querySelector('#ch1');
  check('chapter sections compact', parseFloat(getComputedStyle(first).paddingTop) <= 24 &&
    second.getBoundingClientRect().top - first.getBoundingClientRect().bottom < 30,
    `${getComputedStyle(first).paddingTop}; ${second.getBoundingClientRect().top - first.getBoundingClientRect().bottom}`);
  const marker = document.createElement('output'); marker.id = 'browser-test-result';
  marker.textContent = JSON.stringify(checks); document.body.append(marker);
});
</script>"""


class PagesHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if not path.startswith(PREFIX):
            self.send_error(404)
            return
        relative = path[len(PREFIX):]
        # Jekyll serves individual Markdown at .html, directory README at /.
        if relative.endswith(".html") and not (Path(self.directory) / relative).is_file():
            relative = relative[:-5] + ".md"
        elif relative.endswith("/"):
            relative += "README.md" if (Path(self.directory) / relative / "README.md").is_file() else "index.html"
        self.path = "/" + relative
        super().do_GET()


class BrowserNavigationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RECSYS_REQUIRE_BROWSER") == "1":
            if not CHROME or not Path(CHROME).is_file():
                raise AssertionError(f"CI requires an installed browser: {CHROME!r}")

    def run_browser(self, url):
        completed = subprocess.run(
            [CHROME, "--headless", "--disable-gpu", *SANDBOX_FLAGS, "--no-first-run", "--no-default-browser-check",
             "--virtual-time-budget=4000", "--dump-dom", url],
            capture_output=True, text=True, timeout=25,
        )
        self.assertEqual(0, completed.returncode, completed.stderr[-1000:])
        marker = re.search(r'<output id="browser-test-result">(.*?)</output>', completed.stdout, re.DOTALL)
        self.assertIsNotNone(marker, completed.stderr[-1000:] + completed.stdout[-500:])
        import html
        checks = json.loads(html.unescape(marker.group(1)))
        self.assertTrue(checks and all(row["ok"] for row in checks), checks)
        return checks

    @unittest.skipUnless(CHROME, "Chrome not installed; optional local browser layout test")
    def test_static_catalog_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            website = Path(tmp) / "knowledge-map"
            website.mkdir()
            for name in ("style.css", "document.css", "site-links.js"):
                shutil.copy2(ROOT / "knowledge-map" / name, website / name)
            catalog = website / "catalog.html"
            catalog.write_text((ROOT / "knowledge-map/catalog.html").read_text(encoding="utf-8")
                               .replace("</body>", CATALOG_PROBE + "</body>"), encoding="utf-8")
            self.run_browser(catalog.as_uri())

    @unittest.skipUnless(CHROME, "Chrome not installed; optional local browser smoke test")
    def test_file_and_pages_subpath_navigation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            website = root / "knowledge-map"
            shutil.copytree(ROOT / "knowledge-map", website)
            for name in ("knowledge", "learning-paths"):
                (root / name).symlink_to(ROOT / name, target_is_directory=True)
            index = website / "index.html"
            index.write_text(index.read_text(encoding="utf-8").replace("</body>", PROBE + "</body>"), encoding="utf-8")
            reader = website / "reader.html"
            reader.write_text(reader.read_text(encoding="utf-8").replace("</body>", READER_PROBE + "</body>"), encoding="utf-8")
            code_catalog = website / "code.html"
            code_catalog.write_text(code_catalog.read_text(encoding="utf-8").replace("</body>", CODE_PROBE + "</body>"), encoding="utf-8")
            code_source = website / "code" / next(row["url"].split("/")[-1] for row in json.loads((ROOT / "knowledge-map/code-data.js").read_text(encoding="utf-8").split("= ", 1)[1].removesuffix(";\n")) if row["path"].endswith("/two_tower_debias.py"))
            code_source.write_text(code_source.read_text(encoding="utf-8").replace("</body>", SOURCE_PROBE + "</body>"), encoding="utf-8")
            for chapter in ROOT.glob("ch[0-9][0-9] *"):
                (root / chapter.name).symlink_to(chapter, target_is_directory=True)
            handler = partial(PagesHandler, directory=str(root))
            with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    urls = (
                        (index.as_uri(), PROBE),
                        (f"http://127.0.0.1:{server.server_port}{PREFIX}knowledge-map/index.html", PROBE),
                        (reader.as_uri() + "#ch4&page=2", READER_PROBE),
                        (f"http://127.0.0.1:{server.server_port}{PREFIX}knowledge-map/reader.html#ch4&page=2", READER_PROBE),
                        (code_catalog.as_uri(), CODE_PROBE),
                        (f"http://127.0.0.1:{server.server_port}{PREFIX}knowledge-map/code.html", CODE_PROBE),
                        (code_source.as_uri() + "#L27", SOURCE_PROBE),
                        (f"http://127.0.0.1:{server.server_port}{PREFIX}knowledge-map/code/{code_source.name}#L27", SOURCE_PROBE),
                    )
                    for url, _probe in urls:
                        with self.subTest(url=url):
                            completed = subprocess.run(
                                [CHROME, "--headless", "--disable-gpu", *SANDBOX_FLAGS, "--no-first-run", "--no-default-browser-check",
                                 "--virtual-time-budget=4000", "--dump-dom", url],
                                capture_output=True, text=True, timeout=25,
                            )
                            self.assertEqual(0, completed.returncode, completed.stderr[-1000:])
                            marker = re.search(r'<output id="browser-test-result">(.*?)</output>', completed.stdout, re.DOTALL)
                            self.assertIsNotNone(marker, completed.stderr[-1000:] + completed.stdout[-500:])
                            import html
                            checks = json.loads(html.unescape(marker.group(1)))
                            self.assertTrue(checks and all(row["ok"] for row in checks), checks)
                            for row in checks:
                                if row["name"] == "concept to code page":
                                    target_url = row["detail"]
                                    if url.startswith("http:"):
                                        with urlopen(target_url, timeout=5) as response:
                                            self.assertEqual(200, response.status)
                                            self.assertIn(b'id="source-code"', response.read())
                                    else:
                                        self.assertTrue(Path(unquote(urlsplit(target_url).path)).is_file(), row)
                                if row["name"] == "source hrefs" and row["ok"]:
                                    for target_url in row["detail"].split(" | "):
                                        if url.startswith("http:"):
                                            with urlopen(target_url, timeout=5) as response:
                                                self.assertEqual(200, response.status)
                                        else:
                                            self.assertTrue(Path(unquote(urlsplit(target_url).path)).is_file(), target_url)
                                if row["name"] in ("concept URL", "course URL", "chapter map URL"):
                                    path = unquote(urlsplit(row["detail"]).path)
                                    if url.startswith("file:"):
                                        target = Path(path)
                                    else:
                                        self.assertTrue(path.startswith(PREFIX), path)
                                        target = root / path[len(PREFIX):]
                                        target = target.with_suffix(".md") if target.suffix == ".html" else target
                                    self.assertTrue(target.is_file(), row)
                                    if url.startswith("http:"):
                                        with urlopen(row["detail"], timeout=5) as response:
                                            self.assertEqual(200, response.status)
                finally:
                    server.shutdown()
                    thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
