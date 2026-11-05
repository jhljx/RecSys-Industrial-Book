#!/usr/bin/env python3
"""Notify IndexNow of published sitemap URLs changed between two commits."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

BASE = "https://jhljx.github.io/RecSys-Industrial-Book/"
KEY = "9d4390e1830e48f0b4d52fb03d1a4b2c"
KEY_LOCATION = BASE + KEY + ".txt"
ENDPOINT = "https://www.bing.com/indexnow"
# Only published, indexable URLs. Assets, authoring metadata, and sitemap are not pages.
PUBLISHED = re.compile(r"(?:knowledge-map/(?:catalog|reader|code|agent)\.html|knowledge-map/code/[0-9a-f]+\.html|knowledge-map/|knowledge/(?:chapters|concepts)/[\w-]+\.html|knowledge/|learning-paths/[\w-]+\.html|templates/[^/]+/README\.html|docs/(?:agent-guide|discoverability)\.html|README\.en\.html|ch\d\d%20[^/]+/[^/]+\.pdf)")


def git_bytes(commit: str, path: str) -> bytes | None:
    result = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True)
    return result.stdout if result.returncode == 0 else None


def sitemap_urls(source: bytes) -> set[str]:
    urls = {node.text for node in ET.fromstring(source).iter() if node.tag.endswith("loc")}
    if any(url is None or not url.startswith(BASE) for url in urls):
        raise ValueError("sitemap contains a URL outside this project")
    return urls


def changed_urls(before: str, after: str) -> list[str]:
    previous = sitemap_urls(git_bytes(before, "sitemap.xml") or b"<urlset/>")
    current_data = git_bytes(after, "sitemap.xml")
    if not current_data:
        raise ValueError("published commit has no sitemap.xml")
    current = sitemap_urls(current_data)
    paths = subprocess.check_output(["git", "diff", "--name-only", "-z", before, after]).decode().split("\0")
    changed = set()
    for path in paths:
        if not path:
            continue
        if path == "knowledge-map/index.html":
            changed.add(BASE + "knowledge-map/")
        elif path == "knowledge/README.md":
            changed.add(BASE + "knowledge/")
        elif path == "README.en.md":
            changed.add(BASE + "README.en.html")
        elif path.endswith(".md") and path.startswith(("knowledge/", "learning-paths/", "docs/", "templates/")):
            changed.add(BASE + path.removesuffix(".md") + ".html")
        elif path.endswith((".html", ".pdf")):
            from urllib.parse import quote
            changed.add(BASE + quote(path, safe="/-._~"))
    # Generator/layout changes affect rendered pages even when the inputs are untouched.
    if any(path in paths for path in ("tools/build_search_pages.py", "_layouts/default.html", "_data/concept_links.json")):
        changed.update(url for url in current if url.endswith(".html") and "/knowledge-map/code/" not in url)
    if "tools/build_code_pages.py" in paths:
        changed.update(url for url in current if "/knowledge-map/code/" in url)
    # Deletions are submitted too; unchanged sitemap entries are never mass-pushed.
    candidates = ((current - previous) | (previous - current) | changed) & (current | previous)
    return sorted(url for url in candidates if PUBLISHED.fullmatch(url.removeprefix(BASE)))


def check_live(expected: set[str]) -> None:
    with urlopen(KEY_LOCATION, timeout=15) as response:
        if response.status != 200 or response.read().decode().strip() != KEY:
            raise ValueError("deployed IndexNow key differs")
    with urlopen(BASE + "sitemap.xml", timeout=15) as response:
        live = sitemap_urls(response.read())
    if live != expected:
        raise ValueError("deployed sitemap differs from published commit")


def notify(urls: list[str], after: str) -> int:
    if not urls:
        print("IndexNow: no changed public URLs")
        return 0
    check_live(sitemap_urls(git_bytes(after, "sitemap.xml") or b"<urlset/>"))
    print(f"IndexNow: {len(urls)} changed URLs, key and sitemap live", flush=True)
    payload = {"host": urlsplit(BASE).hostname, "key": KEY, "keyLocation": KEY_LOCATION, "urlList": urls}
    request = Request(ENDPOINT, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    with urlopen(request, timeout=30) as response:
        if response.status not in (200, 202):
            raise ValueError(f"IndexNow rejected request: HTTP {response.status}")
        print(f"IndexNow HTTP {response.status}: received, indexing not guaranteed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="previous successful Pages commit")
    parser.add_argument("--after", required=True, help="newly published Pages commit")
    parser.add_argument("--send", action="store_true", help="send only after confirming Pages deployment")
    args = parser.parse_args()
    urls = changed_urls(args.before, args.after)
    if not args.send:
        print("\n".join(urls) if urls else "IndexNow: no changed public URLs")
        return 0
    try:
        return notify(urls, args.after)
    except (ValueError, HTTPError, URLError) as error:
        parser.exit(1, f"IndexNow not sent: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
