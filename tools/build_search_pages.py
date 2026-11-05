#!/usr/bin/env python3
"""Build crawlable public navigation and sitemap from curated metadata, not PDF text."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from html import escape
from pathlib import Path
from urllib.parse import quote, unquote
from xml.etree import ElementTree as ET

from build_knowledge_map import ROOT, build
from recsys_code import page_url
from recsys_navigation import NavigationIndex

BASE = "https://jhljx.github.io/RecSys-Industrial-Book/"


def url(path: str) -> str:
    return quote(path.lstrip("/"), safe="/#&=")


def link(path: str, label: str) -> str:
    return f'<a href="{escape(url(path), quote=True)}">{escape(label)}</a>'


def chapter_pdf(chapter: int, page: int = 1) -> str:
    return f"reader.html#ch{chapter}&page={page}"


def render(root: Path = ROOT) -> str:
    data = build(root)
    navigation = NavigationIndex(root)
    concepts_by_chapter = defaultdict(list)
    for concept in data["concepts"]:
        concepts_by_chapter[concept["chapter"]].append(concept)
    sections = []
    for chapter in data["chapters"]:
        number = chapter["number"]
        concepts = []
        for concept in concepts_by_chapter[number]:
            card = concept["url"].removeprefix("../").replace(".md", ".html")
            source = navigation.concepts.get(concept["id"])
            page = source["pdf_page"] if source else int(re.search(r"PDF\s*第\s*(\d+)", concept["source"]).group(1))
            evidence = " · " + link(unquote(chapter["pdf"]) + "#page=" + str(page), "PDF 第 " + str(page) + " 页")
            # Concept cards themselves carry definition, applicability and a PDF citation.
            code = ""
            if source:
                candidates = navigation._code(concept["id"], [paper for paper in navigation.papers if concept["id"] in paper["concepts"]])
                pages = ["./" + page_url(path).removeprefix("./") for path in candidates
                         if (root / "knowledge-map" / page_url(path).removeprefix("./")).is_file()]
                if pages:
                    code = " · " + link(pages[0], "相关教学代码")
            concepts.append(f'<li>{link("../" + card, concept["title"])}{evidence}{code}</li>')
        map_path = chapter["map"].replace(".md", ".html")
        code_dir = any((p / "code/README.md").is_file() for p in root.glob(f"ch{number:02d} *"))
        code_link = (" · " + link("code.html#ch" + str(number), "本章代码目录")
                     if code_dir else "")
        items = "".join(concepts) or "<li>本章暂无单独概念卡；以 PDF 和章节阅读地图为准。</li>"
        sections.append(f'''<section id="ch{number}" aria-labelledby="chapter-{number}">
<h2 id="chapter-{number}">第 {number} 章 · {escape(chapter["title"])}</h2>
<p>{link(map_path, "章节阅读地图")} · {link(chapter_pdf(number), "已发布 PDF")}{code_link}</p>
<h3>相关概念与 PDF 页码</h3><ul>{items}</ul>
</section>''')
    other = [concept for concept in data["concepts"] if concept["chapter"] not in {row["number"] for row in data["chapters"]}]
    if other:
        items = "".join(f'<li>{link(concept["url"].replace(".md", ".html"), concept["title"])}：所引章节尚未以独立章节发布，请以概念卡现有的已发布 PDF 依据为准。</li>' for concept in other)
        sections.append(f'<section><h2>其他已引用概念卡</h2><ul>{items}</ul></section>')
    body = "\n".join(sections)
    focus = {concept["id"]: concept for concept in data["concepts"]}
    focus_topics = (("two-tower", "大规模召回如何兼顾效率与采样偏差？"),
                    ("ann-index", "ANN 索引怎样权衡召回率、延迟和内存？"),
                    ("feature-freshness", "实时特征为什么在线仍不够新？"),
                    ("fallback", "推荐链路超时后如何降级或兜底？"))
    focus_links = "".join(f'<li>{escape(question)} {link(focus[ident]["url"].replace(".md", ".html"), focus[ident]["title"])}</li>'
                          for ident, question in focus_topics if ident in focus)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>章节与概念静态索引 · 推荐系统：工业架构与核心算法</title>
<meta name="description" content="浏览已发布推荐系统章节 PDF、章节阅读地图、概念卡和代码目录；每个概念关联可核查的 PDF 页码。">
<link rel="canonical" href="{BASE}knowledge-map/catalog.html">
<link rel="stylesheet" href="./style.css"><link rel="stylesheet" href="./document.css"><script src="./site-links.js" defer></script></head>
<body><a class="skip" href="#main">跳到正文</a>
<header class="site-header"><a class="brand" href="./">RECSYS <span>／ 阅读与实践</span></a><nav aria-label="页面导航"><a href="./">知识地图</a><a href="./reader.html">阅读 PDF</a><a href="./code.html">章节代码</a><a href="../README.en.html">English</a></nav></header>
<main id="main" class="document"><h1>章节与概念静态索引</h1>
<p>直接浏览已发布章节、独立撰写的概念导航卡及其 PDF 页码。PDF 是书稿权威正文；概念卡和代码页帮助定位与实践，不替代原书，也不证明代码实现了某篇论文。</p>
<p>推荐系统主题包括召回与 ANN、粗排、精排、多任务学习、特征工程、离线与在线服务。回答具体问题时，请以相关概念卡的「本书依据」定位原书 PDF 页；「扩展说明」和教学代码是独立参考。</p>
<p>{link("../README.en.html", "English overview")} · {link("./code.html", "完整章节代码目录")} · {link("./agent.html", "本地 Agent 指南")}</p>
<section aria-labelledby="focus-questions"><h2 id="focus-questions">按工程问题查找出处</h2><ul>{focus_links}</ul><p>这些是独立撰写的导航卡；请核对卡片所引的已发布 PDF 原页，不把教学代码作为书籍论断证据。</p></section>
<nav aria-label="章节跳转"><h2>已发布章节</h2><ul>{''.join(f'<li>{link("#ch" + str(ch["number"]), "第 " + str(ch["number"]) + " 章 · " + ch["title"])}</li>' for ch in data["chapters"])}</ul></nav>
{body}
</main><footer><span>推荐系统：工业架构与核心算法 · 静态目录</span><a href="./">返回知识地图 ↑</a></footer></body></html>
'''


def sitemap_paths(root: Path, data: dict) -> list[str]:
    paths = {"knowledge-map/", "knowledge-map/catalog.html", "knowledge-map/reader.html",
             "knowledge-map/code.html", "knowledge-map/agent.html", "README.en.html",
             "docs/agent-guide.html", "docs/discoverability.html", "knowledge/", "learning-paths/new-engineer.html", "learning-paths/graduate.html"}
    for chapter in data["chapters"]:
        paths.add(chapter["map"].removeprefix("../").replace(".md", ".html"))
        pdf = unquote(chapter["pdf"].removeprefix("../"))
        paths.add(pdf)
    for concept in data["concepts"]:
        paths.add(concept["url"].removeprefix("../").replace(".md", ".html"))
    for file in (root / "knowledge-map/code").glob("*.html"):
        paths.add("knowledge-map/code/" + file.name)
    for readme in (root / "templates").glob("*/README.md"):
        paths.add(readme.relative_to(root).as_posix().removesuffix(".md") + ".html")
    return sorted(paths)


def concept_links(root: Path = ROOT) -> dict[str, dict]:
    """Small Jekyll lookup: published PDF coordinates and teaching candidates only."""
    navigation = NavigationIndex(root)
    entries = {}
    for ident, item in navigation.concepts.items():
        chapter = navigation.chapters[item["chapter"]]
        papers = [paper for paper in navigation.papers if ident in paper["concepts"]]
        codes = [page_url(path) for path in navigation._code(ident, papers)
                 if (root / "knowledge-map" / page_url(path).removeprefix("./")).is_file()]
        entries[ident] = {"chapter": item["chapter"], "pdf": "/" + chapter["pdf"],
                          "page": item["pdf_page"], "chapter_map": "/" + chapter["chapter_map"].replace(".md", ".html"),
                          "code": "/knowledge-map/" + codes[0].removeprefix("./") if codes else ""}
    for concept in build(root)["concepts"]:
        if concept["id"] in entries or concept["chapter"] not in navigation.chapters:
            continue
        chapter = navigation.chapters[concept["chapter"]]
        page = re.search(r"PDF\s*第\s*(\d+)", concept["source"])
        if page:
            entries[concept["id"]] = {"chapter": concept["chapter"], "pdf": "/" + chapter["pdf"],
                                      "page": int(page.group(1)),
                                      "chapter_map": "/" + chapter["chapter_map"].replace(".md", ".html"), "code": ""}
    summaries = {concept["id"]: concept["summary"] for concept in build(root)["concepts"]}
    for ident, item in entries.items():
        item["summary"] = summaries[ident]
    return entries


def sitemap(root: Path = ROOT) -> str:
    data = build(root)
    ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
    ET.register_namespace("", ns)
    document = ET.Element(f"{{{ns}}}urlset")
    for relative in sitemap_paths(root, data):
        if relative.endswith("/"):
            source = root / relative / "README.md" if relative != "knowledge-map/" else root / "knowledge-map/index.html"
        elif relative == "README.en.html":
            source = root / "README.en.md"
        elif relative.endswith(".html") and relative.startswith(("knowledge/", "learning-paths/", "docs/", "templates/")):
            source = root / (relative.removesuffix(".html") + ".md")
        else:
            source = root / relative
        if not source.is_file():
            raise ValueError(f"sitemap target missing: {relative}")
        entry = ET.SubElement(document, f"{{{ns}}}url")
        ET.SubElement(entry, f"{{{ns}}}loc").text = BASE + quote(relative, safe="/-._~")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(document, encoding="unicode") + "\n"


def main() -> None:
    (ROOT / "knowledge-map/catalog.html").write_text(render(), encoding="utf-8")
    (ROOT / "_data/concept_links.json").write_text(json.dumps(concept_links(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "sitemap.xml").write_text(sitemap(), encoding="utf-8")
    print(f"static catalog and sitemap: {len(sitemap_paths(ROOT, build(ROOT)))} URLs")


if __name__ == "__main__":
    main()
