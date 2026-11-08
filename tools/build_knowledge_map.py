#!/usr/bin/env python3
"""Generate the public, offline-readable knowledge-map dataset from curated metadata."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from check_knowledge import pdf_page_count

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "knowledge-map/data.js"
# Editorial taxonomy, not a replacement for chapter prose. Every published chapter appears once.
GROUPS = [
    ("foundation", "系统基础", "先建立目标、反馈、架构与评价的共同语言", [0, 1, 2, 3]),
    ("funnel", "推荐漏斗", "从候选供给到最终列表的在线决策链", [4, 5, 6, 7, 8, 9]),
    ("models", "核心模型", "在各层预算约束下选择召回、粗排与精排模型", [10, 11, 12]),
    ("engineering", "数据与服务", "让特征、训练、推理和检索保持可用与一致", [15, 16, 17, 18]),
    ("frontiers", "问题与前沿", "诊断偏差、建模时长，审慎引入探索", [19, 20, 23, 28]),
]


def frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"\A---\n(.*?)\n---", text, re.S)
    if not match:
        raise ValueError("missing frontmatter")
    return dict(re.findall(r"^([a-z_]+):\s*(.+)$", match.group(1), re.M))


def section(text: str, heading: str) -> str:
    match = re.search(rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", text)
    return match.group(1).strip() if match else ""


def load_papers(root: Path, chapters: list[dict], concept_ids: set[str]) -> list[dict]:
    """Validate small editorial paper pointers against released PDFs, never PDF text."""
    catalog = json.loads((root / "knowledge/paper-map.json").read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or catalog.get("version") != 1 or not isinstance(catalog.get("papers"), list):
        raise ValueError("invalid paper-map schema")
    chapter_by_number = {chapter["number"]: chapter for chapter in chapters}
    seen: set[str] = set()
    pages_by_chapter: dict[int, int] = {}
    papers = []
    required = {"id", "method", "title", "chapter", "pages", "concepts"}
    for paper in catalog["papers"]:
        if not isinstance(paper, dict) or set(paper) - (required | {"url"}) or required - set(paper):
            raise ValueError("invalid paper entry fields")
        paper_id = paper["id"]
        if not isinstance(paper_id, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", paper_id) or paper_id in seen:
            raise ValueError(f"invalid or duplicate paper id: {paper_id}")
        seen.add(paper_id)
        if any(not isinstance(paper[key], str) or not paper[key].strip() for key in ("method", "title")):
            raise ValueError(f"invalid paper metadata: {paper_id}")
        chapter = paper["chapter"]
        if type(chapter) is not int or chapter not in chapter_by_number:
            raise ValueError(f"unknown paper chapter: {paper_id}: {chapter}")
        pages = paper["pages"]
        if (not isinstance(pages, list) or len(pages) != 2 or
                any(type(page) is not int for page in pages) or pages[0] < 1 or pages[1] < pages[0]):
            raise ValueError(f"invalid paper pages: {paper_id}")
        if chapter not in pages_by_chapter:
            pdf = root / unquote(chapter_by_number[chapter]["pdf"].removeprefix("../"))
            pages_by_chapter[chapter] = pdf_page_count(pdf)
        if pages[1] > pages_by_chapter[chapter]:
            raise ValueError(f"paper pages exceed PDF bounds: {paper_id}")
        references = paper["concepts"]
        if (not isinstance(references, list) or not references or
                any(not isinstance(item, str) or item not in concept_ids for item in references) or
                len(set(references)) != len(references)):
            raise ValueError(f"unknown or duplicate paper concepts: {paper_id}")
        if "url" in paper:
            url = paper["url"]
            if not isinstance(url, str) or urlparse(url).scheme != "https" or not urlparse(url).netloc:
                raise ValueError(f"invalid paper URL: {paper_id}")
        papers.append(paper)
    return papers


def build(root: Path = ROOT) -> dict:
    groups = [{"id": key, "title": title, "description": description, "chapters": list(chapters)} for key, title, description, chapters in GROUPS]
    chapter_files = sorted((root / "knowledge/chapters").glob("ch[0-9][0-9].md"))
    # Newly released chapters remain visible before an editor assigns a permanent taxonomy.
    published = {int(frontmatter(path.read_text(encoding="utf-8"))["chapter"]) for path in chapter_files}
    ungrouped = sorted(published - {n for group in groups for n in group["chapters"]})
    if ungrouped:
        groups.append({"id": "additional", "title": "新增章节", "description": "待纳入专题分组的已发布章节", "chapters": ungrouped})
    chapters = []
    for path in chapter_files:
        text = path.read_text(encoding="utf-8")
        metadata = frontmatter(text)
        number = int(metadata["chapter"])
        group = next((g["id"] for g in groups if number in g["chapters"]), None)
        if group is None:
            raise ValueError(f"unclassified chapter: {number}")
        pdf = metadata["pdf"]
        if not (root / pdf).is_file():
            raise ValueError(f"missing PDF: {pdf}")
        rows = []
        for line in text.splitlines():
            if not line.startswith("| ") or "PDF 页" in line or "---" in line:
                continue
            cells = [cell.strip() for cell in line.strip("| ").split("|")]
            if len(cells) == 4 and re.search(r"\d", cells[1]):
                rows.append({"section": cells[0], "pages": cells[1], "question": cells[2], "check": cells[3]})
        if number == 0:
            focus = re.findall(r"^\|\s*([\d–-]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|", text, re.M)
            rows = [{"section": title.strip(), "pages": pages.strip(), "question": takeaway.strip(), "check": "区分算法、架构、工程约束与业务权衡。"} for pages, title, takeaway in focus]
        chapters.append({"number": number, "title": metadata["title"], "group": group,
                         "pdf": "../" + quote(pdf, safe="/"), "map": "../knowledge/chapters/" + path.name,
                         "sections": rows})
    graph = json.loads((root / "knowledge/graph.json").read_text(encoding="utf-8"))
    concepts = []
    for path in sorted((root / "knowledge/concepts").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        metadata = frontmatter(text)
        number = int(metadata["chapter"])
        source = section(text, "本书依据")
        if not source or not re.search(r"PDF\s*第\s*\d+", source):
            raise ValueError(f"missing citation: {path}")
        concepts.append({"id": metadata["id"], "title": metadata["title"], "chapter": number,
                         "definition": section(text, "定义"), "summary": section(text, "核心判断"), "check": section(text, "工程检查"),
                         "source": source, "tradeoffs": section(text, "适用与取舍"), "verification": section(text, "如何验证"),
                         "misjudgments": section(text, "容易误判"), "url": "../knowledge/concepts/" + path.name})
    courses = json.loads((root / "learning-paths/courses.json").read_text(encoding="utf-8"))["courses"]
    paths = [{"id": key, "title": course["title"], "audience": course["audience"],
              "url": "../learning-paths/" + key + ".md",
              "units": [{"id": unit["id"], "title": unit["title"], "week": unit["week"],
                         "chapters": unit["chapters"], "concepts": unit["concepts"], "reading": unit["reading"],
                         "deliverable": unit["deliverable"]} for unit in course["units"]]}
             for key, course in courses.items()]
    chapter_ids = {row["number"] for row in chapters}
    concept_ids = {row["id"] for row in concepts}
    if len(chapter_ids) != len(chapters) or chapter_ids != {n for g in groups for n in g["chapters"]}:
        raise ValueError("chapter grouping mismatch")
    if set(graph["nodes"]) != concept_ids or any(a not in concept_ids or b not in concept_ids for a, b in graph["edges"]):
        raise ValueError("concept graph mismatch")
    # Some curated concepts anticipate an unreleased chapter; keep their existing cited PDF
    # visible, but never imply that an unpublished chapter has a released PDF.
    if any(c["chapter"] not in chapter_ids and not c["source"] for c in concepts):
        raise ValueError("unpublished concept lacks released-PDF citation")
    if any(set(u["chapters"]) - chapter_ids or set(u["concepts"]) - concept_ids for p in paths for u in p["units"]):
        raise ValueError("course reference mismatch")
    papers = load_papers(root, chapters, concept_ids)
    return {"version": 1, "groups": groups, "chapters": chapters, "concepts": concepts,
            "edges": graph["edges"], "paths": paths, "papers": papers}



def main() -> None:
    data = build()
    OUTPUT.write_text("/* Generated by tools/build_knowledge_map.py; do not edit manually. */\n" +
                      "window.KNOWLEDGE_MAP = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"knowledge map: {len(data['chapters'])} chapters, {len(data['concepts'])} concepts, "
          f"{len(data['edges'])} dependencies, {len(data['paths'])} courses -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
