#!/usr/bin/env python3
"""Generate the public, offline-readable knowledge-map dataset from curated metadata."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "knowledge-map/data.js"
# Editorial taxonomy, not a replacement for chapter prose. Every published chapter appears once.
GROUPS = [
    ("foundation", "系统基础", "先建立目标、反馈、架构与评价的共同语言", [0, 1, 2, 3]),
    ("funnel", "推荐漏斗", "从候选供给到最终列表的在线决策链", [4, 5, 6, 7, 8, 9]),
    ("models", "核心模型", "在各层预算约束下选择召回、粗排与精排模型", [10, 11, 12]),
    ("engineering", "数据与服务", "让特征、训练、推理和检索保持可用与一致", [15, 16, 17, 18]),
    ("frontiers", "问题与前沿", "诊断偏差、建模时长，审慎引入探索", [19, 23, 28]),
]


def frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"\A---\n(.*?)\n---", text, re.S)
    if not match:
        raise ValueError("missing frontmatter")
    return dict(re.findall(r"^([a-z_]+):\s*(.+)$", match.group(1), re.M))


def section(text: str, heading: str) -> str:
    match = re.search(rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", text)
    return match.group(1).strip() if match else ""


def build(root: Path = ROOT) -> dict:
    groups = [{"id": key, "title": title, "description": description, "chapters": chapters} for key, title, description, chapters in GROUPS]
    chapter_files = sorted((root / "knowledge/chapters").glob("ch[0-9][0-9].md"))
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
                         "summary": section(text, "核心判断"), "check": section(text, "工程检查"),
                         "source": source, "url": "../knowledge/concepts/" + path.name})
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
    return {"version": 1, "groups": groups, "chapters": chapters, "concepts": concepts,
            "edges": graph["edges"], "paths": paths}



def main() -> None:
    data = build()
    OUTPUT.write_text("/* Generated by tools/build_knowledge_map.py; do not edit manually. */\n" +
                      "window.KNOWLEDGE_MAP = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"knowledge map: {len(data['chapters'])} chapters, {len(data['concepts'])} concepts, "
          f"{len(data['edges'])} dependencies, {len(data['paths'])} courses -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
