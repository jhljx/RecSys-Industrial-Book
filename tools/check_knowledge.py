#!/usr/bin/env python3
"""Validate curated knowledge metadata and PDF page citations."""
from __future__ import annotations

import json
import re
from pathlib import Path



def pdf_page_count(path: Path) -> int:
    data = path.read_bytes()
    counts = [int(value) for value in re.findall(rb"/Count\s+(\d+)", data)]
    if not counts:
        raise ValueError(f"cannot determine PDF page count: {path}")
    return max(counts)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    seen: set[int] = set()
    for path in sorted((root / "knowledge/chapters").glob("ch*.md")):
        text = path.read_text(encoding="utf-8")
        chapter_match = re.search(r"^chapter:\s*(\d+)\s*$", text, re.MULTILINE)
        pdf_match = re.search(r"^pdf:\s*(.+?)\s*$", text, re.MULTILINE)
        if not chapter_match or not pdf_match:
            failures.append(f"{path}: missing chapter/pdf frontmatter")
            continue
        chapter = int(chapter_match.group(1))
        seen.add(chapter)
        pdf = root / pdf_match.group(1)
        if not pdf.is_file():
            failures.append(f"{path}: PDF not found: {pdf_match.group(1)}")
            continue
        pages = pdf_page_count(pdf)
        citations = re.findall(r"PDF 第\s*(\d+)(?:[–-](\d+))?\s*页", text)
        if not citations:
            failures.append(f"{path}: no PDF page citation")
        for start, end in citations:
            upper = int(end or start)
            if int(start) < 1 or upper < int(start) or upper > pages:
                failures.append(f"{path}: invalid PDF page range {start}-{upper}; PDF has {pages}")
        if "不是正文转录" not in text and "仅作独立导航" not in text and "本书依据" not in text:
            failures.append(f"{path}: missing authority boundary")
    cards: dict[str, tuple[Path, str]] = {}
    for path in sorted((root / "knowledge/concepts").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        card_id = re.search(r"^id:\s*([a-z0-9-]+)\s*$", text, re.MULTILINE)
        prerequisites = re.search(r"^prerequisites:\s*\[(.*?)\]\s*$", text, re.MULTILINE)
        if not card_id or not prerequisites or card_id.group(1) != path.stem:
            failures.append(f"{path}: invalid id or prerequisites frontmatter")
            continue
        for section in ("核心判断", "工程检查", "本书依据", "扩展说明"):
            if f"## {section}" not in text:
                failures.append(f"{path}: missing section {section}")
        if not re.search(r"PDF 第\s*\d+", text):
            failures.append(f"{path}: no PDF page citation")
        cards[card_id.group(1)] = (path, prerequisites.group(1))
    for _, (path, raw_prerequisites) in cards.items():
        for prerequisite in (item.strip() for item in raw_prerequisites.split(",") if item.strip()):
            if prerequisite not in cards:
                failures.append(f"{path}: unknown prerequisite {prerequisite}")

    graph_path = root / "knowledge/graph.json"
    graph_text = graph_path.read_text(encoding="utf-8")
    graph_nodes = set(re.findall(r'^\s+"([a-z0-9-]+)"[,]?$', graph_text, re.MULTILINE))
    if graph_nodes != set(cards):
        failures.append(f"{graph_path}: node set does not match concept cards")
    courses_path = root / "learning-paths/courses.json"
    try:
        courses = json.loads(courses_path.read_text(encoding="utf-8"))["courses"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        failures.append(f"{courses_path}: invalid course catalog: {error}")
        courses = {}
    expected_chapters = seen
    for course_id, course in courses.items():
        units = course.get("units", [])
        unit_ids = [unit.get("id") for unit in units]
        covered = {chapter for unit in units for chapter in unit.get("chapters", [])}
        concepts = {concept for unit in units for concept in unit.get("concepts", [])}
        if len(unit_ids) != len(set(unit_ids)):
            failures.append(f"{courses_path}: {course_id} has duplicate unit ids")
        if covered != expected_chapters:
            failures.append(f"{courses_path}: {course_id} chapter coverage differs: missing={sorted(expected_chapters-covered)} extra={sorted(covered-expected_chapters)}")
        unknown = concepts - set(cards)
        if unknown:
            failures.append(f"{courses_path}: {course_id} references unknown concepts {sorted(unknown)}")
        for unit in units:
            for field in ("id", "title", "reading", "deliverable", "acceptance", "questions"):
                if not unit.get(field):
                    failures.append(f"{courses_path}: {course_id}/{unit.get('id')} missing {field}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"knowledge check passed: {len(seen)} chapter maps with valid PDF citations")


if __name__ == "__main__":
    main()
