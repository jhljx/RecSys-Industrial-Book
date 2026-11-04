import json
import re
from pathlib import Path


# Only chapter maps represent released PDFs; drafts and untracked local files are not authority.
CITATION = re.compile(r"《([^》]+\.pdf)》")
PAGE = re.compile(r"PDF\s*第\s*(\d+)(?:\s*[–-]\s*(\d+))?(?:\s*[、]\s*(\d+)(?:\s*[–-]\s*(\d+))?)?\s*页")
READING = re.compile(r"ch(\d{2})\s+PDF\s+([^；（]+)")


def pdf_page_count(path: Path) -> int:
    data = path.read_bytes()
    counts = [int(value) for value in re.findall(rb"/Count\s+(\d+)", data)]
    if not counts:
        raise ValueError(f"cannot determine PDF page count: {path}")
    return max(counts)


def validate_references(root: Path, cards: dict[str, tuple[Path, str, str]],
                        released: dict[int, tuple[str, int]], failures: list[str]) -> None:
    """Check book citations against the released catalog and prerequisite graph."""
    released_pdfs = {name: pages for name, pages in released.values()}
    prerequisites_by_id: dict[str, set[str]] = {}
    for card_id, (path, raw_prerequisites, text) in cards.items():
        prerequisites = [item.strip() for item in raw_prerequisites.split(",") if item.strip()]
        prerequisites_by_id[card_id] = set(prerequisites)
        if len(set(prerequisites)) != len(prerequisites):
            failures.append(f"{path}: duplicate prerequisites")
        for prerequisite in prerequisites:
            if prerequisite not in cards:
                failures.append(f"{path}: unknown prerequisite {prerequisite}")
            if prerequisite == card_id:
                failures.append(f"{path}: self prerequisite")
        chapter_match = re.search(r"^chapter:\s*(\d+)\s*$", text, re.MULTILINE)
        if not chapter_match:
            failures.append(f"{path}: missing chapter frontmatter")
            continue
        chapter = int(chapter_match.group(1))
        source = re.search(r"(?ms)^## 本书依据\s*\n(.*?)(?=^## |\Z)", text)
        source_text = source.group(1) if source else ""
        names = CITATION.findall(source_text)
        if not names:
            failures.append(f"{path}: no released PDF citation in 本书依据")
        for name in names:
            if name not in released_pdfs:
                failures.append(f"{path}: unreleased PDF citation: {name}")
        if chapter in released and released[chapter][0] not in names:
            failures.append(f"{path}: chapter {chapter} PDF missing from 本书依据")
        pages = PAGE.findall(source_text)
        if not pages:
            failures.append(f"{path}: no PDF page citation in 本书依据")
        # Match page coordinates to their immediately preceding PDF title, including
        # repeated page references within the same cited PDF clause.
        for match in CITATION.finditer(source_text):
            name = match.group(1)
            next_match = CITATION.search(source_text, match.end())
            end = next_match.start() if next_match else len(source_text)
            cited_pages = PAGE.findall(source_text[match.end():end])
            if not cited_pages:
                failures.append(f"{path}: PDF citation has no page: {name}")
            for start, last, extra, extra_last in cited_pages:
                upper = int(last or start)
                second = int(extra_last or extra) if extra else 0
                if (int(start) < 1 or upper < int(start) or (extra and (int(extra) < 1 or second < int(extra))) or
                        (name in released_pdfs and (upper > released_pdfs[name] or second > released_pdfs[name]))):
                    failures.append(f"{path}: invalid PDF page range {start}-{upper}: {name}")

    graph_path = root / "knowledge/graph.json"
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes, edges = graph["nodes"], graph["edges"]
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("nodes/edges must be lists")
        if len(nodes) != len(set(nodes)) or set(nodes) != set(cards):
            failures.append(f"{graph_path}: node set does not match concept cards or has duplicates")
        expected = {(parent, child) for child, parents in prerequisites_by_id.items() for parent in parents}
        if any(not isinstance(edge, list) or len(edge) != 2 or
               any(not isinstance(node, str) for node in edge) for edge in edges):
            raise ValueError("edges must be pairs of concept ids")
        actual = {tuple(edge) for edge in edges}
        if len(actual) != len(edges) or actual != expected:
            failures.append(f"{graph_path}: edges differ from card prerequisites or have duplicates")
        # Color DFS catches cycles even when graph/card edges are both changed together.
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError(f"prerequisite cycle at {node}")
            if node in visited:
                return
            visiting.add(node)
            for parent in prerequisites_by_id.get(node, set()) & cards.keys():
                visit(parent)
            visiting.remove(node)
            visited.add(node)

        for node in cards:
            visit(node)
    except (OSError, KeyError, TypeError, ValueError) as error:
        failures.append(f"{graph_path}: invalid graph: {error}")


def validate_course_reading(reading: str, chapters: list[int],
                            released: dict[int, tuple[str, int]]) -> list[str]:
    """Check explicitly named chNN PDF reading pages, not free-form revision notes."""
    failures: list[str] = []
    cited: set[int] = set()
    for match in READING.finditer(reading):
        chapter = int(match.group(1))
        cited.add(chapter)
        if chapter not in released or chapter not in chapters:
            failures.append(f"reading references chapter {chapter} outside unit/released chapters")
            continue
        for part in match.group(2).strip().split("、"):
            page = re.fullmatch(r"\s*(\d+)(?:\s*[–-]\s*(\d+))?\s*", part)
            if not page:
                failures.append(f"invalid reading page range: {part}")
                continue
            start, end = int(page.group(1)), int(page.group(2) or page.group(1))
            if start < 1 or end < start or end > released[chapter][1]:
                failures.append(f"chapter {chapter} reading pages {start}-{end} out of bounds")
    if cited and cited != set(chapters):
        failures.append(f"reading chapters differ from unit chapters: {sorted(cited)} != {sorted(set(chapters))}")
    if re.search(r"ch\d{2}\s+PDF", reading) and not cited:
        failures.append("invalid chapter reading format")
    return failures


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    seen: set[int] = set()
    released: dict[int, tuple[str, int]] = {}
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
        released[chapter] = (pdf.name, pages)
        citations = re.findall(r"PDF 第\s*(\d+)(?:[–-](\d+))?\s*页", text)
        if not citations:
            failures.append(f"{path}: no PDF page citation")
        for start, end in citations:
            upper = int(end or start)
            if int(start) < 1 or upper < int(start) or upper > pages:
                failures.append(f"{path}: invalid PDF page range {start}-{upper}; PDF has {pages}")
        if "不是正文转录" not in text and "仅作独立导航" not in text and "本书依据" not in text:
            failures.append(f"{path}: missing authority boundary")
    cards: dict[str, tuple[Path, str, str]] = {}
    for path in sorted((root / "knowledge/concepts").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        card_id = re.search(r"^id:\s*([a-z0-9-]+)\s*$", text, re.MULTILINE)
        prerequisites = re.search(r"^prerequisites:\s*\[(.*?)\]\s*$", text, re.MULTILINE)
        if not card_id or not prerequisites or card_id.group(1) != path.stem:
            failures.append(f"{path}: invalid id or prerequisites frontmatter")
            continue
        for section in ("定义", "核心判断", "工程检查", "本书依据", "扩展说明"):
            if f"## {section}" not in text:
                failures.append(f"{path}: missing section {section}")
        cards[card_id.group(1)] = (path, prerequisites.group(1), text)
    validate_references(root, cards, released, failures)

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
        if course_id in {"new-engineer", "graduate"}:
            graduation_id = "ne-12" if course_id == "new-engineer" else "gr-12"
            graduation = [unit for unit in units if unit.get("id") == graduation_id]
            if len(graduation) != 1:
                failures.append(f"{courses_path}: {course_id} needs exactly one graduation unit {graduation_id}")
            else:
                taught = {chapter for unit in units if unit.get("id") != graduation_id for chapter in unit.get("chapters", [])}
                missing_from_graduation = taught - set(graduation[0].get("chapters", []))
                if missing_from_graduation:
                    failures.append(f"{courses_path}: {course_id}/{graduation_id} missing taught chapters {sorted(missing_from_graduation)}")
        for unit in units:
            for field in ("id", "title", "reading", "deliverable", "acceptance", "questions"):
                if not unit.get(field):
                    failures.append(f"{courses_path}: {course_id}/{unit.get('id')} missing {field}")
            for error in validate_course_reading(unit.get("reading", ""), unit.get("chapters", []), released):
                failures.append(f"{courses_path}: {course_id}/{unit.get('id')}: {error}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"knowledge check passed: {len(seen)} chapter maps with valid PDF citations")


if __name__ == "__main__":
    main()
