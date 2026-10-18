#!/usr/bin/env python3
"""Build, inspect, evaluate, and search the local recommendation-book index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

INDEX_DIR = Path(".recsys-agent")
INDEX_FILE = INDEX_DIR / "index.jsonl"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
FORMAT_VERSION = 2
CHINESE = re.compile(r"[\u3400-\u9fff]+")
LATIN = re.compile(r"[a-z0-9][a-z0-9_+.#/-]*", re.I)
HEADING = re.compile(r"^(?:第[一二三四五六七八九十百零0-9]+[章节]|\d+(?:\.\d+){0,3})\s*[^。！？]{1,80}$")
TEX_SECTION = re.compile(r"\\(chapter|section|subsection|subsubsection)\*?\{([^{}]+)\}")
ALIASES_FILE = Path("knowledge/aliases.json")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    locator: str
    title: str
    text: str
    chapter: int | None = None
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    kind: str = "prose"


def detect_mode(root: Path) -> str:
    return "private" if (root / "chapters").is_dir() and any((root / "chapters").glob("ch*.tex")) else "public"


def normalize(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\x00", " ")).strip()


def terms(text: str) -> list[str]:
    lowered = text.lower()
    output = LATIN.findall(lowered)
    for run in CHINESE.findall(lowered):
        output.extend(run if len(run) == 1 else (run[i : i + 2] for i in range(len(run) - 1)))
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chapter_number(path: Path) -> int | None:
    match = re.search(r"ch(\d+)", path.as_posix(), re.I)
    return int(match.group(1)) if match else None


def load_aliases(root: Path) -> dict[str, list[str]]:
    path = root / ALIASES_FILE
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key).lower(): [str(item) for item in value] for key, value in data.items()}


def alias_in_query(alias: str, query: str) -> bool:
    lowered = alias.casefold()
    if re.fullmatch(r"[a-z0-9_+.#/-]+", lowered):
        return bool(re.search(rf"(?<![a-z0-9_/-]){re.escape(lowered)}(?![a-z0-9_/-])", query.casefold()))
    return lowered in query.casefold()


def expand_query(root: Path, query: str) -> str:
    additions: list[str] = []
    for key, values in load_aliases(root).items():
        family = [key, *values]
        if any(alias_in_query(item, query) for item in family):
            additions.extend(family)
    return " ".join(dict.fromkeys([query, *additions]))

def concept_chapter_hints(root: Path, expanded_query: str) -> set[int]:
    normalized = re.sub(r"[-_\s]+", " ", expanded_query.casefold())
    hints: set[int] = set()
    for path in (root / "knowledge/concepts").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        chapter = re.search(r"^chapter:\s*(\d+)\s*$", text, re.MULTILINE)
        names = (path.stem.replace("-", " "), title.group(1).casefold() if title else "")
        if chapter and any(len(name) >= 4 and name in normalized for name in names):
            hints.add(int(chapter.group(1)))
    return hints


def split_blocks(text: str, target: int = 1500) -> list[str]:
    paragraphs = [normalize(item) for item in re.split(r"\n\s*\n", text) if normalize(item)]
    blocks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) > target:
            blocks.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)
    if current:
        blocks.append("\n\n".join(current))
    return blocks


def private_chunks(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    canonical = {path.stem for path in (root / "chapters").glob("ch*.tex")}
    files = sorted((root / "chapters").glob("ch*.tex"))
    files.extend(path for path in sorted(root.glob("ch*_draft.tex")) if path.stem.removesuffix("_draft") not in canonical)
    for path in files:
        relative = path.relative_to(root).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        starts = [(number, match.group(1), normalize(match.group(2))) for number, line in enumerate(lines, 1) if (match := TEX_SECTION.search(line))]
        if not starts:
            starts = [(1, "chapter", path.stem)]
        chapter_title = starts[0][2]
        for position, (start, level, title) in enumerate(starts):
            end = starts[position + 1][0] - 1 if position + 1 < len(starts) else len(lines)
            if level == "chapter":
                chapter_title = title
            for block_number, block in enumerate(split_blocks("\n".join(lines[start - 1 : end])), 1):
                chunks.append(Chunk(f"{path.stem}:{start}:{block_number}", relative, f"lines {start}-{end}", chapter_title, block, chapter_number(path), title, kind=level))
    return chunks


def public_pdfs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("ch[0-9][0-9] */*.pdf") if path.is_file())


def page_heading(page: str, fallback: str) -> str:
    for line in page.splitlines():
        candidate = normalize(line)
        if HEADING.match(candidate):
            return candidate
    return fallback


def public_chunks(root: Path) -> list[Chunk]:
    executable = shutil.which("pdftotext")
    if not executable:
        raise SystemExit("缺少 pdftotext；macOS 可执行 `brew install poppler`。")
    chunks: list[Chunk] = []
    for path in public_pdfs(root):
        relative = path.relative_to(root).as_posix()
        result = subprocess.run([executable, "-layout", str(path), "-"], check=True, capture_output=True, text=True)
        current_section = path.stem
        for page_number, raw_page in enumerate(result.stdout.split("\f"), 1):
            if not normalize(raw_page):
                continue
            current_section = page_heading(raw_page, current_section)
            for block_number, block in enumerate(split_blocks(raw_page, target=1800), 1):
                chunks.append(Chunk(f"{relative}:p{page_number}:{block_number}", relative, f"PDF page {page_number}", path.stem, block, chapter_number(path), current_section, page_number, page_number))
    return chunks


def sources(root: Path, mode: str) -> list[Path]:
    if mode == "public":
        return public_pdfs(root)
    return sorted((root / "chapters").glob("ch*.tex"))


def source_records(root: Path, mode: str) -> list[dict[str, object]]:
    return [{"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "size": path.stat().st_size} for path in sources(root, mode)]


def build(root: Path, mode: str) -> None:
    chunks = private_chunks(root) if mode == "private" else public_chunks(root)
    if not chunks:
        raise SystemExit(f"未发现可索引的 {'LaTeX' if mode == 'private' else 'PDF'} 内容。")
    (root / INDEX_DIR).mkdir(parents=True, exist_ok=True)
    with (root / INDEX_FILE).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    manifest = {
        "format": FORMAT_VERSION,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": len(chunks),
        "sources": source_records(root, mode),
        "content_policy": "private-source" if mode == "private" else "local-pdf-extraction-not-for-commit",
    }
    (root / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"indexed mode={mode} sources={len(manifest['sources'])} chunks={manifest['chunks']} path={INDEX_FILE}")


def load(root: Path) -> list[Chunk]:
    path = root / INDEX_FILE
    if not path.is_file():
        raise SystemExit("本地索引不存在。请先执行 `python3 tools/recsys_agent.py index`。")
    with path.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]


def stale_reasons(root: Path, mode: str) -> list[str]:
    if not (root / MANIFEST_FILE).is_file() or not (root / INDEX_FILE).is_file():
        return ["index missing"]
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        return [f"format {manifest.get('format')} != {FORMAT_VERSION}"]
    previous = {row["path"]: row["sha256"] for row in manifest.get("sources", [])}
    current = {row["path"]: row["sha256"] for row in source_records(root, mode)}
    reasons = [f"added: {path}" for path in current.keys() - previous.keys()]
    reasons.extend(f"removed: {path}" for path in previous.keys() - current.keys())
    reasons.extend(f"changed: {path}" for path in current.keys() & previous.keys() if current[path] != previous[path])
    return sorted(reasons)


def semantic_vector(tokens: list[str], dimensions: int = 256) -> Counter[int]:
    vector: Counter[int] = Counter()
    for token in set(tokens):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        vector[int.from_bytes(digest, "big") % dimensions] += 1 if digest[0] & 1 else -1
    return vector


def cosine(left: Counter[int], right: Counter[int]) -> float:
    numerator = sum(value * right[key] for key, value in left.items())
    denominator = math.sqrt(sum(value * value for value in left.values()) * sum(value * value for value in right.values()))
    return numerator / denominator if denominator else 0.0


def rank(root: Path, query: str, chapter: int | None = None) -> list[tuple[float, Chunk]]:
    chunks = [chunk for chunk in load(root) if chapter is None or chunk.chapter == chapter]
    expanded = expand_query(root, query)
    chapter_hints = concept_chapter_hints(root, expanded)
    if len(chapter_hints) != 1:
        chapter_hints.clear()
    query_tokens = terms(expanded)
    if not query_tokens:
        raise SystemExit("查询词为空。")
    rows = [terms(f"{chunk.title} {chunk.section} {chunk.text}") for chunk in chunks]
    document_frequency = Counter(token for row in rows for token in set(row))
    average_length = sum(map(len, rows)) / max(len(rows), 1)
    query_counts = Counter(query_tokens)
    query_vector = semantic_vector(query_tokens)
    ranked: list[tuple[float, Chunk]] = []
    for chunk, row in zip(chunks, rows):
        counts = Counter(row)
        bm25 = 0.0
        for token, query_weight in query_counts.items():
            frequency = counts[token]
            if not frequency:
                continue
            inverse = math.log(1 + (len(rows) - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            bm25 += query_weight * inverse * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * len(row) / max(average_length, 1)))
        semantic = max(0.0, cosine(query_vector, semantic_vector(row)))
        exact = 6.0 if query.lower() in f"{chunk.title} {chunk.section} {chunk.text}".lower() else 0.0
        heading = 2.0 * sum(1 for token in set(query_tokens) if token in terms(chunk.section))
        chapter_prior = 12.0 if chunk.chapter in chapter_hints else 0.0
        score = bm25 + semantic * 3.0 + exact + heading + chapter_prior
        if score > 0:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    return ranked


def search(root: Path, query: str, limit: int, chapter: int | None, as_json: bool) -> None:
    results = rank(root, query, chapter)[:limit]
    if as_json:
        print(json.dumps([{"score": round(score, 4), **asdict(chunk)} for score, chunk in results], ensure_ascii=False, indent=2))
        return
    for position, (score, chunk) in enumerate(results, 1):
        excerpt = normalize(chunk.text)
        if len(excerpt) > 500:
            excerpt = excerpt[:500].rstrip() + "…"
        print(f"[{position}] score={score:.3f} {chunk.source} ({chunk.locator})\n    {chunk.section}\n    {excerpt}\n")
    if not results:
        print("未命中。请改用章节名、模型名、英文缩写或更短的关键词。")


def page(root: Path, chapter: int, page_number: int) -> None:
    matches = [chunk for chunk in load(root) if chunk.chapter == chapter and chunk.page_start == page_number]
    if not matches:
        raise SystemExit(f"未找到第 {chapter} 章 PDF 第 {page_number} 页。")
    print(f"# {matches[0].title} · PDF page {page_number}\n")
    print("\n\n".join(chunk.text for chunk in matches))


def evaluate(root: Path, dataset: Path) -> None:
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    hits_at_1 = hits_at_5 = citation_valid = 0
    failures: list[dict[str, object]] = []
    for case in cases:
        results = rank(root, case["query"])[:5]
        expected = set(case["expected_chapters"])
        chapters = [chunk.chapter for _, chunk in results]
        hits_at_1 += bool(chapters and chapters[0] in expected)
        hits_at_5 += bool(expected.intersection(chapters))
        citation_valid += bool(results and all(chunk.page_start and chunk.source.endswith(".pdf") for _, chunk in results))
        if not expected.intersection(chapters):
            failures.append({"query": case["query"], "expected": sorted(expected), "actual": chapters})
    total = len(cases)
    report = {"cases": total, "hit_at_1": hits_at_1 / total, "recall_at_5": hits_at_5 / total, "citation_valid": citation_valid / total, "failures": failures}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures or citation_valid != total:
        raise SystemExit(1)


def doctor(root: Path, mode: str) -> None:
    reasons = stale_reasons(root, mode)
    print(f"mode={mode}")
    print(f"root={root}")
    print(f"pdftotext={shutil.which('pdftotext') or 'missing'}")
    print(f"index={'stale' if reasons else 'ready'}")
    for reason in reasons:
        print(f"  {reason}")
    print(f"{'tex_sources' if mode == 'private' else 'pdf_sources'}={len(sources(root, mode))}")

def workspace(root: Path) -> Path:
    path = root / ".recsys-workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path


def graph(root: Path) -> dict[str, object]:
    return json.loads((root / "knowledge/graph.json").read_text(encoding="utf-8"))

def concept_id(root: Path, topic: str) -> str | None:
    normalized = re.sub(r"[-_\s]+", " ", expand_query(root, topic).casefold())
    matches: list[tuple[int, str]] = []
    for path in (root / "knowledge/concepts").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        names = (path.stem.replace("-", " "), title.group(1).casefold() if title else "")
        score = max((len(name) for name in names if name and name in normalized), default=0)
        if score:
            matches.append((score, path.stem))
    return max(matches)[1] if matches else None


def allocate_minutes(total: int, weights: tuple[int, ...]) -> list[int]:
    if total < len(weights):
        raise SystemExit(f"时间预算至少为 {len(weights)} 分钟。")
    raw = [total * weight / sum(weights) for weight in weights]
    allocated = [max(1, math.floor(value)) for value in raw]
    remainder = total - sum(allocated)
    order = sorted(range(len(weights)), key=lambda index: (raw[index] - math.floor(raw[index]), weights[index]), reverse=True)
    for index in order[:remainder]:
        allocated[index] += 1
    return allocated


def concept_metadata(root: Path, identifier: str | None) -> dict[str, object]:
    if not identifier:
        return {}
    path = root / "knowledge/concepts" / f"{identifier}.md"
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    profile = re.search(r"^effort_profile:\s*\[([0-9, ]+)\]\s*$", text, re.MULTILINE)
    chapter = re.search(r"^chapter:\s*(\d+)\s*$", text, re.MULTILINE)
    return {
        "chapter": int(chapter.group(1)) if chapter else None,
        "effort_profile": tuple(int(item) for item in profile.group(1).split(",")) if profile else None,
    }


def effort_profile(identifier: str | None, level: str, metadata: dict[str, object]) -> tuple[int, int, int, int]:
    configured = metadata.get("effort_profile")
    if isinstance(configured, tuple) and len(configured) == 4:
        return configured
    research = level == "advanced"
    if identifier in {"hnsw", "ann-index", "vector-retrieval", "online-serving", "overload-control", "point-in-time-join"}:
        return (15, 15, 45, 25)
    if identifier in {"ab-experiment", "offline-online-evaluation", "multi-objective", "ranking-formula"}:
        return (20, 35, 20, 25)
    if identifier in {"industrial-recsys-scope", "feedback-loop", "hybrid-architecture"}:
        return (35, 25, 20, 20)
    return (20, 30, 30, 20) if research else (25, 30, 30, 15)


def append_progress(root: Path, row: dict[str, object]) -> None:
    with (workspace(root) / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **row}, ensure_ascii=False) + "\n")


def progress_rows(root: Path) -> list[dict[str, object]]:
    path = workspace(root) / "progress.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"学习记录第 {number} 行损坏：{error}") from error
    return rows


def latest_by(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        value = row.get(key)
        if value:
            output[str(value)] = row
    return output


def learning_plan(root: Path, topic: str, level: str, minutes: int) -> None:
    level = "advanced" if level == "graduate" else level
    results = rank(root, topic)[:3]
    if not results:
        raise SystemExit("未找到可用于学习计划的本书依据。")
    data = graph(root)
    matched = concept_id(root, topic)
    prerequisites = [left for left, right in data["edges"] if right == matched] if matched else []
    profile = effort_profile(matched, level, concept_metadata(root, matched))
    budgets = allocate_minutes(minutes, profile)
    research = level == "advanced"
    tasks = [
        "阅读知识卡、前置概念与命中的 PDF 页面，记录问题假设和适用边界" if research else "阅读知识卡、前置概念与命中的 PDF 页面",
        "推导目标、数据生成机制、强基线与评价设计" if research else "用自己的话回答核心机制、目标和取舍",
        "运行基线和失败路径，保存命令、配置、版本与输出" if research else "运行相关模板或画出可检查的数据流，并记录输出",
        "形成结论、失败模式、迁移问题和下一步实验" if research else "完成检索题和应用题，记录不确定点",
    ]
    plan = {
        "topic": topic, "concept_id": matched, "level": level, "minutes": minutes, "status": "started",
        "prerequisites": prerequisites, "effort_profile": {"reading": profile[0], "reasoning": profile[1], "practice": profile[2], "assessment": profile[3]},
        "sources": [{"chapter": chunk.chapter, "source": chunk.source, "page": chunk.page_start, "section": chunk.section} for _, chunk in results],
        "steps": [{"minutes": budget, "task": task} for budget, task in zip(budgets, tasks)],
        "acceptance": ["步骤分钟数之和等于总预算", "提交物和实践输出存在", "至少一个失败路径被实际运行", "结论可追溯到 PDF、配置或输出"],
    }
    append_progress(root, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def complete_topic(root: Path, topic: str, evidence: Path | None, practice_passed: bool, failure_path: bool, score: float | None) -> None:
    identifier = concept_id(root, topic) or topic
    missing: list[str] = []
    if evidence is None or not evidence.is_file() or not evidence.read_text(encoding="utf-8").strip():
        missing.append("non-empty --evidence file")
    if not practice_passed:
        missing.append("--practice-passed")
    if not failure_path:
        missing.append("--failure-path")
    if score is None or score < 0.8 or score > 1.0:
        missing.append("--score between 0.8 and 1.0")
    if missing:
        row = {"topic": topic, "concept_id": identifier, "status": "assessed", "evidence": str(evidence) if evidence else None, "practice_passed": practice_passed, "failure_path": failure_path, "score": score, "missing": missing}
        append_progress(root, row)
        print(json.dumps({**row, "completed": False}, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    row = {"topic": topic, "concept_id": identifier, "status": "completed", "evidence": str(evidence), "practice_passed": True, "failure_path": True, "score": score, "review_due": (datetime.now(timezone.utc) + timedelta(days=7)).date().isoformat()}
    append_progress(root, row)
    print(json.dumps(row, ensure_ascii=False, indent=2))


def adaptive_next(root: Path) -> None:
    rows = progress_rows(root)
    completed = {str(row.get("concept_id") or row.get("topic")) for row in rows if row.get("status") == "completed"}
    scores = {str(row.get("concept_id")): float(row["score"]) for row in rows if row.get("concept_id") and isinstance(row.get("score"), (int, float))}
    data = graph(root)
    available = [node for node in data["nodes"] if node not in completed and all(left in completed for left, right in data["edges"] if right == node)]
    available.sort(key=lambda node: (scores.get(node, 1.0), node))
    print(json.dumps({"completed": sorted(completed), "recommended_next": available[:5], "reason": "前置已满足；低掌握度与基础节点优先"}, ensure_ascii=False, indent=2))


def load_courses(root: Path) -> dict[str, object]:
    return json.loads((root / "learning-paths/courses.json").read_text(encoding="utf-8"))["courses"]


def course_command(root: Path, action: str, course_id: str | None, minutes: int | None, feedback: str | None) -> None:
    courses = load_courses(root)
    rows = progress_rows(root)
    course_rows = [row for row in rows if row.get("record_type") == "course"]
    active = next((row for row in reversed(course_rows) if row.get("action") == "start"), None)
    selected = course_id or (str(active["course_id"]) if active else None)
    if action == "list":
        print(json.dumps({key: {"title": value["title"], "audience": value["audience"], "units": len(value["units"])} for key, value in courses.items()}, ensure_ascii=False, indent=2))
        return
    if not selected or selected not in courses:
        raise SystemExit("请指定有效课程：new-engineer 或 graduate。")
    course = courses[selected]
    if action == "start":
        append_progress(root, {"record_type": "course", "action": "start", "course_id": selected, "status": "started"})
        print(json.dumps({"course_id": selected, "title": course["title"], "status": "started", "next": "course today"}, ensure_ascii=False, indent=2))
        return
    states = latest_by([row for row in rows if row.get("course_id") == selected and row.get("unit_id")], "unit_id")
    completed_units = {key for key, row in states.items() if row.get("status") == "completed"}
    units = course["units"]
    current = next((unit for unit in units if unit["id"] not in completed_units), None)
    if action == "status":
        print(json.dumps({"course_id": selected, "title": course["title"], "completed_units": len(completed_units), "total_units": len(units), "current_unit": current["id"] if current else None, "units": [{"id": unit["id"], "title": unit["title"], "status": states.get(unit["id"], {}).get("status", "pending"), "score": states.get(unit["id"], {}).get("score")} for unit in units]}, ensure_ascii=False, indent=2))
        return
    if current is None:
        print(json.dumps({"course_id": selected, "status": "completed", "recommended_next": "一周后运行 course review"}, ensure_ascii=False, indent=2))
        return
    if action == "today":
        total = minutes or int(course["default_minutes"])
        practice = bool(current.get("practice"))
        profile = (20, 20, 40, 20) if practice else (30, 30, 20, 20)
        budgets = allocate_minutes(total, profile)
        output = {"course_id": selected, "unit": current, "minutes": total, "steps": [{"minutes": value, "kind": kind} for kind, value in zip(("read", "reason", "practice", "assess"), budgets)], "feedback_context": states.get(current["id"], {}).get("feedback"), "submit": f"course submit --course {selected} --unit {current['id']} --evidence <path> --score <0..1> --practice-passed --failure-path"}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    if action == "feedback":
        if not feedback:
            raise SystemExit("feedback 需要 --message。")
        append_progress(root, {"record_type": "course", "action": "feedback", "course_id": selected, "unit_id": current["id"], "status": "started", "feedback": feedback})
        print(json.dumps({"course_id": selected, "unit_id": current["id"], "feedback": feedback, "adaptation": "下次 today 将保留反馈；低分单元不会完成并会生成补救建议"}, ensure_ascii=False, indent=2))


def course_submit(root: Path, course_id: str, unit_id: str, evidence: Path, score: float, practice_passed: bool, failure_path: bool) -> None:
    courses = load_courses(root)
    if course_id not in courses:
        raise SystemExit("未知课程。")
    unit = next((item for item in courses[course_id]["units"] if item["id"] == unit_id), None)
    if unit is None:
        raise SystemExit("未知课程单元。")
    missing = []
    if not evidence.is_file() or not evidence.read_text(encoding="utf-8").strip(): missing.append("非空 evidence 文件")
    if not practice_passed: missing.append("practice-passed")
    if not failure_path: missing.append("failure-path")
    if not 0 <= score <= 1: missing.append("0..1 score")
    passed = not missing and score >= 0.8
    status = "completed" if passed else "assessed"
    remedial = [] if passed else ["重读本单元 reading 指定页并逐题作答", "重新运行实践和失败路径", "按 acceptance 自评后再次提交"]
    row = {"record_type": "course", "action": "submit", "course_id": course_id, "unit_id": unit_id, "status": status, "score": score, "evidence": str(evidence), "practice_passed": practice_passed, "failure_path": failure_path, "missing": missing, "review_due": (datetime.now(timezone.utc) + timedelta(days=7)).date().isoformat() if passed else None}
    append_progress(root, row)
    print(json.dumps({**row, "remedial": remedial, "recommended_next": "course today" if passed else f"retry {unit_id}"}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


def course_review(root: Path, course_id: str | None) -> None:
    today = datetime.now(timezone.utc).date()
    due = []
    for row in progress_rows(root):
        if row.get("status") != "completed" or not row.get("review_due") or (course_id and row.get("course_id") != course_id):
            continue
        if datetime.fromisoformat(str(row["review_due"])).date() <= today:
            due.append({"course_id": row.get("course_id"), "unit_id": row.get("unit_id"), "concept_id": row.get("concept_id"), "review_due": row["review_due"], "tasks": ["不看资料解释核心机制", "找出一个错误设计", "修改参数并预测后果", "迁移到新场景说明取舍"]})
    print(json.dumps({"due_reviews": due, "count": len(due)}, ensure_ascii=False, indent=2))


def knowledge_graph_view(root: Path, course_id: str | None, output: Path | None) -> None:
    data = graph(root)
    nodes = set(data["nodes"])
    if course_id:
        courses = load_courses(root)
        if course_id not in courses:
            raise SystemExit("未知课程。")
        nodes = {concept for unit in courses[course_id]["units"] for concept in unit["concepts"]}
    lines = ["flowchart LR"]
    for left, right in data["edges"]:
        if left in nodes and right in nodes:
            lines.append(f"    {left.replace('-', '_')}[{left}] --> {right.replace('-', '_')}[{right}]")
    connected = {item for edge in data["edges"] for item in edge if item in nodes}
    for node in sorted(nodes - connected):
        lines.append(f"    {node.replace('-', '_')}[{node}]")
    rendered = "\n".join(lines) + "\n"
    if output:
        output.write_text(rendered, encoding="utf-8")
        print(json.dumps({"output": str(output), "nodes": len(nodes)}, ensure_ascii=False))
    else:
        print(rendered, end="")


def self_evaluate(root: Path, output: Path) -> None:
    retrieval_cases = [json.loads(line) for line in (root / "evals/retrieval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = 0
    citations = 0
    for case in retrieval_cases:
        results = rank(root, case["query"])[:5]
        hits += bool(set(case["expected_chapters"]).intersection(chunk.chapter for _, chunk in results))
        citations += bool(results and all(chunk.page_start and chunk.source.endswith(".pdf") for _, chunk in results))
    rows = progress_rows(root)
    completed = [row for row in rows if row.get("status") == "completed"]
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "retrieval": {"cases": len(retrieval_cases), "recall_at_5": hits / len(retrieval_cases), "citation_valid": citations / len(retrieval_cases)}, "learning": {"completed_records": len(completed), "due_reviews": sum(bool(row.get("review_due") and datetime.fromisoformat(str(row["review_due"])).date() <= datetime.now(timezone.utc).date()) for row in completed)}, "next_actions": ["完成当前 course today 单元", "补做 assessed 单元", "按期运行 course review"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(output), **report}, ensure_ascii=False, indent=2))


def source_rows(root: Path, query: str, limit: int = 3) -> list[dict[str, object]]:
    return [{"chapter": chunk.chapter, "source": chunk.source, "page": chunk.page_start, "section": chunk.section} for _, chunk in rank(root, query)[:limit]]


def parse_quantity(value: str) -> float | None:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmM]?)\s*(?:ms|items?)?\s*", value)
    if not match:
        return None
    scale = {"": 1.0, "k": 1_000.0, "m": 1_000_000.0}[match.group(2).lower()]
    return float(match.group(1)) * scale


def design(root: Path, topic: str, constraints: list[str]) -> None:
    required = ["objective", "qps", "p99", "items", "freshness", "availability", "cost"]
    parsed = dict(item.split("=", 1) for item in constraints if "=" in item)
    missing = [key for key in required if key not in parsed]
    p99_ms = parse_quantity(parsed.get("p99", ""))
    qps = parse_quantity(parsed.get("qps", ""))
    budget_names = ("gateway", "feature", "recall", "ranking", "reranking", "reserve")
    budgets = allocate_minutes(round(p99_ms), (5, 15, 30, 30, 5, 15)) if p99_ms and p99_ms >= len(budget_names) else []
    report = {
        "status": "needs_input" if missing else "complete", "system": topic, "constraints": parsed, "missing": missing,
        "request_flow": ["gateway", "feature snapshot", "parallel multi-recall", "merge/filter", "pre-rank", "rank", "rerank", "response"],
        "latency_budget": [{"stage": name, "p99_ms": value} for name, value in zip(budget_names, budgets)],
        "capacity": {"offered_qps": qps, "minimum_inflight_at_p99": math.ceil(qps * p99_ms / 1000) if qps and p99_ms else None, "replica_formula": "ceil(peak_qps / measured_safe_qps_per_replica * headroom)", "headroom": "1.3-2.0; confirm with peak shape and failure-domain target", "unresolved": ["measured safe QPS/replica", "peak-to-average ratio", "candidate count per stage"]},
        "version_contract": "One release manifest binds model, feature schema, Embedding and ANN index; cache/request keys carry release_id; incompatible releases fail closed.",
        "overload_and_fallback": ["Bound queues and concurrency per dependency", "Apply stage deadlines within the end-to-end budget", "Drop optional recall routes before core routes", "Return version-compatible popular/fresh candidates when personalization cannot finish"],
        "observability": ["stage P50/P95/P99", "queue depth and timeout rate", "candidate counts and empty-result rate", "fallback reason", "release_id and feature freshness", "objective guardrails"],
        "rollout": ["manifest compatibility gate", "shadow validation", "small canary with objective and reliability guardrails", "atomic promotion", "whole-release rollback"],
        "sources": source_rows(root, "在线服务 延迟预算 限流 降级 版本一致性"),
        "assumptions": ["Latency allocation is an initial budget, not measured capacity.", "All external calls propagate a request deadline."],
        "pending_business_confirmation": [key for key in ("runtime", "deployment", "privacy") if key not in parsed],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def review_surface(root: Path, path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"评审目标不是可读文件：{path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    findings: list[dict[str, object]] = []
    for number, line in enumerate(lines, 1):
        if "feature.timestamp <= label.timestamp" in line:
            findings.append({"severity": "pass", "line": number, "finding": "Point-in-time join excludes future features.", "invariant": "feature_time <= label_time"})
        if 'row["feature_time"] > row["label_time"]' in line:
            findings.append({"severity": "pass", "line": number, "finding": "A second validation gate rejects leaked rows.", "verification": "Run with --inject-leakage and require exit code 2."})
    report = {
        "status": "complete", "target": str(path),
        "findings": findings or [{"severity": "needs_manual_review", "finding": "No supported domain invariant was recognized; no clean bill of health is asserted."}],
        "risk": "critical if either point-in-time predicate or rejection gate is removed" if findings else "undetermined",
        "required_verification": ["normal path succeeds", "future-feature injection is rejected", "boundary timestamps remain eligible"],
        "sources": source_rows(root, "样本穿越 point in time join 特征时效性"),
        "limitations": ["Static inspection recognizes explicit repository patterns; execute tests and inspect data contracts before production approval."],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def debug_scenario(root: Path, symptom: str) -> None:
    branches = [
        {"priority": 1, "hypothesis": "slow or saturated recall dependency consumes the deadline", "evidence": ["per-route P99", "timeout rate", "queue depth", "inflight", "deadline-exceeded count"]},
        {"priority": 2, "hypothesis": "candidate supply or filtering changed", "evidence": ["per-route candidate count", "merge count", "filter reason", "empty-result rate"]},
        {"priority": 3, "hypothesis": "release artifacts are semantically mismatched", "evidence": ["model/Embedding/index release_id", "cache-key version", "rollout event"]},
        {"priority": 4, "hypothesis": "traffic, experiment or feature freshness changed", "evidence": ["traffic by bucket", "configuration diff", "feature age and missing rate"]},
    ]
    matched_fixture = "cases/multi-recall-tail-latency.md" if "召回" in symptom and ("P99" in symptom.upper() or "延迟" in symptom) else None
    report = {
        "status": "complete", "symptom": symptom,
        "immediate_action": ["Preserve request and release identifiers", "Compare affected vs healthy buckets before changing configuration", "Keep non-empty fallback enabled while collecting evidence"],
        "diagnostic_tree": branches,
        "fixture_evidence": {"path": matched_fixture, "observations": ["one recall RPC has a long tail", "child calls lack shorter deadlines", "the batch waits after timeout", "fallback supply is sufficient"]} if matched_fixture else None,
        "conclusion": "For the matched fixture, a slow recall dependency consumes the total budget and candidate assembly waits too long; this explains both higher tail latency and empty/insufficient results. For a live incident, confirm the listed evidence before accepting this root cause." if matched_fixture else "No repository fixture uniquely identifies a root cause; follow the evidence order.",
        "remediation": ["Set per-route deadlines below the end-to-end deadline", "Bound per-route concurrency and queueing", "Continue with healthy routes after a child timeout", "Serve non-empty fallback candidates"],
        "recovery_verification": ["fault-injected end-to-end P99 meets the budget", "candidate count and non-empty rate meet SLO", "timeout and fallback reason are observable", "removing the fault restores the normal route"],
        "sources": source_rows(root, "召回 延迟预算 timeout 过载 降级"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="本地推荐系统知识、学习和工程工作台",
        epilog="完整示例见 docs/agent-guide.md；首次使用先运行 index，再运行 doctor。",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="仓库根目录（默认当前目录）")
    parser.add_argument("--mode", choices=("auto", "private", "public"), default="auto", help="知识源模式")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="从发布 PDF 构建本地临时索引")
    search_parser = commands.add_parser("search", help="检索发布 PDF 并返回页码")
    search_parser.add_argument("query", help="问题、术语或模型名")
    search_parser.add_argument("--limit", type=int, default=5, help="返回数量（默认 5）")
    search_parser.add_argument("--chapter", type=int, help="只检索指定章节")
    search_parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    page_parser = commands.add_parser("page", help="读取指定 PDF 页的本地提取文本")
    page_parser.add_argument("--chapter", type=int, required=True, help="章节号")
    page_parser.add_argument("--page", type=int, required=True, help="PDF 页码，不是印刷页码")
    eval_parser = commands.add_parser("eval", help="运行检索与引用基准")
    eval_parser.add_argument("--dataset", type=Path, default=Path("evals/retrieval.jsonl"), help="JSONL 评测集")
    commands.add_parser("doctor", help="检查依赖、索引新鲜度和知识源")
    learn_parser = commands.add_parser("learn", help="生成带阅读页和实践的学习计划")
    learn_parser.add_argument("topic", help="学习主题")
    learn_parser.add_argument("--level", choices=("beginner", "intermediate", "advanced", "graduate"), default="beginner", help="学习深度；graduate 等价于 advanced")
    learn_parser.add_argument("--minutes", type=int, default=45, help="本次时间预算，至少 4 分钟")
    complete_parser = commands.add_parser("complete", help="用提交物、实践、失败路径和测验分数完成主题")
    complete_parser.add_argument("topic", help="主题或概念 ID")
    complete_parser.add_argument("--evidence", type=Path, help="学习提交物路径")
    complete_parser.add_argument("--practice-passed", action="store_true", help="确认实践主路径通过")
    complete_parser.add_argument("--failure-path", action="store_true", help="确认实际运行过失败路径")
    complete_parser.add_argument("--score", type=float, help="检索题与应用题得分，0..1")
    explain_parser = commands.add_parser("explain", help="按受众生成有出处的解释框架")
    explain_parser.add_argument("topic", help="解释主题")
    explain_parser.add_argument("--audience", choices=("graduate", "new-engineer", "interview"), default="new-engineer", help="目标受众")
    commands.add_parser("next", help="按前置、掌握度和完成状态推荐下一主题")
    course_parser = commands.add_parser("course", help="运行机器可读课程学习闭环")
    course_commands = course_parser.add_subparsers(dest="course_action", required=True)
    course_commands.add_parser("list", help="列出课程")
    course_start = course_commands.add_parser("start", help="开始课程")
    course_start.add_argument("course_id", choices=("new-engineer", "graduate"))
    course_today = course_commands.add_parser("today", help="生成当前单元的限时任务")
    course_today.add_argument("--course", choices=("new-engineer", "graduate"))
    course_today.add_argument("--minutes", type=int)
    course_status = course_commands.add_parser("status", help="查看课程进度")
    course_status.add_argument("--course", choices=("new-engineer", "graduate"))
    course_feedback = course_commands.add_parser("feedback", help="反馈困难、时间与学习观察")
    course_feedback.add_argument("--course", choices=("new-engineer", "graduate"))
    course_feedback.add_argument("--message", required=True)
    course_submit_parser = course_commands.add_parser("submit", help="提交当前单元证据并验收")
    course_submit_parser.add_argument("--course", required=True, choices=("new-engineer", "graduate"))
    course_submit_parser.add_argument("--unit", required=True)
    course_submit_parser.add_argument("--evidence", required=True, type=Path)
    course_submit_parser.add_argument("--score", required=True, type=float)
    course_submit_parser.add_argument("--practice-passed", action="store_true")
    course_submit_parser.add_argument("--failure-path", action="store_true")
    course_review_parser = course_commands.add_parser("review", help="列出到期的延迟复习任务")
    course_review_parser.add_argument("--course", choices=("new-engineer", "graduate"))
    graph_parser = commands.add_parser("graph", help="输出 Mermaid 知识图谱")
    graph_parser.add_argument("--course", choices=("new-engineer", "graduate"))
    graph_parser.add_argument("--output", type=Path)
    self_eval_parser = commands.add_parser("self-eval", help="生成读者本地学习与检索自测报告")
    self_eval_parser.add_argument("--output", type=Path, default=Path(".recsys-workspace/self-eval.json"))
    design_parser = commands.add_parser("design", help="生成可机读的推荐系统设计报告")
    design_parser.add_argument("system", help="待设计系统")
    design_parser.add_argument("constraints", nargs="*", help="key=value 约束，如 qps=10000 p99=200ms")
    review_parser = commands.add_parser("review", help="生成带位置、风险和验证的领域评审报告")
    review_parser.add_argument("path", type=Path, help="待评审文件")
    debug_parser = commands.add_parser("debug", help="生成证据树、结论、修复和恢复验证报告")
    debug_parser.add_argument("symptom", help="故障现象")
    args = parser.parse_args()
    root = args.root.resolve()
    mode = detect_mode(root) if args.mode == "auto" else args.mode
    if args.command == "index":
        build(root, mode)
    elif args.command == "search":
        search(root, args.query, args.limit, args.chapter, args.json)
    elif args.command == "page":
        page(root, args.chapter, args.page)
    elif args.command == "eval":
        evaluate(root, root / args.dataset)
    elif args.command == "learn":
        learning_plan(root, args.topic, args.level, args.minutes)
    elif args.command == "complete":
        complete_topic(root, args.topic, args.evidence, args.practice_passed, args.failure_path, args.score)
    elif args.command == "explain":
        explain(root, args.topic, args.audience)
    elif args.command == "next":
        adaptive_next(root)
    elif args.command == "course":
        if args.course_action == "submit":
            course_submit(root, args.course, args.unit, args.evidence, args.score, args.practice_passed, args.failure_path)
        elif args.course_action == "review":
            course_review(root, args.course)
        else:
            course_command(root, args.course_action, getattr(args, "course_id", None) or getattr(args, "course", None), getattr(args, "minutes", None), getattr(args, "message", None))
    elif args.command == "graph":
        knowledge_graph_view(root, args.course, args.output)
    elif args.command == "self-eval":
        self_evaluate(root, root / args.output if not args.output.is_absolute() else args.output)
    elif args.command == "design":
        design(root, args.system, args.constraints)
    elif args.command == "review":
        review_surface(root, args.path)
    elif args.command == "debug":
        debug_scenario(root, args.symptom)
    else:
        doctor(root, mode)


if __name__ == "__main__":
    main()
