#!/usr/bin/env python3
"""Build, inspect, evaluate, and search the local recommendation-book index."""

from __future__ import annotations

import argparse
import ast
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
FORMAT_VERSION = 3
EVIDENCE_SCORE_FLOOR = 8.0  # Conservative heuristic; a score alone does not prove relevance.
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


def alias_spans(alias: str, query: str) -> list[tuple[int, int]]:
    """Find literal term occurrences; Latin abbreviations must have token boundaries."""
    lowered = alias.casefold()
    pattern = re.escape(lowered)
    if re.fullmatch(r"[a-z0-9_+.#/-]+", lowered):
        pattern = rf"(?<![a-z0-9_/-]){pattern}(?![a-z0-9_/-])"
    return [match.span() for match in re.finditer(pattern, query.casefold())]


def matched_terms(families: list[tuple[str, str, list[str]]], query: str) -> list[tuple[str, str]]:
    """Recognize longest non-overlapping occurrences across alias and concept families."""
    candidates = [(start, end, kind, family)
                  for kind, family, terms_in_family in families
                  for term in terms_in_family
                  for start, end in alias_spans(term, query)]
    occupied: list[tuple[int, int]] = []
    matched: set[tuple[str, str]] = set()
    for start, end, kind, family in sorted(candidates, key=lambda row: (-(row[1] - row[0]), row[0], row[2] != "concept", row[3])):
        if any(start < right and left < end for left, right in occupied):
            continue
        occupied.append((start, end))
        matched.add((kind, family))
    return [(kind, family) for kind, family, _ in families if (kind, family) in matched]


def matched_alias_families(aliases: dict[str, list[str]], query: str) -> list[str]:
    """Longest non-overlapping aliases win; separate occurrences express both intents."""
    return [family for _, family in matched_terms(
        [("alias", family, [family, *values]) for family, values in aliases.items()], query)]


def alias_in_query(alias: str, query: str) -> bool:
    return bool(alias_spans(alias, query))


def expand_query(root: Path, query: str) -> str:
    aliases = load_aliases(root)
    additions = [term for family in matched_alias_families(aliases, query)
                 for term in [family, *aliases[family]]]
    return " ".join(dict.fromkeys([query, *additions]))

def concept_chapter_hints(root: Path, query: str) -> set[int]:
    aliases = load_aliases(root)
    families = [("alias", family, [family, *values]) for family, values in aliases.items()]
    chapters: dict[str, int] = {}
    for path in (root / "knowledge/concepts").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        chapter = re.search(r"^chapter:\s*(\d+)\s*$", text, re.MULTILINE)
        if not title or not chapter:
            continue
        families.append(("concept", path.stem, [title.group(1).strip(), path.stem.replace("-", " ")]))
        chapters[path.stem] = int(chapter.group(1))
    recognized = matched_terms(families, query)
    named = {chapters[family] for kind, family in recognized if kind == "concept"}
    if named:
        return named
    canonical: dict[str, list[str]] = {}
    for kind, family, terms_in_family in families:
        if kind == "concept":
            canonical.setdefault(terms_in_family[0], []).append(family)
    return {chapters[canonical[family][0]] for kind, family in recognized
            if kind == "alias" and len(canonical.get(family, [])) == 1}


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


def requests_unreleased_pdf(root: Path, query: str) -> bool:
    """Reject a request for a specific unpublished chapter's PDF, not its topic."""
    digits = {character: number for number, character in enumerate("零一二三四五六七八九")}

    def chapter_number_text(value: str) -> int | None:
        if value.isdecimal():
            return int(value)
        if "十" in value:
            before, after = value.split("十", 1)
            if (before and before not in digits) or (after and after not in digits):
                return None
            return (digits[before] if before else 1) * 10 + (digits[after] if after else 0)
        return digits.get(value)

    if not re.search(r"(?:已发布|正式出版|完整原书|PDF|页码)", query, re.I) or re.search(r"(?:尚未|未发布)", query):
        return False
    mentioned = {n for match in re.finditer(r"第([零一二三四五六七八九十\d]+)章", query)
                 if (n := chapter_number_text(match.group(1))) is not None}
    released = {chapter_number(path) for path in public_pdfs(root)}
    return bool(mentioned and not mentioned.issubset(released))


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
    index_path = root / INDEX_FILE
    with index_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    manifest = {
        "format": FORMAT_VERSION,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": len(chunks),
        "index_sha256": sha256(index_path),
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


def index_health_reasons(root: Path, mode: str) -> list[str]:
    reasons = []
    if not sources(root, mode):
        reasons.append(f"{mode} 模式没有可用知识源")
    reasons.extend(stale_reasons(root, mode))
    if (root / MANIFEST_FILE).is_file():
        manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest.get("mode") != mode:
            reasons.append(f"mode {manifest.get('mode')} != {mode}")
    return reasons


def require_fresh_index(root: Path, mode: str) -> None:
    if not sources(root, mode):
        raise SystemExit(f"{mode} 模式没有可用知识源；请检查 --mode 和仓库路径。")
    try:
        reasons = index_health_reasons(root, mode)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"索引元数据不可用：{error}；请重新执行 index。") from error
    if reasons:
        raise SystemExit("索引已过期或模式不匹配，请执行 `python3 tools/recsys_agent.py index`。\n" + "\n".join(reasons[:5]))


def stale_reasons(root: Path, mode: str) -> list[str]:
    if not (root / MANIFEST_FILE).is_file() or not (root / INDEX_FILE).is_file():
        return ["index missing"]
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        return [f"format {manifest.get('format')} != {FORMAT_VERSION}"]
    if not isinstance(manifest.get("index_sha256"), str) or sha256(root / INDEX_FILE) != manifest["index_sha256"]:
        return ["index content changed; rebuild index"]
    if not isinstance(manifest.get("chunks"), int) or manifest["chunks"] <= 0:
        return ["index chunk count invalid"]
    with (root / INDEX_FILE).open(encoding="utf-8") as handle:
        try:
            indexed = [Chunk(**json.loads(line)) for line in handle if line.strip()]
        except (json.JSONDecodeError, TypeError, KeyError) as error:
            return [f"index content malformed: {error}"]
    if len(indexed) != manifest["chunks"]:
        return ["index chunk count changed; rebuild index"]
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
    # A bare concept title is a navigational query. Prefer its curated, released
    # PDF pointer while keeping longer contextual questions on the normal ranker.
    exact_pointers: set[tuple[str, int]] = set()
    if len(query.strip()) <= 32 and (root / "knowledge/concepts").is_dir():
        from check_knowledge import citation_pointers
        for card in (root / "knowledge/concepts").glob("*.md"):
            content = card.read_text(encoding="utf-8")
            title = re.search(r"(?m)^title:\s*(.+)$", content)
            if not title or query.strip().casefold() != title.group(1).strip().casefold():
                continue
            source = re.search(r"(?ms)^## 本书依据\s*\n(.*?)(?=^## |\Z)", content)
            for pdf_name, start, end in citation_pointers(source.group(1)) if source else []:
                exact_pointers.update((pdf_name, page) for page in range(start, end + 1))
    expanded = expand_query(root, query)
    chapter_hints = concept_chapter_hints(root, query)
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
        pointer_prior = 20.0 if (Path(chunk.source).name, chunk.page_start) in exact_pointers else 0.0
        score = bm25 + semantic * 3.0 + exact + heading + chapter_prior + pointer_prior
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


def evaluate(root: Path, dataset: Path, min_hit_at_1: float = 1.0) -> None:
    if not math.isfinite(min_hit_at_1) or not 0 <= min_hit_at_1 <= 1:
        raise SystemExit("--min-hit-at-1 必须是 0 到 1 之间的有限数值。")
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise SystemExit(f"评测集为空：{dataset}")
    hits_at_1 = hits_at_5 = citation_valid = page_hits = page_total = 0
    failures: list[dict[str, object]] = []
    page_misses: list[dict[str, object]] = []
    top1_misses: list[dict[str, object]] = []
    negative_total = negative_hits = 0
    for case in cases:
        results = [] if requests_unreleased_pdf(root, case["query"]) else rank(root, case["query"])[:5]
        expected = set(case["expected_chapters"])
        if not expected:
            negative_total += 1
            if not results or results[0][0] < EVIDENCE_SCORE_FLOOR:
                negative_hits += 1
            else:
                failures.append({"query": case["query"], "expected": [], "actual": results[0][1].chapter})
            continue
        chapters = [chunk.chapter for _, chunk in results]
        first_hit = bool(chapters and chapters[0] in expected)
        hits_at_1 += first_hit
        hits_at_5 += bool(expected.intersection(chapters))
        citation_valid += bool(results and all(chunk.page_start and chunk.source.endswith(".pdf") for _, chunk in results))
        if "expected_pages" in case:
            page_total += 1
            expected_pages = case["expected_pages"]
            page_hit = bool(results and any(
                chunk.chapter == int(chapter) and int(start) <= chunk.page_start <= int(end)
                for chapter, ranges in expected_pages.items()
                for start, end in ranges
                for _, chunk in results[:1]
            ))
            page_hits += page_hit
            if not page_hit:
                page_misses.append({"query": case["query"], "expected_pages": expected_pages,
                                    "actual": [results[0][1].chapter, results[0][1].page_start] if results else None})
        if not first_hit:
            top1_misses.append({"query": case["query"], "expected": sorted(expected), "actual": chapters[0] if chapters else None})
        if not expected.intersection(chapters):
            failures.append({"query": case["query"], "expected": sorted(expected), "actual": chapters})
    total = len(cases) - negative_total
    report = {"cases": len(cases), "negative_cases": negative_total, "negative_rejection_rate": negative_hits / negative_total if negative_total else None, "hit_at_1": hits_at_1 / total if total else None, "min_hit_at_1": min_hit_at_1,
              "recall_at_5": hits_at_5 / total if total else None, "citation_valid": citation_valid / total if total else None,
              "top1_misses": top1_misses, "failures": failures,
              "page_cases": page_total, "page_hit_at_1": page_hits / page_total if page_total else None,
              "page_misses": page_misses}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (total and hits_at_1 / total < min_hit_at_1) or failures or citation_valid != total or page_misses:
        raise SystemExit(1)


def doctor(root: Path, mode: str) -> None:
    try:
        reasons = index_health_reasons(root, mode)
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons = [f"索引元数据不可用：{error}"]
    print(f"mode={mode}")
    print(f"root={root}")
    print(f"pdftotext={shutil.which('pdftotext') or 'missing'}")
    print(f"index={'stale' if reasons else 'ready'}")
    for reason in reasons:
        print(f"  {reason}")
    print(f"{'tex_sources' if mode == 'private' else 'pdf_sources'}={len(sources(root, mode))}")
    print(f"model_claude={'available' if shutil.which('claude') else 'missing'}")
    print(f"model_codex={'available' if shutil.which('codex') else 'missing'}")
    print("ask_and_coach=requires_model_cli; search_and_navigation=local_index")

def workspace(root: Path) -> Path:
    path = root / ".recsys-workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path


def graph(root: Path) -> dict[str, object]:
    return json.loads((root / "knowledge/graph.json").read_text(encoding="utf-8"))

def concept_id(root: Path, topic: str) -> str | None:
    normalized = re.sub(r"[-_\s]+", " ", topic.casefold()).strip()
    cards = list((root / "knowledge/concepts").glob("*.md"))
    for path in cards:
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        if normalized in {path.stem.replace("-", " "), title.group(1).casefold() if title else ""}:
            return path.stem
    model_aliases = {
        "multi-task-experts": ("mmoe", "ple", "cgc", "多任务专家"),
        "feature-crossing-deep-ranking": ("deepfm", "dlrm", "wide & deep", "特征交叉"),
        "listwise-ranking-loss": ("listwise", "listnet", "列表损失"),
    }
    for identifier, aliases in model_aliases.items():
        if (root / "knowledge/concepts" / f"{identifier}.md").is_file() and any(alias_in_query(alias, topic) for alias in aliases):
            return identifier
    matches: list[tuple[int, str]] = []
    for path in cards:
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        names = (path.stem.replace("-", " "), title.group(1).casefold() if title else "")
        score = max((len(name) for name in names if name and (name in normalized or (len(normalized) >= 2 and normalized in name))), default=0)
        if score:
            matches.append((score, path.stem))
    if matches:
        return max(matches)[1]
    expanded = re.sub(r"[-_\s]+", " ", expand_query(root, topic).casefold())
    for path in cards:
        title = re.search(r"^title:\s*(.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
        names = (path.stem.replace("-", " "), title.group(1).casefold() if title else "")
        score = max((len(name) for name in names if name and name in expanded), default=0)
        if score:
            matches.append((score, path.stem))
    return max(matches)[1] if matches else None


def explicitly_named_concept(root: Path, question: str) -> str | None:
    """Only literal card names may reset a conversation; retrieval expansion must not."""
    normalized = re.sub(r"[-_\s]+", " ", question.casefold())
    matches = []
    for path in (root / "knowledge/concepts").glob("*.md"):
        title = re.search(r"(?m)^title:\s*(.+)$", path.read_text(encoding="utf-8"))
        for name in (path.stem.replace("-", " "), title.group(1).casefold() if title else ""):
            if len(name) >= 3 and name in normalized:
                matches.append((len(name), path.stem))
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


def evidence_gaps(evidence: Path | None, kind: str = "experiment") -> list[str]:
    if evidence is None or not evidence.is_file():
        return ["非空 evidence 文件"]
    text = evidence.read_text(encoding="utf-8").strip()
    if not text:
        return ["非空 evidence 文件"]
    sections = {name: body for name, body in re.findall(r"(?ms)^##\s+([^\n]+)\n(.*?)(?=^##\s+|\Z)", text)}
    required = ("本书依据", "我的解释", "问题答案") + (("实践",) if kind == "experiment" else ("方案验证",))
    missing = [f"提交物章节：{name}" for name in required if not sections.get(name, "").strip()]
    if sections.get("本书依据") and not re.search(r"PDF\s*第\s*\d+\s*页", sections["本书依据"]):
        missing.append("PDF 第 N 页引用")
    if kind == "experiment":
        practice = sections.get("实践", "")
        for label in ("主路径命令", "主路径退出码", "主路径输出", "失败路径命令", "失败路径退出码", "失败路径输出", "失败原因", "恢复条件"):
            if not re.search(rf"(?m)^[-*]?\s*{label}\s*[：:]\s*\S+", practice):
                missing.append(label)
    elif sections.get("方案验证"):
        for label in ("检验方法", "反例或失败条件", "验收依据"):
            if not re.search(rf"(?m)^[-*]?\s*{label}\s*[：:]\s*\S+", sections["方案验证"]):
                missing.append(label)
    return missing


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
        "运行相关章节 code/ 示例或模板的基线和失败路径，保存命令、配置、版本与输出" if research else "选读相关章节 code/ 示例或运行模板，记录输入、输出与失败条件",
        "形成结论、失败模式、迁移问题和下一步实验" if research else "完成检索题和应用题，记录不确定点",
    ]
    import recsys_code

    code_examples = recsys_code.search(root, topic, limit=3)
    plan = {
        "topic": topic, "concept_id": matched, "level": level, "minutes": minutes, "status": "started",
        "prerequisites": prerequisites, "effort_profile": {"reading": profile[0], "reasoning": profile[1], "practice": profile[2], "assessment": profile[3]},
        "sources": [{"chapter": chunk.chapter, "source": chunk.source, "page": chunk.page_start, "section": chunk.section} for _, chunk in results],
        "code_examples": code_examples,
        "code_note": "代码是可选的教学参考，不是 PDF 依据；运行前先检查对应 README 的依赖和耗时。",
        "steps": [{"minutes": budget, "task": task} for budget, task in zip(budgets, tasks)],
        "acceptance": ["步骤分钟数之和等于总预算", "提交物和实践输出存在", "至少一个失败路径被实际运行", "结论可追溯到 PDF、配置或输出"],
    }
    append_progress(root, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def complete_topic(root: Path, topic: str, evidence: Path | None, practice_passed: bool, failure_path: bool, score: float | None) -> None:
    identifier = concept_id(root, topic) or topic
    missing = evidence_gaps(evidence)
    if not practice_passed:
        missing.append("--practice-passed")
    if not failure_path:
        missing.append("--failure-path")
    if score is None or not math.isfinite(score) or score < 0.8 or score > 1.0:
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
    completed = {str(row.get("concept_id") or row.get("topic")) for row in rows if row.get("status") == "completed" and (row.get("concept_id") or row.get("topic"))}
    scores = {str(row.get("concept_id")): float(row["score"]) for row in rows if row.get("concept_id") and isinstance(row.get("score"), (int, float))}
    data = graph(root)
    available = [node for node in data["nodes"] if node not in completed and all(left in completed for left, right in data["edges"] if right == node)]
    available.sort(key=lambda node: (scores.get(node, 1.0), node))
    print(json.dumps({"completed": sorted(completed), "recommended_next": available[:5], "reason": "前置已满足；低掌握度与基础节点优先"}, ensure_ascii=False, indent=2))


def course_unit_kind(unit: dict[str, object]) -> str:
    # Legacy course definitions without concept metadata retain the original gate.
    return "experiment" if unit.get("practice") or "concepts" not in unit else "design"


def course_checkpoints(unit: dict[str, object], feedback: str = "", root: Path | None = None) -> list[dict[str, object]]:
    concepts = list(unit.get("concepts", []))
    # Preserve the published unit ID and order; checkpoints are a reading aid, not new completion records.
    groups = [concepts[i:i + 5] for i in range(0, len(concepts), 5)] or [[]]
    rows = [{"id": f"{unit['id']}-{i + 1}", "concepts": group,
             "task": "核对本组概念对应的阅读页，回答课程问题并记录一项取舍或反例。"}
            for i, group in enumerate(groups)]
    if feedback:
        focus = []
        if root:
            for concept in concepts:
                card = root / "knowledge/concepts" / f"{concept}.md"
                if card.is_file():
                    title = re.search(r"(?m)^title:\s*(.+)$", card.read_text(encoding="utf-8"))
                    if title and title.group(1).casefold() in feedback.casefold(): focus.append(concept)
        focus.extend(concept for concept in concepts if concept in feedback.casefold())
        if focus:
            for row in rows:
                if set(row["concepts"]) & set(focus):
                    row["task"] = "根据反馈重点复核这些概念，再回答课程问题并记录一项反例。"
        else:
            rows[0]["task"] = "先对照本次反馈定位困惑，再阅读本组概念并完成课程问题。"
    return rows


def course_feedback_context(rows: list[dict[str, object]], course_id: str, unit_id: str) -> str | None:
    for row in reversed(rows):
        if row.get("course_id") == course_id and row.get("unit_id") == unit_id and row.get("action") == "feedback":
            return str(row.get("feedback") or "") or None
    return None


def course_checkpoint(root: Path, course_id: str, checkpoint_id: str) -> None:
    courses = load_courses(root)
    if course_id not in courses:
        raise SystemExit("未知课程。")
    rows = progress_rows(root)
    completed = {row.get("unit_id") for row in rows if row.get("course_id") == course_id and row.get("action") == "submit" and row.get("status") == "completed"}
    unit = next((item for item in courses[course_id]["units"] if item["id"] not in completed), None)
    if unit is None:
        raise SystemExit("课程已完成。")
    feedback = course_feedback_context(rows, course_id, unit["id"]) or ""
    checkpoints = course_checkpoints(unit, feedback, root)
    done = {row.get("checkpoint_id") for row in rows if row.get("course_id") == course_id and row.get("unit_id") == unit["id"] and row.get("action") == "checkpoint"}
    remaining = next((item for item in checkpoints if item["id"] not in done), None)
    if remaining is None or checkpoint_id != remaining["id"]:
        raise SystemExit(f"仅可记录当前检查点：{remaining['id'] if remaining else '全部已记录'}。")
    append_progress(root, {"record_type": "course", "action": "checkpoint", "course_id": course_id,
                           "unit_id": unit["id"], "checkpoint_id": checkpoint_id, "status": "visited"})
    print(json.dumps({"course_id": course_id, "unit_id": unit["id"], "checkpoint_id": checkpoint_id,
                      "status": "visited", "note": "记录阅读位置，不代表单元通过或概念已掌握。"}, ensure_ascii=False, indent=2))


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
    states = latest_by([row for row in rows if row.get("course_id") == selected and row.get("unit_id") and row.get("action") == "submit"], "unit_id")
    completed_units = {str(row["unit_id"]) for row in rows if row.get("course_id") == selected and row.get("action") == "submit" and row.get("status") == "completed" and row.get("unit_id")}
    units = course["units"]
    current = next((unit for unit in units if unit["id"] not in completed_units), None)
    if action == "status":
        print(json.dumps({"course_id": selected, "title": course["title"], "completed_units": len(completed_units), "total_units": len(units), "current_unit": current["id"] if current else None, "units": [{"id": unit["id"], "title": unit["title"], "status": "completed" if unit["id"] in completed_units else states.get(unit["id"], {}).get("status", "pending"), "score": states.get(unit["id"], {}).get("score")} for unit in units]}, ensure_ascii=False, indent=2))
        return
    if current is None:
        print(json.dumps({"course_id": selected, "status": "completed", "recommended_next": "一周后运行 course review"}, ensure_ascii=False, indent=2))
        return
    if action == "today":
        total = minutes or int(course["default_minutes"])
        practice = bool(current.get("practice"))
        profile = (20, 20, 40, 20) if practice else (30, 30, 20, 20)
        budgets = allocate_minutes(total, profile)
        import recsys_code

        code_examples = []
        for chapter in current.get("chapters", []):
            for concept in current.get("concepts", []):
                card = root / "knowledge/concepts" / f"{concept}.md"
                if not card.is_file():
                    continue
                title = re.search(r"(?m)^title:\s*(.+)$", card.read_text(encoding="utf-8"))
                if title:
                    code_examples.extend(recsys_code.search(root, title.group(1), chapter=int(chapter), limit=2))
        code_examples = list({item["path"]: item for item in code_examples}.values())[:6]
        kind = course_unit_kind(current)
        feedback_context = course_feedback_context(rows, selected, current["id"])
        checkpoints = course_checkpoints(current, feedback_context or "", root)
        visited = {row.get("checkpoint_id") for row in rows if row.get("course_id") == selected and row.get("unit_id") == current["id"] and row.get("action") == "checkpoint"}
        current_checkpoint = next((item["id"] for item in checkpoints if item["id"] not in visited), None)
        phase = ("read", "reason", "practice", "assess") if kind == "experiment" else ("read", "reason", "verify", "assess")
        submit = f"course submit --course {selected} --unit {current['id']} --evidence <path> --score <0..1>"
        if kind == "experiment": submit += " --practice-passed --failure-path"
        output = {"course_id": selected, "unit": current, "minutes": total, "steps": [{"minutes": value, "kind": step} for step, value in zip(phase, budgets)], "checkpoints": checkpoints, "current_checkpoint": current_checkpoint, "checkpoint_note": "按顺序阅读；用 course checkpoint --course ID --id 检查点ID 记录位置，最终仍由原单元验收。", "evidence_kind": kind, "evidence_sections": ["本书依据", "我的解释", "实践" if kind == "experiment" else "方案验证", "问题答案"], "code_examples": code_examples, "code_note": "章节代码仅为可选参考；需先读 README 和源码，运行前检查依赖及耗时，不能充当 PDF 依据。", "feedback_context": feedback_context, "adaptation_note": "反馈供学习者调整阅读重点；时间预算以 --minutes 为准。" if feedback_context else None, "submit": submit}
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
    completed_units = {str(row["unit_id"]) for row in progress_rows(root) if row.get("course_id") == course_id and row.get("action") == "submit" and row.get("status") == "completed" and row.get("unit_id")}
    current = next((item for item in courses[course_id]["units"] if item["id"] not in completed_units), None)
    if current is None or current["id"] != unit_id:
        raise SystemExit(f"仅可提交当前课程单元：{current['id'] if current else '课程已完成'}。")
    kind = course_unit_kind(unit)
    missing = evidence_gaps(evidence, kind=kind)
    if kind == "experiment":
        if not practice_passed: missing.append("practice-passed")
        if not failure_path: missing.append("failure-path")
    if not math.isfinite(score) or not 0 <= score <= 1: missing.append("0..1 score")
    passed = not missing and score >= 0.8
    status = "completed" if passed else "assessed"
    remedial = [] if passed else ["重读本单元 reading 指定页并逐题作答", "补充实践运行记录" if kind == "experiment" else "补充方案检验、反例与验收依据", "按 acceptance 自评后再次提交"]
    row = {"record_type": "course", "action": "submit", "course_id": course_id, "unit_id": unit_id, "status": status, "score": score, "evidence_kind": kind, "evidence": str(evidence), "practice_passed": practice_passed if kind == "experiment" else None, "failure_path": failure_path if kind == "experiment" else None, "missing": missing, "review_due": (datetime.now(timezone.utc) + timedelta(days=7)).date().isoformat() if passed else None}
    append_progress(root, row)
    print(json.dumps({**row, "remedial": remedial, "recommended_next": "course today" if passed else f"retry {unit_id}"}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


def due_reviews(root: Path, course_id: str | None = None) -> list[dict[str, object]]:
    rows = progress_rows(root)
    latest = latest_by([row for row in rows if row.get("action") == "review_submit"], "review_key")
    today = datetime.now(timezone.utc).date()
    due = []
    seen: set[str] = set()
    for row in rows:
        if row.get("action") == "review_submit" or row.get("status") != "completed" or not row.get("review_due") or (course_id and row.get("course_id") != course_id):
            continue
        key = f"{row.get('course_id') or 'topic'}:{row.get('unit_id') or row.get('concept_id')}:{row['review_due']}"
        if key in seen or latest.get(key, {}).get("status") == "completed":
            continue
        if datetime.fromisoformat(str(row["review_due"])).date() <= today:
            due.append({"review_key": key, "course_id": row.get("course_id"), "unit_id": row.get("unit_id"), "concept_id": row.get("concept_id"), "review_due": row["review_due"], "tasks": ["不看资料解释核心机制", "找出一个错误设计", "修改参数并预测后果", "迁移到新场景说明取舍"]})
            seen.add(key)
    return due


def course_review(root: Path, course_id: str | None, unit_id: str | None = None, evidence: Path | None = None, score: float | None = None) -> None:
    due = due_reviews(root, course_id)
    if unit_id is not None:
        if course_id not in load_courses(root) or not any(unit["id"] == unit_id for unit in load_courses(root)[course_id]["units"]):
            raise SystemExit("未知课程或单元。")
        matches = [item for item in due if item["unit_id"] == unit_id and item["course_id"] == course_id]
        if len(matches) != 1:
            raise SystemExit("没有该课程单元的到期复习任务。")
        item = matches[0]
        content = evidence.read_text(encoding="utf-8") if evidence and evidence.is_file() else ""
        missing = [name for name in ("机制回忆", "错误设计", "参数预测", "迁移取舍") if not re.search(rf"(?m)^##\s*{name}\s*\n\S", content)]
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            missing.append("0..1 score")
        passed = not missing and score is not None and score >= 0.8
        result = {"record_type": "course", "action": "review_submit", "review_key": item["review_key"], "course_id": course_id, "unit_id": unit_id, "status": "completed" if passed else "assessed", "score": score, "evidence": str(evidence) if evidence else None, "missing": missing}
        append_progress(root, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not passed:
            raise SystemExit(2)
        return
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
    if not retrieval_cases:
        raise SystemExit("评测集为空，无法生成自测报告。")
    hits = 0
    citations = 0
    for case in retrieval_cases:
        results = rank(root, case["query"])[:5]
        hits += bool(set(case["expected_chapters"]).intersection(chunk.chapter for _, chunk in results))
        citations += bool(results and all(chunk.page_start and chunk.source.endswith(".pdf") for _, chunk in results))
    rows = progress_rows(root)
    completed = [row for row in rows if row.get("status") == "completed" and row.get("action") != "review_submit"]
    completed_ids = {(row.get("course_id"), row.get("unit_id")) if row.get("unit_id") else ("topic", row.get("concept_id") or row.get("topic")) for row in completed if row.get("unit_id") or row.get("concept_id") or row.get("topic")}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "retrieval": {"cases": len(retrieval_cases), "recall_at_5": hits / len(retrieval_cases), "citation_valid": citations / len(retrieval_cases)}, "learning": {"completed_records": len(completed_ids), "due_reviews": len(due_reviews(root))}, "next_actions": ["完成当前 course today 单元", "补做 assessed 单元", "按期运行 course review"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(output), **report}, ensure_ascii=False, indent=2))


def learning_overview(root: Path) -> None:
    rows = progress_rows(root)
    courses = load_courses(root)
    result = {}
    for course_id, course in courses.items():
        relevant = [row for row in rows if row.get("record_type") == "course" and row.get("course_id") == course_id]
        completed = {row.get("unit_id") for row in relevant if row.get("action") == "submit" and row.get("status") == "completed"}
        current = next((unit for unit in course["units"] if unit["id"] not in completed), None)
        result[course_id] = {"completed_units": len(completed), "total_units": len(course["units"]),
                             "current_unit": current["id"] if current else None,
                             "feedback": course_feedback_context(relevant, course_id, current["id"]) if current else None,
                             "due_reviews": [row for row in due_reviews(root, course_id)],
                             "note": "课程完成不等于每个关联概念均已掌握。"}
    topics = latest_by([row for row in rows if row.get("concept_id") and row.get("record_type") != "course"], "concept_id")
    coach_path = workspace(root) / "coach.jsonl"
    coach = []
    if coach_path.is_file():
        for number, line in enumerate(coach_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                if line.strip(): coach.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"陪练记录第 {number} 行损坏：{error}") from error
    print(json.dumps({"courses": result, "free_topics": {key: {"status": row.get("status"), "score": row.get("score")} for key, row in topics.items()},
                      "coach_follow_up": [{"course_id": row.get("course_id"), "unit_id": row.get("unit_id"), "follow_up": row.get("follow_up")} for row in coach[-2:]],
                      "privacy": "本地状态概览，不包含 PDF 提取文本。"}, ensure_ascii=False, indent=2))


def source_rows(root: Path, query: str, limit: int = 3) -> list[dict[str, object]]:
    return [{"chapter": chunk.chapter, "source": chunk.source, "page": chunk.page_start, "section": chunk.section} for _, chunk in rank(root, query)[:limit]]


def explain(root: Path, topic: str, audience: str) -> None:
    audience_guidance = {
        "graduate": ["定义问题与目标", "解释建模假设和关键机制", "比较强基线、评价方法与研究边界"],
        "new-engineer": ["说明模块解决的问题及其在推荐链路中的位置", "讲清输入、输出和关键工程取舍", "补充常见失败模式与验证方法"],
        "interview": ["用简洁定义说明核心概念", "解释关键机制和适用条件", "总结一项主要取舍与常见追问"],
    }
    results = rank(root, topic)[:5]
    if not results or results[0][0] < EVIDENCE_SCORE_FLOOR:
        results = []
    evidence = [
        {
            "source": chunk.source,
            "chapter": chunk.chapter,
            "page": chunk.page_start,
            "section": chunk.section,
            "excerpt": normalize(chunk.text)[:500],
        }
        for _, chunk in results
    ]
    import recsys_code

    code_examples = recsys_code.search(root, topic, limit=3) if topic.strip() else []
    report = {
        "status": "ready" if evidence else "needs_source",
        "topic": topic,
        "audience": audience,
        "explanation_outline": audience_guidance[audience],
        "evidence": evidence,
        "code_examples": code_examples,
        "instruction": "仅将 evidence 支持的内容表述为本书结论；其他知识标记为扩展说明。code_examples 只是章节源码候选，讲解前须阅读 README 和实际代码，不是 PDF 依据，也不会自动执行。ready 仅表示检索到候选证据，不保证与问题相关，须阅读 PDF 页确认；needs_source 表示没有足够可信的候选。此命令提供证据和讲解提纲，不生成模型撰写的答案。",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def reader_session_path(root: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", name):
        raise SystemExit("--session 只能使用 1–48 位字母、数字、下划线或连字符。")
    return workspace(root) / "reader-sessions" / f"{name}.json"


def reader_session(root: Path, name: str | None) -> dict[str, object]:
    if not name:
        return {}
    path = reader_session_path(root, name)
    if not path.is_file():
        return {}
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise SystemExit("读者会话记录损坏，请检查本地文件。")
    return content


def save_reader_session(root: Path, name: str | None, topic: str, question: str, answer: dict[str, object], concept_id: str | None = None) -> None:
    if not name:
        return
    path = reader_session_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not save PDF extracts or model raw output; this is local ignored state.
    points = answer.get("points", []) if isinstance(answer, dict) else []
    summary = "；".join(str(point.get("text", "")) for point in points if isinstance(point, dict))[:1200]
    path.write_text(json.dumps({"topic": topic, "concept_id": concept_id, "last_question": question,
                                "last_status": answer["status"], "last_summary": summary,
                                "last_boundary": str(answer.get("boundary", ""))[:500]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reader_ask(root: Path, question: str, session: str | None, agent: str, timeout: int, as_json: bool, model: str | None = None) -> None:
    import recsys_qa

    if not question.strip() or timeout <= 0:
        raise SystemExit("问题不能为空，且 --timeout 必须为正数。")
    state = reader_session(root, session)
    prior_topic = str(state.get("topic", "")) if session else ""
    current_concept = concept_id(root, question) if session else None
    previous_concept = state.get("concept_id")
    # A named, different concept starts a new subject; pronoun follow-ups keep the previous one.
    explicit = explicitly_named_concept(root, question) if session and previous_concept else None
    if explicit and previous_concept and explicit != previous_concept:
        prior_topic = ""
        state = {}
        current_concept = explicit
    context = ""
    if state:
        context = json.dumps({key: state.get(key) for key in ("last_question", "last_summary", "last_boundary")}, ensure_ascii=False)
    try:
        rendered, answer, pages = recsys_qa.answer_question(root, question, prior_topic=prior_topic,
                                                              cli=agent, model=model, timeout=timeout, context=context)
    except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
        raise SystemExit(f"问答未完成：{error}") from error
    topic = prior_topic or question
    concept = None
    if answer["status"] == "answered":
        concept = (str(state.get("concept_id") or "") if prior_topic and not explicit else "") or current_concept or concept_id(root, question)
    save_reader_session(root, session, topic, question, answer, concept)
    if as_json:
        print(json.dumps({"question": question, "session": session, "answer": answer,
                          "citations": [{key: page[key] for key in ("id", "chapter", "source", "page")} for page in pages]},
                         ensure_ascii=False, indent=2))
    else:
        print(rendered)


def reader_navigate(root: Path, topic: str, session: str | None) -> None:
    import recsys_navigation

    state = reader_session(root, session)
    subject = topic.strip() or str(state.get("topic", ""))
    if not subject:
        raise SystemExit("需要主题，或已有主题的 --session。")
    report = recsys_navigation.resolve(subject, previous={"context": {"concept_id": state.get("concept_id")}} if state.get("concept_id") else None, root=root)
    if not report["matches"]:
        report["status"] = "not_found"
        report["suggestion"] = "未命中已发布概念或论文；可换用概念名、方法名或先运行 search。"
    else:
        report["status"] = "ready"
    print(json.dumps(report, ensure_ascii=False, indent=2))


def reader_code(root: Path, query: str, chapter: int | None, limit: int) -> None:
    import recsys_code

    try:
        result = recsys_code.report(root, query, chapter=chapter, limit=limit)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


def reader_coach(root: Path, course_id: str, evidence: Path | None, feedback: str, model: str | None = None, semantic: bool = True, agent_cli: str = "claude") -> None:
    import recsys_coach

    try:
        report = recsys_coach.coach(root, course_id, evidence, feedback, semantic=semantic, model=model, agent_cli=agent_cli)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"陪练未完成：{error}") from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_quantity(value: str, expected_unit: str | None = None) -> float | None:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*([kKmM]?)\s*(ms|items?)?\s*", value, re.I)
    if not match:
        return None
    unit = (match.group(3) or "").lower()
    if unit and unit != expected_unit and not (expected_unit == "items" and unit == "item"):
        return None
    scale = {"": 1.0, "k": 1_000.0, "m": 1_000_000.0}[match.group(2).lower()]
    try:
        return float(match.group(1)) * scale
    except OverflowError:
        return math.inf


def design(root: Path, topic: str, constraints: list[str], project: Path | None = None,
           evidence: Path | None = None, domain: str | None = None) -> None:
    import recsys_engineering as engineering
    engineering_context = engineering.project_report(project, evidence, domain)
    required = ["objective", "qps", "p99", "items", "freshness", "availability", "cost"]
    parsed = dict(item.split("=", 1) for item in constraints if "=" in item)
    missing = [key for key in required if not parsed.get(key, "").strip()]
    p99_ms = parse_quantity(parsed.get("p99", ""), "ms")
    qps = parse_quantity(parsed.get("qps", ""))
    items = parse_quantity(parsed.get("items", ""), "items")
    invalid: list[str] = []
    for key, value in (("qps", qps), ("p99", p99_ms), ("items", items)):
        if key not in missing and (value is None or not math.isfinite(value) or value <= 0):
            invalid.append(key)
    availability = parsed.get("availability", "").strip()
    if "availability" not in missing:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)%?", availability)
        if not match or not 0 < float(match.group(1)) <= 100:
            invalid.append("availability")
    budget_names = ("gateway", "feature", "recall", "ranking", "reranking", "reserve")
    if p99_ms and p99_ms > 0:
        raw_budgets = [p99_ms * weight / 100 for weight in (5, 15, 30, 30, 5, 15)]
        budgets = raw_budgets[:-1] + [p99_ms - sum(raw_budgets[:-1])]
    else:
        budgets = []
    minimum_inflight = None
    if qps and p99_ms and "qps" not in invalid and "p99" not in invalid:
        inflight = qps * (p99_ms / 1000)
        if math.isfinite(inflight):
            minimum_inflight = math.ceil(inflight)
        else:
            invalid.append("qps*p99")
    report = {
        "status": "needs_input" if missing or invalid else "draft", "system": topic, "constraints": parsed, "missing": missing, "invalid": invalid,
        "verification": "not_run",
        "request_flow": ["gateway", "feature snapshot", "parallel multi-recall", "merge/filter", "pre-rank", "rank", "rerank", "response"],
        "latency_budget": [{"stage": name, "p99_ms": value} for name, value in zip(budget_names, budgets)],
        "capacity": {"offered_qps": qps, "items": items, "minimum_inflight_at_p99": minimum_inflight, "inflight_note": "P99-based conservative planning estimate, not a mathematical minimum; mean concurrency requires mean latency.", "replica_formula": "ceil(peak_qps / measured_safe_qps_per_replica * headroom)", "headroom": "1.3-2.0; confirm with peak shape and failure-domain target", "unresolved": ["measured safe QPS/replica", "peak-to-average ratio", "candidate count per stage"]},
        "version_contract": "One release manifest binds model, feature schema, Embedding and ANN index; cache/request keys carry release_id; incompatible releases fail closed.",
        "overload_and_fallback": ["Bound queues and concurrency per dependency", "Apply stage deadlines within the end-to-end budget", "Drop optional recall routes before core routes", "Return version-compatible popular/fresh candidates when personalization cannot finish"],
        "observability": ["stage P50/P95/P99", "queue depth and timeout rate", "candidate counts and empty-result rate", "fallback reason", "release_id and feature freshness", "objective guardrails"],
        "rollout": ["manifest compatibility gate", "shadow validation", "small canary with objective and reliability guardrails", "atomic promotion", "whole-release rollback"],
        "sources": source_rows(root, "在线服务 延迟预算 限流 降级 版本一致性"),
        "assumptions": ["Latency allocation is an initial budget, not measured capacity.", "All external calls propagate a request deadline."],
        "pending_business_confirmation": [key for key in ("feature_scale", "consistency", "runtime", "deployment", "privacy") if not parsed.get(key)],
    }
    report.update(engineering_context)
    if domain is not None:
        report["domain_options"] = engineering.PLAYBOOKS[domain]
    print(json.dumps(report, ensure_ascii=False, indent=2))


def review_surface(root: Path, path: Path, project: Path | None = None,
                   evidence: Path | None = None, domain: str | None = None) -> None:
    import recsys_engineering as engineering
    engineering_context = engineering.project_report(project, evidence, domain)
    context = engineering_context.get("project_context")
    if context is not None:
        resolved = (project / path).resolve() if not path.is_absolute() else path.resolve()
        if not resolved.is_relative_to(Path(context["root"])):
            raise SystemExit("评审文件必须位于 --project 目录内。")
        path = resolved
    if not path.exists() or not path.is_file():
        raise SystemExit(f"评审目标不是可读文件：{path}")
    if path.suffix.lower() != ".py":
        print(json.dumps({"status": "unsupported_target", "target": str(path),
                          "target_type": path.suffix.lower().lstrip(".") or "unknown",
                          "message": "当前 review 仅支持 Python 源码；请指定 .py 文件。"}, ensure_ascii=False, indent=2))
        return
    text = path.read_text(encoding="utf-8")
    findings: list[dict[str, object]] = []
    parse_error = None
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        tree = None
        parse_error = f"{error.msg} at line {error.lineno}"

    def attribute_name(node: ast.expr) -> tuple[str, str] | None:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id, node.attr
        return None

    def subscript_name(node: ast.expr) -> tuple[str, str] | None:
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            return node.value.id, node.slice.value
        return None

    if tree:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
                continue
            left, right = node.left, node.comparators[0]
            if isinstance(node.ops[0], ast.LtE) and attribute_name(left) == ("feature", "timestamp") and attribute_name(right) == ("label", "timestamp"):
                findings.append({"severity": "static_candidate", "line": node.lineno, "finding": "Point-in-time predicate found; verify the branch executes and filters future features.", "invariant": "feature_time <= label_time"})
            if isinstance(node.ops[0], ast.Gt) and subscript_name(left) == ("row", "feature_time") and subscript_name(right) == ("row", "label_time"):
                findings.append({"severity": "static_candidate", "line": node.lineno, "finding": "Possible leakage-rejection predicate found; verify it rejects rows at runtime.", "verification": "Run with --inject-leakage and require exit code 2."})
    report = {
        "status": "needs_runtime_verification" if findings else "needs_manual_review", "target": str(path),
        "findings": findings or [{"severity": "needs_manual_review", "finding": "No supported domain invariant was recognized; no clean bill of health is asserted."}],
        "risk": "critical if either point-in-time predicate or rejection gate is removed" if findings else "undetermined",
        "required_verification": ["normal path succeeds", "future-feature injection is rejected", "boundary timestamps remain eligible"],
        "sources": source_rows(root, "样本穿越 point in time join 特征时效性"),
        "limitations": ["Static AST inspection recognizes only explicit Python patterns; a recognized expression is not proof of correct runtime behavior. Execute tests and inspect data contracts before production approval."],
    }
    if parse_error:
        report["parse_error"] = parse_error
    report.update(engineering_context)
    if domain is not None:
        report["domain_acceptance"] = engineering.PLAYBOOKS[domain]["acceptance"]
    print(json.dumps(report, ensure_ascii=False, indent=2))


def debug_scenario(root: Path, symptom: str, fixture: bool = False, project: Path | None = None,
                   evidence: Path | None = None, domain: str | None = None) -> None:
    import recsys_engineering as engineering
    engineering_context = engineering.project_report(project, evidence, domain)
    branches = [
        {"priority": 1, "hypothesis": "traffic or experiment/configuration changed", "evidence": ["traffic by bucket", "configuration diff", "experiment assignment", "first bad time"]},
        {"priority": 2, "hypothesis": "candidate supply or filtering changed", "evidence": ["per-route candidate count", "merge count", "filter reason", "empty-result rate"]},
        {"priority": 3, "hypothesis": "feature freshness or missingness changed", "evidence": ["feature age", "missing rate", "online/offline parity"]},
        {"priority": 4, "hypothesis": "release artifacts are semantically mismatched", "evidence": ["model/Embedding/index release_id", "cache-key version", "rollout event"]},
        {"priority": 5, "hypothesis": "calibration or logging changed", "evidence": ["score distribution", "calibration drift", "logging coverage"]},
        {"priority": 6, "hypothesis": "serving dependency is slow or saturated", "evidence": ["per-route P99", "timeout rate", "queue depth", "inflight", "deadline-exceeded count"]},
    ]
    matched_fixture = "cases/multi-recall-tail-latency.md" if fixture else None
    report = {
        "status": "needs_evidence", "symptom": symptom,
        "immediate_action": ["Preserve request and release identifiers", "Compare affected vs healthy buckets before changing configuration", "Keep non-empty fallback enabled while collecting evidence"],
        "diagnostic_tree": branches,
        "fixture_evidence": {"path": matched_fixture, "scope": "repository teaching case, not live telemetry", "observations": ["one recall RPC has a long tail", "child calls lack shorter deadlines", "the batch waits after timeout", "fallback supply is sufficient"]} if matched_fixture else None,
        "conclusion": "Teaching-case explanation only: slow recall consumes the budget; confirm live telemetry before diagnosing a real incident." if matched_fixture else "No live telemetry supplied; collect the listed evidence before concluding root cause.",
        "remediation": ["Set per-route deadlines below the end-to-end deadline", "Bound per-route concurrency and queueing", "Continue with healthy routes after a child timeout", "Serve non-empty fallback candidates"],
        "recovery_verification": ["fault-injected end-to-end P99 meets the budget", "candidate count and non-empty rate meet SLO", "timeout and fallback reason are observable", "removing the fault restores the normal route"],
        "sources": source_rows(root, "召回 延迟预算 timeout 过载 降级"),
    }
    report.update(engineering_context)
    if domain is not None:
        report["domain_next_check"] = engineering.PLAYBOOKS[domain]["next_check"]
    if evidence is not None:
        checked = engineering_context["evidence_check"]
        report["supported_observation"] = (f"提供的记录中，第 {checked['failed_records']} 条违反 {domain} 契约。"
                                         if checked["failed_records"] else "提供的记录没有违反所检查的契约；不能排除其他原因。")
        report["next_check"] = ("核对记录来源及目标代码路径；" if checked["failed_records"] else
                                "补充故障时段与健康对照样本；") + engineering.PLAYBOOKS[domain]["next_check"]
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
    eval_parser.add_argument("--min-hit-at-1", type=float, default=1.0, help="Hit@1 最低通过率，0..1（默认 1.0）")
    ask_parser = commands.add_parser("ask", help="用已发布 PDF 证据生成可追溯回答并支持追问")
    ask_parser.add_argument("question", help="自然语言问题或追问")
    ask_parser.add_argument("--session", help="本地对话名；续问时复用，保存在忽略的个人工作区")
    ask_parser.add_argument("--agent", default="claude", choices=("claude", "codex"), help="非交互模型 CLI（默认 claude）")
    ask_parser.add_argument("--model", help="可用模型 ID；若本机默认模型不可用请显式指定")
    ask_parser.add_argument("--timeout", type=int, default=120, help="模型超时秒数")
    ask_parser.add_argument("--json", action="store_true", help="输出结构化回答")
    session_clear = commands.add_parser("session-clear", help="清除指定本地问答会话")
    session_clear.add_argument("--session", required=True)
    navigate_parser = commands.add_parser("navigate", help="从主题跳到概念、已发布 PDF、代码和代表论文")
    navigate_parser.add_argument("topic", nargs="?", default="", help="书中主题或概念；有 --session 时可省略")
    navigate_parser.add_argument("--session", help="沿用本地问答会话上下文")
    code_parser = commands.add_parser("code", help="检索各章节 code/ 的真实源码示例及其章节说明")
    code_parser.add_argument("query", nargs="?", default="", help="算法名、主题或方法；配合 --chapter 可省略")
    code_parser.add_argument("--chapter", type=int, help="限定已发布章节号；省略主题时列出本章部分源码")
    code_parser.add_argument("--limit", type=int, default=8, help="最多返回源码数（默认 8）")
    coach_parser = commands.add_parser("coach", help="针对学习提交物生成个性化追问与下一步任务")
    coach_parser.add_argument("--course", required=True, choices=("new-engineer", "graduate"))
    coach_parser.add_argument("--evidence", type=Path, help="学习提交物；也可仅提供 --feedback")
    coach_parser.add_argument("--feedback", default="", help="学习者当前困惑或时间约束")
    coach_parser.add_argument("--model", help="可用模型 ID；本机默认不可用时指定")
    coach_parser.add_argument("--agent", default="claude", choices=("claude", "codex"), help="非交互模型 CLI（默认 claude）")
    coach_parser.add_argument("--no-model", action="store_true", help="仅做离线结构检查，不调用模型")
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
    commands.add_parser("overview", help="查看课程、自由主题与陪练的本地状态概览")
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
    course_checkpoint_parser = course_commands.add_parser("checkpoint", help="记录当前单元小任务的阅读位置")
    course_checkpoint_parser.add_argument("--course", required=True, choices=("new-engineer", "graduate"))
    course_checkpoint_parser.add_argument("--id", required=True, help="course today 返回的当前检查点 ID")
    course_submit_parser = course_commands.add_parser("submit", help="提交当前单元证据并验收")
    course_submit_parser.add_argument("--course", required=True, choices=("new-engineer", "graduate"))
    course_submit_parser.add_argument("--unit", required=True)
    course_submit_parser.add_argument("--evidence", required=True, type=Path)
    course_submit_parser.add_argument("--score", required=True, type=float)
    course_submit_parser.add_argument("--practice-passed", action="store_true")
    course_submit_parser.add_argument("--failure-path", action="store_true")
    course_review_parser = course_commands.add_parser("review", help="列出到期复习，或提交指定单元的复习结果")
    course_review_parser.add_argument("--course", choices=("new-engineer", "graduate"))
    course_review_parser.add_argument("--unit", help="提交复习的课程单元 ID")
    course_review_parser.add_argument("--evidence", type=Path, help="含四项复习答案的文件")
    course_review_parser.add_argument("--score", type=float, help="复习得分，0..1")
    graph_parser = commands.add_parser("graph", help="输出 Mermaid 知识图谱")
    graph_parser.add_argument("--course", choices=("new-engineer", "graduate"))
    graph_parser.add_argument("--output", type=Path)
    self_eval_parser = commands.add_parser("self-eval", help="生成读者本地学习与检索自测报告")
    self_eval_parser.add_argument("--output", type=Path, default=Path(".recsys-workspace/self-eval.json"))
    design_parser = commands.add_parser("design", help="生成可机读的推荐系统设计报告")
    design_parser.add_argument("system", help="待设计系统")
    design_parser.add_argument("constraints", nargs="*", help="key=value 约束，如 qps=10000 p99=200ms")
    design_parser.add_argument("--project", type=Path, help="目标项目目录")
    design_parser.add_argument("--domain", choices=("feature-time", "serving", "release-version"), help="证据契约领域")
    design_parser.add_argument("--evidence", type=Path, help="本地 JSON 证据文件（需同时指定 --domain）")
    review_parser = commands.add_parser("review", help="生成带位置、风险和验证的领域评审报告")
    review_parser.add_argument("path", type=Path, help="待评审文件")
    review_parser.add_argument("--project", type=Path, help="目标项目目录；评审路径相对于此目录")
    review_parser.add_argument("--domain", choices=("feature-time", "serving", "release-version"), help="证据契约领域")
    review_parser.add_argument("--evidence", type=Path, help="本地 JSON 证据文件（需同时指定 --domain）")
    debug_parser = commands.add_parser("debug", help="生成证据树、结论、修复和恢复验证报告")
    debug_parser.add_argument("symptom", help="故障现象")
    debug_parser.add_argument("--fixture", choices=("multi-recall-tail-latency",), help="显式加载教学案例；并非现场遥测")
    debug_parser.add_argument("--project", type=Path, help="目标项目目录")
    debug_parser.add_argument("--domain", choices=("feature-time", "serving", "release-version"), help="证据契约领域")
    debug_parser.add_argument("--evidence", type=Path, help="本地 JSON 证据文件（需同时指定 --domain）")
    verify_parser = commands.add_parser("verify", help="运行目标项目三阶段命令并检查 JSON 契约输出")
    verify_parser.add_argument("--project", type=Path, required=True, help="目标项目目录")
    verify_parser.add_argument("--domain", required=True, choices=("feature-time", "serving", "release-version"))
    for phase in ("baseline", "fault", "recovery"):
        verify_parser.add_argument("--" + phase, required=True, help="以引号包裹的本地命令；输出单个 JSON 记录")
    verify_parser.add_argument("--timeout", type=int, default=30, help="每条命令超时秒数（1–300）")
    args = parser.parse_args()
    if args.command in {"design", "review", "debug"} and bool(args.evidence) != bool(args.domain):
        parser.error("--evidence 与 --domain 必须同时提供。")
    root = args.root.resolve()
    mode = detect_mode(root) if args.mode == "auto" else args.mode
    if args.command in {"search", "page", "eval", "learn", "explain", "design", "review", "debug", "self-eval", "ask", "navigate", "coach"}:
        require_fresh_index(root, mode)
    if args.command == "index":
        build(root, mode)
    elif args.command == "search":
        search(root, args.query, args.limit, args.chapter, args.json)
    elif args.command == "page":
        page(root, args.chapter, args.page)
    elif args.command == "eval":
        evaluate(root, root / args.dataset, args.min_hit_at_1)
    elif args.command == "ask":
        reader_ask(root, args.question, args.session, args.agent, args.timeout, args.json, args.model)
    elif args.command == "session-clear":
        path = reader_session_path(root, args.session)
        if path.is_file():
            path.unlink()
        print(json.dumps({"session": args.session, "status": "cleared"}, ensure_ascii=False))
    elif args.command == "navigate":
        reader_navigate(root, args.topic, args.session)
    elif args.command == "code":
        reader_code(root, args.query, args.chapter, args.limit)
    elif args.command == "coach":
        reader_coach(root, args.course, args.evidence, args.feedback, args.model, not args.no_model, args.agent)
    elif args.command == "learn":
        learning_plan(root, args.topic, args.level, args.minutes)
    elif args.command == "complete":
        complete_topic(root, args.topic, args.evidence, args.practice_passed, args.failure_path, args.score)
    elif args.command == "explain":
        explain(root, args.topic, args.audience)
    elif args.command == "next":
        adaptive_next(root)
    elif args.command == "overview":
        learning_overview(root)
    elif args.command == "course":
        if args.course_action == "checkpoint":
            course_checkpoint(root, args.course, args.id)
        elif args.course_action == "submit":
            course_submit(root, args.course, args.unit, args.evidence, args.score, args.practice_passed, args.failure_path)
        elif args.course_action == "review":
            if args.unit and (not args.course or not args.evidence or args.score is None):
                raise SystemExit("提交复习需要 --course、--unit、--evidence 和 --score。")
            if not args.unit and (args.evidence or args.score is not None):
                raise SystemExit("--evidence 和 --score 仅用于提交指定 --unit 的复习。")
            course_review(root, args.course, args.unit, args.evidence, args.score)
        else:
            course_command(root, args.course_action, getattr(args, "course_id", None) or getattr(args, "course", None), getattr(args, "minutes", None), getattr(args, "message", None))
    elif args.command == "graph":
        knowledge_graph_view(root, args.course, args.output)
    elif args.command == "self-eval":
        self_evaluate(root, root / args.output if not args.output.is_absolute() else args.output)
    elif args.command == "design":
        design(root, args.system, args.constraints, args.project, args.evidence, args.domain)
    elif args.command == "review":
        review_surface(root, args.path, args.project, args.evidence, args.domain)
    elif args.command == "debug":
        debug_scenario(root, args.symptom, bool(args.fixture), args.project, args.evidence, args.domain)
    elif args.command == "verify":
        import recsys_engineering
        result = recsys_engineering.verify_commands(args.project, args.domain,
                    [(phase, getattr(args, phase)) for phase in ("baseline", "fault", "recovery")], args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        doctor(root, mode)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from error
