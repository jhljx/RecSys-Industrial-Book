#!/usr/bin/env python3
"""Build and search a local, disposable index for the recommendation book.

Private repositories are indexed from LaTeX. Public repositories are indexed
from released PDFs. Extracted text is written only below .recsys-agent/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

INDEX_DIR = Path(".recsys-agent")
INDEX_FILE = INDEX_DIR / "index.jsonl"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
CHINESE = re.compile(r"[\u3400-\u9fff]+")
LATIN = re.compile(r"[a-z0-9][a-z0-9_+.#/-]*", re.I)
SECTION = re.compile(r"\\(?:chapter|section|subsection|subsubsection)\*?\{([^{}]+)\}")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    locator: str
    title: str
    text: str


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


def split_blocks(text: str, target: int = 1800) -> list[str]:
    paragraphs = [normalize(p) for p in re.split(r"\n\s*\n", text) if normalize(p)]
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
        starts = [(number, normalize(match.group(1))) for number, line in enumerate(lines, 1) if (match := SECTION.search(line))]
        if not starts:
            starts = [(1, path.stem)]
        for position, (start, title) in enumerate(starts):
            end = starts[position + 1][0] - 1 if position + 1 < len(starts) else len(lines)
            for block_number, block in enumerate(split_blocks("\n".join(lines[start - 1 : end])), 1):
                chunks.append(Chunk(f"{path.stem}:{start}:{block_number}", relative, f"lines {start}-{end}", title, block))
    return chunks


def public_pdfs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("ch[0-9][0-9] */*.pdf") if path.is_file())


def public_chunks(root: Path) -> list[Chunk]:
    executable = shutil.which("pdftotext")
    if not executable:
        raise SystemExit("缺少 pdftotext；macOS 可执行 `brew install poppler`。")
    chunks: list[Chunk] = []
    for path in public_pdfs(root):
        relative = path.relative_to(root).as_posix()
        result = subprocess.run([executable, "-layout", str(path), "-"], check=True, capture_output=True, text=True)
        for page_number, page in enumerate(result.stdout.split("\f"), 1):
            page = normalize(page)
            if not page:
                continue
            for block_number, block in enumerate(split_blocks(page, target=2400), 1):
                chunks.append(Chunk(f"{relative}:p{page_number}:{block_number}", relative, f"PDF page {page_number}", path.stem, block))
    return chunks


def build(root: Path, mode: str) -> None:
    chunks = private_chunks(root) if mode == "private" else public_chunks(root)
    if not chunks:
        raise SystemExit(f"未发现可索引的 {'LaTeX' if mode == 'private' else 'PDF'} 内容。")
    (root / INDEX_DIR).mkdir(parents=True, exist_ok=True)
    with (root / INDEX_FILE).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    manifest = {"format": 1, "mode": mode, "chunks": len(chunks), "sources": len({c.source for c in chunks}), "content_policy": "private-source" if mode == "private" else "local-pdf-extraction-not-for-commit"}
    (root / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"indexed mode={mode} sources={manifest['sources']} chunks={manifest['chunks']} path={INDEX_FILE}")


def load(root: Path) -> list[Chunk]:
    path = root / INDEX_FILE
    if not path.is_file():
        raise SystemExit("本地索引不存在。请先执行 `python3 tools/recsys_agent.py index`。")
    with path.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]


def search(root: Path, query: str, limit: int) -> None:
    chunks = load(root)
    query_terms = Counter(terms(query))
    if not query_terms:
        raise SystemExit("查询词为空。")
    wanted = set(query_terms)
    frequencies = [Counter(term for term in terms(c.title + " " + c.text) if term in wanted) for c in chunks]
    document_frequency = Counter(term for row in frequencies for term in row)
    scores: list[tuple[float, int]] = []
    for index, row in enumerate(frequencies):
        score = sum((1 + math.log(row[t])) * math.log((len(chunks) + 1) / (document_frequency[t] + 1)) * count for t, count in query_terms.items() if row[t])
        if query.lower() in (chunks[index].title + " " + chunks[index].text).lower():
            score += 8
        if score:
            scores.append((score, index))
    scores.sort(reverse=True)
    for rank, (score, index) in enumerate(scores[:limit], 1):
        chunk = chunks[index]
        excerpt = normalize(chunk.text)
        if len(excerpt) > 700:
            excerpt = excerpt[:700].rstrip() + "…"
        print(f"[{rank}] score={score:.3f} {chunk.source} ({chunk.locator})\n    {chunk.title}\n    {excerpt}\n")
    if not scores:
        print("未命中。请改用章节名、模型名、英文缩写或更短的关键词。")


def doctor(root: Path, mode: str) -> None:
    print(f"mode={mode}")
    print(f"root={root}")
    print(f"pdftotext={shutil.which('pdftotext') or 'missing'}")
    print(f"index={'ready' if (root / INDEX_FILE).is_file() else 'missing'}")
    print(f"{'tex_sources' if mode == 'private' else 'pdf_sources'}={len(list((root / 'chapters').glob('ch*.tex'))) if mode == 'private' else len(public_pdfs(root))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("auto", "private", "public"), default="auto")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index")
    query_parser = subparsers.add_parser("search")
    query_parser.add_argument("query")
    query_parser.add_argument("--limit", type=int, default=5)
    subparsers.add_parser("doctor")
    args = parser.parse_args()
    root = args.root.resolve()
    mode = detect_mode(root) if args.mode == "auto" else args.mode
    if args.command == "index":
        build(root, mode)
    elif args.command == "search":
        search(root, args.query, args.limit)
    else:
        doctor(root, mode)


if __name__ == "__main__":
    main()
