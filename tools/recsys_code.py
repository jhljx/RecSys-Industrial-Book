#!/usr/bin/env python3
"""Discover published chapter examples without executing them or treating code as PDF evidence."""
from __future__ import annotations

from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXTENSIONS = {".py", ".java", ".cpp", ".h", ".sql"}
CI_SMOKE_SOURCES = {
    "ch03 推荐系统评价体系/code/ch01_offline_metrics/classification_metrics.py",
    "ch07 重排模块/code/ch05_diversity/mmr_diversification.py",
    "ch08 混排模块/code/ch02_pre_processing/score_normalization.py",
}
_CODE_PAGE_PREFIX = "./code/"

_FILENAME = re.compile(r"(?<![\w./-])([\w.-]+\.(?:py|java|cpp|h|sql))(?![\w./-])", re.I)


def page_name(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:20] + ".html"


def page_url(path: str) -> str:
    return _CODE_PAGE_PREFIX + page_name(path)


def _chapter_number(directory: Path) -> int | None:
    match = re.match(r"ch(\d{2})\s", directory.name)
    return int(match.group(1)) if match else None


def catalog(root: Path = ROOT, chapter: int | None = None) -> list[dict[str, object]]:
    """Use chapter README labels as editorial hints; only return real source paths."""
    root = Path(root)
    records = []
    for directory in sorted(root.glob("ch[0-9][0-9] *")):
        number = _chapter_number(directory)
        if number is None or chapter is not None and chapter != number:
            continue
        code_dir = directory / "code"
        readme = code_dir / "README.md"
        if not readme.is_file():
            continue
        files = [p for p in code_dir.rglob("*") if p.is_file() and p.suffix.lower() in SOURCE_EXTENSIONS
                 and not p.is_symlink() and not {"__pycache__", "target", "build", "dist", ".venv", "venv"}.intersection(p.relative_to(code_dir).parts)]
        names: dict[str, list[Path]] = {}
        for path in files:
            names.setdefault(path.name.casefold(), []).append(path)
        labels: dict[Path, str] = {}
        for line in readme.read_text(encoding="utf-8").splitlines():
            if not re.match(r"^\s*(?:[│├└─ ]{2,}|[-*]\s+|\|\s*)`?[\w.-]+\.(?:py|java|cpp|h|sql)\b", line):
                continue
            filename = _FILENAME.search(line)
            if not filename:
                continue
            hits = names.get(filename.group(1).casefold(), [])
            if len(hits) == 1 and hits[0] not in labels:
                labels[hits[0]] = line.strip(" |├└─-` ")[:240]
        for path in sorted(files):
            relative = path.relative_to(root).as_posix()
            validation = "ci_smoke" if relative in CI_SMOKE_SOURCES else "syntax_only" if path.suffix == ".py" else "source_only"
            records.append({"path": relative, "chapter": number, "validation": validation,
                            "readme": readme.relative_to(root).as_posix(),
                            "description": labels.get(path, ""), "language": path.suffix.lstrip("."),
                            "page": page_url(path.relative_to(root).as_posix())})
    return records


def search(root: Path, query: str, *, chapter: int | None = None, limit: int = 8) -> list[dict[str, object]]:
    if limit < 1 or chapter is not None and chapter < 0:
        raise ValueError("limit 必须为正数，章节号不能为负数。")
    query = query.strip().casefold()
    if not query and chapter is None:
        raise ValueError("请提供代码主题或 --chapter。")
    terms = list(dict.fromkeys(re.findall(r"[a-z][a-z0-9_-]+|[一-鿿]{2,}", query)))
    # A single narrow query is stronger than incidental matches on generic words.
    scored = []
    for item in catalog(root, chapter):
        if not query:
            score = 1
        else:
            name = Path(str(item["path"])).stem.casefold().replace("_", " ").replace("-", " ")
            label = str(item["description"]).casefold()
            short_ascii = query.isascii() and len(query) < 4
            score = 28 if not short_ascii and query in name else 0
            if short_ascii:
                score = 28 if query in re.split(r"[_\W]+", name) else 0
                if not score:
                    continue
            elif not score and query.isascii():
                for token in re.split(r"[_\W]+", name):
                    if token.startswith(query):
                        score = 12
                        break
            if not score and query in label:
                score = 20
            for term in terms:
                if not term.isascii() and term in label and term not in name:
                    score += 4
                elif term in name:
                    score += 2
            if len(terms) > 1 and not score:
                continue
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["chapter"], pair[1]["path"]))
    return [item for _, item in scored[:limit]]


def report(root: Path, query: str, *, chapter: int | None = None, limit: int = 8) -> dict[str, object]:
    items = search(root, query, chapter=chapter, limit=limit)
    return {"status": "ready" if items else "not_found", "query": query, "chapter": chapter,
            "examples": items, "usage": "先读章节 code/README.md 与命中源码，再对照发布 PDF 指定页讲解；page 是知识地图内该源码的 HTML 阅读页。改写新代码时标注复用/改动与假设，独立测试正常和失败路径。此处仅定位文件，不执行代码、不保证依赖齐备，代码不是书中论断的证据。"}
