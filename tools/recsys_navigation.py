#!/usr/bin/env python3
"""Public, citation-first navigation for recommendation-system Q&A.

No PDF text or private authoring files are read. Returned paths are repository-relative.
The caller must still search/read PDF pages before making factual book claims.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CITATION_PAGE = re.compile(r"PDF\s*第\s*(\d+)\s*(?:[–—-]\s*(\d+))?\s*页")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---", re.S)
_FOLLOWUP = re.compile(r"^(?:那|那么|它|这个|这些|其|再|请|能否|可以|给我|看看|以及|的|关于|对应|相关|还有|展开|具体|代码|论文|怎么|如何|为什么|哪里|哪些|有无|是什么|实例|示例|实现|对比|继续|细说|详述|讲讲|呢|？|\?|\s)+$")
# Editorial pointers, not a guarantee that a demonstration implements a production model.
_CODE_HINTS = {
    "two-tower": ("templates/two-tower-ann/app.py", "ch10 召回模型/code/ch04_vector_retrieval/two_tower.py"),
    "negative-sampling": ("ch10 召回模型/code/ch04_vector_retrieval/two_tower_debias.py",),
    "collaborative-filtering-recall": ("ch10 召回模型/code/ch01_collaborative_filtering/item_cf.py",),
    "feature-freshness": ("templates/feature-consistency/app.py",),
    "point-in-time-join": ("templates/feature-consistency/app.py",),
    "version-consistency": ("templates/two-tower-ann/app.py", "templates/serving-reliability/app.py"),
    "fallback": ("templates/serving-reliability/app.py",),
    "overload-control": ("templates/serving-reliability/app.py",),
    "multi-recall": ("templates/recommendation-pipeline/app.py",),
    "reranking": ("templates/recommendation-pipeline/app.py",),
}


def _metadata(path: Path) -> dict[str, str]:
    match = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"missing frontmatter: {path}")
    return dict(re.findall(r"^([a-z_]+):\s*(.+)$", match.group(1), re.M))


def _spans(term: str, text: str) -> list[tuple[int, int]]:
    """Literal matching; ASCII names must not match inside longer model names."""
    pattern = re.escape(term.casefold())
    if term[0].isascii() and term[0].isalnum():
        pattern = rf"(?<![a-z0-9_/-]){pattern}"
    if term[-1].isascii() and term[-1].isalnum():
        pattern += r"(?![a-z0-9_/-])"
    return [hit.span() for hit in re.finditer(pattern, text.casefold())]


class NavigationIndex:
    """Load published metadata once and resolve entity / short follow-up queries.

    resolve(query, previous=None) accepts a previous result (or its context dict).
    It returns {matches, context, used_context}; each match carries id, title,
    concept_card, chapter, chapter_map, pdf, pdf_page, code, papers, related.
    A paper match additionally carries paper_id, paper_title, paper_pages.
    Ambiguous aliases are returned as candidates rather than silently choosing one.
    """

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.chapters: dict[int, dict] = {}
        for path in sorted((self.root / "knowledge/chapters").glob("ch[0-9][0-9].md")):
            meta = _metadata(path)
            pdf = meta["pdf"]
            if (self.root / pdf).is_file():
                number = int(meta["chapter"])
                self.chapters[number] = {"number": number, "title": meta["title"], "pdf": pdf,
                                         "chapter_map": path.relative_to(self.root).as_posix()}
        self.concepts: dict[str, dict] = {}
        for path in sorted((self.root / "knowledge/concepts").glob("*.md")):
            meta = _metadata(path)
            chapter = int(meta["chapter"])
            if chapter not in self.chapters:
                continue
            text = path.read_text(encoding="utf-8")
            citation = re.search(r"(?ms)^## 本书依据\s*\n(.*?)(?=^## |\Z)", text)
            page = _CITATION_PAGE.search(citation.group(1)) if citation else None
            # No PDF page = no trustworthy Q&A citation.
            if not page:
                continue
            ident = meta["id"]
            self.concepts[ident] = {"id": ident, "title": meta["title"], "chapter": chapter,
                                    "concept_card": path.relative_to(self.root).as_posix(),
                                    "pdf_page": int(page.group(1)),
                                    "pdf_page_end": int(page.group(2)) if page.group(2) else None}
        graph = json.loads((self.root / "knowledge/graph.json").read_text(encoding="utf-8"))
        self.edges = [(a, b) for a, b in graph["edges"] if a in self.concepts and b in self.concepts]
        catalog = json.loads((self.root / "knowledge/paper-map.json").read_text(encoding="utf-8"))
        self.papers = [p for p in catalog["papers"] if p["chapter"] in self.chapters and
                       all(c in self.concepts for c in p["concepts"])]
        self.aliases = json.loads((self.root / "knowledge/aliases.json").read_text(encoding="utf-8"))
        self.terms: list[tuple[str, str, str]] = []
        for ident, item in self.concepts.items():
            self.terms.extend((("concept", ident, term) for term in {ident, ident.replace("-", " "), item["title"]} if len(term) > 1))
        for paper in self.papers:
            self.terms.extend((("paper", paper["id"], term) for term in {paper["id"], paper["method"], paper["title"]} if len(term) > 1))
        by_title: dict[str, list[str]] = {}
        for ident, item in self.concepts.items():
            by_title.setdefault(item["title"], []).append(ident)
        for name, variants in self.aliases.items():
            for ident in by_title.get(name, []):
                self.terms.extend(("concept", ident, term) for term in [name, *variants] if len(term) > 1)

    def _code(self, ident: str, papers: list[dict]) -> list[str]:
        import recsys_code

        candidates = list(_CODE_HINTS.get(ident, ()))
        code_dir = self.root / next((p.name for p in self.root.glob(f"ch{self.concepts[ident]['chapter']:02d} *") if p.is_dir()), "") / "code"
        if code_dir.is_dir():
            for paper in papers:
                # Method-to-script matching only for exact stems, not an invented implementation.
                stem = re.sub(r"[^a-z0-9]+", "_", paper["method"].lower()).strip("_")
                candidates.extend(p.relative_to(self.root).as_posix() for p in code_dir.glob(f"*/{stem}.py"))
        # README-described examples are discovery candidates, not proof of equivalence.
        terms = [self.concepts[ident]["title"], ident.replace("-", " ")]
        terms.extend(paper["method"] for paper in papers)
        discovered = {}
        for term in terms:
            for item in recsys_code.search(self.root, term, chapter=self.concepts[ident]["chapter"], limit=3):
                if item["description"] or Path(str(item["path"])).stem.casefold() == ident.replace("-", "_"):
                    discovered[item["path"]] = item
        candidates.extend(discovered)
        return list(dict.fromkeys(path for path in candidates if (self.root / path).is_file()))[:8]

    def _concept(self, ident: str, focus_paper: dict | None = None) -> dict:
        item = self.concepts[ident]
        chapter = self.chapters[item["chapter"]]
        papers = [p for p in self.papers if ident in p["concepts"]]
        if focus_paper:
            papers = [focus_paper]
        related = [{"id": b if a == ident else a, "title": self.concepts[b if a == ident else a]["title"],
                    "relation": "downstream" if a == ident else "prerequisite"}
                   for a, b in self.edges if a == ident or b == ident]
        result = {"id": ident, "title": item["title"], "concept_card": item["concept_card"],
                  "chapter": item["chapter"], "chapter_map": chapter["chapter_map"],
                  "pdf": chapter["pdf"], "pdf_page": item["pdf_page"], "pdf_page_end": item["pdf_page_end"],
                  "code": self._code(ident, papers), "papers": [
                      {"id": p["id"], "method": p["method"], "title": p["title"],
                       "pages": p["pages"], "pdf": self.chapters[p["chapter"]]["pdf"],
                       **({"url": p["url"]} if "url" in p else {})} for p in papers], "related": related}
        import recsys_code

        result["code_pages"] = [{"path": path, "url": recsys_code.page_url(path)} for path in result["code"]]
        if focus_paper:
            result.update(paper_id=focus_paper["id"], paper_title=focus_paper["title"],
                          paper_pages=focus_paper["pages"], pdf_page=focus_paper["pages"][0],
                          pdf_page_end=focus_paper["pages"][1])
        return result

    def resolve(self, query: str, previous: dict | None = None, limit: int = 5) -> dict:
        if limit < 1:
            raise ValueError("limit must be positive")
        candidates = []
        for kind, ident, term in self.terms:
            for start, end in _spans(term, query):
                candidates.append((start, end, kind, ident))
        # Longest non-overlapping term wins; repeated occurrences can reference multiple entities.
        occupied: list[tuple[int, int]] = []
        matches: list[tuple[str, str]] = []
        for start, end, kind, ident in sorted(candidates, key=lambda row: (-(row[1] - row[0]), row[0], row[2] != "paper", row[3])):
            if any(start < right and left < end for left, right in occupied):
                # Identical ambiguous alias must retain all possible concepts.
                if any(start == left and end == right for left, right in occupied) and (kind, ident) not in matches:
                    matches.append((kind, ident))
                continue
            occupied.append((start, end))
            if (kind, ident) not in matches:
                matches.append((kind, ident))
        used_context = False
        if not matches and previous and _FOLLOWUP.fullmatch(query.strip()):
            context = previous.get("context", previous)
            if isinstance(context, dict):
                ident = context.get("concept_id")
                if ident in self.concepts:
                    matches = [("concept", ident)]
                    used_context = True
        output = []
        for kind, ident in matches[:limit]:
            if kind == "paper":
                paper = next(p for p in self.papers if p["id"] == ident)
                output.extend(self._concept(concept, paper) for concept in paper["concepts"] if len(output) < limit)
            else:
                output.append(self._concept(ident))
        context = {"concept_id": output[0]["id"]} if len(output) == 1 else None
        return {"matches": output[:limit], "context": context, "used_context": used_context}


def resolve(query: str, previous: dict | None = None, root: Path = ROOT, limit: int = 5) -> dict:
    """One-shot public navigation; reuse NavigationIndex for multiple turns."""
    return NavigationIndex(root).resolve(query, previous=previous, limit=limit)
