# AGENTS.md

## Purpose

This public repository is both the reader distribution for *推荐系统：工业架构与核心算法* and a local Agent toolkit for recommendation-system Q&A and engineering.

The released PDFs are the authoritative book content. This repository intentionally does not publish LaTeX, prose-equivalent Markdown, or extracted PDF text. Curated concept cards, decision guides, learning paths, experiment instructions, and case summaries are allowed when independently written, concise, source-cited, and non-substitutive for the PDFs.

## Select a mode

### Book Q&A mode

1. Use `.claude/skills/recsys-book-qa/SKILL.md` when available.
2. Build the ignored local PDF index if needed:
   `python3 tools/recsys_agent.py index`
3. Search before answering:
   `python3 tools/recsys_agent.py search "<query>"`
4. Treat search output as discovery evidence. Read the cited PDF pages for full context when precision matters.
5. Cite the released chapter PDF and PDF page. Label non-book material `扩展说明`.
6. If the released PDFs do not cover a topic, state that boundary.

### Development mode

1. Use `.claude/skills/recsys-engineering/SKILL.md` when available.
2. Determine requirements that materially affect the solution: objective, QPS, P99, data/feature scale, freshness, consistency, runtime, deployment, availability, privacy, and cost.
3. Search relevant book guidance and inspect repository code/templates.
4. Treat examples marked `[MOCK]` as teaching substitutes, not production integrations.
5. Produce a complete runnable vertical slice. Include timeout, overload/degradation, observability, version consistency, rollback, capacity, and offline/online parity when relevant.
6. Run the changed path and report exact evidence.

### Chapter code as teaching and implementation context

1. Run `python3 tools/recsys_agent.py code "<concept or method>"` (optionally `--chapter N`) to discover existing `ch*/code/` examples across published chapters. `navigate "<concept>"` links a known concept to its PDF, card, chapter and selected code.
2. Read the matched chapter `code/README.md`, actual source and imports before explaining it. Use `search` / `page` to verify the relevant released PDF pages; code and README are implementation examples, never proof of a book claim. Separate `本书依据` from `代码演示` and `扩展说明`.
3. When teaching, identify the example's input, core transformation, output, assumptions, one parameter experiment, and one failure or counterexample. Explain what the learner should observe; do not claim a run occurred unless it did.
4. When writing new code, use the relevant chapter source as an explicit reference, state what was reused and what changed, test the new normal and failure paths, and label teaching mocks. Do not run every chapter script automatically: dependency, model download, GPU and runtime costs vary; check the specific README first.

## Content boundary

Allowed committed artifacts:

- released PDFs;
- independently written concept cards, decision guides, glossaries, learning paths, experiment instructions, and case summaries with PDF citations;
- curated navigation and metadata;
- source code, tests, dependency manifests, and runnable templates;
- Agent instructions and standards-compliant Skills.

Forbidden committed artifacts:

- `.tex` or authoring sources;
- prose-equivalent chapter Markdown or systematic paragraph-by-paragraph summaries;
- copied passages, bulk PDF extraction output, or `.recsys-agent/`;
- private drafts, editing notes, LaTeX source maps, or authoring history;
- knowledge pages without released-PDF citations or with book material mislabeled as external guidance.

Knowledge pages MUST keep `本书依据` and `扩展说明` separate. PDF filename and PDF page are the only public source coordinates; private LaTeX paths and line numbers MUST NOT be published.

## Response contract

For book answers, distinguish `本书依据` and `扩展说明`. For engineering work, distinguish `本书依据`, `工程实现`, `验证`, and `待业务确认`. Never claim that a teaching example is production-ready without the missing integrations and operational controls.

## Public knowledge map

- `knowledge-map/index.html` is a dependency-free reader page for both local `file://` browsing and GitHub Pages under `/RecSys-Industrial-Book/knowledge-map/`.
- `knowledge-map/data.js` is generated from released chapter maps, concept cards, `knowledge/graph.json`, and `learning-paths/courses.json` by `python3 tools/build_knowledge_map.py`; never edit it by hand or include PDF extraction or private learning state.
- On Pages, directory `README.md` files resolve at the directory URL (for example `knowledge/`), not `README.html`. Individual Markdown documents resolve as `.html`. Shared `_layouts/default.html` and `knowledge-map/document.js` keep document navigation within the themed site.
- The embedded PDF reader requests `#page=N&view=FitH` (fit page width) from the browser viewer. Browser support varies; retain the direct PDF fallback. After changing `reader.js` or `app.js`, update their HTML cache keys and verify live navigation.
- After changing published knowledge, courses or chapter code, regenerate `data.js` and/or code pages, run `python3 -m unittest discover -s tools/tests -v` plus `python3 tools/check_public_boundary.py`. Keep URLs relative so the same page works locally and under the repository Pages subpath.

- The chapter-code explorer `knowledge-map/code.html` and each `knowledge-map/code/*.html` page are generated by `python3 tools/build_code_pages.py` from existing public `ch*/code/` sources and README labels. `code.js` and `code.css` are shared; do not hand-edit generated pages. Rebuild and verify after chapter-code changes. Agent `code` results and `navigate` return matching page URLs. These pages contain code, never extracted PDF text or fabricated claims.

## Discoverability and source links

- `README.en.md` is a concise English project overview, not an English translation of the Chinese book; keep its links and language boundary accurate.
- `python3 tools/build_search_pages.py` regenerates the JS-free chapter/concept catalog and project-path sitemap from public curated metadata and generated code pages. Rebuild it after knowledge or chapter-code changes; do not publish extracted PDF text.
- Concept pages carry their existing PDF citations; `knowledge-map/catalog.html` adds crawlable links to the relevant chapter PDF page and teaching-code candidates. Candidate code is discovery only, not proof of a book claim.
- Pages is hosted under `/RecSys-Industrial-Book/`. A `robots.txt` inside this project path is not the domain-root robots policy. See `docs/discoverability.md` for external links and manual search-console submission; never claim a local change has been deployed or indexed.
