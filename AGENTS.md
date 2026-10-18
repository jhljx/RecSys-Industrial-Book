# AGENTS.md

## Purpose

This public repository is both the reader distribution for *推荐系统：工业架构与核心算法* and a local Agent toolkit for recommendation-system Q&A and engineering.

The released PDFs are the authoritative book content. This repository intentionally does not publish LaTeX, a prose-equivalent Markdown transcription, or extracted PDF text.

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

## Content boundary

Allowed committed artifacts:

- released PDFs;
- curated navigation and independently written metadata;
- source code, tests, dependency manifests, and runnable templates;
- Agent instructions and Skills.

Forbidden committed artifacts:

- `.tex` or authoring sources;
- Markdown/plain-text transcriptions of book prose;
- PDF extraction output or `.recsys-agent/`;
- private drafts, editing notes, source maps, or authoring history.

## Response contract

For book answers, distinguish `本书依据` and `扩展说明`. For engineering work, distinguish `本书依据`, `工程实现`, `验证`, and `待业务确认`. Never claim that a teaching example is production-ready without the missing integrations and operational controls.
