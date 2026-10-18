---
name: recsys-book-qa
description: Answer recommendation-system questions from the released book PDFs with chapter and page citations. Use for explanations, comparisons, formulas, architecture concepts, and questions asking what the book says.
---
# Recommendation book Q&A
1. Run `python3 tools/recsys_agent.py doctor` and index if stale.
2. Search the exact query; retry with Chinese/English aliases and abbreviations.
3. Read cited PDF pages with the `page` command when qualifications, equations, tables, or adjacent paragraphs matter.
4. Answer with conclusion, book-grounded reasoning, tradeoffs, and source PDF/page.
5. Mark unsupported knowledge as `扩展说明`; if released chapters do not cover it, say so.
6. `.recsys-agent/` is disposable local cache. Never commit, publish, or quote it as a replacement edition.
