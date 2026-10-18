---
name: recsys-book-qa
description: Answer recommendation-system questions from the released book PDFs with chapter and page citations. Use for explanations, comparisons, formulas, architecture concepts, and questions asking what the book says.
---

# Recommendation book Q&A

1. Run `python3 tools/recsys_agent.py doctor`.
2. If `.recsys-agent/index.jsonl` is absent or PDFs changed, run `python3 tools/recsys_agent.py index`.
3. Search the exact query; retry with Chinese/English aliases and model abbreviations.
4. Read the cited PDF pages for full context when the answer depends on qualifications, equations, tables, or adjacent paragraphs.
5. Answer with conclusion, book-grounded reasoning, tradeoffs, and source PDF/page.
6. Mark knowledge not supported by released PDFs as `扩展说明`.
7. If released chapters do not cover the topic, say so.

`.recsys-agent/` is a disposable local cache. Never commit, publish, or quote it as a replacement edition of the book.
