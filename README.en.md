# RecSys-Industrial-Book

[简体中文说明](README.md) · [Browsable chapter and concept catalog](knowledge-map/catalog.html) · [Book PDF reader](knowledge-map/reader.html) · [Chapter code](knowledge-map/code.html)

**An open Chinese-language book, *推荐系统：工业架构与核心算法* (Industrial Architecture and Core Algorithms of Recommender Systems), and a local-first learning toolkit.** It is intended for graduate students, recommender-system engineers, architects, and researchers with prior ML or recommendation experience. The book PDFs and most learning materials are **in Chinese; there is no English translation of the book**.

## What you can explore

- **Industrial recommender-system architecture:** data and features, candidate retrieval, pre-ranking, ranking, re-ranking, online serving, evaluation, and reliability.
- **Models and methods:** collaborative filtering, two-tower retrieval, approximate nearest neighbors, deep ranking, multi-task learning, bias, duration prediction, and other topics in the [published chapters](knowledge-map/catalog.html).
- **Book-grounded learning:** the [PDF reader](knowledge-map/reader.html), [static chapter and concept catalog](knowledge-map/catalog.html), curated concept cards with PDF-page references, and learning paths. The released chapter PDFs are authoritative; navigation cards are not a substitute for the book.
- **Engineering examples:** [chapter source-code pages](knowledge-map/code.html) and [runnable templates](templates/) for practicing recommendation workflows. Teaching examples marked `[MOCK]` are not production integrations.
- **Local AI Agent toolkit:** PDF-page-cited retrieval and Q&A, chapter-code discovery, study plans, self-evaluation, and engineering exercises. Search indexes and personal progress stay on your machine, not on the public site.

## Try it locally

With Python 3.10+ and `pdftotext` installed, run from the repository root:

```bash
python3 tools/recsys_agent.py index
python3 tools/recsys_agent.py doctor
python3 tools/recsys_agent.py search "two-tower negative sampling bias"
python3 tools/recsys_agent.py code "two-tower" --chapter 10
```

For commands and citation rules, see the [Agent guide](docs/agent-guide.md) and [repository instructions](AGENTS.md). The public website is a static reader, not a hosted AI assistant.

## Sources, license, and linking

The [released PDFs](knowledge-map/reader.html) are the source for book claims. When discussing a topic, link to its [concept card or chapter map](knowledge-map/catalog.html) and cite the corresponding PDF chapter and page; when discussing an implementation, link to the [specific code page](knowledge-map/code.html). Do not treat a code example as evidence for a book claim. You may link to this repository and its public pages; check the [license](LICENSE) before reusing or redistributing content (CC BY-NC-SA 4.0). This repository does not publish its private LaTeX source, bulk PDF text, or a prose-equivalent Markdown book.

Canonical project entry: https://jhljx.github.io/RecSys-Industrial-Book/knowledge-map/ · Source: https://github.com/jhljx/RecSys-Industrial-Book
