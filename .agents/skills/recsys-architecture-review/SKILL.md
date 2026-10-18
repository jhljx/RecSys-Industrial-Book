---
name: recsys-architecture-review
description: Review recommendation-system code, architecture, experiment, or design for correctness, serving safety, offline-online consistency, capacity, observability, and rollback gaps.
---
# Recommendation architecture review
1. Inspect the target and its callers before judging it.
2. Run `python3 tools/recsys_agent.py review <path>` for the domain checklist.
3. Search released PDFs for each material finding; read cited pages.
4. Prioritize consumer-visible correctness, leakage, version mismatch, unbounded work, timeout/overload, empty fallback, missing metrics, and unsafe rollout.
5. Report findings by severity with exact location, failure scenario, proposed fix, and verification. No style-only filler.
6. Keep book evidence distinct from engineering judgment.
