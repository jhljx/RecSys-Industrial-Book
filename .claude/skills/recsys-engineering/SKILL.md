---
name: recsys-engineering
description: Design, implement, debug, and review industrial recommendation-system architecture or algorithms using the released book PDFs, curated examples, and runnable templates.
---

# Recommendation-system engineering

## Requirements

Resolve the constraints that change implementation: product objective, QPS, P99 latency, corpus/feature scale, update freshness, consistency, runtime, deployment stack, availability, privacy, and cost. Ask only for materially different tradeoffs.

## Evidence

1. Search the local PDF index for relevant book guidance.
2. Read cited PDF pages.
3. Inspect related repository code and templates.
4. Keep `本书依据`, repository convention, and new engineering assumptions separate.

## Build

- Reuse existing interfaces and runnable templates.
- Keep `[MOCK]` dependencies clearly labelled and replace them only when real integration details exist.
- Services require explicit contracts, timeouts, overload/degradation, observability, version consistency, rollback, and capacity assumptions when relevant.
- Models require objective, sampling, feature-time semantics, offline/online parity, complexity, and evaluation criteria when relevant.
- Deliver a complete runnable path without placeholders.

## Verify

Run the changed surface. Report `本书依据`, `工程实现`, `验证`, and unresolved `待业务确认`.
