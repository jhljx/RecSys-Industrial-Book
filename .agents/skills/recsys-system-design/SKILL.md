---
name: recsys-system-design
description: Design an industrial recommendation system or subsystem with explicit scale, latency, freshness, consistency, availability, privacy, cost, degradation, observability, and rollout constraints.
---
# Recommendation system design
1. Resolve objective, QPS, P99, corpus and feature scale, freshness, consistency, runtime, deployment, availability, privacy, and cost. Ask only when alternatives materially differ.
2. Run `python3 tools/recsys_agent.py design "<system>" key=value ...` and close all relevant missing constraints.
3. Search the released PDFs and read the cited pages.
4. Reuse a runnable template; `[MOCK]` dependencies are teaching substitutes.
5. Produce request/data flows, interfaces, capacity and latency budgets, version contract, overload/degradation, metrics, rollout, rollback, and validation.
6. Separate `本书依据`, `工程实现`, `验证`, and `待业务确认`.
