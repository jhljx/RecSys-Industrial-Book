---
name: recsys-engineering
description: Design, implement, debug, and review industrial recommendation-system architecture or algorithms using released PDFs, curated examples, and runnable templates.
---
# Recommendation-system engineering
Resolve constraints that change implementation: objective, QPS, P99, data/feature scale, freshness, consistency, runtime, deployment, availability, privacy, and cost.

Search the local PDF index and read cited pages. Keep `本书依据`, repository convention, and engineering assumptions separate. Reuse runnable templates and label `[MOCK]` teaching substitutes.

Services require explicit contracts, timeouts, overload/degradation, observability, version consistency, rollback, and capacity. Models require objective, sampling, feature-time semantics, offline/online parity, complexity, and evaluation.

Run the changed path. Report `本书依据`, `工程实现`, `验证`, and unresolved `待业务确认`.

For work in a reader's own repository, pass `--project` to `design`, `review` or `debug`; read the target's callers and test instructions before reaching a conclusion. For feature-time, serving, or release-version incidents, accept a bounded JSON observation file with `--domain` and `--evidence` (schema in `docs/agent-guide.md`). `verify` can run three explicit local Python scripts for baseline, fault, and recovery and validate their JSON output. Treat its provenance as a local run, not production telemetry. Compare baseline, fault and recovery records; run the relevant normal and injected failure paths when available. A contract violation in a supplied sample narrows the investigation but does not establish production root cause. Implement a fix only when requested, and retain the original CLI behavior.
