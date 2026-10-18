---
name: recsys-experiments
description: Run and extend the repository's recommendation-system projects covering the ranking funnel, two-tower ANN, point-in-time feature consistency, and online serving reliability.
---
# Recommendation experiments
1. Choose one project under `templates/` based on the learning objective.
2. Read its README and cited PDF pages before changing code.
3. Run the baseline, then the documented fault injection.
4. Change one variable at a time; record input, version, metrics, and expected invariant.
5. Validate the consumer-visible outcome, not internal wiring.
6. Treat every `[MOCK]` as a teaching substitute. A production proposal must add real integrations, capacity, timeout, degradation, observability, versioning, rollback, and offline/online parity.
