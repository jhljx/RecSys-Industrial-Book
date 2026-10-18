---
name: recsys-debugging
description: Diagnose recommendation quality, traffic, feature, model, retrieval, ranking, latency, timeout, overload, degradation, or logging regressions using an evidence-first decision tree.
---
# Recommendation debugging
1. State symptom, first bad time, affected traffic, metric definition, and recent changes.
2. Run `python3 tools/recsys_agent.py debug "<symptom>"`.
3. Follow the emitted order: experiment/config, supply funnel, features, release versions, calibration/logging, then serving health.
4. At each branch collect the named evidence before moving on. Compare against a known-good cohort or version.
5. Reproduce the fault where safe; fix the root cause, then rerun the same scenario.
6. Output timeline, evidence, root cause, corrective change, verification, and prevention. Never suppress an alert as the fix.
