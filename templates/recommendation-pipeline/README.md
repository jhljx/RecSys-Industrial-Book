# Recommendation request pipeline template

A minimal runnable template for Agent-assisted architecture work. It demonstrates explicit request budgets, multi-channel retrieval, deterministic merging, ranking, overload fallback, and metrics without pretending that in-memory components are production dependencies.

## Run

```bash
python3 templates/recommendation-pipeline/app.py --user u1 --limit 5
python3 templates/recommendation-pipeline/app.py --user u1 --limit 5 --simulate-overload
python3 -m unittest discover -s templates/recommendation-pipeline/tests -v
```

## Contract

Input:

- `user_id`: non-empty user identifier;
- `limit`: positive result count;
- `timeout_ms`: end-to-end monotonic-clock budget.

Output includes ranked item IDs, degradation state, and per-stage latency. The pipeline never starts a stage after the request deadline. Overload uses a preconfigured hot-item fallback instead of retrying inside the request.

## Production substitutions

Components tagged `[MOCK]` are educational in-memory implementations. A production implementation must supply:

- RPC clients with per-call deadlines and bounded concurrency;
- real feature/embedding version contracts;
- circuit breaking, rate limits, and isolated resource pools;
- distributed traces and exported metrics;
- model/index/config rollout and rollback;
- capacity tests at target P99 and failure conditions.

Relevant reading: released chapters on recommendation architecture, retrieval/filtering, ranking, offline services, online services, and vector retrieval. Use `tools/recsys_agent.py search` to locate exact PDF pages.
