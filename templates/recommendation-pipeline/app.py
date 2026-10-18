#!/usr/bin/env python3
"""Runnable recommendation request pipeline with deadlines and degradation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable


class DeadlineExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    item_id: str
    retrieval_score: float
    source: str


@dataclass(frozen=True)
class Result:
    items: list[str]
    degraded: bool
    reason: str | None
    stage_latency_ms: dict[str, float]


class Deadline:
    def __init__(self, timeout_ms: int) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self._expires_at = time.monotonic() + timeout_ms / 1000

    def check(self, stage: str) -> None:
        if time.monotonic() >= self._expires_at:
            raise DeadlineExceeded(f"deadline exceeded before {stage}")


class Pipeline:
    def __init__(
        self,
        retrievers: Iterable[Callable[[str], list[Candidate]]],
        hot_items: list[str],
        overloaded: Callable[[], bool] = lambda: False,
    ) -> None:
        self._retrievers = tuple(retrievers)
        self._hot_items = list(hot_items)
        self._overloaded = overloaded

    def recommend(self, user_id: str, limit: int, timeout_ms: int) -> Result:
        if not user_id:
            raise ValueError("user_id must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        started = time.monotonic()
        deadline = Deadline(timeout_ms)
        latency: dict[str, float] = {}
        if self._overloaded():
            return Result(self._hot_items[:limit], True, "overload", latency)
        try:
            candidates: list[Candidate] = []
            stage_start = time.monotonic()
            for retrieve in self._retrievers:
                deadline.check("retrieval")
                candidates.extend(retrieve(user_id))
            latency["retrieval"] = elapsed_ms(stage_start)

            stage_start = time.monotonic()
            deadline.check("merge")
            merged: dict[str, Candidate] = {}
            for candidate in candidates:
                current = merged.get(candidate.item_id)
                if current is None or candidate.retrieval_score > current.retrieval_score:
                    merged[candidate.item_id] = candidate
            latency["merge"] = elapsed_ms(stage_start)

            stage_start = time.monotonic()
            deadline.check("ranking")
            ranked = sorted(merged.values(), key=lambda item: (-item.retrieval_score, item.item_id))
            latency["ranking"] = elapsed_ms(stage_start)
            latency["total"] = elapsed_ms(started)
            return Result([item.item_id for item in ranked[:limit]], False, None, latency)
        except DeadlineExceeded:
            latency["total"] = elapsed_ms(started)
            return Result(self._hot_items[:limit], True, "timeout", latency)


def elapsed_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 3)


def collaborative_filtering(user_id: str) -> list[Candidate]:
    """[MOCK] Replace with a bounded RPC to a retrieval service."""
    return [Candidate("item-1", 0.91, "cf"), Candidate("item-2", 0.74, "cf")]


def vector_retrieval(user_id: str) -> list[Candidate]:
    """[MOCK] Replace with versioned user embedding and ANN index clients."""
    return [Candidate("item-2", 0.83, "ann"), Candidate("item-3", 0.79, "ann")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=100)
    parser.add_argument("--simulate-overload", action="store_true")
    args = parser.parse_args()
    pipeline = Pipeline(
        [collaborative_filtering, vector_retrieval],
        hot_items=["hot-1", "hot-2", "hot-3", "hot-4", "hot-5"],
        overloaded=lambda: args.simulate_overload,
    )
    print(json.dumps(asdict(pipeline.recommend(args.user, args.limit, args.timeout_ms)), ensure_ascii=False))


if __name__ == "__main__":
    main()
