from __future__ import annotations

import importlib.util
import sys
import time
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("recommendation_pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
pipeline_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline_module
SPEC.loader.exec_module(pipeline_module)

Candidate = pipeline_module.Candidate
Pipeline = pipeline_module.Pipeline


class PipelineTest(unittest.TestCase):
    def test_merges_duplicate_candidates_using_best_score(self) -> None:
        first = lambda _: [Candidate("same", 0.2, "first"), Candidate("other", 0.6, "first")]
        second = lambda _: [Candidate("same", 0.9, "second")]
        result = Pipeline([first, second], ["hot"]).recommend("u1", 2, 100)
        self.assertEqual(["same", "other"], result.items)
        self.assertFalse(result.degraded)

    def test_overload_uses_bounded_fallback(self) -> None:
        never_called = lambda _: self.fail("retriever must not run during overload")
        result = Pipeline([never_called], ["hot-1", "hot-2"], overloaded=lambda: True).recommend("u1", 1, 100)
        self.assertEqual(["hot-1"], result.items)
        self.assertEqual("overload", result.reason)

    def test_timeout_degrades_instead_of_starting_later_stage(self) -> None:
        def slow(_: str) -> list[Candidate]:
            time.sleep(0.01)
            return [Candidate("late", 1.0, "slow")]

        result = Pipeline([slow], ["hot"]).recommend("u1", 1, 1)
        self.assertEqual(["hot"], result.items)
        self.assertEqual("timeout", result.reason)


if __name__ == "__main__":
    unittest.main()
