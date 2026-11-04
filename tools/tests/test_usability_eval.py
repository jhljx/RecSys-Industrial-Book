"""Validation and explicit third-reviewer verdicts for blinded usability runs."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import usability_eval as evaluator


def review(task="Q1", label="A", reviewer="first", score=1):
    return {"task_id": task, "label": label, "reviewer": reviewer,
            **dict.fromkeys(evaluator.SCORE_FIELDS, score), "first_success_seconds": 2.5}


class UsabilityEvalTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reviews = self.root / "reviews.jsonl"
        self.assignment = self.root / "assignment.json"
        self.assignment.write_text(json.dumps({task: {"A": "old", "B": "new"}
                                                for task in evaluator.TASKS}))

    def write(self, rows):
        self.reviews.write_text("\n".join(json.dumps(row) for row in rows))

    def summarize(self, rows, complete):
        self.write(rows)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            if complete:
                evaluator.summarize(argparse.Namespace(reviews=self.reviews, assignment=self.assignment))
            else:
                with self.assertRaises(SystemExit) as error:
                    evaluator.summarize(argparse.Namespace(reviews=self.reviews, assignment=self.assignment))
                self.assertEqual(1, error.exception.code)
        result = json.loads(output.getvalue())
        self.assertEqual("complete" if complete else "incomplete", result["status"])
        return result

    def rows(self):
        return [review(task, label, reviewer) for task in evaluator.TASKS
                for label in evaluator.LABELS for reviewer in ("first", "second")]

    def conflict(self):
        rows = self.rows()
        rows[0]["correctness"] = 0
        rows[1]["correctness"] = 2
        return rows

    def verdict(self):
        third = review(reviewer="third", score=2)
        third["adjudications"] = {"correctness": {
            "reviewers": ["first", "second"], "score": 2,
            "reason": "Checked the cited PDF and runnable output."}}
        return third

    def test_accepts_complete_independent_reviews(self):
        result = self.summarize(self.rows(), True)
        self.assertEqual(1, result["reviewer_agreement_within_one"])

    def test_rejects_boolean_scores_and_invalid_elapsed(self):
        for field in evaluator.SCORE_FIELDS:
            for value in (True, False, -1, 3, 1.0, "1", None):
                with self.subTest(field=field, value=value):
                    row = review()
                    row[field] = value
                    self.write([row])
                    with self.assertRaises(SystemExit):
                        evaluator.load_reviews(self.reviews)
        for value in (True, False, float("nan"), float("inf"), -float("inf"), -1, None, "2"):
            with self.subTest(elapsed=value):
                row = review()
                row["first_success_seconds"] = value
                self.write([row])
                with self.assertRaises(SystemExit):
                    evaluator.load_reviews(self.reviews)

    def test_accepts_finite_nonnegative_elapsed(self):
        for elapsed in (0, 2, 0.5):
            row = review()
            row["first_success_seconds"] = elapsed
            self.write([row])
            self.assertEqual([row], evaluator.load_reviews(self.reviews))

    def test_third_rating_does_not_resolve_conflict(self):
        for extras in ([], [review(reviewer="third")],
                       [review(reviewer="third"), review(reviewer="fourth")]):
            result = self.summarize(self.conflict() + extras, False)
            self.assertEqual(["Q1/A/correctness"], result["requires_third_reviewer"])

    def test_explicit_verdict_resolves_and_controls_score(self):
        result = self.summarize(self.conflict() + [self.verdict()], True)
        self.assertEqual({"correctness": 2}, result["tasks"][0]["old"]["adjudicated_scores"])
        self.assertEqual(6, result["tasks"][0]["old"]["median_score"])
        self.assertLess(result["reviewer_agreement_within_one"], 1)

    def test_invalid_or_ambiguous_verdicts_do_not_resolve(self):
        for change in ({"reviewers": ["first", "third"]}, {"reviewers": ["first", "first"]},
                       {"score": True}, {"score": 3}, {"score": 0}, {"reason": " "}):
            with self.subTest(change=change):
                third = self.verdict()
                third["adjudications"]["correctness"].update(change)
                self.summarize(self.conflict() + [third], False)
        fourth = self.verdict()
        fourth["reviewer"] = "fourth"
        self.summarize(self.conflict() + [self.verdict(), fourth], False)

    def test_missing_and_duplicate_reviewers_rejected(self):
        self.summarize([review()], False)
        self.write([review(), review()])
        with self.assertRaises(SystemExit):
            evaluator.load_reviews(self.reviews)


if __name__ == "__main__":
    unittest.main()
