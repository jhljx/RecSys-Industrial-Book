"""Small, deterministic, standard-library-only checks for released chapter code.

Import just four explicitly selected scripts; do not recursively execute model demos
that require torch, datasets, pretrained weights, services or GPU resources.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


def chapter_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ChapterSmokeTest(unittest.TestCase):
    def test_classification_confusion_matrix(self):
        module = chapter_module("chapter_classification", "ch03 推荐系统评价体系/code/ch01_offline_metrics/classification_metrics.py")
        result = module.confusion_matrix([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
        self.assertEqual((1, 1, 1, 1), (result.tp, result.fp, result.fn, result.tn))
        self.assertAlmostEqual(0.5, result.f1())
        self.assertEqual(0.0, module.all_negative_baseline([1, 0]).recall())

    def test_mmr_diversity_tradeoff(self):
        module = chapter_module("chapter_mmr", "ch07 重排模块/code/ch05_diversity/mmr_diversification.py")
        similarities = [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertEqual([0, 1], module.mmr_select([0.9, 0.8, 0.5], similarities, 2, 1.0))
        self.assertEqual([0, 2], module.mmr_select([0.9, 0.8, 0.5], similarities, 2, 0.3))

    def test_normalization_keeps_order_and_handles_constant_input(self):
        module = chapter_module("chapter_normalization", "ch08 混排模块/code/ch02_pre_processing/score_normalization.py")
        for label, normalizer in module.NORMALIZERS.items():
            with self.subTest(label=label):
                values = normalizer([1.0, 2.0, 3.0])
                self.assertTrue(values[0] < values[1] < values[2])
        self.assertEqual([0.5, 0.5], module.minmax([3.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
