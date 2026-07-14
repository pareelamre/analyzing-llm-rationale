from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def load_evaluate_metrics_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_metrics.py"
    spec = importlib.util.spec_from_file_location("evaluate_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvaluateMetricsTests(unittest.TestCase):
    def test_parse_temperature_handles_abbreviated_directory_tags(self):
        module = load_evaluate_metrics_module()

        cases = {
            "temperature_0": 0.0,
            "temperature_00": 0.0,
            "temperature_000": 0.0,
            "temperature_025": 0.25,
            "temperature_0025": 0.25,
            "temperature_075": 0.75,
            "temperature_0075": 0.75,
            "temperature_125": 1.25,
            "temperature_175": 1.75,
            "temperature_2": 2.0,
            "temperature_200": 2.0,
        }

        for dirname, expected in cases.items():
            with self.subTest(dirname=dirname):
                self.assertEqual(module.parse_temperature(dirname), expected)

    def test_score_result_rows_reports_conditional_and_coverage_penalized_metrics(self):
        module = load_evaluate_metrics_module()

        targets = {
            1: 1,
            2: 0,
            3: 1,
            4: 0,
        }
        rows = [
            {"id": "1", "predicted_answer": "yes", "confidence": 0.8},
            {"id": 2, "predicted_answer": "yes", "confidence": 0.7},
            {"id": 3, "predicted_answer": "maybe", "confidence": 0.6},
            {"id": 99, "predicted_answer": "no", "confidence": 0.9},
        ]

        metrics = module.score_result_rows(rows, targets, bins=10)

        self.assertEqual(metrics["n_expected"], 4)
        self.assertEqual(metrics["n_valid"], 2)
        self.assertEqual(metrics["n_invalid"], 1)
        self.assertEqual(metrics["n_absent"], 1)
        self.assertEqual(metrics["n_invalid_or_missing"], 2)
        self.assertEqual(metrics["n_extra_rows"], 1)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["conditional_accuracy"], 0.5)
        self.assertEqual(metrics["coverage_penalized_accuracy"], 0.25)
        self.assertAlmostEqual(metrics["conditional_brier_score"], (0.04 + 0.49) / 2)
        self.assertAlmostEqual(
            metrics["coverage_penalized_brier_score"],
            (0.04 + 0.49 + 2.0) / 4,
        )
        self.assertEqual(metrics["accuracy"], metrics["conditional_accuracy"])
        self.assertEqual(metrics["brier_score"], metrics["conditional_brier_score"])


if __name__ == "__main__":
    unittest.main()
