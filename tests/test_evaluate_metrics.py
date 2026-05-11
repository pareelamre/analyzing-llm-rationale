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


if __name__ == "__main__":
    unittest.main()
