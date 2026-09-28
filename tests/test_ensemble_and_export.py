import unittest

from analyzing_llm_rationale import analytics_export, ensemble


class TestEnsemble(unittest.TestCase):
    def test_compute_brier_weights(self):
        models = ["gemma-4-26b-a4b-it", "crowd-follow"]
        weights = ensemble.compute_brier_weights(models)
        self.assertIn("gemma-4-26b-a4b-it", weights)
        self.assertIn("crowd-follow", weights)
        # Lower Brier score gets higher weight
        self.assertGreater(weights["gemma-4-26b-a4b-it"], weights["crowd-follow"])
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)

    def test_aggregate_ensemble_predictions(self):
        forecasts = [
            {"model": "gemma-4-26b-a4b-it", "probability": 0.70, "rationale": "Bullish"},
            {"model": "gpt-oss-120b", "probability": 0.65, "rationale": "Moderate"},
            {"model": "crowd-follow", "probability": 0.40, "rationale": "Market average"},
        ]
        result = ensemble.aggregate_ensemble_predictions(forecasts, market_price=0.50)
        self.assertIn("ensemble_probability", result)
        self.assertGreater(result["ensemble_probability"], 0.50)
        self.assertIsNotNone(result["edge"])
        self.assertIn(result["consensus_level"], ["high", "moderate", "divergent"])
        self.assertEqual(len(result["member_contributions"]), 3)
        self.assertIn("rationale_summary", result)

    def test_empty_forecasts(self):
        result = ensemble.aggregate_ensemble_predictions([])
        self.assertEqual(result["ensemble_probability"], 0.5)


class TestAnalyticsExport(unittest.TestCase):
    def test_export_dataset_to_parquet(self):
        records = [
            {"market_id": "m1", "prob": 0.65, "edge": 0.15},
            {"market_id": "m2", "prob": 0.40, "edge": -0.10},
        ]
        parquet_bytes = analytics_export.export_dataset_to_parquet("test_table", records)
        self.assertTrue(len(parquet_bytes) > 0)
        # Parquet files start with magic bytes PAR1
        self.assertTrue(parquet_bytes.startswith(b"PAR1"))

    def test_empty_export_dataset_to_parquet(self):
        parquet_bytes = analytics_export.export_dataset_to_parquet("empty_table", [])
        self.assertTrue(len(parquet_bytes) > 0)
        self.assertTrue(parquet_bytes.startswith(b"PAR1"))


if __name__ == "__main__":
    unittest.main()
