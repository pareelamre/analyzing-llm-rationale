import unittest

from analyzing_llm_rationale import forecast_factors


class TestForecastFactors(unittest.TestCase):
    def test_extract_factors_and_catalysts(self):
        rationale = (
            "Accelerated solar deployment and favorable policy tailwinds increase the likelihood of meeting targets. "
            "However, supply chain barriers and grid interconnection delays create risk. "
            "A key milestone is the upcoming COP30 summit and the scheduled policy announcement in Q3 2026."
        )
        res = forecast_factors.extract_factors_and_catalysts(rationale, probability=0.68)
        self.assertIn("bull_factors", res)
        self.assertIn("bear_factors", res)
        self.assertIn("key_catalysts", res)

        self.assertTrue(len(res["bull_factors"]) > 0)
        self.assertTrue(len(res["bear_factors"]) > 0)
        self.assertTrue(len(res["key_catalysts"]) > 0)

        # Verify structure
        bull = res["bull_factors"][0]
        self.assertIn("factor", bull)
        self.assertIn("impact", bull)
        self.assertIn("weight", bull)

    def test_empty_rationale(self):
        res = forecast_factors.extract_factors_and_catalysts("")
        self.assertEqual(res["bull_factors"], [])
        self.assertEqual(res["bear_factors"], [])
        self.assertEqual(res["key_catalysts"], [])


if __name__ == "__main__":
    unittest.main()
