"""Unit tests for Foresea Quantitative Kelly Portfolio Optimizer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.portfolio_optimizer import (  # noqa: E402
    KellyPortfolioOptimizer,
    optimize_portfolio_allocation,
)


class PortfolioOptimizerTests(unittest.TestCase):
    def test_single_market_quarter_kelly_allocation(self):
        opps = [
            {
                "question": "Will CPI exceed 3.0% in Q3?",
                "platform": "Kalshi",
                "market_probability": 0.40,
                "model_probability": 0.60,  # 20% edge
                "credibility_score": 0.90,
                "credibility_grade": "A",
            }
        ]
        res = optimize_portfolio_allocation(opps, bankroll_usd=1000.0, kelly_fraction=0.25)
        self.assertEqual(res["bankroll_usd"], 1000.0)
        self.assertEqual(res["n_positions"], 1)
        self.assertGreater(res["allocated_usd"], 0.0)
        self.assertLessEqual(res["capital_utilization_pct"], 80.0)
        self.assertIn("allocations", res)
        pos = res["allocations"][0]
        self.assertEqual(pos["side"], "YES")
        self.assertEqual(pos["entry_price"], 0.40)
        self.assertGreater(pos["contracts"], 0)

    def test_multi_market_risk_caps_and_cash_reserve(self):
        opps = [
            {
                "question": f"Market {i}",
                "platform": "Polymarket",
                "market_probability": 0.30,
                "model_probability": 0.70,  # Huge edge
                "credibility_score": 0.85,
            }
            for i in range(10)
        ]
        optimizer = KellyPortfolioOptimizer(bankroll_usd=5000.0, max_single_position_pct=0.15, max_total_exposure_pct=0.75)
        res = optimizer.optimize(opps)
        self.assertEqual(res["n_positions"], 10)
        # Total allocation must be <= 75% of bankroll
        self.assertLessEqual(res["allocated_usd"], 5000.0 * 0.75 + 5.0)
        self.assertGreaterEqual(res["cash_reserve_usd"], 5000.0 * 0.25 - 5.0)

    def test_low_credibility_filtering(self):
        opps = [
            {
                "question": "Low Credibility Market",
                "platform": "Kalshi",
                "market_probability": 0.20,
                "model_probability": 0.50,
                "credibility_score": 0.40,  # Below threshold
            }
        ]
        res = optimize_portfolio_allocation(opps, bankroll_usd=1000.0, min_credibility=0.60)
        self.assertEqual(res["n_positions"], 0)
        self.assertEqual(res["allocated_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
