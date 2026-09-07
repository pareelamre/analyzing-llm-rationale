"""The Kelly formula that sizes every position, asserted by amount.

The optimizer's own tests checked that it allocated *something*
(``allocated_usd > 0``, ``capital_utilization_pct <= 80``) and never how
much. So the sizing formula was free: removing the division by the odds,

    full_kelly = (win_prob * (b + 1.0) - 1.0) / b   ->   ... - 1.0)

left all 1,711 tests green while changing the quarter-Kelly allocation on a
0.40 market from 8.33% of bankroll to 12.50%, and on a 0.10 market from
2.78% to the 15% single-position cap -- a 5.4x position.

These assert the number.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.portfolio_optimizer import (  # noqa: E402
    KellyPortfolioOptimizer,
)


def _opp(market_p: float, model_p: float, **extra):
    row = {
        "market_probability": market_p,
        "model_probability": model_p,
        "credibility_score": 0.90,
        "ident": "KXTEST",
        "question": "Test market",
    }
    row.update(extra)
    return row


def _only(result):
    positions = result["allocations"]
    assert len(positions) == 1, f"expected one allocation, got {len(positions)}"
    return positions[0]


class KellyFormulaTests(unittest.TestCase):
    def test_an_even_money_market_sizes_at_the_documented_fraction(self):
        """entry 0.40, win 0.60 -> f* = (0.60-0.40)/(1-0.40) = 1/3."""
        opt = KellyPortfolioOptimizer(bankroll_usd=1000.0, kelly_fraction=0.25)
        pos = _only(opt.optimize([_opp(0.40, 0.60)]))

        self.assertEqual(pos["entry_price"], 0.40)
        self.assertAlmostEqual(pos["full_kelly_pct"], 33.33, places=2)
        # Quarter of it, below the 15% single-position cap.
        self.assertAlmostEqual(pos["allocated_pct"], 8.33, places=2)

    def test_a_longshot_sizes_far_smaller_than_the_odds_alone_suggest(self):
        """entry 0.10, win 0.20 -> f* = (0.20-0.10)/(1-0.10) = 1/9.

        This is the price where dropping the division by b hurts most: the
        odds b are 9, so the unnormalised figure is 9x too large and runs
        straight into the single-position cap.
        """
        opt = KellyPortfolioOptimizer(bankroll_usd=1000.0, kelly_fraction=0.25)
        pos = _only(opt.optimize([_opp(0.10, 0.20)]))

        self.assertEqual(pos["entry_price"], 0.10)
        self.assertAlmostEqual(pos["full_kelly_pct"], 11.11, places=2)
        self.assertAlmostEqual(pos["allocated_pct"], 2.78, places=2)

    def test_the_formula_matches_its_own_documented_simplification(self):
        """The code says f* = (win_prob - entry)/(1 - entry). Hold it to that.

        Checked across the price range rather than at one point, because the
        two forms agree exactly at 0.50 -- where b is 1 -- and diverge as the
        price moves away from it.
        """
        opt = KellyPortfolioOptimizer(bankroll_usd=1000.0, kelly_fraction=0.25)
        for market_cents in range(5, 95, 5):
            market_p = market_cents / 100.0
            model_p = min(0.98, market_p + 0.10)
            with self.subTest(market_p=market_p):
                pos = _only(opt.optimize([_opp(market_p, model_p)]))
                entry, win = market_p, model_p
                expected = (win - entry) / (1.0 - entry)
                self.assertAlmostEqual(
                    pos["full_kelly_pct"], round(expected * 100, 2), places=2,
                )


class SizingGuardTests(unittest.TestCase):
    def test_a_position_is_capped_at_the_single_market_limit(self):
        opt = KellyPortfolioOptimizer(
            bankroll_usd=1000.0, kelly_fraction=1.0, max_single_position_pct=0.15,
        )
        pos = _only(opt.optimize([_opp(0.20, 0.80)]))

        # f* = (0.80-0.20)/0.80 = 0.75 at full Kelly; the cap binds.
        self.assertAlmostEqual(pos["full_kelly_pct"], 75.0, places=2)
        self.assertAlmostEqual(pos["allocated_pct"], 15.0, places=2)

    def test_kelly_fraction_is_clamped_to_a_sane_range(self):
        self.assertEqual(
            KellyPortfolioOptimizer(kelly_fraction=0.0).kelly_fraction, 0.05,
        )
        self.assertEqual(
            KellyPortfolioOptimizer(kelly_fraction=5.0).kelly_fraction, 1.0,
        )
        self.assertEqual(
            KellyPortfolioOptimizer(kelly_fraction=0.25).kelly_fraction, 0.25,
        )

    def test_a_zero_edge_market_is_not_allocated(self):
        """Reachable only with min_edge=0, which is why the guard is `<= 0`.

        With model and market agreeing exactly, f* is 0. Under `< 0` that
        passes the guard and emits a 0.00% allocation -- a position in a
        market the model has no view on.
        """
        opt = KellyPortfolioOptimizer(bankroll_usd=1000.0, min_edge=0.0)
        result = opt.optimize([_opp(0.40, 0.40)])

        self.assertEqual(result["allocations"], [])
        self.assertEqual(result["n_positions"], 0)


if __name__ == "__main__":
    unittest.main()
