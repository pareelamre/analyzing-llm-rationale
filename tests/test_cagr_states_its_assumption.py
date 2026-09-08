"""The annualised CAGR is exponential in a constant nothing checks.

estimated_annualized_cagr_pct is exp(weekly_growth x 52) - 1. The 52 says
the whole book resolves and is redeployed weekly. The markets it is
computed over do not:

    live edge board, 27 rows, lead_days min 1.8 / median 23.1 / max 115.2
    median implies 15.8 turnovers a year, against the 52 assumed

Because the figure is exponential in that count, a 3.3x error in the
exponent is four orders of magnitude in the output. The live endpoint
returned 2,049,923,397.7 percent, which is arithmetically exactly
exp(0.3238 x 52) - 1 -- not a coding slip, the constant.

This does not change the published number. Substituting a different
constant is a decision about a metric people may already be reading. It
publishes the assumption beside the figure, and what the held markets
imply, so the gap is visible rather than inferred -- and carries the
resolution dates the board already provides onto each allocation, which
is what any of the possible fixes would need.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.portfolio_optimizer import (  # noqa: E402
    _ASSUMED_TURNOVERS_PER_YEAR,
    KellyPortfolioOptimizer,
    _implied_turnovers_per_year,
)


def _opp(**over):
    row = {
        "question": "Will it happen?",
        "platform": "Kalshi",
        "ident": "KXTEST",
        "market_probability": 0.47,
        "model_probability": 0.75,
        "credibility_score": 0.90,
        "lead_days": 22.1,
        "horizon": "14-30d",
        "resolve_time": "2026-09-30",
    }
    row.update(over)
    return row


class TheNumberIsUnchangedTests(unittest.TestCase):
    def test_the_cagr_is_still_exp_of_weekly_growth_times_52(self):
        """The point is to explain the figure, not to move it.

        Compared with a relative tolerance: the published weekly rate is
        rounded to three decimals, and exponentiating it 52 times amplifies
        that rounding into whole percentage points of the answer.
        """
        result = KellyPortfolioOptimizer(bankroll_usd=1000.0).optimize([_opp()])
        weekly = result["expected_weekly_growth_rate_pct"] / 100.0
        expected = (math.exp(weekly * 52) - 1.0) * 100
        self.assertLess(
            abs(result["estimated_annualized_cagr_pct"] - expected) / expected, 0.001,
        )

    def test_the_figure_is_extremely_sensitive_to_the_growth_rate(self):
        """Why the exponent deserves scrutiny, as arithmetic.

        A rounding difference in the fourth decimal of the weekly rate moves
        the annualised figure by whole percentage points; a change in the
        turnover count moves it by orders of magnitude.
        """
        weekly = 0.10802
        base = (math.exp(weekly * 52) - 1.0) * 100
        nudged = (math.exp((weekly + 0.0001) * 52) - 1.0) * 100
        self.assertGreater(nudged - base, 10.0)

        # At this growth rate, compounding 52 times instead of the ~16 the
        # horizons imply multiplies the answer by 59. At the live book's
        # higher growth rate the same swap spans four orders of magnitude.
        at_implied = (math.exp(weekly * 16) - 1.0) * 100
        self.assertGreater(base / at_implied, 20.0)

        live_weekly = 0.32377          # the rate the endpoint published
        self.assertGreater(
            ((math.exp(live_weekly * 52) - 1.0) * 100)
            / ((math.exp(live_weekly * 16) - 1.0) * 100),
            10_000.0,
        )

    def test_the_constant_is_still_52(self):
        self.assertEqual(_ASSUMED_TURNOVERS_PER_YEAR, 52)


class TheAssumptionIsPublishedTests(unittest.TestCase):
    def test_the_response_says_what_it_assumed(self):
        result = KellyPortfolioOptimizer(bankroll_usd=1000.0).optimize([_opp()])
        assumptions = result["cagr_assumptions"]
        self.assertEqual(assumptions["turnovers_per_year_assumed"], 52)
        self.assertIn("exp(weekly_growth x 52)", assumptions["basis"])

    def test_it_reports_what_the_held_markets_imply(self):
        result = KellyPortfolioOptimizer(bankroll_usd=1000.0).optimize(
            [_opp(lead_days=23.1)],
        )
        self.assertAlmostEqual(
            result["cagr_assumptions"]["turnovers_per_year_implied_by_horizons"],
            round(365.0 / 23.1, 1),
        )

    def test_the_gap_is_visible_on_the_live_shaped_book(self):
        """22 and 115 days: the assumption is an order of magnitude out."""
        result = KellyPortfolioOptimizer(bankroll_usd=1000.0).optimize([
            _opp(ident="a", lead_days=22.1),
            _opp(ident="b", lead_days=115.2, market_probability=0.41,
                 model_probability=0.62),
        ])
        implied = result["cagr_assumptions"]["turnovers_per_year_implied_by_horizons"]
        self.assertLess(implied, 20.0)
        self.assertGreater(result["cagr_assumptions"]["turnovers_per_year_assumed"], implied * 2)


class ImpliedTurnoverTests(unittest.TestCase):
    def test_it_uses_the_median(self):
        """Distinct from the mean, the shortest and the longest.

        A fixture where several of those coincide proves nothing: my first
        version used (7, 7, 7, 365), whose median and minimum are both 7, so
        a mutation reading leads[0] survived it.
        """
        allocations = [{"lead_days": d} for d in (7.0, 30.0, 60.0)]
        leads = [7.0, 30.0, 60.0]
        self.assertAlmostEqual(
            _implied_turnovers_per_year(allocations), round(365.0 / 30.0, 1),
        )
        self.assertNotAlmostEqual(round(365.0 / 30.0, 1), round(365.0 / min(leads), 1))
        self.assertNotAlmostEqual(round(365.0 / 30.0, 1), round(365.0 / max(leads), 1))
        self.assertNotAlmostEqual(
            round(365.0 / 30.0, 1), round(365.0 / (sum(leads) / len(leads)), 1),
        )

    def test_one_long_market_does_not_swamp_a_book_of_short_ones(self):
        allocations = [{"lead_days": d} for d in (7.0, 7.0, 7.0, 365.0)]
        self.assertAlmostEqual(
            _implied_turnovers_per_year(allocations), round(365.0 / 7.0, 1),
        )

    def test_without_horizons_it_says_nothing_rather_than_guessing(self):
        self.assertIsNone(_implied_turnovers_per_year([{"question": "x"}]))
        self.assertIsNone(_implied_turnovers_per_year([]))

    def test_unusable_values_are_ignored(self):
        for bad in (None, 0, -5, "soon"):
            with self.subTest(bad=bad):
                self.assertIsNone(_implied_turnovers_per_year([{"lead_days": bad}]))


class AllocationsCarryTheHorizonTests(unittest.TestCase):
    def test_each_allocation_reports_when_it_resolves(self):
        [alloc] = KellyPortfolioOptimizer(bankroll_usd=1000.0).optimize([_opp()])["allocations"]
        self.assertEqual(alloc["lead_days"], 22.1)
        self.assertEqual(alloc["horizon"], "14-30d")
        self.assertEqual(alloc["resolve_time"], "2026-09-30")

    def test_a_row_without_them_reports_none_rather_than_omitting(self):
        row = {k: v for k, v in _opp().items()
               if k not in ("lead_days", "horizon", "resolve_time")}
        [alloc] = KellyPortfolioOptimizer(bankroll_usd=1000.0).optimize([row])["allocations"]
        for key in ("lead_days", "horizon", "resolve_time"):
            self.assertIn(key, alloc)
            self.assertIsNone(alloc[key])


if __name__ == "__main__":
    unittest.main()
