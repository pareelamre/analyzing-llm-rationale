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


class UnassessedCredibilityIsNotAnATests(unittest.TestCase):
    """An opportunity with no credibility assessment was published as grade A.

    The filter admits it -- `if cred_score is not None and cred_score <
    self.min_credibility` lets a missing score through by design. The output
    then wrote `opp.get("credibility_grade", "A")` and
    `opp.get("credibility_score", 0.90)`, so "we did not assess this" became
    the best grade available, on the output that sizes real capital.

    The server's own response model types both as Optional, so None is the
    representation that already exists for unknown.
    """

    def _opp(self, **extra):
        base = {
            "question": "Unassessed market",
            "platform": "Kalshi",
            "model_probability": 0.70,
            "market_probability": 0.50,
        }
        base.update(extra)
        return base

    def test_a_missing_grade_is_none_not_a(self):
        res = optimize_portfolio_allocation([self._opp()], bankroll_usd=1000.0)
        [alloc] = res["allocations"]
        self.assertIsNone(alloc["credibility_grade"])
        self.assertNotEqual(alloc["credibility_grade"], "A")

    def test_a_missing_score_is_none_not_nine_tenths(self):
        res = optimize_portfolio_allocation([self._opp()], bankroll_usd=1000.0)
        [alloc] = res["allocations"]
        self.assertIsNone(alloc["credibility_score"])
        self.assertNotEqual(alloc["credibility_score"], 0.90)

    def test_a_real_assessment_is_still_reported(self):
        res = optimize_portfolio_allocation(
            [self._opp(credibility_grade="B", credibility_score=0.72)],
            bankroll_usd=1000.0,
        )
        [alloc] = res["allocations"]
        self.assertEqual(alloc["credibility_grade"], "B")
        self.assertAlmostEqual(alloc["credibility_score"], 0.72)

    def test_an_unassessed_market_is_still_admitted(self):
        """Scope: this PR relabels, it does not change what is admitted."""
        res = optimize_portfolio_allocation([self._opp()], bankroll_usd=1000.0)
        self.assertEqual(len(res["allocations"]), 1)


class ExecutabilityIsVisibleTests(unittest.TestCase):
    """The optimizer read no liquidity signal at all.

    Its largest live allocation -- 1,499.90 of a 10,000 bankroll, at the 15%
    concentration cap -- went to KXNFLRETIRE-MSTAFFORD9-2627, which the edge
    board publishes as market_bid 0.0, market_volume 0.0, discrepancy_status
    "thin_market". Buying 2,830 contracts into a book with no bid is not a
    trade. These fields ride on the same row the optimizer already reads.
    """

    _THIN = {
        "question": "Thin market",
        "platform": "Kalshi",
        "model_probability": 0.03,
        "market_probability": 0.465,
        "credibility_grade": "A",
        "credibility_score": 0.90,
        "discrepancy_status": "thin_market",
        "market_bid": 0.0,
        "market_ask": 0.93,
        "market_volume": 0.0,
        "market_liquidity": None,
    }

    def test_the_thin_market_flag_reaches_the_caller(self):
        res = optimize_portfolio_allocation([dict(self._THIN)], bankroll_usd=10_000.0)
        [alloc] = res["allocations"]
        self.assertEqual(alloc["discrepancy_status"], "thin_market")
        self.assertEqual(alloc["market_bid"], 0.0)
        self.assertEqual(alloc["market_volume"], 0.0)

    def test_a_caller_can_now_separate_tradeable_from_not(self):
        deep = dict(self._THIN, question="Deep market", market_bid=0.52,
                    market_volume=250_000.0, discrepancy_status=None)
        res = optimize_portfolio_allocation(
            [dict(self._THIN), deep], bankroll_usd=10_000.0
        )
        untradeable = [
            a for a in res["allocations"]
            if a["discrepancy_status"] == "thin_market" or not a["market_bid"]
        ]
        self.assertEqual(len(untradeable), 1)
        self.assertEqual(untradeable[0]["question"], "Thin market")

    def test_absent_liquidity_fields_are_none_not_invented(self):
        res = optimize_portfolio_allocation(
            [{"question": "Q", "platform": "Kalshi",
              "model_probability": 0.70, "market_probability": 0.50}],
            bankroll_usd=1000.0,
        )
        [alloc] = res["allocations"]
        for key in ("discrepancy_status", "market_bid", "market_ask",
                    "market_volume", "market_liquidity"):
            self.assertIsNone(alloc[key], key)

    def test_sizing_is_unchanged_by_the_new_fields(self):
        """Allocations must be identical with and without liquidity data."""
        bare = {"question": "Q", "platform": "Kalshi",
                "model_probability": 0.70, "market_probability": 0.50}
        with_liq = dict(bare, market_bid=0.0, market_volume=0.0,
                        discrepancy_status="thin_market")
        a = optimize_portfolio_allocation([bare], bankroll_usd=1000.0)
        b = optimize_portfolio_allocation([with_liq], bankroll_usd=1000.0)
        self.assertEqual(a["allocated_usd"], b["allocated_usd"])
        self.assertEqual(
            a["allocations"][0]["allocated_pct"], b["allocations"][0]["allocated_pct"]
        )


class AllocationsAreIdentifiableTests(unittest.TestCase):
    """Every allocation came back with ticker "".

    The resolver read `ticker`, `slug`, `id`. A live 26-row edge board
    carries none of them; it carries `ident` on all 26. So a tool telling
    you to place 1,499.90 named no instrument -- only a URL.
    """

    def _opp(self, **over):
        row = {
            "question": "Will X happen?",
            "platform": "Kalshi",
            "ident": "KXNFLRETIRE-MSTAFFORD9-2627",
            "model_probability": 0.70,
            "market_probability": 0.50,
        }
        row.update(over)
        return row

    def test_ident_is_used_when_ticker_is_absent(self):
        res = optimize_portfolio_allocation([self._opp()], bankroll_usd=1000.0)
        self.assertEqual(res["allocations"][0]["ticker"], "KXNFLRETIRE-MSTAFFORD9-2627")

    def test_an_explicit_ticker_still_wins(self):
        res = optimize_portfolio_allocation(
            [self._opp(ticker="KXEXPLICIT-1")], bankroll_usd=1000.0
        )
        self.assertEqual(res["allocations"][0]["ticker"], "KXEXPLICIT-1")

    def test_slug_and_id_remain_fallbacks(self):
        for key, value in (("slug", "some-slug"), ("id", "12345")):
            with self.subTest(key=key):
                opp = self._opp(**{key: value})
                opp.pop("ident")
                res = optimize_portfolio_allocation([opp], bankroll_usd=1000.0)
                self.assertEqual(res["allocations"][0]["ticker"], value)

    def test_nothing_identifying_is_still_empty_not_invented(self):
        opp = self._opp()
        opp.pop("ident")
        res = optimize_portfolio_allocation([opp], bankroll_usd=1000.0)
        self.assertEqual(res["allocations"][0]["ticker"], "")

    def test_no_allocation_is_anonymous_on_board_shaped_rows(self):
        rows = [self._opp(ident=f"KX-{n}", question=f"Q{n}") for n in range(4)]
        res = optimize_portfolio_allocation(rows, bankroll_usd=10_000.0)
        self.assertTrue(all(a["ticker"] for a in res["allocations"]))
