"""What decides whether a 5-minute crypto trade happens, and how large.

_strategy_from_edge gates on `abs_edge >= min_edge and expected_value > 0`
and sizes with a normalised Kelly. Three of those pieces were unpinned --
each mutation passed the suite:

    min_edge = max(threshold, fee)        -> threshold
    raw_kelly = (fair-market)/(1-market)  -> (fair-market)
    should_trade = ... and ev > 0.0       -> drop the EV term

Dropping the payoff normalisation is the same shape as the portfolio Kelly
bug in #541: exact at even money and worst at the extremes, which is where
5-minute crypto markets spend much of their time.

    market  normalised  unnormalised  ratio
    0.50    0.1000      0.0500        2.0x
    0.70    0.1667      0.0500        3.3x
    0.90    0.5000      0.0500       10.0x
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.crypto_5m import _strategy_from_edge  # noqa: E402


def _strategy(**over):
    kwargs = {
        "edge": 0.05,
        "probability_up": 0.55,
        "sigma_1m": 0.0005,
        "horizon": 5.0,
        "edge_threshold": 0.03,
        "fee_bps": 0.0,
    }
    kwargs.update(over)
    return _strategy_from_edge(**kwargs)


class TheFeeFloorsTheEdgeGateTests(unittest.TestCase):
    """A trade must clear the fee, whatever the configured threshold says."""

    def test_the_gate_is_the_fee_when_it_exceeds_the_threshold(self):
        result = _strategy(edge=0.02, edge_threshold=0.005, fee_bps=300.0)
        self.assertAlmostEqual(result["min_trade_edge"], 0.03)
        self.assertEqual(result["recommendation"], "hold")

    def test_an_edge_under_the_fee_is_not_traded_even_with_a_low_threshold(self):
        """Without the floor, min_edge would be 0.005 and this would trade."""
        result = _strategy(edge=0.02, edge_threshold=0.005, fee_bps=300.0)
        self.assertEqual(result["max_position_fraction"], 0.0)

    def test_the_threshold_still_governs_when_it_exceeds_the_fee(self):
        result = _strategy(edge=0.02, edge_threshold=0.04, fee_bps=10.0)
        self.assertAlmostEqual(result["min_trade_edge"], 0.04)
        self.assertEqual(result["recommendation"], "hold")


class ZeroExpectedValueIsNotTradedTests(unittest.TestCase):
    def test_an_edge_exactly_equal_to_the_fee_is_held(self):
        """abs_edge >= min_edge passes here; only the EV term refuses it.

        Reachable exactly when threshold and fee coincide with the edge, and
        a trade whose expected value is zero is one worth not taking.
        """
        result = _strategy(edge=0.03, edge_threshold=0.03, fee_bps=300.0)
        self.assertEqual(result["recommendation"], "hold")
        self.assertEqual(result["max_position_fraction"], 0.0)

    def test_an_edge_just_above_the_fee_is_traded(self):
        result = _strategy(edge=0.04, edge_threshold=0.03, fee_bps=300.0)
        self.assertEqual(result["recommendation"], "buy_up")
        self.assertGreater(result["max_position_fraction"], 0.0)


class KellyIsNormalisedByThePayoffTests(unittest.TestCase):
    def test_a_high_priced_market_sizes_off_the_normalised_edge(self):
        """market 0.90, fair 0.95.

        Normalised raw Kelly is (0.95-0.90)/(1-0.90) = 0.50, which after the
        quarter-Kelly and volatility factors still clears the 5% cap.
        Unnormalised it is 0.05, giving 0.0125 -- a quarter of the position.
        """
        result = _strategy(edge=0.05, probability_up=0.95)
        self.assertEqual(result["recommendation"], "buy_up")
        self.assertAlmostEqual(result["kelly_fraction"], 0.05)

    def test_the_same_edge_at_even_money_sizes_smaller(self):
        """The normalisation is what makes these differ; without it the two
        are identical, since the numerator is the same 0.05 edge."""
        high = _strategy(edge=0.05, probability_up=0.95)["kelly_fraction"]
        even = _strategy(edge=0.05, probability_up=0.55)["kelly_fraction"]
        self.assertGreater(high, even)

    def test_the_position_is_capped(self):
        result = _strategy(edge=0.40, probability_up=0.95)
        self.assertLessEqual(result["kelly_fraction"], 0.05)
        self.assertLessEqual(result["max_position_fraction"], 0.05)

    def test_no_edge_means_no_position(self):
        result = _strategy(edge=None)
        self.assertEqual(result["max_position_fraction"], 0.0)
        self.assertEqual(result["confidence"], "none")


class ConfidenceBandsTests(unittest.TestCase):
    """Labels, not sizing -- but they are published beside the position."""

    def test_below_the_gate_is_low(self):
        self.assertEqual(_strategy(edge=0.01)["confidence"], "low")

    def test_between_the_gate_and_eight_points_is_medium(self):
        self.assertEqual(_strategy(edge=0.05)["confidence"], "medium")

    def test_at_or_above_eight_points_is_high(self):
        self.assertEqual(_strategy(edge=0.08)["confidence"], "high")

    def test_the_band_edges_are_inclusive_the_way_the_gate_is(self):
        """An edge exactly at the gate is tradable, so it is not 'low'."""
        self.assertEqual(_strategy(edge=0.03, edge_threshold=0.03)["confidence"], "medium")


if __name__ == "__main__":
    unittest.main()
