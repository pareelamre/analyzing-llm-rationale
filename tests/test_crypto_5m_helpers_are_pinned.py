"""Two helpers every crypto_5m signal passes through, and one that looks unpinned but is not.

A mutation sweep of lines 419-460 left three survivors against
tests.test_crypto_5m, and two of them also survive the full suite.

_normalize_market_probability accepts either a fraction or a percentage
and tells them apart with `p > 1.0`. As `p >= 1.0` a market priced at
certainty is read as a percentage and divided by a hundred: 1.0 becomes
0.01. The interesting value is the only one where the two forms
disagree, and no fixture used it.

_robust_z divides by `max(scale, 1e-9)`, a floor that stops a flat
market dividing by zero. As `min(scale, 1e-9)` the floor becomes a
ceiling, the divisor is never more than 1e-9, and every z-score clips to
+/-4. It feeds momentum_3m_z, momentum_5m_z, momentum_15m_z and
reversal_1m_z, so all four features would saturate at once. The same
shape -- a cap written as min() that could be max() -- turned up in
twin/risk, portfolio_optimizer and benchmark_tools this week.

The third survivor, _sigmoid's `x >= 0` against `x > 0`, is recorded
here as equivalent rather than left to be rediscovered. The branch is for
numerical stability, and at exactly zero both halves return 0.5 to the
bit, so no input can tell the two forms apart.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.crypto_5m import (  # noqa: E402
    _normalize_market_probability,
    _robust_z,
    _sigmoid,
)


class MarketProbabilityTests(unittest.TestCase):
    def test_certainty_is_a_fraction_not_a_percentage(self):
        """The one value where `>` and `>=` disagree."""
        self.assertEqual(_normalize_market_probability(1.0), 1.0)

    def test_just_below_certainty_is_left_alone(self):
        self.assertEqual(_normalize_market_probability(0.99), 0.99)

    def test_a_percentage_is_scaled_down(self):
        self.assertAlmostEqual(_normalize_market_probability(55.0), 0.55)
        self.assertAlmostEqual(_normalize_market_probability(100.0), 1.0)

    def test_just_above_one_is_read_as_a_percentage(self):
        self.assertAlmostEqual(_normalize_market_probability(1.01), 0.0101)

    def test_out_of_range_values_are_clamped(self):
        self.assertEqual(_normalize_market_probability(-0.2), 0.0)
        self.assertEqual(_normalize_market_probability(250.0), 1.0)

    def test_missing_stays_missing(self):
        self.assertIsNone(_normalize_market_probability(None))


class RobustZTests(unittest.TestCase):
    def test_an_ordinary_move_is_scaled_not_saturated(self):
        self.assertAlmostEqual(_robust_z(0.001, 0.002), 0.5)
        self.assertAlmostEqual(_robust_z(-0.001, 0.01), -0.1)

    def test_distinct_moves_give_distinct_scores(self):
        """Under the inverted floor every one of these reads 4.0."""
        scores = {round(_robust_z(v, 0.01), 6) for v in (0.001, 0.002, 0.005)}
        self.assertEqual(len(scores), 3)
        self.assertLess(max(scores), 4.0)

    def test_large_moves_are_clipped_to_four(self):
        self.assertEqual(_robust_z(1.0, 0.01), 4.0)
        self.assertEqual(_robust_z(-1.0, 0.01), -4.0)

    def test_a_flat_market_does_not_divide_by_zero(self):
        """What the floor is for."""
        self.assertEqual(_robust_z(0.0, 0.0), 0.0)
        self.assertEqual(_robust_z(0.001, 0.0), 4.0)


class SigmoidEquivalenceTests(unittest.TestCase):
    """Why `x >= 0` vs `x > 0` is not worth a test of its own."""

    def test_both_branches_agree_exactly_at_zero(self):
        upper = 1.0 / (1.0 + math.exp(-0.0))
        lower = math.exp(0.0) / (1.0 + math.exp(0.0))
        self.assertEqual(upper, lower)
        self.assertEqual(_sigmoid(0.0), 0.5)

    def test_it_is_stable_at_the_extremes_either_side(self):
        self.assertAlmostEqual(_sigmoid(800.0), 1.0)
        self.assertAlmostEqual(_sigmoid(-800.0), 0.0)


if __name__ == "__main__":
    unittest.main()
