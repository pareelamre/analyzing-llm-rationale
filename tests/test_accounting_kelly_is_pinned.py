"""The Kelly formula behind the published paper-PnL.

_kelly_fraction sizes simulate_validated_kelly_account, which produces the
quarter_kelly and edge_kelly strategy results the track record publishes.
It was unpinned: removing the division by the odds passed the full suite,
all 1,865 tests, as did flipping the sign and inverting the odds.

    f* = (p_win * b - q_win) / b,   b = (1 - p_side) / p_side

Dropping the division is exact at even money and worse as the price falls,
which is the third time this session the same normalisation has been found
unheld -- after the portfolio optimizer (#541) and the crypto strategy
(#562). Two other Kelly implementations, in benchmark_tools and the caller
of this one, were already pinned.

    p_win  p_side   correct   unnormalised  ratio
    0.60   0.50     +0.2000   +0.2000        1.0x
    0.60   0.40     +0.3333   +0.5000        1.5x
    0.60   0.20     +0.5000   +2.0000        4.0x
    0.60   0.10     +0.5556   +5.0000        9.0x

A fraction of 5.0 is five times the bankroll on one bet.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.accounting import _kelly_fraction  # noqa: E402


class TheFormulaTests(unittest.TestCase):
    def test_even_money_with_a_ten_point_edge(self):
        """b = 1, so f* = p - q = 0.20."""
        self.assertAlmostEqual(_kelly_fraction(0.60, 0.50), 0.20)

    def test_a_cheaper_side_sizes_larger(self):
        """b = 1.5 at 0.40: (0.60*1.5 - 0.40)/1.5 = 0.3333."""
        self.assertAlmostEqual(_kelly_fraction(0.60, 0.40), 1.0 / 3.0, places=6)

    def test_a_longshot_is_where_the_division_matters_most(self):
        """b = 9 at 0.10: (0.60*9 - 0.40)/9 = 0.5556, not 5.0."""
        self.assertAlmostEqual(_kelly_fraction(0.60, 0.10), 0.5556, places=4)
        self.assertLess(_kelly_fraction(0.60, 0.10), 1.0)

    def test_it_matches_its_own_documented_formula_across_the_range(self):
        for p_side in [round(x * 0.05, 2) for x in range(1, 20)]:
            for p_win in (0.35, 0.50, 0.65, 0.80):
                with self.subTest(p_win=p_win, p_side=p_side):
                    b = (1.0 - p_side) / p_side
                    self.assertAlmostEqual(
                        _kelly_fraction(p_win, p_side),
                        (p_win * b - (1.0 - p_win)) / b,
                        places=9,
                    )

    def test_no_edge_is_zero(self):
        self.assertAlmostEqual(_kelly_fraction(0.50, 0.50), 0.0)

    def test_a_negative_edge_stays_negative(self):
        """The docstring says it may be negative; callers clamp, not this."""
        self.assertLess(_kelly_fraction(0.40, 0.50), 0.0)
        self.assertAlmostEqual(_kelly_fraction(0.40, 0.50), -0.20)


class TheOddsDerivationTests(unittest.TestCase):
    def test_the_odds_come_from_the_side_actually_held(self):
        """b = (1 - p_side)/p_side. Inverting it changes every size."""
        cheap = _kelly_fraction(0.60, 0.20)
        dear = _kelly_fraction(0.60, 0.80)
        self.assertGreater(cheap, dear)

    def test_a_dear_side_with_the_same_win_probability_sizes_smaller(self):
        """b = 0.25 at 0.80: (0.60*0.25 - 0.40)/0.25 = -1.0, no edge."""
        self.assertLess(_kelly_fraction(0.60, 0.80), 0.0)


class UnusableInputsTests(unittest.TestCase):
    def test_a_degenerate_price_is_refused(self):
        for p_side in (0.0, 1.0, -0.1, 1.5):
            with self.subTest(p_side=p_side):
                self.assertEqual(_kelly_fraction(0.60, p_side), 0.0)

    def test_a_certain_win_still_sizes_below_the_bankroll(self):
        """f* = 1.0 at most: never more than the bankroll on one bet."""
        self.assertLessEqual(_kelly_fraction(1.0, 0.10), 1.0)


if __name__ == "__main__":
    unittest.main()
