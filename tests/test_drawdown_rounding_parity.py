"""current_drawdown and max_drawdown must be comparable.

They are published side by side in the board's equity_curves block, and
current is by definition one of the values max is the maximum of, so
`current <= max` always holds in arithmetic.

It did not hold on the published board. current_drawdown rounded to 6
places and max_drawdown to 4, so max could round *below* current. Four of
eight agents shipped an impossible pair:

    gemma-4-26b-a4b-it   current 0.007707  max 0.0077
    qwen3-8-27b          current 0.000117  max 0.0001
    glm-5-3-flash        current 0.003125  max 0.0031
    deepseek-v4-flash    current 0.005513  max 0.0055

Rounding, not accounting -- but anything checking the invariant to decide
whether an agent may trade saw it break, and on the four healthiest
agents rather than the two in real drawdown.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.agent_trading_stats import _current_drawdown  # noqa: E402
from analyzing_llm_rationale.track_record_live import _sharpe_and_max_drawdown  # noqa: E402


def _curve(*values: float):
    return [{"account_value": v} for v in values]


class DrawdownRoundingParityTests(unittest.TestCase):
    #: Each reproduces a pair the live board actually published.
    _LIVE_PAIRS = (0.007707, 0.000117, 0.003125, 0.005513)

    def test_the_live_pairs_no_longer_invert(self):
        for drop in self._LIVE_PAIRS:
            with self.subTest(drop=drop):
                curve = _curve(10_000.0, 10_000.0 * (1 - drop))
                current = _current_drawdown(curve)
                maximum = _sharpe_and_max_drawdown(curve)["max_drawdown"]
                self.assertLessEqual(
                    current, maximum,
                    f"current {current} > max {maximum}, which cannot happen",
                )

    def test_both_keep_the_same_number_of_places(self):
        """The invariant holds because neither is coarser than the other."""
        curve = _curve(10_000.0, 9_922.93)
        current = _current_drawdown(curve)
        maximum = _sharpe_and_max_drawdown(curve)["max_drawdown"]
        self.assertEqual(current, maximum)

    def test_a_trough_before_the_end_keeps_max_above_current(self):
        """The case the two numbers exist to distinguish."""
        curve = _curve(10_000.0, 7_000.0, 9_000.0)
        current = _current_drawdown(curve)
        maximum = _sharpe_and_max_drawdown(curve)["max_drawdown"]
        self.assertAlmostEqual(current, 0.1, places=6)
        self.assertAlmostEqual(maximum, 0.3, places=6)
        self.assertLess(current, maximum)

    def test_the_invariant_holds_across_many_shapes(self):
        """Not just the four that happened to be caught."""
        import random

        rng = random.Random(20260907)
        for trial in range(300):
            values = [10_000.0]
            for _ in range(rng.randint(2, 12)):
                values.append(max(1.0, values[-1] * (1 + rng.uniform(-0.09, 0.09))))
            curve = _curve(*values)
            current = _current_drawdown(curve)
            maximum = _sharpe_and_max_drawdown(curve)["max_drawdown"]
            with self.subTest(trial=trial):
                self.assertIsNotNone(current)
                self.assertLessEqual(current, maximum)


if __name__ == "__main__":
    unittest.main()
