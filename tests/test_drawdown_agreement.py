"""current_drawdown can never exceed max_drawdown.

Both are published in the same equity_curves block on the agent board.
current is how far below its running peak the account sits now; max is the
worst that figure has ever been. current is therefore one of the values max
is the maximum of, so `current <= max` holds by construction -- unless the
two are rounded to different precisions.

They were. max_drawdown rounded to 4dp against current_drawdown's 6dp, and
the live board showed four of eight agents with current greater than max.
It was rounding rather than accounting, but anything checking the invariant
saw it break. The fix -- matching both at 6dp -- had nothing holding it:
putting 4dp back passed the whole suite.

This asserts the invariant rather than the constant, so it survives a
change of precision as long as the two stay consistent.

Not covered here: the codebase has four max-drawdown implementations.
crypto_5m has two and both are held; this one and _current_drawdown are
held by the tests below. The fourth is inline in
crypto_kalshi.kalshi_btc_equity and is unpinned -- it reads a JSONL log,
so exercising it needs a fixture rather than a curve, and it is left for
its own change.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.agent_trading_stats import _current_drawdown  # noqa: E402
from analyzing_llm_rationale.track_record_live import (  # noqa: E402
    _sharpe_and_max_drawdown,
)

#: Curves whose drawdown needs more than four decimals to express.
_AWKWARD_CURVES = (
    [10000.0, 12345.0, 9857.6],
    [1000.0, 1337.0, 1071.3],
    [500.0, 733.0, 611.17],
    [10000.0, 10001.0, 9999.7],
)


def _curve(values):
    return [{"account_value": v} for v in values]


class TheInvariantTests(unittest.TestCase):
    def test_current_never_exceeds_max(self):
        for values in _AWKWARD_CURVES:
            with self.subTest(values=values):
                curve = _curve(values)
                current = _current_drawdown(curve)
                worst = _sharpe_and_max_drawdown(curve)["max_drawdown"]
                self.assertIsNotNone(current)
                self.assertIsNotNone(worst)
                self.assertLessEqual(
                    current, worst,
                    f"current {current} exceeds max {worst} on {values}",
                )

    def test_a_curve_ending_at_its_low_reports_the_same_figure_twice(self):
        """When now is the worst it has been, the two must agree exactly."""
        for values in _AWKWARD_CURVES:
            with self.subTest(values=values):
                curve = _curve(values)
                self.assertEqual(
                    _current_drawdown(curve),
                    _sharpe_and_max_drawdown(curve)["max_drawdown"],
                )

    def test_a_recovery_leaves_max_above_current(self):
        """Down then partly back up: max remembers, current does not."""
        curve = _curve([1000.0, 1500.0, 900.0, 1200.0])
        current = _current_drawdown(curve)
        worst = _sharpe_and_max_drawdown(curve)["max_drawdown"]
        self.assertLess(current, worst)
        self.assertAlmostEqual(worst, 0.4, places=6)
        self.assertAlmostEqual(current, 0.2, places=6)

    def test_both_round_to_the_same_precision(self):
        """The mechanism, stated directly.

        At different precisions the invariant breaks on curves like these
        without either function being wrong about the account.
        """
        for values in _AWKWARD_CURVES:
            with self.subTest(values=values):
                curve = _curve(values)
                current = _current_drawdown(curve)
                worst = _sharpe_and_max_drawdown(curve)["max_drawdown"]
                self.assertEqual(round(current, 6), current)
                self.assertEqual(round(worst, 6), worst)


if __name__ == "__main__":
    unittest.main()
