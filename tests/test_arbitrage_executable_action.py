"""The numbers a reader would actually trade on.

test_arbitrage_thresholds pins whether an opportunity is reported. This
pins what the opportunity says once it is. Two lines survived the whole
2,052-test suite:

    "short_price_no": round(1.0 - sell_price, 4)  ->  1.0 + sell_price
    ... if total_entry_cost > 0 else 0.0          ->  >= 0

Both are in executable_action, which is the part of the payload that
names a price to pay.

The arithmetic is worth stating because it is not arbitrary. The trade
is: buy YES on the cheap venue at buy_price, buy NO on the dear one at
1 - sell_price. Exactly one of those two legs pays $1, so

    net_cost == long_price + short_price_no == 1 - spread

Both identities are asserted rather than the constants, so they survive
a change of rounding while still failing the moment a sign flips. Under
`1.0 + sell_price` the short leg costs 1.55 on a market priced 0.55 --
above the $1 the pair can ever return -- and neither identity holds.

The division guard is reachable, not defensive decoration: a pair priced
0.00 against 1.00 has a spread of 1.0 and therefore a net cost of
exactly zero, so `>= 0` divides by it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.arbitrage_scanner import (  # noqa: E402
    scan_cross_venue_arbitrage,
)

#: Seven shared tokens out of twenty: an overlap of 0.35, the default floor.
_SHARED = "alpha bravo charlie delta echo foxtrot golf"
_TITLE = _SHARED + " hotel india juliet kilo lima mike"


def scan(p_prob, k_prob, **kwargs):
    return scan_cross_venue_arbitrage(
        polymarket_markets=[{"question": _TITLE, "probability": p_prob}],
        kalshi_markets=[{"question": _TITLE, "probability": k_prob}],
        **kwargs,
    )


class ExecutableActionTests(unittest.TestCase):
    def _action(self, p_prob, k_prob):
        found = scan(p_prob, k_prob)
        self.assertEqual(len(found), 1, "expected one opportunity")
        return found[0], found[0]["executable_action"]

    def test_the_two_legs_add_up_to_the_quoted_net_cost(self):
        _opportunity, action = self._action(0.30, 0.45)
        self.assertAlmostEqual(
            action["long_price"] + action["short_price_no"],
            action["net_cost"],
            places=4,
        )

    def test_the_net_cost_is_a_dollar_less_the_spread(self):
        """One leg always pays $1, so the edge is what you did not pay."""
        opportunity, action = self._action(0.30, 0.45)
        self.assertAlmostEqual(
            action["net_cost"], 1.0 - opportunity["spread"], places=4
        )

    def test_the_short_leg_is_a_price_not_a_sum(self):
        """1.0 + sell_price would cost more than the pair can return."""
        _opportunity, action = self._action(0.30, 0.45)
        self.assertAlmostEqual(action["short_price_no"], 0.55, places=4)
        self.assertGreaterEqual(action["short_price_no"], 0.0)
        self.assertLessEqual(action["short_price_no"], 1.0)

    def test_the_long_leg_is_the_cheaper_venue_whichever_it_is(self):
        _o, cheap_on_poly = self._action(0.30, 0.45)
        self.assertEqual(cheap_on_poly["long_venue"], "Polymarket")
        self.assertAlmostEqual(cheap_on_poly["long_price"], 0.30)

        _o, cheap_on_kalshi = self._action(0.45, 0.30)
        self.assertEqual(cheap_on_kalshi["long_venue"], "Kalshi")
        self.assertAlmostEqual(cheap_on_kalshi["long_price"], 0.30)

    def test_the_identities_hold_from_either_side(self):
        for p_prob, k_prob in ((0.30, 0.45), (0.45, 0.30), (0.05, 0.95)):
            opportunity, action = self._action(p_prob, k_prob)
            with self.subTest(p=p_prob, k=k_prob):
                self.assertAlmostEqual(
                    action["long_price"] + action["short_price_no"],
                    action["net_cost"],
                    places=4,
                )
                self.assertAlmostEqual(
                    action["net_cost"], 1.0 - opportunity["spread"], places=4
                )


class CostlessPairTests(unittest.TestCase):
    """A market at 0.00 against one at 1.00 costs nothing to enter."""

    def test_a_zero_net_cost_does_not_divide(self):
        opportunity, action = None, None
        found = scan(0.0, 1.0)
        self.assertEqual(len(found), 1)
        opportunity, action = found[0], found[0]["executable_action"]
        self.assertAlmostEqual(opportunity["spread"], 1.0)
        self.assertAlmostEqual(action["net_cost"], 0.0)

    def test_the_return_on_nothing_is_reported_as_zero(self):
        """Recorded, not endorsed.

        Entering at no cost is an unbounded return, and 0.0 is a sentinel
        standing in for "cannot divide". It reads as no return at all.
        Nothing downstream is misled today because the board sorts by
        spread rather than by this figure, so the pair still lands first.
        """
        self.assertEqual(scan(0.0, 1.0)[0]["gross_roi_pct"], 0.0)

    def test_an_ordinary_pair_reports_a_real_return(self):
        opportunity = scan(0.30, 0.45)[0]
        self.assertAlmostEqual(
            opportunity["gross_roi_pct"],
            round(opportunity["spread"] / opportunity["executable_action"]["net_cost"] * 100, 2),
            places=2,
        )
        self.assertGreater(opportunity["gross_roi_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
