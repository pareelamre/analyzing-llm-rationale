"""An order priced exactly at the touch must fill.

``_depth_fill`` decides which book levels are reachable with

    if bid + 1e-9 >= threshold and quantity > 0:

where ``threshold = 1.0 - limit_price``. The tolerance is not decoration.
Cent-denominated prices do not survive that subtraction exactly: with a
limit of 0.41 the threshold is 0.5900000000000001, which is greater than a
resting bid of 0.59. Without the tolerance the level at the limit is
skipped, the order fills zero instead of the resting size, and nothing
reports an error -- it looks like the book was empty.

Mutating the tolerance away left the whole suite green, which is why this
exists. Liquidity is already the binding constraint on the agent board;
dropping a level that is exactly at the limit is the wrong direction to be
wrong in.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402


def _book(bid: str, quantity: str):
    """A Kalshi book with a single resting level on the opposite side."""
    return {"orderbook_fp": {"no_dollars": [[bid, quantity]]}}


class ExactlyAtTheLimitFillsTests(unittest.TestCase):
    def test_a_level_at_the_limit_fills_completely(self):
        with mock.patch.object(market_data, "fetch_kalshi_orderbook",
                               return_value=_book("0.5900", "100")):
            total, vwap = benchmark_tools._depth_fill("KXTEST", "yes", 0.41, 100)

        self.assertEqual(total, 100.0)
        self.assertAlmostEqual(vwap, 0.41)

    def test_every_cent_price_whose_bid_sits_exactly_at_the_limit_fills(self):
        """The property, not one example. 0.41/0.59 is not special."""
        for cents in range(1, 100):
            limit_price = cents / 100.0
            bid = (100 - cents) / 100.0
            with self.subTest(limit=limit_price):
                with mock.patch.object(
                    market_data, "fetch_kalshi_orderbook",
                    return_value=_book(f"{bid:.4f}", "50"),
                ):
                    total, _vwap = benchmark_tools._depth_fill(
                        "KXTEST", "yes", limit_price, 50,
                    )
                self.assertEqual(
                    total, 50.0,
                    f"a bid of {bid:.2f} is exactly at a limit of {limit_price:.2f} "
                    "and must be reachable",
                )

    def test_a_level_worse_than_the_limit_is_still_excluded(self):
        """The tolerance must not swallow a level that is genuinely too dear.

        Checked at every price, not one: a single example one cent out sits
        so close to the boundary that even a tolerance of a whole cent leaves
        it excluded by rounding, so it does not distinguish 1e-9 from 0.01.
        Sweeping the grid does.
        """
        for cents in range(2, 100):
            limit_price = cents / 100.0
            # One cent dearer than the level that sits exactly at the limit.
            bid = (100 - cents - 1) / 100.0
            if bid <= 0:
                continue
            with self.subTest(limit=limit_price, bid=bid):
                with mock.patch.object(
                    market_data, "fetch_kalshi_orderbook",
                    return_value=_book(f"{bid:.4f}", "100"),
                ):
                    total, vwap = benchmark_tools._depth_fill(
                        "KXTEST", "yes", limit_price, 100,
                    )
                self.assertEqual(
                    total, 0.0,
                    f"a bid of {bid:.2f} is past a limit of {limit_price:.2f} "
                    "and must not be reachable",
                )
                self.assertIsNone(vwap)


class TheToleranceIsLoadBearingTests(unittest.TestCase):
    """Why 1e-9 is there at all, stated as arithmetic rather than a comment."""

    def _pairs_needing_the_tolerance(self):
        pairs = []
        for cents in range(1, 100):
            limit_price = cents / 100.0
            bid = (100 - cents) / 100.0
            threshold = 1.0 - limit_price
            if (bid + 1e-9 >= threshold) != (bid >= threshold):
                pairs.append((bid, limit_price))
        return pairs

    def test_exact_comparison_would_drop_real_prices(self):
        pairs = self._pairs_needing_the_tolerance()
        self.assertGreater(
            len(pairs), 0,
            "if this is ever empty the tolerance may be removable -- check "
            "the arithmetic before deleting it",
        )
        self.assertIn((0.59, 0.41), pairs)

    def test_the_dropped_side_is_always_the_reachable_one(self):
        """Every disagreement is the exact test wrongly excluding a level.

        The tolerance only ever admits levels; it never admits one the exact
        comparison would have rejected as too expensive.
        """
        for bid, limit_price in self._pairs_needing_the_tolerance():
            with self.subTest(bid=bid, limit=limit_price):
                self.assertTrue(bid + 1e-9 >= 1.0 - limit_price)
                self.assertFalse(bid >= 1.0 - limit_price)
                # The level is at the limit to the cent, not beyond it.
                self.assertAlmostEqual(bid, 1.0 - limit_price, places=9)


if __name__ == "__main__":
    unittest.main()
