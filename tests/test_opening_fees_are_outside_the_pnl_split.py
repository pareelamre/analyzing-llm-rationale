"""Where an opening fill's fee lands, and where it does not.

`account_value - starting_cash` is the account's truth and reconciles: the
board's total_pnl matches it, and account_value matches cash plus positions
at market. The published *split* of that total into realized_pnl and
unrealized_pnl does not add back up to it.

The reason is exact. On a fill, fee is allocated to realized PnL in
proportion to the quantity that closed existing exposure:

    fee_alloc = fee * (realized_pairs / quantity)

A fill that opens has realized_pairs == 0, so it allocates nothing. Its fee
leaves cash -- and therefore lands in total_pnl -- while appearing in
neither realized_pnl nor unrealized_pnl, which is `mark - cost_basis` and
cost_basis excludes fees.

So:

    (realized_pnl + unrealized_pnl) - (account_value - starting_cash)
        == fees paid on fills that opened exposure

and the split reads *better* than the account, never worse. On the live
board llama-3.3-70b-instruct shows realized+unrealized of -1975.74 against
a total of -2015.36: summing the published parts understates the loss by
39.62.

These tests do not argue for a different convention. They pin the one in
force so it is specified rather than discovered, and so a change to it has
to be deliberate.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.accounting import PredictionMarketAccount  # noqa: E402

_START = 10_000.0
_MARK = {("kalshi", "M1"): {"bid": 0.50, "ask": 0.50}}


def _residual(*fills, quotes=None):
    """(realized + unrealized) - (account_value - starting_cash)."""
    account = PredictionMarketAccount(starting_cash=_START)
    for fill in fills:
        account.buy(platform="kalshi", ident="M1", **fill)
    snap = account.snapshot(quotes if quotes is not None else _MARK)
    total = snap["account_value"] - _START
    split = snap["realized_pnl"] + snap["unrealized_pnl"]
    return round(split - total, 9), snap


def _open(fee=0.0):
    return dict(side="YES", quantity=10.0, price=0.50, fee=fee)


def _close(fee=0.0):
    return dict(side="NO", quantity=10.0, price=0.50, fee=fee)


class OpeningFeesSitOutsideTheSplitTests(unittest.TestCase):
    def test_no_fee_reconciles_exactly(self):
        residual, _ = _residual(_open())
        self.assertEqual(residual, 0.0)

    def test_an_opening_fee_is_the_whole_residual(self):
        residual, snap = _residual(_open(fee=0.20))
        self.assertEqual(residual, 0.20)
        self.assertEqual(snap["realized_pnl"], 0.0)
        self.assertEqual(snap["fees_paid"], 0.20)

    def test_a_closing_fee_is_allocated_and_reconciles(self):
        """The close path already does the right thing."""
        residual, snap = _residual(_open(), _close(fee=0.20))
        self.assertEqual(residual, 0.0)
        self.assertEqual(snap["realized_pnl"], -0.20)

    def test_only_the_opening_leg_escapes(self):
        residual, snap = _residual(_open(fee=0.20), _close(fee=0.20))
        self.assertEqual(residual, 0.20)
        self.assertEqual(snap["fees_paid"], 0.40)
        self.assertEqual(snap["realized_pnl"], -0.20)

    def test_a_partly_closing_fill_splits_its_fee_in_proportion(self):
        """The case that distinguishes proportional allocation from all-or-nothing.

        Hold YES 4, then buy NO 10: four contracts close, six open. The fee is
        allocated `fee * (realized_pairs / quantity)` = 40% of it, so 60%
        escapes into the opening side. A test that only ever closes a whole
        position cannot tell this formula from `fee_alloc = fee`.
        """
        account = PredictionMarketAccount(starting_cash=_START)
        account.buy(platform="kalshi", ident="M1", side="YES",
                    quantity=4.0, price=0.50, fee=0.0)
        account.buy(platform="kalshi", ident="M1", side="NO",
                    quantity=10.0, price=0.50, fee=0.20)
        snap = account.snapshot(_MARK)

        self.assertAlmostEqual(snap["realized_pnl"], -0.20 * (4 / 10), places=9)
        residual = (snap["realized_pnl"] + snap["unrealized_pnl"]) - (
            snap["account_value"] - _START
        )
        self.assertAlmostEqual(residual, 0.20 * (6 / 10), places=9)

    def test_the_split_never_reads_worse_than_the_account(self):
        """The error flatters performance; it does not understate it."""
        for fills in (
            (_open(fee=0.05),),
            (_open(fee=0.05), _close(fee=0.05)),
            (_open(fee=0.31), _close()),
            (_open(), _close(fee=0.44)),
        ):
            with self.subTest(fills=len(fills)):
                residual, _ = _residual(*fills)
                self.assertGreaterEqual(residual, 0.0)

    def test_the_residual_is_bounded_by_total_fees(self):
        residual, snap = _residual(_open(fee=0.20), _close(fee=0.20))
        self.assertLessEqual(residual, snap["fees_paid"])

    def test_cash_is_charged_the_full_fee_on_either_leg(self):
        """The account is never wrong -- only the published split is."""
        opened = PredictionMarketAccount(starting_cash=_START)
        opened.buy(platform="kalshi", ident="M1", **_open(fee=0.20))
        self.assertAlmostEqual(
            opened.snapshot(_MARK)["cash"], _START - (10.0 * 0.50) - 0.20, places=6
        )

        closed = PredictionMarketAccount(starting_cash=_START)
        closed.buy(platform="kalshi", ident="M1", **_open())
        closed.buy(platform="kalshi", ident="M1", **_close(fee=0.20))
        snap = closed.snapshot(_MARK)
        # Both legs paid 0.50 on a pair worth 1.00, so only the fee is lost.
        self.assertAlmostEqual(snap["account_value"], _START - 0.20, places=6)
        self.assertAlmostEqual(snap["cash"], snap["account_value"], places=6)

if __name__ == "__main__":
    unittest.main()
