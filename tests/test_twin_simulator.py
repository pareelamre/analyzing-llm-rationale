import unittest
from decimal import Decimal

from analyzing_llm_rationale.twin.simulator import ShadowVenue


class TwinSimulatorTests(unittest.TestCase):
    def test_seeded_partial_depth_and_idempotent_order(self):
        venue = ShadowVenue(account_id="shadow-001", seed=1, fee_rate=Decimal(".01"))
        first = venue.submit("order-001", quantity=Decimal("5"), ask=Decimal(".5"), depth=Decimal("3"))
        self.assertEqual(first, venue.submit("order-001", quantity=Decimal("5"), ask=Decimal(".5"), depth=Decimal("3")))
        self.assertLessEqual(first.filled_quantity, Decimal("3"))
        self.assertEqual(venue.cancel("order-001").status, "cancelled")

    def test_depth_cash_cancel_race_and_delayed_settlement_are_accounted_once(self):
        venue = ShadowVenue(
            account_id="shadow-002", seed=7, fee_rate=Decimal(".01"), starting_cash=Decimal("10"),
            adverse_no_fill_probability=Decimal("0"),
        )
        receipt = venue.submit(
            "order-002", quantity=Decimal("10"), ask=Decimal(".5"), depth=Decimal("4"),
            instrument_id="kalshi:shadow:KXTEST", outcome="yes",
        )
        self.assertEqual(receipt.filled_quantity, Decimal("4"))
        cancelled = venue.cancel("order-002", fill_race_quantity=Decimal("2"))
        self.assertEqual(cancelled.filled_quantity, Decimal("6"))
        self.assertEqual(cancelled.status, "cancelled")
        before_settlement = venue.account().cash
        settled = venue.settle("order-002", resolved_outcome="yes")
        self.assertEqual(settled.settled_payout, Decimal("6"))
        self.assertEqual(venue.account().cash, before_settlement + Decimal("6"))
        self.assertEqual(venue.settle("order-002", resolved_outcome="yes"), settled)
        self.assertEqual(venue.account().positions, ())
