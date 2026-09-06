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
