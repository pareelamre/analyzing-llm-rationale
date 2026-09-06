from __future__ import annotations

import unittest
from datetime import datetime, timezone

from analyzing_llm_rationale.twin.account import synchronize_account

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def page(items, complete=True):
    return {"complete": complete, "items": items}


def inputs(**updates):
    result = {
        "balances": [page([{"available": "7", "total": "10", "reserved": "3"}])],
        "positions": [page([{"position_id": "position-001", "quantity": "2"}])],
        "orders": [page([{"order_id": "order-001", "client_order_id": "command-001"}])],
        "fills": [page([{"fill_id": "fill-001", "order_id": "order-001"}])],
        "settlements": [page([{"settlement_id": "settlement-001", "amount": "1"}])],
        "local_command_ids": {"command-001"},
    }
    result.update(updates)
    return result


class TwinAccountTests(unittest.TestCase):
    def sync(self, **updates):
        return synchronize_account("scope-001", generation=1, received_at=NOW, **inputs(**updates))

    def test_multi_page_snapshot_deduplicates_immutable_ids_and_is_complete(self):
        result = self.sync(positions=[page([{"position_id": "position-001", "quantity": "2"}]), page([{"position_id": "position-002", "quantity": "1"}])])
        self.assertFalse(result.retained_previous)
        self.assertEqual(len(result.snapshot.positions), 2)
        self.assertEqual(str(result.snapshot.available_cash), "7")

    def test_repeated_fill_conflicting_duplicate_and_partial_page_retain_previous(self):
        good = self.sync().snapshot
        repeated = self.sync(fills=[page([{"fill_id": "fill-001"}, {"fill_id": "fill-001"}])])
        conflict = self.sync(fills=[page([{"fill_id": "fill-001", "quantity": "1"}, {"fill_id": "fill-001", "quantity": "2"}])], previous=good)
        partial = self.sync(orders=[page([], complete=False)], previous=good)
        self.assertEqual(len(repeated.snapshot.fills), 1)
        self.assertTrue(conflict.retained_previous)
        self.assertTrue(partial.retained_previous)

    def test_external_manual_order_marks_drift_without_discarding_complete_inventory(self):
        result = self.sync(orders=[page([{"order_id": "external-order", "client_order_id": "manual-order"}])])
        self.assertTrue(result.snapshot.divergence)
        self.assertEqual(result.snapshot.external_activity_ids, ("manual-order",))

    def test_unavailable_or_inconsistent_balance_never_increases_spending(self):
        good = self.sync().snapshot
        unavailable = self.sync(balances=[page([{"available": None, "total": "10"}])], previous=good)
        inconsistent = self.sync(balances=[page([{"available": "9", "total": "10", "reserved": "2"}])], previous=good)
        self.assertEqual(unavailable.snapshot.available_cash, good.available_cash)
        self.assertEqual(inconsistent.snapshot.available_cash, good.available_cash)

    def test_settlement_correction_and_concurrent_snapshot_activity_are_deterministic(self):
        corrected = self.sync(settlements=[page([{"settlement_id": "settlement-001", "amount": "1"}, {"settlement_id": "settlement-002", "amount": "-0.1"}])])
        concurrent = self.sync(orders=[page([{"order_id": "order-001"}]), page([{"order_id": "order-002"}])])
        self.assertEqual(len(corrected.snapshot.settlements), 2)
        self.assertEqual([row["order_id"] for row in concurrent.snapshot.orders], ["order-001", "order-002"])


if __name__ == "__main__":
    unittest.main()
