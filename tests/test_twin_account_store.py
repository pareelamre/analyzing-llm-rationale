from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timezone

from analyzing_llm_rationale.twin.account import synchronize_account
from analyzing_llm_rationale.twin.account_store import (
    AccountSnapshotStoreError,
    InMemoryAccountSnapshotStore,
    _canonical_json,
    _restore,
)
from analyzing_llm_rationale.twin.models import Completeness

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def page(items):
    return {"complete": True, "items": items}


def snapshot(*, generation=1, received_at=NOW):
    result = synchronize_account(
        "scope-001", generation=generation, received_at=received_at,
        balances=[page([{"available": "7", "total": "10", "reserved": "3", "settled_cash": "10"}])],
        positions=[page([{"position_id": "position-001", "quantity": "2", "average_price": "0.4", "liquidation_value": "0.6"}])],
        orders=[page([{"order_id": "order-001"}])], fills=[page([{"fill_id": "fill-001", "fee": "0.1"}])],
        settlements=[page([{"settlement_id": "settlement-001", "fee": "0", "status": "settled"}])],
        local_command_ids=set(),
    )
    assert result.snapshot is not None
    return result.snapshot


class AccountSnapshotStoreTests(unittest.TestCase):
    def test_round_trip_preserves_complete_economics(self):
        original = snapshot()
        restored = _restore(__import__("json").loads(_canonical_json(original)))
        self.assertEqual(restored, original)

    def test_restart_reimport_is_idempotent_but_conflict_is_rejected(self):
        store = InMemoryAccountSnapshotStore()
        original = snapshot()
        self.assertEqual(store.save(original), original)
        same_generation = snapshot(received_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
        self.assertEqual(store.save(same_generation), original)
        with self.assertRaisesRegex(AccountSnapshotStoreError, "conflicting economics"):
            store.save(replace(same_generation, available_cash=same_generation.available_cash + 1))

    def test_stale_generation_cannot_replace_inventory_and_incomplete_is_rejected(self):
        store = InMemoryAccountSnapshotStore()
        newest = snapshot(generation=2)
        store.save(newest)
        self.assertEqual(store.save(snapshot(generation=1)), newest)
        with self.assertRaisesRegex(AccountSnapshotStoreError, "only complete"):
            store.save(replace(newest, completeness=Completeness.INCOMPLETE))


if __name__ == "__main__":
    unittest.main()
