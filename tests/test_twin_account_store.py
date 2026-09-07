from __future__ import annotations

import json
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock

from analyzing_llm_rationale.twin.account import synchronize_account
from analyzing_llm_rationale.twin.account_store import (
    AccountSnapshotStoreError,
    DatastoreAccountSnapshotStore,
    InMemoryAccountSnapshotStore,
    _canonical_json,
    _fingerprint,
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
    def test_restore_rejects_corrupted_authority(self):
        for updates in (
            {"available_cash": "999"}, {"generation": -1}, {"generation": True},
            {"reserved_cash": "-1"}, {"position_basis": "999"},
            {"fees_paid": "0"}, {"conservative_liquidation_value": "999"},
            {"divergence": "false"}, {"scope_id": ""},
        ):
            with self.subTest(updates=updates):
                payload = json.loads(_canonical_json(snapshot()))
                payload.update(updates)
                with self.assertRaises(AccountSnapshotStoreError):
                    _restore(payload)

    def test_datastore_load_checks_scope_and_integrity(self):
        original = snapshot()
        entity = {"payload_json": _canonical_json(original),
                  "fingerprint": _fingerprint(original), "generation": original.generation}
        client = Mock()
        client.get.return_value = entity
        store = DatastoreAccountSnapshotStore(client)
        self.assertEqual(store.load(original.scope_id), original)
        with self.assertRaises(AccountSnapshotStoreError):
            store.load("another-scope")
        for updates in ({"fingerprint": "corrupted"}, {"generation": 99}):
            client.get.return_value = {**entity, **updates}
            with self.assertRaises(AccountSnapshotStoreError):
                store.load(original.scope_id)

    def test_new_invalid_snapshot_is_never_saved(self):
        store = InMemoryAccountSnapshotStore()
        with self.assertRaises(AccountSnapshotStoreError):
            store.save(replace(snapshot(), generation=-1))
        self.assertIsNone(store.load("scope-001"))

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
            store.save(replace(same_generation, available_cash=same_generation.available_cash - 1))

    def test_stale_generation_cannot_replace_inventory_and_incomplete_is_rejected(self):
        store = InMemoryAccountSnapshotStore()
        newest = snapshot(generation=2)
        store.save(newest)
        self.assertEqual(store.save(snapshot(generation=1)), newest)
        with self.assertRaisesRegex(AccountSnapshotStoreError, "only complete"):
            store.save(replace(newest, completeness=Completeness.INCOMPLETE))


if __name__ == "__main__":
    unittest.main()
