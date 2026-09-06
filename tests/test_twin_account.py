from __future__ import annotations

import unittest
from datetime import datetime, timezone

from analyzing_llm_rationale.twin.account import (
    portfolio_pages_from_complete_read,
    synchronize_account,
)
from analyzing_llm_rationale.twin.models import SchemaValidationError
from analyzing_llm_rationale.twin.reconcile import (
    AccountReadError,
    read_complete_collection,
    synchronize_complete_account,
)

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

    def test_incomplete_cursor_and_replayed_generation_retain_prior_complete_snapshot(self):
        good = self.sync().snapshot
        cursor_incomplete = self.sync(orders=[{"complete": True, "has_more": True, "items": []}], previous=good)
        replayed = synchronize_account("scope-001", generation=1, received_at=NOW, previous=good, **inputs())
        self.assertTrue(cursor_incomplete.retained_previous)
        self.assertEqual(cursor_incomplete.snapshot.generation, good.generation)
        self.assertEqual(replayed.issues, ("stale_or_replayed_generation",))

    def test_unknown_fill_marks_account_drift(self):
        result = self.sync(fills=[page([{"fill_id": "external-fill", "client_order_id": "unmanaged-command"}])])
        self.assertTrue(result.snapshot.divergence)
        self.assertEqual(result.snapshot.external_activity_ids, ("unmanaged-command",))

    def test_portfolio_adapter_refuses_display_limited_reads(self):
        with self.assertRaises(SchemaValidationError):
            portfolio_pages_from_complete_read({"balance": {"available": "1"}})
        pages = portfolio_pages_from_complete_read({
            "complete": True,
            "balance": {"available": "7", "total": "10", "reserved": "3"},
            "positions": [{"position_id": "position-001"}],
            "orders": [{"order_id": "order-001"}],
            "fills": [{"fill_id": "fill-001"}],
            "settlements": [{"settlement_id": "settlement-001"}],
        })
        self.assertTrue(pages["balances"][0]["complete"])

    def test_cursor_reader_collects_all_pages_and_rejects_a_repeated_cursor(self):
        pages = {
            None: {"items": [{"position_id": "one"}], "cursor": "page-2", "generation_token": "v1"},
            "page-2": {"items": [{"position_id": "two"}], "cursor": None, "generation_token": "v1"},
        }
        collection = read_complete_collection("positions", lambda cursor: pages[cursor])
        self.assertEqual([row["position_id"] for row in collection.items], ["one", "two"])
        self.assertEqual(collection.pages_read, 2)
        with self.assertRaisesRegex(AccountReadError, "cursor_repeated"):
            read_complete_collection(
                "orders",
                lambda _: {"items": [], "cursor": "same"},
                max_pages=3,
            )
        with self.assertRaisesRegex(AccountReadError, "generation_changed"):
            read_complete_collection(
                "fills",
                lambda cursor: (
                    {"items": [], "cursor": "page-2"}
                    if cursor is None else {"items": [], "cursor": None, "generation_token": "late-version"}
                ),
            )

    def test_failed_second_page_or_changed_generation_keeps_prior_snapshot(self):
        prior = self.sync().snapshot
        complete = {
            "balances": lambda _: {"items": [{"available": "7", "total": "10", "reserved": "3"}], "cursor": None},
            "positions": lambda _: {"items": [{"position_id": "position-001", "quantity": "2"}], "cursor": None},
            "orders": lambda cursor: (
                {"items": [{"order_id": "order-001"}], "cursor": "more", "generation_token": "v1"}
                if cursor is None else (_ for _ in ()).throw(RuntimeError("timeout"))
            ),
            "fills": lambda _: {"items": [{"fill_id": "fill-001"}], "cursor": None},
            "settlements": lambda _: {"items": [{"settlement_id": "settlement-001"}], "cursor": None},
        }
        failed = synchronize_complete_account(
            "scope-001", generation=2, received_at=NOW, fetchers=complete,
            local_command_ids={"command-001"}, previous=prior,
        )
        self.assertTrue(failed.retained_previous)
        self.assertEqual(failed.snapshot, prior)
        complete["orders"] = lambda cursor: (
            {"items": [{"order_id": "order-001"}], "cursor": "more", "generation_token": "v1"}
            if cursor is None else {"items": [{"order_id": "order-002"}], "cursor": None, "generation_token": "v2"}
        )
        changed = synchronize_complete_account(
            "scope-001", generation=2, received_at=NOW, fetchers=complete,
            local_command_ids={"command-001"}, previous=prior,
        )
        self.assertTrue(changed.retained_previous)
        self.assertIn("orders_generation_changed", changed.issues)


if __name__ == "__main__":
    unittest.main()
