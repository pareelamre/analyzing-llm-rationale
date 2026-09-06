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
    cursor_page_fetcher,
    offset_page_fetcher,
    read_complete_collection,
    synchronize_complete_account,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def page(items, complete=True):
    return {"complete": complete, "items": items}


def inputs(**updates):
    result = {
        "balances": [page([{"available": "7", "total": "10", "reserved": "3", "settled_cash": "10"}])],
        "positions": [page([{"position_id": "position-001", "quantity": "2", "average_price": "0.4", "liquidation_value": "0.6"}])],
        "orders": [page([{"order_id": "order-001", "client_order_id": "command-001"}])],
        "fills": [page([{"fill_id": "fill-001", "order_id": "order-001", "fee": "0.1"}])],
        "settlements": [page([{"settlement_id": "settlement-001", "amount": "1", "fee": "0", "status": "settled"}])],
        "local_command_ids": {"command-001"},
    }
    result.update(updates)
    return result


class TwinAccountTests(unittest.TestCase):
    def sync(self, generation=1, **updates):
        return synchronize_account("scope-001", generation=generation, received_at=NOW, **inputs(**updates))

    def test_multi_page_snapshot_deduplicates_immutable_ids_and_is_complete(self):
        result = self.sync(positions=[page([{"position_id": "position-001", "quantity": "2", "average_price": "0.4", "liquidation_value": "0.6"}]), page([{"position_id": "position-002", "quantity": "1", "basis": "0.2", "liquidation_value": "0.1"}])])
        self.assertFalse(result.retained_previous)
        self.assertEqual(len(result.snapshot.positions), 2)
        self.assertEqual(str(result.snapshot.available_cash), "7")
        self.assertEqual(str(result.snapshot.position_basis), "1.0")
        self.assertEqual(str(result.snapshot.conservative_liquidation_value), "10.7")

    def test_repeated_fill_conflicting_duplicate_and_partial_page_retain_previous(self):
        good = self.sync().snapshot
        repeated = self.sync(fills=[page([{"fill_id": "fill-001", "fee": "0.1"}, {"fill_id": "fill-001", "fee": "0.1"}])])
        conflict = self.sync(fills=[page([{"fill_id": "fill-001", "quantity": "1", "fee": "0.1"}, {"fill_id": "fill-001", "quantity": "2", "fee": "0.1"}])], previous=good)
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
        unavailable = self.sync(balances=[page([{"available": None, "total": "10", "reserved": "3", "settled_cash": "10"}])], previous=good)
        inconsistent = self.sync(balances=[page([{"available": "9", "total": "10", "reserved": "2", "settled_cash": "10"}])], previous=good)
        self.assertEqual(unavailable.snapshot.available_cash, good.available_cash)
        self.assertEqual(inconsistent.snapshot.available_cash, good.available_cash)

    def test_settlement_correction_and_concurrent_snapshot_activity_are_deterministic(self):
        corrected = self.sync(settlements=[page([{"settlement_id": "settlement-001", "amount": "1", "fee": "0", "status": "settled"}, {"settlement_id": "settlement-002", "amount": "-0.1", "fee": "0.02", "status": "final"}])])
        concurrent = self.sync(orders=[page([{"order_id": "order-001"}]), page([{"order_id": "order-002"}])])
        self.assertEqual(len(corrected.snapshot.settlements), 2)
        self.assertEqual(str(corrected.snapshot.fees_paid), "0.12")
        self.assertEqual([row["order_id"] for row in concurrent.snapshot.orders], ["order-001", "order-002"])

    def test_economics_are_stable_on_a_new_complete_generation(self):
        original = self.sync().snapshot
        reimported = synchronize_account(
            "scope-001", generation=2, received_at=NOW, previous=original, **inputs()
        )
        self.assertFalse(reimported.retained_previous)
        self.assertEqual(reimported.snapshot.holdings, original.holdings)
        self.assertEqual(reimported.snapshot.position_basis, original.position_basis)
        self.assertEqual(reimported.snapshot.fees_paid, original.fees_paid)
        self.assertEqual(
            reimported.snapshot.conservative_liquidation_value,
            original.conservative_liquidation_value,
        )

    def test_unknown_position_economics_or_nonimmutable_settlement_retain_prior(self):
        good = self.sync().snapshot
        missing_liquidation = self.sync(
            generation=2,
            positions=[page([{"position_id": "position-001", "quantity": "2", "average_price": "0.4"}])],
            previous=good,
        )
        anonymous_settlement = self.sync(
            generation=2,
            settlements=[page([{"ticker": "TICKER", "fee": "0"}])], previous=good
        )
        self.assertTrue(missing_liquidation.retained_previous)
        self.assertEqual(missing_liquidation.issues, ("position_economics_unavailable",))
        self.assertTrue(anonymous_settlement.retained_previous)
        self.assertEqual(anonymous_settlement.issues, ("settlements_missing_immutable_id",))

    def test_missing_reserved_cash_or_provisional_settlement_retain_prior(self):
        good = self.sync().snapshot
        missing_reserved = self.sync(
            generation=2,
            balances=[page([{"available": "7", "total": "10", "settled_cash": "10"}])],
            previous=good,
        )
        provisional = self.sync(
            generation=2,
            settlements=[page([{"settlement_id": "settlement-001", "fee": "0", "status": "pending"}])],
            previous=good,
        )
        self.assertEqual(missing_reserved.issues, ("balance_unavailable_or_inconsistent",))
        self.assertEqual(provisional.issues, ("settlement_not_final",))

    def test_incomplete_cursor_and_replayed_generation_retain_prior_complete_snapshot(self):
        good = self.sync().snapshot
        cursor_incomplete = self.sync(orders=[{"complete": True, "has_more": True, "items": []}], previous=good)
        replayed = synchronize_account("scope-001", generation=1, received_at=NOW, previous=good, **inputs())
        self.assertTrue(cursor_incomplete.retained_previous)
        self.assertEqual(cursor_incomplete.snapshot.generation, good.generation)
        self.assertEqual(replayed.issues, ("stale_or_replayed_generation",))

    def test_unknown_fill_marks_account_drift(self):
        result = self.sync(fills=[page([{"fill_id": "external-fill", "client_order_id": "unmanaged-command", "fee": "0"}])])
        self.assertTrue(result.snapshot.divergence)
        self.assertEqual(result.snapshot.external_activity_ids, ("unmanaged-command",))

    def test_portfolio_adapter_refuses_display_limited_reads(self):
        with self.assertRaises(SchemaValidationError):
            portfolio_pages_from_complete_read({"balance": {"available": "1"}})
        pages = portfolio_pages_from_complete_read({
            "complete": True,
            "balance": {"available": "7", "total": "10", "reserved": "3", "settled_cash": "10"},
            "positions": [{"position_id": "position-001", "quantity": "2", "average_price": "0.4", "liquidation_value": "0.6"}],
            "orders": [{"order_id": "order-001"}],
            "fills": [{"fill_id": "fill-001", "fee": "0"}],
            "settlements": [{"settlement_id": "settlement-001", "fee": "0", "status": "settled"}],
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

    def test_venue_cursor_and_offset_adapters_preserve_continuations(self):
        calls = []

        def reader(venue, operation, parameters, **kwargs):
            calls.append((venue, operation, dict(parameters), kwargs))
            if operation == "orders":
                if parameters.get("next_cursor") == "page-2":
                    return {"data": {"orders": [{"order_id": "two"}]}, "next_cursor": None}
                return {"data": {"orders": [{"order_id": "one"}]}, "next_cursor": "page-2"}
            if parameters["offset"] == 0:
                return {"data": [{"token_id": "one"}], "next_offset": 2}
            return {"data": [{"token_id": "two"}], "next_offset": None}

        orders = read_complete_collection(
            "orders",
            cursor_page_fetcher(
                "polymarket", "orders", item_key="orders", reader=reader, creds={"api": "x"},
                cursor_parameter="next_cursor",
            ),
        )
        positions = read_complete_collection(
            "positions",
            offset_page_fetcher("polymarket", "positions", reader=reader, creds={"api": "x"}, limit=2),
        )
        self.assertEqual([row["order_id"] for row in orders.items], ["one", "two"])
        self.assertEqual([row["token_id"] for row in positions.items], ["one", "two"])
        self.assertEqual(calls[1][2]["next_cursor"], "page-2")
        self.assertEqual(calls[3][2]["offset"], 2)
        self.assertEqual(calls[0][3]["access"], "account")

    def test_adapter_refuses_a_venue_pagination_cap_as_complete_inventory(self):
        fetch = offset_page_fetcher(
            "polymarket", "positions", reader=lambda *_args, **_kwargs: {
                "data": [], "next_offset": None, "pagination_limit_reached": True,
            }, creds={"api": "x"}, limit=500,
        )
        with self.assertRaisesRegex(AccountReadError, "pagination_limit_reached"):
            read_complete_collection("positions", fetch)


if __name__ == "__main__":
    unittest.main()
