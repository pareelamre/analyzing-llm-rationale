from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin import (
    AccountScope,
    CommandState,
    InMemoryTwinStore,
    InsufficientReservationCapacity,
    TradeIntent,
    TwinStoreError,
    require_durable_store,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def scope() -> AccountScope:
    return AccountScope("scope-001", "owner-001", "kalshi", "account-ref", "demo", "USD", "connection-001", 1, NOW)


def intent(*, quantity: str = "1", suffix: str = "001", instrument_id: str = "kalshi:demo:KXTEST") -> TradeIntent:
    return TradeIntent(
        id=f"intent-{suffix}", account_scope_id="scope-001", account_epoch=1,
        instrument_id=instrument_id, action="BUY_YES", quantity=Decimal(quantity),
        limit_price=Decimal("0.40"), time_in_force="IOC", forecast_id="forecast-001", exit_reason=None,
        policy_version="policy-v1", strategy_version="strategy-v1", market_version="market-v1",
        fee_allowance=Decimal("0.01"), slippage_allowance=Decimal("0.01"),
        expires_at=NOW + timedelta(days=1), created_at=NOW,
    )


class TwinStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryTwinStore()
        self.store.register_account(scope(), venue_available_cash=Decimal("10"), loss_limit=Decimal("6"))

    def test_event_deduplication_advances_projection_once(self):
        first = self.store.append_event("scope-001", event_id="event-001", event_type="market_seen", payload={"v": 1}, occurred_at=NOW, observed_at=NOW)
        repeat = self.store.append_event("scope-001", event_id="event-001", event_type="market_seen", payload={"v": 2}, occurred_at=NOW, observed_at=NOW)
        self.assertEqual(first, repeat)
        self.assertEqual(self.store.projection("scope-001").revision, 1)

    def test_reservation_is_idempotent_and_does_not_double_count_venue_holds(self):
        created = self.store.reserve_intent(intent(), cash=Decimal("4"), max_loss=Decimal("3"), now=NOW)
        repeat = self.store.reserve_intent(intent(), cash=Decimal("9"), max_loss=Decimal("5"), now=NOW)
        projection = self.store.projection("scope-001")
        self.assertEqual(created, repeat)
        self.assertEqual(projection.reserved_cash, Decimal("4"))
        self.assertEqual(projection.available_cash_for_reservation, Decimal("6"))
        self.assertEqual(projection.reserved_max_loss, Decimal("3"))

    def test_competing_final_capacity_allows_exactly_one_reservation(self):
        self.store.reserve_intent(intent(suffix="001"), cash=Decimal("6"), max_loss=Decimal("3"), now=NOW)
        with self.assertRaises(InsufficientReservationCapacity):
            self.store.reserve_intent(
                intent(suffix="002", instrument_id="kalshi:demo:KXOTHER"),
                cash=Decimal("5"), max_loss=Decimal("3"), now=NOW,
            )

    def test_fenced_claim_rejects_stale_worker_after_lease_expiry(self):
        order = intent()
        self.store.reserve_intent(order, cash=Decimal("1"), max_loss=Decimal("1"), now=NOW)
        command = self.store.command_for_intent(order)
        first = self.store.claim_command(command.id, worker_id="worker-a", now=NOW, lease_seconds=5)
        second = self.store.claim_command(command.id, worker_id="worker-b", now=NOW + timedelta(seconds=6), lease_seconds=5)
        self.assertEqual(second.fence, first.fence + 1)
        with self.assertRaisesRegex(TwinStoreError, "stale worker fence"):
            self.store.transition_command(command.id, target=CommandState.SUBMISSION_UNKNOWN, fence=first.fence, worker_id="worker-a")
        updated = self.store.transition_command(command.id, target=CommandState.SUBMISSION_UNKNOWN, fence=second.fence, worker_id="worker-b")
        self.assertEqual(updated.state, CommandState.SUBMISSION_UNKNOWN)

    def test_unknown_reservation_cannot_be_released_by_ttl_cleanup(self):
        order = intent()
        reservation = self.store.reserve_intent(order, cash=Decimal("1"), max_loss=Decimal("1"), now=NOW)
        command = self.store.command_for_intent(order)
        claim = self.store.claim_command(command.id, worker_id="worker", now=NOW)
        self.store.transition_command(command.id, target=CommandState.SUBMISSION_UNKNOWN, fence=claim.fence, worker_id="worker")
        with self.assertRaisesRegex(TwinStoreError, "cannot be released"):
            self.store.release_reservation(reservation.id, confirmed_no_order=False)
        released = self.store.release_reservation(reservation.id, confirmed_no_order=True)
        self.assertEqual(released.state.value, "released")

    def test_outbox_inbox_and_projection_rebuild_are_idempotent(self):
        order = intent()
        self.store.reserve_intent(order, cash=Decimal("2"), max_loss=Decimal("1"), now=NOW)
        command = self.store.command_for_intent(order)
        message = self.store.mark_outbox_delivered(command.id, delivered_at=NOW)
        self.assertEqual(message, self.store.mark_outbox_delivered(command.id, delivered_at=NOW))
        self.assertTrue(self.store.receive_inbox("scope-001", "delivery-001"))
        self.assertFalse(self.store.receive_inbox("scope-001", "delivery-001"))
        original = self.store.projection("scope-001")
        rebuilt = self.store.rebuild_projection("scope-001")
        self.assertEqual(rebuilt.reserved_cash, original.reserved_cash)
        self.assertEqual(rebuilt.reserved_max_loss, original.reserved_max_loss)
        self.assertEqual(rebuilt.revision, original.revision)

    def test_live_mode_refuses_non_durable_store(self):
        self.assertIs(require_durable_store(self.store, live=False), self.store)
        with self.assertRaisesRegex(TwinStoreError, "durable"):
            require_durable_store(self.store, live=True)


if __name__ == "__main__":
    unittest.main()
