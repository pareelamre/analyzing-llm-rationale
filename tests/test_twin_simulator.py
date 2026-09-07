from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin import (
    CommandState,
    Completeness,
    Instrument,
    MarketSnapshot,
    TradeIntent,
)
from analyzing_llm_rationale.twin.risk import RiskResult
from analyzing_llm_rationale.twin.simulator import (
    CapturedBook,
    DepthLevel,
    ShadowAssumptions,
    ShadowVenue,
)
from analyzing_llm_rationale.twin.store import ExecutionCommand

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
INSTRUMENT_ID = "kalshi:demo:KXTEST"


def instrument() -> Instrument:
    return Instrument(
        INSTRUMENT_ID, "kalshi", "demo", "KXTEST", None, None, None,
        "settlement-hash", "politics", "event-001", "cluster-001",
        Decimal(".01"), Decimal("1"), "fee-v1", "cap-v1", "open",
        NOW + timedelta(days=30), NOW + timedelta(days=31), NOW - timedelta(days=1),
    )


def snapshot(*, suffix: str = "001", received_at: datetime | None = None, yes_ask: str = ".40") -> MarketSnapshot:
    received = received_at or NOW - timedelta(seconds=1)
    return MarketSnapshot(
        f"snapshot-{suffix}", INSTRUMENT_ID, received - timedelta(seconds=1), received,
        int(suffix), "captured-rest", Completeness.COMPLETE, 10,
        Decimal(".39"), Decimal(yes_ask), Decimal(".59"), Decimal(".60"),
        "fee-v1", received,
    )


def intent(*, action: str = "BUY_YES", quantity: str = "5", market_version: str = "snapshot-001") -> TradeIntent:
    return TradeIntent(
        "intent-001", "shadow-scope:shadow-account-001", 1, INSTRUMENT_ID,
        action, Decimal(quantity), Decimal(".45") if action.startswith("BUY") else Decimal(".35"),
        "IOC", "forecast-001" if action.startswith("BUY") else None,
        None if action.startswith("BUY") else "policy_exit", "policy-v1", "strategy-v1",
        market_version, Decimal(".02"), Decimal(".01"),
        NOW + timedelta(minutes=5), NOW,
    )


def risk(order: TradeIntent, *, cash: str = "10") -> RiskResult:
    return RiskResult(
        order.quantity, Decimal(cash), Decimal(cash) if order.action.value.startswith("BUY") else Decimal("0"),
        market_snapshot_id=order.market_version,
    )


def command(order: TradeIntent, *, suffix: str = "001") -> ExecutionCommand:
    return ExecutionCommand(
        f"command-{suffix}", order.account_scope_id, order.id, order.intent_hash,
        CommandState.SUBMITTING, f"reservation-{suffix}", f"client-{suffix}", NOW,
        request_fingerprint="request-hash",
    )


def book(*, suffix: str = "001", received_at: datetime | None = None, levels=None) -> CapturedBook:
    snap = snapshot(suffix=suffix, received_at=received_at)
    return CapturedBook(snap, "yes", tuple(levels or (
        DepthLevel(Decimal(".40"), Decimal("2")),
        DepthLevel(Decimal(".42"), Decimal("1")),
    )))


class TwinSimulatorTests(unittest.TestCase):
    def test_preview_submit_partial_fill_and_account_are_hand_calculable(self):
        venue = ShadowVenue(
            account_id="shadow-account-001", seed=7, starting_cash=Decimal("10"),
            assumptions=ShadowAssumptions(fee_rate=Decimal(".01"), latency_ms=75),
        )
        order = intent()
        preview = venue.preview(order, risk(order), instrument(), book(), now=NOW)
        self.assertEqual(preview.planned_quantity, Decimal("3"))
        self.assertEqual(preview.planned_fee, Decimal(".0122"))
        self.assertEqual(preview.planned_cash_delta, Decimal("-1.2322"))

        acknowledgement = venue.submit(command(order), preview, now=NOW)
        order_id = acknowledgement["acknowledgement"]["venue_order_id"]
        receipt = venue.status(order_id)
        self.assertEqual((receipt.status, receipt.filled_quantity, receipt.remaining_quantity), ("partial", Decimal("3"), Decimal("2")))
        account = venue.account(received_at=NOW)
        self.assertEqual(account.available_cash, Decimal("8.7678"))
        self.assertEqual(account.position_basis, Decimal("1.2322"))
        self.assertEqual(account.fees_paid, Decimal(".0122"))
        self.assertEqual(len(account.fills), 1)
        self.assertIn(venue.run.run_id[:12], venue.events()[0].id)
        self.assertEqual(venue.run.simulator_version, "shadow-venue-v3")

    def test_same_seed_inputs_and_event_stream_reproduce_exactly(self):
        def cycle():
            venue = ShadowVenue(
                account_id="shadow-account-001", seed=3,
                assumptions=ShadowAssumptions(fee_rate=Decimal(".01"), adverse_price_ticks=1),
            )
            order = intent()
            preview = venue.preview(order, risk(order), instrument(), book(), now=NOW)
            response = venue.submit(command(order), preview, now=NOW)
            return preview, response, venue.events(), venue.account(received_at=NOW)

        self.assertEqual(cycle(), cycle())

    def test_no_fill_and_adverse_move_use_captured_ask_not_midpoint(self):
        no_fill = ShadowVenue(
            account_id="shadow-account-001", seed=1,
            assumptions=ShadowAssumptions(no_fill_probability=Decimal("1")),
        )
        order = intent()
        preview = no_fill.preview(order, risk(order), instrument(), book(), now=NOW)
        self.assertEqual(preview.planned_fills, ())
        response = no_fill.submit(command(order), preview, now=NOW)
        self.assertEqual(no_fill.status(response["acknowledgement"]["venue_order_id"]).status, "open")

        empty_depth = ShadowVenue(account_id="shadow-empty", seed=1)
        empty_preview = empty_depth.preview(
            order, risk(order), instrument(), CapturedBook(snapshot(), "yes", ()), now=NOW,
        )
        self.assertEqual(empty_preview.planned_fills, ())

        adverse = ShadowVenue(
            account_id="shadow-account-001", seed=1,
            assumptions=ShadowAssumptions(adverse_price_ticks=6),
        )
        moved = adverse.preview(order, risk(order), instrument(), book(), now=NOW)
        self.assertEqual(moved.planned_fills, ())
        with self.assertRaisesRegex(ValueError, "executable snapshot price"):
            adverse.preview(
                order, risk(order), instrument(),
                CapturedBook(snapshot(), "yes", (DepthLevel(Decimal(".395"), Decimal("5")),)),
                now=NOW,
            )

    def test_cancel_fill_race_requires_later_captured_depth_and_cancels_remainder(self):
        venue = ShadowVenue(account_id="shadow-account-001", seed=2, starting_cash=Decimal("10"))
        order = intent()
        preview = venue.preview(order, risk(order), instrument(), book(), now=NOW)
        response = venue.submit(command(order), preview, now=NOW)
        order_id = response["acknowledgement"]["venue_order_id"]
        later = CapturedBook(
            snapshot(suffix="002", received_at=NOW + timedelta(seconds=1)), "yes",
            (DepthLevel(Decimal(".40"), Decimal("1")),),
        )
        cancelled = venue.cancel(order_id, now=NOW + timedelta(seconds=2), race_book=later, instrument=instrument())
        self.assertEqual(cancelled.filled_quantity, Decimal("4"))
        self.assertEqual(cancelled.cancelled_quantity, Decimal("1"))
        self.assertEqual(cancelled.remaining_quantity, Decimal("0"))
        self.assertEqual([event.event_type for event in venue.events()], [
            "order_acknowledged", "fill", "fill_after_cancel_request", "cancelled",
        ])

    def test_delayed_settlement_and_repeated_commands_are_idempotent(self):
        venue = ShadowVenue(account_id="shadow-account-001", seed=2, starting_cash=Decimal("10"))
        order = intent(quantity="3")
        preview = venue.preview(order, risk(order), instrument(), book(), now=NOW)
        first = venue.submit(command(order), preview, now=NOW)
        self.assertEqual(first, venue.submit(command(order), preview, now=NOW + timedelta(seconds=1)))
        order_id = first["acknowledgement"]["venue_order_id"]
        before = venue.account(received_at=NOW).available_cash
        settled = venue.settle(order_id, resolved_outcome="yes", settled_at=NOW + timedelta(days=1))
        self.assertEqual(settled.settled_payout, Decimal("3"))
        self.assertEqual(venue.account(received_at=NOW + timedelta(days=1)).available_cash, before + Decimal("3"))
        self.assertEqual(
            venue.settle(order_id, resolved_outcome="yes", settled_at=NOW + timedelta(days=2)),
            settled,
        )
        self.assertEqual(len(venue.account(received_at=NOW + timedelta(days=2)).settlements), 1)

    def test_sell_preview_reduces_verified_inventory_and_basis(self):
        venue = ShadowVenue(account_id="shadow-account-001", seed=2, starting_cash=Decimal("10"))
        buy = intent(quantity="3")
        buy_preview = venue.preview(buy, risk(buy), instrument(), book(), now=NOW)
        venue.submit(command(buy, suffix="buy"), buy_preview, now=NOW)

        sell = intent(action="SELL_YES", quantity="3", market_version="snapshot-002")
        sell_snapshot = snapshot(suffix="002", received_at=NOW + timedelta(seconds=1))
        sell_book = CapturedBook(
            sell_snapshot, "yes", (DepthLevel(Decimal(".39"), Decimal("3")),),
        )
        sell_preview = venue.preview(
            sell, risk(sell, cash="0"), instrument(), sell_book,
            now=NOW + timedelta(seconds=1),
        )
        self.assertEqual(sell_preview.planned_cash_delta, Decimal("1.17"))
        venue.submit(command(sell, suffix="sell"), sell_preview, now=NOW + timedelta(seconds=1))
        updated = venue.account(received_at=NOW + timedelta(seconds=1))
        self.assertEqual(updated.holdings, ())
        self.assertEqual(updated.position_basis, Decimal("0"))
        self.assertEqual(updated.available_cash, Decimal("9.95"))

    def test_shadow_identifiers_and_historical_snapshot_binding_are_strict(self):
        with self.assertRaisesRegex(ValueError, "start with shadow"):
            ShadowVenue(account_id="live-account", seed=1)
        venue = ShadowVenue(account_id="shadow-account-001", seed=1)
        order = intent()
        with self.assertRaisesRegex(ValueError, "validated market version"):
            venue.preview(order, risk(order), instrument(), book(suffix="002"), now=NOW)
        stale = book(received_at=NOW - timedelta(seconds=20))
        with self.assertRaisesRegex(ValueError, "freshness"):
            venue.preview(order, risk(order), instrument(), stale, now=NOW)


if __name__ == "__main__":
    unittest.main()
