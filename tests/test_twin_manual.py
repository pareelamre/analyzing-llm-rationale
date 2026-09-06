from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin import (
    InMemoryTwinStore,
    InsufficientReservationCapacity,
    TradeIntent,
)
from analyzing_llm_rationale.twin.manual import (
    ManualReservationConflict,
    reserve_confirmed_manual_order,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def guardrails(*, available: str = "10", loss: str = "10"):
    return {
        "portfolio": {"available": available},
        "policy": {"max_daily_risk_notional": loss},
        "quote": {"market_ident": "KXTEST", "fetched_at": NOW.isoformat()},
    }


def normalized(*, action: str = "buy", outcome: str = "yes", quantity: str = "2", price: str = "0.5"):
    return {
        "platform": "kalshi",
        "action": action,
        "outcome": outcome,
        "ticker": "KXTEST",
        "quantity": quantity,
        "price": price,
        "time_in_force": "immediate_or_cancel",
    }


class ManualTwinBoundaryTests(unittest.TestCase):
    def test_manual_and_autonomous_intents_contend_for_one_account_capacity(self):
        store = InMemoryTwinStore()
        claim = reserve_confirmed_manual_order(
            store,
            user_id="user-001",
            venue="kalshi",
            payload={"client_order_id": "manual-001"},
            normalized=normalized(quantity="12", price="0.5"),
            guardrails=guardrails(),
            authority_ref="direct-manual-001",
            now=NOW,
        )
        scope_id = claim.command.scope_id
        self.assertEqual(claim.command.client_order_id, "manual-001")
        self.assertTrue(claim.command.request_fingerprint)
        autonomous = TradeIntent(
            id="autonomous-001",
            account_scope_id=scope_id,
            account_epoch=1,
            instrument_id="kalshi:live:KXOTHER",
            action="BUY_YES",
            quantity=Decimal("10"),
            limit_price=Decimal("0.5"),
            time_in_force="IOC",
            forecast_id="forecast-001",
            exit_reason=None,
            policy_version="policy-v1",
            strategy_version="strategy-v1",
            market_version="market-v1",
            fee_allowance=Decimal("0"),
            slippage_allowance=Decimal("0"),
            expires_at=NOW + timedelta(days=1),
            created_at=NOW,
        )
        with self.assertRaises(InsufficientReservationCapacity):
            store.reserve_intent(autonomous, cash=Decimal("5"), max_loss=Decimal("5"), now=NOW)

    def test_duplicate_manual_authority_never_receives_a_second_claim(self):
        store = InMemoryTwinStore()
        kwargs = {
            "user_id": "user-001",
            "venue": "kalshi",
            "payload": {"client_order_id": "manual-001"},
            "normalized": normalized(),
            "guardrails": guardrails(),
            "authority_ref": "direct-manual-001",
            "now": NOW,
        }
        first = reserve_confirmed_manual_order(store, **kwargs)
        with self.assertRaises(ManualReservationConflict):
            reserve_confirmed_manual_order(store, **kwargs)
        self.assertEqual(store.projection(first.command.scope_id).reserved_cash, Decimal("1.0"))

    def test_manual_sell_reserves_a_command_without_reserving_new_cash_or_loss(self):
        store = InMemoryTwinStore()
        claim = reserve_confirmed_manual_order(
            store,
            user_id="user-001",
            venue="kalshi",
            payload={"client_order_id": "manual-sell-001"},
            normalized=normalized(action="sell", outcome="no"),
            guardrails=guardrails(),
            authority_ref="direct-manual-sell-001",
            now=NOW,
        )
        projection = store.projection(claim.command.scope_id)
        self.assertEqual(projection.reserved_cash, Decimal("0"))
        self.assertEqual(projection.reserved_max_loss, Decimal("0"))


if __name__ == "__main__":
    unittest.main()
