"""Datastore-emulator integration checks for account reservation contention.

Run with:
  DATASTORE_EMULATOR_HOST=127.0.0.1:8765 GOOGLE_CLOUD_PROJECT=foresea-twin-test \
  PYTHONPATH=src py -m unittest tests.test_twin_store_integration
"""
from __future__ import annotations

import multiprocessing
import os
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from analyzing_llm_rationale.twin import AccountScope, TradeIntent
from analyzing_llm_rationale.twin.manual import reserve_confirmed_manual_order
from analyzing_llm_rationale.twin.store import DatastoreTwinStore, InsufficientReservationCapacity


def _reserve_in_process(scope_id: str, intent_id: str, instrument_id: str, result_queue: multiprocessing.Queue) -> None:
    from google.cloud import datastore

    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    store = DatastoreTwinStore(datastore.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"]))
    intent = TradeIntent(
        id=intent_id, account_scope_id=scope_id, account_epoch=1, instrument_id=instrument_id,
        action="BUY_YES", quantity=Decimal("1"), limit_price=Decimal("0.4"), time_in_force="IOC",
        forecast_id="forecast-001", exit_reason=None, policy_version="policy-v1", strategy_version="strategy-v1",
        market_version="market-v1", fee_allowance=Decimal("0.01"), slippage_allowance=Decimal("0.01"),
        expires_at=now + timedelta(days=1), created_at=now,
    )
    try:
        store.reserve_intent(intent, cash=Decimal("6"), max_loss=Decimal("3"), now=now)
        result_queue.put("reserved")
    except InsufficientReservationCapacity:
        result_queue.put("capacity_blocked")


@unittest.skipUnless(os.environ.get("DATASTORE_EMULATOR_HOST"), "requires DATASTORE_EMULATOR_HOST")
class DatastoreTwinStoreIntegrationTests(unittest.TestCase):
    def test_manual_and_autonomous_commands_share_datastore_capacity(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        store = DatastoreTwinStore(datastore.Client(project=project))
        claim = reserve_confirmed_manual_order(
            store,
            user_id=f"manual-user-{uuid4().hex}",
            venue="kalshi",
            payload={"client_order_id": "manual-001"},
            normalized={
                "platform": "kalshi", "action": "buy", "outcome": "yes", "ticker": "KXTEST",
                "quantity": "12", "price": "0.5", "time_in_force": "immediate_or_cancel",
            },
            guardrails={
                "portfolio": {"available": "10"},
                "policy": {"max_daily_risk_notional": "10"},
                "quote": {"market_ident": "KXTEST"},
            },
            authority_ref="direct-manual-001",
            now=now,
        )
        autonomous = TradeIntent(
            id="autonomous-001", account_scope_id=claim.command.scope_id, account_epoch=1,
            instrument_id="kalshi:live:KXOTHER", action="BUY_YES", quantity=Decimal("10"),
            limit_price=Decimal("0.5"), time_in_force="IOC", forecast_id="forecast-001", exit_reason=None,
            policy_version="policy-v1", strategy_version="strategy-v1", market_version="market-v1",
            fee_allowance=Decimal("0"), slippage_allowance=Decimal("0"),
            expires_at=now + timedelta(days=1), created_at=now,
        )
        with self.assertRaises(InsufficientReservationCapacity):
            store.reserve_intent(autonomous, cash=Decimal("5"), max_loss=Decimal("5"), now=now)
        self.assertEqual(store.projection(claim.command.scope_id).reserved_cash, Decimal("6.0"))

    def test_fenced_datastore_claim_retains_account_scope(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        scope_id = f"claim-{uuid4().hex}"
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        store = DatastoreTwinStore(datastore.Client(project=project))
        store.register_account(
            AccountScope(scope_id, "owner-001", "kalshi", "account-ref", "demo", "USD", "connection-001", 1, now),
            venue_available_cash=Decimal("10"), loss_limit=Decimal("6"),
        )
        order = TradeIntent(
            id="claim-intent", account_scope_id=scope_id, account_epoch=1, instrument_id="kalshi:demo:KXCLAIM",
            action="BUY_YES", quantity=Decimal("1"), limit_price=Decimal("0.4"), time_in_force="IOC",
            forecast_id="forecast-001", exit_reason=None, policy_version="policy-v1", strategy_version="strategy-v1",
            market_version="market-v1", fee_allowance=Decimal("0.01"), slippage_allowance=Decimal("0.01"),
            expires_at=now + timedelta(days=1), created_at=now,
        )
        store.reserve_intent(order, cash=Decimal("2"), max_loss=Decimal("1"), now=now)
        command = store.command_for_intent(order)
        first = store.claim_command(command.id, worker_id="worker-a", now=now, lease_seconds=5)
        self.assertIsNotNone(first)
        self.assertIsNone(store.claim_command(command.id, worker_id="worker-b", now=now, lease_seconds=5))
        second = store.claim_command(command.id, worker_id="worker-b", now=now + timedelta(seconds=6), lease_seconds=5)
        self.assertEqual(second.fence, first.fence + 1)

    def test_two_processes_compete_for_last_account_capacity(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        scope_id = f"integration-{uuid4().hex}"
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        store = DatastoreTwinStore(datastore.Client(project=project))
        store.register_account(
            AccountScope(scope_id, "owner-001", "kalshi", "account-ref", "demo", "USD", "connection-001", 1, now),
            venue_available_cash=Decimal("10"), loss_limit=Decimal("6"),
        )
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        processes = [
            context.Process(target=_reserve_in_process, args=(scope_id, f"intent-{index}", f"kalshi:demo:KX{index}", result_queue))
            for index in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            self.assertEqual(process.exitcode, 0)
        outcomes = sorted(result_queue.get(timeout=5) for _ in processes)
        self.assertEqual(outcomes, ["capacity_blocked", "reserved"])
        self.assertEqual(store.projection(scope_id).reserved_cash, Decimal("6"))


if __name__ == "__main__":
    unittest.main()
