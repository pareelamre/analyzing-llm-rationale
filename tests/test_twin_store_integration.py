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

from analyzing_llm_rationale.twin import AccountScope, CommandState, TradeIntent
from analyzing_llm_rationale.twin.budget import (
    BudgetAlreadyClaimed,
    BudgetExceeded,
    BudgetPolicy,
    DatastoreResearchBudget,
    call_with_budget,
)
from analyzing_llm_rationale.twin.cycle_runtime import (
    DatastoreStrategyRunStore,
    StrategyRun,
    StrategyRunError,
    StrategyRunPhase,
)
from analyzing_llm_rationale.twin.mandates import DatastoreMandateStore, Mandate
from analyzing_llm_rationale.twin.manual import reserve_confirmed_manual_order
from analyzing_llm_rationale.twin.market_capture import (
    DatastoreMarketCaptureStore,
    MarketCaptureBatch,
    MarketCaptureError,
)
from analyzing_llm_rationale.twin.operator import DatastorePauseStore
from analyzing_llm_rationale.twin.recovery import (
    DatastoreLifecycleStore,
    FillObservation,
    LifecycleProjection,
    RecoveryBlocked,
    apply_lifecycle_observations,
)
from analyzing_llm_rationale.twin.store import (
    DatastoreTwinStore,
    InsufficientReservationCapacity,
    ReservationState,
)
from analyzing_llm_rationale.twin.strategy import DatastoreStrategyStore, StrategyCycle
from analyzing_llm_rationale.twin.worker import (
    DatastoreWorkerJobs,
    WorkerJob,
    WorkerJobError,
    WorkerJobKind,
    WorkerJobStatus,
)


def _research_in_process(key, reservation_id, result_queue):
    from google.api_core.exceptions import Aborted, Conflict
    from google.cloud import datastore

    store = DatastoreResearchBudget(datastore.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"]))

    def provider():
        result_queue.put("provider-called")
        return {"usage": {"cost_usd": "0", "total_tokens": 100}}

    try:
        call_with_budget(store, reservation_id, key=key, estimated_usd=Decimal("0"), estimated_tokens=100, policy=BudgetPolicy(Decimal("0"), 100, 10), operation=provider)
    except (BudgetAlreadyClaimed, BudgetExceeded, Aborted, Conflict):
        result_queue.put("blocked")


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
    def test_market_capture_survives_restart_and_rejects_conflicting_cycle(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        client = datastore.Client(project=project)
        store = DatastoreMarketCaptureStore(client)
        cycle_id = f"market-capture-{uuid4().hex}"
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        batch = MarketCaptureBatch(now, (), ())
        self.assertTrue(store.record(cycle_id, batch))
        self.assertFalse(store.record(cycle_id, batch))
        restarted = DatastoreMarketCaptureStore(datastore.Client(project=project))
        self.assertEqual(restarted.get(cycle_id), batch)
        with self.assertRaisesRegex(MarketCaptureError, "conflicting"):
            restarted.record(
                cycle_id, MarketCaptureBatch(now + timedelta(seconds=1), (), ()),
            )

    def test_strategy_run_phase_survives_restart_and_fences_stale_writer(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        client = datastore.Client(project=project)
        store = DatastoreStrategyRunStore(client)
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        initial = store.create(StrategyRun(
            f"strategy-cycle-{uuid4().hex}",
            "shadow-scope:foresea-edge-v1", 1,
            "foresea-edge-shadow-v1", now,
        ))
        blocked = initial.advance(
            StrategyRunPhase.BLOCKED, now=now + timedelta(seconds=1),
            reason="fixture_complete",
        )
        store.save(blocked, expected_revision=0)

        restarted = DatastoreStrategyRunStore(datastore.Client(project=project))
        self.assertEqual(restarted.get(initial.id), blocked)
        self.assertEqual(restarted.create(initial), blocked)
        with self.assertRaisesRegex(StrategyRunError, "revision conflict"):
            restarted.save(blocked, expected_revision=0)

    def test_operator_queries_and_controls_survive_store_recreation(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        client = datastore.Client(project=project)
        owner_id = f"operator-{uuid4().hex}"
        scope_id = f"scope-{uuid4().hex}"
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        twin = DatastoreTwinStore(client)
        twin.register_account(
            AccountScope(
                scope_id, owner_id, "kalshi", "account-ref", "shadow", "USD",
                "connection-ref", 1, now,
            ),
            venue_available_cash=Decimal("10"), loss_limit=Decimal("5"),
        )
        order = TradeIntent(
            id="operator-intent", account_scope_id=scope_id, account_epoch=1,
            instrument_id="kalshi:demo:KXOPERATOR", action="BUY_YES",
            quantity=Decimal("1"), limit_price=Decimal("0.4"), time_in_force="IOC",
            forecast_id="forecast-001", exit_reason=None, policy_version="policy-v1",
            strategy_version="strategy-v1", market_version="market-v1",
            fee_allowance=Decimal("0.01"), slippage_allowance=Decimal("0.01"),
            expires_at=now + timedelta(days=1), created_at=now,
        )
        twin.reserve_intent(order, cash=Decimal("1"), max_loss=Decimal("1"), now=now)

        strategies = DatastoreStrategyStore(client)
        strategies.record_cycle(StrategyCycle(
            f"cycle-{uuid4().hex}", "PASS", "operator_fixture",
            created_at=now, account_scope_id=scope_id,
        ))
        mandates = DatastoreMandateStore(client)
        draft = Mandate(
            f"mandate-{uuid4().hex}", owner_id, scope_id, "strategy-v1",
            now + timedelta(days=1), account_epoch=1, venue="kalshi",
            max_capital="10", max_loss="5", max_model_usd="1",
            max_model_tokens=1000, max_model_requests=5,
            model_hash="a" * 64, config_hash="b" * 64,
            readiness_hash="c" * 64, release_hash="d" * 64,
            identity_hash="e" * 64, created_at=now,
        )
        mandates.create(draft)
        pauses = DatastorePauseStore(client)
        pauses.set(
            owner_id, paused=True, reason="operator_fixture",
            idempotency_key="operator-pause-001", now=now,
        )

        recreated = DatastoreTwinStore(datastore.Client(project=project))
        self.assertEqual([item.id for item in recreated.account_scopes(owner_id)], [scope_id])
        self.assertEqual([item.intent_id for item in recreated.commands(scope_id)], [order.id])
        self.assertEqual(
            [item.account_scope_id for item in DatastoreStrategyStore(client).cycles(
                frozenset({scope_id}), limit=10,
            )],
            [scope_id],
        )
        self.assertEqual(DatastoreMandateStore(client).latest_for_owner(owner_id), (draft,))
        self.assertTrue(DatastorePauseStore(client).get(owner_id).paused)

    def test_durable_worker_claims_survive_restart_and_fence_stale_completion(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        client = datastore.Client(project=project)
        store = DatastoreWorkerJobs(client)
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        job_id = f"worker-{uuid4().hex}"
        created = store.add(WorkerJob(
            job_id, "scope-001", WorkerJobKind.RECONCILE,
            {"account_snapshot_id": "snapshot-001"}, now + timedelta(minutes=1),
        ))
        self.assertEqual(store.add(created).id, job_id)
        first = store.claim(job_id, worker_id="same-worker", now=now, lease_seconds=1)
        self.assertIsNone(store.claim(job_id, worker_id="other", now=now, lease_seconds=1))
        self.assertEqual(
            [item.id for item in store.stale(now=now + timedelta(seconds=2))],
            [job_id],
        )
        second = store.claim(
            job_id, worker_id="same-worker", now=now + timedelta(seconds=2),
            lease_seconds=10,
        )
        self.assertEqual(second.fence, first.fence + 1)
        with self.assertRaisesRegex(WorkerJobError, "stale"):
            store.complete(
                job_id, worker_id="same-worker", fence=first.fence,
                result={"status": "complete"}, now=now + timedelta(seconds=2),
            )
        completed = store.complete(
            job_id, worker_id="same-worker", fence=second.fence,
            result={"status": "complete"}, now=now + timedelta(seconds=2),
        )
        self.assertEqual(completed.status, WorkerJobStatus.COMPLETED)
        restarted = DatastoreWorkerJobs(datastore.Client(project=project))
        self.assertEqual(restarted.get(job_id).completed_result, {"status": "complete"})

    def test_datastore_budget_reservation_is_idempotent_and_unknown_spend_stays_reserved(self):
        from google.cloud import datastore

        budget = DatastoreResearchBudget(datastore.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")))
        key = f"budget-{uuid4().hex}"
        policy = BudgetPolicy(Decimal("1"), 100, 1)
        budget.reserve("call-001", key=key, estimated_usd=Decimal("0.8"), estimated_tokens=50, policy=policy)
        self.assertEqual(budget.reserve("call-001", key=key, estimated_usd=Decimal("0.8"), estimated_tokens=50, policy=policy).estimated_usd, Decimal("0.8"))
        with self.assertRaises(ValueError):
            budget.reserve("call-001", key=key, estimated_usd=Decimal("0.9"), estimated_tokens=90, policy=policy)
        budget.reconcile("call-001", key=key, actual_usd=None, actual_tokens=None)
        with self.assertRaises(BudgetExceeded):
            budget.reserve("call-002", key=key, estimated_usd=Decimal("0.3"), estimated_tokens=10, policy=policy)

    def test_research_processes_cannot_reuse_a_claim_or_final_tokens(self):
        from google.cloud import datastore

        context = multiprocessing.get_context("spawn")
        for duplicate in (True, False):
            with self.subTest(duplicate=duplicate):
                key = f"research-race-{uuid4().hex}"
                queue = context.Queue()
                processes = [context.Process(target=_research_in_process, args=(key, "same" if duplicate else f"request-{i}", queue)) for i in range(2)]
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=30)
                    self.assertEqual(process.exitcode, 0)
                self.assertCountEqual([queue.get(timeout=5) for _ in processes], ["provider-called", "blocked"])
                store = DatastoreResearchBudget(datastore.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"]))
                self.assertEqual(store.usage(key).actual_tokens, 100)

    def test_uncertain_tokens_and_legacy_usage_fail_closed(self):
        from google.cloud import datastore

        client = datastore.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
        store = DatastoreResearchBudget(client)
        key = f"unknown-{uuid4().hex}"
        policy = BudgetPolicy(Decimal("0"), 100, 10)
        store.reserve("unknown", key=key, estimated_usd=Decimal("0"), estimated_tokens=100, policy=policy)
        store.claim("unknown", key=key)
        store.reconcile("unknown", key=key, actual_usd=None, actual_tokens=None)
        self.assertEqual(store.usage(key).uncertain_tokens, 100)
        with self.assertRaises(BudgetExceeded):
            store.reserve("again", key=key, estimated_usd=Decimal("0"), estimated_tokens=1, policy=policy)
        with self.assertRaises(BudgetAlreadyClaimed):
            store.claim("unknown", key=key)
        actual = store.reconcile("unknown", key=key, actual_usd=Decimal("0"), actual_tokens=80)
        self.assertEqual(actual.uncertain_tokens, 0)
        self.assertEqual(actual.actual_tokens, 80)
        self.assertEqual(store.reconcile("unknown", key=key, actual_usd=Decimal("0"), actual_tokens=80), actual)
        store.reserve("remaining", key=key, estimated_usd=Decimal("0"), estimated_tokens=20, policy=policy)
        legacy_key = f"legacy-{uuid4().hex}"
        legacy = datastore.Entity(key=client.key("TwinResearchBudget", legacy_key))
        legacy.update({"requests": 1, "uncertain_usd": "0"})
        client.put(legacy)
        with self.assertRaisesRegex(BudgetExceeded, "legacy budget"):
            store.reserve("new", key=legacy_key, estimated_usd=Decimal("0"), estimated_tokens=1, policy=policy)

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
        self.assertEqual(
            store.reservation(scope_id, command.reservation_id).state,
            ReservationState.SUBMITTING,
        )
        self.assertIsNone(store.claim_command(command.id, worker_id="worker-b", now=now, lease_seconds=5))
        second = store.claim_command(command.id, worker_id="worker-b", now=now + timedelta(seconds=6), lease_seconds=5)
        self.assertEqual(second.fence, first.fence + 1)
        store.transition_command(
            command.id, target=CommandState.ACKNOWLEDGED,
            fence=second.fence, worker_id=second.worker_id,
        )
        third = store.claim_command(
            command.id, worker_id="worker-c", now=now + timedelta(seconds=12),
            lease_seconds=5,
        )
        self.assertEqual(third.fence, second.fence + 1)
        store.transition_command(
            command.id, target=CommandState.PARTIALLY_FILLED,
            fence=third.fence, worker_id=third.worker_id,
        )
        fourth = store.claim_command(
            command.id, worker_id="worker-d", now=now + timedelta(seconds=18),
            lease_seconds=5,
        )
        self.assertEqual(fourth.fence, third.fence + 1)

    def test_confirmed_rejection_atomically_releases_datastore_reservation(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        scope_id = f"reject-{uuid4().hex}"
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        store = DatastoreTwinStore(datastore.Client(project=project))
        store.register_account(
            AccountScope(
                scope_id, "owner-001", "kalshi", "account-ref", "demo", "USD",
                "connection-001", 1, now,
            ),
            venue_available_cash=Decimal("10"), loss_limit=Decimal("6"),
        )
        order = TradeIntent(
            "reject-intent", scope_id, 1, "kalshi:demo:KXREJECT", "BUY_YES",
            Decimal("1"), Decimal("0.4"), "IOC", "forecast-001", None,
            "policy-v1", "strategy-v1", "market-v1", Decimal("0"), Decimal("0"),
            now + timedelta(days=1), now,
        )
        reservation = store.reserve_intent(
            order, cash=Decimal("1"), max_loss=Decimal("1"), now=now,
        )
        command = store.command_for_intent(order)
        claim = store.claim_command(command.id, worker_id="worker", now=now)
        store.transition_command(
            command.id, target=CommandState.REJECTED,
            fence=claim.fence, worker_id=claim.worker_id,
        )
        self.assertEqual(store.projection(scope_id).reserved_cash, Decimal("0"))
        self.assertEqual(
            store.reservation(scope_id, reservation.id).state,
            ReservationState.RELEASED,
        )

    def test_lifecycle_projection_is_integrity_checked_and_compare_and_swapped(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        scope_id = f"lifecycle-{uuid4().hex}"
        store = DatastoreLifecycleStore(datastore.Client(project=project))
        base = LifecycleProjection(
            "command-1", scope_id, "venue-order-1", "client-1",
            "kalshi:demo:KX", Decimal("2"),
        )
        fill = FillObservation(
            "fill-1", 1, "venue-order-1", "client-1", base.instrument_id,
            Decimal("1"), datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        updated = apply_lifecycle_observations(
            base, fills=[fill], observed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(store.save(updated, expected_revision=0), updated)
        self.assertEqual(store.load(scope_id, base.command_id), updated)
        with self.assertRaisesRegex(RecoveryBlocked, "changed"):
            store.save(updated, expected_revision=0)

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
