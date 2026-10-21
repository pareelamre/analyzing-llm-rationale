"""Datastore-emulator integration checks for account reservation contention.

Run with:
  DATASTORE_EMULATOR_HOST=127.0.0.1:8765 GOOGLE_CLOUD_PROJECT=foresea-twin-test \
  PYTHONPATH=src py -m unittest tests.test_twin_store_integration
"""
from __future__ import annotations

import multiprocessing
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Event
from unittest import mock
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
    TwinStoreError,
)
from analyzing_llm_rationale.twin.strategy import DatastoreStrategyStore, StrategyCycle
from analyzing_llm_rationale.twin.worker import (
    DatastoreResearchCompletionStore,
    DatastoreWorkerJobs,
    MaintenanceResearchJobGateway,
    ResearchCompletion,
    TwinResearchWorker,
    WorkerJob,
    WorkerJobError,
    WorkerJobKind,
    WorkerJobStatus,
)
from tests import test_twin_account_journal as account_journal
from tests import test_twin_account_maintenance as native_account_maintenance
from tests import test_twin_native_lifecycle as native_lifecycle
from tests import test_twin_native_pause as native_pause
from tests import test_twin_opening as opening


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
    def test_one_manual_authority_cannot_bind_multiple_intents_concurrently(self):
        from google.cloud import datastore

        project = os.environ.get('GOOGLE_CLOUD_PROJECT', 'foresea-twin-test')
        now = datetime.now(timezone.utc)
        scope = AccountScope('authority-'+uuid4().hex, 'owner', 'kalshi', 'account', 'live',
                             'USD', 'connection', 1, now)
        store = DatastoreTwinStore(datastore.Client(project=project))
        store.register_account(scope, venue_available_cash=Decimal('10'), loss_limit=Decimal('10'))

        def reserve(index):
            other = DatastoreTwinStore(datastore.Client(project=project))
            intent = TradeIntent('intent-'+str(index), scope.id, 1, 'kalshi:live:TEST'+str(index),
                'BUY_YES', Decimal('1'), Decimal('.5'), 'IOC', 'forecast', None, 'policy',
                'strategy', 'market', Decimal('0'), Decimal('0'), now+timedelta(minutes=5), now)
            try:
                other.reserve_intent(intent, cash=Decimal('1'), max_loss=Decimal('1'), now=now,
                                     authority_id='one-confirmed-request', authority_alias='client-'+str(index))
                return 'reserved'
            except TwinStoreError:
                return 'blocked'

        with ThreadPoolExecutor(4) as workers:
            outcomes = list(workers.map(reserve, range(4)))
        self.assertEqual(outcomes.count('reserved'), 1)
        self.assertEqual(store.projection(scope.id).reserved_cash, Decimal('1'))
        self.assertEqual(len(store.commands(scope.id)), 1)
        query = store._client.query(kind='TwinRequestAuthority', ancestor=store._key(scope.id))
        self.assertEqual(len(list(query.fetch())), 2)  # Losing aliases were rolled back.
        command = store.commands(scope.id)[0]
        claim = store.claim_command(command.id, worker_id='dispatch', now=now)
        self.assertTrue(store.begin_dispatch(command.id, worker_id=claim.worker_id, fence=claim.fence, now=now))
        with self.assertRaisesRegex(TwinStoreError, 'unstarted'):
            store.transition_command(command.id, target=CommandState.REJECTED,
                worker_id=claim.worker_id, fence=claim.fence, require_unstarted=True)
        self.assertEqual(store.projection(scope.id).reserved_cash, Decimal('1'))

    def test_prepared_request_and_dispatch_fence_are_atomic_across_clients(self):
        from google.api_core.exceptions import Aborted
        from google.cloud import datastore

        now = datetime.now(timezone.utc)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        scope = AccountScope(f"native-{uuid4().hex}", "fixture-owner", "polymarket",
                             "fixture-account", "live", "USD", "fixture-connection", 1, now)
        client = datastore.Client(project=project)
        store = DatastoreTwinStore(client)
        store.register_account(scope, venue_available_cash=Decimal("20"), loss_limit=Decimal("20"))
        intent = TradeIntent("fixture-native", scope.id, 1, "polymarket:live:fixture", "BUY_YES",
                             Decimal("5"), Decimal(".4"), "IOC", "fixture-forecast", None,
                             "fixture-policy", "fixture-strategy", "fixture-market", Decimal(".1"),
                             Decimal(".1"), now+timedelta(minutes=5), now)
        from tests.test_twin_native_polymarket import TEST_ENCRYPTION_KEY
        key_patch = mock.patch.dict(os.environ, {"FORESEA_TWIN_ORDER_ENCRYPTION_KEY": TEST_ENCRYPTION_KEY})
        key_patch.start()
        self.addCleanup(key_patch.stop)
        payload = {"schema_version": 1, "signature": "fixture-only-"+"a"*2000}
        store.reserve_intent(intent, cash=Decimal("2.2"), max_loss=Decimal("2.2"), now=now,
                             client_order_id="fixture-native-hash", prepared_request=payload)
        command = store.command_for_intent(intent)
        entity = client.get(store._key(scope.id, "TwinCommand", command.id))
        self.assertIn("prepared_request_json", entity.exclude_from_indexes)
        claim = store.claim_command(command.id, worker_id="fixture", now=now)

        def start_dispatch(_):
            other = DatastoreTwinStore(datastore.Client(project=project))
            observed = other.command_for_intent(intent)
            self.assertEqual(observed.prepared_request_json, command.prepared_request_json)
            try:
                return other.begin_dispatch(command.id, worker_id=claim.worker_id,
                                             fence=claim.fence, now=now)
            except Aborted:
                return False  # Contention may reject a writer; it must never send.

        with ThreadPoolExecutor(4) as workers:
            results = list(workers.map(start_dispatch, range(4)))
        self.assertEqual(sum(results), 1)
        self.assertTrue(store.command_for_intent(intent).dispatch_started)
        self.assertEqual(store.projection(scope.id).reserved_cash, Decimal("2.2"))

    def test_research_completion_replays_across_datastore_clients(self):
        from google.cloud import datastore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "foresea-twin-test")
        now = datetime.now(timezone.utc)
        job_id = f"research-replay-{uuid4().hex}"
        client = datastore.Client(project=project)
        jobs = DatastoreWorkerJobs(client)
        jobs.add(WorkerJob(
            job_id, "scope-001", WorkerJobKind.RESEARCH,
            {
                "research_assignment_id": f"assignment-{uuid4().hex}",
                "budget_reservation_id": f"budget-{uuid4().hex}",
                "market_snapshot_id": "snapshot-001",
                "evidence_set_id": "evidence-001",
                "model_config_id": "model-001",
                "budget_key_id": "budget-key-001",
                "strategy_cycle_id": "strategy-cycle-001",
            },
            now + timedelta(minutes=5),
        ))
        model_calls = []
        finalizations = []

        def research(_assignment):
            model_calls.append(1)
            return ResearchCompletion(
                "completed", result_payload={"schema_version": 1},
                actual_usd="0", actual_tokens=12,
            )

        def finalize(_assignment, completion):
            finalizations.append(completion.actual_tokens)
            if len(finalizations) == 1:
                raise TimeoutError("injected finalizer failure")
            return ResearchCompletion(
                "completed", research_result_id="result-001",
                usage_record_id="usage-001",
            )

        first = TwinResearchWorker(
            MaintenanceResearchJobGateway(
                jobs, authorize_assignment=lambda _: None,
                finalize_result=finalize,
                completion_store=DatastoreResearchCompletionStore(client),
            ), worker_id="first-worker",
        )
        with self.assertRaisesRegex(TimeoutError, "finalizer failure"):
            first.handle(job_id, now=now, research=research)
        self.assertEqual(jobs.get(job_id).status, WorkerJobStatus.RUNNING)

        restarted_client = datastore.Client(project=project)
        restarted_jobs = DatastoreWorkerJobs(restarted_client)
        replay = TwinResearchWorker(
            MaintenanceResearchJobGateway(
                restarted_jobs,
                authorize_assignment=lambda _: self.fail("model budget was reclaimed"),
                finalize_result=finalize,
                completion_store=DatastoreResearchCompletionStore(restarted_client),
            ), worker_id="restart-worker",
        )
        result = replay.handle(
            job_id, now=now + timedelta(seconds=126), research=research,
        )
        self.assertEqual(result["research_result_id"], "result-001")
        self.assertEqual(model_calls, [1])
        self.assertEqual(finalizations, [12, 12])
        self.assertEqual(restarted_jobs.get(job_id).status, WorkerJobStatus.COMPLETED)
        self.assertIsNone(DatastoreResearchCompletionStore(restarted_client).get(job_id))

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

        recover_id = f"worker-recover-{uuid4().hex}"
        store.add(WorkerJob(
            recover_id, "scope-001", WorkerJobKind.RECONCILE,
            {"account_snapshot_id": "snapshot-002"}, now + timedelta(minutes=1),
        ))
        lost = store.claim(
            recover_id, worker_id="lost-worker", now=now, lease_seconds=1,
        )
        recovered = store.recover_stale(now=now + timedelta(seconds=2))
        self.assertIn(recover_id, [item.id for item in recovered])
        self.assertEqual(store.get(recover_id).status, WorkerJobStatus.QUEUED)
        replacement = store.claim(
            recover_id, worker_id="replacement", now=now + timedelta(seconds=2),
        )
        self.assertEqual(replacement.fence, lost.fence + 1)

        expired_id = f"worker-expired-{uuid4().hex}"
        store.add(WorkerJob(
            expired_id, "scope-001", WorkerJobKind.RECONCILE,
            {"account_snapshot_id": "snapshot-003"}, now + timedelta(seconds=1),
        ))
        store.recover_stale(now=now + timedelta(seconds=2))
        expired = store.get(expired_id)
        self.assertEqual(expired.status, WorkerJobStatus.EXPIRED)
        self.assertEqual(expired.completed_result, {"status": "expired"})

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


from tests import test_twin_lifecycle_atomic as atomic_lifecycle  # noqa: E402
from tests import test_twin_recovery_review as review_lifecycle  # noqa: E402


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastoreAtomicLifecycleTests(atomic_lifecycle.AtomicLifecycleTests):
    backends = ('kalshi', 'polymarket')

    def pair(self, backend, *, acknowledge=True):
        from google.cloud import datastore

        client = datastore.Client(project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'foresea-twin-test'),
                                  namespace='atomic-'+uuid4().hex)
        self.addCleanup(client.close)
        store = DatastoreTwinStore(client)
        ledger = DatastoreLifecycleStore(client)
        store, intent, command, claim = atomic_lifecycle.prepared_unknown(
            venue=backend, environment='live', store=store)
        if acknowledge:
            atomic_lifecycle.recover_submission(store, command=command, intent=intent, claim=claim,
                now=atomic_lifecycle.NOW+timedelta(seconds=3),
                lookups=[atomic_lifecycle.lookup(command, intent, found=True)], clock=lambda: atomic_lifecycle.NOW+timedelta(seconds=3))
        return store, ledger, intent, store.command_for_intent(intent), claim

    def test_takeover_during_expired_reconciliation_retains_cash_and_identity(self):
        from google.cloud import datastore

        for venue in self.backends:
            with self.subTest(venue=venue):
                values = self.pair(venue)
                store, ledger, intent, command, old_claim = values
                client = datastore.Client(project=store._client.project, namespace=store._client.namespace)
                self.addCleanup(client.close)
                replacement_store = DatastoreTwinStore(client)
                saved, release, attempting = Event(), Event(), Event()
                clock = [atomic_lifecycle.NOW+timedelta(seconds=3)]
                original = ledger.save

                def hold_after_save(*args, _save=original, _saved=saved, _release=release, **kwargs):
                    result = _save(*args, **kwargs)
                    _saved.set()
                    if not _release.wait(10):
                        raise AssertionError('test failed to release the held transaction')
                    return result

                def take_over(_store=replacement_store, _command=command, _attempting=attempting):
                    _attempting.set()
                    return _store.claim_command(_command.id, worker_id='new-generation',
                        now=atomic_lifecycle.NOW+timedelta(seconds=33), lease_seconds=30)

                with mock.patch.object(ledger, 'save', side_effect=hold_after_save):
                    with ThreadPoolExecutor(2) as pool:
                        stale = pool.submit(self.run_reconcile, values, at=clock, order_status='open')
                        try:
                            self.assertTrue(saved.wait(10))
                            newer = pool.submit(take_over)
                            self.assertTrue(attempting.wait(10))
                            clock[0] = atomic_lifecycle.NOW+timedelta(seconds=33)
                        finally:
                            release.set()
                        with self.assertRaises(RecoveryBlocked):
                            stale.result(timeout=20)
                        claim = newer.result(timeout=20)
                self.assertEqual(claim.fence, old_claim.fence+1)
                self.assertEqual(store.command_for_intent(intent).claim, claim)
                self.assertEqual(store.command_for_intent(intent).client_order_id, command.client_order_id)
                self.assert_held_without_evidence(values)

    def test_concurrent_commit_order_respects_fences_without_expiring_the_old_clock(self):
        from google.cloud import datastore

        for venue in self.backends:
            with self.subTest(venue=venue):
                values = self.pair(venue)
                store, ledger, intent, command, claim = values
                client = datastore.Client(project=store._client.project, namespace=store._client.namespace)
                self.addCleanup(client.close)
                replacement_store = DatastoreTwinStore(client)
                saved, release = Event(), Event()
                original = ledger.save

                def hold(*args, _save=original, _saved=saved, _release=release, **kwargs):
                    result = _save(*args, **kwargs)
                    _saved.set()
                    if not _release.wait(10):
                        raise AssertionError('test failed to release held transaction')
                    return result

                with mock.patch.object(ledger, 'save', side_effect=hold):
                    with ThreadPoolExecutor(2) as pool:
                        older = pool.submit(self.run_reconcile, values,
                            fills=[atomic_lifecycle.fill(command,intent,'f','1')])
                        committed_first = False
                        try:
                            self.assertTrue(saved.wait(10))
                            newer = pool.submit(replacement_store.claim_command, command.id,
                                worker_id='new-fence', now=atomic_lifecycle.NOW+timedelta(seconds=33))
                            try:
                                newer.result(timeout=1)
                                committed_first = True  # Result returned: replacement actually committed.
                            except FutureTimeout:
                                pass
                        finally:
                            release.set()
                        failed = False
                        try:
                            result = older.result(timeout=20)
                        except RecoveryBlocked:
                            failed = True
                        replacement = newer.result(timeout=20)
                self.assertEqual(replacement.fence, claim.fence+1)
                if committed_first:
                    self.assertTrue(failed, 'a transaction cannot commit over an already committed newer fence')
                if failed:
                    self.assertIsNone(ledger.load(command.scope_id,command.id))
                    self.assertEqual(store.command_for_intent(intent).state, CommandState.ACKNOWLEDGED)
                else:
                    self.assertFalse(committed_first)
                    self.assertEqual(result.filled_quantity, 1)
                    self.assertEqual(store.command_for_intent(intent).state, CommandState.PARTIALLY_FILLED)
                self.assertEqual(store.command_for_intent(intent).claim, replacement)
                self.assertEqual(store.projection(command.scope_id).reserved_cash, 1)
                print('fence_commit_order', venue, 'replacement_before_release' if committed_first else 'overlapping_writer',
                      'old_aborted' if failed else 'old_committed_first')


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastoreLifecycleReviewTests(review_lifecycle.LifecycleReviewTests):
    backends = ('kalshi', 'polymarket')
    pair = DatastoreAtomicLifecycleTests.pair


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastoreNativeAccountMaintenanceTests(native_account_maintenance.NativeAccountMaintenanceTests):
    def make_client(self):
        from google.cloud import datastore

        self.enterContext(mock.patch.dict(os.environ, {'FORESEA_DATASTORE_BACKEND':'gcp'}))
        self.namespace = 'native-account-'+uuid4().hex
        return datastore.Client(project='foresea-twin-test', namespace=self.namespace)

    def test_committed_job_replacement_aborts_nonexpired_snapshot_commit(self):
        from google.cloud import datastore

        from analyzing_llm_rationale.twin.worker import DatastoreWorkerJobs, WorkerDegraded

        replacement_client = datastore.Client(project='foresea-twin-test', namespace=self.namespace)
        replacement_jobs = DatastoreWorkerJobs(replacement_client)
        saved, release = Event(), Event()
        original = self.snapshots.save

        def held_save(snapshot, *, validate):
            calls = 0
            def guard():
                nonlocal calls
                validate()
                calls += 1
                if calls == 2:
                    saved.set()
                    if not release.wait(15):
                        raise AssertionError('test failed to release snapshot transaction')
            return original(snapshot, validate=guard)

        with mock.patch.object(self.snapshots, 'save', side_effect=held_save), ThreadPoolExecutor(2) as pool:
            stale = pool.submit(self.operation(), self.job)
            try:
                self.assertTrue(saved.wait(10))
                newer = pool.submit(replacement_jobs.claim, self.job.id, worker_id='replacement',
                    now=native_account_maintenance.NOW+timedelta(seconds=121), lease_seconds=40)
                replacement = newer.result(timeout=10)
                self.assertIsNotNone(replacement)
                self.assertEqual(replacement.fence, self.job.fence+1)
            finally:
                release.set()
            with self.assertRaises(WorkerDegraded):
                stale.result(timeout=15)
        self.assertIsNone(self.snapshots.load(self.scope.id))
        self.assertEqual(self.jobs.get(self.job.id), replacement)
        self.assertEqual(self.store.command_for_intent(self.intent), self.command)
        print('native_capture_commit_order', self.venue, 'replacement_before_release', 'old_aborted')


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastorePolymarketAccountMaintenanceTests(DatastoreNativeAccountMaintenanceTests):
    venue = 'polymarket'


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastoreNativeLifecycleTests(native_lifecycle.NativeLifecycleTests):
    def make_client(self):
        from google.cloud import datastore

        self.enterContext(mock.patch.dict(os.environ, {'FORESEA_DATASTORE_BACKEND':'gcp'}))
        self.namespace = 'native-lifecycle-'+uuid4().hex
        return datastore.Client(project='foresea-twin-test', namespace=self.namespace)

    def test_committed_job_replacement_aborts_history_cursor_write(self):
        from google.cloud import datastore

        from analyzing_llm_rationale.twin.native_history import NativeHistoryCapture
        from analyzing_llm_rationale.twin.worker import DatastoreWorkerJobs, WorkerDegraded

        second = datastore.Client(project='foresea-twin-test', namespace=self.namespace)
        replacement_jobs = DatastoreWorkerJobs(second)
        saved, release = Event(), Event()
        original = NativeHistoryCapture._put

        def held(capture, key, payload):
            original(capture, key, payload)
            if key.kind == 'TwinNativeHistoryChunk':
                saved.set()
                if not release.wait(15):
                    raise AssertionError('test failed to release history transaction')

        with mock.patch.object(NativeHistoryCapture, '_put', held), ThreadPoolExecutor(2) as pool:
            stale = pool.submit(self.operation(), self.job)
            try:
                self.assertTrue(saved.wait(10))
                newer = pool.submit(replacement_jobs.claim, self.job.id, worker_id='new-history-job',
                    now=native_lifecycle.NOW + timedelta(seconds=121), lease_seconds=30)
                claim = newer.result(timeout=10)
                self.assertEqual(claim.fence, self.job.fence + 1)
            finally:
                release.set()
            with self.assertRaises(WorkerDegraded):
                stale.result(timeout=15)
        self.assertEqual(list(self.db.query(kind='TwinNativeHistoryChunk').fetch()), [])
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal('1'))
        self.assertEqual(self.store.command_for_intent(self.intent).state, CommandState.SUBMISSION_UNKNOWN)
        print('native_history_commit_order', self.venue, 'replacement_before_release', 'old_aborted')

    def test_committed_job_replacement_aborts_nonexpired_financial_commit(self):
        from google.cloud import datastore

        from analyzing_llm_rationale.twin.worker import DatastoreWorkerJobs, WorkerDegraded

        second = datastore.Client(project='foresea-twin-test',namespace=self.namespace)
        replacement_jobs = DatastoreWorkerJobs(second)
        self.assertEqual(DatastoreTwinStore(second).intent_for_command(self.scope.id,self.command.id),self.intent)
        saved, release = Event(), Event()
        original = self.lifecycle.save
        def held_save(*args, **kwargs):
            value = original(*args, **kwargs)
            saved.set()
            if not release.wait(15):
                raise AssertionError('test failed to release financial transaction')
            return value
        with mock.patch.object(self.lifecycle,'save',side_effect=held_save), ThreadPoolExecutor(2) as pool:
            stale = pool.submit(self.operation(),self.job)
            try:
                self.assertTrue(saved.wait(10))
                newer = pool.submit(replacement_jobs.claim,self.job.id,worker_id='new-job',
                    now=native_lifecycle.NOW+timedelta(seconds=121),lease_seconds=30)
                claim = newer.result(timeout=10)
                self.assertEqual(claim.fence,self.job.fence+1)
            finally:
                release.set()
            with self.assertRaises(WorkerDegraded):
                stale.result(timeout=15)
        self.assertIsNone(self.lifecycle.load(self.scope.id,self.command.id))
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash,Decimal('1'))
        self.assertEqual(self.store.command_for_intent(self.intent).state,CommandState.ACKNOWLEDGED)
        print('native_lifecycle_commit_order',self.venue,'replacement_before_release','old_aborted')


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastorePolymarketNativeLifecycleTests(
    DatastoreNativeLifecycleTests, native_lifecycle.PolymarketNativeLifecycleTests,
):
    venue = 'polymarket'


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastoreNativeRecoveryPauseTests(native_pause.NativeRecoveryPauseTests):
    fixture_type = DatastoreNativeLifecycleTests


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastoreAccountJournalTests(account_journal.AccountJournalTests):
    fixture_type = DatastoreNativeAccountMaintenanceTests

    def test_committed_journal_replacement_aborts_snapshot_commit(self):
        from google.cloud import datastore

        from analyzing_llm_rationale.twin.account_journal import DatastoreAccountJournal
        from analyzing_llm_rationale.twin.worker import WorkerDegraded

        journal = self.seed_matching_native_journal()
        other_client = datastore.Client(project='foresea-twin-test', namespace=self._case.namespace)
        other = DatastoreAccountJournal(DatastoreTwinStore(other_client))
        saved, release = Event(), Event()
        original = self.snapshots.save

        def held_save(snapshot, *, validate):
            calls = 0
            def guard():
                nonlocal calls
                validate()
                calls += 1
                if calls == 2:
                    saved.set()
                    if not release.wait(15):
                        raise AssertionError('journal race did not release snapshot transaction')
            return original(snapshot, validate=guard)

        with mock.patch.object(self.snapshots, 'save', side_effect=held_save), ThreadPoolExecutor(2) as pool:
            stale = pool.submit(self.operation(), self.job)
            try:
                self.assertTrue(saved.wait(10))
                newer = pool.submit(other.append, self.scope,
                    (self.fact('fee', kind='fee', cash_delta=Decimal('-.01'), positions=(), fee=Decimal('.01')),),
                    expected_revision=journal.revision)
                replacement = newer.result(timeout=10)
                self.assertEqual(replacement.revision, journal.revision+1)
            finally:
                release.set()
            with self.assertRaises(WorkerDegraded):
                stale.result(timeout=15)
        self.assertIsNone(self.snapshots.load(self.scope.id))
        self.assertEqual(self.journal.load(self.scope), replacement)
        print('account_journal_commit_order', self._case.venue, 'replacement_before_release', 'old_aborted')


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastorePolymarketAccountJournalTests(DatastoreAccountJournalTests):
    fixture_type = DatastorePolymarketAccountMaintenanceTests


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastoreAccountOpeningTests(opening.AccountOpeningTests):
    def make_client(self):
        from google.cloud import datastore

        self.enterContext(mock.patch.dict(os.environ, {'FORESEA_DATASTORE_BACKEND': 'gcp'}))
        if not hasattr(self, 'namespace'):
            self.namespace = 'owner-opening-' + uuid4().hex
        client = datastore.Client(project='foresea-twin-test', namespace=self.namespace)
        self.addCleanup(client.close)
        return client

    def test_concurrently_replaced_snapshot_aborts_opening_commit(self):
        from google.api_core.exceptions import Aborted, Conflict

        from analyzing_llm_rationale.twin.account_journal import DatastoreAccountJournal
        from analyzing_llm_rationale.twin.account_store import DatastoreAccountSnapshotStore
        from analyzing_llm_rationale.twin.opening import OpeningConfirmation

        body = OpeningConfirmation(**self.body())
        other = DatastoreAccountSnapshotStore(self.make_client())
        staged, release = Event(), Event()
        original = self.client.put
        def hold_approval(entity):
            result = original(entity)
            if entity.key.kind == 'TwinAccountOpeningApproval':
                staged.set()
                if not release.wait(15):
                    raise AssertionError('opening race did not release the transaction')
            return result
        with mock.patch.object(self.client, 'put', side_effect=hold_approval), ThreadPoolExecutor(2) as pool:
            stale = pool.submit(self.service.confirm, 'owner', self.scope.id, body)
            try:
                self.assertTrue(staged.wait(10))
                replacement = pool.submit(other.save, self.make_snapshot(generation=2))
                self.assertEqual(replacement.result(timeout=10).generation, 2)
            finally:
                release.set()
            with self.assertRaises((Aborted, Conflict)):
                stale.result(timeout=15)
        self.assertIsNone(DatastoreAccountJournal(self.store).load(self.scope))
        self.assertIsNone(self.client.get(self.service._key(self.scope)))
        self.assertEqual(self.snapshots.load(self.scope.id).generation, 2)

    def _race_opening_confirmations(self, *, duplicate):
        import json

        from google.api_core.exceptions import Aborted, Conflict

        from analyzing_llm_rationale.twin.account_journal import DatastoreAccountJournal
        from analyzing_llm_rationale.twin.account_store import DatastoreAccountSnapshotStore
        from analyzing_llm_rationale.twin.opening import (
            AccountOpeningService,
            OpeningConfirmation,
            OpeningConflict,
        )

        body = OpeningConfirmation(**self.body())
        competing = body if duplicate else body.model_copy(update={'evidence_ref': 'f' * 64})
        other_client = self.make_client()
        other = AccountOpeningService(DatastoreTwinStore(other_client), DatastoreAccountSnapshotStore(other_client),
                                      clock=lambda: self.time)
        before = self.store.projection(self.scope.id)
        staged, release = Event(), Event()
        original = self.client.put
        def hold_approval(entity):
            result = original(entity)
            if entity.key.kind == 'TwinAccountOpeningApproval':
                staged.set()
                if not release.wait(15):
                    raise AssertionError('competing opening confirmation did not release')
            return result
        with mock.patch.object(self.client, 'put', side_effect=hold_approval), ThreadPoolExecutor(2) as pool:
            stale = pool.submit(self.service.confirm, 'owner', self.scope.id, body)
            try:
                self.assertTrue(staged.wait(10))
                winner = pool.submit(other.confirm, 'owner', self.scope.id, competing).result(timeout=10)
            finally:
                release.set()
            with self.assertRaises((Aborted, Conflict)):
                stale.result(timeout=15)
        journal = DatastoreAccountJournal(self.store).load(self.scope)
        record = json.loads(self.client.get(self.service._key(self.scope))['payload_json'])
        self.assertEqual(record['evidence_ref'], competing.evidence_ref)
        self.assertEqual(journal.opening.source_ref, winner['opening_ref'])
        self.assertEqual((journal.revision, journal.facts), (1, ()))
        self.assertEqual(self.store.projection(self.scope.id).revision, before.revision + 1)
        self.assertEqual(other.confirm('owner', self.scope.id, competing), winner)
        if duplicate:
            self.assertEqual(self.service.confirm('owner', self.scope.id, body), winner)
        else:
            with self.assertRaises(OpeningConflict):
                self.service.confirm('owner', self.scope.id, body)

    def test_competing_distinct_confirmations_create_one_opening(self):
        self._race_opening_confirmations(duplicate=False)

    def test_concurrent_duplicate_confirmation_has_safe_retry(self):
        self._race_opening_confirmations(duplicate=True)


@unittest.skipUnless(os.environ.get('DATASTORE_EMULATOR_HOST'), 'local Datastore emulator required')
class DatastorePolymarketAccountOpeningTests(DatastoreAccountOpeningTests):
    venue = 'polymarket'


if __name__ == "__main__":
    unittest.main()
