"""Environment-built private worker app used by the two Cloud Run services."""
from __future__ import annotations

import logging
import os
import socket
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from opentelemetry import metrics, trace

from ..forecast_ledger import ForecastLedger
from ..observability import init_observability
from .account_store import DatastoreAccountSnapshotStore
from .budget import (
    BudgetAlreadyClaimed,
    BudgetExceeded,
    DatastoreResearchBudget,
    estimate_request_cost,
)
from .cycle_runtime import DatastoreStrategyRunStore, StrategyRun, StrategyRunPhase
from .execution import (
    ExecutionContext,
    SubmissionDisposition,
    SubmissionUnknown,
    submit_claimed_command,
)
from .market_capture import (
    CapturedMarket,
    DatastoreMarketCaptureStore,
    LiveMarketDataGateway,
    MarketCapturePolicy,
    capture_markets,
)
from .models import AccountScope, CommandState, Instrument, ProposalAction, TradeIntent
from .public_evidence import (
    NewsPipelinePublicArticleGateway,
    PublicEvidenceError,
    PublicEvidencePolicy,
    acquire_public_evidence,
    captured_market_listing_evidence,
)
from .research_gateway import (
    DatastorePublicEvidenceCache,
    DatastoreResearchCaptureStore,
    DatastoreResearchResultStore,
    PublicResearchCapture,
    ResearchRuntimePolicy,
    execute_preclaimed_research,
    load_research_runtime_policy,
    public_evidence_set_id,
    record_research_forecast,
    research_capture_payload,
    research_request_hash,
    research_result_payload,
    restore_research_capture,
    restore_research_result,
)
from .risk import RiskExposure
from .runtime import (
    HttpResearchJobGateway,
    PrivateTwinRuntime,
    RuntimeConfigurationError,
    RuntimeIdentityPolicy,
    create_private_worker_app,
)
from .scheduler import CloudTasksConfig, CloudTasksDispatcher, ShadowCycleSchedule
from .simulator import CapturedBook, DepthLevel, ShadowAssumptions, ShadowVenue
from .store import DatastoreTwinStore, TwinStoreError
from .strategy import (
    DatastoreStrategyStore,
    ForeseaEdgeStrategy,
    HeldPosition,
    StrategyAccountState,
    StrategyCandidate,
    StrategyCycle,
    StrategyStep,
    load_strategy_policy,
)
from .worker import (
    DatastoreWorkerJobs,
    MaintenanceResearchJobGateway,
    ResearchAssignment,
    ResearchCompletion,
    TwinResearchWorker,
    TwinWorker,
    WorkerDegraded,
    WorkerJob,
    WorkerJobError,
    WorkerJobKind,
    WorkerJobStatus,
    WorkerPaused,
    WorkerRole,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
repair_authorizations = metrics.get_meter(__name__).create_counter(
    "twin.research.repair_authorizations", unit="1",
)
research_preparations = metrics.get_meter(__name__).create_counter(
    "twin.strategy.research_preparations", unit="1",
)


@tracer.start_as_current_span("twin.strategy.load_calibration")
def _calibration_observations(ledger: ForecastLedger, *, as_of: datetime) -> tuple[dict[str, object], ...]:
    """Expose only resolved, audit-grade forecasts available before a decision."""
    observations = []
    for row in ledger.resolved_forecasts():
        try:
            resolved_at = datetime.fromisoformat(str(row["resolved_at"]))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            resolved_at >= as_of or row.get("ledger_audit_grade") is not True
            or row.get("source") != "twin_research_v1"
            or not all(str(row.get(name) or "").strip() for name in (
                "forecast_id", "instrument_id", "cluster_id", "model_hash",
                "prompt_hash", "category_family",
            ))
        ):
            continue
        observations.append({
            "id": str(row["forecast_id"]),
            "instrument_id": str(row["instrument_id"]),
            "cluster_id": str(row["cluster_id"]),
            "probability": str(row["model_probability"]),
            "outcome": int(row["outcome"]),
            "forecast_at": str(row["forecasted_at"]),
            "resolved_at": str(row["resolved_at"]),
            "model_hash": str(row["model_hash"]),
            "prompt_hash": str(row["prompt_hash"]),
            "category_family": str(row["category_family"]),
        })
    return tuple(sorted(observations, key=lambda item: (str(item["resolved_at"]), str(item["id"]))))


@tracer.start_as_current_span("twin.shadow_account.reconcile")
def _reconcile_shadow_account(
    snapshot_store, twin_store, *, scope_id: str, account_epoch: int,
    now: datetime, strategy_policy,
) -> tuple[AccountScope, StrategyAccountState]:
    """Refresh one durable, zero-authority account generation for shadow decisions."""
    scope = AccountScope(
        scope_id, "foresea-edge-v1", "shadow", "foresea-edge-v1", "shadow",
        "USD", "shadow-venue-v3", account_epoch,
        datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    prior = snapshot_store.load(scope_id)
    if prior is None:
        venue = ShadowVenue(
            account_id="shadow-account-foresea-edge-v1",
            scope_id=scope_id,
            seed=20260913,
            starting_cash=Decimal("1000"),
        )
    else:
        venue = ShadowVenue.from_account_snapshot(
            account_id="shadow-account-foresea-edge-v1",
            seed=20260913,
            snapshot=prior,
        )
    snapshot = venue.account(received_at=now)
    snapshot = snapshot_store.save(snapshot)
    projection = twin_store.register_account(
        scope, venue_available_cash=snapshot.available_cash,
        loss_limit=strategy_policy.risk_limits.max_total_loss,
    )
    if (
        projection.venue_available_cash != snapshot.available_cash
        or projection.loss_limit != strategy_policy.risk_limits.max_total_loss
    ):
        projection = twin_store.refresh_account_capacity(
            scope.id, venue_available_cash=snapshot.available_cash,
            loss_limit=strategy_policy.risk_limits.max_total_loss,
        )
    positions: list[HeldPosition] = []
    exposures: list[RiskExposure] = []
    portfolio_complete = not snapshot.divergence
    order_rows = [dict(row) for row in snapshot.orders]
    for holding in snapshot.holdings:
        instrument_id, separator, outcome = holding.instrument_id.rpartition(":")
        matching = [
            row for row in order_rows
            if row.get("instrument_id") == instrument_id
            and row.get("outcome") == outcome
            and str(row.get("action") or "").startswith("BUY_")
            and isinstance(row.get("instrument"), dict)
            and isinstance(row.get("intent"), dict)
        ]
        if not separator or not matching:
            portfolio_complete = False
            continue
        try:
            instruments = [Instrument(**dict(row["instrument"])) for row in matching]
            intents = [TradeIntent.from_storage(dict(row["intent"])) for row in matching]
            instrument = instruments[0]
            if any(item != instrument for item in instruments):
                raise ValueError("position instrument versions diverged")
            opened_at = min(item.created_at for item in intents)
            pending_sell = sum((
                Decimal(str(row.get("remaining_quantity", "0")))
                for row in order_rows
                if row.get("instrument_id") == instrument_id
                and row.get("outcome") == outcome
                and str(row.get("action") or "").startswith("SELL_")
            ), Decimal("0"))
            positions.append(HeldPosition(
                instrument, outcome, holding.quantity, opened_at,
                min(
                    instrument.close_at,
                    opened_at + timedelta(seconds=strategy_policy.maximum_holding_seconds),
                ),
                instrument.settlement_spec_hash, strategy_policy.config_version,
                pending_sell_quantity=min(pending_sell, holding.quantity),
            ))
            exposures.append(RiskExposure(
                instrument.id, instrument.cluster_id, instrument.venue,
                holding.basis, "inventory",
            ))
        except (ArithmeticError, KeyError, TypeError, ValueError):
            portfolio_complete = False
    if snapshot.settlements or any(
        str(row.get("action") or "").startswith("SELL_") for row in order_rows
    ):
        # Realized-PnL state is deliberately not inferred from partial history.
        portfolio_complete = False
    current_equity = snapshot.conservative_liquidation_value
    state = StrategyAccountState(
        snapshot, projection, tuple(positions), tuple(exposures),
        Decimal("0"), Decimal("0"), max(Decimal("1000"), current_equity),
        current_equity,
        portfolio_complete=portfolio_complete,
    )
    return scope, state


def _shadow_book(market: CapturedMarket, intent: TradeIntent) -> CapturedBook:
    outcome = "yes" if intent.action in {ProposalAction.BUY_YES, ProposalAction.SELL_YES} else "no"
    buying = intent.action in {ProposalAction.BUY_YES, ProposalAction.BUY_NO}
    if outcome == "yes":
        price = market.snapshot.yes_ask if buying else market.snapshot.yes_bid
        depth = market.yes_ask_depth if buying else market.yes_bid_depth
    else:
        price = market.snapshot.no_ask if buying else market.snapshot.no_bid
        depth = market.no_ask_depth if buying else market.no_bid_depth
    if price is None or depth is None or depth <= 0:
        raise ValueError("captured executable side has no exact depth")
    return CapturedBook(market.snapshot, outcome, (DepthLevel(price, depth),))


@tracer.start_as_current_span("twin.shadow.execute_cycle")
def _execute_shadow_cycle(
    cycle: StrategyCycle, *, capture, snapshot_store, twin_store,
    scope: AccountScope, now: datetime, worker_id: str,
):
    """Reserve and simulate one accepted intent with restart-safe identities."""
    if cycle.decision != "INTENT" or cycle.intent is None or cycle.risk_result is None:
        return None
    intent, risk = cycle.intent, cycle.risk_result
    market = next((
        item for item in capture.markets
        if item.instrument.id == intent.instrument_id
        and item.snapshot.id == intent.market_version
    ), None)
    if market is None:
        raise ValueError("accepted intent has no matching immutable market capture")
    prior = snapshot_store.load(scope.id)
    if prior is None:
        raise ValueError("shadow account snapshot is unavailable")
    price = intent.limit_price
    fee_per_share = (
        market.trading_cost.fee_per_share_at(price)
        if market.trading_cost is not None else Decimal("0")
    )
    fee_rate = fee_per_share / price if price > 0 else Decimal("0")
    venue = ShadowVenue.from_account_snapshot(
        account_id="shadow-account-foresea-edge-v1", seed=20260913,
        snapshot=prior, assumptions=ShadowAssumptions(fee_rate=fee_rate),
    )
    twin_store.reserve_intent(
        intent, cash=risk.cash, max_loss=risk.max_loss, now=now,
        preconditions=risk.reservation_preconditions,
    )
    command = twin_store.command_for_intent(intent)
    order_id = f"shadow-order:{command.client_order_id}"
    receipt = next((
        item for item in prior.orders if item.get("order_id") == order_id
    ), None)
    if command.state in {
        CommandState.FILLED, CommandState.REJECTED, CommandState.CANCELLED,
    }:
        return command
    claim = twin_store.claim_command(command.id, worker_id=worker_id, now=now)
    if claim is None:
        current = twin_store.command_for_intent(intent)
        if current.state in {
            CommandState.FILLED, CommandState.REJECTED, CommandState.CANCELLED,
        }:
            return current
        raise WorkerPaused("shadow_command_claim_in_flight")

    if receipt is not None and command.state in {
        CommandState.SUBMITTING, CommandState.SUBMISSION_UNKNOWN,
        CommandState.ACKNOWLEDGED, CommandState.PARTIALLY_FILLED,
    }:
        status = str(receipt.get("status") or "")
        target = (
            CommandState.FILLED if status == "filled"
            else CommandState.PARTIALLY_FILLED if status == "partial"
            else CommandState.CANCELLED if status == "cancelled"
            else CommandState.ACKNOWLEDGED
        )
        current = command
        if (
            current.state is CommandState.SUBMITTING
            or current.state is CommandState.SUBMISSION_UNKNOWN
            and target is CommandState.CANCELLED
        ):
            current = twin_store.transition_command(
                command.id, target=CommandState.ACKNOWLEDGED,
                fence=claim.fence, worker_id=claim.worker_id,
            )
        if current.state is target:
            return current
        return twin_store.transition_command(
            command.id, target=target, fence=claim.fence,
            worker_id=claim.worker_id,
        )
    if command.state is CommandState.SUBMISSION_UNKNOWN:
        return twin_store.transition_command(
            command.id, target=CommandState.REJECTED,
            fence=claim.fence, worker_id=claim.worker_id,
        )

    preview = venue.preview(
        intent, risk, market.instrument, _shadow_book(market, intent), now=now,
    )

    def persist_simulation(current):
        response = venue.submit(current, preview, now=now)
        simulated = venue.status(order_id)
        if intent.time_in_force == "IOC" and simulated.remaining_quantity > 0:
            venue.cancel(order_id, now=now)
        proposed = venue.account(received_at=now)
        saved = snapshot_store.save(proposed)
        if saved != proposed:
            raise TwinStoreError("a newer shadow account generation won the submission race")
        return response

    try:
        result = submit_claimed_command(
            twin_store, command=command, intent=intent, claim=claim,
            context=ExecutionContext(
                scope, intent.policy_version, intent.strategy_version,
                intent.market_version, False, autonomous=True, simulation=True,
            ),
            now=now, submit=persist_simulation,
        )
    except SubmissionUnknown as exc:
        raise WorkerPaused("shadow_submission_requires_reconciliation") from exc
    if result.disposition is not SubmissionDisposition.ACKNOWLEDGED:
        return result.command
    status = venue.status(order_id).status
    target = (
        CommandState.FILLED if status == "filled"
        else CommandState.PARTIALLY_FILLED if status == "partial"
        else CommandState.CANCELLED if status == "cancelled"
        else None
    )
    try:
        return (
            twin_store.transition_command(
            result.command.id, target=target, fence=claim.fence,
            worker_id=claim.worker_id,
            ) if target is not None else result.command
        )
    except TwinStoreError as exc:
        raise WorkerPaused("shadow_fill_state_requires_reconciliation") from exc


@tracer.start_as_current_span("twin.strategy.refresh_decision_market")
def _refresh_decision_markets(
    run: StrategyRun, *, market_capture_store, market_data_gateway,
    market_capture_policy, now: datetime,
):
    """Capture executable prices after research and retain only compatible contracts."""
    capture_id = run.id + ":decision"
    capture = market_capture_store.get(capture_id)
    if capture is None:
        capture = capture_markets(
            market_data_gateway, now=now, policy=market_capture_policy,
            observation_clock=lambda: datetime.now(timezone.utc),
        )
        market_capture_store.record(capture_id, capture)
    originals = {item.instrument.id: item for item in run.candidates}
    candidates = []
    for market in capture.markets:
        original = originals.get(market.instrument.id)
        if original is None or market.trading_cost is None:
            continue
        if (
            market.instrument.settlement_spec_hash != original.instrument.settlement_spec_hash
            or market.instrument.capability_version != original.instrument.capability_version
            or market.instrument.tick_size != original.instrument.tick_size
            or market.instrument.min_quantity != original.instrument.min_quantity
            or market.instrument.fee_version != original.instrument.fee_version
        ):
            continue
        candidates.append(StrategyCandidate(
            market.instrument, market.snapshot,
            market.yes_ask_depth, market.no_ask_depth,
            max(
                market.trading_cost.yes_fee_per_share,
                market.trading_cost.no_fee_per_share,
            ),
            market.instrument.tick_size,
            original.calibration_observations,
            (original.snapshot.id,),
        ))
    return capture, tuple(candidates)


def _stable_id(prefix: str, *parts: object) -> str:
    return prefix + sha256("|".join(str(item) for item in parts).encode()).hexdigest()[:24]


@tracer.start_as_current_span("twin.strategy.prepare_research")
def _prepare_strategy_research(
    *, run: StrategyRun, capture, jobs, budget, captures, evidence_cache,
    evidence_gateway, research_policy: ResearchRuntimePolicy, now: datetime,
    calibration_observations=(),
) -> tuple[tuple[StrategyCandidate, ...], tuple[str, ...]]:
    """Persist bounded research inputs and jobs, skipping unsafe candidates."""
    candidates: list[StrategyCandidate] = []
    job_ids: list[str] = []
    if research_policy.model.price_valid_until <= now:
        research_preparations.add(1, {"outcome": "expired_price", "candidate_count": 0})
        return (), ()
    estimate = estimate_request_cost(
        input_tokens=research_policy.model.max_input_tokens,
        output_tokens=research_policy.model.max_output_tokens,
        price=research_policy.model.price,
        require_usd_ceiling=True,
    )
    estimated_tokens = (
        research_policy.model.max_input_tokens + research_policy.model.max_output_tokens
    )
    budget_key = f"foresea-edge:{run.account_scope_id}:{now.date().isoformat()}"
    for market in capture.markets[:research_policy.candidates_per_cycle]:
        if market.trading_cost is None:
            continue
        assignment_id = _stable_id("assignment-", run.id, market.instrument.id)
        reservation_id = _stable_id("budget-", run.id, market.instrument.id)
        job_id = _stable_id("research-job-", run.id, market.instrument.id)
        try:
            research_capture = captures.get_capture(assignment_id)
            if research_capture is None:
                try:
                    evidence = acquire_public_evidence(
                        evidence_gateway, instrument=market.instrument, now=now,
                        policy=PublicEvidencePolicy(max_evidence=5),
                    )
                except PublicEvidenceError:
                    evidence = captured_market_listing_evidence(
                        instrument=market.instrument,
                        rules=market.settlement_rules,
                        retrieved_at=now,
                    )
                # All inputs in this capture are stamped against the bounded
                # preparation instant. Slow public-source failures must not age
                # an otherwise fresh snapshot while the fallback is assembled.
                as_of = now
                research_capture = PublicResearchCapture(
                    market.instrument, market.snapshot, market.settlement_rules,
                    as_of, evidence,
                )
                research_capture.validate_at_decision()
                evidence_cache.put(market.instrument.id, as_of, evidence)
                captures.record_capture(assignment_id, research_capture)
            evidence_id = public_evidence_set_id(
                research_capture.instrument.id, research_capture.as_of,
                research_capture.evidence,
            )
            budget.reserve(
                reservation_id, key=budget_key, estimated_usd=estimate,
                estimated_tokens=estimated_tokens, policy=research_policy.budget,
            )
            jobs.add(WorkerJob(
                job_id, run.account_scope_id, WorkerJobKind.RESEARCH,
                {
                    "research_assignment_id": assignment_id,
                    "budget_reservation_id": reservation_id,
                    "market_snapshot_id": research_capture.snapshot.id,
                    "evidence_set_id": evidence_id,
                    "model_config_id": research_policy.id,
                    "budget_key_id": budget_key,
                    "strategy_cycle_id": run.id,
                },
                min(research_capture.instrument.close_at, now + timedelta(minutes=4)),
                created_at=now,
            ))
        except (ArithmeticError, BudgetExceeded, PublicEvidenceError, ValueError):
            continue
        candidates.append(StrategyCandidate(
            market.instrument, market.snapshot,
            market.yes_ask_depth, market.no_ask_depth,
            max(
                market.trading_cost.yes_fee_per_share,
                market.trading_cost.no_fee_per_share,
            ),
            market.instrument.tick_size,
            tuple(calibration_observations),
        ))
        job_ids.append(job_id)
    research_preparations.add(1, {
        "outcome": "prepared" if job_ids else "empty",
        "candidate_count": len(job_ids),
    })
    return tuple(candidates), tuple(job_ids)


def _stage_strategy_continuation(
    jobs, strategy_run_store, assignment: ResearchAssignment | WorkerJob, *, now: datetime,
) -> WorkerJob | None:
    """Durably stage one unique continuation after a research terminal result."""
    if isinstance(assignment, WorkerJob):
        cycle_id = assignment.payload.get("strategy_cycle_id") or (
            "legacy-research-cycle:" + sha256(assignment.id.encode()).hexdigest()[:24]
        )
        research_job_id = assignment.id
    else:
        cycle_id = assignment.strategy_cycle_id
        research_job_id = assignment.job_id
    if cycle_id.startswith("legacy-research-cycle:"):
        return None
    run = strategy_run_store.get(cycle_id)
    if run is None:
        raise WorkerJobError("strategy run is unavailable for research continuation")
    return jobs.add(WorkerJob(
        _stable_id("strategy-continuation-", run.id, research_job_id),
        run.account_scope_id, WorkerJobKind.STRATEGY,
        {
            "strategy_cycle_id": run.id,
            "config_release_id": run.config_release_id,
            "account_epoch_id": str(run.account_epoch),
        },
        now + timedelta(minutes=10), created_at=now,
    ))


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeConfigurationError(f"{name} is required")
    return value


def _accounts(name: str, *, required: bool = True) -> frozenset[str]:
    values = frozenset(
        item.strip() for item in os.environ.get(name, "").split(",") if item.strip()
    )
    if required and not values:
        raise RuntimeConfigurationError(f"{name} requires at least one service account")
    return values


def _assert_shadow_only() -> None:
    if os.environ.get("FORESEA_TWIN_MODE", "").strip().lower() != "shadow":
        raise RuntimeConfigurationError("private twin runtime is shadow-only")
    if os.environ.get("FORESEA_TWIN_LIVE_CAPITAL", "").strip() != "0":
        raise RuntimeConfigurationError("private twin runtime requires zero live capital")
    if os.environ.get("FORESEA_TWIN_LIVE_MANDATE", "").strip():
        raise RuntimeConfigurationError("private twin runtime cannot receive a live mandate")


def _runtime_worker_id(role: WorkerRole, hostname: str) -> str:
    """Return a bounded stable ID without exposing a platform hostname."""
    if not hostname.strip():
        raise RuntimeConfigurationError("private twin runtime requires an instance hostname")
    return f"{role.value}-{sha256(hostname.encode()).hexdigest()[:24]}"


class _DatastoreLedgerAdapter:
    """Provide ForecastLedger's immutable insert surface over Cloud Datastore."""

    def __init__(self, client) -> None:
        self._client = client

    def key(self, kind: str, id_: str):
        return self._client.key(kind, id_)

    def get(self, key):
        return self._client.get(key)

    def query(self, kind: str):
        return self._client.query(kind=kind)

    def insert_immutable(self, source) -> bool:
        from google.cloud import datastore

        key = self._client.key(source.key.kind, source.key.id)
        with self._client.transaction():
            if self._client.get(key) is not None:
                return False
            entity = datastore.Entity(key=key, exclude_from_indexes=("payload",))
            entity.update(dict(source))
            self._client.put(entity)
        return True


def _recover_stale_research_budgets(
    jobs: DatastoreWorkerJobs, budget: DatastoreResearchBudget, *, now: datetime,
) -> int:
    """Classify expired research leases as uncertain without releasing capacity."""
    recovered = 0
    for stale in jobs.stale(now=now):
        if stale.kind is not WorkerJobKind.RESEARCH:
            continue
        try:
            budget.mark_uncertain(
                stale.payload["budget_reservation_id"],
                key=stale.payload["budget_key_id"],
            )
        except (BudgetExceeded, KeyError, ValueError):
            # The original reservation remains counted if durable recovery is
            # unavailable, so maintenance cannot create spending capacity.
            continue
        try:
            budget.mark_uncertain(
                stale.payload["budget_reservation_id"] + ":repair",
                key=stale.payload["budget_key_id"],
            )
        except (BudgetExceeded, KeyError, ValueError):
            pass
        recovered += 1
    return recovered


def _reconcile_worker_startup(
    jobs: DatastoreWorkerJobs, budget: DatastoreResearchBudget, *, now: datetime,
    dispatcher=None, strategy_run_store=None,
) -> bool:
    """Resolve durable expired work before opening maintenance readiness."""
    _recover_stale_research_budgets(jobs, budget, now=now)
    recovered = jobs.recover_stale(now=now)
    if dispatcher is not None:
        for recovered_job in recovered:
            if recovered_job.status is WorkerJobStatus.QUEUED:
                dispatcher.enqueue(recovered_job)
            elif (
                recovered_job.kind is WorkerJobKind.RESEARCH
                and recovered_job.status is WorkerJobStatus.EXPIRED
                and strategy_run_store is not None
            ):
                continuation = _stage_strategy_continuation(
                    jobs, strategy_run_store, recovered_job, now=now,
                )
                if continuation is not None:
                    dispatcher.enqueue(continuation)
    return not jobs.stale(now=now)


def _maintenance_operation(
    jobs: DatastoreWorkerJobs, budget: DatastoreResearchBudget, job: WorkerJob,
    strategy_store: DatastoreStrategyStore | None = None,
    strategy_run_store: DatastoreStrategyRunStore | None = None,
    market_capture_store: DatastoreMarketCaptureStore | None = None,
    market_data_gateway: LiveMarketDataGateway | None = None,
    market_capture_policy: MarketCapturePolicy | None = None,
    research_capture_store=None,
    research_result_store=None,
    evidence_cache=None,
    evidence_gateway=None,
    research_policy: ResearchRuntimePolicy | None = None,
    dispatcher=None,
    account_snapshot_store=None,
    twin_store=None,
    strategy_policy=None,
    calibration_observations=(),
) -> dict[str, object]:
    """Safe staging operation until venue/account adapters are configured."""
    if job.kind is WorkerJobKind.RECOVERY:
        now = datetime.now(timezone.utc)
        stale = jobs.stale(now=now)
        uncertain = _recover_stale_research_budgets(jobs, budget, now=now)
        return {
            "status": "complete",
            "stale_jobs_detected": len(stale),
            "research_budgets_marked_uncertain": uncertain,
        }
    if job.kind in {WorkerJobKind.RECONCILE, WorkerJobKind.EXIT}:
        raise WorkerDegraded("account_maintenance_adapter_unconfigured")
    if job.kind is WorkerJobKind.STRATEGY:
        execution_command = None
        if (
            strategy_store is None or strategy_run_store is None
            or market_capture_store is None or market_data_gateway is None
        ):
            raise WorkerDegraded("strategy_store_unconfigured")
        run = strategy_run_store.get(job.payload["strategy_cycle_id"])
        if run is None:
            run = strategy_run_store.create(StrategyRun(
                id=job.payload["strategy_cycle_id"],
                account_scope_id=job.account_scope_id,
                account_epoch=int(job.payload["account_epoch_id"]),
                config_release_id=job.payload["config_release_id"],
                observed_at=job.created_at,
            ))
        elif (
            run.account_scope_id != job.account_scope_id
            or run.account_epoch != int(job.payload["account_epoch_id"])
            or run.config_release_id != job.payload["config_release_id"]
        ):
            raise WorkerJobError("strategy continuation identity changed")
        if run.phase is StrategyRunPhase.QUEUED:
            capture = market_capture_store.get(run.id)
            if capture is None:
                capture = capture_markets(
                    market_data_gateway, now=datetime.now(timezone.utc),
                    policy=market_capture_policy,
                    observation_clock=lambda: datetime.now(timezone.utc),
                )
                market_capture_store.record(run.id, capture)
            prepared_at = datetime.now(timezone.utc)
            if not capture.markets:
                run = strategy_run_store.save(
                    run.advance(
                        StrategyRunPhase.BLOCKED, now=prepared_at,
                        reason="no_eligible_markets",
                    ),
                    expected_revision=run.revision,
                )
            elif any(item is None for item in (
                research_capture_store, research_result_store, evidence_cache,
                evidence_gateway, research_policy, dispatcher,
            )):
                run = strategy_run_store.save(
                    run.advance(
                        StrategyRunPhase.BLOCKED, now=prepared_at,
                        reason="strategy_research_unconfigured",
                    ),
                    expected_revision=run.revision,
                )
            else:
                candidates, research_job_ids = _prepare_strategy_research(
                    run=run, capture=capture, jobs=jobs, budget=budget,
                    captures=research_capture_store, evidence_cache=evidence_cache,
                    evidence_gateway=evidence_gateway,
                    research_policy=research_policy, now=prepared_at,
                    calibration_observations=(
                        calibration_observations()
                        if callable(calibration_observations)
                        else calibration_observations
                    ),
                )
                if not research_job_ids:
                    run = strategy_run_store.save(
                        run.advance(
                            StrategyRunPhase.BLOCKED, now=datetime.now(timezone.utc),
                            reason="research_inputs_unavailable",
                        ),
                        expected_revision=run.revision,
                    )
                else:
                    run = strategy_run_store.save(
                        run.advance(
                            StrategyRunPhase.RESEARCH_PENDING,
                            now=datetime.now(timezone.utc), candidates=candidates,
                            research_job_ids=research_job_ids,
                        ),
                        expected_revision=run.revision,
                    )
                    for research_job_id in research_job_ids:
                        dispatcher.enqueue(jobs.get(research_job_id))
        durable_results = ()
        if run.phase in {StrategyRunPhase.RESEARCH_PENDING, StrategyRunPhase.READY}:
            research_jobs = tuple(jobs.get(item) for item in run.research_job_ids)
            if any(item.completed_result is None for item in research_jobs):
                if run.phase is StrategyRunPhase.READY:
                    raise WorkerPaused("ready_strategy_research_not_terminal")
                return {
                    "status": "pending", "reason": "research_pending",
                    "strategy_cycle_id": run.id,
                    "config_release_id": run.config_release_id,
                    "run_phase": run.phase.value, "run_revision": run.revision,
                    "research_job_count": len(research_jobs),
                }
            durable_results = tuple(
                research_result_store.get_result(item.payload["budget_reservation_id"])
                for item in research_jobs
            )
            if not any(item is not None for item in durable_results):
                run = strategy_run_store.save(
                    run.advance(
                        StrategyRunPhase.BLOCKED, now=datetime.now(timezone.utc),
                        reason="research_results_unavailable",
                    ),
                    expected_revision=run.revision,
                )
            elif run.phase is StrategyRunPhase.RESEARCH_PENDING:
                run = strategy_run_store.save(
                    run.advance(StrategyRunPhase.READY, now=datetime.now(timezone.utc)),
                    expected_revision=run.revision,
                )
        if run.phase is StrategyRunPhase.READY:
            decision_at = datetime.now(timezone.utc)
            if any(item is None for item in (
                account_snapshot_store, twin_store, strategy_policy,
            )):
                run = strategy_run_store.save(
                    run.advance(
                        StrategyRunPhase.BLOCKED, now=decision_at,
                        reason="account_maintenance_adapter_unconfigured",
                    ),
                    expected_revision=run.revision,
                )
            else:
                decision_capture, decision_candidates = _refresh_decision_markets(
                    run, market_capture_store=market_capture_store,
                    market_data_gateway=market_data_gateway,
                    market_capture_policy=market_capture_policy,
                    now=decision_at,
                )
                decision_at = datetime.now(timezone.utc)
                scope, account_state = _reconcile_shadow_account(
                    account_snapshot_store, twin_store,
                    scope_id=run.account_scope_id, account_epoch=run.account_epoch,
                    now=decision_at, strategy_policy=strategy_policy,
                )
                results_by_instrument = {
                    candidate.instrument.id: result
                    for candidate, result in zip(run.candidates, durable_results)
                    if result is not None
                }
                candidates_by_instrument = {
                    candidate.instrument.id: candidate for candidate in decision_candidates
                }
                cycle = ForeseaEdgeStrategy(
                    store=strategy_store, policy=strategy_policy,
                ).run_cycle(
                    scope=scope, now=decision_at,
                    cycle_identity_at=run.observed_at,
                    reconcile=lambda: account_state,
                    load_position_market=lambda position: candidates_by_instrument.get(
                        position.instrument.id,
                    ),
                    discover=lambda: tuple(
                        candidate for candidate in decision_candidates
                        if candidate.instrument.id in results_by_instrument
                    ),
                    research=lambda candidate: results_by_instrument[candidate.instrument.id],
                )
                execution_capture = decision_capture
                try:
                    if execution_capture is None:
                        raise ValueError("strategy market capture is unavailable")
                    execution_command = _execute_shadow_cycle(
                        cycle, capture=execution_capture,
                        snapshot_store=account_snapshot_store, twin_store=twin_store,
                        scope=scope, now=decision_at,
                        worker_id=f"shadow-executor:{job.id}",
                    )
                except WorkerPaused:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Shadow strategy execution failed (%s: %s)",
                        type(exc).__name__, str(exc),
                    )
                    run = strategy_run_store.save(
                        run.advance(
                            StrategyRunPhase.BLOCKED, now=decision_at,
                            reason="shadow_execution_failed",
                        ),
                        expected_revision=run.revision,
                    )
                else:
                    run = strategy_run_store.save(
                        run.advance(
                            StrategyRunPhase.COMPLETE, now=decision_at,
                            reason=cycle.reason,
                        ),
                        expected_revision=run.revision,
                    )
        capture = market_capture_store.get(run.id)
        if run.phase not in {StrategyRunPhase.BLOCKED, StrategyRunPhase.COMPLETE}:
            raise WorkerPaused("strategy_run_not_terminal")
        cycle = strategy_store.get_cycle(job.payload["strategy_cycle_id"])
        recorded = False
        if cycle is None:
            recorded = strategy_store.record_cycle(StrategyCycle(
                key=job.payload["strategy_cycle_id"],
                decision="PASS",
                reason=str(run.reason),
                steps=(StrategyStep(
                    "market_capture", "blocked", str(run.reason),
                    job.payload["config_release_id"],
                ),),
                created_at=job.created_at,
                account_scope_id=job.account_scope_id,
            ))
            cycle = strategy_store.get_cycle(job.payload["strategy_cycle_id"])
        if (
            execution_command is None and cycle is not None
            and cycle.intent is not None and twin_store is not None
        ):
            try:
                execution_command = twin_store.command_for_intent(cycle.intent)
            except TwinStoreError:
                pass
        return {
            "status": "complete" if run.phase is StrategyRunPhase.COMPLETE else "blocked",
            "reason": str(run.reason),
            "decision": cycle.decision if cycle is not None else "PASS",
            "command_state": execution_command.state.value if execution_command is not None else None,
            "strategy_cycle_id": job.payload["strategy_cycle_id"],
            "config_release_id": job.payload["config_release_id"],
            "observation_recorded": recorded,
            "run_phase": run.phase.value,
            "run_revision": run.revision,
            "market_capture_count": len(capture.markets) if capture else 0,
            "market_rejection_count": len(capture.rejections) if capture else 0,
        }
    raise WorkerPaused("unsupported_maintenance_job")


@tracer.start_as_current_span("twin.research.authorize_repair")
def _authorize_research_repair(
    budget, assignment: ResearchAssignment, policy: ResearchRuntimePolicy,
    actual_usd: str | None, actual_tokens: int | None,
) -> bool:
    """Reconcile the primary call, then atomically claim one repair allowance."""
    span = trace.get_current_span()
    try:
        amount = None if actual_usd is None else Decimal(actual_usd)
        budget.reconcile(
            assignment.budget_reservation_id, key=assignment.budget_key_id,
            actual_usd=amount, actual_tokens=actual_tokens,
        )
        repair_id = assignment.budget_reservation_id + ":repair"
        estimate = estimate_request_cost(
            input_tokens=policy.model.max_input_tokens,
            output_tokens=policy.model.max_output_tokens,
            price=policy.model.price,
            require_usd_ceiling=True,
        )
        budget.reserve(
            repair_id, key=assignment.budget_key_id,
            estimated_usd=estimate,
            estimated_tokens=(
                policy.model.max_input_tokens + policy.model.max_output_tokens
            ),
            policy=policy.budget,
        )
        budget.claim(repair_id, key=assignment.budget_key_id)
    except (ArithmeticError, BudgetAlreadyClaimed, BudgetExceeded, KeyError, ValueError) as exc:
        span.set_attribute("outcome", "denied")
        span.set_attribute("twin.research.repair_denial", type(exc).__name__)
        repair_authorizations.add(1, {"outcome": "denied"})
        return False
    span.set_attribute("outcome", "authorized")
    repair_authorizations.add(1, {"outcome": "authorized"})
    return True


def _research_operation(
    assignment: ResearchAssignment, policy: ResearchRuntimePolicy,
    gateway: HttpResearchJobGateway, provider,
) -> ResearchCompletion:
    if assignment.model_config_id != policy.id:
        raise WorkerDegraded("research_model_config_mismatch")
    try:
        capture = restore_research_capture(gateway.load_capture(assignment))
    except (ValueError, WorkerJobError) as exc:
        raise WorkerDegraded("research_capture_unavailable") from exc
    if (
        capture.snapshot.id != assignment.market_snapshot_id
        or public_evidence_set_id(
            capture.instrument.id, capture.as_of, capture.evidence,
        ) != assignment.evidence_set_id
    ):
        raise WorkerDegraded("research_capture_identity_mismatch")
    execution = execute_preclaimed_research(
        provider, capture=capture, config=policy.model,
        now=datetime.now(timezone.utc),
        authorize_repair=lambda usd, tokens: gateway.authorize_repair(
            assignment,
            actual_usd=None if usd is None else str(usd),
            actual_tokens=tokens,
        ),
    )
    usage = {}
    if execution.actual_usd is not None and execution.actual_tokens is not None:
        usage = {
            "actual_usd": str(execution.actual_usd),
            "actual_tokens": execution.actual_tokens,
        }
    return ResearchCompletion(
        "completed", result_payload=research_result_payload(execution.result),
        repair_attempted=execution.repair_attempted,
        repair_actual_usd=(
            None if execution.repair_actual_usd is None
            else str(execution.repair_actual_usd)
        ),
        repair_actual_tokens=execution.repair_actual_tokens,
        **usage,
    )


def create_environment_app():
    """Build a role-specific app from secret-free deployment configuration."""
    _assert_shadow_only()
    role = WorkerRole(_required("FORESEA_TWIN_WORKER_ROLE"))

    def now() -> datetime:
        return datetime.now(timezone.utc)
    audience = _required(
        "FORESEA_TWIN_MAINTENANCE_AUDIENCE"
        if role is WorkerRole.MAINTENANCE else "FORESEA_TWIN_RESEARCH_AUDIENCE"
    )
    identities = RuntimeIdentityPolicy(
        audience,
        _accounts("FORESEA_TWIN_SCHEDULER_ACCOUNTS", required=role is WorkerRole.MAINTENANCE),
        _accounts("FORESEA_TWIN_DISPATCHER_ACCOUNTS"),
        _accounts("FORESEA_TWIN_RESEARCH_ACCOUNTS", required=role is WorkerRole.MAINTENANCE),
    )
    worker_id = _runtime_worker_id(role, socket.gethostname())
    repository_root = Path(__file__).resolve().parents[3]
    research_policy = load_research_runtime_policy(
        repository_root / "configs" / "twin.yaml",
        repository_root / "configs" / "models.yaml",
    )
    strategy_policy = load_strategy_policy(
        repository_root / "configs" / "twin.yaml", shadow_trial=True,
    )

    if role is WorkerRole.MAINTENANCE:
        from google.cloud import datastore

        client = datastore.Client(project=_required("GOOGLE_CLOUD_PROJECT"))
        jobs = DatastoreWorkerJobs(client)
        budget = DatastoreResearchBudget(client)
        captures = DatastoreResearchCaptureStore(client)
        results = DatastoreResearchResultStore(client)
        strategy_store = DatastoreStrategyStore(client)
        account_snapshot_store = DatastoreAccountSnapshotStore(client)
        twin_store = DatastoreTwinStore(client)
        strategy_run_store = DatastoreStrategyRunStore(client)
        market_capture_store = DatastoreMarketCaptureStore(client)
        market_data_gateway = LiveMarketDataGateway()
        evidence_cache = DatastorePublicEvidenceCache(client)
        evidence_gateway = NewsPipelinePublicArticleGateway()
        ledger = ForecastLedger(_DatastoreLedgerAdapter(client))
        dispatcher = CloudTasksDispatcher(CloudTasksConfig(
            _required("GOOGLE_CLOUD_PROJECT"), _required("FORESEA_TWIN_TASKS_LOCATION"),
            _required("FORESEA_TWIN_MAINTENANCE_QUEUE"),
            _required("FORESEA_TWIN_RESEARCH_QUEUE"),
            _required("FORESEA_TWIN_MAINTENANCE_URL") + "/internal/twin/maintain",
            _required("FORESEA_TWIN_RESEARCH_URL") + "/internal/twin/research",
            _required("FORESEA_TWIN_DISPATCHER_SERVICE_ACCOUNT"),
            _required("FORESEA_TWIN_MAINTENANCE_AUDIENCE"),
            _required("FORESEA_TWIN_RESEARCH_AUDIENCE"),
        ))

        def stage_continuation(assignment: ResearchAssignment) -> WorkerJob | None:
            return _stage_strategy_continuation(
                jobs, strategy_run_store, assignment, now=now(),
            )

        def authorize(assignment: ResearchAssignment) -> None:
            try:
                budget.claim(
                    assignment.budget_reservation_id, key=assignment.budget_key_id,
                )
            except BudgetAlreadyClaimed as exc:
                raise WorkerDegraded("research_budget_claim_unknown") from exc
            except (BudgetExceeded, KeyError, ValueError) as exc:
                raise WorkerDegraded("research_budget_unavailable") from exc

        def load_capture(assignment: ResearchAssignment):
            if assignment.model_config_id != research_policy.id:
                raise WorkerDegraded("research_model_config_mismatch")
            capture = captures.get_capture(assignment.research_assignment_id)
            if capture is None:
                raise WorkerDegraded("research_capture_unavailable")
            if (
                capture.snapshot.id != assignment.market_snapshot_id
                or public_evidence_set_id(
                    capture.instrument.id, capture.as_of, capture.evidence,
                ) != assignment.evidence_set_id
            ):
                raise WorkerDegraded("research_capture_identity_mismatch")
            return research_capture_payload(capture)

        def authorize_repair(
            assignment: ResearchAssignment, actual_usd: str | None,
            actual_tokens: int | None,
        ) -> bool:
            return _authorize_research_repair(
                budget, assignment, research_policy, actual_usd, actual_tokens,
            )

        def finalize_result(
            assignment: ResearchAssignment, completion: ResearchCompletion,
        ) -> ResearchCompletion:
            if completion.status == "degraded":
                budget.mark_uncertain(
                    assignment.budget_reservation_id, key=assignment.budget_key_id,
                )
                stage_continuation(assignment)
                return ResearchCompletion("degraded", reason=completion.reason)
            if completion.result_payload is None:
                raise WorkerDegraded("research_result_payload_missing")
            capture = restore_research_capture(load_capture(assignment))
            result = restore_research_result(completion.result_payload)
            if (
                result.request_hash != research_request_hash(capture, research_policy.model)
                or result.proposal.market_snapshot_id != assignment.market_snapshot_id
                or (
                    result.forecast is not None
                    and result.forecast.instrument_id != capture.instrument.id
                )
            ):
                raise WorkerDegraded("research_result_identity_mismatch")
            results.record_result(assignment.budget_reservation_id, result)
            record_research_forecast(ledger, result, capture, research_policy.model)
            actual_usd = (
                None if completion.actual_usd is None
                else Decimal(completion.actual_usd)
            )
            budget.reconcile(
                assignment.budget_reservation_id, key=assignment.budget_key_id,
                actual_usd=actual_usd, actual_tokens=completion.actual_tokens,
            )
            if completion.repair_attempted:
                repair_usd = (
                    None if completion.repair_actual_usd is None
                    else Decimal(completion.repair_actual_usd)
                )
                budget.reconcile(
                    assignment.budget_reservation_id + ":repair",
                    key=assignment.budget_key_id,
                    actual_usd=repair_usd,
                    actual_tokens=completion.repair_actual_tokens,
                )
            result_id = (
                result.forecast.id if result.forecast is not None else result.proposal.id
            )
            usage_id = "usage-" + sha256(
                (
                    f"{assignment.budget_reservation_id}|"
                    f"{completion.actual_usd}|{completion.actual_tokens}"
                ).encode()
            ).hexdigest()[:24]
            stage_continuation(assignment)
            return ResearchCompletion(
                "completed", research_result_id=result_id, usage_record_id=usage_id,
            )

        gateway = MaintenanceResearchJobGateway(
            jobs, authorize_assignment=authorize, capture_loader=load_capture,
            authorize_repair=authorize_repair, finalize_result=finalize_result,
            after_complete=lambda assignment, _completed: (
                dispatcher.enqueue(continuation)
                if (continuation := stage_continuation(assignment)) is not None
                else None
            ),
        )
        worker = TwinWorker(
            jobs, worker_id=worker_id,
            reconcile_startup=lambda: _reconcile_worker_startup(
                jobs, budget, now=now(), dispatcher=dispatcher,
                strategy_run_store=strategy_run_store,
            ),
            # Cloud Run bounds maintenance requests at 120 seconds. Keep the
            # claim fenced slightly longer so readiness recovery cannot reclaim
            # work that the platform is still allowing to finish.
            lease_seconds=125,
        )
        runtime = PrivateTwinRuntime(
            role, identities, now, jobs=jobs, dispatcher=dispatcher,
            cycle_schedule=ShadowCycleSchedule(
                "shadow-scope:foresea-edge-v1", strategy_policy.config_version,
                bucket_seconds=strategy_policy.cycle_bucket_seconds,
                deadline_seconds=strategy_policy.cycle_bucket_seconds,
            ),
            maintenance_worker=worker,
            maintenance_operation=lambda job: _maintenance_operation(
                jobs, budget, job, strategy_store, strategy_run_store,
                market_capture_store, market_data_gateway,
                MarketCapturePolicy(
                    max_candidates=strategy_policy.max_research_candidates,
                    candidates_per_venue=strategy_policy.max_research_candidates,
                ),
                captures, results, evidence_cache, evidence_gateway,
                research_policy, dispatcher,
                account_snapshot_store, twin_store, strategy_policy,
                lambda: _calibration_observations(ledger, as_of=now()),
            ),
            research_gateway=gateway,
        )
    else:
        # Startup validates the configured secret, model identity, endpoint,
        # timeout and price before this revision can report ready.
        provider = research_policy.build_provider()
        gateway = HttpResearchJobGateway(
            _required("FORESEA_TWIN_MAINTENANCE_URL"),
            audience=_required("FORESEA_TWIN_MAINTENANCE_AUDIENCE"),
        )
        runtime = PrivateTwinRuntime(
            role, identities, now,
            research_worker=TwinResearchWorker(gateway, worker_id=worker_id),
            research_operation=lambda assignment: _research_operation(
                assignment, research_policy, gateway,
                provider,
            ),
        )
    app = create_private_worker_app(runtime)
    init_observability(app)
    return app
