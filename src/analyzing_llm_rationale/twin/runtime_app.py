"""Environment-built private worker app used by the two Cloud Run services."""
from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from opentelemetry import metrics, trace

from ..forecast_ledger import ForecastLedger
from ..observability import init_observability
from .budget import (
    BudgetAlreadyClaimed,
    BudgetExceeded,
    DatastoreResearchBudget,
    estimate_request_cost,
)
from .cycle_runtime import DatastoreStrategyRunStore, StrategyRun, StrategyRunPhase
from .market_capture import (
    DatastoreMarketCaptureStore,
    LiveMarketDataGateway,
    MarketCapturePolicy,
    capture_markets,
)
from .research_gateway import (
    DatastoreResearchCaptureStore,
    DatastoreResearchResultStore,
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
from .runtime import (
    HttpResearchJobGateway,
    PrivateTwinRuntime,
    RuntimeConfigurationError,
    RuntimeIdentityPolicy,
    create_private_worker_app,
)
from .scheduler import CloudTasksConfig, CloudTasksDispatcher, ShadowCycleSchedule
from .strategy import DatastoreStrategyStore, StrategyCycle, StrategyStep, load_strategy_policy
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
    WorkerPaused,
    WorkerRole,
)

tracer = trace.get_tracer(__name__)
repair_authorizations = metrics.get_meter(__name__).create_counter(
    "twin.research.repair_authorizations", unit="1",
)


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


def _maintenance_operation(
    jobs: DatastoreWorkerJobs, budget: DatastoreResearchBudget, job: WorkerJob,
    strategy_store: DatastoreStrategyStore | None = None,
    strategy_run_store: DatastoreStrategyRunStore | None = None,
    market_capture_store: DatastoreMarketCaptureStore | None = None,
    market_data_gateway: LiveMarketDataGateway | None = None,
    market_capture_policy: MarketCapturePolicy | None = None,
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
        if (
            strategy_store is None or strategy_run_store is None
            or market_capture_store is None or market_data_gateway is None
        ):
            raise WorkerDegraded("strategy_store_unconfigured")
        run = strategy_run_store.create(StrategyRun(
            id=job.payload["strategy_cycle_id"],
            account_scope_id=job.account_scope_id,
            account_epoch=int(job.payload["account_epoch_id"]),
            config_release_id=job.payload["config_release_id"],
            observed_at=job.created_at,
        ))
        if run.phase is StrategyRunPhase.QUEUED:
            capture = market_capture_store.get(run.id)
            if capture is None:
                capture = capture_markets(
                    market_data_gateway, now=datetime.now(timezone.utc),
                    policy=market_capture_policy,
                )
                market_capture_store.record(run.id, capture)
            reason = (
                "strategy_research_unconfigured" if capture.markets
                else "no_eligible_markets"
            )
            run = strategy_run_store.save(
                run.advance(
                    StrategyRunPhase.BLOCKED,
                    now=datetime.now(timezone.utc),
                    reason=reason,
                ),
                expected_revision=run.revision,
            )
        capture = market_capture_store.get(run.id)
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
        return {
            "status": "blocked",
            "reason": "strategy_dependencies_unconfigured",
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
    strategy_policy = load_strategy_policy(repository_root / "configs" / "twin.yaml")

    if role is WorkerRole.MAINTENANCE:
        from google.cloud import datastore

        client = datastore.Client(project=_required("GOOGLE_CLOUD_PROJECT"))
        jobs = DatastoreWorkerJobs(client)
        budget = DatastoreResearchBudget(client)
        captures = DatastoreResearchCaptureStore(client)
        results = DatastoreResearchResultStore(client)
        strategy_store = DatastoreStrategyStore(client)
        strategy_run_store = DatastoreStrategyRunStore(client)
        market_capture_store = DatastoreMarketCaptureStore(client)
        market_data_gateway = LiveMarketDataGateway()
        ledger = ForecastLedger(_DatastoreLedgerAdapter(client))

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
            return ResearchCompletion(
                "completed", research_result_id=result_id, usage_record_id=usage_id,
            )

        gateway = MaintenanceResearchJobGateway(
            jobs, authorize_assignment=authorize, capture_loader=load_capture,
            authorize_repair=authorize_repair, finalize_result=finalize_result,
        )
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
        worker = TwinWorker(
            jobs, worker_id=worker_id,
            reconcile_startup=lambda: not jobs.stale(now=now()),
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
