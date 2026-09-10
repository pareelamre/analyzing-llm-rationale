"""Environment-built private worker app used by the two Cloud Run services."""
from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from ..observability import init_observability
from .budget import BudgetAlreadyClaimed, BudgetExceeded, DatastoreResearchBudget
from .research_gateway import (
    DatastoreResearchCaptureStore,
    ResearchRuntimePolicy,
    load_research_runtime_policy,
    public_evidence_set_id,
    research_capture_payload,
    restore_research_capture,
)
from .runtime import (
    HttpResearchJobGateway,
    PrivateTwinRuntime,
    RuntimeConfigurationError,
    RuntimeIdentityPolicy,
    create_private_worker_app,
)
from .scheduler import CloudTasksConfig, CloudTasksDispatcher
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
        recovered += 1
    return recovered


def _maintenance_operation(
    jobs: DatastoreWorkerJobs, budget: DatastoreResearchBudget, job: WorkerJob,
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
    raise WorkerPaused("unsupported_maintenance_job")


def _research_operation(
    assignment: ResearchAssignment, policy: ResearchRuntimePolicy,
    gateway: HttpResearchJobGateway,
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
    raise WorkerDegraded("research_result_transport_unavailable")


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

    if role is WorkerRole.MAINTENANCE:
        from google.cloud import datastore

        client = datastore.Client(project=_required("GOOGLE_CLOUD_PROJECT"))
        jobs = DatastoreWorkerJobs(client)
        budget = DatastoreResearchBudget(client)
        captures = DatastoreResearchCaptureStore(client)

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

        gateway = MaintenanceResearchJobGateway(
            jobs, authorize_assignment=authorize, capture_loader=load_capture,
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
            maintenance_worker=worker,
            maintenance_operation=lambda job: _maintenance_operation(jobs, budget, job),
            research_gateway=gateway,
        )
    else:
        # Startup validates the configured secret, model identity, endpoint,
        # timeout and price before this revision can report ready.
        research_policy.build_provider()
        gateway = HttpResearchJobGateway(
            _required("FORESEA_TWIN_MAINTENANCE_URL"),
            audience=_required("FORESEA_TWIN_MAINTENANCE_AUDIENCE"),
        )
        runtime = PrivateTwinRuntime(
            role, identities, now,
            research_worker=TwinResearchWorker(gateway, worker_id=worker_id),
            research_operation=lambda assignment: _research_operation(
                assignment, research_policy, gateway,
            ),
        )
    app = create_private_worker_app(runtime)
    init_observability(app)
    return app
