"""Environment-built private worker app used by the two Cloud Run services."""
from __future__ import annotations

import os
import socket
from datetime import datetime, timezone

from ..observability import init_observability
from .budget import BudgetAlreadyClaimed, BudgetExceeded, DatastoreResearchBudget
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


def _maintenance_operation(jobs: DatastoreWorkerJobs, job: WorkerJob) -> dict[str, object]:
    """Safe staging operation until venue/account adapters are configured."""
    if job.kind is WorkerJobKind.RECOVERY:
        stale = jobs.stale(now=datetime.now(timezone.utc))
        return {"status": "complete", "stale_jobs_detected": len(stale)}
    if job.kind in {WorkerJobKind.RECONCILE, WorkerJobKind.EXIT}:
        raise WorkerDegraded("account_maintenance_adapter_unconfigured")
    raise WorkerPaused("unsupported_maintenance_job")


def _research_operation(_: ResearchAssignment) -> ResearchCompletion:
    raise WorkerDegraded("research_pipeline_unconfigured")


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
    worker_id = f"{role.value}-{socket.gethostname()}"

    if role is WorkerRole.MAINTENANCE:
        from google.cloud import datastore

        client = datastore.Client(project=_required("GOOGLE_CLOUD_PROJECT"))
        jobs = DatastoreWorkerJobs(client)
        budget = DatastoreResearchBudget(client)

        def authorize(assignment: ResearchAssignment) -> None:
            try:
                budget.claim(
                    assignment.budget_reservation_id, key=assignment.budget_key_id,
                )
            except BudgetAlreadyClaimed as exc:
                raise WorkerDegraded("research_budget_claim_unknown") from exc
            except (BudgetExceeded, KeyError, ValueError) as exc:
                raise WorkerDegraded("research_budget_unavailable") from exc

        gateway = MaintenanceResearchJobGateway(jobs, authorize_assignment=authorize)
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
            maintenance_operation=lambda job: _maintenance_operation(jobs, job),
            research_gateway=gateway,
        )
    else:
        gateway = HttpResearchJobGateway(
            _required("FORESEA_TWIN_MAINTENANCE_URL"),
            audience=_required("FORESEA_TWIN_MAINTENANCE_AUDIENCE"),
        )
        runtime = PrivateTwinRuntime(
            role, identities, now,
            research_worker=TwinResearchWorker(gateway, worker_id=worker_id),
            research_operation=_research_operation,
        )
    app = create_private_worker_app(runtime)
    init_observability(app)
    return app
