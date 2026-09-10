"""Role-scoped FastAPI surface for the private autonomous-twin services."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Callable, Mapping, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .scheduler import TaskDispatcher, dispatch_due_jobs
from .worker import (
    MaintenanceResearchJobGateway,
    ResearchAssignment,
    ResearchCompletion,
    TwinResearchWorker,
    TwinWorker,
    WorkerAuthenticationError,
    WorkerJobError,
    WorkerJobs,
    WorkerRole,
    verify_google_worker_oidc,
)


class RuntimeConfigurationError(RuntimeError):
    pass


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$")


class ResearchResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fence: int = Field(ge=1)
    status: str
    research_result_id: Optional[str] = None
    usage_record_id: Optional[str] = None
    reason: Optional[str] = None
    result_payload: Optional[dict[str, Any]] = None
    actual_usd: Optional[str] = None
    actual_tokens: Optional[int] = Field(default=None, ge=0)
    repair_attempted: bool = False
    repair_actual_usd: Optional[str] = None
    repair_actual_tokens: Optional[int] = Field(default=None, ge=0)


class ResearchRepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fence: int = Field(ge=1)
    actual_usd: Optional[str] = None
    actual_tokens: Optional[int] = Field(default=None, ge=0)


TokenVerifier = Callable[[str, str], Mapping[str, Any]]
TokenFetcher = Callable[[str], str]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class RuntimeIdentityPolicy:
    audience: str
    scheduler_accounts: frozenset[str]
    dispatcher_accounts: frozenset[str]
    research_accounts: frozenset[str]

    def __post_init__(self) -> None:
        if not self.audience.startswith("https://"):
            raise RuntimeConfigurationError("private worker audience must use HTTPS")
        if not self.dispatcher_accounts:
            raise RuntimeConfigurationError("a dispatcher service identity is required")


class HttpResearchJobGateway:
    """Research-side client for the narrow maintenance claim/result interface."""

    def __init__(
        self, base_url: str, *, audience: str, timeout_seconds: float = 10,
        session: Any = requests, token_fetcher: Optional[TokenFetcher] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._audience = audience
        self._timeout = float(timeout_seconds)
        self._session = session
        self._token_fetcher = token_fetcher or self._google_token
        if not self._base_url.startswith("https://") or not audience.startswith("https://"):
            raise RuntimeConfigurationError("research gateway requires HTTPS URLs")
        if not 0 < self._timeout <= 30:
            raise RuntimeConfigurationError("research gateway timeout must be within 30 seconds")

    @staticmethod
    def _google_token(audience: str) -> str:
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2.id_token import fetch_id_token

        return fetch_id_token(GoogleRequest(), audience)

    def _call(self, method: str, path: str, *, body: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        token = self._token_fetcher(self._audience)
        try:
            response = self._session.request(
                method, self._base_url + path,
                headers={"Authorization": f"Bearer {token}"},
                json=dict(body) if body is not None else None,
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise WorkerJobError("maintenance research gateway request failed") from exc
        if not isinstance(payload, Mapping):
            raise WorkerJobError("maintenance research gateway returned an invalid body")
        return payload

    def completed_result(self, job_id: str) -> Optional[Mapping[str, Any]]:
        payload = self._call("GET", f"/internal/twin/research-jobs/{job_id}")
        result = payload.get("completed_result")
        if result is not None and not isinstance(result, Mapping):
            raise WorkerJobError("maintenance returned an invalid research result")
        return result

    def claim(self, job_id: str, *, worker_id: str, now: datetime) -> Optional[ResearchAssignment]:
        del worker_id, now  # Maintenance derives identity and time from the authenticated request.
        payload = self._call("POST", f"/internal/twin/research-jobs/{job_id}/claim")
        if payload.get("status") != "claimed":
            return None
        assignment = payload.get("assignment")
        if not isinstance(assignment, Mapping):
            raise WorkerJobError("maintenance omitted the research assignment")
        try:
            return ResearchAssignment(**dict(assignment))
        except (TypeError, ValueError) as exc:
            raise WorkerJobError("maintenance returned an invalid research assignment") from exc

    def load_capture(self, assignment: ResearchAssignment) -> Mapping[str, Any]:
        payload = self._call(
            "GET", f"/internal/twin/research-jobs/{assignment.job_id}/capture",
        )
        capture = payload.get("capture")
        if not isinstance(capture, Mapping):
            raise WorkerJobError("maintenance omitted the research capture")
        return capture

    def authorize_repair(
        self, assignment: ResearchAssignment, *, actual_usd: Optional[str],
        actual_tokens: Optional[int],
    ) -> bool:
        payload = self._call(
            "POST", f"/internal/twin/research-jobs/{assignment.job_id}/repair",
            body={
                "fence": assignment.fence,
                "actual_usd": actual_usd,
                "actual_tokens": actual_tokens,
            },
        )
        return payload.get("status") == "authorized"

    def complete(
        self, assignment: ResearchAssignment, result: ResearchCompletion, *, now: datetime,
    ) -> Mapping[str, Any]:
        del now  # Maintenance timestamps the durable result.
        payload = self._call(
            "POST", f"/internal/twin/research-jobs/{assignment.job_id}/result",
            body={"fence": assignment.fence, **result.to_mapping(include_transport=True)},
        )
        return payload


@dataclass
class PrivateTwinRuntime:
    role: WorkerRole
    identities: RuntimeIdentityPolicy
    clock: Clock
    jobs: Optional[WorkerJobs] = None
    dispatcher: Optional[TaskDispatcher] = None
    maintenance_worker: Optional[TwinWorker] = None
    maintenance_operation: Optional[Callable[[Any], Mapping[str, Any]]] = None
    research_gateway: Optional[MaintenanceResearchJobGateway] = None
    research_worker: Optional[TwinResearchWorker] = None
    research_operation: Optional[Callable[[ResearchAssignment], ResearchCompletion]] = None
    token_verifier: Optional[TokenVerifier] = None

    def __post_init__(self) -> None:
        self.role = WorkerRole(self.role)
        if self.role is WorkerRole.MAINTENANCE:
            required = (
                self.jobs, self.dispatcher, self.maintenance_worker,
                self.maintenance_operation, self.research_gateway,
            )
        else:
            required = (self.research_worker, self.research_operation)
        if any(value is None for value in required):
            raise RuntimeConfigurationError(f"{self.role.value} runtime bindings are incomplete")

    def start(self) -> bool:
        if self.role is WorkerRole.MAINTENANCE:
            return bool(self.maintenance_worker and self.maintenance_worker.start())
        return True

    def shutdown(self) -> None:
        if self.maintenance_worker is not None:
            self.maintenance_worker.shutdown()
        if self.research_worker is not None:
            self.research_worker.shutdown()

    @property
    def ready(self) -> bool:
        if self.role is WorkerRole.MAINTENANCE:
            return bool(self.maintenance_worker and self.maintenance_worker.execution_ready)
        return bool(self.research_worker and self.research_worker.accepting_work)

    def authenticate(self, request: Request, allowed: frozenset[str]) -> str:
        header = request.headers.get("authorization", "")
        scheme, separator, token = header.partition(" ")
        if scheme.lower() != "bearer" or not separator or not token.strip():
            raise WorkerAuthenticationError("private worker requires a bearer token")
        principal = verify_google_worker_oidc(
            token.strip(), expected_audience=self.identities.audience,
            allowed_service_accounts=allowed, now=self.clock(),
            verifier=self.token_verifier,
        )
        return principal.service_account_email

    @staticmethod
    def research_worker_id(service_account_email: str) -> str:
        return "research-" + sha256(service_account_email.encode()).hexdigest()[:24]


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, WorkerAuthenticationError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, WorkerJobError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=503, detail="Private twin runtime is unavailable.")


def create_private_worker_app(runtime: PrivateTwinRuntime) -> FastAPI:
    """Create a standalone worker app with no public or trading routes."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(title=f"Foresea twin {runtime.role.value} worker", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "role": runtime.role.value}

    @app.get("/ready")
    async def ready():
        payload = {"status": "ready" if runtime.ready else "unready", "role": runtime.role.value}
        return payload if runtime.ready else JSONResponse(payload, status_code=503)

    if runtime.role is WorkerRole.MAINTENANCE:

        @app.post("/internal/twin/dispatch")
        async def dispatch(request: Request):
            try:
                runtime.authenticate(request, runtime.identities.scheduler_accounts)
                names = dispatch_due_jobs(
                    runtime.jobs, runtime.dispatcher, now=runtime.clock(), limit=25,
                )
                return {"status": "complete", "tasks_enqueued": len(names)}
            except Exception as exc:
                raise _http_error(exc) from exc

        @app.post("/internal/twin/maintain")
        async def maintain(body: JobRequest, request: Request):
            try:
                runtime.authenticate(request, runtime.identities.dispatcher_accounts)
                return runtime.maintenance_worker.handle(
                    body.job_id, now=runtime.clock(), maintain=runtime.maintenance_operation,
                )
            except Exception as exc:
                raise _http_error(exc) from exc

        @app.get("/internal/twin/research-jobs/{job_id}")
        async def research_status(job_id: str, request: Request):
            try:
                runtime.authenticate(request, runtime.identities.research_accounts)
                completed_result = runtime.research_gateway.completed_result(job_id)
                job = runtime.jobs.get(job_id)
                return {
                    "status": job.status.value,
                    "completed_result": completed_result,
                }
            except Exception as exc:
                raise _http_error(exc) from exc

        @app.post("/internal/twin/research-jobs/{job_id}/claim")
        async def research_claim(job_id: str, request: Request):
            try:
                email = runtime.authenticate(request, runtime.identities.research_accounts)
                assignment = runtime.research_gateway.claim(
                    job_id, worker_id=runtime.research_worker_id(email), now=runtime.clock(),
                )
                if assignment is None:
                    job = runtime.jobs.get(job_id)
                    return {
                        "status": job.status.value,
                        "completed_result": job.completed_result,
                    }
                return {"status": "claimed", "assignment": asdict(assignment)}
            except Exception as exc:
                raise _http_error(exc) from exc

        @app.post("/internal/twin/research-jobs/{job_id}/result")
        async def research_result(job_id: str, body: ResearchResultRequest, request: Request):
            try:
                email = runtime.authenticate(request, runtime.identities.research_accounts)
                job = runtime.jobs.get(job_id)
                assignment = ResearchAssignment.from_job(job)
                if assignment.worker_id != runtime.research_worker_id(email):
                    raise WorkerAuthenticationError("research claim belongs to another identity")
                if assignment.fence != body.fence:
                    raise WorkerJobError("research result has a stale fence")
                result = ResearchCompletion(
                    body.status, research_result_id=body.research_result_id,
                    usage_record_id=body.usage_record_id, reason=body.reason,
                    result_payload=body.result_payload, actual_usd=body.actual_usd,
                    actual_tokens=body.actual_tokens,
                    repair_attempted=body.repair_attempted,
                    repair_actual_usd=body.repair_actual_usd,
                    repair_actual_tokens=body.repair_actual_tokens,
                )
                completed = runtime.research_gateway.complete(
                    assignment, result, now=runtime.clock(),
                )
                return completed
            except Exception as exc:
                raise _http_error(exc) from exc

        @app.post("/internal/twin/research-jobs/{job_id}/repair")
        async def research_repair(job_id: str, body: ResearchRepairRequest, request: Request):
            try:
                email = runtime.authenticate(request, runtime.identities.research_accounts)
                assignment = ResearchAssignment.from_job(runtime.jobs.get(job_id))
                if assignment.worker_id != runtime.research_worker_id(email):
                    raise WorkerAuthenticationError("research claim belongs to another identity")
                if assignment.fence != body.fence:
                    raise WorkerJobError("research repair has a stale fence")
                authorized = runtime.research_gateway.authorize_repair(
                    assignment, actual_usd=body.actual_usd,
                    actual_tokens=body.actual_tokens,
                )
                return {"status": "authorized" if authorized else "denied"}
            except Exception as exc:
                raise _http_error(exc) from exc

        @app.get("/internal/twin/research-jobs/{job_id}/capture")
        async def research_capture(job_id: str, request: Request):
            try:
                email = runtime.authenticate(request, runtime.identities.research_accounts)
                assignment = ResearchAssignment.from_job(runtime.jobs.get(job_id))
                if assignment.worker_id != runtime.research_worker_id(email):
                    raise WorkerAuthenticationError("research claim belongs to another identity")
                capture = runtime.research_gateway.load_capture(assignment)
                return {"capture": capture}
            except Exception as exc:
                raise _http_error(exc) from exc

    else:

        @app.post("/internal/twin/research")
        async def research(body: JobRequest, request: Request):
            try:
                runtime.authenticate(request, runtime.identities.dispatcher_accounts)
                return runtime.research_worker.handle(
                    body.job_id, now=runtime.clock(), research=runtime.research_operation,
                )
            except Exception as exc:
                raise _http_error(exc) from exc

    return app
