"""Private bounded worker primitives for autonomous twin maintenance and research."""
from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Mapping, Optional


class WorkerAuthenticationError(PermissionError):
    pass


class WorkerJobError(RuntimeError):
    pass


_GOOGLE_OIDC_ISSUERS = frozenset({"accounts.google.com", "https://accounts.google.com"})


@dataclass(frozen=True)
class WorkerOidcPrincipal:
    """Verified identity for a private Cloud Run worker request.

    Cloud Tasks headers are delivery metadata only.  They never establish this
    principal: the caller must present a Google-issued ID token for the exact
    Cloud Run service audience.
    """

    issuer: str
    audience: str
    service_account_email: str
    expires_at: datetime


def require_worker_oidc(
    principal: WorkerOidcPrincipal | None,
    *,
    expected_audience: str,
    allowed_service_accounts: frozenset[str],
    now: datetime,
) -> WorkerOidcPrincipal:
    """Accept only an unexpired Google OIDC principal for one worker route."""
    if now.tzinfo is None:
        raise WorkerAuthenticationError("worker authentication needs an aware time")
    if not expected_audience.strip() or not allowed_service_accounts:
        raise WorkerAuthenticationError("worker authentication is not configured")
    if principal is None:
        raise WorkerAuthenticationError("private worker requires a Google OIDC identity token")
    if principal.expires_at.tzinfo is None or principal.expires_at <= now:
        raise WorkerAuthenticationError("private worker identity token is expired")
    if principal.issuer not in _GOOGLE_OIDC_ISSUERS:
        raise WorkerAuthenticationError("private worker identity token has an invalid issuer")
    if principal.audience != expected_audience:
        raise WorkerAuthenticationError("private worker identity token has the wrong audience")
    if principal.service_account_email not in allowed_service_accounts:
        raise WorkerAuthenticationError("private worker service identity is not authorized")
    return principal


def verify_google_worker_oidc(
    token: str | None,
    *,
    expected_audience: str,
    allowed_service_accounts: frozenset[str],
    now: datetime,
    verifier: Callable[[str, str], Mapping[str, Any]] | None = None,
) -> WorkerOidcPrincipal:
    """Verify a Google ID token before applying narrow worker-route identity rules.

    ``verifier`` exists solely for deterministic tests.  Production calls the
    Google verifier with the exact expected audience; this code never derives
    authority from task or scheduler headers.
    """
    if not token:
        raise WorkerAuthenticationError("private worker requires a bearer token")
    if verifier is None:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.id_token import verify_oauth2_token
        except ImportError as exc:  # pragma: no cover - covered by serve dependency
            raise WorkerAuthenticationError("Google OIDC verification is unavailable") from exc

        def verifier(value: str, audience: str) -> Mapping[str, Any]:
            return verify_oauth2_token(value, Request(), audience=audience)

    try:
        claims = verifier(token, expected_audience)
        expires_at = datetime.fromtimestamp(float(claims["exp"]), tz=now.tzinfo)
        principal = WorkerOidcPrincipal(
            issuer=str(claims["iss"]),
            audience=str(claims["aud"]),
            service_account_email=str(claims["email"]),
            expires_at=expires_at,
        )
        if claims.get("email_verified") is not True:
            raise WorkerAuthenticationError("private worker identity email is not verified")
    except WorkerAuthenticationError:
        raise
    except Exception as exc:
        raise WorkerAuthenticationError("private worker identity token has invalid claims") from exc
    return require_worker_oidc(
        principal,
        expected_audience=expected_audience,
        allowed_service_accounts=allowed_service_accounts,
        now=now,
    )


class WorkerJobKind(str, Enum):
    RECOVERY = "recovery"
    RECONCILE = "reconcile"
    EXIT = "exit"
    RESEARCH = "research"


_PRIORITY = {WorkerJobKind.RECOVERY: 0, WorkerJobKind.RECONCILE: 1, WorkerJobKind.EXIT: 2, WorkerJobKind.RESEARCH: 3}


@dataclass(frozen=True)
class WorkerJob:
    id: str
    account_scope_id: str
    kind: WorkerJobKind
    payload: Mapping[str, str]
    deadline: datetime
    completed_result: Optional[Mapping[str, Any]] = None
    worker_id: Optional[str] = None
    lease_expires_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.account_scope_id.strip():
            raise WorkerJobError("worker jobs need stable IDs")
        if self.deadline.tzinfo is None:
            raise WorkerJobError("worker job deadlines must be timezone-aware")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.payload.items()):
            raise WorkerJobError("worker payloads contain stable string IDs only")
        forbidden = {"credential", "token", "secret", "url", "execute", "live"}
        if any(key.lower() in forbidden for key in self.payload):
            raise WorkerJobError("worker payload contains forbidden authority or credential data")


class InMemoryWorkerJobs:
    """Thread-safe test queue; production T17 replaces it with Cloud Tasks/Datastore."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, WorkerJob] = {}

    def add(self, job: WorkerJob) -> WorkerJob:
        with self._lock:
            existing = self._jobs.get(job.id)
            if existing is not None:
                return existing
            self._jobs[job.id] = job
            return job

    def claim(self, job_id: str, *, worker_id: str, now: datetime, lease_seconds: int = 30) -> Optional[WorkerJob]:
        if now.tzinfo is None or lease_seconds <= 0:
            raise WorkerJobError("claim needs an aware time and positive lease")
        with self._lock:
            job = self._jobs[job_id]
            if job.completed_result is not None or job.deadline <= now:
                return None
            if job.lease_expires_at is not None and job.lease_expires_at > now:
                return None
            claimed = replace(job, worker_id=worker_id, lease_expires_at=now + timedelta(seconds=lease_seconds))
            self._jobs[job_id] = claimed
            return claimed

    def complete(self, job_id: str, *, worker_id: str, result: Mapping[str, Any]) -> WorkerJob:
        with self._lock:
            job = self._jobs[job_id]
            if job.completed_result is not None:
                return job
            if job.worker_id != worker_id:
                raise WorkerJobError("stale worker cannot complete this job")
            completed = replace(job, completed_result=dict(result), lease_expires_at=None)
            self._jobs[job_id] = completed
            return completed

    def get(self, job_id: str) -> WorkerJob:
        with self._lock:
            return self._jobs[job_id]

    def due(self, *, now: datetime) -> tuple[WorkerJob, ...]:
        with self._lock:
            return tuple(sorted(
                (job for job in self._jobs.values() if job.completed_result is None and job.deadline > now),
                key=lambda job: (_PRIORITY[job.kind], job.deadline, job.id),
            ))


def require_worker_request(token: str | None, *, expected_token: str) -> None:
    if not expected_token or token != expected_token:
        raise WorkerAuthenticationError("private worker authentication failed")


class TwinWorker:
    """One-shot private handler; no background loop or trading capability."""

    def __init__(self, jobs: InMemoryWorkerJobs, *, worker_id: str, reconcile_startup: Callable[[], bool]) -> None:
        self._jobs, self._worker_id, self._reconcile_startup = jobs, worker_id, reconcile_startup
        self.execution_ready = False

    def start(self) -> bool:
        self.execution_ready = bool(self._reconcile_startup())
        return self.execution_ready

    def handle(
        self, job_id: str, *, now: datetime, maintain: Callable[[WorkerJob], Mapping[str, Any]],
        research: Callable[[WorkerJob], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        existing = self._jobs.get(job_id)
        if existing.completed_result is not None:
            return existing.completed_result
        job = self._jobs.claim(job_id, worker_id=self._worker_id, now=now)
        if job is None:
            current = self._jobs.get(job_id)
            return current.completed_result or {"status": "in_progress"}
        if job.kind is WorkerJobKind.RESEARCH:
            result = research(job)
        else:
            result = maintain(job)
        return self._jobs.complete(job.id, worker_id=self._worker_id, result=result).completed_result or {}
