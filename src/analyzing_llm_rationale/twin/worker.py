"""Private bounded worker primitives for autonomous twin maintenance and research."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, Mapping, Optional, Protocol

from opentelemetry import metrics, trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
worker_operations = metrics.get_meter(__name__).create_counter(
    "twin.worker.operations", unit="1"
)
duplicate_suppressions = metrics.get_meter(__name__).create_counter(
    "twin.duplicate_suppressions", unit="1"
)
queue_lag_seconds = metrics.get_meter(__name__).create_histogram(
    "twin.queue.lag", unit="s"
)
retry_exhaustions = metrics.get_meter(__name__).create_counter(
    "twin.retries.exhausted", unit="1"
)


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
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,254}$")
_MAX_RESULT_BYTES = 64 * 1024
_RESEARCH_PAYLOAD_FIELDS = frozenset({
    "research_assignment_id", "budget_reservation_id", "market_snapshot_id",
    "evidence_set_id", "model_config_id", "budget_key_id",
})


class WorkerRole(str, Enum):
    MAINTENANCE = "maintenance"
    RESEARCH = "research"


class WorkerJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    PAUSED = "paused"
    EXPIRED = "expired"


class WorkerDegraded(RuntimeError):
    """A bounded dependency failure that should be stored, not retried forever."""

    def __init__(self, reason: str) -> None:
        if not _STABLE_ID.fullmatch(str(reason)):
            raise WorkerJobError("degradation reason must be a stable identifier")
        self.reason = str(reason)
        super().__init__(self.reason)


class WorkerPaused(RuntimeError):
    """A hard ambiguity that requires an operator or later reconciliation."""

    def __init__(self, reason: str) -> None:
        if not _STABLE_ID.fullmatch(str(reason)):
            raise WorkerJobError("pause reason must be a stable identifier")
        self.reason = str(reason)
        super().__init__(self.reason)


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
    fence: int = 0
    attempts: int = 0
    status: WorkerJobStatus = WorkerJobStatus.QUEUED
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_error: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise WorkerJobError("worker job schema is unsupported")
        if (
            not isinstance(self.id, str) or not _STABLE_ID.fullmatch(self.id)
            or not isinstance(self.account_scope_id, str)
            or not _STABLE_ID.fullmatch(self.account_scope_id)
        ):
            raise WorkerJobError("worker jobs need stable IDs")
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise WorkerJobError("worker job deadlines must be timezone-aware")
        try:
            object.__setattr__(self, "kind", WorkerJobKind(self.kind))
            object.__setattr__(self, "status", WorkerJobStatus(self.status))
        except ValueError as exc:
            raise WorkerJobError("worker job kind or status is unsupported") from exc
        if type(self.fence) is not int or self.fence < 0 or type(self.attempts) is not int or self.attempts < 0:
            raise WorkerJobError("worker job counters must be non-negative integers")
        if not isinstance(self.payload, Mapping):
            raise WorkerJobError("worker payload must be an object")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.payload.items()):
            raise WorkerJobError("worker payloads contain stable string IDs only")
        if any(
            not key.endswith("_id") or not _STABLE_ID.fullmatch(value)
            for key, value in self.payload.items()
        ):
            raise WorkerJobError("worker payloads may contain stable ID fields only")
        if self.kind is WorkerJobKind.RESEARCH and set(self.payload) != _RESEARCH_PAYLOAD_FIELDS:
            raise WorkerJobError("research job is missing its exact budgeted assignment IDs")
        for timestamp in (self.created_at, self.completed_at, self.lease_expires_at):
            if timestamp is not None and (timestamp.tzinfo is None or timestamp.utcoffset() is None):
                raise WorkerJobError("worker job timestamps must be timezone-aware")
        if self.completed_result is not None:
            _validated_result(self.completed_result)
        if self.status in {
            WorkerJobStatus.COMPLETED, WorkerJobStatus.DEGRADED, WorkerJobStatus.PAUSED,
        } and self.completed_result is None:
            raise WorkerJobError("finished worker job is missing its result")


def _validated_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise WorkerJobError("worker result must be an object")
    try:
        encoded = json.dumps(dict(result), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise WorkerJobError("worker result is not safely serializable") from exc
    if len(encoded.encode("utf-8")) > _MAX_RESULT_BYTES:
        raise WorkerJobError("worker result exceeds the durable size limit")
    return json.loads(encoded)


class WorkerJobs(Protocol):
    durable: bool

    def add(self, job: WorkerJob) -> WorkerJob: ...
    def claim(self, job_id: str, *, worker_id: str, now: datetime, lease_seconds: int = 30) -> Optional[WorkerJob]: ...
    def complete(
        self, job_id: str, *, worker_id: str, fence: int,
        result: Mapping[str, Any], now: datetime, degraded: bool = False,
        paused: bool = False,
    ) -> WorkerJob: ...
    def get(self, job_id: str) -> WorkerJob: ...
    def due(self, *, now: datetime) -> tuple[WorkerJob, ...]: ...
    def stale(self, *, now: datetime) -> tuple[WorkerJob, ...]: ...


class InMemoryWorkerJobs:
    """Thread-safe test queue; production T17 replaces it with Cloud Tasks/Datastore."""

    durable = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, WorkerJob] = {}

    def add(self, job: WorkerJob) -> WorkerJob:
        with self._lock:
            existing = self._jobs.get(job.id)
            if existing is not None:
                if existing.account_scope_id != job.account_scope_id or existing.kind is not job.kind or existing.payload != job.payload:
                    raise WorkerJobError("worker job ID was reused with different work")
                return existing
            stored = replace(job, created_at=job.created_at or datetime.now(timezone.utc))
            self._jobs[job.id] = stored
            return stored

    def claim(self, job_id: str, *, worker_id: str, now: datetime, lease_seconds: int = 30) -> Optional[WorkerJob]:
        if now.tzinfo is None or lease_seconds <= 0:
            raise WorkerJobError("claim needs an aware time and positive lease")
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise WorkerJobError("worker job was not found") from exc
            if job.completed_result is not None:
                return None
            if job.deadline <= now:
                self._jobs[job_id] = replace(job, status=WorkerJobStatus.EXPIRED)
                return None
            if job.lease_expires_at is not None and job.lease_expires_at > now:
                return None
            lease_expires_at = min(job.deadline, now + timedelta(seconds=lease_seconds))
            claimed = replace(
                job, worker_id=worker_id, lease_expires_at=lease_expires_at,
                fence=job.fence + 1, attempts=job.attempts + 1,
                status=WorkerJobStatus.RUNNING, last_error=None,
            )
            self._jobs[job_id] = claimed
            return claimed

    def complete(
        self, job_id: str, *, worker_id: str, fence: int,
        result: Mapping[str, Any], now: datetime, degraded: bool = False,
        paused: bool = False,
    ) -> WorkerJob:
        if degraded and paused:
            raise WorkerJobError("worker result cannot be both degraded and paused")
        if now.tzinfo is None or now.utcoffset() is None:
            raise WorkerJobError("completion needs an aware time")
        result = _validated_result(result)
        with self._lock:
            job = self._jobs[job_id]
            if job.completed_result is not None:
                return job
            if job.worker_id != worker_id or job.fence != fence:
                raise WorkerJobError("stale worker cannot complete this job")
            completed = replace(
                job, completed_result=result, lease_expires_at=None,
                completed_at=now,
                status=(
                    WorkerJobStatus.PAUSED if paused else
                    WorkerJobStatus.DEGRADED if degraded else WorkerJobStatus.COMPLETED
                ),
            )
            self._jobs[job_id] = completed
            return completed

    def get(self, job_id: str) -> WorkerJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise WorkerJobError("worker job was not found") from exc

    def due(self, *, now: datetime) -> tuple[WorkerJob, ...]:
        with self._lock:
            return tuple(sorted(
                (
                    job for job in self._jobs.values()
                    if job.completed_result is None and job.deadline > now
                    and (job.lease_expires_at is None or job.lease_expires_at <= now)
                ),
                key=lambda job: (_PRIORITY[job.kind], job.deadline, job.id),
            ))

    def stale(self, *, now: datetime) -> tuple[WorkerJob, ...]:
        with self._lock:
            return tuple(sorted((
                job for job in self._jobs.values()
                if job.completed_result is None and (
                    job.deadline <= now or (
                        job.status is WorkerJobStatus.RUNNING
                        and job.lease_expires_at is not None
                        and job.lease_expires_at <= now
                    )
                )
            ), key=lambda job: (job.deadline, job.id)))


class DatastoreWorkerJobs:
    """Strong-key durable claims; global due scans only enqueue idempotent work."""

    durable = True

    def __init__(self, client: Any) -> None:
        self._client = client

    def _key(self, job_id: str):
        return self._client.key("TwinWorkerJob", job_id)

    @staticmethod
    def _identity(job: WorkerJob) -> str:
        encoded = json.dumps({
            "id": job.id, "account_scope_id": job.account_scope_id,
            "kind": job.kind.value, "payload": dict(job.payload),
            "deadline": job.deadline.isoformat(), "schema_version": job.schema_version,
        }, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode()).hexdigest()

    @classmethod
    def _from_entity(cls, entity: Any) -> WorkerJob:
        try:
            result = json.loads(str(entity["result_json"])) if entity.get("result_json") else None
            job = WorkerJob(
                id=str(entity.key.name), account_scope_id=str(entity["account_scope_id"]),
                kind=WorkerJobKind(str(entity["kind"])),
                payload=json.loads(str(entity["payload_json"])), deadline=entity["deadline"],
                completed_result=result, worker_id=entity.get("worker_id"),
                lease_expires_at=entity.get("lease_expires_at"), fence=int(entity.get("fence", 0)),
                attempts=int(entity.get("attempts", 0)), status=WorkerJobStatus(str(entity["status"])),
                created_at=entity.get("created_at"), completed_at=entity.get("completed_at"),
                last_error=entity.get("last_error"), schema_version=int(entity["schema_version"]),
            )
            if entity.get("identity_hash") != cls._identity(job):
                raise WorkerJobError("durable worker job identity failed validation")
            return job
        except WorkerJobError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkerJobError("durable worker job is malformed") from exc

    @classmethod
    def _entity(cls, job: WorkerJob, key: Any):
        from google.cloud import datastore

        entity = datastore.Entity(key=key, exclude_from_indexes=("payload_json", "result_json"))
        entity.update({
            "account_scope_id": job.account_scope_id, "kind": job.kind.value,
            "payload_json": json.dumps(dict(job.payload), sort_keys=True, separators=(",", ":")),
            "deadline": job.deadline, "result_json": (
                json.dumps(dict(job.completed_result), sort_keys=True, separators=(",", ":"))
                if job.completed_result is not None else None
            ),
            "worker_id": job.worker_id, "lease_expires_at": job.lease_expires_at,
            "fence": job.fence, "attempts": job.attempts, "status": job.status.value,
            "created_at": job.created_at, "completed_at": job.completed_at,
            "last_error": job.last_error, "schema_version": job.schema_version,
            "identity_hash": cls._identity(job),
        })
        return entity

    def add(self, job: WorkerJob) -> WorkerJob:
        key = self._key(job.id)
        with self._client.transaction():
            existing = self._client.get(key)
            if existing is not None:
                stored = self._from_entity(existing)
                if self._identity(stored) != self._identity(job):
                    raise WorkerJobError("worker job ID was reused with different work")
                return stored
            stored = replace(job, created_at=job.created_at or datetime.now(timezone.utc))
            self._client.put(self._entity(stored, key))
            return stored

    def get(self, job_id: str) -> WorkerJob:
        entity = self._client.get(self._key(job_id))
        if entity is None:
            raise WorkerJobError("worker job was not found")
        return self._from_entity(entity)

    def claim(self, job_id: str, *, worker_id: str, now: datetime, lease_seconds: int = 30) -> Optional[WorkerJob]:
        if now.tzinfo is None or now.utcoffset() is None or lease_seconds <= 0:
            raise WorkerJobError("claim needs an aware time and positive lease")
        key = self._key(job_id)
        with self._client.transaction():
            entity = self._client.get(key)
            if entity is None:
                raise WorkerJobError("worker job was not found")
            job = self._from_entity(entity)
            if job.completed_result is not None:
                return None
            if job.deadline <= now:
                expired = replace(job, status=WorkerJobStatus.EXPIRED, lease_expires_at=None)
                self._client.put(self._entity(expired, key))
                return None
            if job.lease_expires_at is not None and job.lease_expires_at > now:
                return None
            claimed = replace(
                job, worker_id=worker_id,
                lease_expires_at=min(job.deadline, now + timedelta(seconds=lease_seconds)),
                fence=job.fence + 1, attempts=job.attempts + 1,
                status=WorkerJobStatus.RUNNING, last_error=None,
            )
            self._client.put(self._entity(claimed, key))
            return claimed

    def complete(
        self, job_id: str, *, worker_id: str, fence: int,
        result: Mapping[str, Any], now: datetime, degraded: bool = False,
        paused: bool = False,
    ) -> WorkerJob:
        if degraded and paused:
            raise WorkerJobError("worker result cannot be both degraded and paused")
        if now.tzinfo is None or now.utcoffset() is None:
            raise WorkerJobError("completion needs an aware time")
        result = _validated_result(result)
        key = self._key(job_id)
        with self._client.transaction():
            entity = self._client.get(key)
            if entity is None:
                raise WorkerJobError("worker job was not found")
            job = self._from_entity(entity)
            if job.completed_result is not None:
                return job
            if job.worker_id != worker_id or job.fence != fence:
                raise WorkerJobError("stale worker cannot complete this job")
            completed = replace(
                job, completed_result=result, completed_at=now, lease_expires_at=None,
                status=(
                    WorkerJobStatus.PAUSED if paused else
                    WorkerJobStatus.DEGRADED if degraded else WorkerJobStatus.COMPLETED
                ),
            )
            self._client.put(self._entity(completed, key))
            return completed

    def due(self, *, now: datetime) -> tuple[WorkerJob, ...]:
        query = self._client.query(kind="TwinWorkerJob")
        jobs = (self._from_entity(entity) for entity in query.fetch())
        return tuple(sorted((
            job for job in jobs
            if job.completed_result is None and job.deadline > now
            and (job.lease_expires_at is None or job.lease_expires_at <= now)
        ), key=lambda job: (_PRIORITY[job.kind], job.deadline, job.id)))

    def stale(self, *, now: datetime) -> tuple[WorkerJob, ...]:
        query = self._client.query(kind="TwinWorkerJob")
        jobs = (self._from_entity(entity) for entity in query.fetch())
        return tuple(sorted((
            job for job in jobs
            if job.completed_result is None and (
                job.deadline <= now or (
                    job.status is WorkerJobStatus.RUNNING
                    and job.lease_expires_at is not None
                    and job.lease_expires_at <= now
                )
            )
        ), key=lambda job: (job.deadline, job.id)))


def require_worker_request(token: str | None, *, expected_token: str) -> None:
    if not expected_token or token != expected_token:
        raise WorkerAuthenticationError("private worker authentication failed")


class TwinWorker:
    """One-shot private handler; no background loop or trading capability."""

    def __init__(
        self, jobs: WorkerJobs, *, worker_id: str,
        reconcile_startup: Callable[[], bool], role: WorkerRole = WorkerRole.MAINTENANCE,
    ) -> None:
        if WorkerRole(role) is not WorkerRole.MAINTENANCE:
            raise WorkerJobError("research must use the narrow TwinResearchWorker boundary")
        self._jobs, self._worker_id, self._reconcile_startup = jobs, worker_id, reconcile_startup
        self.execution_ready = False
        self.accepting_work = False

    def start(self) -> bool:
        try:
            self.execution_ready = bool(self._reconcile_startup())
        except Exception as exc:
            self.execution_ready = False
            self.accepting_work = False
            logger.error("Twin maintenance startup reconciliation failed (%s)", type(exc).__name__)
            worker_operations.add(1, {"role": "maintenance", "outcome": "startup_failed"})
            return False
        self.accepting_work = True
        return self.execution_ready

    def shutdown(self) -> None:
        self.accepting_work = False
        self.execution_ready = False

    @tracer.start_as_current_span("twin.worker.maintenance")
    def handle(
        self, job_id: str, *, now: datetime,
        maintain: Callable[[WorkerJob], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if not self.accepting_work:
            worker_operations.add(1, {"role": "maintenance", "outcome": "draining"})
            return {"status": "draining"}
        existing = self._jobs.get(job_id)
        if existing.completed_result is not None:
            worker_operations.add(1, {"role": "maintenance", "outcome": "duplicate"})
            duplicate_suppressions.add(1, {"operation": "worker_delivery", "role": "maintenance"})
            return existing.completed_result
        if existing.kind is WorkerJobKind.RESEARCH:
            raise WorkerJobError("worker role cannot process this job kind")
        job = self._jobs.claim(job_id, worker_id=self._worker_id, now=now)
        if job is None:
            current = self._jobs.get(job_id)
            if current.status is WorkerJobStatus.EXPIRED:
                worker_operations.add(1, {"role": "maintenance", "outcome": "expired"})
                return {"status": "expired"}
            worker_operations.add(1, {"role": "maintenance", "outcome": "in_progress"})
            return current.completed_result or {"status": "in_progress"}
        if job.created_at is not None:
            queue_lag_seconds.record(
                max(0.0, (now - job.created_at).total_seconds()),
                {"role": "maintenance", "kind": job.kind.value},
            )
        try:
            result = maintain(job)
            degraded = False
        except WorkerDegraded as exc:
            result = {"status": "degraded", "reason": exc.reason}
            degraded = True
            paused = False
        except WorkerPaused as exc:
            result = {"status": "paused", "reason": exc.reason}
            degraded = False
            paused = True
        else:
            paused = False
        completed = self._jobs.complete(
            job.id, worker_id=self._worker_id, fence=job.fence,
            result=result, now=now, degraded=degraded, paused=paused,
        )
        worker_operations.add(1, {"role": "maintenance", "outcome": completed.status.value})
        return completed.completed_result or {}


def bounded_safe_read(
    operation: Callable[[], Any], *, attempts: int = 3,
    base_delay_seconds: float = 0.1, sleep: Callable[[float], None] = time.sleep,
    degradation_reason: str = "data_unavailable",
) -> Any:
    """Retry a read-only dependency call with a small, explicit budget."""
    if attempts < 1 or attempts > 5 or base_delay_seconds < 0:
        raise WorkerJobError("safe-read retry policy is outside its bounded range")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = operation()
            worker_operations.add(1, {"role": "dependency_read", "outcome": "success"})
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "bounded worker dependency read failed",
                extra={"attempt": attempt + 1, "max_attempts": attempts},
            )
            if attempt + 1 < attempts:
                sleep(base_delay_seconds * (2 ** attempt))
    worker_operations.add(1, {"role": "dependency_read", "outcome": "degraded"})
    retry_exhaustions.add(1, {"operation": "dependency_read"})
    raise WorkerDegraded(degradation_reason) from last_error


@dataclass(frozen=True)
class ResearchAssignment:
    job_id: str
    worker_id: str
    fence: int
    deadline: datetime
    research_assignment_id: str
    budget_reservation_id: str
    market_snapshot_id: str
    evidence_set_id: str
    model_config_id: str
    budget_key_id: str

    @classmethod
    def from_job(cls, job: WorkerJob) -> "ResearchAssignment":
        if job.kind is not WorkerJobKind.RESEARCH or job.worker_id is None:
            raise WorkerJobError("research assignment requires a claimed research job")
        if set(job.payload) != _RESEARCH_PAYLOAD_FIELDS:
            raise WorkerJobError("research assignment payload is incomplete")
        return cls(
            job.id, job.worker_id, job.fence, job.deadline,
            job.payload["research_assignment_id"], job.payload["budget_reservation_id"],
            job.payload["market_snapshot_id"], job.payload["evidence_set_id"],
            job.payload["model_config_id"], job.payload["budget_key_id"],
        )


@dataclass(frozen=True)
class ResearchCompletion:
    status: str
    research_result_id: Optional[str] = None
    usage_record_id: Optional[str] = None
    reason: Optional[str] = None
    result_payload: Optional[Mapping[str, Any]] = None
    actual_usd: Optional[str] = None
    actual_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "degraded"}:
            raise WorkerJobError("research completion status is unsupported")
        for value in (self.research_result_id, self.usage_record_id, self.reason):
            if value is not None and (
                not isinstance(value, str) or not _STABLE_ID.fullmatch(value)
            ):
                raise WorkerJobError("research completion contains an invalid identifier")
        if self.status == "completed" and (
            self.reason is not None or (
                self.result_payload is None and (
                    self.research_result_id is None or self.usage_record_id is None
                )
            )
        ):
            raise WorkerJobError("completed research requires result and usage IDs")
        if self.status == "degraded" and (self.reason is None or self.research_result_id is not None):
            raise WorkerJobError("degraded research requires only a stable reason")
        if (self.actual_usd is None) != (self.actual_tokens is None):
            raise WorkerJobError("research usage must include both USD and tokens")
        if self.actual_tokens is not None and (
            type(self.actual_tokens) is not int or self.actual_tokens < 0
        ):
            raise WorkerJobError("research token usage is invalid")

    def to_mapping(self, *, include_transport: bool = False) -> dict[str, Any]:
        values = {
            "status": self.status, "research_result_id": self.research_result_id,
            "usage_record_id": self.usage_record_id, "reason": self.reason,
        }
        if include_transport:
            values.update({
                "result_payload": self.result_payload,
                "actual_usd": self.actual_usd,
                "actual_tokens": self.actual_tokens,
            })
        return {
            key: value for key, value in values.items() if value is not None
        }


class ResearchJobGateway(Protocol):
    """Narrow maintenance-owned API implemented remotely for the research role."""

    def completed_result(self, job_id: str) -> Optional[Mapping[str, Any]]: ...
    def claim(self, job_id: str, *, worker_id: str, now: datetime) -> Optional[ResearchAssignment]: ...
    def load_capture(self, assignment: ResearchAssignment) -> Mapping[str, Any]: ...
    def complete(
        self, assignment: ResearchAssignment, result: ResearchCompletion, *, now: datetime,
    ) -> Mapping[str, Any]: ...


class MaintenanceResearchJobGateway:
    """Maintenance-side adapter; the research process receives only this API over HTTP."""

    def __init__(
        self, jobs: WorkerJobs, *,
        authorize_assignment: Callable[[ResearchAssignment], None],
        capture_loader: Optional[Callable[[ResearchAssignment], Mapping[str, Any]]] = None,
        finalize_result: Optional[
            Callable[[ResearchAssignment, ResearchCompletion], ResearchCompletion]
        ] = None,
    ) -> None:
        self._jobs = jobs
        self._authorize_assignment = authorize_assignment
        self._capture_loader = capture_loader
        self._finalize_result = finalize_result

    def completed_result(self, job_id: str) -> Optional[Mapping[str, Any]]:
        job = self._jobs.get(job_id)
        if job.kind is not WorkerJobKind.RESEARCH:
            raise WorkerJobError("job is not a research assignment")
        return job.completed_result

    def claim(self, job_id: str, *, worker_id: str, now: datetime) -> Optional[ResearchAssignment]:
        current = self._jobs.get(job_id)
        if current.kind is not WorkerJobKind.RESEARCH:
            raise WorkerJobError("job is not a research assignment")
        claimed = self._jobs.claim(job_id, worker_id=worker_id, now=now)
        if claimed is None:
            return None
        assignment = ResearchAssignment.from_job(claimed)
        try:
            self._authorize_assignment(assignment)
        except WorkerDegraded as exc:
            self._jobs.complete(
                assignment.job_id, worker_id=assignment.worker_id,
                fence=assignment.fence,
                result={"status": "degraded", "reason": exc.reason},
                now=now, degraded=True,
            )
            return None
        return assignment

    def complete(
        self, assignment: ResearchAssignment, result: ResearchCompletion, *, now: datetime,
    ) -> Mapping[str, Any]:
        if self._finalize_result is not None:
            result = self._finalize_result(assignment, result)
        completed = self._jobs.complete(
            assignment.job_id, worker_id=assignment.worker_id, fence=assignment.fence,
            result=result.to_mapping(), now=now, degraded=result.status == "degraded",
        )
        return completed.completed_result or {}

    def load_capture(self, assignment: ResearchAssignment) -> Mapping[str, Any]:
        current = ResearchAssignment.from_job(self._jobs.get(assignment.job_id))
        if current != assignment:
            raise WorkerJobError("research capture request has a stale assignment")
        if self._capture_loader is None:
            raise WorkerDegraded("research_capture_unavailable")
        capture = self._capture_loader(assignment)
        if not isinstance(capture, Mapping):
            raise WorkerJobError("research capture loader returned an invalid payload")
        return capture


class TwinResearchWorker:
    """Research process boundary with no trading store or execution callback."""

    def __init__(self, gateway: ResearchJobGateway, *, worker_id: str) -> None:
        self._gateway = gateway
        self._worker_id = worker_id
        self.accepting_work = True

    def shutdown(self) -> None:
        self.accepting_work = False

    @tracer.start_as_current_span("twin.worker.research")
    def handle(
        self, job_id: str, *, now: datetime,
        research: Callable[[ResearchAssignment], ResearchCompletion],
    ) -> Mapping[str, Any]:
        if not self.accepting_work:
            worker_operations.add(1, {"role": "research", "outcome": "draining"})
            return {"status": "draining"}
        existing = self._gateway.completed_result(job_id)
        if existing is not None:
            worker_operations.add(1, {"role": "research", "outcome": "duplicate"})
            duplicate_suppressions.add(1, {"operation": "worker_delivery", "role": "research"})
            return existing
        assignment = self._gateway.claim(job_id, worker_id=self._worker_id, now=now)
        if assignment is None:
            worker_operations.add(1, {"role": "research", "outcome": "in_progress"})
            return self._gateway.completed_result(job_id) or {"status": "in_progress"}
        try:
            result = research(assignment)
        except WorkerDegraded as exc:
            result = ResearchCompletion("degraded", reason=exc.reason)
        if not isinstance(result, ResearchCompletion):
            raise WorkerJobError("research worker returned an untyped result")
        completed = self._gateway.complete(assignment, result, now=now)
        worker_operations.add(1, {"role": "research", "outcome": result.status})
        return completed
