"""Durable phase state for asynchronous private shadow strategy cycles."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping, Optional, Protocol

from opentelemetry import metrics, trace

from .models import Instrument, MarketSnapshot
from .strategy import StrategyCandidate

tracer = trace.get_tracer(__name__)
cycle_phase_transitions = metrics.get_meter(__name__).create_counter(
    "twin.strategy.phase_transitions", unit="1",
)


class StrategyRunError(RuntimeError):
    pass


class StrategyRunPhase(str, Enum):
    QUEUED = "queued"
    RESEARCH_PENDING = "research_pending"
    READY = "ready"
    COMPLETE = "complete"
    BLOCKED = "blocked"


_TERMINAL_PHASES = {StrategyRunPhase.COMPLETE, StrategyRunPhase.BLOCKED}
_ALLOWED_TRANSITIONS = {
    StrategyRunPhase.QUEUED: {
        StrategyRunPhase.RESEARCH_PENDING, StrategyRunPhase.COMPLETE,
        StrategyRunPhase.BLOCKED,
    },
    StrategyRunPhase.RESEARCH_PENDING: {
        StrategyRunPhase.READY, StrategyRunPhase.BLOCKED,
    },
    StrategyRunPhase.READY: {
        StrategyRunPhase.COMPLETE, StrategyRunPhase.BLOCKED,
    },
    StrategyRunPhase.COMPLETE: set(),
    StrategyRunPhase.BLOCKED: set(),
}


def _aware(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise StrategyRunError(f"{name} must be timezone-aware")
    return value


def _candidate_payload(candidate: StrategyCandidate) -> dict[str, Any]:
    return {
        "instrument": candidate.instrument.to_storage(),
        "snapshot": candidate.snapshot.to_storage(),
        "yes_depth": str(candidate.yes_depth),
        "no_depth": str(candidate.no_depth),
        "fee_per_share": str(candidate.fee_per_share),
        "slippage_per_share": str(candidate.slippage_per_share),
        "calibration_observations": [dict(item) for item in candidate.calibration_observations],
    }


def _restore_candidate(payload: Any) -> StrategyCandidate:
    if not isinstance(payload, Mapping) or set(payload) != {
        "instrument", "snapshot", "yes_depth", "no_depth", "fee_per_share",
        "slippage_per_share", "calibration_observations",
    }:
        raise StrategyRunError("stored strategy candidate schema is invalid")
    observations = payload["calibration_observations"]
    if not isinstance(observations, list) or any(not isinstance(item, Mapping) for item in observations):
        raise StrategyRunError("stored calibration observations are invalid")
    try:
        return StrategyCandidate(
            Instrument(**dict(payload["instrument"])),
            MarketSnapshot(**dict(payload["snapshot"])),
            Decimal(str(payload["yes_depth"])),
            Decimal(str(payload["no_depth"])),
            Decimal(str(payload["fee_per_share"])),
            Decimal(str(payload["slippage_per_share"])),
            tuple(dict(item) for item in observations),
        )
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        raise StrategyRunError("stored strategy candidate is malformed") from exc


@dataclass(frozen=True)
class StrategyRun:
    id: str
    account_scope_id: str
    account_epoch: int
    config_release_id: str
    observed_at: datetime
    phase: StrategyRunPhase = StrategyRunPhase.QUEUED
    candidates: tuple[StrategyCandidate, ...] = ()
    research_job_ids: tuple[str, ...] = ()
    reason: Optional[str] = None
    revision: int = 0
    updated_at: Optional[datetime] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise StrategyRunError("strategy run schema is unsupported")
        for value in (self.id, self.account_scope_id, self.config_release_id):
            if not isinstance(value, str) or not value.strip():
                raise StrategyRunError("strategy run identity is incomplete")
        if not self.account_scope_id.startswith("shadow-"):
            raise StrategyRunError("strategy run must use a shadow account scope")
        if type(self.account_epoch) is not int or self.account_epoch < 1:
            raise StrategyRunError("strategy run account epoch must be positive")
        if type(self.revision) is not int or self.revision < 0:
            raise StrategyRunError("strategy run revision must be non-negative")
        object.__setattr__(self, "phase", StrategyRunPhase(self.phase))
        object.__setattr__(self, "observed_at", _aware("observed_at", self.observed_at))
        updated_at = self.updated_at or self.observed_at
        object.__setattr__(self, "updated_at", _aware("updated_at", updated_at))
        candidates = tuple(self.candidates)
        jobs = tuple(str(item).strip() for item in self.research_job_ids)
        if any(not isinstance(item, StrategyCandidate) for item in candidates):
            raise StrategyRunError("strategy run candidates must be typed")
        if any(not item for item in jobs) or len(set(jobs)) != len(jobs):
            raise StrategyRunError("strategy run research job IDs must be unique")
        if self.phase is StrategyRunPhase.QUEUED and (candidates or jobs or self.reason):
            raise StrategyRunError("queued strategy run cannot contain phase results")
        if self.phase in {StrategyRunPhase.RESEARCH_PENDING, StrategyRunPhase.READY}:
            if not candidates or len(candidates) != len(jobs) or self.reason:
                raise StrategyRunError("research phase requires one job per captured candidate")
        if self.phase in _TERMINAL_PHASES and not str(self.reason or "").strip():
            raise StrategyRunError("terminal strategy run requires a reason")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "research_job_ids", jobs)

    def advance(
        self, phase: StrategyRunPhase, *, now: datetime,
        candidates: Optional[tuple[StrategyCandidate, ...]] = None,
        research_job_ids: Optional[tuple[str, ...]] = None,
        reason: Optional[str] = None,
    ) -> "StrategyRun":
        phase = StrategyRunPhase(phase)
        if phase not in _ALLOWED_TRANSITIONS[self.phase]:
            raise StrategyRunError(f"strategy run cannot transition from {self.phase.value} to {phase.value}")
        return replace(
            self, phase=phase,
            candidates=self.candidates if candidates is None else tuple(candidates),
            research_job_ids=(
                self.research_job_ids if research_job_ids is None
                else tuple(research_job_ids)
            ),
            reason=reason, revision=self.revision + 1, updated_at=_aware("updated_at", now),
        )

    def to_storage(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "account_scope_id": self.account_scope_id,
            "account_epoch": self.account_epoch,
            "config_release_id": self.config_release_id,
            "observed_at": self.observed_at.isoformat(),
            "phase": self.phase.value,
            "candidates": [_candidate_payload(item) for item in self.candidates],
            "research_job_ids": list(self.research_job_ids),
            "reason": self.reason,
            "revision": self.revision,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_storage(cls, payload: Any) -> "StrategyRun":
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version", "id", "account_scope_id", "account_epoch",
            "config_release_id", "observed_at", "phase", "candidates",
            "research_job_ids", "reason", "revision", "updated_at",
        }:
            raise StrategyRunError("stored strategy run schema is invalid")
        if not isinstance(payload["candidates"], list) or not isinstance(payload["research_job_ids"], list):
            raise StrategyRunError("stored strategy run collections are invalid")
        try:
            return cls(
                id=str(payload["id"]),
                account_scope_id=str(payload["account_scope_id"]),
                account_epoch=int(payload["account_epoch"]),
                config_release_id=str(payload["config_release_id"]),
                observed_at=datetime.fromisoformat(str(payload["observed_at"])),
                phase=StrategyRunPhase(str(payload["phase"])),
                candidates=tuple(_restore_candidate(item) for item in payload["candidates"]),
                research_job_ids=tuple(str(item) for item in payload["research_job_ids"]),
                reason=(str(payload["reason"]) if payload["reason"] is not None else None),
                revision=int(payload["revision"]),
                updated_at=datetime.fromisoformat(str(payload["updated_at"])),
                schema_version=int(payload["schema_version"]),
            )
        except (TypeError, ValueError) as exc:
            raise StrategyRunError("stored strategy run is malformed") from exc


class StrategyRunStore(Protocol):
    durable: bool

    def create(self, run: StrategyRun) -> StrategyRun: ...
    def get(self, run_id: str) -> Optional[StrategyRun]: ...
    def save(self, run: StrategyRun, *, expected_revision: int) -> StrategyRun: ...


def _encoded(run: StrategyRun) -> str:
    return json.dumps(
        run.to_storage(), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _identity(run: StrategyRun) -> tuple[object, ...]:
    return (
        run.id, run.account_scope_id, run.account_epoch,
        run.config_release_id, run.observed_at,
    )


class InMemoryStrategyRunStore:
    durable = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, str] = {}

    @tracer.start_as_current_span("twin.strategy.run_state.create")
    def create(self, run: StrategyRun) -> StrategyRun:
        if run.revision != 0 or run.phase is not StrategyRunPhase.QUEUED:
            raise StrategyRunError("new strategy run must be queued at revision zero")
        encoded = _encoded(run)
        with self._lock:
            existing = self._items.get(run.id)
            if existing is not None:
                restored = StrategyRun.from_storage(json.loads(existing))
                if _identity(restored) != _identity(run):
                    raise StrategyRunError("strategy run ID was reused with different work")
                return restored
            self._items[run.id] = encoded
        cycle_phase_transitions.add(1, {"from": "none", "to": run.phase.value})
        return run

    def get(self, run_id: str) -> Optional[StrategyRun]:
        with self._lock:
            encoded = self._items.get(str(run_id))
        return None if encoded is None else StrategyRun.from_storage(json.loads(encoded))

    @tracer.start_as_current_span("twin.strategy.run_state.save")
    def save(self, run: StrategyRun, *, expected_revision: int) -> StrategyRun:
        with self._lock:
            current = self.get(run.id)
            if current is None:
                raise StrategyRunError("strategy run was not found")
            _validate_update(current, run, expected_revision=expected_revision)
            self._items[run.id] = _encoded(run)
        cycle_phase_transitions.add(1, {"from": current.phase.value, "to": run.phase.value})
        return run


class DatastoreStrategyRunStore:
    durable = True

    def __init__(self, client: Any, *, namespace: str = "foresea-twin") -> None:
        self._client = client
        self._namespace = namespace

    def _key(self, run_id: str):
        return self._client.key(
            "TwinStrategyRun", sha256(run_id.encode("utf-8")).hexdigest(),
            namespace=self._namespace,
        )

    @staticmethod
    def _restore(entity: Any) -> StrategyRun:
        encoded = str(entity.get("payload_json") or "")
        if entity.get("fingerprint") != sha256(encoded.encode("utf-8")).hexdigest():
            raise StrategyRunError("stored strategy run failed integrity validation")
        run = StrategyRun.from_storage(json.loads(encoded))
        if entity.get("run_id") != run.id or int(entity.get("revision", -1)) != run.revision:
            raise StrategyRunError("stored strategy run identity is inconsistent")
        return run

    @staticmethod
    def _entity(key: Any, run: StrategyRun):
        from google.cloud import datastore

        encoded = _encoded(run)
        if len(encoded.encode("utf-8")) > 900_000:
            raise StrategyRunError("strategy run exceeds the durable size limit")
        entity = datastore.Entity(key=key, exclude_from_indexes=("payload_json",))
        entity.update({
            "run_id": run.id,
            "account_scope_id": run.account_scope_id,
            "phase": run.phase.value,
            "revision": run.revision,
            "updated_at": run.updated_at,
            "payload_json": encoded,
            "fingerprint": sha256(encoded.encode("utf-8")).hexdigest(),
        })
        return entity

    @tracer.start_as_current_span("twin.strategy.run_state.create")
    def create(self, run: StrategyRun) -> StrategyRun:
        if run.revision != 0 or run.phase is not StrategyRunPhase.QUEUED:
            raise StrategyRunError("new strategy run must be queued at revision zero")
        key = self._key(run.id)
        with self._client.transaction():
            entity = self._client.get(key)
            if entity is not None:
                existing = self._restore(entity)
                if _identity(existing) != _identity(run):
                    raise StrategyRunError("strategy run ID was reused with different work")
                return existing
            self._client.put(self._entity(key, run))
        cycle_phase_transitions.add(1, {"from": "none", "to": run.phase.value})
        return run

    def get(self, run_id: str) -> Optional[StrategyRun]:
        entity = self._client.get(self._key(str(run_id)))
        return None if entity is None else self._restore(entity)

    @tracer.start_as_current_span("twin.strategy.run_state.save")
    def save(self, run: StrategyRun, *, expected_revision: int) -> StrategyRun:
        key = self._key(run.id)
        with self._client.transaction():
            entity = self._client.get(key)
            if entity is None:
                raise StrategyRunError("strategy run was not found")
            current = self._restore(entity)
            _validate_update(current, run, expected_revision=expected_revision)
            self._client.put(self._entity(key, run))
        cycle_phase_transitions.add(1, {"from": current.phase.value, "to": run.phase.value})
        return run


def _validate_update(
    current: StrategyRun, updated: StrategyRun, *, expected_revision: int,
) -> None:
    if current.revision != expected_revision or updated.revision != expected_revision + 1:
        raise StrategyRunError("strategy run revision conflict")
    if _identity(current) != _identity(updated):
        raise StrategyRunError("strategy run immutable identity changed")
    if updated.phase not in _ALLOWED_TRANSITIONS[current.phase]:
        raise StrategyRunError("strategy run transition is not allowed")
