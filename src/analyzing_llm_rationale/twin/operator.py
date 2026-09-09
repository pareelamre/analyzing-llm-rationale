"""Owner-scoped operator projections and durable control requests."""
from __future__ import annotations

import base64
import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from opentelemetry import metrics, trace

from .account_store import AccountSnapshotRepository
from .mandates import Mandate, MandateStore
from .models import CommandState
from .store import AccountProjection, ExecutionCommand, TwinStore
from .strategy import StrategyCycle, StrategyStore
from .worker import WorkerJob, WorkerJobError, WorkerJobKind, WorkerJobs

tracer = trace.get_tracer(__name__)
operator_operations = metrics.get_meter(__name__).create_counter("twin.operator.operations", unit="1")
_REASON = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_TERMINAL_COMMANDS = frozenset({
    CommandState.FILLED, CommandState.CANCELLED, CommandState.REJECTED,
    CommandState.EXPIRED, CommandState.BLOCKED,
})
_CANCELLABLE_COMMANDS = frozenset({
    CommandState.ACKNOWLEDGED, CommandState.PARTIALLY_FILLED,
    CommandState.CANCEL_REQUESTED, CommandState.SUBMISSION_UNKNOWN,
})


class OperatorError(RuntimeError):
    pass


class OperatorConflict(OperatorError):
    pass


@dataclass(frozen=True)
class OwnerPause:
    owner_id: str
    paused: bool
    reason: str
    changed_at: datetime
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.owner_id.strip() or not _REASON.fullmatch(self.reason):
            raise OperatorError("operator pause identity or reason is invalid")
        if self.changed_at.tzinfo is None or self.changed_at.utcoffset() is None:
            raise OperatorError("operator pause time must be timezone-aware")
        if type(self.revision) is not int or self.revision < 0:
            raise OperatorError("operator pause revision is invalid")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "paused": self.paused, "reason": self.reason,
            "changed_at": self.changed_at.isoformat(), "revision": self.revision,
        }


class PauseStore(Protocol):
    durable: bool

    def get(self, owner_id: str) -> OwnerPause: ...
    def set(
        self, owner_id: str, *, paused: bool, reason: str,
        idempotency_key: str, now: datetime,
    ) -> OwnerPause: ...


class InMemoryPauseStore:
    durable = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: dict[str, OwnerPause] = {}
        self._requests: dict[tuple[str, str], tuple[str, OwnerPause]] = {}

    @staticmethod
    def _default(owner_id: str) -> OwnerPause:
        return OwnerPause(owner_id, False, "not_paused", datetime.fromtimestamp(0, tz=timezone.utc))

    def get(self, owner_id: str) -> OwnerPause:
        with self._lock:
            return self._states.get(owner_id, self._default(owner_id))

    def set(
        self, owner_id: str, *, paused: bool, reason: str,
        idempotency_key: str, now: datetime,
    ) -> OwnerPause:
        fingerprint = _control_fingerprint(paused, reason)
        with self._lock:
            replay = self._requests.get((owner_id, idempotency_key))
            if replay is not None:
                if replay[0] != fingerprint:
                    raise OperatorConflict("idempotency key was used for another pause transition")
                return replay[1]
            current = self.get(owner_id)
            result = OwnerPause(owner_id, paused, reason, now, current.revision + 1)
            self._states[owner_id] = result
            self._requests[(owner_id, idempotency_key)] = (fingerprint, result)
            return result


class DatastorePauseStore:
    """Owner-ancestor pause state with idempotent transitions."""

    durable = True

    def __init__(self, client: Any) -> None:
        self._client = client

    def _state_key(self, owner_id: str) -> Any:
        return self._client.key("User", owner_id, "TwinPauseState", "current")

    def _request_key(self, owner_id: str, idempotency_key: str) -> Any:
        return self._client.key(
            "User", owner_id, "TwinPauseRequest", sha256(idempotency_key.encode()).hexdigest(),
        )

    @staticmethod
    def _from_entity(owner_id: str, entity: Any) -> OwnerPause:
        try:
            changed_at = entity["changed_at"]
            if isinstance(changed_at, str):
                changed_at = datetime.fromisoformat(changed_at.replace("Z", "+00:00"))
            return OwnerPause(
                owner_id, bool(entity["paused"]), str(entity["reason"]),
                changed_at, int(entity["revision"]),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise OperatorError("stored operator pause is malformed") from exc

    def get(self, owner_id: str) -> OwnerPause:
        entity = self._client.get(self._state_key(owner_id))
        return self._from_entity(owner_id, entity) if entity is not None else OwnerPause(
            owner_id, False, "not_paused", datetime.fromtimestamp(0, tz=timezone.utc),
        )

    def set(
        self, owner_id: str, *, paused: bool, reason: str,
        idempotency_key: str, now: datetime,
    ) -> OwnerPause:
        from google.cloud import datastore

        fingerprint = _control_fingerprint(paused, reason)
        state_key, request_key = self._state_key(owner_id), self._request_key(owner_id, idempotency_key)
        with self._client.transaction():
            replay = self._client.get(request_key)
            if replay is not None:
                if replay.get("fingerprint") != fingerprint:
                    raise OperatorConflict("idempotency key was used for another pause transition")
                return self._from_entity(owner_id, replay)
            current = self._client.get(state_key)
            revision = int(current.get("revision", 0)) + 1 if current else 1
            result = OwnerPause(owner_id, paused, reason, now, revision)
            state, request = datastore.Entity(key=state_key), datastore.Entity(key=request_key)
            stored = {
                "paused": result.paused, "reason": result.reason,
                "changed_at": result.changed_at, "revision": result.revision,
            }
            state.update(stored)
            request.update({**stored, "fingerprint": fingerprint})
            self._client.put_multi([state, request])
            return result


@dataclass(frozen=True)
class OperatorPage:
    items: tuple[Mapping[str, Any], ...]
    next_cursor: Optional[str]

    def to_mapping(self) -> dict[str, Any]:
        return {"items": list(self.items), "next_cursor": self.next_cursor}


ReadinessResolver = Callable[[str, Any], Mapping[str, Any]]
Clock = Callable[[], datetime]


class OperatorService:
    """Build credential-free views and enqueue durable control work."""

    def __init__(
        self, twin_store: TwinStore, snapshots: AccountSnapshotRepository,
        strategies: StrategyStore, mandates: MandateStore, pauses: PauseStore,
        jobs: WorkerJobs, *, readiness: ReadinessResolver, clock: Clock,
        account_stale_seconds: int = 120,
    ) -> None:
        if account_stale_seconds <= 0:
            raise OperatorError("operator account freshness window must be positive")
        self.twin_store, self.snapshots = twin_store, snapshots
        self.strategies, self.mandates = strategies, mandates
        self.pauses, self.jobs = pauses, jobs
        self.readiness_resolver, self.clock = readiness, clock
        self.account_stale_seconds = account_stale_seconds

    def _scopes(self, owner_id: str) -> tuple[Any, ...]:
        return self.twin_store.account_scopes(owner_id, limit=25)

    def _commands(self, scopes: Sequence[Any]) -> tuple[ExecutionCommand, ...]:
        commands = []
        for scope in scopes:
            commands.extend(self.twin_store.commands(scope.id, limit=200))
        return tuple(sorted(commands, key=lambda item: (item.created_at, item.id), reverse=True))

    @tracer.start_as_current_span("twin.operator.status")
    def status(self, owner_id: str) -> Mapping[str, Any]:
        now, scopes = self.clock(), self._scopes(owner_id)
        commands, pause = self._commands(scopes), self.pauses.get(owner_id)
        mandates = tuple(self.mandates.latest_for_owner(owner_id, limit=100))
        active = tuple(mandate for mandate in mandates if mandate.active(now=now))
        snapshots = [self.snapshots.load(scope.id) for scope in scopes]
        stale_accounts = sum(
            snapshot is None or (now - snapshot.received_at).total_seconds() > self.account_stale_seconds
            for snapshot in snapshots
        )
        unknown = sum(command.state is CommandState.SUBMISSION_UNKNOWN for command in commands)
        readiness = [dict(self.readiness_resolver(owner_id, scope)) for scope in scopes]
        readiness_met = bool(readiness) and all(
            item.get("status") == "ready_for_shadow_trial" or item.get("live_eligible") is True
            for item in readiness
        )
        blockers = []
        if not active:
            blockers.append("no_active_mandate")
        if pause.paused:
            blockers.append("owner_paused")
        if unknown:
            blockers.append("submission_unknown")
        if stale_accounts:
            blockers.append("account_snapshot_stale")
        if not readiness_met:
            blockers.append("readiness_not_met")
        result = {
            "mode": "live" if any(mandate.live for mandate in active) else "shadow",
            "generated_at": now.isoformat(), "account_count": len(scopes),
            "active_mandate_count": len(active), "paused": pause.paused,
            "pause": pause.to_mapping(), "unknown_commands": unknown,
            "stale_accounts": stale_accounts,
            "readiness_met": readiness_met, "blockers": blockers,
            "new_exposure_allowed": not blockers,
            "mandates": [_mandate_view(mandate, now) for mandate in mandates],
        }
        operator_operations.add(1, {"operation": "status", "outcome": "complete"})
        return result

    @tracer.start_as_current_span("twin.operator.portfolio")
    def portfolio(self, owner_id: str) -> Mapping[str, Any]:
        now, accounts = self.clock(), []
        for scope in self._scopes(owner_id):
            accounts.append(_portfolio_view(
                scope, self.twin_store.projection(scope.id), self.snapshots.load(scope.id),
                now=now, stale_seconds=self.account_stale_seconds,
            ))
        operator_operations.add(1, {"operation": "portfolio", "outcome": "complete"})
        return {"generated_at": now.isoformat(), "accounts": accounts}

    def decisions(self, owner_id: str, *, limit: int, cursor: Optional[str]) -> OperatorPage:
        scopes = self._scopes(owner_id)
        cycles = self.strategies.cycles(
            frozenset(scope.id for scope in scopes), limit=min(200, limit + 100),
        )
        return _page(
            tuple(_decision_view(cycle) for cycle in cycles), limit=limit, cursor=cursor,
            time_field="created_at", id_field="key",
        )

    def commands(self, owner_id: str, *, limit: int, cursor: Optional[str]) -> OperatorPage:
        return _page(
            tuple(_command_view(command) for command in self._commands(self._scopes(owner_id))),
            limit=limit, cursor=cursor, time_field="created_at", id_field="id",
        )

    def readiness(self, owner_id: str) -> Mapping[str, Any]:
        now, items = self.clock(), []
        for scope in self._scopes(owner_id):
            items.append({
                "account_scope_id": scope.id, "venue": scope.venue,
                "environment": scope.environment,
                **dict(self.readiness_resolver(owner_id, scope)),
            })
        return {"generated_at": now.isoformat(), "accounts": items}

    def set_pause(
        self, owner_id: str, *, paused: bool, reason: str, idempotency_key: str,
    ) -> Mapping[str, Any]:
        now = self.clock()
        _validate_control(reason, idempotency_key, now)
        result = self.pauses.set(
            owner_id, paused=paused, reason=reason, idempotency_key=idempotency_key, now=now,
        )
        operator_operations.add(1, {"operation": "pause", "outcome": "paused" if paused else "resumed"})
        return result.to_mapping()

    def request_cancel(
        self, owner_id: str, command_id: str, *, idempotency_key: str,
    ) -> Mapping[str, Any]:
        now = self.clock()
        _validate_control("operator_cancel", idempotency_key, now)
        scope = None
        command = None
        for candidate_scope in self._scopes(owner_id):
            matches = [
                item for item in self.twin_store.commands(candidate_scope.id, limit=200)
                if item.id == command_id
            ]
            if matches:
                scope, command = candidate_scope, matches[0]
                break
        if scope is None or command is None:
            raise LookupError("command not found")
        if command.state in _TERMINAL_COMMANDS:
            raise OperatorConflict("terminal command cannot be cancelled")
        if command.state not in _CANCELLABLE_COMMANDS:
            raise OperatorConflict("command is not yet cancellable")
        request_id = "cancel-" + sha256(
            f"{owner_id}\x1f{command_id}\x1f{idempotency_key}".encode(),
        ).hexdigest()[:32]
        payload = {"command_id": command.id, "cancel_request_id": request_id}
        try:
            job = self.jobs.get(request_id)
            if job.account_scope_id != scope.id or job.kind is not WorkerJobKind.EXIT or job.payload != payload:
                raise OperatorConflict("cancel request identity conflicts with existing work")
        except WorkerJobError:
            job = self.jobs.add(WorkerJob(
                request_id, scope.id, WorkerJobKind.EXIT, payload, now + timedelta(days=1),
            ))
        operator_operations.add(1, {"operation": "cancel", "outcome": "queued"})
        return {
            "status": "queued", "cancel_request_id": request_id,
            "command_id": command.id, "command_state": command.state.value,
            "job_status": job.status.value, "requested_at": (job.created_at or now).isoformat(),
        }


def _validate_control(reason: str, idempotency_key: str, now: datetime) -> None:
    if not _REASON.fullmatch(str(reason)):
        raise OperatorError("operator reason must be a stable identifier")
    if not 8 <= len(str(idempotency_key)) <= 128:
        raise OperatorError("operator idempotency key length is invalid")
    if now.tzinfo is None or now.utcoffset() is None:
        raise OperatorError("operator control time must be timezone-aware")


def _control_fingerprint(paused: bool, reason: str) -> str:
    if not _REASON.fullmatch(str(reason)):
        raise OperatorError("operator reason must be a stable identifier")
    return sha256(json.dumps(
        {"paused": bool(paused), "reason": str(reason)}, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _number(value: Decimal | str) -> str:
    return format(Decimal(str(value)), "f")


def _portfolio_view(
    scope: Any, projection: AccountProjection, snapshot: Any, *,
    now: datetime, stale_seconds: int,
) -> Mapping[str, Any]:
    stale = snapshot is None or (now - snapshot.received_at).total_seconds() > stale_seconds
    return {
        "account_scope_id": scope.id, "venue": scope.venue,
        "environment": scope.environment, "collateral_asset": scope.collateral_asset,
        "account_epoch": scope.account_epoch, "projection_revision": projection.revision,
        "venue_available_cash": _number(projection.venue_available_cash),
        "available_cash_for_reservation": _number(projection.available_cash_for_reservation),
        "reserved_cash": _number(projection.reserved_cash),
        "reserved_max_loss": _number(projection.reserved_max_loss),
        "loss_limit": _number(projection.loss_limit),
        "snapshot_status": "missing" if snapshot is None else "stale" if stale else "fresh",
        "snapshot_received_at": snapshot.received_at.isoformat() if snapshot else None,
        "snapshot_generation": snapshot.generation if snapshot else None,
        "position_count": len(snapshot.holdings) if snapshot else None,
        "conservative_liquidation_value": _number(snapshot.conservative_liquidation_value) if snapshot else None,
        "divergence": snapshot.divergence if snapshot else None,
        "drift_reasons": list(snapshot.drift_reasons) if snapshot else [],
    }


def _command_view(command: ExecutionCommand) -> Mapping[str, Any]:
    reason = {
        CommandState.SUBMISSION_UNKNOWN: "venue_acknowledgement_unknown",
        CommandState.BLOCKED: "policy_or_authority_blocked",
        CommandState.EXPIRED: "intent_expired", CommandState.REJECTED: "venue_rejected",
    }.get(command.state)
    return {
        "id": command.id, "account_scope_id": command.scope_id,
        "intent_id": command.intent_id, "state": command.state.value,
        "created_at": command.created_at.isoformat(), "reason": reason,
        "uncertain": command.state is CommandState.SUBMISSION_UNKNOWN,
        "cancellable": command.state in _CANCELLABLE_COMMANDS,
        "claim_expires_at": command.claim.lease_expires_at.isoformat() if command.claim else None,
    }


def _decision_view(cycle: StrategyCycle) -> Mapping[str, Any]:
    intent = cycle.intent
    return {
        "key": cycle.key, "account_scope_id": cycle.account_scope_id,
        "decision": cycle.decision, "reason": cycle.reason,
        "strategy_version": cycle.strategy_version,
        "created_at": cycle.created_at.isoformat() if cycle.created_at else None,
        "intent": ({
            "id": intent.id, "instrument_id": intent.instrument_id,
            "action": intent.action.value, "quantity": _number(intent.quantity),
            "limit_price": _number(intent.limit_price),
        } if intent else None),
        "steps": [asdict(step) for step in cycle.steps],
    }


def _mandate_view(mandate: Mandate, now: datetime) -> Mapping[str, Any]:
    return {
        "id": mandate.id, "version": mandate.version,
        "account_scope_id": mandate.account_scope_id,
        "strategy_version": mandate.strategy_version, "venue": mandate.venue,
        "mode": "live" if mandate.live else "shadow", "live": mandate.live,
        "expires_at": mandate.expires_at.isoformat(),
        "active": mandate.active(now=now), "approved": mandate.approved_at is not None,
        "revoked": mandate.revoked, "authority_hash": mandate.digest(),
        "max_capital": _number(mandate.max_capital),
        "max_loss": _number(mandate.max_loss),
        "max_model_usd": _number(mandate.max_model_usd),
        "max_model_tokens": mandate.max_model_tokens,
        "max_model_requests": mandate.max_model_requests,
        "limits": {
            "max_capital": _number(mandate.max_capital), "max_loss": _number(mandate.max_loss),
            "max_model_usd": _number(mandate.max_model_usd),
            "max_model_tokens": mandate.max_model_tokens,
            "max_model_requests": mandate.max_model_requests,
        },
    }


def _page(
    items: tuple[Mapping[str, Any], ...], *, limit: int, cursor: Optional[str],
    time_field: str, id_field: str,
) -> OperatorPage:
    if not 1 <= limit <= 100:
        raise OperatorError("page limit must be within 1..100")
    after = _decode_cursor(cursor) if cursor else None
    selected = []
    for item in items:
        value = item.get(time_field)
        if not isinstance(value, str):
            continue
        key = (value, str(item[id_field]))
        if after is None or key < after:
            selected.append(item)
    page = tuple(selected[:limit])
    next_cursor = None
    if len(selected) > limit and page:
        last = page[-1]
        next_cursor = _encode_cursor(str(last[time_field]), str(last[id_field]))
    return OperatorPage(page, next_cursor)


def _encode_cursor(timestamp: str, identity: str) -> str:
    return base64.urlsafe_b64encode(
        json.dumps([timestamp, identity], separators=(",", ":")).encode(),
    ).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, str]:
    if len(cursor) > 512:
        raise OperatorError("page cursor is invalid")
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
        if not isinstance(payload, list) or len(payload) != 2 or not all(
            isinstance(item, str) and item for item in payload
        ):
            raise ValueError
        parsed = datetime.fromisoformat(payload[0].replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError
        return payload[0], payload[1]
    except Exception as exc:
        raise OperatorError("page cursor is invalid") from exc


def operator_status(
    *, mandate_active: bool, paused: bool, unknown_commands: int, shadow_only: bool = True,
) -> Mapping[str, object]:
    """Compatibility projection retained for existing callers and tests."""
    return {
        "mode": "shadow" if shadow_only else "live", "mandate_active": mandate_active,
        "paused": paused, "unknown_commands": unknown_commands,
        "new_exposure_allowed": mandate_active and not paused and unknown_commands == 0,
    }
