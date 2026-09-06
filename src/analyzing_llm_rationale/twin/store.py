"""Durable-state primitives for the autonomous trading twin.

The in-memory store is for unit tests and shadow development only.  Production
callers must provide a durable implementation; no venue or model callback is
accepted by transaction methods, so external work cannot run while the account
serialization lock is held.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping, Optional, Protocol

from .models import (
    AccountScope,
    CommandState,
    SchemaValidationError,
    TradeIntent,
    can_transition_command,
)


class TwinStoreError(RuntimeError):
    """The store cannot safely perform the requested state transition."""


class InsufficientReservationCapacity(TwinStoreError):
    """A reservation would exceed account cash or worst-case-loss limits."""


class ReservationState(str, Enum):
    RESERVED = "reserved"
    SUBMITTING = "submitting"
    SUBMISSION_UNKNOWN = "submission_unknown"
    RELEASED = "released"
    SETTLED = "settled"


@dataclass(frozen=True)
class AccountProjection:
    scope_id: str
    account_epoch: int
    venue_available_cash: Decimal
    loss_limit: Decimal
    revision: int = 0
    reserved_cash: Decimal = Decimal("0")
    reserved_max_loss: Decimal = Decimal("0")

    @property
    def available_cash_for_reservation(self) -> Decimal:
        """Venue availability already excludes venue-held funds; subtract local only once."""
        return self.venue_available_cash - self.reserved_cash

    @property
    def available_loss_for_reservation(self) -> Decimal:
        return self.loss_limit - self.reserved_max_loss


@dataclass(frozen=True)
class Reservation:
    id: str
    scope_id: str
    intent_id: str
    intent_hash: str
    cash: Decimal
    max_loss: Decimal
    account_revision: int
    state: ReservationState
    created_at: datetime
    reconciliation_ref: Optional[str] = None


@dataclass(frozen=True)
class CommandClaim:
    command_id: str
    worker_id: str
    fence: int
    lease_expires_at: datetime


@dataclass(frozen=True)
class ExecutionCommand:
    id: str
    scope_id: str
    intent_id: str
    intent_hash: str
    state: CommandState
    reservation_id: str
    client_order_id: str
    created_at: datetime
    claim: Optional[CommandClaim] = None


@dataclass(frozen=True)
class TwinEvent:
    id: str
    scope_id: str
    sequence: int
    event_type: str
    occurred_at: datetime
    observed_at: datetime
    payload_hash: str
    previous_projection_revision: int


@dataclass(frozen=True)
class OutboxMessage:
    id: str
    scope_id: str
    command_id: str
    payload_hash: str
    created_at: datetime
    delivered_at: Optional[datetime] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _decimal(name: str, value: Decimal) -> Decimal:
    value = Decimal(value)
    if not value.is_finite() or value < 0:
        raise SchemaValidationError(f"{name} must be a non-negative finite decimal")
    return value


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


class TwinStore(Protocol):
    """Account-serialized persistence API used by manual and twin execution."""

    durable: bool

    def register_account(self, scope: AccountScope, *, venue_available_cash: Decimal, loss_limit: Decimal) -> AccountProjection: ...

    def reserve_intent(self, intent: TradeIntent, *, cash: Decimal, max_loss: Decimal, now: Optional[datetime] = None) -> Reservation: ...

    def claim_command(self, command_id: str, *, worker_id: str, now: Optional[datetime] = None, lease_seconds: int = 30) -> Optional[CommandClaim]: ...


class InMemoryTwinStore:
    """Thread-safe deterministic store for tests; never valid for live execution."""

    durable = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._projections: dict[str, AccountProjection] = {}
        self._events: dict[str, dict[str, TwinEvent]] = {}
        self._reservations: dict[str, Reservation] = {}
        self._reservation_by_intent: dict[tuple[str, str], str] = {}
        self._commands: dict[str, ExecutionCommand] = {}
        self._outbox: dict[str, OutboxMessage] = {}
        self._inbox: set[tuple[str, str]] = set()

    def register_account(
        self, scope: AccountScope, *, venue_available_cash: Decimal, loss_limit: Decimal
    ) -> AccountProjection:
        cash = _decimal("venue_available_cash", venue_available_cash)
        loss = _decimal("loss_limit", loss_limit)
        with self._lock:
            existing = self._projections.get(scope.id)
            if existing is not None:
                if existing.account_epoch != scope.account_epoch:
                    raise TwinStoreError("account epoch changed; pause and reconcile before registering a new scope")
                return existing
            projection = AccountProjection(scope.id, scope.account_epoch, cash, loss)
            self._projections[scope.id] = projection
            self._events[scope.id] = {}
            return projection

    def projection(self, scope_id: str) -> AccountProjection:
        with self._lock:
            try:
                return self._projections[scope_id]
            except KeyError as exc:
                raise TwinStoreError("account scope is not registered") from exc

    def append_event(
        self,
        scope_id: str,
        *,
        event_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        occurred_at: Optional[datetime] = None,
        observed_at: Optional[datetime] = None,
        advance_projection: bool = True,
    ) -> TwinEvent:
        """Deduplicate one event and advance the account projection revision once."""
        occurred = occurred_at or _now()
        observed = observed_at or _now()
        if occurred.tzinfo is None or observed.tzinfo is None:
            raise TwinStoreError("event timestamps must be timezone-aware")
        with self._lock:
            projection = self.projection(scope_id)
            existing = self._events[scope_id].get(event_id)
            if existing is not None:
                return existing
            event = TwinEvent(
                id=event_id,
                scope_id=scope_id,
                sequence=len(self._events[scope_id]) + 1,
                event_type=str(event_type),
                occurred_at=occurred,
                observed_at=observed,
                payload_hash=_payload_hash(payload),
                previous_projection_revision=projection.revision,
            )
            self._events[scope_id][event_id] = event
            if advance_projection:
                self._projections[scope_id] = replace(projection, revision=projection.revision + 1)
            return event

    def reserve_intent(
        self,
        intent: TradeIntent,
        *,
        cash: Decimal,
        max_loss: Decimal,
        now: Optional[datetime] = None,
    ) -> Reservation:
        """Reserve cash and incremental max loss in one account-scoped transition."""
        reserved_cash = _decimal("cash", cash)
        reserved_loss = _decimal("max_loss", max_loss)
        now = now or _now()
        with self._lock:
            projection = self.projection(intent.account_scope_id)
            if intent.account_epoch != projection.account_epoch:
                raise TwinStoreError("intent account epoch is stale")
            key = (intent.account_scope_id, intent.intent_hash)
            existing_id = self._reservation_by_intent.get(key)
            if existing_id is not None:
                return self._reservations[existing_id]
            if reserved_cash > projection.available_cash_for_reservation:
                raise InsufficientReservationCapacity("insufficient account cash after existing local reservations")
            if reserved_loss > projection.available_loss_for_reservation:
                raise InsufficientReservationCapacity("insufficient worst-case-loss capacity after reservations")
            reservation = Reservation(
                id=f"reservation-{intent.intent_hash[:24]}",
                scope_id=intent.account_scope_id,
                intent_id=intent.id,
                intent_hash=intent.intent_hash,
                cash=reserved_cash,
                max_loss=reserved_loss,
                account_revision=projection.revision + 1,
                state=ReservationState.RESERVED,
                created_at=now,
            )
            self._reservations[reservation.id] = reservation
            self._reservation_by_intent[key] = reservation.id
            self._projections[intent.account_scope_id] = replace(
                projection,
                revision=projection.revision + 1,
                reserved_cash=projection.reserved_cash + reserved_cash,
                reserved_max_loss=projection.reserved_max_loss + reserved_loss,
            )
            self.append_event(
                intent.account_scope_id,
                event_id=f"reserve:{intent.intent_hash}",
                event_type="reservation_created",
                payload={"reservation_id": reservation.id, "intent_hash": intent.intent_hash},
                occurred_at=now,
                observed_at=now,
                advance_projection=False,
            )
            command = ExecutionCommand(
                id=f"command-{intent.intent_hash[:24]}",
                scope_id=intent.account_scope_id,
                intent_id=intent.id,
                intent_hash=intent.intent_hash,
                state=CommandState.RESERVED,
                reservation_id=reservation.id,
                client_order_id=f"foresea-{intent.intent_hash[:20]}",
                created_at=now,
            )
            self._commands[command.id] = command
            self._outbox[command.id] = OutboxMessage(
                id=f"outbox-{intent.intent_hash[:24]}",
                scope_id=intent.account_scope_id,
                command_id=command.id,
                payload_hash=_payload_hash({"command_id": command.id, "intent_hash": intent.intent_hash}),
                created_at=now,
            )
            return reservation

    def command_for_intent(self, intent: TradeIntent) -> ExecutionCommand:
        with self._lock:
            reservation_id = self._reservation_by_intent.get((intent.account_scope_id, intent.intent_hash))
            if reservation_id is None:
                raise TwinStoreError("intent has no reservation")
            command_id = f"command-{intent.intent_hash[:24]}"
            return self._commands[command_id]

    def claim_command(
        self, command_id: str, *, worker_id: str, now: Optional[datetime] = None, lease_seconds: int = 30
    ) -> Optional[CommandClaim]:
        now = now or _now()
        if lease_seconds <= 0:
            raise TwinStoreError("lease_seconds must be positive")
        with self._lock:
            command = self._commands[command_id]
            old_claim = command.claim
            if old_claim is not None and old_claim.lease_expires_at > now:
                return None
            fence = (old_claim.fence if old_claim is not None else 0) + 1
            claim = CommandClaim(command_id, worker_id, fence, now + timedelta(seconds=lease_seconds))
            state = CommandState.SUBMITTING if command.state is CommandState.RESERVED else command.state
            if state is not command.state and not can_transition_command(command.state, state):
                raise TwinStoreError("command cannot enter submitting state")
            self._commands[command_id] = replace(command, state=state, claim=claim)
            reservation = self._reservations[command.reservation_id]
            if reservation.state is ReservationState.RESERVED:
                self._reservations[reservation.id] = replace(reservation, state=ReservationState.SUBMITTING)
            return claim

    def transition_command(
        self, command_id: str, *, target: CommandState, fence: int, worker_id: str
    ) -> ExecutionCommand:
        """Apply a state only from the current fenced worker generation."""
        with self._lock:
            command = self._commands[command_id]
            claim = command.claim
            if claim is None or claim.worker_id != worker_id or claim.fence != fence:
                raise TwinStoreError("stale worker fence cannot progress command")
            if not can_transition_command(command.state, target):
                raise TwinStoreError(f"invalid command transition {command.state.value}->{target.value}")
            updated = replace(command, state=target)
            self._commands[command_id] = updated
            reservation = self._reservations[command.reservation_id]
            if target is CommandState.SUBMISSION_UNKNOWN:
                self._reservations[reservation.id] = replace(reservation, state=ReservationState.SUBMISSION_UNKNOWN)
            return updated

    def release_reservation(self, reservation_id: str, *, confirmed_no_order: bool) -> Reservation:
        """Release only after confirmed no-order/terminal reconciliation evidence."""
        with self._lock:
            reservation = self._reservations[reservation_id]
            if reservation.state in {ReservationState.SUBMITTING, ReservationState.SUBMISSION_UNKNOWN} and not confirmed_no_order:
                raise TwinStoreError("submitting or unknown reservation cannot be released by expiry cleanup")
            if reservation.state is ReservationState.RELEASED:
                return reservation
            projection = self.projection(reservation.scope_id)
            released = replace(reservation, state=ReservationState.RELEASED)
            self._reservations[reservation_id] = released
            self._projections[reservation.scope_id] = replace(
                projection,
                revision=projection.revision + 1,
                reserved_cash=projection.reserved_cash - reservation.cash,
                reserved_max_loss=projection.reserved_max_loss - reservation.max_loss,
            )
            return released

    def receive_inbox(self, scope_id: str, message_id: str) -> bool:
        """Return true once per delivery; repeated task/event IDs are ignored."""
        with self._lock:
            key = (scope_id, message_id)
            if key in self._inbox:
                return False
            self._inbox.add(key)
            return True

    def mark_outbox_delivered(self, command_id: str, *, delivered_at: Optional[datetime] = None) -> OutboxMessage:
        with self._lock:
            message = self._outbox[command_id]
            if message.delivered_at is not None:
                return message
            updated = replace(message, delivered_at=delivered_at or _now())
            self._outbox[command_id] = updated
            return updated

    def rebuild_projection(self, scope_id: str) -> AccountProjection:
        """Rebuild revision and active reservations solely from durable local events/state."""
        with self._lock:
            original = self.projection(scope_id)
            active = [
                item for item in self._reservations.values()
                if item.scope_id == scope_id and item.state not in {ReservationState.RELEASED, ReservationState.SETTLED}
            ]
            rebuilt = replace(
                original,
                revision=len(self._events[scope_id]),
                reserved_cash=sum((item.cash for item in active), Decimal("0")),
                reserved_max_loss=sum((item.max_loss for item in active), Decimal("0")),
            )
            self._projections[scope_id] = rebuilt
            return rebuilt


def require_durable_store(store: TwinStore, *, live: bool) -> TwinStore:
    if live and not store.durable:
        raise TwinStoreError("live twin execution requires a durable account store")
    return store


class DatastoreTwinStore:
    """Datastore-backed account transaction adapter for runtime use.

    Every mutation reads the account root by key in one transaction, then writes
    child entities below that root.  This avoids eventually-consistent global
    queries when approving cash/risk and makes the account the serialization
    point across manual and autonomous callers.
    """

    durable = True
    _KIND = "TwinAccount"

    def __init__(self, client: Any) -> None:
        self._client = client

    def _key(self, scope_id: str, *path: str) -> Any:
        return self._client.key(self._KIND, scope_id, *path)

    @staticmethod
    def _projection(entity: Any) -> AccountProjection:
        return AccountProjection(
            scope_id=str(entity["scope_id"]), account_epoch=int(entity["account_epoch"]),
            venue_available_cash=Decimal(str(entity["venue_available_cash"])), loss_limit=Decimal(str(entity["loss_limit"])),
            revision=int(entity.get("revision", 0)), reserved_cash=Decimal(str(entity.get("reserved_cash", "0"))),
            reserved_max_loss=Decimal(str(entity.get("reserved_max_loss", "0"))),
        )

    @staticmethod
    def _write_projection(entity: Any, projection: AccountProjection) -> None:
        entity.update({
            "scope_id": projection.scope_id, "account_epoch": projection.account_epoch,
            "venue_available_cash": str(projection.venue_available_cash), "loss_limit": str(projection.loss_limit),
            "revision": projection.revision, "reserved_cash": str(projection.reserved_cash),
            "reserved_max_loss": str(projection.reserved_max_loss),
        })

    def register_account(self, scope: AccountScope, *, venue_available_cash: Decimal, loss_limit: Decimal) -> AccountProjection:
        from google.cloud import datastore

        cash, loss = _decimal("venue_available_cash", venue_available_cash), _decimal("loss_limit", loss_limit)
        key = self._key(scope.id)
        with self._client.transaction():
            entity = self._client.get(key)
            if entity is not None:
                projection = self._projection(entity)
                if projection.account_epoch != scope.account_epoch:
                    raise TwinStoreError("account epoch changed; pause and reconcile before registering a new scope")
                return projection
            entity = datastore.Entity(key=key)
            projection = AccountProjection(scope.id, scope.account_epoch, cash, loss)
            self._write_projection(entity, projection)
            self._client.put(entity)
            return projection

    def projection(self, scope_id: str) -> AccountProjection:
        entity = self._client.get(self._key(scope_id))
        if entity is None:
            raise TwinStoreError("account scope is not registered")
        return self._projection(entity)

    def _reserve_intent_once(self, intent: TradeIntent, *, cash: Decimal, max_loss: Decimal, now: Optional[datetime] = None) -> Reservation:
        from google.cloud import datastore

        reserved_cash, reserved_loss, now = _decimal("cash", cash), _decimal("max_loss", max_loss), now or _now()
        root = self._key(intent.account_scope_id)
        reservation_id = f"reservation-{intent.intent_hash[:24]}"
        with self._client.transaction():
            account = self._client.get(root)
            if account is None:
                raise TwinStoreError("account scope is not registered")
            projection = self._projection(account)
            if intent.account_epoch != projection.account_epoch:
                raise TwinStoreError("intent account epoch is stale")
            reservation_key = self._key(intent.account_scope_id, "TwinReservation", reservation_id)
            existing = self._client.get(reservation_key)
            if existing is not None:
                return Reservation(
                    id=reservation_id, scope_id=intent.account_scope_id, intent_id=str(existing["intent_id"]),
                    intent_hash=str(existing["intent_hash"]), cash=Decimal(str(existing["cash"])),
                    max_loss=Decimal(str(existing["max_loss"])), account_revision=int(existing["account_revision"]),
                    state=ReservationState(str(existing["state"])), created_at=existing["created_at"],
                )
            if reserved_cash > projection.available_cash_for_reservation or reserved_loss > projection.available_loss_for_reservation:
                raise InsufficientReservationCapacity("insufficient account reservation capacity")
            updated = replace(
                projection, revision=projection.revision + 1, reserved_cash=projection.reserved_cash + reserved_cash,
                reserved_max_loss=projection.reserved_max_loss + reserved_loss,
            )
            self._write_projection(account, updated)
            reservation = Reservation(reservation_id, intent.account_scope_id, intent.id, intent.intent_hash, reserved_cash, reserved_loss, updated.revision, ReservationState.RESERVED, now)
            entity = datastore.Entity(key=reservation_key)
            entity.update({"intent_id": intent.id, "intent_hash": intent.intent_hash, "cash": str(reserved_cash), "max_loss": str(reserved_loss), "account_revision": updated.revision, "state": reservation.state.value, "created_at": now})
            command_id = f"{intent.account_scope_id}:command-{intent.intent_hash[:24]}"
            command = datastore.Entity(key=self._key(intent.account_scope_id, "TwinCommand", command_id))
            command.update({"intent_id": intent.id, "intent_hash": intent.intent_hash, "reservation_id": reservation_id, "state": CommandState.RESERVED.value, "client_order_id": f"foresea-{intent.intent_hash[:20]}", "created_at": now, "fence": 0})
            event = datastore.Entity(key=self._key(intent.account_scope_id, "TwinEvent", f"reserve:{intent.intent_hash}"))
            event.update({"event_type": "reservation_created", "sequence": updated.revision, "occurred_at": now, "observed_at": now, "payload_hash": _payload_hash({"reservation_id": reservation_id})})
            outbox = datastore.Entity(key=self._key(intent.account_scope_id, "TwinOutbox", command_id))
            outbox.update({"command_id": command_id, "payload_hash": _payload_hash({"command_id": command_id}), "created_at": now, "delivered_at": None})
            self._client.put_multi([account, entity, command, event, outbox])
            return reservation

    def reserve_intent(self, intent: TradeIntent, *, cash: Decimal, max_loss: Decimal, now: Optional[datetime] = None) -> Reservation:
        """Retry only Datastore transaction conflicts; never retry venue/model work."""
        from google.api_core.exceptions import Aborted

        last_error: Optional[Aborted] = None
        for _ in range(4):
            try:
                return self._reserve_intent_once(intent, cash=cash, max_loss=max_loss, now=now)
            except Aborted as exc:
                last_error = exc
        raise TwinStoreError("Datastore contention exceeded the bounded reservation retry budget") from last_error

    def command_for_intent(self, intent: TradeIntent) -> ExecutionCommand:
        command_id = f"{intent.account_scope_id}:command-{intent.intent_hash[:24]}"
        entity = self._client.get(self._key(intent.account_scope_id, "TwinCommand", command_id))
        if entity is None:
            raise TwinStoreError("intent has no reservation command")
        claim = None
        if entity.get("claim_worker_id"):
            claim = CommandClaim(
                command_id=entity.key.name, worker_id=str(entity["claim_worker_id"]), fence=int(entity["fence"]),
                lease_expires_at=entity["lease_expires_at"],
            )
        return ExecutionCommand(
            id=entity.key.name, scope_id=intent.account_scope_id, intent_id=str(entity["intent_id"]),
            intent_hash=str(entity["intent_hash"]), state=CommandState(str(entity["state"])),
            reservation_id=str(entity["reservation_id"]), client_order_id=str(entity["client_order_id"]),
            created_at=entity["created_at"], claim=claim,
        )

    def claim_command(self, command_id: str, *, worker_id: str, now: Optional[datetime] = None, lease_seconds: int = 30) -> Optional[CommandClaim]:
        if lease_seconds <= 0:
            raise TwinStoreError("lease_seconds must be positive")
        now = now or _now()
        scope_id = self._command_scope(command_id)
        key = self._key(scope_id, "TwinCommand", command_id)
        with self._client.transaction():
            entity = self._client.get(key)
            if entity is None:
                raise TwinStoreError("command was not found")
            old_expiry = entity.get("lease_expires_at")
            if old_expiry is not None and old_expiry > now:
                return None
            old_fence = int(entity.get("fence", 0))
            state = CommandState(str(entity["state"]))
            if state is CommandState.RESERVED:
                state = CommandState.SUBMITTING
            elif state not in {CommandState.SUBMITTING, CommandState.SUBMISSION_UNKNOWN}:
                return None
            claim = CommandClaim(command_id, worker_id, old_fence + 1, now + timedelta(seconds=lease_seconds))
            entity.update({"state": state.value, "claim_worker_id": worker_id, "fence": claim.fence, "lease_expires_at": claim.lease_expires_at})
            self._client.put(entity)
            return claim

    def transition_command(self, command_id: str, *, target: CommandState, fence: int, worker_id: str) -> ExecutionCommand:
        scope_id = self._command_scope(command_id)
        command_key = self._key(scope_id, "TwinCommand", command_id)
        with self._client.transaction():
            entity = self._client.get(command_key)
            if entity is None:
                raise TwinStoreError("command was not found")
            current = CommandState(str(entity["state"]))
            if str(entity.get("claim_worker_id") or "") != worker_id or int(entity.get("fence", 0)) != fence:
                raise TwinStoreError("stale worker fence cannot progress command")
            if not can_transition_command(current, target):
                raise TwinStoreError(f"invalid command transition {current.value}->{target.value}")
            entity["state"] = target.value
            reservation_key = self._key(scope_id, "TwinReservation", str(entity["reservation_id"]))
            reservation = self._client.get(reservation_key)
            if reservation is not None and target is CommandState.SUBMISSION_UNKNOWN:
                reservation["state"] = ReservationState.SUBMISSION_UNKNOWN.value
                self._client.put(reservation)
            self._client.put(entity)
        return self.command_for_intent_by_id(scope_id, command_id)

    def command_for_intent_by_id(self, scope_id: str, command_id: str) -> ExecutionCommand:
        entity = self._client.get(self._key(scope_id, "TwinCommand", command_id))
        if entity is None:
            raise TwinStoreError("command was not found")
        claim = CommandClaim(command_id, str(entity["claim_worker_id"]), int(entity["fence"]), entity["lease_expires_at"]) if entity.get("claim_worker_id") else None
        return ExecutionCommand(command_id, scope_id, str(entity["intent_id"]), str(entity["intent_hash"]), CommandState(str(entity["state"])), str(entity["reservation_id"]), str(entity["client_order_id"]), entity["created_at"], claim)

    def _command_scope(self, command_id: str) -> str:
        # Commands are intentionally addressed under their account root. Callers
        # retain scope context rather than using eventually consistent global lookup.
        parts = command_id.split(":", 1)
        if len(parts) == 2:
            return parts[0]
        raise TwinStoreError("Datastore command operations require an account-scoped command id")
