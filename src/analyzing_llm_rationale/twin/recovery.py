"""Conservative recovery of ambiguous command submissions.

Recovery owns no venue submission capability.  It can only reconcile the
prepared identity that was persisted before dispatch, and it keeps the cash
reservation when a venue cannot provide complete evidence of absence.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

from opentelemetry import metrics, trace

from .account import AccountSnapshot
from .models import AccountScope, CommandState, Completeness, TradeIntent
from .store import CommandClaim, ExecutionCommand, TwinStore, TwinStoreError

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
recovery_operations = metrics.get_meter(__name__).create_counter("twin.recovery.operations", unit="1")
ambiguous_submissions = metrics.get_meter(__name__).create_counter(
    "twin.submissions.ambiguous", unit="1"
)
_ORDER_STATUSES = frozenset({
    "acknowledged", "open", "partially_filled", "filled",
    "cancel_requested", "cancelled", "rejected",
})


class RecoveryBlocked(RuntimeError):
    """Recovery did not have enough evidence to change durable exposure state."""


class RecoveryAction(str, Enum):
    TERMINAL = "terminal"
    RECONCILED = "reconciled"
    CONFIRMED_ABSENT = "confirmed_absent"
    OPERATOR_ATTENTION = "operator_attention"


@dataclass(frozen=True)
class AbsencePolicy:
    required_observations: int
    minimum_interval_seconds: int

    def __post_init__(self) -> None:
        if self.required_observations < 2 or self.minimum_interval_seconds < 1:
            raise RecoveryBlocked("absence policy must require repeated separated observations")


def venue_absence_policy(venue: str) -> AbsencePolicy:
    """Return Foresea's conservative policy; this is not a venue guarantee."""
    normalized = str(venue).strip().lower()
    if normalized == "kalshi":
        return AbsencePolicy(2, 1)
    if normalized == "polymarket":
        return AbsencePolicy(3, 5)
    raise RecoveryBlocked("unknown venue cannot prove order absence")


@dataclass(frozen=True)
class VenueOrderLookup:
    """One complete, identity-bound order/fill query result."""

    account_scope_id: str
    instrument_id: str
    client_order_id: str
    request_fingerprint: str
    complete: bool
    order_found: Optional[bool]
    observed_at: datetime

    def __post_init__(self) -> None:
        if type(self.complete) is not bool or (
            self.order_found is not None and type(self.order_found) is not bool
        ):
            raise RecoveryBlocked("order lookup completeness fields must be boolean")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise RecoveryBlocked("recovery observation time must be timezone-aware")
        if not self.complete and self.order_found is not None:
            raise RecoveryBlocked("incomplete lookup cannot assert order presence or absence")
        if self.order_found is not None and not self.client_order_id:
            raise RecoveryBlocked("order lookup must retain the prepared client identity")


@dataclass(frozen=True)
class RecoveryResult:
    action: RecoveryAction
    command: ExecutionCommand
    reservation_released: bool = False


VenueLookup = Callable[[ExecutionCommand], VenueOrderLookup]


def _require_current_scope(
    store: TwinStore, command: ExecutionCommand, intent: TradeIntent,
) -> AccountScope:
    try:
        scope = store.account_scope(command.scope_id)
    except TwinStoreError as exc:
        raise RecoveryBlocked("recovery account scope is unavailable") from exc
    instrument_parts = intent.instrument_id.split(":", 2)
    if (
        command.scope_id != intent.account_scope_id
        or scope.account_epoch != intent.account_epoch
        or len(instrument_parts) != 3
        or instrument_parts[0] != scope.venue
        or instrument_parts[1] != scope.environment
    ):
        raise RecoveryBlocked("account reconnect or venue binding changed during recovery")
    return scope


def lookup_from_complete_account(
    snapshot: AccountSnapshot, *, command: ExecutionCommand,
    intent: TradeIntent, observed_at: datetime,
) -> VenueOrderLookup:
    """Bind a complete account generation to one prepared order identity."""
    if snapshot.scope_id != command.scope_id or command.intent_hash != intent.intent_hash:
        raise RecoveryBlocked("account generation does not match the recovery command")
    if snapshot.received_at != observed_at:
        raise RecoveryBlocked("account observation time does not match its durable generation")
    if snapshot.completeness is not Completeness.COMPLETE or snapshot.blocks_new_exposure:
        return VenueOrderLookup(
            command.scope_id, intent.instrument_id, command.client_order_id,
            command.request_fingerprint, False, None, snapshot.received_at,
        )
    candidates = [
        row for row in (*snapshot.orders, *snapshot.fills)
        if str(row.get("client_order_id") or "") == command.client_order_id
    ]
    for row in candidates:
        identity = str(
            row.get("instrument_id") or row.get("ticker") or row.get("token_id") or ""
        ).strip()
        if not identity:
            return VenueOrderLookup(
                command.scope_id, intent.instrument_id, command.client_order_id,
                command.request_fingerprint, False, None, observed_at,
            )
        if identity != intent.instrument_id and identity != intent.instrument_id.rsplit(":", 1)[-1]:
            raise RecoveryBlocked("venue evidence reused the client identity for another instrument")
    return VenueOrderLookup(
        command.scope_id, intent.instrument_id, command.client_order_id,
        command.request_fingerprint, True, bool(candidates), observed_at,
    )


def _recovery_result(
    action: RecoveryAction, command: ExecutionCommand, *, reservation_released: bool = False,
) -> RecoveryResult:
    recovery_operations.add(1, {"operation": "submission", "state": action.value})
    if action is RecoveryAction.OPERATOR_ATTENTION:
        ambiguous_submissions.add(1, {"venue": "unknown", "stage": "recovery"})
        logger.warning(
            "twin recovery requires operator attention command_ref=%s account_ref=%s",
            sha256(command.id.encode()).hexdigest()[:16],
            sha256(command.scope_id.encode()).hexdigest()[:16],
        )
    return RecoveryResult(action, command, reservation_released)


def _decimal(name: str, value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RecoveryBlocked(f"{name} must be a decimal") from exc
    if not result.is_finite() or result < 0:
        raise RecoveryBlocked(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class FillObservation:
    """One immutable or explicitly revised venue fill."""

    fill_id: str
    version: int
    order_id: str
    client_order_id: str
    instrument_id: str
    quantity: Decimal
    occurred_at: datetime
    observed_at: datetime

    def __post_init__(self) -> None:
        if not all(str(getattr(self, name)).strip() for name in (
            "fill_id", "order_id", "client_order_id", "instrument_id",
        )):
            raise RecoveryBlocked("fill identity is incomplete")
        if type(self.version) is not int or self.version < 1:
            raise RecoveryBlocked("fill version must be positive")
        object.__setattr__(self, "quantity", _decimal("fill quantity", self.quantity))
        if self.quantity == 0:
            raise RecoveryBlocked("fill quantity must be positive")
        if (
            self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None
            or self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None
        ):
            raise RecoveryBlocked("fill timestamps must be timezone-aware")
        if self.observed_at < self.occurred_at:
            raise RecoveryBlocked("fill cannot be observed before it occurs")

    def to_storage(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id, "version": self.version, "order_id": self.order_id,
            "client_order_id": self.client_order_id, "instrument_id": self.instrument_id,
            "quantity": str(self.quantity), "occurred_at": self.occurred_at.isoformat(),
            "observed_at": self.observed_at.isoformat(),
        }

    @classmethod
    def from_storage(cls, value: Mapping[str, Any]) -> "FillObservation":
        if type(value.get("version")) is not int:
            raise RecoveryBlocked("stored fill version must be an integer")
        return cls(
            str(value["fill_id"]), value["version"], str(value["order_id"]),
            str(value["client_order_id"]), str(value["instrument_id"]),
            Decimal(str(value["quantity"])),
            datetime.fromisoformat(str(value["occurred_at"]).replace("Z", "+00:00")),
            datetime.fromisoformat(str(value["observed_at"]).replace("Z", "+00:00")),
        )


@dataclass(frozen=True)
class SettlementObservation:
    settlement_id: str
    version: int
    order_id: str
    client_order_id: str
    instrument_id: str
    amount: Decimal
    final: bool
    occurred_at: datetime
    observed_at: datetime

    def __post_init__(self) -> None:
        if not all(str(getattr(self, name)).strip() for name in (
            "settlement_id", "order_id", "client_order_id", "instrument_id",
        )) or type(self.version) is not int or self.version < 1:
            raise RecoveryBlocked("settlement identity and positive version are required")
        if type(self.final) is not bool:
            raise RecoveryBlocked("settlement final flag must be boolean")
        try:
            amount = Decimal(str(self.amount))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise RecoveryBlocked("settlement amount must be a decimal") from exc
        if not amount.is_finite():
            raise RecoveryBlocked("settlement amount must be finite")
        object.__setattr__(self, "amount", amount)
        if (
            self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None
            or self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None
        ):
            raise RecoveryBlocked("settlement timestamps must be timezone-aware")
        if self.observed_at < self.occurred_at:
            raise RecoveryBlocked("settlement cannot be observed before it occurs")

    def to_storage(self) -> dict[str, Any]:
        return {
            "settlement_id": self.settlement_id, "version": self.version,
            "order_id": self.order_id, "client_order_id": self.client_order_id,
            "instrument_id": self.instrument_id,
            "amount": str(self.amount), "final": self.final,
            "occurred_at": self.occurred_at.isoformat(), "observed_at": self.observed_at.isoformat(),
        }

    @classmethod
    def from_storage(cls, value: Mapping[str, Any]) -> "SettlementObservation":
        if type(value.get("version")) is not int or type(value.get("final")) is not bool:
            raise RecoveryBlocked("stored settlement version or final flag is malformed")
        return cls(
            str(value["settlement_id"]), value["version"], str(value["order_id"]),
            str(value["client_order_id"]), str(value["instrument_id"]),
            Decimal(str(value["amount"])), value["final"],
            datetime.fromisoformat(str(value["occurred_at"]).replace("Z", "+00:00")),
            datetime.fromisoformat(str(value["observed_at"]).replace("Z", "+00:00")),
        )


@dataclass(frozen=True)
class LifecycleProjection:
    command_id: str
    scope_id: str
    venue_order_id: str
    client_order_id: str
    instrument_id: str
    order_quantity: Decimal
    revision: int = 0
    order_status: str = "acknowledged"
    fills: tuple[FillObservation, ...] = ()
    settlements: tuple[SettlementObservation, ...] = ()
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not all(str(getattr(self, name)).strip() for name in (
            "command_id", "scope_id", "venue_order_id", "client_order_id", "instrument_id",
        )):
            raise RecoveryBlocked("lifecycle identity is incomplete")
        object.__setattr__(self, "order_quantity", _decimal("order quantity", self.order_quantity))
        if self.order_quantity == 0 or type(self.revision) is not int or self.revision < 0:
            raise RecoveryBlocked("order quantity must be positive and revision non-negative")
        if self.order_status not in _ORDER_STATUSES:
            raise RecoveryBlocked("venue order status is unsupported")
        if self.updated_at is not None and (
            self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None
        ):
            raise RecoveryBlocked("lifecycle timestamp must be timezone-aware")
        if any(not isinstance(item, FillObservation) for item in self.fills):
            raise RecoveryBlocked("lifecycle fills are malformed")
        if any(not isinstance(item, SettlementObservation) for item in self.settlements):
            raise RecoveryBlocked("lifecycle settlements are malformed")
        fill_ids = [item.fill_id for item in self.fills]
        settlement_ids = [item.settlement_id for item in self.settlements]
        if len(fill_ids) != len(set(fill_ids)) or len(settlement_ids) != len(set(settlement_ids)):
            raise RecoveryBlocked("lifecycle contains duplicate evidence identities")
        for item in (*self.fills, *self.settlements):
            if (
                item.order_id != self.venue_order_id
                or item.client_order_id != self.client_order_id
                or item.instrument_id != self.instrument_id
            ):
                raise RecoveryBlocked("lifecycle evidence belongs to another order identity")
        if self.filled_quantity > self.order_quantity:
            raise RecoveryBlocked("fills exceed the original order quantity")

    @property
    def filled_quantity(self) -> Decimal:
        return sum((fill.quantity for fill in self.fills), Decimal("0"))

    @property
    def settlement_final(self) -> bool:
        return bool(self.settlements) and all(item.final for item in self.settlements)

    @property
    def settled_amount(self) -> Decimal | None:
        if not self.settlement_final:
            return None
        return sum((item.amount for item in self.settlements), Decimal("0"))

    def to_storage(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "command_id": self.command_id, "scope_id": self.scope_id,
            "venue_order_id": self.venue_order_id, "client_order_id": self.client_order_id,
            "instrument_id": self.instrument_id, "order_quantity": str(self.order_quantity),
            "revision": self.revision, "order_status": self.order_status,
            "fills": [item.to_storage() for item in self.fills],
            "settlements": [item.to_storage() for item in self.settlements],
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_storage(cls, value: Mapping[str, Any]) -> "LifecycleProjection":
        if value.get("schema_version") != 1 or type(value.get("revision")) is not int:
            raise RecoveryBlocked("stored lifecycle schema or revision is unsupported")
        return cls(
            str(value["command_id"]), str(value["scope_id"]), str(value["venue_order_id"]),
            str(value["client_order_id"]), str(value["instrument_id"]),
            Decimal(str(value["order_quantity"])), revision=value["revision"],
            order_status=str(value["order_status"]),
            fills=tuple(FillObservation.from_storage(item) for item in value["fills"]),
            settlements=tuple(SettlementObservation.from_storage(item) for item in value["settlements"]),
            updated_at=(datetime.fromisoformat(str(value["updated_at"]).replace("Z", "+00:00")) if value.get("updated_at") else None),
        )


def _merge_versioned(current: Sequence[Any], incoming: Sequence[Any], identity: str) -> tuple[Any, ...]:
    merged = {str(getattr(item, identity)): item for item in current}
    for item in incoming:
        key = str(getattr(item, identity))
        prior = merged.get(key)
        if prior is None or item.version > prior.version:
            merged[key] = item
        elif item.version == prior.version and item != prior:
            raise RecoveryBlocked(f"conflicting {identity} revision")
    return tuple(merged[key] for key in sorted(merged))


def apply_lifecycle_observations(
    current: LifecycleProjection, *, fills: Sequence[FillObservation] = (),
    settlements: Sequence[SettlementObservation] = (), order_status: str | None = None,
    observed_at: datetime,
) -> LifecycleProjection:
    """Idempotently reduce complete venue evidence into one order lifecycle."""
    if observed_at.tzinfo is None:
        raise RecoveryBlocked("lifecycle observation time must be timezone-aware")
    for fill in fills:
        if (
            fill.order_id != current.venue_order_id
            or fill.client_order_id != current.client_order_id
            or fill.instrument_id != current.instrument_id
            or fill.observed_at > observed_at
        ):
            raise RecoveryBlocked("fill does not match the recovered order identity")
    for settlement in settlements:
        if (
            settlement.order_id != current.venue_order_id
            or settlement.client_order_id != current.client_order_id
            or settlement.instrument_id != current.instrument_id
            or settlement.observed_at > observed_at
        ):
            raise RecoveryBlocked("settlement does not match the recovered order identity")
    merged_fills = _merge_versioned(current.fills, fills, "fill_id")
    merged_settlements = _merge_versioned(current.settlements, settlements, "settlement_id")
    if sum((item.quantity for item in merged_fills), Decimal("0")) > current.order_quantity:
        raise RecoveryBlocked("fills exceed the original order quantity")
    status = str(order_status or current.order_status).strip().lower()
    if status not in _ORDER_STATUSES:
        raise RecoveryBlocked("venue order status is unsupported")
    changed = (
        merged_fills != current.fills or merged_settlements != current.settlements
        or status != current.order_status
    )
    return LifecycleProjection(
        current.command_id, current.scope_id, current.venue_order_id,
        current.client_order_id, current.instrument_id, current.order_quantity,
        revision=current.revision + (1 if changed else 0), order_status=status,
        fills=merged_fills, settlements=merged_settlements,
        updated_at=observed_at if changed else current.updated_at,
    )


class LifecycleStore(Protocol):
    durable: bool
    def load(self, scope_id: str, command_id: str) -> LifecycleProjection | None: ...
    def save(self, projection: LifecycleProjection, *, expected_revision: int) -> LifecycleProjection: ...


class InMemoryLifecycleStore:
    durable = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[tuple[str, str], LifecycleProjection] = {}

    def load(self, scope_id: str, command_id: str) -> LifecycleProjection | None:
        with self._lock:
            return self._items.get((scope_id, command_id))

    def save(self, projection: LifecycleProjection, *, expected_revision: int) -> LifecycleProjection:
        with self._lock:
            key = (projection.scope_id, projection.command_id)
            current = self._items.get(key)
            actual = current.revision if current is not None else 0
            if actual != expected_revision:
                raise RecoveryBlocked("lifecycle changed during reconciliation")
            self._items[key] = projection
            return projection


class DatastoreLifecycleStore:
    durable = True

    def __init__(self, client: Any) -> None:
        self._client = client

    def _key(self, scope_id: str, command_id: str):
        return self._client.key("TwinAccount", scope_id, "TwinLifecycle", command_id)

    @staticmethod
    def _decode(entity: Any) -> LifecycleProjection:
        try:
            payload = json.loads(str(entity["payload_json"]))
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if entity.get("fingerprint") != sha256(encoded.encode()).hexdigest():
                raise RecoveryBlocked("stored lifecycle failed integrity validation")
            return LifecycleProjection.from_storage(payload)
        except RecoveryBlocked:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RecoveryBlocked("stored lifecycle is malformed") from exc

    def load(self, scope_id: str, command_id: str) -> LifecycleProjection | None:
        entity = self._client.get(self._key(scope_id, command_id))
        return self._decode(entity) if entity is not None else None

    def save(self, projection: LifecycleProjection, *, expected_revision: int) -> LifecycleProjection:
        from google.cloud import datastore

        key = self._key(projection.scope_id, projection.command_id)
        with self._client.transaction():
            current = self._client.get(key)
            actual = self._decode(current).revision if current is not None else 0
            if actual != expected_revision:
                raise RecoveryBlocked("lifecycle changed during reconciliation")
            payload = projection.to_storage()
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            entity = datastore.Entity(key=key, exclude_from_indexes=("payload_json",))
            entity.update({
                "revision": projection.revision,
                "fingerprint": sha256(encoded.encode()).hexdigest(),
                "payload_json": encoded,
                "updated_at": projection.updated_at or datetime.now(timezone.utc),
            })
            self._client.put(entity)
            return projection


def lifecycle_command_state(projection: LifecycleProjection) -> CommandState:
    """Derive command state from unique fills before mutable order status."""
    if projection.filled_quantity >= projection.order_quantity:
        return CommandState.FILLED
    if projection.filled_quantity > 0:
        return CommandState.PARTIALLY_FILLED
    return {
        "filled": CommandState.FILLED,
        "partially_filled": CommandState.PARTIALLY_FILLED,
        "cancel_requested": CommandState.CANCEL_REQUESTED,
        "cancelled": CommandState.CANCELLED,
        "rejected": CommandState.REJECTED,
        "acknowledged": CommandState.ACKNOWLEDGED,
        "open": CommandState.ACKNOWLEDGED,
    }[projection.order_status]


@tracer.start_as_current_span("twin.recovery.lifecycle")
def reconcile_lifecycle(
    command_store: TwinStore, lifecycle_store: LifecycleStore, *,
    command: ExecutionCommand, intent: TradeIntent, claim: CommandClaim,
    venue_order_id: str, fills: Sequence[FillObservation] = (),
    settlements: Sequence[SettlementObservation] = (), order_status: str | None = None,
    observed_at: datetime,
) -> LifecycleProjection:
    """Persist lifecycle evidence once and then advance the fenced command."""
    scope = _require_current_scope(command_store, command, intent)
    if scope.environment == "live" and not lifecycle_store.durable:
        raise RecoveryBlocked("live lifecycle reconciliation requires durable storage")
    current_command = command_store.command_for_intent(intent)
    if current_command.id != command.id or current_command.intent_hash != intent.intent_hash:
        raise RecoveryBlocked("lifecycle command is not bound to the immutable intent")
    if (
        current_command.claim is None
        or current_command.claim.worker_id != claim.worker_id
        or current_command.claim.fence != claim.fence
    ):
        raise RecoveryBlocked("lifecycle worker no longer owns the command fence")
    projection = lifecycle_store.load(command.scope_id, command.id)
    if projection is None:
        projection = LifecycleProjection(
            command.id, command.scope_id, venue_order_id, command.client_order_id,
            intent.instrument_id, intent.quantity,
        )
    elif (
        projection.venue_order_id != venue_order_id
        or projection.client_order_id != command.client_order_id
        or projection.instrument_id != intent.instrument_id
    ):
        raise RecoveryBlocked("lifecycle evidence belongs to another order identity")
    previous_revision = projection.revision
    updated = apply_lifecycle_observations(
        projection, fills=fills, settlements=settlements,
        order_status=order_status, observed_at=observed_at,
    )
    lifecycle_store.save(updated, expected_revision=previous_revision)
    target = lifecycle_command_state(updated)
    recovery_operations.add(1, {"operation": "lifecycle", "state": target.value})
    if target is not current_command.state and current_command.state is not CommandState.FILLED:
        try:
            command_store.transition_command(
                current_command.id, target=target, fence=claim.fence, worker_id=claim.worker_id,
            )
        except TwinStoreError as exc:
            raise RecoveryBlocked("lifecycle evidence was saved but command projection needs operator reconciliation") from exc
    if updated.settlement_final:
        settlement_ref = sha256(json.dumps(
            [item.to_storage() for item in updated.settlements],
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        try:
            command_store.settle_reservation(
                command.scope_id, command.reservation_id, settlement_ref=settlement_ref,
            )
        except TwinStoreError as exc:
            raise RecoveryBlocked("final settlement could not release the reservation") from exc
    return updated


def startup_recovery_action(command: ExecutionCommand, *, lease_expired: bool) -> RecoveryAction:
    """Classify restart work without treating a stale lease as proof of absence."""
    if command.state is CommandState.REJECTED:
        return RecoveryAction.TERMINAL
    if command.state in {CommandState.FILLED, CommandState.CANCELLED}:
        # These order states still need read-only late-fill and settlement
        # reconciliation until the reservation has reached a final state.
        return RecoveryAction.RECONCILED
    if command.state is CommandState.RESERVED:
        return RecoveryAction.RECONCILED
    if command.state in {
        CommandState.SUBMITTING, CommandState.SUBMISSION_UNKNOWN,
        CommandState.CANCEL_REQUESTED,
    }:
        return RecoveryAction.OPERATOR_ATTENTION
    if lease_expired:
        return RecoveryAction.OPERATOR_ATTENTION
    return RecoveryAction.RECONCILED


CancelSubmitter = Callable[[ExecutionCommand], Mapping[str, Any]]


@tracer.start_as_current_span("twin.recovery.cancel")
def cancel_after_reconciliation(
    store: TwinStore, *, command: ExecutionCommand, intent: TradeIntent,
    claim: CommandClaim, lookup: VenueOrderLookup, cancel: CancelSubmitter,
    now: datetime, max_lookup_age_seconds: int = 5,
) -> Mapping[str, Any]:
    """Reconcile the exact identity before each bounded cancellation attempt."""
    _require_current_scope(store, command, intent)
    current = store.command_for_intent(intent)
    if not _matching_lookup(current, intent, lookup) or lookup.order_found is not True:
        raise RecoveryBlocked("cancellation requires a complete matching live-order observation")
    if now.tzinfo is None or now.utcoffset() is None or max_lookup_age_seconds < 1:
        raise RecoveryBlocked("cancellation requires an aware time and positive freshness window")
    lookup_age = (now - lookup.observed_at).total_seconds()
    if lookup_age < 0 or lookup_age > max_lookup_age_seconds:
        raise RecoveryBlocked("cancellation lookup is stale or from the future")
    if current.claim is None or current.claim != claim:
        raise RecoveryBlocked("cancellation worker no longer owns the command fence")
    if current.state not in {
        CommandState.ACKNOWLEDGED, CommandState.PARTIALLY_FILLED,
        CommandState.CANCEL_REQUESTED,
    }:
        raise RecoveryBlocked(f"command cannot be cancelled from {current.state.value}")
    if current.state is not CommandState.CANCEL_REQUESTED:
        current = store.transition_command(
            current.id, target=CommandState.CANCEL_REQUESTED,
            fence=claim.fence, worker_id=claim.worker_id,
        )
    try:
        response = cancel(current)
    except Exception as exc:
        logger.warning(
            "twin cancellation outcome unknown command_ref=%s account_ref=%s",
            sha256(current.id.encode()).hexdigest()[:16],
            sha256(current.scope_id.encode()).hexdigest()[:16],
        )
        recovery_operations.add(1, {"operation": "cancel", "state": "unknown"})
        ambiguous_submissions.add(1, {"venue": "unknown", "stage": "cancel"})
        raise RecoveryBlocked("cancellation outcome is unknown; reconcile before another attempt") from exc
    if not isinstance(response, Mapping):
        raise RecoveryBlocked("cancellation response is malformed; reconcile before another attempt")
    envelope = response.get("venue_response", response)
    if not isinstance(envelope, Mapping):
        raise RecoveryBlocked("cancellation response is malformed; reconcile before another attempt")
    status = str(envelope.get("status") or envelope.get("venue_status") or "").strip().lower()
    if envelope.get("cancelled") is not True and envelope.get("canceled") is not True and status not in {
        "cancelled", "canceled",
    }:
        raise RecoveryBlocked("cancellation response is unconfirmed; reconcile before another attempt")
    recovery_operations.add(1, {"operation": "cancel", "state": "acknowledged"})
    return dict(response)


def _matching_lookup(command: ExecutionCommand, intent: TradeIntent, lookup: VenueOrderLookup) -> bool:
    return (
        lookup.complete
        and lookup.account_scope_id == command.scope_id
        and lookup.instrument_id == intent.instrument_id
        and lookup.client_order_id == command.client_order_id
        and lookup.request_fingerprint == command.request_fingerprint
    )


@tracer.start_as_current_span("twin.recovery.submission")
def recover_submission(
    store: TwinStore,
    *,
    command: ExecutionCommand,
    intent: TradeIntent,
    claim: CommandClaim,
    now: datetime,
    lookups: Iterable[VenueOrderLookup],
    required_absence_observations: int | None = None,
) -> RecoveryResult:
    """Recover a command without resubmitting it.

    A confirmed order becomes acknowledged.  Releasing the reservation needs
    independent complete absence observations; any incomplete or mismatched
    observation is deliberately a paused operator state.
    """
    policy = venue_absence_policy(intent.instrument_id.split(":", 1)[0])
    required = required_absence_observations or policy.required_observations
    if required < policy.required_observations:
        raise RecoveryBlocked("absence observations cannot weaken the venue policy")
    if now.tzinfo is None:
        raise RecoveryBlocked("recovery time must be timezone-aware")
    _require_current_scope(store, command, intent)
    current = store.command_for_intent(intent)
    if current.id != command.id or current.intent_hash != intent.intent_hash:
        raise RecoveryBlocked("recovery command is not bound to the immutable intent")
    if current.state in {CommandState.FILLED, CommandState.CANCELLED, CommandState.REJECTED}:
        return _recovery_result(RecoveryAction.TERMINAL, current)
    if current.state not in {CommandState.SUBMITTING, CommandState.SUBMISSION_UNKNOWN}:
        raise RecoveryBlocked(f"command cannot be recovered from {current.state.value}")
    if current.claim is None or current.claim.worker_id != claim.worker_id or current.claim.fence != claim.fence:
        raise RecoveryBlocked("recovery worker no longer owns the command fence")
    if current.claim.lease_expires_at <= now:
        raise RecoveryBlocked("recovery claim has expired")

    complete_absences = 0
    saw_incomplete = False
    absence_observation_times: set[datetime] = set()
    for lookup in lookups:
        identity_matches = (
            lookup.account_scope_id == current.scope_id
            and lookup.instrument_id == intent.instrument_id
            and lookup.client_order_id == current.client_order_id
            and lookup.request_fingerprint == current.request_fingerprint
        )
        if lookup.observed_at > now or not identity_matches:
            return _recovery_result(RecoveryAction.OPERATOR_ATTENTION, current)
        if not lookup.complete:
            saw_incomplete = True
            continue
        if lookup.order_found is True:
            updated = store.transition_command(
                current.id, target=CommandState.ACKNOWLEDGED, fence=claim.fence, worker_id=claim.worker_id
            )
            return _recovery_result(RecoveryAction.RECONCILED, updated)
        if lookup.order_found is False:
            if lookup.observed_at in absence_observation_times:
                return _recovery_result(RecoveryAction.OPERATOR_ATTENTION, current)
            absence_observation_times.add(lookup.observed_at)
            complete_absences += 1

    ordered_absences = sorted(absence_observation_times)
    separated = all(
        (later - earlier).total_seconds() >= policy.minimum_interval_seconds
        for earlier, later in zip(ordered_absences, ordered_absences[1:])
    )
    if saw_incomplete or complete_absences < required or not separated:
        return _recovery_result(RecoveryAction.OPERATOR_ATTENTION, current)
    updated = store.transition_command(
        current.id, target=CommandState.REJECTED, fence=claim.fence, worker_id=claim.worker_id
    )
    reservation_id = updated.reservation_id
    if store.durable:
        reservation_id = f"{updated.scope_id}:{reservation_id}"
    try:
        store.release_reservation(reservation_id, confirmed_no_order=True)
    except TwinStoreError as exc:
        raise RecoveryBlocked("confirmed absence could not release its reservation") from exc
    return _recovery_result(
        RecoveryAction.CONFIRMED_ABSENT, updated, reservation_released=True,
    )


def recovery_action(command_state: str, venue_order_found: bool | None) -> str:
    """Legacy simple classifier retained for callers without identity evidence."""
    if command_state in {"filled", "cancelled", "rejected"}:
        return "terminal"
    if command_state in {"submitting", "submission_unknown"}:
        if venue_order_found is True:
            return "reconcile"
        if venue_order_found is False:
            return "mark_no_order"
        return "pause_and_reconcile"
    return "resume"
