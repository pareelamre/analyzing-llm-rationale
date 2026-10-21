"""Durable, generation-fenced persistence for complete account snapshots."""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, replace
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, Callable, Mapping, Optional, Protocol

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from .account import (
    AccountHolding,
    AccountSnapshot,
    _account_sum,
    _fees_paid,
    _position_holdings,
    _settlements_are_final,
)
from .models import Completeness, SchemaValidationError

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
snapshot_events = metrics.get_meter(__name__).create_counter("twin.account_snapshot.persisted", unit="1")


class AccountSnapshotStoreError(RuntimeError):
    """A complete account snapshot could not be durably retained."""


class AccountSnapshotRepository(Protocol):
    durable: bool

    def load(self, scope_id: str) -> Optional[AccountSnapshot]: ...

    def save(self, snapshot: AccountSnapshot, *, validate: Callable[[], None] | None = None) -> AccountSnapshot: ...


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported account snapshot value: {type(value).__name__}")


def _validate_snapshot(snapshot: AccountSnapshot) -> None:
    """Recheck persisted financial invariants before accepting account authority."""
    if (snapshot.journal_revision is not None or snapshot.journal_fingerprint is not None) and (
        type(snapshot.journal_revision) is not int or snapshot.journal_revision < 1
        or not isinstance(snapshot.journal_fingerprint, str)
        or re.fullmatch('[0-9a-f]{64}', snapshot.journal_fingerprint) is None
    ):
        raise AccountSnapshotStoreError('account snapshot journal provenance is invalid')
    if snapshot.account_epoch is not None and (type(snapshot.account_epoch) is not int or snapshot.account_epoch < 1):
        raise AccountSnapshotStoreError('account snapshot epoch is invalid')
    if snapshot.completeness is not Completeness.COMPLETE:
        raise AccountSnapshotStoreError("only complete account snapshots may be persisted")
    if (
        not isinstance(snapshot.scope_id, str) or not snapshot.scope_id.strip()
        or type(snapshot.generation) is not int or snapshot.generation < 1
        or snapshot.received_at.tzinfo is None
        or type(snapshot.divergence) is not bool
    ):
        raise AccountSnapshotStoreError("account snapshot identity is invalid")
    for name in ("available_cash", "total_cash", "reserved_cash", "settled_cash",
                 "position_basis", "fees_paid", "conservative_liquidation_value"):
        value = getattr(snapshot, name)
        if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
            raise AccountSnapshotStoreError("account snapshot monetary value is invalid")
    if (snapshot.available_cash + snapshot.reserved_cash > snapshot.total_cash
            or snapshot.settled_cash > snapshot.total_cash):
        raise AccountSnapshotStoreError("account snapshot cash is inconsistent")
    holdings, error = _position_holdings(snapshot.positions)
    fees, fee_error = _fees_paid(snapshot.fills, snapshot.settlements)
    if error or fee_error or _settlements_are_final(snapshot.settlements):
        raise AccountSnapshotStoreError("account snapshot source economics are invalid")
    if (holdings != snapshot.holdings or fees != snapshot.fees_paid
            or _account_sum(h.basis for h in holdings) != snapshot.position_basis
            or _account_sum((snapshot.settled_cash, _account_sum(h.liquidation_value for h in holdings)))
            != snapshot.conservative_liquidation_value
            or bool(snapshot.external_activity_ids or snapshot.drift_reasons) != snapshot.divergence):
        raise AccountSnapshotStoreError("account snapshot derived economics are inconsistent")


def _payload(snapshot: AccountSnapshot) -> dict[str, Any]:
    _validate_snapshot(snapshot)
    if snapshot.completeness is not Completeness.COMPLETE:
        raise AccountSnapshotStoreError("only complete account snapshots may be persisted")
    return {
        **({'account_epoch':snapshot.account_epoch} if snapshot.account_epoch is not None else {}),
        **({'journal_revision': snapshot.journal_revision, 'journal_fingerprint': snapshot.journal_fingerprint}
           if snapshot.journal_revision is not None else {}),
        "scope_id": snapshot.scope_id,
        "generation": snapshot.generation,
        "received_at": snapshot.received_at.isoformat(),
        "completeness": snapshot.completeness.value,
        "available_cash": str(snapshot.available_cash),
        "total_cash": str(snapshot.total_cash),
        "reserved_cash": str(snapshot.reserved_cash),
        "settled_cash": str(snapshot.settled_cash),
        "holdings": [asdict(holding) for holding in snapshot.holdings],
        "position_basis": str(snapshot.position_basis),
        "fees_paid": str(snapshot.fees_paid),
        "conservative_liquidation_value": str(snapshot.conservative_liquidation_value),
        "positions": list(snapshot.positions),
        "orders": list(snapshot.orders),
        "fills": list(snapshot.fills),
        "settlements": list(snapshot.settlements),
        "external_activity_ids": list(snapshot.external_activity_ids),
        "divergence": snapshot.divergence,
        "drift_reasons": list(snapshot.drift_reasons),
    }


def _canonical_json(snapshot: AccountSnapshot) -> str:
    try:
        return json.dumps(_payload(snapshot), sort_keys=True, separators=(",", ":"), default=_json_default)
    except (TypeError, ValueError) as exc:
        raise AccountSnapshotStoreError("account snapshot is not safely serializable") from exc


def _fingerprint(snapshot: AccountSnapshot) -> str:
    payload = _payload(snapshot)
    payload.pop("received_at")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return sha256(encoded).hexdigest()


def _decimal(name: str, value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as exc:
        raise AccountSnapshotStoreError(f"persisted {name} is invalid") from exc
    if not parsed.is_finite():
        raise AccountSnapshotStoreError(f"persisted {name} is invalid")
    return parsed


def _restore(payload: Mapping[str, Any]) -> AccountSnapshot:
    try:
        received_at = datetime.fromisoformat(str(payload["received_at"]))
        if received_at.tzinfo is None or payload["completeness"] != Completeness.COMPLETE.value:
            raise ValueError
        holdings = tuple(
            AccountHolding(
                instrument_id=str(row["instrument_id"]), quantity=_decimal("holding quantity", row["quantity"]),
                basis=_decimal("holding basis", row["basis"]),
                liquidation_value=_decimal("holding liquidation", row["liquidation_value"]),
            )
            for row in payload["holdings"]
        )
        collections = {name: tuple(payload[name]) for name in ("positions", "orders", "fills", "settlements")}
        if any(any(not isinstance(row, Mapping) for row in rows) for rows in collections.values()):
            raise ValueError
        if type(payload["generation"]) is not int or type(payload["divergence"]) is not bool:
            raise ValueError
        snapshot = AccountSnapshot(
            scope_id=str(payload["scope_id"]), generation=int(payload["generation"]), received_at=received_at,
            completeness=Completeness.COMPLETE, available_cash=_decimal("available_cash", payload["available_cash"]),
            total_cash=_decimal("total_cash", payload["total_cash"]), reserved_cash=_decimal("reserved_cash", payload["reserved_cash"]),
            settled_cash=_decimal("settled_cash", payload["settled_cash"]), holdings=holdings,
            position_basis=_decimal("position_basis", payload["position_basis"]),
            fees_paid=_decimal("fees_paid", payload["fees_paid"]),
            conservative_liquidation_value=_decimal("conservative_liquidation_value", payload["conservative_liquidation_value"]),
            positions=collections["positions"], orders=collections["orders"], fills=collections["fills"],
            settlements=collections["settlements"], external_activity_ids=tuple(str(item) for item in payload["external_activity_ids"]),
            divergence=bool(payload["divergence"]), drift_reasons=tuple(str(item) for item in payload["drift_reasons"]),
            account_epoch=payload.get('account_epoch'),
            journal_revision=payload.get('journal_revision'),
            journal_fingerprint=payload.get('journal_fingerprint'),
        )
        _validate_snapshot(snapshot)
        return snapshot
    except (KeyError, TypeError, ValueError, AccountSnapshotStoreError) as exc:
        raise AccountSnapshotStoreError("persisted account snapshot is malformed") from exc


class InMemoryAccountSnapshotStore:
    """Process-local implementation for deterministic unit tests only."""

    durable = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshots: dict[str, AccountSnapshot] = {}

    def load(self, scope_id: str) -> Optional[AccountSnapshot]:
        with self._lock:
            return self._snapshots.get(scope_id)

    def save(self, snapshot: AccountSnapshot, *, validate: Callable[[], None] | None = None) -> AccountSnapshot:
        _validate_snapshot(snapshot)
        with self._lock:
            if validate is not None:
                validate()
            prior = self._snapshots.get(snapshot.scope_id)
            if prior is not None:
                if prior.generation > snapshot.generation:
                    return prior
                if prior.generation == snapshot.generation:
                    if _fingerprint(prior) != _fingerprint(snapshot):
                        raise AccountSnapshotStoreError("same account generation has conflicting economics")
                    return prior
            if validate is not None:
                validate()
            self._snapshots[snapshot.scope_id] = snapshot
            return snapshot


class DatastoreAccountSnapshotStore:
    """Datastore-backed latest-complete snapshot store under the account root."""

    durable = True

    def __init__(self, client: Any, *, max_payload_bytes: int = 900_000) -> None:
        if max_payload_bytes < 1:
            raise SchemaValidationError("account snapshot payload limit must be positive")
        self._client = client
        self._max_payload_bytes = max_payload_bytes

    def _key(self, scope_id: str) -> Any:
        return self._client.key("TwinAccount", scope_id, "TwinAccountSnapshot", "latest")

    def load(self, scope_id: str) -> Optional[AccountSnapshot]:
        entity = self._client.get(self._key(scope_id))
        if entity is None:
            return None
        try:
            snapshot = _restore(json.loads(str(entity["payload_json"])))
            if snapshot.scope_id != scope_id or entity.get("fingerprint") != _fingerprint(snapshot):
                raise AccountSnapshotStoreError("stored snapshot identity or fingerprint mismatch")
            if entity.get("generation") != snapshot.generation:
                raise AccountSnapshotStoreError("stored snapshot generation mismatch")
            return self._current_journal_view(snapshot)
        except (KeyError, TypeError, ValueError, AccountSnapshotStoreError) as exc:
            raise AccountSnapshotStoreError("stored account snapshot cannot be read") from exc

    def _current_journal_view(self, snapshot: AccountSnapshot) -> AccountSnapshot:
        if snapshot.journal_revision is None:
            return snapshot  # Legacy remains readable, but blocks_new_exposure is true.
        try:
            from .account_journal import DatastoreAccountJournal, compare_journal_snapshot
            from .store import DatastoreTwinStore

            store = DatastoreTwinStore(self._client)
            scope = store.account_scope(snapshot.scope_id)
            if scope.environment == 'shadow' and scope.venue == 'shadow':
                return snapshot  # Simulator's own economic event digest, never native authority.
            journal = DatastoreAccountJournal(store).load(scope)
            if (journal is not None and snapshot.account_epoch == scope.account_epoch
                    and journal.revision == snapshot.journal_revision
                    and journal.fingerprint == snapshot.journal_fingerprint):
                return compare_journal_snapshot(snapshot, journal)
        # aqg: top-level boundary -- private journal errors only produce a fixed blocking view
        except Exception:
            pass
        return replace(snapshot, divergence=True,
            drift_reasons=tuple(sorted(set((*snapshot.drift_reasons, 'journal_authority_changed')))))

    @tracer.start_as_current_span("twin.account_snapshot.persist", record_exception=False, set_status_on_exception=False)
    def save(self, snapshot: AccountSnapshot, *, validate: Callable[[], None] | None = None) -> AccountSnapshot:
        span = trace.get_current_span()
        span.set_attributes({"account.scope_id": snapshot.scope_id, "account.generation": snapshot.generation})
        try:
            encoded = _canonical_json(snapshot)
            if len(encoded.encode("utf-8")) > self._max_payload_bytes:
                raise AccountSnapshotStoreError("account snapshot exceeds durable payload limit")
            fingerprint = _fingerprint(snapshot)
            key = self._key(snapshot.scope_id)
            with self._client.transaction():
                if validate is not None:
                    validate()
                existing = self._client.get(key)
                if existing is not None:
                    existing_snapshot = _restore(json.loads(str(existing["payload_json"])))
                    if (existing_snapshot.scope_id != snapshot.scope_id
                            or existing.get("fingerprint") != _fingerprint(existing_snapshot)
                            or existing.get("generation") != existing_snapshot.generation):
                        raise AccountSnapshotStoreError("stored snapshot integrity mismatch")
                    if existing_snapshot.generation > snapshot.generation:
                        snapshot_events.add(1, {"outcome": "stale"})
                        span.set_attribute("outcome", "stale")
                        return existing_snapshot
                    if existing_snapshot.generation == snapshot.generation:
                        if str(existing.get("fingerprint")) != fingerprint:
                            raise AccountSnapshotStoreError("same account generation has conflicting economics")
                        snapshot_events.add(1, {"outcome": "duplicate"})
                        span.set_attribute("outcome", "duplicate")
                        return existing_snapshot
                from analyzing_llm_rationale import datastore_backend as datastore

                entity = datastore.Entity(key=key, exclude_from_indexes=("payload_json",))
                entity.update({"generation": snapshot.generation, "fingerprint": fingerprint, "payload_json": encoded})
                self._client.put(entity)
                if validate is not None:
                    validate()
            snapshot_events.add(1, {"outcome": "stored"})
            span.set_attribute("outcome", "stored")
            logger.info("Stored complete account snapshot generation %s", snapshot.generation)
            return snapshot
        except Exception:  # aqg: top-level boundary -- retain durable state and emit only fixed error text
            snapshot_events.add(1, {"outcome": "error"})
            span.record_exception(ValueError('account_snapshot_persist_failed'))
            span.set_status(Status(StatusCode.ERROR))
            logger.warning("Could not persist complete account snapshot")
            raise
