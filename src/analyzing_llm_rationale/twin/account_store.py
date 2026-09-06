"""Durable, generation-fenced persistence for complete account snapshots."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, Mapping, Optional, Protocol

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from .account import AccountHolding, AccountSnapshot
from .models import Completeness, SchemaValidationError

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
snapshot_events = metrics.get_meter(__name__).create_counter("twin.account_snapshot.persisted", unit="1")


class AccountSnapshotStoreError(RuntimeError):
    """A complete account snapshot could not be durably retained."""


class AccountSnapshotRepository(Protocol):
    durable: bool

    def load(self, scope_id: str) -> Optional[AccountSnapshot]: ...

    def save(self, snapshot: AccountSnapshot) -> AccountSnapshot: ...


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported account snapshot value: {type(value).__name__}")


def _payload(snapshot: AccountSnapshot) -> dict[str, Any]:
    if snapshot.completeness is not Completeness.COMPLETE:
        raise AccountSnapshotStoreError("only complete account snapshots may be persisted")
    return {
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
        return AccountSnapshot(
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
        )
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

    def save(self, snapshot: AccountSnapshot) -> AccountSnapshot:
        with self._lock:
            prior = self._snapshots.get(snapshot.scope_id)
            if prior is not None:
                if prior.generation > snapshot.generation:
                    return prior
                if prior.generation == snapshot.generation:
                    if _fingerprint(prior) != _fingerprint(snapshot):
                        raise AccountSnapshotStoreError("same account generation has conflicting economics")
                    return prior
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
            return _restore(json.loads(str(entity["payload_json"])))
        except (KeyError, TypeError, ValueError, AccountSnapshotStoreError) as exc:
            raise AccountSnapshotStoreError("stored account snapshot cannot be read") from exc

    @tracer.start_as_current_span("twin.account_snapshot.persist")
    def save(self, snapshot: AccountSnapshot) -> AccountSnapshot:
        span = trace.get_current_span()
        span.set_attributes({"account.scope_id": snapshot.scope_id, "account.generation": snapshot.generation})
        try:
            encoded = _canonical_json(snapshot)
            if len(encoded.encode("utf-8")) > self._max_payload_bytes:
                raise AccountSnapshotStoreError("account snapshot exceeds durable payload limit")
            fingerprint = _fingerprint(snapshot)
            key = self._key(snapshot.scope_id)
            with self._client.transaction():
                existing = self._client.get(key)
                if existing is not None:
                    existing_snapshot = _restore(json.loads(str(existing["payload_json"])))
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
                from google.cloud import datastore

                entity = datastore.Entity(key=key, exclude_from_indexes=("payload_json",))
                entity.update({"generation": snapshot.generation, "fingerprint": fingerprint, "payload_json": encoded})
                self._client.put(entity)
            snapshot_events.add(1, {"outcome": "stored"})
            span.set_attribute("outcome", "stored")
            logger.info("Stored complete account snapshot generation %s", snapshot.generation)
            return snapshot
        except Exception as exc:
            snapshot_events.add(1, {"outcome": "error"})
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.warning("Could not persist complete account snapshot")
            raise
