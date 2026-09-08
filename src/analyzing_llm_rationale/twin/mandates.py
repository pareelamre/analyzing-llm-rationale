"""Immutable, owner-scoped authority for autonomous execution."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, Mapping, Optional, Protocol, Sequence

from .evaluation import validate_readiness_artifact

_ACTIONS = frozenset({"BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"})


class MandateError(ValueError):
    pass


class MandateConflict(MandateError):
    pass


def _decimal(name: str, value: Any) -> str:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MandateError(f"{name} must be a decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise MandateError(f"{name} must be finite and non-negative")
    return str(parsed)


def _hash(name: str, value: Optional[str], *, required: bool = False) -> Optional[str]:
    if value is None and not required:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise MandateError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise MandateError(f"{name} must be a SHA-256 digest") from exc
    return value


@dataclass(frozen=True)
class Mandate:
    id: str
    owner_id: str
    account_scope_id: str
    strategy_version: str
    expires_at: datetime
    live: bool = False
    approved_hash: str | None = None
    revoked: bool = False
    account_epoch: int = 1
    venue: str = "shadow"
    allowed_actions: tuple[str, ...] = ("BUY_YES", "BUY_NO")
    max_capital: str = "0"
    max_loss: str = "0"
    model_hash: str | None = None
    config_hash: str | None = None
    readiness_hash: str | None = None
    release_hash: str | None = None
    max_model_usd: str = "0"
    max_model_tokens: int = 0
    max_model_requests: int = 0
    version: int = 1
    parent_hash: str | None = None
    identity_hash: str | None = None
    created_at: datetime | None = None
    approved_at: datetime | None = None
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("id", "owner_id", "account_scope_id", "strategy_version", "venue"):
            if not str(getattr(self, name)).strip():
                raise MandateError(f"{name} is required")
        if self.expires_at.tzinfo is None:
            raise MandateError("mandate expiry must be timezone-aware")
        if not isinstance(self.account_epoch, int) or self.account_epoch < 1:
            raise MandateError("account_epoch must be positive")
        if not isinstance(self.version, int) or self.version < 1:
            raise MandateError("mandate version must be positive")
        actions = tuple(sorted(set(self.allowed_actions)))
        if not actions or any(action not in _ACTIONS for action in actions):
            raise MandateError("mandate actions must be bounded prediction-market order actions")
        object.__setattr__(self, "allowed_actions", actions)
        for name in ("max_capital", "max_loss", "max_model_usd"):
            object.__setattr__(self, name, _decimal(name, getattr(self, name)))
        for name in ("max_model_tokens", "max_model_requests"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise MandateError(f"{name} must be a non-negative integer")
        for name in ("model_hash", "config_hash", "readiness_hash", "release_hash", "parent_hash", "identity_hash"):
            object.__setattr__(self, name, _hash(name, getattr(self, name)))
        for name in ("created_at", "approved_at", "revoked_at"):
            value = getattr(self, name)
            if value is not None and value.tzinfo is None:
                raise MandateError(f"{name} must be timezone-aware")
        if self.approved_hash is not None:
            _hash("approved_hash", self.approved_hash, required=True)
        if self.revoked != (self.revoked_at is not None):
            raise MandateError("revocation state and timestamp must agree")

    def authority_payload(self) -> dict[str, Any]:
        return {
            "id": self.id, "version": self.version, "parent_hash": self.parent_hash,
            "owner_id": self.owner_id, "identity_hash": self.identity_hash,
            "account_scope_id": self.account_scope_id, "account_epoch": self.account_epoch,
            "venue": self.venue, "strategy_version": self.strategy_version,
            "expires_at": self.expires_at.isoformat(), "live": self.live,
            "allowed_actions": list(self.allowed_actions), "max_capital": self.max_capital,
            "max_loss": self.max_loss, "max_model_usd": self.max_model_usd,
            "max_model_tokens": self.max_model_tokens, "max_model_requests": self.max_model_requests,
            "model_hash": self.model_hash, "config_hash": self.config_hash,
            "readiness_hash": self.readiness_hash, "release_hash": self.release_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def digest(self) -> str:
        encoded = json.dumps(self.authority_payload(), sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    def active(self, *, now: datetime) -> bool:
        return (
            now.tzinfo is not None and not self.revoked
            and self.approved_hash == self.digest() and now < self.expires_at
        )

    def to_storage(self) -> dict[str, Any]:
        return {
            **self.authority_payload(), "approved_hash": self.approved_hash,
            "revoked": self.revoked,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
        }

    @classmethod
    def from_storage(cls, value: Mapping[str, Any]) -> "Mandate":
        allowed = {
            "id", "version", "parent_hash", "owner_id", "identity_hash", "account_scope_id",
            "account_epoch", "venue", "strategy_version", "expires_at", "live", "allowed_actions",
            "max_capital", "max_loss", "max_model_usd", "max_model_tokens", "max_model_requests",
            "model_hash", "config_hash", "readiness_hash", "release_hash", "created_at",
            "approved_hash", "revoked", "approved_at", "revoked_at",
        }
        if set(value) != allowed:
            raise MandateError("stored mandate schema is invalid")
        data = dict(value)
        for name in ("expires_at", "created_at", "approved_at", "revoked_at"):
            if data.get(name) is not None:
                data[name] = datetime.fromisoformat(str(data[name]).replace("Z", "+00:00"))
        data["allowed_actions"] = tuple(data["allowed_actions"])
        return cls(**data)


def approve(
    draft: Mandate, *, owner_id: str, readiness_hash: str | None = None,
    expected_hash: str | None = None, now: datetime | None = None,
    identity_hash: str | None = None, release_hash: str | None = None,
    config_hash: str | None = None, model_hash: str | None = None,
    readiness_artifact: Mapping[str, Any] | None = None,
) -> Mandate:
    now = now or datetime.now(timezone.utc)
    if draft.owner_id != owner_id or draft.revoked:
        raise PermissionError("only the owner can approve an active mandate draft")
    if expected_hash is not None and expected_hash != draft.digest():
        raise MandateConflict("mandate changed after owner review")
    if draft.expires_at <= now:
        raise MandateError("expired mandates cannot be approved")
    for name, expected, current in (
        ("identity", draft.identity_hash, identity_hash), ("release", draft.release_hash, release_hash),
        ("config", draft.config_hash, config_hash), ("model", draft.model_hash, model_hash),
    ):
        if current is not None and expected != current:
            raise MandateConflict(f"current {name} does not match the reviewed mandate")
    if draft.live:
        required = (draft.identity_hash, draft.release_hash, draft.config_hash, draft.model_hash, draft.readiness_hash)
        if any(value is None for value in required):
            raise PermissionError("live activation requires identity, release, config, model, and readiness hashes")
        if readiness_hash != draft.readiness_hash or readiness_artifact is None:
            raise PermissionError("live activation requires the current verified readiness artifact")
        validate_readiness_artifact(readiness_artifact, now=now, max_age_seconds=7 * 24 * 60 * 60)
        if readiness_artifact.get("artifact_hash") != draft.readiness_hash:
            raise MandateConflict("readiness artifact does not match the reviewed mandate")
        # T12 deliberately cannot authorize live capital. T21/T22 later supply
        # a live-eligible artifact through a versioned validator extension.
        if readiness_artifact.get("live_eligible") is not True:
            raise PermissionError("current readiness artifact is not live eligible")
    approved = replace(draft, approved_hash=None, approved_at=now)
    return replace(approved, approved_hash=approved.digest())


def revise(mandate: Mandate, **changes: Any) -> Mandate:
    forbidden = {"id", "owner_id", "approved_hash", "approved_at", "revoked", "revoked_at", "version", "parent_hash"}
    if forbidden.intersection(changes):
        raise MandateError("immutable mandate identity or state cannot be edited")
    return replace(
        mandate, **changes, version=mandate.version + 1, parent_hash=mandate.digest(),
        approved_hash=None, approved_at=None, revoked=False, revoked_at=None,
    )


def revoke(mandate: Mandate, *, owner_id: str, now: datetime | None = None) -> Mandate:
    if mandate.owner_id != owner_id:
        raise PermissionError("only the owner can revoke a mandate")
    if mandate.revoked:
        return mandate
    return replace(mandate, revoked=True, revoked_at=now or datetime.now(timezone.utc))


class MandateStore(Protocol):
    durable: bool
    def create(self, mandate: Mandate) -> Mandate: ...
    def get(self, owner_id: str, mandate_id: str, version: int | None = None) -> Mandate | None: ...
    def save_transition(self, before: Mandate, after: Mandate, *, idempotency_key: str) -> Mandate: ...
    def versions(self, owner_id: str, mandate_id: str) -> Sequence[Mandate]: ...


class InMemoryMandateStore:
    durable = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[tuple[str, str, int], Mandate] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, Mandate]] = {}

    def create(self, mandate: Mandate) -> Mandate:
        with self._lock:
            key = (mandate.owner_id, mandate.id, mandate.version)
            existing = self._items.get(key)
            if existing is not None:
                if existing != mandate:
                    raise MandateConflict("mandate version already exists")
                return existing
            self._items[key] = mandate
            return mandate

    def get(self, owner_id: str, mandate_id: str, version: int | None = None) -> Mandate | None:
        with self._lock:
            matches = [item for (owner, mid, _), item in self._items.items() if owner == owner_id and mid == mandate_id]
            if version is not None:
                return self._items.get((owner_id, mandate_id, version))
            return max(matches, key=lambda item: item.version) if matches else None

    def save_transition(self, before: Mandate, after: Mandate, *, idempotency_key: str) -> Mandate:
        if not idempotency_key.strip():
            raise MandateError("idempotency key is required")
        with self._lock:
            replay = self._idempotency.get((before.owner_id, idempotency_key))
            if replay is not None:
                fingerprint, result = replay
                if fingerprint != _transition_fingerprint(before, after):
                    raise MandateConflict("idempotency key was used for another transition")
                return result
            current = self.get(before.owner_id, before.id, before.version)
            if current != before:
                raise MandateConflict("mandate changed before transition")
            self._items[(after.owner_id, after.id, after.version)] = after
            self._idempotency[(before.owner_id, idempotency_key)] = (_transition_fingerprint(before, after), after)
            return after

    def versions(self, owner_id: str, mandate_id: str) -> Sequence[Mandate]:
        with self._lock:
            return tuple(sorted(
                (item for (owner, mid, _), item in self._items.items() if owner == owner_id and mid == mandate_id),
                key=lambda item: item.version,
            ))


class DatastoreMandateStore:
    """Durable owner-ancestor mandate versions and idempotent transitions."""

    durable = True

    def __init__(self, client: Any) -> None:
        self._client = client

    def _version_key(self, owner_id: str, mandate_id: str, version: int):
        return self._client.key("User", owner_id, "TwinMandate", f"{mandate_id}:v{version}")

    def _pointer_key(self, owner_id: str, mandate_id: str):
        return self._client.key("User", owner_id, "TwinMandatePointer", mandate_id)

    def _transition_key(self, owner_id: str, idempotency_key: str):
        identity = sha256(idempotency_key.encode("utf-8")).hexdigest()
        return self._client.key("User", owner_id, "TwinMandateTransition", identity)

    @staticmethod
    def _entity_payload(entity: Any) -> Mandate:
        payload = json.loads(str(entity["payload_json"]))
        if entity.get("fingerprint") != sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest():
            raise MandateConflict("stored mandate failed integrity validation")
        return Mandate.from_storage(payload)

    @staticmethod
    def _entity(key: Any, mandate: Mandate):
        from google.cloud import datastore

        payload = mandate.to_storage()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 100_000:
            raise MandateError("mandate exceeds persistence boundary")
        entity = datastore.Entity(key=key, exclude_from_indexes=("payload_json",))
        entity.update({
            "mandate_id": mandate.id, "version": mandate.version,
            "fingerprint": sha256(encoded.encode()).hexdigest(), "payload_json": encoded,
        })
        return entity

    def create(self, mandate: Mandate) -> Mandate:
        from google.cloud import datastore

        key = self._version_key(mandate.owner_id, mandate.id, mandate.version)
        pointer_key = self._pointer_key(mandate.owner_id, mandate.id)
        with self._client.transaction():
            existing = self._client.get(key)
            if existing is not None:
                stored = self._entity_payload(existing)
                if stored != mandate:
                    raise MandateConflict("mandate version already exists")
                return stored
            if self._client.get(pointer_key) is not None:
                raise MandateConflict("mandate identity already exists")
            pointer = datastore.Entity(key=pointer_key)
            pointer.update({"latest_version": mandate.version})
            self._client.put_multi([self._entity(key, mandate), pointer])
            return mandate

    def get(self, owner_id: str, mandate_id: str, version: int | None = None) -> Mandate | None:
        if version is None:
            pointer = self._client.get(self._pointer_key(owner_id, mandate_id))
            if pointer is None:
                return None
            version = int(pointer["latest_version"])
        entity = self._client.get(self._version_key(owner_id, mandate_id, version))
        return self._entity_payload(entity) if entity is not None else None

    def save_transition(self, before: Mandate, after: Mandate, *, idempotency_key: str) -> Mandate:
        from google.cloud import datastore

        if not idempotency_key.strip():
            raise MandateError("idempotency key is required")
        transition_key = self._transition_key(before.owner_id, idempotency_key)
        before_key = self._version_key(before.owner_id, before.id, before.version)
        after_key = self._version_key(after.owner_id, after.id, after.version)
        pointer_key = self._pointer_key(before.owner_id, before.id)
        with self._client.transaction():
            replay = self._client.get(transition_key)
            if replay is not None:
                if replay.get("fingerprint") != _transition_fingerprint(before, after):
                    raise MandateConflict("idempotency key was used for another transition")
                return Mandate.from_storage(json.loads(str(replay["result_json"])))
            current_entity = self._client.get(before_key)
            if current_entity is None or self._entity_payload(current_entity) != before:
                raise MandateConflict("mandate changed before transition")
            if after.version != before.version and self._client.get(after_key) is not None:
                raise MandateConflict("mandate version already exists")
            transition = datastore.Entity(key=transition_key, exclude_from_indexes=("result_json",))
            transition.update({
                "mandate_id": before.id,
                "fingerprint": _transition_fingerprint(before, after),
                "result_json": json.dumps(after.to_storage(), sort_keys=True, separators=(",", ":")),
            })
            pointer = self._client.get(pointer_key)
            if pointer is None:
                raise MandateConflict("mandate pointer is missing")
            pointer["latest_version"] = max(int(pointer["latest_version"]), after.version)
            self._client.put_multi([self._entity(after_key, after), pointer, transition])
            return after

    def versions(self, owner_id: str, mandate_id: str) -> Sequence[Mandate]:
        pointer = self._client.get(self._pointer_key(owner_id, mandate_id))
        if pointer is None:
            return ()
        keys = [self._version_key(owner_id, mandate_id, version) for version in range(1, int(pointer["latest_version"]) + 1)]
        mandates = (self._entity_payload(entity) for entity in self._client.get_multi(keys) if entity is not None)
        return tuple(sorted(mandates, key=lambda mandate: mandate.version))


@dataclass(frozen=True)
class PauseState:
    global_pause: bool = False
    account_scopes: frozenset[str] = frozenset()
    strategies: frozenset[str] = frozenset()
    venues: frozenset[str] = frozenset()

    def blocks(self, mandate: Mandate) -> bool:
        return (
            self.global_pause or mandate.account_scope_id in self.account_scopes
            or mandate.strategy_version in self.strategies or mandate.venue in self.venues
        )


def authorize_mandate(
    mandate: Mandate, *, now: datetime, pause: PauseState,
    account_epoch: int, action: str,
) -> None:
    if not mandate.active(now=now):
        raise PermissionError("mandate is inactive")
    if pause.blocks(mandate):
        raise PermissionError("autonomous execution is paused")
    if account_epoch != mandate.account_epoch:
        raise MandateConflict("mandate account epoch is stale")
    if action not in mandate.allowed_actions:
        raise PermissionError("mandate does not allow this action")


def _transition_fingerprint(before: Mandate, after: Mandate) -> str:
    operation = "revision" if after.version > before.version else "revoke" if after.revoked else "approve"
    payload = {"mandate_id": before.id, "version": before.version, "authority_hash": before.digest(), "operation": operation}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
