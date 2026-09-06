"""Atomic worst-case research-budget reservations before provider calls."""
from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable


class BudgetExceeded(RuntimeError):
    pass


class PriceUnavailable(BudgetExceeded):
    """A paid research request has no configured, auditable price."""


@dataclass(frozen=True)
class BudgetPolicy:
    usd_limit: Decimal
    token_limit: int
    request_limit: int


@dataclass(frozen=True)
class BudgetUsage:
    key: str
    reserved_usd: Decimal = Decimal("0")
    actual_usd: Decimal = Decimal("0")
    uncertain_usd: Decimal = Decimal("0")
    reserved_tokens: int = 0
    actual_tokens: int = 0
    requests: int = 0


@dataclass(frozen=True)
class BudgetReservation:
    id: str
    key: str
    estimated_usd: Decimal
    estimated_tokens: int
    state: str = "reserved"


@dataclass(frozen=True)
class ModelPrice:
    """USD price per one million tokens; zero-cost is explicit, never inferred."""

    input_per_million: Decimal | None
    output_per_million: Decimal | None


def estimate_request_cost(
    *, input_tokens: int, output_tokens: int, price: ModelPrice, require_usd_ceiling: bool
) -> Decimal:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token estimates must be non-negative")
    if price.input_per_million is None or price.output_per_million is None:
        if require_usd_ceiling:
            raise PriceUnavailable("paid research is blocked until model token prices are configured")
        return Decimal("0")
    return (
        Decimal(input_tokens) * price.input_per_million
        + Decimal(output_tokens) * price.output_per_million
    ) / Decimal("1000000")


def call_with_budget(
    budget: Any,
    reservation_id: str,
    *,
    key: str,
    estimated_usd: Decimal,
    estimated_tokens: int,
    policy: BudgetPolicy,
    operation: Callable[[], Any],
) -> Any:
    """Reserve before a provider/tool call and retain unknown charges safely."""
    budget.reserve(reservation_id, key=key, estimated_usd=estimated_usd, estimated_tokens=estimated_tokens, policy=policy)
    try:
        result = operation()
    except Exception:
        budget.reconcile(reservation_id, key=key, actual_usd=None, actual_tokens=None)
        raise
    usage = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(usage, dict):
        budget.reconcile(reservation_id, key=key, actual_usd=None, actual_tokens=None)
        return result
    actual_tokens = usage.get("total_tokens")
    actual_usd = usage.get("cost_usd")
    budget.reconcile(
        reservation_id,
        key=key,
        actual_usd=Decimal(str(actual_usd)) if actual_usd is not None else None,
        actual_tokens=int(actual_tokens) if isinstance(actual_tokens, int) else None,
    )
    return result


class InMemoryResearchBudget:
    """Thread-safe test implementation; live workers must use a durable adapter."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._usage: dict[str, BudgetUsage] = {}
        self._reservations: dict[str, BudgetReservation] = {}

    @staticmethod
    def key(strategy_id: str, account_scope_id: str, now: datetime) -> str:
        if now.tzinfo is None:
            raise ValueError("budget timestamps must be timezone-aware")
        return f"{strategy_id}:{account_scope_id}:{now.astimezone(timezone.utc).date().isoformat()}"

    def reserve(self, reservation_id: str, *, key: str, estimated_usd: Decimal, estimated_tokens: int, policy: BudgetPolicy) -> BudgetReservation:
        estimate = Decimal(str(estimated_usd))
        if estimate < 0 or estimated_tokens < 0:
            raise ValueError("budget estimates must be non-negative")
        with self._lock:
            existing = self._reservations.get(reservation_id)
            if existing is not None:
                return existing
            usage = self._usage.get(key, BudgetUsage(key))
            if usage.reserved_usd + usage.actual_usd + estimate > policy.usd_limit:
                raise BudgetExceeded("worst-case research USD budget is exhausted")
            if usage.reserved_tokens + usage.actual_tokens + estimated_tokens > policy.token_limit:
                raise BudgetExceeded("research token budget is exhausted")
            if usage.requests + 1 > policy.request_limit:
                raise BudgetExceeded("research request budget is exhausted")
            reservation = BudgetReservation(reservation_id, key, estimate, estimated_tokens)
            self._reservations[reservation_id] = reservation
            self._usage[key] = replace(usage, reserved_usd=usage.reserved_usd + estimate, reserved_tokens=usage.reserved_tokens + estimated_tokens, requests=usage.requests + 1)
            return reservation

    def reconcile(self, reservation_id: str, *, actual_usd: Decimal | None, actual_tokens: int | None) -> BudgetUsage:
        with self._lock:
            reservation = self._reservations[reservation_id]
            if reservation.state != "reserved":
                return self._usage[reservation.key]
            usage = self._usage[reservation.key]
            if actual_usd is None or actual_tokens is None:
                updated = replace(usage, reserved_usd=usage.reserved_usd - reservation.estimated_usd, reserved_tokens=usage.reserved_tokens - reservation.estimated_tokens, uncertain_usd=usage.uncertain_usd + reservation.estimated_usd)
                state = "uncertain"
            else:
                amount = Decimal(str(actual_usd))
                updated = replace(usage, reserved_usd=usage.reserved_usd - reservation.estimated_usd, reserved_tokens=usage.reserved_tokens - reservation.estimated_tokens, actual_usd=usage.actual_usd + amount, actual_tokens=usage.actual_tokens + actual_tokens)
                state = "reconciled"
            self._usage[reservation.key] = updated
            self._reservations[reservation_id] = replace(reservation, state=state)
            return updated

    def usage(self, key: str) -> BudgetUsage:
        with self._lock:
            return self._usage.get(key, BudgetUsage(key))


class DatastoreResearchBudget:
    """Durable daily budget aggregate and idempotent reservation adapter."""

    durable = True

    def __init__(self, client) -> None:
        self._client = client

    def _usage_key(self, key: str):
        return self._client.key("TwinResearchBudget", key)

    def _reservation_key(self, key: str, reservation_id: str):
        return self._client.key("TwinResearchBudget", key, "Reservation", reservation_id)

    @staticmethod
    def _usage(key: str, entity) -> BudgetUsage:
        return BudgetUsage(
            key=key,
            reserved_usd=Decimal(str(entity.get("reserved_usd", "0"))),
            actual_usd=Decimal(str(entity.get("actual_usd", "0"))),
            uncertain_usd=Decimal(str(entity.get("uncertain_usd", "0"))),
            reserved_tokens=int(entity.get("reserved_tokens", 0)),
            actual_tokens=int(entity.get("actual_tokens", 0)),
            requests=int(entity.get("requests", 0)),
        )

    @staticmethod
    def _write(entity, usage: BudgetUsage) -> None:
        entity.update({
            "reserved_usd": str(usage.reserved_usd), "actual_usd": str(usage.actual_usd),
            "uncertain_usd": str(usage.uncertain_usd), "reserved_tokens": usage.reserved_tokens,
            "actual_tokens": usage.actual_tokens, "requests": usage.requests,
        })

    def reserve(self, reservation_id: str, *, key: str, estimated_usd: Decimal, estimated_tokens: int, policy: BudgetPolicy) -> BudgetReservation:
        from google.cloud import datastore

        estimate = Decimal(str(estimated_usd))
        if estimate < 0 or estimated_tokens < 0:
            raise ValueError("budget estimates must be non-negative")
        usage_key, reservation_key = self._usage_key(key), self._reservation_key(key, reservation_id)
        with self._client.transaction():
            existing = self._client.get(reservation_key)
            if existing is not None:
                return BudgetReservation(reservation_id, key, Decimal(str(existing["estimated_usd"])), int(existing["estimated_tokens"]), str(existing["state"]))
            entity = self._client.get(usage_key) or datastore.Entity(key=usage_key)
            usage = self._usage(key, entity)
            if usage.reserved_usd + usage.actual_usd + usage.uncertain_usd + estimate > policy.usd_limit:
                raise BudgetExceeded("worst-case research USD budget is exhausted")
            if usage.reserved_tokens + usage.actual_tokens + estimated_tokens > policy.token_limit or usage.requests + 1 > policy.request_limit:
                raise BudgetExceeded("research token or request budget is exhausted")
            updated = replace(usage, reserved_usd=usage.reserved_usd + estimate, reserved_tokens=usage.reserved_tokens + estimated_tokens, requests=usage.requests + 1)
            self._write(entity, updated)
            reservation = datastore.Entity(key=reservation_key)
            reservation.update({"estimated_usd": str(estimate), "estimated_tokens": estimated_tokens, "state": "reserved"})
            self._client.put_multi([entity, reservation])
            return BudgetReservation(reservation_id, key, estimate, estimated_tokens)

    def reconcile(self, reservation_id: str, *, key: str, actual_usd: Decimal | None, actual_tokens: int | None) -> BudgetUsage:
        reservation_key, usage_key = self._reservation_key(key, reservation_id), self._usage_key(key)
        with self._client.transaction():
            reservation, entity = self._client.get(reservation_key), self._client.get(usage_key)
            if reservation is None or entity is None:
                raise KeyError("budget reservation was not found")
            usage = self._usage(key, entity)
            if reservation["state"] != "reserved":
                return usage
            estimate, tokens = Decimal(str(reservation["estimated_usd"])), int(reservation["estimated_tokens"])
            if actual_usd is None or actual_tokens is None:
                updated, state = replace(usage, reserved_usd=usage.reserved_usd - estimate, reserved_tokens=usage.reserved_tokens - tokens, uncertain_usd=usage.uncertain_usd + estimate), "uncertain"
            else:
                updated, state = replace(usage, reserved_usd=usage.reserved_usd - estimate, reserved_tokens=usage.reserved_tokens - tokens, actual_usd=usage.actual_usd + Decimal(str(actual_usd)), actual_tokens=usage.actual_tokens + actual_tokens), "reconciled"
            self._write(entity, updated)
            reservation["state"] = state
            self._client.put_multi([entity, reservation])
            return updated

    def usage(self, key: str) -> BudgetUsage:
        entity = self._client.get(self._usage_key(key))
        return self._usage(key, entity) if entity is not None else BudgetUsage(key)
