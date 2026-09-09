"""Atomic worst-case research-budget reservations before provider calls."""
from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

from opentelemetry import metrics, trace

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
research_budget_operations = meter.create_counter("twin.research.budget.operations", unit="1")
research_budget_usd = meter.create_histogram("twin.research.budget.usd", unit="USD")
research_budget_tokens = meter.create_histogram("twin.research.budget.tokens", unit="{token}")

class BudgetExceeded(RuntimeError):
    pass


class PriceUnavailable(BudgetExceeded):
    """A paid research request has no configured, auditable price."""


class BudgetAlreadyClaimed(BudgetExceeded):
    """Duplicate delivery cannot dispatch a possibly billed request again."""


def _amount(value: Any) -> Decimal:
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("budget amounts must be finite and non-negative")
    return amount


def _tokens(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("budget counts must be non-negative integers")
    return value


@dataclass(frozen=True)
class BudgetPolicy:
    usd_limit: Decimal
    token_limit: int
    request_limit: int

    def __post_init__(self) -> None:
        _amount(self.usd_limit)
        _tokens(self.token_limit)
        _tokens(self.request_limit)


@dataclass(frozen=True)
class BudgetUsage:
    key: str
    reserved_usd: Decimal = Decimal("0")
    actual_usd: Decimal = Decimal("0")
    uncertain_usd: Decimal = Decimal("0")
    reserved_tokens: int = 0
    actual_tokens: int = 0
    requests: int = 0
    uncertain_tokens: int = 0


@dataclass(frozen=True)
class BudgetReservation:
    id: str
    key: str
    estimated_usd: Decimal
    estimated_tokens: int
    state: str = "reserved"
    charged_usd: Decimal = Decimal("0")
    charged_tokens: int = 0


def _reconcile_usage(usage: BudgetUsage, reservation: BudgetReservation, actual_usd: Decimal | None, actual_tokens: int | None) -> tuple[BudgetUsage, BudgetReservation]:
    amount = _amount(actual_usd) if actual_usd is not None else None
    tokens = _tokens(actual_tokens) if actual_tokens is not None else None
    if reservation.state == "reconciled":
        if amount is not None and amount != reservation.charged_usd or tokens is not None and tokens != reservation.charged_tokens:
            raise ValueError("reconciled usage changed; an audited correction is required")
        return usage, reservation
    if reservation.state not in {"reserved", "claimed", "uncertain"}:
        raise ValueError("invalid budget reservation state")
    if reservation.state == "uncertain":
        remaining = replace(usage, uncertain_usd=usage.uncertain_usd - reservation.charged_usd, uncertain_tokens=usage.uncertain_tokens - reservation.charged_tokens)
    else:
        remaining = replace(usage, reserved_usd=usage.reserved_usd - reservation.estimated_usd, reserved_tokens=usage.reserved_tokens - reservation.estimated_tokens)
    _amount(remaining.reserved_usd)
    _amount(remaining.uncertain_usd)
    _tokens(remaining.reserved_tokens)
    _tokens(remaining.uncertain_tokens)
    if amount is None or tokens is None:
        held_usd = max(reservation.estimated_usd, reservation.charged_usd, amount or Decimal("0"))
        held_tokens = max(reservation.estimated_tokens, reservation.charged_tokens, tokens or 0)
        return replace(remaining, uncertain_usd=remaining.uncertain_usd + held_usd, uncertain_tokens=remaining.uncertain_tokens + held_tokens), replace(reservation, state="uncertain", charged_usd=held_usd, charged_tokens=held_tokens)
    return replace(remaining, actual_usd=remaining.actual_usd + amount, actual_tokens=remaining.actual_tokens + tokens), replace(reservation, state="reconciled", charged_usd=amount, charged_tokens=tokens)


@dataclass(frozen=True)
class ModelPrice:
    """USD price per one million tokens; zero-cost is explicit, never inferred."""

    input_per_million: Decimal | None
    output_per_million: Decimal | None


def estimate_request_cost(
    *, input_tokens: int, output_tokens: int, price: ModelPrice, require_usd_ceiling: bool
) -> Decimal:
    _tokens(input_tokens)
    _tokens(output_tokens)
    if price.input_per_million is None or price.output_per_million is None:
        if require_usd_ceiling:
            raise PriceUnavailable("paid research is blocked until model token prices are configured")
        return Decimal("0")
    return (
        Decimal(input_tokens) * _amount(price.input_per_million)
        + Decimal(output_tokens) * _amount(price.output_per_million)
    ) / Decimal("1000000")


@tracer.start_as_current_span("twin.research.budget")
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
    span = trace.get_current_span()
    span.set_attribute("twin.operation", "research_budget")
    research_budget_usd.record(float(estimated_usd), {"kind": "reserved"})
    research_budget_tokens.record(estimated_tokens, {"kind": "reserved"})
    budget.reserve(reservation_id, key=key, estimated_usd=estimated_usd, estimated_tokens=estimated_tokens, policy=policy)
    budget.claim(reservation_id, key=key)
    try:
        result = operation()
    except Exception:
        budget.reconcile(reservation_id, key=key, actual_usd=None, actual_tokens=None)
        span.set_attribute("outcome", "uncertain")
        research_budget_operations.add(1, {"outcome": "uncertain"})
        raise
    usage = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(usage, dict):
        budget.reconcile(reservation_id, key=key, actual_usd=None, actual_tokens=None)
        span.set_attribute("outcome", "uncertain")
        research_budget_operations.add(1, {"outcome": "uncertain"})
        return result
    actual_tokens = usage.get("total_tokens")
    actual_usd = usage.get("cost_usd")
    amount, tokens, invalid = None, None, False
    try:
        amount = _amount(actual_usd) if actual_usd is not None else None
    except (ValueError, ArithmeticError, TypeError):
        invalid = True
    try:
        tokens = _tokens(actual_tokens) if actual_tokens is not None else None
    except (ValueError, ArithmeticError, TypeError):
        invalid = True
    if invalid:
        # Retain valid above-estimate facts even when the other field is bad.
        budget.reconcile(reservation_id, key=key, actual_usd=amount, actual_tokens=tokens)
        if amount is not None:
            research_budget_usd.record(float(amount), {"kind": "actual"})
        if tokens is not None:
            research_budget_tokens.record(tokens, {"kind": "actual"})
        span.set_attribute("outcome", "invalid_usage")
        research_budget_operations.add(1, {"outcome": "invalid_usage"})
        raise ValueError("provider returned invalid budget usage") from None
    budget.reconcile(
        reservation_id,
        key=key,
        actual_usd=amount,
        actual_tokens=tokens,
    )
    if amount is not None:
        research_budget_usd.record(float(amount), {"kind": "actual"})
    if tokens is not None:
        research_budget_tokens.record(tokens, {"kind": "actual"})
    outcome = "reconciled" if amount is not None and tokens is not None else "uncertain"
    span.set_attribute("outcome", outcome)
    research_budget_operations.add(1, {"outcome": outcome})
    return result


class InMemoryResearchBudget:
    """Thread-safe test implementation; live workers must use a durable adapter."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._usage: dict[str, BudgetUsage] = {}
        self._reservations: dict[tuple[str, str], BudgetReservation] = {}

    @staticmethod
    def key(strategy_id: str, account_scope_id: str, now: datetime) -> str:
        if now.tzinfo is None:
            raise ValueError("budget timestamps must be timezone-aware")
        return f"{strategy_id}:{account_scope_id}:{now.astimezone(timezone.utc).date().isoformat()}"

    def reserve(self, reservation_id: str, *, key: str, estimated_usd: Decimal, estimated_tokens: int, policy: BudgetPolicy) -> BudgetReservation:
        estimate = _amount(estimated_usd)
        _tokens(estimated_tokens)
        with self._lock:
            existing = self._reservations.get((key, reservation_id))
            if existing is not None:
                if (existing.estimated_usd, existing.estimated_tokens) != (estimate, estimated_tokens):
                    raise ValueError("budget reservation identity changed")
                return existing
            usage = self._usage.get(key, BudgetUsage(key))
            if usage.reserved_usd + usage.actual_usd + usage.uncertain_usd + estimate > policy.usd_limit:
                raise BudgetExceeded("worst-case research USD budget is exhausted")
            if usage.reserved_tokens + usage.actual_tokens + usage.uncertain_tokens + estimated_tokens > policy.token_limit:
                raise BudgetExceeded("research token budget is exhausted")
            if usage.requests + 1 > policy.request_limit:
                raise BudgetExceeded("research request budget is exhausted")
            reservation = BudgetReservation(reservation_id, key, estimate, estimated_tokens)
            self._reservations[(key, reservation_id)] = reservation
            self._usage[key] = replace(usage, reserved_usd=usage.reserved_usd + estimate, reserved_tokens=usage.reserved_tokens + estimated_tokens, requests=usage.requests + 1)
            return reservation

    def claim(self, reservation_id: str, *, key: str) -> None:
        with self._lock:
            reservation = self._reservations[(key, reservation_id)]
            if reservation.state != "reserved":
                raise BudgetAlreadyClaimed("research request was already claimed")
            self._reservations[(key, reservation_id)] = replace(reservation, state="claimed")

    def reconcile(
        self,
        reservation_id: str,
        *,
        actual_usd: Decimal | None,
        actual_tokens: int | None,
        key: str | None = None,
    ) -> BudgetUsage:
        with self._lock:
            if actual_usd is not None:
                _amount(actual_usd)
            if actual_tokens is not None:
                _tokens(actual_tokens)
            if key is None:
                matches = [r for (k, rid), r in self._reservations.items() if rid == reservation_id]
                if len(matches) != 1:
                    raise ValueError("daily key is required for ambiguous reservation identity")
                key = matches[0].key
            reservation = self._reservations[(key, reservation_id)]
            if key is not None and key != reservation.key:
                raise ValueError("budget reservation belongs to a different daily key")
            usage = self._usage[reservation.key]
            updated, reconciled = _reconcile_usage(usage, reservation, actual_usd, actual_tokens)
            self._usage[reservation.key] = updated
            self._reservations[(key, reservation_id)] = reconciled
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
        if entity.get("requests", 0) and "uncertain_tokens" not in entity:
            raise BudgetExceeded("legacy budget requires usage audit before new research")
        return BudgetUsage(
            key=key,
            reserved_usd=_amount(entity.get("reserved_usd", "0")),
            actual_usd=_amount(entity.get("actual_usd", "0")),
            uncertain_usd=_amount(entity.get("uncertain_usd", "0")),
            reserved_tokens=_tokens(entity.get("reserved_tokens", 0)),
            actual_tokens=_tokens(entity.get("actual_tokens", 0)),
            requests=_tokens(entity.get("requests", 0)),
            uncertain_tokens=_tokens(entity.get("uncertain_tokens", 0)),
        )

    @staticmethod
    def _write(entity, usage: BudgetUsage) -> None:
        entity.update({
            "reserved_usd": str(usage.reserved_usd), "actual_usd": str(usage.actual_usd),
            "uncertain_usd": str(usage.uncertain_usd), "reserved_tokens": usage.reserved_tokens,
            "actual_tokens": usage.actual_tokens, "requests": usage.requests,
            "uncertain_tokens": usage.uncertain_tokens,
        })

    def reserve(self, reservation_id: str, *, key: str, estimated_usd: Decimal, estimated_tokens: int, policy: BudgetPolicy) -> BudgetReservation:
        from google.cloud import datastore

        estimate = _amount(estimated_usd)
        _tokens(estimated_tokens)
        usage_key, reservation_key = self._usage_key(key), self._reservation_key(key, reservation_id)
        with self._client.transaction():
            existing = self._client.get(reservation_key)
            if existing is not None:
                if (Decimal(str(existing["estimated_usd"])), int(existing["estimated_tokens"])) != (estimate, estimated_tokens):
                    raise ValueError("budget reservation identity changed")
                return BudgetReservation(reservation_id, key, _amount(existing["estimated_usd"]), _tokens(existing["estimated_tokens"]), str(existing["state"]), _amount(existing.get("charged_usd", "0")), _tokens(existing.get("charged_tokens", 0)))
            entity = self._client.get(usage_key) or datastore.Entity(key=usage_key)
            usage = self._usage(key, entity)
            if usage.reserved_usd + usage.actual_usd + usage.uncertain_usd + estimate > policy.usd_limit:
                raise BudgetExceeded("worst-case research USD budget is exhausted")
            if usage.reserved_tokens + usage.actual_tokens + usage.uncertain_tokens + estimated_tokens > policy.token_limit or usage.requests + 1 > policy.request_limit:
                raise BudgetExceeded("research token or request budget is exhausted")
            updated = replace(usage, reserved_usd=usage.reserved_usd + estimate, reserved_tokens=usage.reserved_tokens + estimated_tokens, requests=usage.requests + 1)
            self._write(entity, updated)
            reservation = datastore.Entity(key=reservation_key)
            reservation.update({"estimated_usd": str(estimate), "estimated_tokens": estimated_tokens, "state": "reserved", "charged_usd": "0", "charged_tokens": 0})
            self._client.put_multi([entity, reservation])
            return BudgetReservation(reservation_id, key, estimate, estimated_tokens)

    def claim(self, reservation_id: str, *, key: str) -> None:
        with self._client.transaction():
            reservation = self._client.get(self._reservation_key(key, reservation_id))
            if reservation is None:
                raise KeyError("budget reservation was not found")
            # An old aggregate may have released unknown billed tokens.
            entity = self._client.get(self._usage_key(key))
            if entity is None:
                raise BudgetExceeded("budget aggregate was not found")
            usage = self._usage(key, entity)
            if reservation["state"] != "reserved":
                raise BudgetAlreadyClaimed("research request was already claimed")
            if usage.reserved_usd < _amount(reservation["estimated_usd"]) or usage.reserved_tokens < _tokens(reservation["estimated_tokens"]):
                raise BudgetExceeded("reservation exceeds persisted aggregate")
            reservation["state"] = "claimed"
            self._client.put(reservation)

    def reconcile(self, reservation_id: str, *, key: str, actual_usd: Decimal | None, actual_tokens: int | None) -> BudgetUsage:
        if actual_usd is not None:
            _amount(actual_usd)
        if actual_tokens is not None:
            _tokens(actual_tokens)
        reservation_key, usage_key = self._reservation_key(key, reservation_id), self._usage_key(key)
        with self._client.transaction():
            reservation, entity = self._client.get(reservation_key), self._client.get(usage_key)
            if reservation is None or entity is None:
                raise KeyError("budget reservation was not found")
            usage = self._usage(key, entity)
            record = BudgetReservation(reservation_id, key, _amount(reservation["estimated_usd"]), _tokens(reservation["estimated_tokens"]), str(reservation["state"]), _amount(reservation.get("charged_usd", "0")), _tokens(reservation.get("charged_tokens", 0)))
            updated, reconciled = _reconcile_usage(usage, record, actual_usd, actual_tokens)
            self._write(entity, updated)
            reservation.update({"state": reconciled.state, "charged_usd": str(reconciled.charged_usd), "charged_tokens": reconciled.charged_tokens})
            self._client.put_multi([entity, reservation])
            return updated

    def usage(self, key: str) -> BudgetUsage:
        entity = self._client.get(self._usage_key(key))
        return self._usage(key, entity) if entity is not None else BudgetUsage(key)
