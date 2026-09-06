"""Atomic worst-case research-budget reservations before provider calls."""
from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal


class BudgetExceeded(RuntimeError):
    pass


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


class InMemoryResearchBudget:
    """Thread-safe contract used by workers; durable adapter follows in T16."""

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
