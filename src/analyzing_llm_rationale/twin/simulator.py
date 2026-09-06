"""Seeded, network-free binary venue adapter for shadow execution.

It implements the small submit/status/cancel/account/settle surface the twin
needs. Prices and depth are supplied from captured snapshots; this adapter
never fetches a current book and never writes to a venue.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from random import Random
from typing import Optional

_ZERO = Decimal("0")
_ONE = Decimal("1")


@dataclass(frozen=True)
class ShadowReceipt:
    order_id: str
    filled_quantity: Decimal
    remaining_quantity: Decimal
    fee: Decimal
    status: str
    instrument_id: str = ""
    outcome: str = "yes"
    spent: Decimal = _ZERO
    settled_payout: Decimal = _ZERO


@dataclass(frozen=True)
class ShadowAccount:
    account_id: str
    cash: Decimal
    positions: tuple[tuple[str, str, Decimal], ...]
    seed: int
    simulator_version: str = "shadow-venue-v2"


class ShadowVenue:
    """A deterministic, depth-limited shadow adapter with explicit cancel races."""

    def __init__(
        self, *, account_id: str, seed: int, fee_rate: Decimal = _ZERO,
        starting_cash: Decimal = Decimal("1000"), adverse_no_fill_probability: Decimal = Decimal("0.1"),
    ) -> None:
        if not account_id or account_id.lower().startswith("live"):
            raise ValueError("shadow account identifiers must not overlap live accounts")
        fee = _decimal("fee_rate", fee_rate)
        cash = _decimal("starting_cash", starting_cash)
        no_fill = _decimal("adverse_no_fill_probability", adverse_no_fill_probability)
        if fee < _ZERO or not _ZERO <= no_fill <= _ONE:
            raise ValueError("invalid shadow venue configuration")
        self.account_id = account_id
        self.seed = seed
        self._random = Random(seed)
        self.fee_rate = fee
        self.adverse_no_fill_probability = no_fill
        self._cash = cash
        self._orders: dict[str, ShadowReceipt] = {}
        self._positions: dict[tuple[str, str], Decimal] = {}

    def submit(
        self, order_id: str, *, quantity: Decimal, ask: Decimal, depth: Decimal,
        instrument_id: Optional[str] = None, outcome: str = "yes",
    ) -> ShadowReceipt:
        """Fill only against captured visible depth and account cash."""
        if order_id in self._orders:
            return self._orders[order_id]
        quantity, price, visible_depth = _positive("quantity", quantity), _price(ask), _nonnegative("depth", depth)
        instrument = str(instrument_id or order_id).strip()
        side = str(outcome).lower().strip()
        if not instrument or side not in {"yes", "no"}:
            raise ValueError("shadow orders require an instrument and yes/no outcome")
        requested = min(quantity, visible_depth)
        if self._random.random() < float(self.adverse_no_fill_probability):
            requested = _ZERO
        unit_cost = price * (_ONE + self.fee_rate)
        affordable = self._cash / unit_cost if unit_cost else _ZERO
        filled = min(requested, affordable)
        fee = filled * price * self.fee_rate
        spent = filled * price + fee
        self._cash -= spent
        self._positions[(instrument, side)] = self._positions.get((instrument, side), _ZERO) + filled
        receipt = ShadowReceipt(
            order_id, filled, quantity - filled, fee,
            "filled" if filled == quantity else "partial" if filled else "open",
            instrument, side, spent,
        )
        self._orders[order_id] = receipt
        return receipt

    def status(self, order_id: str) -> ShadowReceipt:
        return self._orders[order_id]

    def cancel(self, order_id: str, *, fill_race_quantity: Decimal = _ZERO) -> ShadowReceipt:
        """Cancel the remainder, retaining any explicit fill that won the race."""
        current = self._orders[order_id]
        if current.status in {"cancelled", "settled"}:
            return current
        race = min(_nonnegative("fill_race_quantity", fill_race_quantity), current.remaining_quantity)
        if race:
            price = current.spent / (current.filled_quantity * (_ONE + self.fee_rate)) if current.filled_quantity else _ZERO
            fee = race * price * self.fee_rate
            spent = race * price + fee
            if spent > self._cash:
                race = self._cash / (price * (_ONE + self.fee_rate)) if price else _ZERO
                fee, spent = race * price * self.fee_rate, race * price * (_ONE + self.fee_rate)
            self._cash -= spent
            self._positions[(current.instrument_id, current.outcome)] = self._positions.get((current.instrument_id, current.outcome), _ZERO) + race
            current = replace(current, filled_quantity=current.filled_quantity + race, remaining_quantity=current.remaining_quantity - race, fee=current.fee + fee, spent=current.spent + spent)
        receipt = replace(current, status="cancelled")
        self._orders[order_id] = receipt
        return receipt

    def settle(self, order_id: str, *, resolved_outcome: str) -> ShadowReceipt:
        """Settle held filled shares once; delayed settlement cannot double-credit cash."""
        current = self._orders[order_id]
        if current.status == "settled":
            return current
        resolution = str(resolved_outcome).lower().strip()
        if resolution not in {"yes", "no"}:
            raise ValueError("resolved_outcome must be yes or no")
        payout = current.filled_quantity if resolution == current.outcome else _ZERO
        self._cash += payout
        key = (current.instrument_id, current.outcome)
        self._positions[key] = max(_ZERO, self._positions.get(key, _ZERO) - current.filled_quantity)
        receipt = replace(current, status="settled", settled_payout=payout)
        self._orders[order_id] = receipt
        return receipt

    def account(self) -> ShadowAccount:
        positions = tuple(sorted((instrument, outcome, quantity) for (instrument, outcome), quantity in self._positions.items() if quantity > _ZERO))
        return ShadowAccount(self.account_id, self._cash, positions, self.seed)


def _decimal(name: str, value: Decimal) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative(name: str, value: Decimal) -> Decimal:
    parsed = _decimal(name, value)
    if parsed < _ZERO:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive(name: str, value: Decimal) -> Decimal:
    parsed = _nonnegative(name, value)
    if parsed <= _ZERO:
        raise ValueError(f"{name} must be positive")
    return parsed


def _price(value: Decimal) -> Decimal:
    parsed = _positive("ask", value)
    if parsed >= _ONE:
        raise ValueError("ask must be below one")
    return parsed
