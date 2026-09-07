"""Deterministic, network-free venue adapter for shadow execution.

Every fill comes from a versioned captured book.  Preview freezes the exact
depth consumed; submit never fetches or substitutes a current book.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, Mapping, Optional, Sequence

from .account import AccountHolding, AccountSnapshot
from .models import (
    CommandState,
    Completeness,
    Instrument,
    MarketSnapshot,
    ProposalAction,
    TradeIntent,
)
from .risk import RiskResult
from .store import ExecutionCommand

_ZERO = Decimal("0")
_ONE = Decimal("1")
SIMULATOR_VERSION = "shadow-venue-v3"


@dataclass(frozen=True)
class DepthLevel:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", _price(self.price))
        object.__setattr__(self, "quantity", _positive("quantity", self.quantity))


@dataclass(frozen=True)
class CapturedBook:
    snapshot: MarketSnapshot
    outcome: str
    levels: tuple[DepthLevel, ...]

    def __post_init__(self) -> None:
        outcome = str(self.outcome).lower().strip()
        if outcome not in {"yes", "no"}:
            raise ValueError("captured book outcome must be yes or no")
        if not isinstance(self.levels, tuple) or any(not isinstance(level, DepthLevel) for level in self.levels):
            raise ValueError("captured book requires typed immutable levels")
        object.__setattr__(self, "outcome", outcome)


@dataclass(frozen=True)
class ShadowAssumptions:
    fee_rate: Decimal = _ZERO
    latency_ms: int = 0
    adverse_price_ticks: int = 0
    no_fill_probability: Decimal = _ZERO

    def __post_init__(self) -> None:
        fee = _nonnegative("fee_rate", self.fee_rate)
        no_fill = _nonnegative("no_fill_probability", self.no_fill_probability)
        if fee >= _ONE or no_fill > _ONE:
            raise ValueError("shadow fee and no-fill assumptions must be below valid bounds")
        if not isinstance(self.latency_ms, int) or self.latency_ms < 0:
            raise ValueError("latency_ms must be a non-negative integer")
        if not isinstance(self.adverse_price_ticks, int) or self.adverse_price_ticks < 0:
            raise ValueError("adverse_price_ticks must be a non-negative integer")
        object.__setattr__(self, "fee_rate", fee)
        object.__setattr__(self, "no_fill_probability", no_fill)

    def payload(self) -> dict[str, Any]:
        return {
            "fee_rate": str(self.fee_rate), "latency_ms": self.latency_ms,
            "adverse_price_ticks": self.adverse_price_ticks,
            "no_fill_probability": str(self.no_fill_probability),
        }


@dataclass(frozen=True)
class ShadowPreview:
    preview_hash: str
    intent_hash: str
    market_snapshot_id: str
    market_received_at: datetime
    instrument_id: str
    action: ProposalAction
    outcome: str
    requested_quantity: Decimal
    limit_price: Decimal
    liquidation_bid: Decimal
    planned_fills: tuple[DepthLevel, ...]
    planned_fee: Decimal
    planned_cash_delta: Decimal
    seed: int
    simulator_version: str
    assumptions_hash: str

    @property
    def planned_quantity(self) -> Decimal:
        return sum((level.quantity for level in self.planned_fills), _ZERO)


@dataclass(frozen=True)
class ShadowReceipt:
    order_id: str
    client_order_id: str
    intent_hash: str
    preview_hash: str
    filled_quantity: Decimal
    remaining_quantity: Decimal
    cancelled_quantity: Decimal
    fee: Decimal
    status: str
    instrument_id: str
    outcome: str
    action: ProposalAction
    cash_delta: Decimal
    settled_payout: Decimal = _ZERO


@dataclass(frozen=True)
class ShadowEvent:
    id: str
    sequence: int
    event_type: str
    order_id: str
    occurred_at: datetime
    quantity: Decimal = _ZERO
    cash_delta: Decimal = _ZERO
    fee: Decimal = _ZERO
    market_snapshot_id: Optional[str] = None


@dataclass(frozen=True)
class ShadowRun:
    run_id: str
    account_id: str
    scope_id: str
    seed: int
    simulator_version: str
    assumptions: Mapping[str, Any]
    assumptions_hash: str


class ShadowVenue:
    """A venue-shaped adapter driven only by captured immutable inputs."""

    def __init__(
        self, *, account_id: str, seed: int, scope_id: Optional[str] = None,
        starting_cash: Decimal = Decimal("1000"), assumptions: Optional[ShadowAssumptions] = None,
        fee_rate: Decimal = _ZERO, adverse_no_fill_probability: Decimal = _ZERO,
    ) -> None:
        account = str(account_id).strip()
        if not account or not account.lower().startswith("shadow-"):
            raise ValueError("shadow account identifiers must start with shadow-")
        scope = str(scope_id or f"shadow-scope:{account}").strip()
        if not scope.startswith("shadow-"):
            raise ValueError("shadow scope identifiers must start with shadow-")
        self.account_id = account
        self.scope_id = scope
        self.seed = int(seed)
        self.assumptions = assumptions or ShadowAssumptions(
            fee_rate=fee_rate, no_fill_probability=adverse_no_fill_probability,
        )
        payload = self.assumptions.payload()
        self.assumptions_hash = _hash(payload)
        self.run = ShadowRun(
            _hash({"account_id": account, "scope_id": scope, "seed": self.seed,
                   "version": SIMULATOR_VERSION, "assumptions_hash": self.assumptions_hash}),
            account, scope, self.seed, SIMULATOR_VERSION, payload, self.assumptions_hash,
        )
        self._cash = _nonnegative("starting_cash", starting_cash)
        self._orders: dict[str, ShadowReceipt] = {}
        self._previews: dict[str, ShadowPreview] = {}
        self._positions: dict[tuple[str, str], Decimal] = {}
        self._basis: dict[tuple[str, str], Decimal] = {}
        self._marks: dict[tuple[str, str], Decimal] = {}
        self._events: list[ShadowEvent] = []
        self._fills: list[dict[str, Any]] = []
        self._settlements: list[dict[str, Any]] = []

    def preview(
        self, intent: TradeIntent, risk: RiskResult, instrument: Instrument,
        book: CapturedBook, *, now: datetime,
    ) -> ShadowPreview:
        """Freeze a fill plan from one captured book and validated risk result."""
        _aware("preview time", now)
        snapshot = book.snapshot
        if snapshot.complete is not Completeness.COMPLETE:
            raise ValueError("shadow preview requires a complete captured book")
        age = (now.astimezone(timezone.utc) - snapshot.received_at.astimezone(timezone.utc)).total_seconds()
        if age < 0 or age > snapshot.stale_after_seconds:
            raise ValueError("captured book is outside its declared freshness window")
        if instrument.id != intent.instrument_id or snapshot.instrument_id != intent.instrument_id:
            raise ValueError("intent, instrument, and book identities do not match")
        if intent.market_version != snapshot.id or risk.market_snapshot_id != snapshot.id:
            raise ValueError("shadow preview is not bound to the validated market version")
        if risk.reason is not None or risk.quantity != intent.quantity:
            raise ValueError("shadow preview requires a matching accepted risk result")
        outcome, buying = _action(intent.action)
        if outcome != book.outcome:
            raise ValueError("captured book outcome does not match the intent")
        if intent.quantity < instrument.min_quantity or intent.quantity % instrument.min_quantity:
            raise ValueError("intent quantity violates captured venue size rules")
        expected_top = (
            snapshot.yes_ask if buying and outcome == "yes" else
            snapshot.no_ask if buying else
            snapshot.yes_bid if outcome == "yes" else snapshot.no_bid
        )
        if expected_top is None or (book.levels and book.levels[0].price != expected_top):
            raise ValueError("captured depth does not start at the executable snapshot price")
        if any(level.price % instrument.tick_size for level in book.levels):
            raise ValueError("captured depth violates venue tick size")
        if any(level.quantity % instrument.min_quantity for level in book.levels):
            raise ValueError("captured depth violates venue size increment")
        prices = tuple(level.price for level in book.levels)
        if prices != tuple(sorted(prices, reverse=not buying)):
            raise ValueError("captured depth is not in executable price order")
        liquidation_bid = snapshot.yes_bid if outcome == "yes" else snapshot.no_bid
        if liquidation_bid is None:
            raise ValueError("captured book is missing a conservative liquidation bid")
        planned = () if self._draw(intent.intent_hash, "submit") < self.assumptions.no_fill_probability else self._plan(
            book.levels, quantity=intent.quantity, limit=intent.limit_price,
            buying=buying, tick=instrument.tick_size,
        )
        notional = sum((level.price * level.quantity for level in planned), _ZERO)
        fee = notional * self.assumptions.fee_rate
        cash_delta = -(notional + fee) if buying else notional - fee
        if buying and -cash_delta > risk.cash:
            raise ValueError("captured fill would exceed the validated cash flow")
        body = {
            "intent_hash": intent.intent_hash, "snapshot": snapshot.id, "instrument": instrument.id,
            "action": intent.action.value, "quantity": str(intent.quantity), "limit": str(intent.limit_price),
            "liquidation_bid": str(liquidation_bid),
            "fills": [(str(level.price), str(level.quantity)) for level in planned],
            "fee": str(fee), "cash_delta": str(cash_delta), "seed": self.seed,
            "simulator_version": SIMULATOR_VERSION, "assumptions_hash": self.assumptions_hash,
        }
        preview = ShadowPreview(
            _hash(body), intent.intent_hash, snapshot.id, snapshot.received_at, instrument.id,
            intent.action, outcome, intent.quantity, intent.limit_price, liquidation_bid, tuple(planned), fee,
            cash_delta, self.seed, SIMULATOR_VERSION, self.assumptions_hash,
        )
        existing = self._previews.get(intent.intent_hash)
        if existing is not None and existing != preview:
            raise ValueError("intent already has a different shadow preview")
        self._previews[intent.intent_hash] = preview
        return preview

    def submit(
        self, command: ExecutionCommand, preview: ShadowPreview, *, now: datetime,
    ) -> dict[str, Any]:
        """Acknowledge one prepared command and apply its frozen fill plan once."""
        _aware("submit time", now)
        order_id = f"shadow-order:{command.client_order_id}"
        existing = self._orders.get(order_id)
        if existing is not None:
            if existing.preview_hash != preview.preview_hash:
                raise ValueError("client order identity already has a different preview")
            return self._ack(existing)
        if command.intent_hash != preview.intent_hash or self._previews.get(command.intent_hash) != preview:
            raise ValueError("command is not bound to a stored shadow preview")
        if command.state is not CommandState.SUBMITTING:
            raise ValueError("shadow submit requires a claimed submitting command")
        quantity = preview.planned_quantity
        outcome, buying = _action(preview.action)
        key = (preview.instrument_id, outcome)
        if buying:
            required = -preview.planned_cash_delta
            if required > self._cash:
                raise ValueError("shadow account cash changed after preview")
            self._cash -= required
            self._positions[key] = self._positions.get(key, _ZERO) + quantity
            self._basis[key] = self._basis.get(key, _ZERO) + required
        else:
            if quantity > self._positions.get(key, _ZERO):
                raise ValueError("shadow inventory changed after reduce-only preview")
            self._cash += preview.planned_cash_delta
            self._positions[key] -= quantity
            old_quantity = self._positions[key] + quantity
            old_basis = self._basis.get(key, _ZERO)
            self._basis[key] = old_basis * (self._positions[key] / old_quantity) if old_quantity else _ZERO
        self._marks[key] = preview.liquidation_bid
        status = "filled" if quantity == preview.requested_quantity else "partial" if quantity else "open"
        receipt = ShadowReceipt(
            order_id, command.client_order_id, command.intent_hash, preview.preview_hash,
            quantity, preview.requested_quantity - quantity, _ZERO, preview.planned_fee,
            status, preview.instrument_id, outcome, preview.action, preview.planned_cash_delta,
        )
        self._orders[order_id] = receipt
        self._event("order_acknowledged", order_id, now, market_snapshot_id=preview.market_snapshot_id)
        if quantity:
            self._fills.append({
                "fill_id": f"shadow-fill:{order_id}:1", "order_id": order_id,
                "client_order_id": command.client_order_id, "instrument_id": preview.instrument_id,
                "outcome": outcome, "quantity": quantity,
                "price": sum((level.price * level.quantity for level in preview.planned_fills), _ZERO) / quantity,
                "fee": preview.planned_fee, "filled_at": now,
            })
            self._event(
                "fill", order_id, now, quantity=quantity, cash_delta=preview.planned_cash_delta,
                fee=preview.planned_fee, market_snapshot_id=preview.market_snapshot_id,
            )
        return self._ack(receipt)

    def status(self, order_id: str) -> ShadowReceipt:
        return self._orders[order_id]

    def cancel(
        self, order_id: str, *, now: datetime, race_book: Optional[CapturedBook] = None,
        instrument: Optional[Instrument] = None,
    ) -> ShadowReceipt:
        """Cancel remaining quantity; optional race fills must come from a later book."""
        _aware("cancel time", now)
        current = self._orders[order_id]
        if current.status in {"cancelled", "settled", "filled"}:
            return current
        race_quantity = _ZERO
        race_fee = _ZERO
        race_cash = _ZERO
        if race_book is not None:
            if instrument is None or race_book.snapshot.instrument_id != current.instrument_id:
                raise ValueError("cancel-race book does not match the order instrument")
            preview = self._previews[current.intent_hash]
            if race_book.snapshot.received_at <= preview.market_received_at:
                raise ValueError("cancel-race fill requires a later captured book")
            outcome, buying = _action(current.action)
            if race_book.outcome != outcome:
                raise ValueError("cancel-race book outcome does not match the order")
            fills = self._plan(
                race_book.levels, quantity=current.remaining_quantity, limit=preview.limit_price,
                buying=buying, tick=instrument.tick_size,
            )
            race_quantity = sum((level.quantity for level in fills), _ZERO)
            notional = sum((level.price * level.quantity for level in fills), _ZERO)
            race_fee = notional * self.assumptions.fee_rate
            race_cash = -(notional + race_fee) if buying else notional - race_fee
            key = (current.instrument_id, current.outcome)
            if buying:
                affordable = self._cash / (notional / race_quantity * (_ONE + self.assumptions.fee_rate)) if race_quantity else _ZERO
                if affordable < race_quantity:
                    raise ValueError("cancel-race fill exceeds shadow account cash")
                self._cash += race_cash
                self._positions[key] = self._positions.get(key, _ZERO) + race_quantity
                self._basis[key] = self._basis.get(key, _ZERO) - race_cash
            else:
                if race_quantity > self._positions.get(key, _ZERO):
                    raise ValueError("cancel-race fill exceeds shadow inventory")
                self._cash += race_cash
                old_quantity = self._positions[key]
                old_basis = self._basis.get(key, _ZERO)
                self._positions[key] -= race_quantity
                self._basis[key] = old_basis * (self._positions[key] / old_quantity) if old_quantity else _ZERO
            if race_quantity:
                self._fills.append({
                    "fill_id": f"shadow-fill:{order_id}:cancel-race", "order_id": order_id,
                    "client_order_id": current.client_order_id, "instrument_id": current.instrument_id,
                    "outcome": current.outcome, "quantity": race_quantity,
                    "price": notional / race_quantity, "fee": race_fee, "filled_at": now,
                })
                mark = race_book.snapshot.yes_bid if current.outcome == "yes" else race_book.snapshot.no_bid
                if mark is not None:
                    self._marks[key] = mark
                self._event(
                    "fill_after_cancel_request", order_id, now, quantity=race_quantity,
                    cash_delta=race_cash, fee=race_fee,
                    market_snapshot_id=race_book.snapshot.id,
                )
        cancelled = current.remaining_quantity - race_quantity
        updated = replace(
            current, filled_quantity=current.filled_quantity + race_quantity,
            remaining_quantity=_ZERO, cancelled_quantity=cancelled,
            fee=current.fee + race_fee, cash_delta=current.cash_delta + race_cash,
            status="cancelled",
        )
        self._orders[order_id] = updated
        self._event("cancelled", order_id, now, quantity=cancelled)
        return updated

    def settle(self, order_id: str, *, resolved_outcome: str, settled_at: datetime) -> ShadowReceipt:
        """Credit a delayed final outcome once and append an immutable event."""
        _aware("settlement time", settled_at)
        current = self._orders[order_id]
        if current.status == "settled":
            return current
        resolution = str(resolved_outcome).lower().strip()
        if resolution not in {"yes", "no"}:
            raise ValueError("resolved_outcome must be yes or no")
        key = (current.instrument_id, current.outcome)
        buying = current.action in {ProposalAction.BUY_YES, ProposalAction.BUY_NO}
        held = min(current.filled_quantity, self._positions.get(key, _ZERO)) if buying else _ZERO
        payout = held if resolution == current.outcome else _ZERO
        self._cash += payout
        old_quantity = self._positions.get(key, _ZERO)
        old_basis = self._basis.get(key, _ZERO)
        self._positions[key] = old_quantity - held
        self._basis[key] = old_basis * (self._positions[key] / old_quantity) if old_quantity else _ZERO
        settlement_id = f"shadow-settlement:{order_id}"
        self._settlements.append({
            "settlement_id": settlement_id, "order_id": order_id, "instrument_id": current.instrument_id,
            "outcome": resolution, "quantity": held, "payout": payout, "status": "final",
            "settled_at": settled_at,
        })
        updated = replace(current, status="settled", settled_payout=payout)
        self._orders[order_id] = updated
        self._event("settled", order_id, settled_at, quantity=held, cash_delta=payout)
        return updated

    def account(self, *, received_at: datetime) -> AccountSnapshot:
        """Return the canonical account abstraction used by reconciliation/risk."""
        _aware("account time", received_at)
        holdings = tuple(
            AccountHolding(
                f"{instrument_id}:{outcome}", quantity, self._basis.get((instrument_id, outcome), _ZERO),
                quantity * self._marks.get((instrument_id, outcome), _ZERO),
            )
            for (instrument_id, outcome), quantity in sorted(self._positions.items()) if quantity > _ZERO
        )
        positions = tuple({
            "position_id": holding.instrument_id, "instrument_id": holding.instrument_id,
            "quantity": holding.quantity, "basis": holding.basis,
            "liquidation_value": holding.liquidation_value,
        } for holding in holdings)
        orders = tuple({
            "order_id": receipt.order_id, "client_order_id": receipt.client_order_id,
            "instrument_id": receipt.instrument_id, "outcome": receipt.outcome,
            "status": receipt.status, "filled_quantity": receipt.filled_quantity,
            "remaining_quantity": receipt.remaining_quantity,
        } for receipt in sorted(self._orders.values(), key=lambda item: item.order_id))
        return AccountSnapshot(
            scope_id=self.scope_id, generation=len(self._events), received_at=received_at,
            completeness=Completeness.COMPLETE, available_cash=self._cash, total_cash=self._cash,
            reserved_cash=_ZERO, settled_cash=self._cash, holdings=holdings,
            position_basis=sum((item.basis for item in holdings), _ZERO),
            fees_paid=sum((Decimal(str(item["fee"])) for item in self._fills), _ZERO),
            conservative_liquidation_value=self._cash + sum((item.liquidation_value for item in holdings), _ZERO),
            positions=positions, orders=orders, fills=tuple(self._fills), settlements=tuple(self._settlements),
            external_activity_ids=(), divergence=False, drift_reasons=(),
        )

    def events(self) -> tuple[ShadowEvent, ...]:
        return tuple(self._events)

    def _plan(
        self, levels: Sequence[DepthLevel], *, quantity: Decimal, limit: Decimal,
        buying: bool, tick: Decimal,
    ) -> tuple[DepthLevel, ...]:
        remaining = quantity
        planned: list[DepthLevel] = []
        adverse = tick * self.assumptions.adverse_price_ticks
        for level in levels:
            price = level.price + adverse if buying else level.price - adverse
            if not _ZERO < price < _ONE or (buying and price > limit) or (not buying and price < limit):
                continue
            filled = min(remaining, level.quantity)
            if filled:
                planned.append(DepthLevel(price, filled))
                remaining -= filled
            if remaining <= _ZERO:
                break
        return tuple(planned)

    def _draw(self, identity: str, phase: str) -> Decimal:
        digest = sha256(f"{self.seed}:{identity}:{phase}".encode("utf-8")).digest()
        return Decimal(int.from_bytes(digest, "big")) / Decimal(2 ** (8 * len(digest)))

    def _event(
        self, event_type: str, order_id: str, occurred_at: datetime, *, quantity: Decimal = _ZERO,
        cash_delta: Decimal = _ZERO, fee: Decimal = _ZERO, market_snapshot_id: Optional[str] = None,
    ) -> None:
        sequence = len(self._events) + 1
        self._events.append(ShadowEvent(
            f"shadow-event:{self.run.run_id[:12]}:{sequence}", sequence, event_type, order_id,
            occurred_at, quantity, cash_delta, fee, market_snapshot_id,
        ))

    @staticmethod
    def _ack(receipt: ShadowReceipt) -> dict[str, Any]:
        return {
            "acknowledgement": {
                "acknowledged": True, "status": "accepted", "venue_order_id": receipt.order_id,
                "client_order_id": receipt.client_order_id,
            }
        }


def _action(action: ProposalAction) -> tuple[str, bool]:
    mapping = {
        ProposalAction.BUY_YES: ("yes", True), ProposalAction.BUY_NO: ("no", True),
        ProposalAction.SELL_YES: ("yes", False), ProposalAction.SELL_NO: ("no", False),
    }
    try:
        return mapping[action]
    except KeyError as exc:
        raise ValueError("shadow venue supports only binary buy or sell actions") from exc


def _hash(payload: Mapping[str, Any]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _aware(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")


def _decimal(name: str, value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative(name: str, value: Any) -> Decimal:
    parsed = _decimal(name, value)
    if parsed < _ZERO:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive(name: str, value: Any) -> Decimal:
    parsed = _nonnegative(name, value)
    if parsed <= _ZERO:
        raise ValueError(f"{name} must be positive")
    return parsed


def _price(value: Any) -> Decimal:
    parsed = _positive("price", value)
    if parsed >= _ONE:
        raise ValueError("price must be below one")
    return parsed
