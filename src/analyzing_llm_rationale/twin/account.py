"""Complete-generation account normalization for the autonomous twin.

Venue adapters provide already-fetched pages.  This layer never treats a
missing page, nullable financial field, or duplicate mutable row as evidence
that inventory is empty.  It retains the previous complete generation instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Sequence

from .models import Completeness, SchemaValidationError


@dataclass(frozen=True)
class AccountHolding:
    """A reconciled holding with only conservative, venue-observed economics."""

    instrument_id: str
    quantity: Decimal
    basis: Decimal
    liquidation_value: Decimal


@dataclass(frozen=True)
class AccountTolerance:
    """Venue-declared comparison precision; arbitrary broad epsilons are forbidden."""

    currency: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for field in ("currency", "quantity"):
            value = Decimal(str(getattr(self, field)))
            if not value.is_finite() or value < 0:
                raise SchemaValidationError(f"{field} tolerance must be a non-negative finite decimal")
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class AccountSnapshot:
    scope_id: str
    generation: int
    received_at: datetime
    completeness: Completeness
    available_cash: Decimal
    total_cash: Decimal
    reserved_cash: Decimal
    settled_cash: Decimal
    holdings: tuple[AccountHolding, ...]
    position_basis: Decimal
    fees_paid: Decimal
    conservative_liquidation_value: Decimal
    positions: tuple[Mapping[str, Any], ...]
    orders: tuple[Mapping[str, Any], ...]
    fills: tuple[Mapping[str, Any], ...]
    settlements: tuple[Mapping[str, Any], ...]
    external_activity_ids: tuple[str, ...]
    divergence: bool
    drift_reasons: tuple[str, ...]

    @property
    def blocks_new_exposure(self) -> bool:
        """A reconciler must pause entries until any discrepancy is explained."""
        return self.divergence


@dataclass(frozen=True)
class AccountSyncResult:
    snapshot: Optional[AccountSnapshot]
    retained_previous: bool
    issues: tuple[str, ...]


def portfolio_pages_from_complete_read(payload: Mapping[str, Any]) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Adapt a venue read only when its pagination contract is explicit.

    Existing UI reconciliation payloads are deliberately not accepted: their
    first-page limits are display bounds, not a proof that account inventory is
    complete.  A venue adapter must attest to a completed read before it can
    become capital authority for the autonomous twin.
    """
    if not isinstance(payload, Mapping) or payload.get("complete") is not True:
        raise SchemaValidationError("account portfolio read is not pagination-complete")
    required = ("balance", "positions", "orders", "fills", "settlements")
    if any(key not in payload for key in required):
        raise SchemaValidationError("complete portfolio read omitted an account collection")
    collections: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for key in required:
        value = payload[key]
        if key == "balance":
            rows = [value] if isinstance(value, Mapping) else []
        else:
            rows = list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []
        if (key == "balance" and not rows) or any(not isinstance(row, Mapping) for row in rows):
            raise SchemaValidationError(f"complete portfolio {key} collection is malformed")
        collections[f"{key}s" if key == "balance" else key] = ({"complete": True, "items": rows},)
    return collections


def _decimal(name: str, value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def _signed_decimal(name: str, value: Any) -> Optional[Decimal]:
    """Parse a finite signed quantity without making missing data look like zero."""
    if value in (None, ""):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return amount if amount.is_finite() else None


def _first_decimal(row: Mapping[str, Any], fields: tuple[str, ...]) -> Optional[Decimal]:
    for field in fields:
        if field in row:
            return _decimal(field, row[field])
    return None


def _position_holdings(
    positions: Sequence[Mapping[str, Any]],
) -> tuple[Optional[tuple[AccountHolding, ...]], Optional[str]]:
    holdings: list[AccountHolding] = []
    for row in positions:
        instrument_id = next(
            (str(row[field]).strip() for field in ("position_id", "token_id", "ticker") if row.get(field) not in (None, "")),
            "",
        )
        quantity = _signed_decimal("quantity", row.get("quantity", row.get("position")))
        basis = _first_decimal(row, ("basis", "cost_basis", "cost_basis_dollars"))
        if basis is None and quantity is not None:
            price = _first_decimal(row, ("average_price", "avg_price", "avgPrice"))
            basis = abs(quantity) * price if price is not None else None
        liquidation = _first_decimal(
            row,
            ("conservative_liquidation_value", "liquidation_value", "bid_value"),
        )
        if not instrument_id or quantity is None or basis is None or liquidation is None:
            return None, "position_economics_unavailable"
        holdings.append(AccountHolding(instrument_id, quantity, basis, liquidation))
    return tuple(sorted(holdings, key=lambda holding: holding.instrument_id)), None


def _fees_paid(
    fills: Sequence[Mapping[str, Any]], settlements: Sequence[Mapping[str, Any]]
) -> tuple[Optional[Decimal], Optional[str]]:
    fees = Decimal("0")
    for label, rows in (("fill", fills), ("settlement", settlements)):
        for row in rows:
            fee = _first_decimal(row, ("fee", "fee_cost", "fee_cost_dollars", "fees_paid"))
            if fee is None:
                return None, f"{label}_fee_unavailable"
            fees += fee
    return fees, None


def _settlements_are_final(settlements: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Require an explicit final state before settlement cash can be authoritative."""
    final_statuses = {"final", "settled", "completed", "confirmed"}
    for settlement in settlements:
        if settlement.get("final") is True:
            continue
        if str(settlement.get("status", "")).strip().lower() in final_statuses:
            continue
        return "settlement_not_final"
    return None


def _drift_reasons(
    *,
    available_cash: Decimal,
    reserved_cash: Decimal,
    holdings: Sequence[AccountHolding],
    local_available_cash: Optional[Any],
    local_reserved_cash: Optional[Any],
    local_holdings: Optional[Mapping[str, Any]],
    tolerance: AccountTolerance,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for label, observed, expected in (
        ("available_cash", available_cash, local_available_cash),
        ("reserved_cash", reserved_cash, local_reserved_cash),
    ):
        if expected is None:
            continue
        expected_value = _decimal(label, expected)
        if expected_value is None:
            reasons.append(f"local_{label}_invalid")
        elif abs(observed - expected_value) > tolerance.currency:
            reasons.append(f"{label}_mismatch")
    if local_holdings is None:
        return tuple(reasons)
    if not isinstance(local_holdings, Mapping):
        return tuple([*reasons, "local_holdings_invalid"])
    observed_holdings = {holding.instrument_id: holding.quantity for holding in holdings}
    expected_holdings: dict[str, Decimal] = {}
    for instrument_id, quantity in local_holdings.items():
        key = str(instrument_id).strip()
        value = _signed_decimal("local_holding", quantity)
        if not key or value is None:
            reasons.append("local_holdings_invalid")
            continue
        expected_holdings[key] = value
    for instrument_id in sorted(set(observed_holdings) | set(expected_holdings)):
        if abs(observed_holdings.get(instrument_id, Decimal("0")) - expected_holdings.get(instrument_id, Decimal("0"))) > tolerance.quantity:
            reasons.append(f"holding_mismatch:{instrument_id}")
    return tuple(reasons)


def _page_rows(label: str, pages: Sequence[Mapping[str, Any]]) -> tuple[Optional[list[Mapping[str, Any]]], Optional[str]]:
    rows: list[Mapping[str, Any]] = []
    if not pages:
        return None, f"{label}_pages_missing"
    for page in pages:
        if (
            not isinstance(page, Mapping)
            or page.get("complete") is not True
            or page.get("has_more") is True
        ):
            return None, f"{label}_pagination_incomplete"
        value = page.get("items")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return None, f"{label}_malformed"
        if any(not isinstance(item, Mapping) for item in value):
            return None, f"{label}_malformed"
        rows.extend(value)
    return rows, None


def _dedupe(label: str, rows: Sequence[Mapping[str, Any]], *, id_fields: tuple[str, ...]) -> tuple[Optional[tuple[Mapping[str, Any], ...]], Optional[str]]:
    found: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = next((str(row[field]).strip() for field in id_fields if row.get(field) not in (None, "")), "")
        if not identity:
            return None, f"{label}_missing_immutable_id"
        prior = found.get(identity)
        if prior is not None and dict(prior) != dict(row):
            return None, f"{label}_duplicate_conflict"
        found[identity] = row
    return tuple(found[key] for key in sorted(found)), None


def synchronize_account(
    scope_id: str,
    *,
    generation: int,
    received_at: datetime,
    balances: Sequence[Mapping[str, Any]],
    positions: Sequence[Mapping[str, Any]],
    orders: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]],
    settlements: Sequence[Mapping[str, Any]],
    local_command_ids: set[str],
    previous: Optional[AccountSnapshot] = None,
    local_available_cash: Optional[Any] = None,
    local_reserved_cash: Optional[Any] = None,
    local_holdings: Optional[Mapping[str, Any]] = None,
    tolerance: Optional[AccountTolerance] = None,
) -> AccountSyncResult:
    """Atomically validate a full venue generation or retain the last one."""
    if not scope_id or generation < 1 or received_at.tzinfo is None:
        raise SchemaValidationError("scope_id, positive generation, and UTC-aware received_at are required")
    if previous is not None:
        if previous.scope_id != scope_id:
            raise SchemaValidationError("previous snapshot belongs to a different account scope")
        if generation <= previous.generation:
            return AccountSyncResult(previous, True, ("stale_or_replayed_generation",))
    pages = {"balances": balances, "positions": positions, "orders": orders, "fills": fills, "settlements": settlements}
    flat: dict[str, list[Mapping[str, Any]]] = {}
    issues: list[str] = []
    for label, source in pages.items():
        rows, error = _page_rows(label, source)
        if error:
            issues.append(error)
        else:
            flat[label] = rows or []
    if issues:
        return AccountSyncResult(previous, previous is not None, tuple(issues))

    balance_rows = flat["balances"]
    if len(balance_rows) != 1:
        return AccountSyncResult(previous, previous is not None, ("balance_ambiguous",))
    balance = balance_rows[0]
    available, total, reserved, settled = (
        _decimal("available", balance.get("available")),
        _decimal("total", balance.get("total")),
        _decimal("reserved", balance.get("reserved")),
        _decimal("settled_cash", balance.get("settled_cash")),
    )
    if (
        available is None
        or total is None
        or reserved is None
        or settled is None
        or available + reserved > total
        or settled > total
    ):
        return AccountSyncResult(previous, previous is not None, ("balance_unavailable_or_inconsistent",))

    deduped: dict[str, tuple[Mapping[str, Any], ...]] = {}
    field_map = {"positions": ("position_id", "token_id", "ticker"), "orders": ("order_id", "id"), "fills": ("fill_id", "trade_id", "id"), "settlements": ("settlement_id", "id")}
    for label, fields in field_map.items():
        rows, error = _dedupe(label, flat[label], id_fields=fields)
        if error:
            issues.append(error)
        else:
            deduped[label] = rows or ()
    if issues:
        return AccountSyncResult(previous, previous is not None, tuple(issues))

    settlement_error = _settlements_are_final(deduped["settlements"])
    holdings, holdings_error = _position_holdings(deduped["positions"])
    fees_paid, fees_error = _fees_paid(deduped["fills"], deduped["settlements"])
    if settlement_error or holdings_error or fees_error:
        return AccountSyncResult(
            previous,
            previous is not None,
            tuple(error for error in (settlement_error, holdings_error, fees_error) if error),
        )
    assert holdings is not None and fees_paid is not None
    comparison_tolerance = tolerance or AccountTolerance()
    drift_reasons = _drift_reasons(
        available_cash=available,
        reserved_cash=reserved,
        holdings=holdings,
        local_available_cash=local_available_cash,
        local_reserved_cash=local_reserved_cash,
        local_holdings=local_holdings,
        tolerance=comparison_tolerance,
    )
    position_basis = sum((holding.basis for holding in holdings), Decimal("0"))
    conservative_liquidation_value = settled + sum(
        (holding.liquidation_value for holding in holdings), Decimal("0")
    )

    external: list[str] = []
    for order in deduped["orders"]:
        command_id = str(order.get("client_order_id") or order.get("command_id") or "").strip()
        if command_id and command_id not in local_command_ids:
            external.append(command_id)
    for fill in deduped["fills"]:
        command_id = str(fill.get("client_order_id") or fill.get("command_id") or "").strip()
        if command_id and command_id not in local_command_ids:
            external.append(command_id)
    snapshot = AccountSnapshot(
        scope_id=scope_id, generation=generation, received_at=received_at, completeness=Completeness.COMPLETE,
        available_cash=available, total_cash=total, reserved_cash=reserved, settled_cash=settled,
        holdings=holdings, position_basis=position_basis, fees_paid=fees_paid,
        conservative_liquidation_value=conservative_liquidation_value, positions=deduped["positions"],
        orders=deduped["orders"], fills=deduped["fills"], settlements=deduped["settlements"],
        external_activity_ids=tuple(sorted(set(external))),
        divergence=bool(external or drift_reasons), drift_reasons=drift_reasons,
    )
    return AccountSyncResult(snapshot, False, ())
