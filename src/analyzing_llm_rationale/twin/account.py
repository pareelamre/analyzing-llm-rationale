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
class AccountSnapshot:
    scope_id: str
    generation: int
    received_at: datetime
    completeness: Completeness
    available_cash: Decimal
    total_cash: Decimal
    reserved_cash: Decimal
    positions: tuple[Mapping[str, Any], ...]
    orders: tuple[Mapping[str, Any], ...]
    fills: tuple[Mapping[str, Any], ...]
    settlements: tuple[Mapping[str, Any], ...]
    external_activity_ids: tuple[str, ...]
    divergence: bool


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
    available, total, reserved = (_decimal("available", balance.get("available")), _decimal("total", balance.get("total")), _decimal("reserved", balance.get("reserved", "0")))
    if available is None or total is None or reserved is None or available + reserved > total:
        return AccountSyncResult(previous, previous is not None, ("balance_unavailable_or_inconsistent",))

    deduped: dict[str, tuple[Mapping[str, Any], ...]] = {}
    field_map = {"positions": ("position_id", "token_id", "ticker"), "orders": ("order_id", "id"), "fills": ("fill_id", "trade_id", "id"), "settlements": ("settlement_id", "id", "ticker")}
    for label, fields in field_map.items():
        rows, error = _dedupe(label, flat[label], id_fields=fields)
        if error:
            issues.append(error)
        else:
            deduped[label] = rows or ()
    if issues:
        return AccountSyncResult(previous, previous is not None, tuple(issues))

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
        available_cash=available, total_cash=total, reserved_cash=reserved, positions=deduped["positions"],
        orders=deduped["orders"], fills=deduped["fills"], settlements=deduped["settlements"],
        external_activity_ids=tuple(sorted(set(external))), divergence=bool(external),
    )
    return AccountSyncResult(snapshot, False, ())
