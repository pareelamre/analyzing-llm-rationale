"""Native account-read adapters that refuse undocumented capital semantics.

The generic reconciler accepts canonical rows only after every collection is
complete.  These adapters make the boundary with Kalshi and Polymarket
explicit: a displayed portfolio, allowance, mark value, or a short page never
becomes an account cash ledger or a liquidation estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .models import SchemaValidationError
from .reconcile import (
    AccountReadError,
    PageFetcher,
    RowNormalizer,
    VenueReader,
    cursor_page_fetcher,
    offset_page_fetcher,
)


@dataclass(frozen=True)
class VenueAccountCapability:
    """Whether documented provider data can establish a complete cash ledger."""

    venue: str
    cash_ledger_supported: bool
    settlement_identity_supported: bool
    blockers: tuple[str, ...]


KALSHI_ACCOUNT_CAPABILITY = VenueAccountCapability(
    venue="kalshi",
    cash_ledger_supported=False,
    settlement_identity_supported=False,
    blockers=(
        "kalshi_balance_cash_breakdown_unavailable",
        "kalshi_settlement_immutable_id_unavailable",
        "kalshi_position_liquidation_unavailable",
    ),
)

POLYMARKET_ACCOUNT_CAPABILITY = VenueAccountCapability(
    venue="polymarket",
    cash_ledger_supported=False,
    settlement_identity_supported=False,
    blockers=(
        "polymarket_cash_authority_unavailable",
        "polymarket_settlement_authority_unavailable",
        "polymarket_position_liquidation_unavailable",
    ),
)


def account_capability(venue: str) -> VenueAccountCapability:
    """Return the documented capital-authority status for a supported venue."""
    normalized = str(venue).strip().lower()
    if normalized == "kalshi":
        return KALSHI_ACCOUNT_CAPABILITY
    if normalized == "polymarket":
        return POLYMARKET_ACCOUNT_CAPABILITY
    raise SchemaValidationError("unsupported venue account adapter")


def _unavailable(reason: str) -> PageFetcher:
    def fetch(_cursor: Optional[str]) -> Mapping[str, Any]:
        raise AccountReadError(reason)

    return fetch


def _expect_account_object(result: Mapping[str, Any], operation: str) -> Mapping[str, Any]:
    data = result.get("data") if isinstance(result, Mapping) else None
    if not isinstance(data, Mapping):
        raise AccountReadError(f"{operation}_response_malformed")
    return data


def kalshi_balance_page_fetcher(
    *, reader: VenueReader, creds: Mapping[str, Any], parameters: Optional[Mapping[str, Any]] = None
) -> PageFetcher:
    """Read Kalshi's balance without confusing portfolio value for cash.

    Kalshi documents ``balance`` as available cents and ``portfolio_value`` as
    a value that includes positions. It does not supply the total, reserved,
    and settled cash fields required by ``AccountSnapshot``.  Keep the prior
    generation until a documented, account-scoped ledger source is available.
    """
    base_parameters = dict(parameters or {})

    def fetch(cursor: Optional[str]) -> Mapping[str, Any]:
        if cursor is not None:
            raise AccountReadError("kalshi_balance_not_paginated")
        result = reader("kalshi", "balance", base_parameters, access="account", creds=dict(creds))
        data = _expect_account_object(result, "kalshi_balance")
        if data.get("balance") in (None, ""):
            raise AccountReadError("kalshi_balance_unavailable")
        # Intentionally do not map portfolio_value to total cash or invent
        # reserved/settled cash from an available-cash response.
        raise AccountReadError("kalshi_balance_cash_breakdown_unavailable")

    return fetch


def _kalshi_position(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep only directly observed Kalshi position fields.

    ``market_exposure_dollars`` is not a conservative executable bid.  The
    missing liquidation/basis fields are therefore left absent for the account
    reconciler to reject rather than estimated from display data.
    """
    return {
        "position_id": row.get("ticker"),
        "ticker": row.get("ticker"),
        "quantity": row.get("position_fp"),
        "fees_paid": row.get("fees_paid_dollars"),
    }


def _kalshi_order(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "order_id": row.get("order_id"),
        "client_order_id": row.get("client_order_id"),
    }


def _kalshi_fill(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "fill_id": row.get("fill_id"),
        "trade_id": row.get("trade_id"),
        "order_id": row.get("order_id"),
        "fee": row.get("fee_cost"),
        "client_order_id": row.get("client_order_id"),
    }


def _polymarket_position(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize known position identity/quantity/basis without using marks."""
    return {
        "position_id": row.get("asset"),
        "token_id": row.get("asset"),
        "quantity": row.get("size"),
        # The provider documents initialValue as fee-exclusive basis. Do not
        # use currentValue: it is a mark, not executable liquidation value.
        "basis": row.get("initialValue"),
    }


def _polymarket_order(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "order_id": row.get("id") or row.get("order_id"),
        "client_order_id": row.get("client_order_id"),
    }


def _polymarket_fill(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "fill_id": row.get("id") or row.get("trade_id"),
        "order_id": row.get("order_id") or row.get("taker_order_id"),
        "fee": row.get("fee") or row.get("fee_usdc"),
        "client_order_id": row.get("client_order_id"),
    }


def _cursor_collection(
    venue: str,
    operation: str,
    *,
    item_key: str,
    reader: VenueReader,
    creds: Mapping[str, Any],
    parameters: Mapping[str, Any],
    normalizer: RowNormalizer,
    cursor_parameter: str = "cursor",
) -> PageFetcher:
    return cursor_page_fetcher(
        venue,
        operation,
        item_key=item_key,
        reader=reader,
        creds=creds,
        cursor_parameter=cursor_parameter,
        parameters=parameters,
        item_normalizer=normalizer,
    )


def complete_account_fetchers(
    venue: str,
    *,
    reader: VenueReader,
    creds: Mapping[str, Any],
    page_limit: int = 1_000,
) -> Mapping[str, PageFetcher]:
    """Build all required fetchers with their current safe provider semantics.

    The mapping is deliberately usable by ``synchronize_complete_account``:
    it captures collection-specific failures and retains an old durable
    snapshot. It must not be used as proof that either venue supports live
    capital authority while its documented fields remain incomplete.
    """
    normalized = str(venue).strip().lower()
    if page_limit < 1:
        raise SchemaValidationError("venue account page limit must be positive")
    if normalized == "kalshi":
        params = {"limit": min(page_limit, 1_000)}
        return {
            "balances": kalshi_balance_page_fetcher(reader=reader, creds=creds),
            "positions": _cursor_collection(
                "kalshi", "positions", item_key="market_positions", reader=reader, creds=creds,
                parameters=params, normalizer=_kalshi_position,
            ),
            "orders": _cursor_collection(
                "kalshi", "orders", item_key="orders", reader=reader, creds=creds,
                parameters=params, normalizer=_kalshi_order,
            ),
            "fills": _cursor_collection(
                "kalshi", "fills", item_key="fills", reader=reader, creds=creds,
                parameters=params, normalizer=_kalshi_fill,
            ),
            # The documented settlement rows have no immutable settlement ID.
            "settlements": _unavailable("kalshi_settlement_immutable_id_unavailable"),
        }
    if normalized == "polymarket":
        return {
            # CLOB allowance and Data API position value are not a complete
            # spendable-cash ledger.
            "balances": _unavailable("polymarket_cash_authority_unavailable"),
            "positions": offset_page_fetcher(
                "polymarket", "positions", reader=reader, creds=creds,
                limit=min(page_limit, 500), item_normalizer=_polymarket_position,
            ),
            "orders": _cursor_collection(
                "polymarket", "orders", item_key="orders", reader=reader, creds=creds,
                parameters={}, normalizer=_polymarket_order, cursor_parameter="next_cursor",
            ),
            "fills": _cursor_collection(
                "polymarket", "fills", item_key="trades", reader=reader, creds=creds,
                parameters={}, normalizer=_polymarket_fill, cursor_parameter="next_cursor",
            ),
            "settlements": _unavailable("polymarket_settlement_authority_unavailable"),
        }
    raise SchemaValidationError("unsupported venue account adapter")
