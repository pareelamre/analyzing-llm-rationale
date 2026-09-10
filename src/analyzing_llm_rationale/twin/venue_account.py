"""Native account-read adapters that refuse undocumented capital semantics.

The generic reconciler accepts canonical rows only after every collection is
complete.  These adapters make the boundary with Kalshi and Polymarket
explicit: a displayed portfolio, allowance, mark value, or a short page never
becomes an account cash ledger or a liquidation estimate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, Callable, Mapping, Optional

from .. import market_data
from .models import SchemaValidationError
from .reconcile import (
    AccountReadError,
    PageFetcher,
    RowNormalizer,
    VenueReader,
    cursor_page_fetcher,
    offset_page_fetcher,
    read_complete_collection,
)


@dataclass(frozen=True)
class VenueAccountCapability:
    """Whether documented provider data can establish a complete cash ledger."""

    venue: str
    cash_ledger_supported: bool
    settlement_identity_supported: bool
    blockers: tuple[str, ...]


GenerationFence = Callable[[], str]
AccountingSnapshotReader = Callable[[], Mapping[str, Any]]
LiquidationReader = Callable[[str, str, Decimal, str], Decimal]


@dataclass(frozen=True)
class VenueAccountReadPlan:
    """Runtime adapter set plus a before/after whole-account read fence."""

    venue: str
    fetchers: Mapping[str, PageFetcher]
    generation_fence: GenerationFence


KALSHI_ACCOUNT_CAPABILITY = VenueAccountCapability(
    venue="kalshi",
    cash_ledger_supported=True,
    settlement_identity_supported=True,
    blockers=(),
)

POLYMARKET_ACCOUNT_CAPABILITY = VenueAccountCapability(
    venue="polymarket",
    cash_ledger_supported=True,
    settlement_identity_supported=True,
    blockers=("fee_enabled_trade_without_actual_fee",),
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


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def _finite_decimal(value: Any, label: str, *, signed: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise AccountReadError(f"{label}_unavailable") from exc
    if not result.is_finite() or (not signed and result < 0):
        raise AccountReadError(f"{label}_unavailable")
    return result


def conservative_bid_liquidation(orderbook: Mapping[str, Any], quantity: Decimal) -> Decimal:
    """Walk executable bids and value any depth shortfall at zero."""
    remaining = abs(_finite_decimal(quantity, "position_quantity", signed=True))
    levels = orderbook.get("bids")
    if not isinstance(levels, list):
        raise AccountReadError("position_orderbook_unavailable")
    parsed: list[tuple[Decimal, Decimal]] = []
    for level in levels:
        if isinstance(level, Mapping):
            price, size = level.get("price"), level.get("size")
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price, size = level[0], level[1]
        else:
            raise AccountReadError("position_orderbook_malformed")
        parsed.append((_finite_decimal(price, "bid_price"), _finite_decimal(size, "bid_size")))
    value = Decimal("0")
    for price, size in sorted(parsed, reverse=True):
        if price > 1:
            price /= Decimal("100")
        take = min(remaining, size)
        value += take * price
        remaining -= take
        if remaining == 0:
            break
    return value


def _default_liquidation_reader(venue: str, instrument_id: str, quantity: Decimal, side: str) -> Decimal:
    if venue == "polymarket":
        return conservative_bid_liquidation(
            market_data.fetch_polymarket_orderbook(instrument_id), quantity
        )
    book = market_data.fetch_kalshi_orderbook(instrument_id)
    levels = book.get(f"{side}_dollars") or book.get(side)
    return conservative_bid_liquidation({"bids": levels}, quantity)


def kalshi_balance_page_fetcher(
    *, reader: VenueReader, creds: Mapping[str, Any], parameters: Optional[Mapping[str, Any]] = None
) -> PageFetcher:
    """Read documented available cash; reservations are derived from open orders."""
    base_parameters = dict(parameters or {})

    def fetch(cursor: Optional[str]) -> Mapping[str, Any]:
        if cursor is not None:
            raise AccountReadError("kalshi_balance_not_paginated")
        result = reader("kalshi", "balance", base_parameters, access="account", creds=dict(creds))
        data = _expect_account_object(result, "kalshi_balance")
        raw_balance = data.get("balance_dollars")
        if raw_balance in (None, "") and data.get("balance") not in (None, ""):
            raw_balance = _finite_decimal(data["balance"], "kalshi_balance") / Decimal("100")
        if raw_balance in (None, ""):
            raise AccountReadError("kalshi_balance_unavailable")
        return {
            "items": [{
                "cash_semantics": "kalshi_available",
                "cash": str(raw_balance),
                "portfolio_value": data.get("portfolio_value"),
                "updated_ts": data.get("updated_ts"),
            }],
            "cursor": None,
            "generation_token": None,
        }

    return fetch


def _kalshi_position(row: Mapping[str, Any], liquidation_reader: LiquidationReader) -> Mapping[str, Any]:
    """Normalize fixed-point position basis and executable liquidation depth."""
    quantity = _finite_decimal(row.get("position_fp"), "kalshi_position", signed=True)
    ticker = str(row.get("ticker") or "").strip()
    if not ticker:
        raise AccountReadError("kalshi_position_identity_unavailable")
    side = "yes" if quantity >= 0 else "no"
    return {
        "position_id": ticker,
        "ticker": ticker,
        "quantity": str(quantity),
        "basis": row.get("market_exposure_dollars"),
        "liquidation_value": str(liquidation_reader("kalshi", ticker, quantity, side)),
        "fees_paid": row.get("fees_paid_dollars"),
    }


def _kalshi_order(row: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = {
        "order_id": row.get("order_id"),
        "client_order_id": row.get("client_order_id"),
    }
    # Orders are mutable. Retain the observed lifecycle and economics so the
    # complete-generation deduper can detect an order changing mid-read.
    for field in (
        "ticker", "status", "side", "action", "type", "initial_count",
        "initial_count_fp", "fill_count", "fill_count_fp", "remaining_count",
        "remaining_count_fp", "yes_price", "yes_price_dollars", "no_price",
        "no_price_dollars", "expiration_time",
    ):
        if field in row:
            normalized[field] = row[field]
    return normalized


def _kalshi_fill(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "fill_id": row.get("fill_id"),
        "trade_id": row.get("trade_id"),
        "order_id": row.get("order_id"),
        "fee": row.get("fee_cost") if "fee_cost" in row else row.get("fee_cost_dollars"),
        "client_order_id": row.get("client_order_id"),
    }


def _kalshi_settlement(row: Mapping[str, Any]) -> Mapping[str, Any]:
    identity_fields = {
        key: row.get(key) for key in (
            "ticker", "event_ticker", "market_result", "yes_count_fp",
            "yes_total_cost_dollars", "no_count_fp", "no_total_cost_dollars",
            "revenue", "settled_time", "fee_cost", "value", "exchange_index",
        )
    }
    if not identity_fields["ticker"] or not identity_fields["settled_time"]:
        raise AccountReadError("kalshi_settlement_identity_unavailable")
    # A market ticker identifies the one settlement ledger entry for this
    # account scope. Keep the ID stable when Kalshi corrects payout fields in a
    # later generation so replacement cannot be mistaken for a second payout.
    normalized = dict(identity_fields)
    normalized.update({
        "settlement_id": f"kalshi:{identity_fields['ticker']}",
        "status": "settled",
        "amount": _finite_decimal(row.get("revenue"), "kalshi_settlement_revenue") / Decimal("100"),
        "fee": row.get("fee_cost"),
    })
    return normalized


def _polymarket_position(row: Mapping[str, Any], liquidation_reader: LiquidationReader) -> Mapping[str, Any]:
    """Normalize accounting holdings without treating ``curPrice`` as executable."""
    asset = str(row.get("asset") or "").strip()
    quantity = _finite_decimal(row.get("size"), "polymarket_position")
    if not asset:
        raise AccountReadError("polymarket_position_identity_unavailable")
    basis = row.get("initialValue")
    if basis in (None, ""):
        average = _finite_decimal(row.get("avgPrice"), "polymarket_position_price")
        basis = quantity * average
    return {
        "position_id": asset,
        "token_id": asset,
        "condition_id": row.get("conditionId"),
        "quantity": str(quantity),
        "basis": str(basis),
        "liquidation_value": str(
            liquidation_reader("polymarket", asset, quantity, "token")
        ),
        "valuation_time": row.get("valuationTime"),
    }


def _polymarket_order(row: Mapping[str, Any]) -> Mapping[str, Any]:
    result = {
        "order_id": row.get("id") or row.get("order_id"),
        "client_order_id": row.get("client_order_id"),
    }
    for field in (
        "status", "side", "price", "original_size", "size_matched",
        "asset_id", "market", "created_at",
    ):
        if field in row:
            result[field] = row[field]
    return result


def _polymarket_fill(row: Mapping[str, Any]) -> Mapping[str, Any]:
    fee = row.get("fee") if "fee" in row else row.get("fee_usdc")
    if fee in (None, ""):
        rate = _finite_decimal(row.get("fee_rate_bps"), "polymarket_fill_fee_rate")
        if rate != 0:
            raise AccountReadError("fee_enabled_trade_without_actual_fee")
        fee = Decimal("0")
    return {
        "fill_id": row.get("id") or row.get("trade_id"),
        "order_id": row.get("order_id") or row.get("taker_order_id"),
        "fee": fee,
        "client_order_id": row.get("client_order_id"),
    }


def _polymarket_settlement(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if str(row.get("type") or "").upper() != "REDEEM":
        raise AccountReadError("polymarket_non_redeem_settlement")
    transaction_hash = str(row.get("transactionHash") or "").strip()
    condition_id = str(row.get("conditionId") or "").strip()
    if not transaction_hash or not condition_id:
        raise AccountReadError("polymarket_settlement_identity_unavailable")
    return {
        "settlement_id": f"polymarket:{transaction_hash}:{condition_id}:{row.get('asset') or ''}",
        "transaction_hash": transaction_hash,
        "condition_id": condition_id,
        "asset": row.get("asset"),
        "amount": row.get("usdcSize"),
        "fee": "0",
        "status": "settled",
        "settled_time": row.get("timestamp"),
    }


def _snapshot_token(snapshot: Mapping[str, Any]) -> str:
    positions = snapshot.get("positions")
    equity = snapshot.get("equity")
    if not isinstance(positions, list) or not isinstance(equity, Mapping):
        raise AccountReadError("polymarket_accounting_snapshot_malformed")
    valuation_times = {
        str(row.get("valuationTime") or "").strip() for row in [*positions, equity]
    }
    if len(valuation_times) != 1 or "" in valuation_times:
        raise AccountReadError("polymarket_accounting_snapshot_generation_mixed")
    return _canonical_hash(snapshot)


def _complete_account_fingerprint(fetchers: Mapping[str, PageFetcher]) -> str:
    """Fingerprint complete collections so concurrent mutations cannot straddle a read."""
    payload = {
        name: list(read_complete_collection(name, fetchers[name]).items)
        for name in ("balances", "positions", "orders", "fills", "settlements")
    }
    return _canonical_hash(payload)


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


def _tiered_cursor_collection(
    venue: str,
    live_operation: str,
    historical_operation: str,
    *,
    item_key: str,
    reader: VenueReader,
    creds: Mapping[str, Any],
    parameters: Mapping[str, Any],
    normalizer: RowNormalizer,
) -> PageFetcher:
    """Traverse live then historical partitions without dropping either tier."""
    live = _cursor_collection(
        venue, live_operation, item_key=item_key, reader=reader, creds=creds,
        parameters=parameters, normalizer=normalizer,
    )
    historical = _cursor_collection(
        venue, historical_operation, item_key=item_key, reader=reader, creds=creds,
        parameters=parameters, normalizer=normalizer,
    )

    def fetch(cursor: Optional[str]) -> Mapping[str, Any]:
        tier, upstream = "live", None
        if cursor is not None:
            tier, separator, raw = cursor.partition(":")
            if not separator or tier not in {"live", "historical"}:
                raise AccountReadError(f"{live_operation}_tier_cursor_invalid")
            upstream = raw or None
        page = dict((live if tier == "live" else historical)(upstream))
        next_cursor = page.get("cursor")
        if tier == "live":
            page["cursor"] = f"live:{next_cursor}" if next_cursor else "historical:"
        else:
            page["cursor"] = f"historical:{next_cursor}" if next_cursor else None
        # Partition movement is the fence. Per-tier generation markers cannot
        # be compared across distinct providers.
        page["generation_token"] = None
        return page

    return fetch


def complete_account_fetchers(
    venue: str,
    *,
    reader: VenueReader,
    creds: Mapping[str, Any],
    page_limit: int = 1_000,
    accounting_snapshot_reader: Optional[AccountingSnapshotReader] = None,
    liquidation_reader: Optional[LiquidationReader] = None,
) -> Mapping[str, PageFetcher]:
    """Build complete venue fetchers; prefer the fenced read-plan API."""
    return complete_account_read_plan(
        venue, reader=reader, creds=creds, page_limit=page_limit,
        accounting_snapshot_reader=accounting_snapshot_reader,
        liquidation_reader=liquidation_reader,
    ).fetchers


def complete_account_read_plan(
    venue: str,
    *,
    reader: VenueReader,
    creds: Mapping[str, Any],
    page_limit: int = 1_000,
    accounting_snapshot_reader: Optional[AccountingSnapshotReader] = None,
    liquidation_reader: Optional[LiquidationReader] = None,
) -> VenueAccountReadPlan:
    """Build the runtime read boundary and stable before/after generation fence."""
    normalized = str(venue).strip().lower()
    if page_limit < 1:
        raise SchemaValidationError("venue account page limit must be positive")
    liquidate = liquidation_reader or _default_liquidation_reader
    if normalized == "kalshi":
        params = {"limit": min(page_limit, 1_000)}
        fetchers = {
            "balances": kalshi_balance_page_fetcher(reader=reader, creds=creds),
            "positions": _cursor_collection(
                "kalshi", "positions", item_key="market_positions", reader=reader, creds=creds,
                parameters=params, normalizer=lambda row: _kalshi_position(row, liquidate),
            ),
            "orders": _tiered_cursor_collection(
                "kalshi", "orders", "historical_orders", item_key="orders",
                reader=reader, creds=creds, parameters=params, normalizer=_kalshi_order,
            ),
            "fills": _tiered_cursor_collection(
                "kalshi", "fills", "historical_fills", item_key="fills",
                reader=reader, creds=creds, parameters=params, normalizer=_kalshi_fill,
            ),
            "settlements": _cursor_collection(
                "kalshi", "settlements", item_key="settlements", reader=reader, creds=creds,
                parameters=params, normalizer=_kalshi_settlement,
            ),
        }
        def kalshi_fence() -> str:
            return _complete_account_fingerprint(fetchers)

        return VenueAccountReadPlan("kalshi", fetchers, kalshi_fence)
    if normalized == "polymarket":
        if accounting_snapshot_reader is None:
            from ..venue_api import read_polymarket_accounting_snapshot

            def default_snapshot_reader() -> Mapping[str, Any]:
                return read_polymarket_accounting_snapshot(creds=dict(creds))

            accounting_snapshot_reader = default_snapshot_reader
        current: dict[str, Mapping[str, Any]] = {}

        def load_snapshot(*, fresh: bool = False) -> Mapping[str, Any]:
            if fresh or "snapshot" not in current:
                value = accounting_snapshot_reader()
                if not isinstance(value, Mapping):
                    raise AccountReadError("polymarket_accounting_snapshot_malformed")
                _snapshot_token(value)
                current["snapshot"] = value
            return current["snapshot"]

        def snapshot_fence() -> str:
            load_snapshot(fresh=True)
            return _complete_account_fingerprint(fetchers)

        def snapshot_balance(cursor: Optional[str]) -> Mapping[str, Any]:
            if cursor is not None:
                raise AccountReadError("polymarket_accounting_snapshot_not_paginated")
            snapshot = load_snapshot()
            equity = snapshot["equity"]
            return {
                "items": [{"cash_semantics": "polymarket_wallet", "cash": equity.get("cashBalance")}],
                "cursor": None,
                "generation_token": _snapshot_token(snapshot),
            }

        def snapshot_positions(cursor: Optional[str]) -> Mapping[str, Any]:
            if cursor is not None:
                raise AccountReadError("polymarket_accounting_snapshot_not_paginated")
            snapshot = load_snapshot()
            return {
                "items": [_polymarket_position(row, liquidate) for row in snapshot["positions"]],
                "cursor": None,
                "generation_token": _snapshot_token(snapshot),
            }

        fetchers = {
            "balances": snapshot_balance,
            "positions": snapshot_positions,
            "orders": _cursor_collection(
                "polymarket", "orders", item_key="data", reader=reader, creds=creds,
                parameters={}, normalizer=_polymarket_order, cursor_parameter="next_cursor",
            ),
            "fills": _cursor_collection(
                "polymarket", "fills", item_key="data", reader=reader, creds=creds,
                parameters={}, normalizer=_polymarket_fill, cursor_parameter="next_cursor",
            ),
            "settlements": offset_page_fetcher(
                "polymarket", "activity", reader=reader, creds=creds,
                limit=min(page_limit, 500), parameters={"type": ["REDEEM"]},
                item_normalizer=_polymarket_settlement,
            ),
        }
        return VenueAccountReadPlan("polymarket", fetchers, snapshot_fence)
    raise SchemaValidationError("unsupported venue account adapter")
