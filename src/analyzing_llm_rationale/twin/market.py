"""Normalize venue market data into immutable executable-twin records.

This module deliberately accepts raw venue payloads rather than calling either
venue. Fetching, retries, and pagination stay outside this deterministic
validation boundary. A malformed or partial payload is a PASS, never a best
effort tradable instrument.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, Mapping, Optional, Sequence

from .models import (
    Completeness,
    Instrument,
    MarketSnapshot,
    RejectionReason,
    SchemaValidationError,
    canonical_instrument_id,
)


@dataclass(frozen=True)
class MarketAssessment:
    """One exact venue market plus explicit reasons it cannot be traded."""

    instrument: Optional[Instrument]
    snapshot: Optional[MarketSnapshot]
    reasons: tuple[RejectionReason, ...]

    @property
    def eligible(self) -> bool:
        return self.instrument is not None and self.snapshot is not None and not self.reasons


def _text(raw: Any) -> str:
    return str(raw or "").strip()


def _identifier(raw: Any) -> Optional[str]:
    value = _text(raw)
    return value or None


def _decimal(raw: Any) -> Optional[Decimal]:
    if raw in (None, ""):
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return value if value.is_finite() and value >= 0 else None


def _timestamp(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else None
    value = _text(raw)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else None


def _list(raw: Any) -> list[Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return list(raw) if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)) else []


def _hash(*values: str) -> str:
    return sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def _book_levels(raw: Any) -> tuple[Optional[Decimal], Decimal]:
    """Return best price and visible size from CLOB-shaped levels."""
    levels = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else []
    best: Optional[Decimal] = None
    size = Decimal("0")
    for level in levels:
        if isinstance(level, Mapping):
            price, quantity = _decimal(level.get("price")), _decimal(level.get("size") or level.get("quantity"))
        elif isinstance(level, Sequence) and len(level) >= 2:
            price, quantity = _decimal(level[0]), _decimal(level[1])
        else:
            continue
        if price is None or quantity is None:
            continue
        best = price if best is None else max(best, price)
        size += quantity
    return best, size


def _reason_set(*reasons: RejectionReason) -> tuple[RejectionReason, ...]:
    return tuple(dict.fromkeys(reasons))


def normalize_market(
    venue: str,
    market: Mapping[str, Any],
    *,
    received_at: datetime,
    sequence: int,
    orderbooks: Optional[Mapping[str, Mapping[str, Any]]] = None,
    allowed_categories: Optional[set[str]] = None,
    min_horizon_seconds: int = 0,
    max_horizon_seconds: Optional[int] = None,
    stale_after_seconds: int = 30,
) -> MarketAssessment:
    """Create precise records only when market semantics and depth are known."""
    if received_at.tzinfo is None:
        raise SchemaValidationError("received_at must be timezone-aware")
    venue = _text(venue).lower()
    if venue not in {"kalshi", "polymarket"}:
        raise SchemaValidationError("unsupported venue")
    if sequence < 1 or stale_after_seconds < 1:
        raise SchemaValidationError("sequence and stale_after_seconds must be positive")
    if not isinstance(market, Mapping):
        return MarketAssessment(None, None, (RejectionReason.PASS_INCOMPLETE_DATA,))

    if venue == "kalshi":
        return _normalize_kalshi(
            market, received_at=received_at, sequence=sequence, allowed_categories=allowed_categories,
            min_horizon_seconds=min_horizon_seconds, max_horizon_seconds=max_horizon_seconds,
            stale_after_seconds=stale_after_seconds,
        )
    return _normalize_polymarket(
        market, received_at=received_at, sequence=sequence, orderbooks=orderbooks or {},
        allowed_categories=allowed_categories, min_horizon_seconds=min_horizon_seconds,
        max_horizon_seconds=max_horizon_seconds, stale_after_seconds=stale_after_seconds,
    )


def _common_reasons(
    *, status: str, category: str, settlement: str, close_at: Optional[datetime], received_at: datetime,
    allowed_categories: Optional[set[str]], min_horizon_seconds: int, max_horizon_seconds: Optional[int],
) -> tuple[RejectionReason, ...]:
    reasons: list[RejectionReason] = []
    if status not in {"active", "open", "live", "trading"}:
        reasons.append(RejectionReason.PASS_UNSUPPORTED_INSTRUMENT)
    if not settlement:
        reasons.append(RejectionReason.PASS_INCOMPLETE_DATA)
    if allowed_categories is not None and category.lower() not in {item.lower() for item in allowed_categories}:
        reasons.append(RejectionReason.PASS_UNSUPPORTED_INSTRUMENT)
    if close_at is None:
        reasons.append(RejectionReason.PASS_INCOMPLETE_DATA)
    else:
        horizon = (close_at - received_at).total_seconds()
        if horizon < min_horizon_seconds or (max_horizon_seconds is not None and horizon > max_horizon_seconds):
            reasons.append(RejectionReason.PASS_UNSUPPORTED_INSTRUMENT)
    return _reason_set(*reasons)


def _normalize_kalshi(
    market: Mapping[str, Any], *, received_at: datetime, sequence: int, allowed_categories: Optional[set[str]],
    min_horizon_seconds: int, max_horizon_seconds: Optional[int], stale_after_seconds: int,
) -> MarketAssessment:
    ticker = _identifier(market.get("ticker"))
    close_at = _timestamp(market.get("close_time") or market.get("close_time_iso") or market.get("expiration_time"))
    venue_at = _timestamp(market.get("updated_time") or market.get("last_updated_ts")) or received_at
    category = _text(market.get("category") or "other")
    settlement = _text(market.get("rules_primary") or market.get("settlement_value") or market.get("subtitle"))
    reasons = list(_common_reasons(
        status=_text(market.get("status")).lower(), category=category, settlement=settlement, close_at=close_at,
        received_at=received_at, allowed_categories=allowed_categories, min_horizon_seconds=min_horizon_seconds,
        max_horizon_seconds=max_horizon_seconds,
    ))
    tick = _decimal(market.get("price_level_structure") or market.get("tick_size") or "0.01")
    minimum = _decimal(market.get("min_contracts") or market.get("min_quantity") or "1")
    yes_bid, yes_ask = _decimal(market.get("yes_bid_dollars") or market.get("yes_bid")), _decimal(market.get("yes_ask_dollars") or market.get("yes_ask"))
    no_bid, no_ask = _decimal(market.get("no_bid_dollars") or market.get("no_bid")), _decimal(market.get("no_ask_dollars") or market.get("no_ask"))
    if not ticker or tick is None or minimum is None or tick == 0 or minimum == 0:
        reasons.append(RejectionReason.PASS_INCOMPLETE_DATA)
    if yes_ask is None or no_ask is None or yes_ask == 0 or no_ask == 0:
        reasons.append(RejectionReason.PASS_INCOMPLETE_DATA)
    if venue_at > received_at:
        reasons.append(RejectionReason.PASS_STALE_DATA)
    if reasons:
        return MarketAssessment(None, None, _reason_set(*reasons))
    assert ticker and close_at and tick is not None and minimum is not None
    instrument = Instrument(
        id=canonical_instrument_id(venue="kalshi", environment="live", venue_instrument_id=ticker), venue="kalshi",
        environment="live", venue_instrument_id=ticker, condition_id=None, yes_token_id=None, no_token_id=None,
        settlement_spec_hash=_hash("kalshi", ticker, settlement), category=category or "other", event_id=ticker,
        cluster_id=ticker, tick_size=tick, min_quantity=minimum, fee_version=_text(market.get("fee_version") or "kalshi-default"),
        capability_version="kalshi-v2-limit", status="open", close_at=close_at, resolution_at=close_at, created_at=received_at,
    )
    snapshot = MarketSnapshot(
        id=f"snapshot-{_hash(instrument.id, str(sequence), venue_at.isoformat())[:32]}", instrument_id=instrument.id,
        venue_at=venue_at, received_at=received_at, sequence=sequence, source="kalshi-rest", complete=Completeness.COMPLETE,
        stale_after_seconds=stale_after_seconds, yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask,
        fee_version=instrument.fee_version, created_at=received_at,
    )
    return MarketAssessment(instrument, snapshot, ())


def _normalize_polymarket(
    market: Mapping[str, Any], *, received_at: datetime, sequence: int, orderbooks: Mapping[str, Mapping[str, Any]],
    allowed_categories: Optional[set[str]], min_horizon_seconds: int, max_horizon_seconds: Optional[int],
    stale_after_seconds: int,
) -> MarketAssessment:
    condition_id, market_id = _identifier(market.get("conditionId") or market.get("condition_id")), _identifier(market.get("id"))
    outcomes, tokens = [_text(item).lower() for item in _list(market.get("outcomes"))], [_identifier(item) for item in _list(market.get("clobTokenIds"))]
    close_at = _timestamp(market.get("endDateIso") or market.get("endDate"))
    venue_at = _timestamp(market.get("updatedAt") or market.get("updated_at")) or received_at
    category = _text(market.get("category") or "other")
    settlement = _text(market.get("rules") or market.get("description") or market.get("resolutionSource"))
    status = "open" if bool(market.get("active", True)) and not bool(market.get("closed", False)) and bool(market.get("acceptingOrders", True)) else "closed"
    reasons = list(_common_reasons(
        status=status, category=category, settlement=settlement, close_at=close_at, received_at=received_at,
        allowed_categories=allowed_categories, min_horizon_seconds=min_horizon_seconds, max_horizon_seconds=max_horizon_seconds,
    ))
    if outcomes != ["yes", "no"] or len(tokens) != 2 or any(token is None for token in tokens):
        reasons.append(RejectionReason.PASS_UNSUPPORTED_INSTRUMENT)
    tick, minimum = _decimal(market.get("minimum_tick_size") or market.get("tick_size")), _decimal(market.get("minimum_order_size") or market.get("min_order_size"))
    if not condition_id or not market_id or tick is None or minimum is None or tick == 0 or minimum == 0:
        reasons.append(RejectionReason.PASS_INCOMPLETE_DATA)
    if venue_at > received_at:
        reasons.append(RejectionReason.PASS_STALE_DATA)
    if reasons:
        return MarketAssessment(None, None, _reason_set(*reasons))
    yes_token, no_token = str(tokens[0]), str(tokens[1])
    yes_book, no_book = orderbooks.get(yes_token), orderbooks.get(no_token)
    if not isinstance(yes_book, Mapping) or not isinstance(no_book, Mapping):
        return MarketAssessment(None, None, (RejectionReason.PASS_INCOMPLETE_DATA,))
    yes_bid, yes_size = _book_levels(yes_book.get("bids"))
    yes_ask, yes_ask_size = _book_levels(yes_book.get("asks"))
    no_bid, no_size = _book_levels(no_book.get("bids"))
    no_ask, no_ask_size = _book_levels(no_book.get("asks"))
    if min(yes_size, yes_ask_size, no_size, no_ask_size) <= 0 or yes_ask is None or no_ask is None:
        return MarketAssessment(None, None, (RejectionReason.PASS_INCOMPLETE_DATA,))
    assert condition_id and market_id and close_at and tick is not None and minimum is not None
    instrument = Instrument(
        id=canonical_instrument_id(venue="polymarket", environment="live", condition_id=condition_id, venue_instrument_id=market_id),
        venue="polymarket", environment="live", venue_instrument_id=market_id, condition_id=condition_id,
        yes_token_id=yes_token, no_token_id=no_token, settlement_spec_hash=_hash("polymarket", condition_id, market_id, settlement),
        category=category or "other", event_id=condition_id, cluster_id=condition_id, tick_size=tick, min_quantity=minimum,
        fee_version=_text(market.get("fee_version") or "polymarket-default"), capability_version="polymarket-clob-limit",
        status="open", close_at=close_at, resolution_at=close_at, created_at=received_at,
    )
    snapshot = MarketSnapshot(
        id=f"snapshot-{_hash(instrument.id, str(sequence), venue_at.isoformat())[:32]}", instrument_id=instrument.id,
        venue_at=venue_at, received_at=received_at, sequence=sequence, source="polymarket-clob", complete=Completeness.COMPLETE,
        stale_after_seconds=stale_after_seconds, yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask,
        fee_version=instrument.fee_version, created_at=received_at,
    )
    return MarketAssessment(instrument, snapshot, ())
