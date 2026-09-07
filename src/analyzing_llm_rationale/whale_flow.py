"""Foresea Whale & Smart Money Trade Flow Intelligence.

Aggregates, filters, and analyzes institutional-grade block trades (> $500)
across Polymarket and Kalshi to detect smart money positioning and net sentiment.

Metrics:
- Total Whale Volume (USD)
- Net Bullish Flow vs. Bearish Flow
- Whale Sentiment Index (0% = Extreme Bearish, 100% = Extreme Bullish)
- Real-time large trade feed
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("foresea-whale-flow")


def _first(mapping: Dict[str, Any], *keys: str) -> Any:
    """First key present and not blank."""
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _platform_of(t: Dict[str, Any]) -> str:
    """The venue, inferred from the tape when the row does not carry it.

    fetch_recent_trades returns each venue's own rows unchanged, and neither
    carries a `platform`. Every print therefore read "Venue".
    """
    stated = t.get("platform")
    if stated:
        return str(stated)
    if "taker_side" in t or "count_fp" in t:
        return "Kalshi"
    if "conditionId" in t or "asset" in t or "proxyWallet" in t:
        return "Polymarket"
    return "Venue"


def _ticker_of(t: Dict[str, Any]) -> str:
    """Kalshi publishes `ticker`; Polymarket publishes `slug` and `asset`."""
    return str(_first(t, "ticker", "market", "token_id", "slug", "asset", "conditionId") or "")


def _title_of(t: Dict[str, Any]) -> str:
    """Polymarket publishes `title`. Kalshi's trade tape carries no title, so
    the ticker names the market rather than the word "Market"."""
    return str(
        _first(t, "market_title", "question", "title", "eventSlug", "ticker") or "Market"
    )


def _price_size_side(t: Dict[str, Any]) -> Tuple[float, float, str]:
    """Price, contracts and position side, in each venue's own vocabulary.

    Kalshi's tape has yes_price_dollars / no_price_dollars and count_fp, all
    strings; it has no `price` or `size`. The old parser looked only for
    price/size, so every Kalshi row priced at the 0.50 default with a size of
    0 -- a notional of 0, which no min_notional can clear. Kalshi trades could
    not appear in this endpoint at all.

    Polymarket's tape has price and size, which did resolve, but its `side` is
    BUY/SELL and the position is in `outcome`. Reading BUY as bullish counted
    a buy of NO as YES.
    """
    taker = str(t.get("taker_side") or "").strip().lower()
    if taker in ("yes", "no"):
        side = taker.upper()
        raw_price = t.get("yes_price_dollars") if side == "YES" else t.get("no_price_dollars")
        price = _as_float(raw_price, _as_float(t.get("price"), 0.50))
        return price, _as_float(_first(t, "count_fp", "size", "quantity", "count"), 0.0), side

    price = _as_float(_first(t, "price", "yes_price"), 0.50)
    size = _as_float(_first(t, "size", "quantity", "count", "count_fp"), 0.0)

    outcome = str(t.get("outcome") or "").strip().upper()
    action = str(t.get("side") or t.get("action") or "YES").upper()
    if outcome in ("YES", "NO"):
        # Selling YES is the same exposure as buying NO.
        selling = "SELL" in action
        side = outcome if not selling else ("NO" if outcome == "YES" else "YES")
        return price, size, side

    return price, size, ("YES" if ("YES" in action or "BUY" in action) else "NO")


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def analyze_whale_trades(
    trades: List[Dict[str, Any]],
    min_notional_usd: float = 250.0,
    limit: int = 50,
) -> Dict[str, Any]:
    """Filter and calculate smart-money flow metrics from a trade tape."""
    whale_prints: List[Dict[str, Any]] = []
    total_yes_usd = 0.0
    total_no_usd = 0.0

    for t in trades:
        price, size, side = _price_size_side(t)

        notional_usd = round(price * size, 2)
        if notional_usd >= min_notional_usd:
            if side == "YES":
                total_yes_usd += notional_usd
                clean_side = "YES"
            else:
                total_no_usd += notional_usd
                clean_side = "NO"

            whale_prints.append({
                "platform": _platform_of(t),
                "ticker": _ticker_of(t),
                "market_title": _title_of(t),
                "side": clean_side,
                "price": round(price, 2),
                "size": int(size),
                "notional_usd": notional_usd,
                "timestamp": t.get("timestamp") or t.get("created_time") or datetime.now(timezone.utc).isoformat(),
            })

    total_volume_usd = round(total_yes_usd + total_no_usd, 2)
    sentiment_index = round((total_yes_usd / total_volume_usd) * 100, 1) if total_volume_usd > 0 else 50.0

    if sentiment_index >= 65.0:
        sentiment_label = "BULLISH ACCUMULATION"
    elif sentiment_index <= 35.0:
        sentiment_label = "BEARISH DISTRIBUTION"
    else:
        sentiment_label = "NEUTRAL / BALANCED"

    # Sort descending by notional
    whale_prints.sort(key=lambda x: x["notional_usd"], reverse=True)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "min_notional_usd": min_notional_usd,
        "n_whale_prints": len(whale_prints[:limit]),
        "total_whale_volume_usd": total_volume_usd,
        "whale_yes_volume_usd": round(total_yes_usd, 2),
        "whale_no_volume_usd": round(total_no_usd, 2),
        "whale_sentiment_index_pct": sentiment_index,
        "whale_sentiment_label": sentiment_label,
        "top_prints": whale_prints[:limit],
    }


def fetch_live_whale_flow(min_notional_usd: float = 250.0, limit: int = 50) -> Dict[str, Any]:
    """Fetch live trades from market_data and extract smart money flow."""
    try:
        from analyzing_llm_rationale import market_data
        trades: List[Dict[str, Any]] = []

        # Tag the venue here rather than inferring it downstream: this is the
        # only place that knows for certain, and neither tape carries it.
        for venue in ("Polymarket", "Kalshi"):
            rows = market_data.fetch_recent_trades(venue.lower(), limit=40) or []
            trades.extend({**row, "platform": venue} for row in rows if isinstance(row, dict))

        return analyze_whale_trades(trades, min_notional_usd=min_notional_usd, limit=limit)
    except Exception as exc:
        logger.warning("Failed to fetch live whale flow: %s", exc)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "min_notional_usd": min_notional_usd,
            "n_whale_prints": 0,
            "total_whale_volume_usd": 0.0,
            "whale_yes_volume_usd": 0.0,
            "whale_no_volume_usd": 0.0,
            "whale_sentiment_index_pct": 50.0,
            "whale_sentiment_label": "NEUTRAL / BALANCED",
            "top_prints": [],
        }
