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
from typing import Any, Dict, List

logger = logging.getLogger("foresea-whale-flow")


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
        price = float(t.get("price") or t.get("yes_price") or 0.50)
        size = float(t.get("size") or t.get("quantity") or t.get("count") or 0)
        side = str(t.get("side") or t.get("action") or "YES").upper()

        notional_usd = round(price * size, 2)
        if notional_usd >= min_notional_usd:
            if "YES" in side or "BUY" in side:
                total_yes_usd += notional_usd
                clean_side = "YES"
            else:
                total_no_usd += notional_usd
                clean_side = "NO"

            whale_prints.append({
                "platform": t.get("platform", "Venue"),
                "ticker": t.get("ticker") or t.get("market") or t.get("token_id", ""),
                "market_title": t.get("market_title") or t.get("question", "Market"),
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

        # Fetch sample of Polymarket trades
        poly_trades = market_data.fetch_recent_trades("polymarket", limit=40)
        trades.extend(poly_trades or [])

        # Fetch sample of Kalshi trades
        kalshi_trades = market_data.fetch_recent_trades("kalshi", limit=40)
        trades.extend(kalshi_trades or [])

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
