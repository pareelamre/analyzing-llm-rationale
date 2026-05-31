"""Fetch live prices from prediction-market venues (Polymarket, Kalshi).

Each fetcher returns a normalized dict compatible with the ``/predict``
``market_*`` fields, so a quote can be piped straight into an edge analysis::

    {
        "platform": "Polymarket",
        "question": "...",
        "market_url": "https://polymarket.com/market/...",
        "outcome": "Yes",
        "probability": 0.54,            # 0..1, or None when unpriced
        "outcomes": [{"label": "Yes", "probability": 0.54}, ...],
    }
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
_TIMEOUT_S = 12
_HEADERS = {"User-Agent": "foresea-market-bot/1.0"}


class MarketDataError(RuntimeError):
    """Raised when a market cannot be fetched or parsed."""


def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    import requests

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT_S)
    except Exception as exc:  # network error
        raise MarketDataError(f"Market request failed: {exc}") from exc
    if resp.status_code == 404:
        raise MarketDataError("Market not found.")
    if resp.status_code != 200:
        raise MarketDataError(f"Market provider returned status {resp.status_code}.")
    try:
        return resp.json()
    except ValueError as exc:
        raise MarketDataError("Market provider returned invalid JSON.") from exc


def _as_list(value: Any) -> List[Any]:
    """Polymarket returns outcomes/prices as JSON-encoded strings or lists."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _primary_outcome(options: List[Dict[str, Any]]) -> tuple[str, Optional[float]]:
    """Prefer a 'Yes' outcome, else the highest-probability option."""
    for opt in options:
        if str(opt.get("label", "")).strip().lower() == "yes":
            return opt["label"], opt.get("probability")
    priced = [o for o in options if o.get("probability") is not None]
    if priced:
        top = max(priced, key=lambda o: o["probability"])
        return top["label"], top["probability"]
    return (options[0]["label"], options[0].get("probability")) if options else ("", None)


def fetch_polymarket(slug: Optional[str] = None, market_id: Optional[str] = None) -> Dict[str, Any]:
    """Fetch a Polymarket market by slug or numeric id via the Gamma API."""
    if not slug and not market_id:
        raise MarketDataError("Provide a Polymarket market slug or id.")
    params: Dict[str, Any] = {}
    if slug:
        params["slug"] = slug
    if market_id:
        params["id"] = market_id
    data = _get_json(POLYMARKET_GAMMA_URL, params=params)
    if isinstance(data, list):
        market = data[0] if data else None
    elif isinstance(data, dict):
        market = data
    else:
        market = None
    if not market:
        raise MarketDataError("Polymarket market not found.")

    labels = _as_list(market.get("outcomes"))
    prices = [_to_float(p) for p in _as_list(market.get("outcomePrices"))]
    options = [
        {"label": str(label), "probability": prices[i] if i < len(prices) else None}
        for i, label in enumerate(labels)
    ]
    outcome, probability = _primary_outcome(options)
    resolved_slug = market.get("slug") or slug or ""
    return {
        "platform": "Polymarket",
        "question": market.get("question") or market.get("title") or "",
        "market_url": f"https://polymarket.com/market/{resolved_slug}" if resolved_slug else "",
        "outcome": outcome,
        "probability": probability,
        "outcomes": options,
    }


def fetch_kalshi(ticker: str) -> Dict[str, Any]:
    """Fetch a Kalshi market by ticker via the public trade API v2."""
    if not ticker:
        raise MarketDataError("Provide a Kalshi market ticker.")
    ticker = ticker.strip().upper()
    data = _get_json(f"{KALSHI_API_URL}/{ticker}")
    market = data.get("market") if isinstance(data, dict) else None
    if not market:
        raise MarketDataError("Kalshi market not found.")

    # Kalshi prices are in cents (0..100). Prefer last trade, else bid/ask midpoint.
    last = market.get("last_price")
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")
    cents: Optional[float]
    if last:
        cents = float(last)
    elif yes_bid is not None and yes_ask is not None:
        cents = (float(yes_bid) + float(yes_ask)) / 2.0
    elif yes_bid is not None:
        cents = float(yes_bid)
    else:
        cents = None
    probability = round(cents / 100.0, 4) if cents is not None else None
    no_probability = round(1.0 - probability, 4) if probability is not None else None
    return {
        "platform": "Kalshi",
        "question": market.get("title") or market.get("subtitle") or ticker,
        "market_url": f"https://kalshi.com/markets/{ticker}",
        "outcome": "Yes",
        "probability": probability,
        "outcomes": [
            {"label": "Yes", "probability": probability},
            {"label": "No", "probability": no_probability},
        ],
    }
