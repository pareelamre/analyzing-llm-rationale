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

# The edge scan only considers genuinely contested markets. Near-certain
# longshots (e.g. a 0.2% World Cup outcome) have huge volume but produce
# spurious "edges" from small model differences, so they're filtered out.
_SCAN_MIN_PRICE = 0.05
_SCAN_MAX_PRICE = 0.95


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


def _polymarket_quote(market: Dict[str, Any]) -> Dict[str, Any]:
    labels = _as_list(market.get("outcomes"))
    prices = [_to_float(p) for p in _as_list(market.get("outcomePrices"))]
    options = [
        {"label": str(label), "probability": prices[i] if i < len(prices) else None}
        for i, label in enumerate(labels)
    ]
    outcome, probability = _primary_outcome(options)
    slug = market.get("slug") or ""
    return {
        "platform": "Polymarket",
        "question": market.get("question") or market.get("title") or "",
        "market_url": f"https://polymarket.com/market/{slug}" if slug else "",
        "outcome": outcome,
        "probability": probability,
        "outcomes": options,
        "close_time": market.get("endDate") or market.get("endDateIso"),
    }


def resolve_polymarket(slug: str) -> Optional[int]:
    """Return 1/0 if a binary Polymarket market has resolved YES/NO, else None.

    A market is resolved when ``closed`` and ``umaResolutionStatus == resolved``;
    the winning outcome's ``outcomePrices`` entry settles to ~1. Used by the live
    track record to score a forecast once its market resolves.
    """
    if not slug:
        return None
    data = _get_json(POLYMARKET_GAMMA_URL, params={"slug": slug})
    if isinstance(data, list):
        market = data[0] if data else None
    elif isinstance(data, dict):
        market = data
    else:
        market = None
    if not market:
        return None
    closed = bool(market.get("closed"))
    status = str(market.get("umaResolutionStatus") or "").strip().lower()
    if not (closed and status == "resolved"):
        return None
    labels = _as_list(market.get("outcomes"))
    prices = [_to_float(p) for p in _as_list(market.get("outcomePrices"))]
    for i, label in enumerate(labels):
        if str(label).strip().lower() == "yes" and i < len(prices) and prices[i] is not None:
            return 1 if prices[i] >= 0.5 else 0
    return None


def resolve_kalshi(ticker: str) -> Optional[int]:
    """Return 1/0 if a Kalshi market has settled YES/NO, else None.

    A settled market has ``status`` in {settled, finalized} and ``result`` in
    {yes, no}.
    """
    if not ticker:
        return None
    data = _get_json(f"{KALSHI_API_URL}/{ticker.strip().upper()}")
    market = data.get("market") if isinstance(data, dict) else None
    if not market:
        return None
    status = str(market.get("status") or "").strip().lower()
    result = str(market.get("result") or "").strip().lower()
    if status in ("settled", "finalized") and result in ("yes", "no"):
        return 1 if result == "yes" else 0
    return None


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
    return _polymarket_quote(market)


def list_polymarket(limit: int = 5, query: Optional[str] = None) -> List[Dict[str, Any]]:
    """List liquid, contested binary Polymarket markets (for the edge scan).

    Pulls high-volume markets, then keeps binary Yes/No markets priced in the
    mid-range so the scan focuses on genuinely contested questions. When
    ``query`` is given, only markets whose question contains that keyword are
    kept (best-effort, over the high-volume candidate pool).
    """
    limit = max(1, min(int(limit), 20))
    want = (query or "").strip().lower()
    # Search deeper when filtering by keyword, since matches may not be top-volume.
    candidate_cap = min(500, limit * (60 if want else 10))
    data = _get_json(
        POLYMARKET_GAMMA_URL,
        params={
            "active": "true",
            "closed": "false",
            "limit": candidate_cap,
            "order": "volume24hr",
            "ascending": "false",
        },
    )
    quotes: List[Dict[str, Any]] = []
    for market in data if isinstance(data, list) else []:
        quote = _polymarket_quote(market)
        labels = {str(o["label"]).strip().lower() for o in quote["outcomes"]}
        prob = quote["probability"]
        if prob is None or labels != {"yes", "no"}:
            continue
        if not (_SCAN_MIN_PRICE <= prob <= _SCAN_MAX_PRICE):
            continue
        if want and want not in (quote["question"] or "").lower():
            continue
        quotes.append(quote)
        if len(quotes) >= limit:
            break
    return quotes


def _kalshi_quote(market: Dict[str, Any]) -> Dict[str, Any]:
    ticker = (market.get("ticker") or "").strip().upper()
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
        "market_url": f"https://kalshi.com/markets/{ticker}" if ticker else "",
        "outcome": "Yes",
        "probability": probability,
        "outcomes": [
            {"label": "Yes", "probability": probability},
            {"label": "No", "probability": no_probability},
        ],
        "close_time": market.get("close_time"),
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
    return _kalshi_quote(market)


def list_kalshi(limit: int = 5, query: Optional[str] = None) -> List[Dict[str, Any]]:
    """List open, priced Kalshi markets (for the edge scan).

    When ``query`` is given, only markets whose title contains that keyword are
    kept (best-effort, over the open-market candidate pool).
    """
    limit = max(1, min(int(limit), 20))
    want = (query or "").strip().lower()
    candidate_cap = min(1000, limit * (100 if want else 20))
    data = _get_json(KALSHI_API_URL, params={"status": "open", "limit": candidate_cap})
    markets = data.get("markets", []) if isinstance(data, dict) else []
    quotes: List[Dict[str, Any]] = []
    for market in markets:
        quote = _kalshi_quote(market)
        prob = quote["probability"]
        if prob is None or not (_SCAN_MIN_PRICE <= prob <= _SCAN_MAX_PRICE):
            continue
        if want and want not in (quote["question"] or "").lower():
            continue
        quotes.append(quote)
        if len(quotes) >= limit:
            break
    return quotes
