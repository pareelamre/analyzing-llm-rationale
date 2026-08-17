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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
POLYMARKET_TAGS_URL = "https://gamma-api.polymarket.com/tags"
POLYMARKET_CLOB_BOOK_URL = "https://clob.polymarket.com/book"
POLYMARKET_HISTORY_URL = "https://clob.polymarket.com/prices-history"
POLYMARKET_SPORTS_URL = "https://gamma-api.polymarket.com/sports"
POLYMARKET_COMMENTS_URL = "https://gamma-api.polymarket.com/comments"
POLYMARKET_SERIES_URL = "https://gamma-api.polymarket.com/series"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
KALSHI_EXCHANGE_STATUS_URL = "https://api.elections.kalshi.com/trade-api/v2/exchange/status"
KALSHI_EXCHANGE_SCHEDULE_URL = "https://api.elections.kalshi.com/trade-api/v2/exchange/schedule"
KALSHI_LIVE_DATA_URL = "https://api.elections.kalshi.com/trade-api/v2/live-data"
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


def _venue_news_articles(payload: Dict[str, Any], platform: str) -> List[Dict[str, Any]]:
    """Normalize any venue-supplied article objects without inventing sources.

    Neither venue guarantees an article collection in every market response, but
    both evolve their public payloads. Preserve recognized article collections
    when present so the forecasting API can merge them with its live news search.
    """
    articles: List[Dict[str, Any]] = []
    containers = [payload]
    containers.extend(
        item for item in (payload.get("events") or [])
        if isinstance(item, dict)
    )
    for container in containers:
        for field in ("news", "news_articles", "newsArticles", "articles", "relatedArticles"):
            values = container.get(field)
            if isinstance(values, dict):
                values = values.get("items") or values.get("articles") or []
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or item.get("headline")
                url = item.get("url") or item.get("link")
                summary = item.get("summary") or item.get("description")
                if not any((title, url, summary)):
                    continue
                articles.append({
                    "title": str(title) if title else None,
                    "url": str(url) if url else None,
                    "summary": str(summary) if summary else None,
                    "source": str(item.get("source") or item.get("publisher") or platform),
                    "publish_date": item.get("publish_date")
                    or item.get("publishedAt")
                    or item.get("published_at"),
                })
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for article in articles:
        key = (
            (article.get("url") or "").strip().lower()
            or (article.get("title") or "").strip().lower()
        )
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped[:20]


def _polymarket_rules(market: Dict[str, Any]) -> Optional[str]:
    parts: List[str] = []
    for value in (
        market.get("rules"),
        market.get("description"),
        *[
            event.get("description")
            for event in (market.get("events") or [])
            if isinstance(event, dict)
        ],
    ):
        text = str(value or "").strip()
        if text and text not in parts:
            parts.append(text)
    return "\n\n".join(parts) or None


def _within_close_window(close_time: Any, min_days: Optional[float], max_days: Optional[float]) -> bool:
    """Whether a market's resolution time falls in [min_days, max_days] from now.

    Used to keep the track record to markets that will actually resolve in a
    useful window (not same-day noise, not multi-decade markets that never
    score). ``None`` bounds disable that side; an unparseable/absent close time
    is kept only when there's no max bound.
    """
    if min_days is None and max_days is None:
        return True
    if not close_time:
        return max_days is None
    try:
        cdt = datetime.fromisoformat(str(close_time).strip().replace("Z", "+00:00"))
    except ValueError:
        return max_days is None
    if cdt.tzinfo is None:
        cdt = cdt.replace(tzinfo=timezone.utc)
    lead = (cdt - datetime.now(timezone.utc)).total_seconds() / 86400.0
    if min_days is not None and lead < min_days:
        return False
    if max_days is not None and lead > max_days:
        return False
    return True


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
    # The YES-outcome CLOB token id -- what /market/batch needs to fetch order
    # book depth / price history for this market from marketd. Only set for a
    # clean binary Yes/No market (matches services/marketd/polymarket.go's
    # normalizePolymarket, which is equally conservative).
    token_ids = _as_list(market.get("clobTokenIds"))
    token_id = None
    for i, label in enumerate(labels):
        if str(label).strip().lower() == "yes" and i < len(token_ids):
            token_id = str(token_ids[i]).strip() or None
            break
    slug = market.get("slug") or ""
    events = market.get("events") or []
    event = events[0] if events and isinstance(events[0], dict) else {}
    event_metadata = event.get("eventMetadata") or {}
    description = (
        event_metadata.get("context_description")
        or event.get("description")
        or market.get("description")
        or ""
    ).strip() or None
    event_sources = [
        str(event.get("resolutionSource") or "").strip()
        for event in events
        if isinstance(event, dict) and event.get("resolutionSource")
    ]
    return {
        "platform": "Polymarket",
        "question": market.get("question") or market.get("title") or "",
        "market_url": f"https://polymarket.com/market/{slug}" if slug else "",
        "ident": slug,
        "outcome": outcome,
        "probability": probability,
        "outcomes": options,
        "close_time": market.get("endDate") or market.get("endDateIso"),
        "created_time": market.get("startDate") or market.get("createdAt"),
        "description": description,
        "resolution_criteria": _polymarket_rules(market),
        "volume": _to_float(market.get("volume24hr") or market.get("volume")),
        "liquidity": _to_float(market.get("liquidity") or market.get("liquidityNum")),
        "price_change_24h": _to_float(
            market.get("oneDayPriceChange")
            or market.get("priceChange24hr")
            or market.get("priceChange24h")
        ),
        "yes_bid": _to_float(market.get("bestBid")),
        "yes_ask": _to_float(market.get("bestAsk")),
        "last_trade_price": _to_float(market.get("lastTradePrice") or market.get("last_trade_price")),
        "price_change_7d": _to_float(market.get("oneWeekPriceChange") or market.get("weekPriceChange")),
        "resolution_source": (
            market.get("resolverUrl")
            or market.get("resolutionSource")
            or (event_sources[0] if event_sources else "")
            or ""
        ).strip() or None,
        "venue_news_articles": _venue_news_articles(market, "Polymarket"),
        "category": market.get("category"),
        "token_id": token_id,
    }


def resolve_polymarket(slug: str) -> Optional[int]:
    """Return 1/0 if a binary Polymarket market has resolved YES/NO, else None.

    A market is resolved when ``closed`` and ``umaResolutionStatus == resolved``;
    the winning outcome's ``outcomePrices`` entry settles to ~1. Used by the live
    track record to score a forecast once its market resolves.
    """
    if not slug:
        return None
    # The Gamma /markets endpoint excludes closed markets by default, so a plain
    # slug lookup returns nothing once a market resolves — which silently blocked
    # all resolution. closed=true is required to see a settled market.
    data = _get_json(POLYMARKET_GAMMA_URL, params={"slug": slug, "closed": "true"})
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


_CATEGORY_KEYWORDS = {
    "Politics": ["election", "president", "senate", "congress", "trump", "biden",
                 "vote", "governor", "parliament", "prime minister", "government",
                 "shutdown", "poll", "democrat", "republican", "mayor", "cabinet"],
    "Crypto": ["bitcoin", "btc", "ethereum", " eth ", "crypto", "solana",
               "dogecoin", "token", "blockchain", "coinbase", "stablecoin"],
    "Sports": ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
               "world cup", "super bowl", "premier league", "champion", "playoff",
               "tournament", " vs ", "ufc", "formula 1", " f1 ", "golf", "tennis",
               "olympic", "world series"],
    "Economics": ["fed", "interest rate", "inflation", "gdp", "jobs report",
                  "recession", "cpi", "unemployment", "economy", "jobless", "rate cut"],
    "Entertainment": ["movie", "film", "oscar", "album", "song", "box office",
                      "grammy", "emmy", "netflix", "celebrity", "tv show", "season",
                      "rotten tomatoes", "spotify", "billboard"],
    "Tech": ["openai", "tesla", "apple", "google", "nvidia", "chip", "gpt",
             "spacex", "artificial intelligence", "iphone", "software", "startup"],
    "World": ["ukraine", "russia", "china", "israel", "gaza", "war", "nuclear",
              "climate", "nato", "united nations", "summit", "ceasefire"],
}


def _market_category(question: Optional[str], raw: Optional[str] = None) -> str:
    """Normalise a market into a browse category. Uses keyword heuristics on the
    question (works for Polymarket, whose listing has no category), then falls
    back to the venue's own category (Kalshi provides one)."""
    q = (question or "").lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(k in q for k in keywords):
            return category
    return str(raw) if raw else "Other"


def list_polymarket(limit: int = 5, query: Optional[str] = None,
                    min_close_days: Optional[float] = None,
                    max_close_days: Optional[float] = None,
                    contested_only: bool = True,
                    category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List liquid, contested binary Polymarket markets (for the edge scan).

    Pulls high-volume markets, then keeps binary Yes/No markets priced in the
    mid-range so the scan focuses on genuinely contested questions. When
    ``query`` is given, only markets whose question contains that keyword are
    kept. ``category`` filters by the market's category (substring, case-
    insensitive). ``min_close_days``/``max_close_days`` optionally restrict to a
    resolution-horizon window (used by the live track record).
    """
    limit = max(1, min(int(limit), 200))
    want = (query or "").strip().lower()
    cat = (category or "").strip().lower()
    # Search deeper when filtering, since matches may not be top-volume.
    deeper = bool(want or cat or min_close_days is not None or max_close_days is not None)
    candidate_cap = min(500, limit * (60 if deeper else 10))
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
        if contested_only and not (_SCAN_MIN_PRICE <= prob <= _SCAN_MAX_PRICE):
            continue
        if want and want not in (quote["question"] or "").lower():
            continue
        quote["category"] = _market_category(quote["question"], quote.get("category"))
        if cat and cat not in quote["category"].lower():
            continue
        if not _within_close_window(quote.get("close_time"), min_close_days, max_close_days):
            continue
        quotes.append(quote)
        if len(quotes) >= limit:
            break
    return quotes


def _kalshi_series_ticker(market: Dict[str, Any]) -> str:
    """The series ticker — the root of a Kalshi web market URL.

    Kalshi web pages live at ``/markets/<series_ticker>`` (lowercase), NOT at the
    raw event ticker (which carries a date suffix and 404s). Prefer the explicit
    ``series_ticker``; else strip the event ticker's trailing ``-<date>`` segment;
    else fall back to the market ticker's first segment.
    """
    series = (market.get("series_ticker") or "").strip()
    if not series:
        event = (market.get("event_ticker") or "").strip()
        base = event or (market.get("ticker") or "").strip()
        series = base.rsplit("-", 1)[0] if "-" in base else base
    return series.lower()


def _kalshi_quote(
    market: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    events = market.get("events") or []
    event = event or (
        events[0] if events and isinstance(events[0], dict) else {}
    )
    ticker = (market.get("ticker") or "").strip().upper()
    # The raw series ticker -- what /market/batch needs to fetch candlesticks
    # for this market from marketd (which requires both series_ticker and
    # ticker). Distinct from _kalshi_series_ticker() below, which lowercases
    # and falls back to a derived value for building a *display* URL; the
    # candlesticks API needs the exact, canonically-cased value or nothing.
    series_ticker = (market.get("series_ticker") or event.get("series_ticker") or "").strip() or None
    # Kalshi prices are in the *_dollars fields (0..1). Use the current
    # actionable book midpoint before last trade: a thin contract's last print
    # can be days old and create a fake council-vs-crowd discrepancy even though
    # today's bid/ask agrees with the council.
    last = _to_float(market.get("last_price_dollars"))
    yes_bid = _to_float(market.get("yes_bid_dollars"))
    yes_ask = _to_float(market.get("yes_ask_dollars"))
    valid_book = (
        yes_bid is not None
        and yes_ask is not None
        and 0.0 <= yes_bid <= yes_ask <= 1.0
        and (yes_bid > 0.0 or yes_ask < 1.0)
    )
    if valid_book:
        probability = (yes_bid + yes_ask) / 2.0
    elif last is not None and 0.0 < last < 1.0:
        probability = last
    elif yes_bid is not None and yes_bid > 0.0:
        probability = yes_bid
    else:
        probability = None
    probability = round(probability, 4) if probability is not None else None
    no_probability = round(1.0 - probability, 4) if probability is not None else None
    settlement_sources = [
        source for source in (event.get("settlement_sources") or [])
        if isinstance(source, dict)
    ]
    source_labels = [
        " — ".join(
            value for value in (
                str(source.get("name") or "").strip(),
                str(source.get("url") or "").strip(),
            )
            if value
        )
        for source in settlement_sources
    ]
    return {
        "platform": "Kalshi",
        "question": market.get("title") or market.get("yes_sub_title") or market.get("subtitle") or ticker,
        "market_url": (lambda s: f"https://kalshi.com/markets/{s}" if s else "")(
            _kalshi_series_ticker(market)),
        "ident": ticker,
        "outcome": "Yes",
        "probability": probability,
        "outcomes": [
            {"label": "Yes", "probability": probability},
            {"label": "No", "probability": no_probability},
        ],
        "close_time": market.get("close_time"),
        "created_time": market.get("created_time") or market.get("open_time"),
        "description": (event.get("sub_title") or "").strip() or None,
        "volume": _to_float(market.get("volume_24h_fp") or market.get("volume")),
        "liquidity": _to_float(market.get("open_interest") or market.get("liquidity")),
        "price_change_24h": _to_float(
            market.get("price_change_24h")
            or market.get("last_price_change_dollars")
            or market.get("change_dollars")
        ),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "last_trade_price": last if last is not None else None,
        "price_change_7d": None,  # not in Kalshi API
        "resolution_source": "; ".join(source_labels) or "Kalshi",
        "resolution_criteria": " ".join(filter(None, [
            (market.get("rules_primary") or "").strip(),
            (market.get("rules_secondary") or "").strip(),
        ])) or None,
        "venue_news_articles": _venue_news_articles(market, "Kalshi"),
        "no_sub_title": (market.get("no_sub_title") or "").strip() or None,
        "expected_expiration_time": market.get("expected_expiration_time"),
        "floor_strike": market.get("floor_strike"),
        "cap_strike": market.get("cap_strike"),
        "category": None,  # set from the event in list_kalshi
        "series_ticker": series_ticker,
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
    event_ticker = str(market.get("event_ticker") or "").strip()
    if event_ticker:
        try:
            event_data = _get_json(f"{KALSHI_EVENTS_URL}/{event_ticker}")
            event = event_data.get("event") if isinstance(event_data, dict) else None
            if isinstance(event, dict):
                market = {**market, "events": [event]}
        except MarketDataError:
            # The market detail still contains the canonical contract rules.
            # Event-level article enrichment is best effort.
            pass
    return _kalshi_quote(market)


def list_kalshi(limit: int = 5, query: Optional[str] = None,
                min_close_days: Optional[float] = None,
                max_close_days: Optional[float] = None,
                contested_only: bool = True,
                category: Optional[str] = None,
                paginate: bool = False) -> List[Dict[str, Any]]:
    """List open, priced Kalshi markets via the ``/events`` endpoint.

    The flat ``/markets?status=open`` listing is saturated by auto-generated
    multi-leg "MVE" parlay markets, so we pull real markets grouped under events
    instead, skip MVE legs, and build a readable question from the event title
    (plus the candidate sub-title for multi-outcome events). ``query`` filters by
    keyword; ``min_close_days``/``max_close_days`` restrict the resolution
    horizon. Results are sorted soonest-resolving first.
    Set ``paginate=True`` to follow cursors up to 1000 events (used by the tick).
    """
    limit = max(1, min(int(limit), 200))
    want = (query or "").strip().lower()
    cat = (category or "").strip().lower()
    events: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while len(events) < (1000 if paginate else 200):
        params: Dict[str, Any] = {
            "status": "open", "with_nested_markets": "true", "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        data = _get_json(KALSHI_EVENTS_URL, params=params)
        if not isinstance(data, dict):
            break
        page = data.get("events") or []
        events.extend(page)
        cursor = data.get("cursor")
        if not paginate or not cursor or len(page) < 200:
            break
    quotes: List[Dict[str, Any]] = []
    for event in events:
        title = event.get("title") or ""
        event_category = event.get("category")
        for market in event.get("markets", []) or []:
            if market.get("mve_collection_ticker"):
                continue  # skip multi-leg parlay markets
            quote = _kalshi_quote(market, event)
            prob = quote["probability"]
            if prob is None:
                continue
            if contested_only and not (_SCAN_MIN_PRICE <= prob <= _SCAN_MAX_PRICE):
                continue
            sub = (market.get("yes_sub_title") or "").strip()
            question = f"{title} — {sub}" if (sub and sub.lower() not in title.lower()) else (title or quote["question"])
            quote["question"] = question
            quote["category"] = _market_category(question, event_category)
            if want and want not in question.lower():
                continue
            if cat and cat not in quote["category"].lower():
                continue
            if not _within_close_window(quote.get("close_time"), min_close_days, max_close_days):
                continue
            quotes.append(quote)
    # Prefer soonest-resolving so the track record accrues scored outcomes sooner.
    quotes.sort(key=lambda q: q.get("close_time") or "9999")
    return quotes[:limit]


def fetch_kalshi_exchange_status() -> Dict[str, Any]:
    """Fetch live exchange status from Kalshi API (/exchange/status)."""
    data = _get_json(KALSHI_EXCHANGE_STATUS_URL)
    return data if isinstance(data, dict) else {"exchange_active": True, "trading_active": True}


def fetch_kalshi_exchange_schedule() -> Dict[str, Any]:
    """Fetch trading schedule and maintenance windows from Kalshi API (/exchange/schedule)."""
    data = _get_json(KALSHI_EXCHANGE_SCHEDULE_URL)
    return data if isinstance(data, dict) else {"schedule": []}


def fetch_kalshi_orderbook(ticker: str) -> Dict[str, Any]:
    """Fetch live orderbook depth for a Kalshi market ticker."""
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook"
    data = _get_json(url)
    if isinstance(data, dict) and "orderbook" in data:
        return data["orderbook"]
    return data if isinstance(data, dict) else {}


def fetch_polymarket_tags() -> List[Dict[str, Any]]:
    """Fetch active categories/tags from Polymarket Gamma API."""
    data = _get_json(POLYMARKET_TAGS_URL)
    return [t for t in data if isinstance(t, dict)] if isinstance(data, list) else []


def fetch_polymarket_orderbook(token_id: str) -> Dict[str, Any]:
    """Fetch live CLOB orderbook for a Polymarket token ID."""
    data = _get_json(POLYMARKET_CLOB_BOOK_URL, params={"token_id": token_id})
    return data if isinstance(data, dict) else {}


def fetch_polymarket_price_history(market: str, interval: str = "1d") -> List[Dict[str, Any]]:
    """Fetch historical prices for a Polymarket market condition or token."""
    data = _get_json(POLYMARKET_HISTORY_URL, params={"market": market, "interval": interval})
    if isinstance(data, dict) and "history" in data:
        return data["history"]
    return data if isinstance(data, list) else []


def fetch_kalshi_candlesticks(ticker: str, series_ticker: str = "") -> List[Dict[str, Any]]:
    """Fetch historical OHLC candlesticks for a Kalshi market ticker."""
    s_ticker = series_ticker or ticker.split("-")[0]
    url = f"https://api.elections.kalshi.com/trade-api/v2/series/{s_ticker}/markets/{ticker}/candlesticks"
    data = _get_json(url)
    if isinstance(data, dict) and "candlesticks" in data:
        return data["candlesticks"]
    return data if isinstance(data, list) else []
def fetch_kalshi_live_data(event_ticker: str = "", data_type: str = "") -> Dict[str, Any]:
    """Fetch real-time sports game stats and live event feeds from Kalshi (/live-data)."""
    params = {}
    if event_ticker:
        params["event_ticker"] = event_ticker
    url = f"{KALSHI_LIVE_DATA_URL}/{data_type}" if data_type else KALSHI_LIVE_DATA_URL
    data = _get_json(url, params=params if not data_type else None)
    return data if isinstance(data, dict) else {}


def fetch_polymarket_sports() -> List[Dict[str, Any]]:
    """Fetch active sports leagues and market types from Polymarket Gamma API."""
    data = _get_json(POLYMARKET_SPORTS_URL)
    return [s for s in data if isinstance(s, dict)] if isinstance(data, list) else []


def fetch_polymarket_comments(market_id: str = "") -> List[Dict[str, Any]]:
    """Fetch public community comments for a Polymarket event/market."""
    params = {"market": market_id} if market_id else None
    data = _get_json(POLYMARKET_COMMENTS_URL, params=params)
    return [c for c in data if isinstance(c, dict)] if isinstance(data, list) else []


def fetch_polymarket_series() -> List[Dict[str, Any]]:
    """Fetch active event series listings from Polymarket Gamma API."""
    data = _get_json(POLYMARKET_SERIES_URL)
    return [s for s in data if isinstance(s, dict)] if isinstance(data, list) else []
