"""Foresea Radar: public discovery surface for interesting prediction markets.

Radar intentionally does not call the forecasting model while a visitor loads
the page. It composes public market listings, Reddit discussion signals, and the
latest committed Foresea track-record snapshots. That keeps the product surface
cheap and avoids reintroducing the Cloud Run OOM pattern the track-record move
was designed to remove.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from analyzing_llm_rationale import market_data

_REDDIT_RSS = "https://www.reddit.com/r/{sub}/new/.rss?limit=25"
_HEADERS = {"User-Agent": "foresea-radar/1.0 (by /u/pareelamre)"}

_SEED_MARKETS = [
    {
        "platform": "Polymarket",
        "ident": "will-an-artwork-sell-for-150-million-by-december-31",
        "source": "research_seed",
        "reddit_url": "https://old.reddit.com/r/PredictionMarkets/comments/1twvyv3/is_anyone_actually_buying_a_150m_painting_this/",
        "note": "Clean auction threshold, thin but discussion-driven.",
    },
    {
        "platform": "Polymarket",
        "ident": "will-victor-marx-win-the-2026-colorado-governor-republican-primary-election",
        "source": "research_seed",
        "reddit_url": "https://old.reddit.com/r/PredictionMarkets/comments/1twvtiy/gop_governor_candidate_victor_marx_admits_to/",
        "note": "Political primary with a near-term catalyst and public controversy.",
    },
    {
        "platform": "Kalshi",
        "ident": "KXAGICO-COMP-26Q2",
        "source": "track_record_seed",
        "note": "Already part of Foresea's live track-record store.",
    },
]

_SPORT_WORDS = {
    "nba", "nfl", "mlb", "nhl", "wnba", "soccer", "ufc", "fifa", "world cup",
    "dota", "lol", "league of legends", "cs2", "valorant", "championship",
}
_CATALYST_WORDS = {
    "earnings", "election", "primary", "vote", "court", "trial", "decision",
    "rate", "fed", "inflation", "cpi", "release", "call", "debate", "launch",
    "auction", "sale", "deadline",
}
_RULE_RISK_WORDS = {
    "mention", "mentions", "say", "says", "tweet", "post", "announce",
    "announcement", "before", "by", "first", "highest", "lowest",
}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, dict) and "__dt__" in value:
        value = value["__dt__"]
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _days_until(value: Any, now: Optional[datetime] = None) -> Optional[float]:
    dt = _parse_dt(value)
    if dt is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (dt - now).total_seconds() / 86400.0


def _wordish_topic(title: str) -> str:
    words = re.findall(r"[a-z0-9]+", title.lower())
    stop = {
        "will", "what", "when", "who", "does", "do", "did", "the", "a", "an",
        "is", "are", "this", "that", "market", "markets", "polymarket", "kalshi",
        "prediction", "bet", "bets", "yes", "no", "by", "before", "after",
    }
    keep = [w for w in words if w not in stop and len(w) > 2]
    return " ".join(keep[:4])


def _contains_any(text: str, words: Iterable[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def _liquidity_tag(volume: Optional[float], liquidity: Optional[float]) -> str:
    size = max(volume or 0.0, liquidity or 0.0)
    if size < 1000:
        return "thin liquidity"
    if size < 25000:
        return "medium liquidity"
    return "liquid"


def market_tags(item: Dict[str, Any], now: Optional[datetime] = None) -> List[str]:
    question = item.get("question") or item.get("title") or ""
    tags: List[str] = []
    tags.append(_liquidity_tag(_to_float(item.get("volume")), _to_float(item.get("liquidity"))))
    if _contains_any(question, _RULE_RISK_WORDS):
        tags.append("resolution-rule risk")
    if _contains_any(question, _CATALYST_WORDS):
        tags.append("news catalyst")
    if _contains_any(question, _SPORT_WORDS) or str(item.get("category") or "").lower() == "sports":
        tags.append("crowded sports market")
    days = _days_until(item.get("close_time"), now)
    if days is not None and 0 <= days <= 14:
        tags.append("near-term")
    if item.get("reddit_url") or item.get("reddit_mentions"):
        tags.append("reddit-discussed")
    if item.get("tracked_live"):
        tags.append("tracked live")
    return list(dict.fromkeys(tags))


def credibility_score(item: Dict[str, Any], now: Optional[datetime] = None) -> int:
    score = 35
    if item.get("market_url"):
        score += 12
    if item.get("probability") is not None:
        score += 10
    if item.get("close_time"):
        score += 10
    volume = _to_float(item.get("volume")) or 0.0
    liquidity = _to_float(item.get("liquidity")) or 0.0
    if max(volume, liquidity) >= 1000:
        score += 10
    if max(volume, liquidity) >= 25000:
        score += 5
    if item.get("reddit_url") or item.get("reddit_mentions"):
        score += 8
    if item.get("tracked_live"):
        score += 10
    if "resolution-rule risk" in market_tags(item, now):
        score -= 8
    if "thin liquidity" in market_tags(item, now):
        score -= 8
    if not item.get("market_url"):
        score -= 18
    return max(0, min(100, score))


def unusual_score(item: Dict[str, Any], now: Optional[datetime] = None) -> float:
    change = abs(_to_float(item.get("price_change_24h")) or 0.0)
    volume = max(_to_float(item.get("volume")) or 0.0, 0.0)
    prob = _to_float(item.get("probability"))
    contested = 1.0 - min(abs((prob or 0.5) - 0.5) * 2.0, 1.0)
    days = _days_until(item.get("close_time"), now)
    near = 1.0 if days is not None and 0 <= days <= 14 else 0.0
    reddit = 1.0 if (item.get("reddit_url") or item.get("reddit_mentions")) else 0.0
    return round((change * 100.0) + math.log10(volume + 10.0) * 6.0 + contested * 8.0 + near * 8.0 + reddit * 7.0, 3)


def _normalise_quote(quote: Dict[str, Any], source: str = "venue_scan") -> Dict[str, Any]:
    ident = quote.get("ident")
    if not ident and quote.get("market_url"):
        ident = str(quote["market_url"]).rstrip("/").split("/")[-1]
    return {
        "platform": quote.get("platform"),
        "ident": ident,
        "question": quote.get("question") or "",
        "market_url": quote.get("market_url"),
        "outcome": quote.get("outcome") or "Yes",
        "market_probability": quote.get("probability"),
        "probability": quote.get("probability"),
        "foresea_probability": None,
        "edge": None,
        "close_time": quote.get("close_time"),
        "volume": quote.get("volume"),
        "liquidity": quote.get("liquidity"),
        "price_change_24h": quote.get("price_change_24h"),
        "category": quote.get("category"),
        "source": source,
        "evidence_links": [],
        "tracked_live": False,
    }


def _fetch_seed(seed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        if seed["platform"].lower() == "polymarket":
            quote = market_data.fetch_polymarket(slug=seed["ident"])
        else:
            quote = market_data.fetch_kalshi(seed["ident"])
    except market_data.MarketDataError:
        return None
    item = _normalise_quote(quote, source=seed.get("source") or "research_seed")
    item["reddit_url"] = seed.get("reddit_url")
    item["note"] = seed.get("note")
    return item


def fetch_reddit_discussions(subreddits: Iterable[str] = ("PredictionMarkets", "Polymarket", "Kalshi"),
                             limit_per_sub: int = 8) -> List[Dict[str, Any]]:
    import requests

    _ATOM = "http://www.w3.org/2005/Atom"
    rows: List[Dict[str, Any]] = []
    for sub in subreddits:
        try:
            import xml.etree.ElementTree as ET

            resp = requests.get(_REDDIT_RSS.format(sub=sub), headers=_HEADERS, timeout=8)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.text)
        except Exception:
            continue
        for entry in list(root.findall(f"{{{_ATOM}}}entry"))[:limit_per_sub]:
            title_el = entry.find(f"{{{_ATOM}}}title")
            link_el = entry.find(f"{{{_ATOM}}}link")
            title = (title_el.text or "").strip() if title_el is not None else ""
            if not title:
                continue
            url = link_el.get("href", "") if link_el is not None else ""
            rows.append({
                "subreddit": sub,
                "title": title,
                "url": url,
                "score": None,
                "topic": _wordish_topic(title),
            })
    return rows[:30]


def _latest_snapshots(store_path: Path) -> Dict[str, Dict[str, Any]]:
    if not store_path.exists():
        return {}
    try:
        raw = json.loads(store_path.read_text())
    except Exception:
        return {}
    snapshots = raw.get("ForecastSnapshot") or {}
    latest: Dict[str, Dict[str, Any]] = {}
    for snap in snapshots.values():
        ident = str(snap.get("ident") or "").strip()
        url = str(snap.get("market_url") or "").rstrip("/")
        keys = [k for k in [ident, url, url.split("/")[-1] if url else ""] if k]
        if not keys:
            continue
        ts = _parse_dt(snap.get("snapshot_ts"))
        current = latest.get(keys[0])
        current_ts = _parse_dt(current.get("snapshot_ts")) if current else None
        if current is not None and ts is not None and current_ts is not None and ts <= current_ts:
            continue
        for key in keys:
            latest[key] = snap
    return latest


def _merge_snapshot(item: Dict[str, Any], snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        str(item.get("ident") or "").strip(),
        str(item.get("market_url") or "").rstrip("/"),
        str(item.get("market_url") or "").rstrip("/").split("/")[-1],
    ]
    snap = next((snapshots[k] for k in keys if k and k in snapshots), None)
    if not snap:
        return item
    item["tracked_live"] = True
    item["foresea_probability"] = snap.get("model_probability")
    item["market_probability"] = snap.get("market_probability", item.get("market_probability"))
    item["probability"] = item.get("market_probability")
    item["edge"] = (
        item["foresea_probability"] - item["market_probability"]
        if item.get("foresea_probability") is not None and item.get("market_probability") is not None
        else None
    )
    item["snapshot_date"] = snap.get("snapshot_date")
    item["evidence_count"] = snap.get("evidence_count")
    return item


def _dedupe_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = str(item.get("market_url") or item.get("ident") or item.get("question") or "").lower()
        if not key:
            continue
        existing = out.get(key)
        if existing is None:
            out[key] = item
            continue
        if item.get("reddit_url") and not existing.get("reddit_url"):
            existing["reddit_url"] = item["reddit_url"]
        existing["source"] = existing.get("source") or item.get("source")
    return list(out.values())


def _add_evidence(item: Dict[str, Any]) -> None:
    links = []
    if item.get("market_url"):
        links.append({"label": item.get("platform") or "Market", "url": item["market_url"]})
    if item.get("reddit_url"):
        links.append({"label": "Reddit discussion", "url": item["reddit_url"]})
    item["evidence_links"] = links


def enrich_items(items: Iterable[Dict[str, Any]], repo_root: Path) -> List[Dict[str, Any]]:
    """Merge committed Foresea snapshots, score, tag, dedupe, and add links."""
    now = datetime.now(timezone.utc)
    store_path = repo_root / "data" / "track_record_store.json"
    snapshots = _latest_snapshots(store_path)
    enriched = []
    for item in _dedupe_items(items):
        _merge_snapshot(item, snapshots)
        item["tags"] = market_tags(item, now)
        item["credibility_score"] = credibility_score(item, now)
        item["unusual_score"] = unusual_score(item, now)
        _add_evidence(item)
        enriched.append(item)
    return enriched


def rank_items(items: Iterable[Dict[str, Any]], *, limit: int = 10) -> List[Dict[str, Any]]:
    ranked = sorted(
        list(items),
        key=lambda x: (
            x.get("tracked_live") is True,
            x.get("foresea_probability") is not None,
            abs(x.get("edge") or 0.0),
            x.get("unusual_score") or 0,
            x.get("credibility_score") or 0,
        ),
        reverse=True,
    )
    return ranked[: max(1, min(limit, 25))]


def collect_candidates(*, include_reddit: bool = True) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch public Radar candidates without forecasting them."""
    items: List[Dict[str, Any]] = []

    for seed in _SEED_MARKETS:
        fetched = _fetch_seed(seed)
        if fetched:
            items.append(fetched)

    for lister, platform in [(market_data.list_polymarket, "Polymarket"), (market_data.list_kalshi, "Kalshi")]:
        try:
            for quote in lister(limit=10, min_close_days=1, max_close_days=45):
                items.append(_normalise_quote(quote, source=f"{platform.lower()}_scan"))
        except market_data.MarketDataError:
            continue

    reddit = fetch_reddit_discussions() if include_reddit else []
    for post in reddit[:8]:
        items.append({
            "platform": None,
            "ident": None,
            "question": post["title"],
            "market_url": None,
            "outcome": "Yes",
            "market_probability": None,
            "probability": None,
            "foresea_probability": None,
            "edge": None,
            "close_time": None,
            "volume": None,
            "liquidity": None,
            "price_change_24h": None,
            "category": None,
            "source": "reddit_discussion",
            "reddit_url": post.get("url"),
            "reddit_subreddit": post.get("subreddit"),
            "reddit_score": post.get("score"),
            "reddit_topic": post.get("topic"),
            "evidence_links": [],
            "tracked_live": False,
        })
    return items, reddit[:10]


def render_radar_payload(
    items: Iterable[Dict[str, Any]],
    reddit: List[Dict[str, Any]],
    *,
    limit: int = 10,
    generated_by: str = "request",
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    enriched = list(items)
    top = rank_items(enriched, limit=limit)
    return {
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "generated_by": generated_by,
        "methodology": (
            "Radar ranks public market listings, Reddit discussion signals, and "
            "Foresea's committed live snapshots. The public endpoint serves this "
            "committed JSON artifact; scheduled GitHub Actions may add fresh "
            "Foresea forecasts before committing it."
        ),
        "items": top,
        "unusual_moves": sorted(
            [x for x in enriched if x.get("market_url")],
            key=lambda x: x.get("unusual_score") or 0,
            reverse=True,
        )[:10],
        "reddit_discussions": reddit[:10],
        "tracked_live": [x for x in top if x.get("tracked_live")],
        "tag_definitions": {
            "thin liquidity": "Low venue volume/open interest; expect noisy prices and slippage.",
            "resolution-rule risk": "Wording may be sensitive to platform adjudication or exact definitions.",
            "news catalyst": "A dated external event could move the price quickly.",
            "crowded sports market": "Popular sports/esports markets can be efficient and sentiment-heavy.",
            "near-term": "Scheduled to close within roughly two weeks.",
            "tracked live": "Foresea has a committed point-in-time snapshot for this market.",
        },
    }


def build_radar(repo_root: Path, *, limit: int = 10, include_reddit: bool = True) -> Dict[str, Any]:
    items, reddit = collect_candidates(include_reddit=include_reddit)
    enriched = enrich_items(items, repo_root)
    return render_radar_payload(enriched, reddit, limit=limit)
