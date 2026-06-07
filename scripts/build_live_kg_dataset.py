"""Build a KG dataset from live prediction-market resolved questions.

Pulls resolved binary markets from Polymarket (Gamma API), Kalshi, and
Manifold Markets after a cutoff date. Tags each with entity_tagger and
writes a clean dataset ready for calibration / knowledge-graph enrichment.

These sources have no pre-attached news articles, so the output focuses on
question + outcome + entities + domain. That's sufficient for:
  - Domain-level calibration curves
  - Entity co-occurrence / track record enrichment

Usage:
    python scripts/build_live_kg_dataset.py [--output PATH] [--after 2024-12-01]
    python scripts/build_live_kg_dataset.py --sources polymarket kalshi
    python scripts/build_live_kg_dataset.py --limit 2000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from analyzing_llm_rationale.entity_tagger import tag_question  # noqa: E402

_DEFAULT_OUTPUT = _REPO / "data" / "kg_dataset_live.json"
_DEFAULT_AFTER = "2024-12-01"

_HEADERS = {"User-Agent": "foresea-kg-builder/1.0"}
_TIMEOUT = 15


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, params: Optional[Dict] = None) -> Any:
    import requests
    r = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip().rstrip("Z").replace(" ", "T").replace("+00:00", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_list(v: Any) -> List:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return []
    return []


_NOISE_PATTERNS = [
    # ultra-short horizon price targets
    r'\b\d+\s*min\b',
    r'\b\d+\s*minute',
    r'\b\d+\s*hr\b',
    r'\btarget\s*(?:price|level)\s*[:\$]',
    r'·\s*\$[\d,]+',
    r'\bprice\s+(?:above|below|hits?|reaches?)\s+\$',
    # Kalshi sports game lines (over/under, spread, totals)
    r'\bover\s+\d+\.?\d*\s+(?:runs?|points?|goals?|assists?|rebounds?)',
    r'\bunder\s+\d+\.?\d*\s+(?:runs?|points?|goals?|assists?|rebounds?)',
    r'\b(?:first|last)\s+\d+\s+innings?\s+total',
    r'\b(?:moneyline|spread|handicap)\b',
    r'\bvs\.?\s+\w+.*:\s+(?:total|spread|moneyline)',
    # Crypto price level automation (BTC/ETH/SOL above $X at T)
    r'\b(?:btc|eth|sol|xrp|bnb|ada|doge)\b.*\b(?:above|below|hits?|over|under)\b.*\$[\d,]+',
    r'\$[\d,]+\s+(?:target|level|strike)',
]
_NOISE_RE = [__import__('re').compile(p, __import__('re').IGNORECASE) for p in _NOISE_PATTERNS]


def _is_noise(question: str) -> bool:
    """Return True for low-quality / non-substantive market questions."""
    if len(question) < 25:
        return True
    return any(pat.search(question) for pat in _NOISE_RE)


def _record(
    source: str,
    ident: str,
    question: str,
    outcome: int,
    resolve_time: datetime,
    created_time: Optional[datetime],
    market_url: str,
    category: Optional[str] = None,
    market_prob: Optional[float] = None,
) -> Dict[str, Any]:
    tags = tag_question(question, category)
    horizon_days: Optional[float] = None
    if created_time and resolve_time:
        horizon_days = round((resolve_time - created_time).total_seconds() / 86400, 1)
    return {
        "id": f"{source}:{ident}",
        "source": source,
        "question": question.strip(),
        "outcome": outcome,
        "resolve_time": resolve_time.isoformat(),
        "created_time": created_time.isoformat() if created_time else None,
        "horizon_days": horizon_days,
        "market_probability": market_prob,
        "domain": tags["domain"],
        "entities": tags["entities"],
        "category": category or "",
        "market_url": market_url,
    }


# ── Polymarket ────────────────────────────────────────────────────────────────

def _fetch_polymarket(after_dt: datetime, limit: int) -> List[Dict[str, Any]]:
    """Pull resolved binary Polymarket markets after `after_dt`."""
    GAMMA = "https://gamma-api.polymarket.com/markets"
    results: List[Dict[str, Any]] = []
    offset = 0
    batch = min(500, limit * 4)
    seen: set = set()

    print("  Polymarket: fetching resolved markets…")
    while len(results) < limit:
        try:
            data = _get(GAMMA, params={
                "closed": "true",
                "limit": batch,
                "offset": offset,
                "order": "endDate",
                "ascending": "false",
            })
        except Exception as e:
            print(f"  Polymarket error at offset {offset}: {e}")
            break

        markets = data if isinstance(data, list) else []
        if not markets:
            break

        for m in markets:
            slug = m.get("slug") or ""
            if not slug or slug in seen:
                continue
            # Filter by resolution date
            end_dt = _parse_dt(m.get("endDate") or m.get("endDateIso"))
            if not end_dt or end_dt < after_dt:
                continue
            # Must be binary Yes/No
            labels = [str(l).strip().lower() for l in _as_list(m.get("outcomes"))]
            prices = [_to_float(p) for p in _as_list(m.get("outcomePrices"))]
            if set(labels) != {"yes", "no"}:
                continue
            # Determine outcome from settled prices
            outcome: Optional[int] = None
            market_prob: Optional[float] = None
            for i, label in enumerate(labels):
                p = prices[i] if i < len(prices) else None
                if label == "yes" and p is not None:
                    market_prob = p
                    if p >= 0.5:
                        outcome = 1
                    else:
                        outcome = 0
            if outcome is None:
                continue
            question = m.get("question") or m.get("title") or ""
            if _is_noise(question):
                continue
            seen.add(slug)
            results.append(_record(
                source="polymarket",
                ident=slug,
                question=question,
                outcome=outcome,
                resolve_time=end_dt,
                created_time=_parse_dt(m.get("startDate") or m.get("createdAt")),
                market_url=f"https://polymarket.com/market/{slug}",
                category=m.get("category"),
                market_prob=round(market_prob, 4) if market_prob is not None else None,
            ))
            if len(results) >= limit:
                break

        offset += len(markets)
        if len(markets) < batch:
            break
        time.sleep(0.3)

    print(f"  Polymarket: {len(results)} records")
    return results


# ── Kalshi ────────────────────────────────────────────────────────────────────

def _fetch_kalshi(after_dt: datetime, limit: int) -> List[Dict[str, Any]]:
    """Pull settled Kalshi markets after `after_dt`."""
    KALSHI_EVENTS = "https://api.elections.kalshi.com/trade-api/v2/events"
    results: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    seen_tickers: set = set()
    seen_events: set = set()   # one market per event_ticker
    seen_title_keys: set = set()  # deduplicate "which X wins this week" type patterns

    print("  Kalshi: fetching settled markets…")
    while len(results) < limit:
        params: Dict[str, Any] = {"status": "settled", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            data = _get(KALSHI_EVENTS, params=params)
        except Exception as e:
            print(f"  Kalshi error: {e}")
            break

        events = data.get("events", []) if isinstance(data, dict) else []
        cursor = data.get("cursor") if isinstance(data, dict) else None

        if not events:
            break

        for event in events:
            title = event.get("title") or ""
            category = event.get("category")
            event_ticker = (event.get("event_ticker") or "").strip().upper()

            # Skip automated/non-substantive categories
            if category in ("Crypto", "Financials", "Sports", "Mentions", "Climate and Weather"):
                continue

            # One record per event_ticker
            if event_ticker and event_ticker in seen_events:
                continue

            # Deduplicate "which X wins this week/today" style events by normalised title.
            # These produce many distinct event_tickers (one per song/show/candidate)
            # all sharing the same human-readable question title.
            title_key = re.sub(r'\s+', ' ', title.lower().strip())
            # Strip trailing date/ordinal so "Top Netflix Show this week" unifies
            title_key = re.sub(r'\b(this week|today|tonight|on \w+ \d+[,\s]|\d{4})\b.*$', '', title_key).strip()
            if title_key in seen_title_keys:
                continue

            for m in event.get("markets", []) or []:
                if m.get("mve_collection_ticker"):
                    continue
                ticker = (m.get("ticker") or "").strip().upper()
                if not ticker or ticker in seen_tickers:
                    continue
                status = str(m.get("status") or "").strip().lower()
                result = str(m.get("result") or "").strip().lower()
                if status not in ("settled", "finalized") or result not in ("yes", "no"):
                    continue
                close_dt = _parse_dt(m.get("close_time"))
                if not close_dt or close_dt < after_dt:
                    continue

                outcome = 1 if result == "yes" else 0
                sub = (m.get("yes_sub_title") or "").strip()
                question = f"{title} — {sub}" if sub and sub.lower() not in title.lower() else title
                et = m.get("event_ticker") or event_ticker
                url = (f"https://kalshi.com/markets/{et}/{ticker}"
                       if et else f"https://kalshi.com/markets/{ticker}")

                if _is_noise(question):
                    continue
                seen_tickers.add(ticker)
                if event_ticker:
                    seen_events.add(event_ticker)
                seen_title_keys.add(title_key)
                results.append(_record(
                    source="kalshi",
                    ident=ticker,
                    question=question,
                    outcome=outcome,
                    resolve_time=close_dt,
                    created_time=None,
                    market_url=url,
                    category=category,
                    market_prob=None,
                ))
                # One market per event — stop after the first kept market
                break
            if len(results) >= limit:
                break

        if not cursor:
            break
        time.sleep(0.3)

    print(f"  Kalshi: {len(results)} records")
    return results


# ── Manifold Markets ──────────────────────────────────────────────────────────

def _fetch_manifold(after_dt: datetime, limit: int) -> List[Dict[str, Any]]:
    """Pull resolved BINARY Manifold markets after `after_dt`."""
    MANIFOLD = "https://api.manifold.markets/v0/search-markets"
    results: List[Dict[str, Any]] = []
    offset = 0
    batch = 1000
    seen: set = set()

    print("  Manifold: fetching resolved binary markets…")
    while len(results) < limit:
        params: Dict[str, Any] = {
            "filter": "resolved",
            "contractType": "BINARY",
            "limit": batch,
            "sort": "newest",
            "offset": offset,
        }

        try:
            markets = _get(MANIFOLD, params=params)
        except Exception as e:
            print(f"  Manifold error: {e}")
            break

        if not isinstance(markets, list) or not markets:
            break

        all_before_cutoff = True
        for m in markets:
            mid = m.get("id") or m.get("slug") or ""
            if not mid or mid in seen:
                continue
            resolve_ts = m.get("resolutionTime")
            if not resolve_ts:
                continue
            try:
                resolve_dt = datetime.fromtimestamp(int(resolve_ts) / 1000, tz=timezone.utc)
            except (ValueError, OSError):
                continue
            if resolve_dt >= after_dt:
                all_before_cutoff = False

            if resolve_dt < after_dt:
                continue

            resolution = str(m.get("resolution") or "").strip().upper()
            if resolution not in ("YES", "NO"):
                continue

            outcome = 1 if resolution == "YES" else 0
            slug = m.get("slug") or mid
            question = m.get("question") or ""
            created_ts = m.get("createdTime")
            try:
                created_dt = datetime.fromtimestamp(int(created_ts) / 1000, tz=timezone.utc) if created_ts else None
            except (ValueError, OSError):
                created_dt = None
            market_prob = _to_float(m.get("resolutionProbability") or m.get("probability"))

            if _is_noise(question):
                continue
            seen.add(mid)
            results.append(_record(
                source="manifold",
                ident=slug,
                question=question,
                outcome=outcome,
                resolve_time=resolve_dt,
                created_time=created_dt,
                market_url=f"https://manifold.markets/{m.get('creatorUsername', 'unknown')}/{slug}",
                category=m.get("category") or (m.get("groups") or [None])[0] or None,
                market_prob=round(market_prob, 4) if market_prob is not None else None,
            ))
            if len(results) >= limit:
                break

        if all_before_cutoff:
            break
        offset += len(markets)
        if len(markets) < batch:
            break
        time.sleep(0.2)

    print(f"  Manifold: {len(results)} records")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def build(
    output_path: Path,
    sources: List[str],
    after: str = _DEFAULT_AFTER,
    limit_per_source: int = 2000,
) -> List[Dict[str, Any]]:
    after_dt = _parse_dt(after)
    if not after_dt:
        raise ValueError(f"Cannot parse --after date: {after!r}")

    print(f"Building live KG dataset (sources={sources}, after={after})")

    all_records: List[Dict[str, Any]] = []
    if "polymarket" in sources:
        all_records.extend(_fetch_polymarket(after_dt, limit_per_source))
    if "kalshi" in sources:
        all_records.extend(_fetch_kalshi(after_dt, limit_per_source))
    if "manifold" in sources:
        all_records.extend(_fetch_manifold(after_dt, limit_per_source))

    # Sort by resolve_time descending (newest first)
    all_records.sort(key=lambda r: r["resolve_time"], reverse=True)

    from collections import Counter
    domains = Counter(r["domain"] for r in all_records)
    sources_ct = Counter(r["source"] for r in all_records)
    yes_rate = sum(1 for r in all_records if r["outcome"] == 1) / max(1, len(all_records))

    print(f"\nTotal records: {len(all_records)}")
    print(f"Yes-rate: {yes_rate:.1%}")
    print(f"By source: {dict(sources_ct)}")
    print("\nDomain distribution:")
    for d, n in domains.most_common():
        yr = sum(1 for r in all_records if r["domain"] == d and r["outcome"] == 1)
        print(f"  {d:20s} {n:5d}  yes-rate={100*yr//n}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved → {output_path}")
    return all_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--after", default=_DEFAULT_AFTER,
                        help="Only include markets resolved after this date (YYYY-MM-DD)")
    parser.add_argument("--sources", nargs="+",
                        choices=["polymarket", "kalshi", "manifold"],
                        default=["polymarket", "kalshi", "manifold"])
    parser.add_argument("--limit", type=int, default=50000,
                        help="Max records per source (default 50k — fetch all after --after date)")
    args = parser.parse_args()
    build(Path(args.output), args.sources, args.after, args.limit)


if __name__ == "__main__":
    main()
