"""Build the Foresea knowledge-graph dataset.

Sources:
  metaculus  — local FRS-format JSON with news articles; articles filtered to
               strictly pre-resolution dates (no look-ahead bias)
  kalshi     — live Kalshi API (settled events after --after date)
  polymarket — live Polymarket Gamma API (closed markets after --after date)
  manifold   — live Manifold Markets API (resolved BINARY after --after date)

All records share a common schema. Only Metaculus records carry articles.

Usage:
    python scripts/build_kg_dataset.py                        # all sources
    python scripts/build_kg_dataset.py --sources metaculus    # Metaculus only
    python scripts/build_kg_dataset.py --sources kalshi manifold
    python scripts/build_kg_dataset.py --min-created 2024-12-01 --min-articles 1

Output schema (data/kg_dataset.json):
    {
        "id":                str,      # "{source}:{ident}"
        "source":            str,      # metaculus | kalshi | polymarket | manifold
        "question":          str,
        "outcome":           int,      # 1=yes  0=no
        "resolve_time":      str,      # ISO-8601
        "created_time":      str|null,
        "horizon_days":      float|null,
        "market_probability":float|null,  # pre-resolution market price (live sources)
        "domain":            str,      # entity_tagger domain label
        "entities":          list[str],
        "category":          str,      # original source category
        "market_url":        str,
        "n_articles":        int,      # 0 for non-Metaculus
        "articles":          list,     # non-empty only for Metaculus
    }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from analyzing_llm_rationale.entity_tagger import tag_question  # noqa: E402

_DEFAULT_INPUT = _REPO / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
_DEFAULT_OUTPUT = _REPO / "data" / "kg_dataset.json"
_DEFAULT_AFTER = "2024-12-01"
_HEADERS = {"User-Agent": "foresea-kg-builder/1.0"}
_HTTP_TIMEOUT = 15


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get(url: str, params: Optional[Dict] = None) -> Any:
    import requests
    r = requests.get(url, params=params, headers=_HEADERS, timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _parse_dt(s: Any) -> Optional[datetime]:
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


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:
        return ""


# ── Noise filter for live markets ─────────────────────────────────────────────

_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in [
    r'\b\d+\s*min(?:ute)?\b.*\$',                           # "15 min · $1.15"
    r'\btarget\s*(?:price|level)\s*[:$]',                   # "Target Price: $X"
    r'·\s*\$[\d,]+',                                        # "· $1.1541"
    r'\bprice\s+(?:above|below|hits?|reaches?)\s+\$',
    r'\bover\s+\d+(?:\.\d+)?\s+(?:runs?|points?|goals?|assists?|rebounds?|hits?)',
    r'\bunder\s+\d+(?:\.\d+)?\s+(?:runs?|points?|goals?|assists?|rebounds?|hits?)',
    r'\b(?:first|last)\s+\d+\s+innings?\b',
    r'\b(?:moneyline|spread|handicap)\b',
    r'\b(?:btc|eth|sol|xrp|bnb|ada|doge)\b.*\b(?:above|below|hits?|over|under)\b.*\$[\d,]+',
    r'\$[\d,]+\s+(?:target|level|strike)',
]]


def _is_noise(q: str) -> bool:
    return len(q) < 25 or any(p.search(q) for p in _NOISE_RE)


# ── Common record constructor ─────────────────────────────────────────────────

def _make_record(
    source: str,
    ident: str,
    question: str,
    outcome: int,
    resolve_time: datetime,
    created_time: Optional[datetime],
    market_url: str,
    category: Optional[str] = None,
    market_prob: Optional[float] = None,
    articles: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    tags = tag_question(question, category)
    horizon_days: Optional[float] = None
    if created_time and resolve_time:
        horizon_days = round((resolve_time - created_time).total_seconds() / 86400, 1)
    arts = articles or []
    return {
        "id":                 f"{source}:{ident}",
        "source":             source,
        "question":           question.strip(),
        "outcome":            outcome,
        "resolve_time":       resolve_time.isoformat(),
        "created_time":       created_time.isoformat() if created_time else None,
        "horizon_days":       horizon_days,
        "market_probability": market_prob,
        "domain":             tags["domain"],
        "entities":           tags["entities"],
        "category":           category or "",
        "market_url":         market_url,
        "n_articles":         len(arts),
        "articles":           arts,
    }


# ── Metaculus ─────────────────────────────────────────────────────────────────

def _fetch_metaculus(
    input_path: Path,
    min_articles: int,
    created_after: Optional[datetime],
) -> List[Dict[str, Any]]:
    print(f"  Metaculus: reading {input_path.name} …")
    with open(input_path) as f:
        raw = json.load(f)
    print(f"    {len(raw)} raw records")

    results: List[Dict[str, Any]] = []
    skipped = Counter()

    for i, rec in enumerate(raw):
        answer = (rec.get("answer") or "").strip().lower()
        if answer == "yes":
            outcome = 1
        elif answer == "no":
            outcome = 0
        else:
            skipped["no_outcome"] += 1
            continue

        resolve_time = _parse_dt(rec.get("resolve_time"))
        if not resolve_time:
            skipped["no_resolve"] += 1
            continue

        articles = []
        for art in rec.get("news_articles") or []:
            pub = _parse_dt(art.get("publish_date"))
            if pub and pub < resolve_time:
                articles.append({
                    "title":        (art.get("title") or "").strip(),
                    "url":          (art.get("url") or "").strip(),
                    "publish_date": art["publish_date"],
                    "text":         (art.get("text") or "").strip(),
                    "summary":      (art.get("summary") or art.get("summary_llm") or "").strip(),
                    "source":       _host(art.get("url") or ""),
                })

        if len(articles) < min_articles:
            skipped["min_articles"] += 1
            continue

        created_time = _parse_dt(rec.get("created_time") or rec.get("publish_time"))
        if created_after and (not created_time or created_time < created_after):
            skipped["too_old"] += 1
            continue

        question = (rec.get("question") or "").strip()
        categories = [c for c in (rec.get("categories") or []) if isinstance(c, str)]
        qid = rec.get("id", i)

        results.append(_make_record(
            source="metaculus",
            ident=str(qid),
            question=question,
            outcome=outcome,
            resolve_time=resolve_time,
            created_time=created_time,
            market_url=f"https://www.metaculus.com/questions/{qid}/",
            category=categories[0] if categories else None,
            articles=articles,
        ))

    print(f"    kept {len(results)}, skipped {dict(skipped)}")
    return results


# ── Polymarket ────────────────────────────────────────────────────────────────

def _fetch_polymarket(after_dt: datetime, limit: int) -> List[Dict[str, Any]]:
    GAMMA = "https://gamma-api.polymarket.com/markets"
    results: List[Dict[str, Any]] = []
    offset = 0
    batch = min(500, limit * 4)
    seen: set = set()

    print("  Polymarket: fetching resolved markets…")
    while len(results) < limit:
        try:
            data = _get(GAMMA, params={"closed": "true", "limit": batch,
                                       "offset": offset, "order": "endDate",
                                       "ascending": "false"})
        except Exception as e:
            print(f"    error at offset {offset}: {e}")
            break

        markets = data if isinstance(data, list) else []
        if not markets:
            break

        for m in markets:
            slug = m.get("slug") or ""
            if not slug or slug in seen:
                continue
            end_dt = _parse_dt(m.get("endDate") or m.get("endDateIso"))
            if not end_dt or end_dt < after_dt:
                continue
            labels = [str(l).strip().lower() for l in _as_list(m.get("outcomes"))]
            prices = [_to_float(p) for p in _as_list(m.get("outcomePrices"))]
            if set(labels) != {"yes", "no"}:
                continue
            outcome: Optional[int] = None
            market_prob: Optional[float] = None
            for idx, label in enumerate(labels):
                p = prices[idx] if idx < len(prices) else None
                if label == "yes" and p is not None:
                    market_prob = p
                    outcome = 1 if p >= 0.5 else 0
            if outcome is None:
                continue
            question = m.get("question") or m.get("title") or ""
            if _is_noise(question):
                continue
            seen.add(slug)
            results.append(_make_record(
                source="polymarket", ident=slug, question=question, outcome=outcome,
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

    print(f"    {len(results)} records")
    return results


# ── Kalshi ────────────────────────────────────────────────────────────────────

_KALSHI_SKIP_CATS = {"Crypto", "Financials", "Sports", "Mentions", "Climate and Weather"}


def _fetch_kalshi(after_dt: datetime, limit: int) -> List[Dict[str, Any]]:
    KALSHI_EVENTS = "https://api.elections.kalshi.com/trade-api/v2/events"
    results: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    seen_tickers: set = set()
    seen_events: set = set()
    seen_title_keys: set = set()

    print("  Kalshi: fetching settled markets…")
    while len(results) < limit:
        params: Dict[str, Any] = {"status": "settled", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            data = _get(KALSHI_EVENTS, params=params)
        except Exception as e:
            print(f"    error: {e}")
            break

        events = data.get("events", []) if isinstance(data, dict) else []
        cursor = data.get("cursor") if isinstance(data, dict) else None
        if not events:
            break

        for event in events:
            title = event.get("title") or ""
            category = event.get("category")
            if category in _KALSHI_SKIP_CATS:
                continue
            event_ticker = (event.get("event_ticker") or "").strip().upper()
            if event_ticker and event_ticker in seen_events:
                continue
            # Collapse "which X wins this week" patterns that spawn many event_tickers
            title_key = re.sub(r'\s+', ' ', title.lower().strip())
            title_key = re.sub(r'\b(this week|today|tonight|on \w+ \d+[,\s]|\d{4})\b.*$',
                               '', title_key).strip()
            if title_key in seen_title_keys:
                continue

            for m in event.get("markets", []) or []:
                if m.get("mve_collection_ticker"):
                    continue
                ticker = (m.get("ticker") or "").strip().upper()
                if not ticker or ticker in seen_tickers:
                    continue
                if str(m.get("status") or "").lower() not in ("settled", "finalized"):
                    continue
                result = str(m.get("result") or "").lower()
                if result not in ("yes", "no"):
                    continue
                close_dt = _parse_dt(m.get("close_time"))
                if not close_dt or close_dt < after_dt:
                    continue

                sub = (m.get("yes_sub_title") or "").strip()
                question = f"{title} — {sub}" if sub and sub.lower() not in title.lower() else title
                et = m.get("event_ticker") or event_ticker
                url = (f"https://kalshi.com/markets/{et}/{ticker}" if et
                       else f"https://kalshi.com/markets/{ticker}")

                if _is_noise(question):
                    continue
                seen_tickers.add(ticker)
                if event_ticker:
                    seen_events.add(event_ticker)
                seen_title_keys.add(title_key)
                results.append(_make_record(
                    source="kalshi", ident=ticker, question=question,
                    outcome=1 if result == "yes" else 0,
                    resolve_time=close_dt, created_time=None,
                    market_url=url, category=category,
                ))
                break  # one market per event

            if len(results) >= limit:
                break

        if not cursor:
            break
        time.sleep(0.3)

    print(f"    {len(results)} records")
    return results


# ── Manifold Markets ──────────────────────────────────────────────────────────

def _fetch_manifold(after_dt: datetime, limit: int) -> List[Dict[str, Any]]:
    MANIFOLD = "https://api.manifold.markets/v0/search-markets"
    results: List[Dict[str, Any]] = []
    offset = 0
    batch = 1000
    seen: set = set()

    print("  Manifold: fetching resolved binary markets…")
    while len(results) < limit:
        try:
            markets = _get(MANIFOLD, params={"filter": "resolved", "contractType": "BINARY",
                                             "limit": batch, "sort": "newest", "offset": offset})
        except Exception as e:
            print(f"    error: {e}")
            break

        if not isinstance(markets, list) or not markets:
            break

        all_before = True
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
                all_before = False
            if resolve_dt < after_dt:
                continue
            resolution = str(m.get("resolution") or "").upper()
            if resolution not in ("YES", "NO"):
                continue
            question = m.get("question") or ""
            if _is_noise(question):
                continue
            slug = m.get("slug") or mid
            created_ts = m.get("createdTime")
            try:
                created_dt = datetime.fromtimestamp(int(created_ts) / 1000, tz=timezone.utc) if created_ts else None
            except (ValueError, OSError):
                created_dt = None
            market_prob = _to_float(m.get("resolutionProbability") or m.get("probability"))
            seen.add(mid)
            results.append(_make_record(
                source="manifold", ident=slug, question=question,
                outcome=1 if resolution == "YES" else 0,
                resolve_time=resolve_dt, created_time=created_dt,
                market_url=f"https://manifold.markets/{m.get('creatorUsername','unknown')}/{slug}",
                category=m.get("category") or (m.get("groups") or [None])[0] or None,
                market_prob=round(market_prob, 4) if market_prob is not None else None,
            ))
            if len(results) >= limit:
                break

        if all_before:
            break
        offset += len(markets)
        if len(markets) < batch:
            break
        time.sleep(0.2)

    print(f"    {len(results)} records")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def build(
    output_path: Path,
    sources: List[str],
    after: str = _DEFAULT_AFTER,
    limit_per_source: int = 50_000,
    metaculus_input: Optional[Path] = None,
    min_articles: int = 0,
    created_after: Optional[str] = None,
) -> List[Dict[str, Any]]:
    after_dt = _parse_dt(after)
    if not after_dt:
        raise ValueError(f"Cannot parse --after date: {after!r}")
    created_after_dt = _parse_dt(created_after) if created_after else None

    print(f"Building KG dataset  sources={sources}  after={after}")

    all_records: List[Dict[str, Any]] = []

    if "metaculus" in sources:
        inp = metaculus_input or _DEFAULT_INPUT
        all_records.extend(_fetch_metaculus(inp, min_articles, created_after_dt))
    if "polymarket" in sources:
        all_records.extend(_fetch_polymarket(after_dt, limit_per_source))
    if "kalshi" in sources:
        all_records.extend(_fetch_kalshi(after_dt, limit_per_source))
    if "manifold" in sources:
        all_records.extend(_fetch_manifold(after_dt, limit_per_source))

    # Dedup by id, preserving order
    seen: set = set()
    deduped = []
    for r in all_records:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)
    deduped.sort(key=lambda r: r["resolve_time"] or "")

    sources_ct = Counter(r["source"] for r in deduped)
    domains = Counter(r["domain"] for r in deduped)
    yes_rate = sum(1 for r in deduped if r["outcome"] == 1) / max(1, len(deduped))

    print(f"\nTotal: {len(deduped)} records  yes-rate={yes_rate:.1%}")
    print(f"By source: {dict(sources_ct)}")
    print("\nDomain distribution:")
    for d, n in domains.most_common():
        yr = sum(1 for r in deduped if r["domain"] == d and r["outcome"] == 1)
        print(f"  {d:20s} {n:6d}  yes-rate={100*yr//n}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"\nSaved → {output_path}")
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--sources", nargs="+",
                        choices=["metaculus", "polymarket", "kalshi", "manifold"],
                        default=["metaculus", "polymarket", "kalshi", "manifold"])
    parser.add_argument("--after", default=_DEFAULT_AFTER,
                        help="Only include markets resolved/created after this date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=50_000,
                        help="Max records per live source (default 50k)")
    parser.add_argument("--metaculus-input", default=None,
                        help="Path to the Metaculus FRS JSON file")
    parser.add_argument("--min-articles", type=int, default=0,
                        help="Metaculus: drop records with fewer than N pre-resolution articles")
    parser.add_argument("--min-created", default=None,
                        help="Metaculus: only include questions created on or after YYYY-MM-DD")
    args = parser.parse_args()
    build(
        output_path=Path(args.output),
        sources=args.sources,
        after=args.after,
        limit_per_source=args.limit,
        metaculus_input=Path(args.metaculus_input) if args.metaculus_input else None,
        min_articles=args.min_articles,
        created_after=args.min_created,
    )


if __name__ == "__main__":
    main()
