#!/usr/bin/env python3
"""Collect timestamped headlines from GDELT for the fund's news-sentiment evaluation.

GDELT DOC 2.0 (free with attribution: "Data from the GDELT Project, gdeltproject.org")
serves article lists for the last three months. Each group's query is asked in 12-hour
windows so the 250-article cap per answer samples the whole day, one request every 6.5
seconds as GDELT asks, backing off when it answers 429. Output: one JSON line per
headline in OUT_DIR/<group>.jsonl with its GDELT seen time (UTC), title and domain.

Usage: fund_news_collect.py [--days 89] [--out data/fund_news]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import requests

API = "https://api.gdeltproject.org/api/v2/doc/doc"
GROUPS = {
    "india_equity": '(nifty OR sensex OR "indian stocks" OR "indian shares") sourcelang:english',
    "us_equity": '("wall street" OR "s&p 500" OR "stock market") sourcelang:english',
    "gold": '("gold price" OR "gold prices" OR bullion) sourcelang:english',
    "crude": '("crude oil" OR "oil prices" OR brent OR opec) sourcelang:english',
}
WINDOW_HOURS = 12
SPACING_S = 6.5
MAX_RETRIES = 6


def fetch(query: str, start: datetime, end: datetime, *, session: requests.Session) -> list[dict]:
    url = (f"{API}?query={quote(query)}&mode=artlist&format=json&maxrecords=250&sort=datedesc"
           f"&startdatetime={start:%Y%m%d%H%M%S}&enddatetime={end:%Y%m%d%H%M%S}")
    for attempt in range(MAX_RETRIES):
        resp = session.get(url, timeout=60, headers={"User-Agent": "Foresea/fund-news-eval"})
        if resp.status_code == 429:
            time.sleep(SPACING_S * (2 ** attempt))
            continue
        resp.raise_for_status()
        try:
            return list((resp.json() or {}).get("articles") or [])
        except ValueError:  # GDELT answers an empty window with an empty body
            return []
    raise RuntimeError(f"GDELT kept rate-limiting {start:%Y-%m-%d %H}h")


def seen_at(article: dict) -> str:
    """GDELT's seendate (20261108T171500Z) as ISO 8601 UTC."""
    raw = str(article.get("seendate") or "")
    return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()


def collect(days: int, out_dir: Path, *, now: datetime | None = None) -> dict[str, int]:
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    counts = {}
    for group, query in GROUPS.items():
        seen: set[str] = set()
        path = out_dir / f"{group}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            end = now
            start_limit = now - timedelta(days=days)
            while end > start_limit:
                start = max(start_limit, end - timedelta(hours=WINDOW_HOURS))
                try:
                    articles = fetch(query, start, end, session=session)
                except (requests.RequestException, RuntimeError) as exc:
                    print(f"  {group} {start:%Y-%m-%d %H}h skipped: {exc}", file=sys.stderr)
                    articles = []
                for a in articles:
                    key = a.get("url") or a.get("title")
                    if not key or key in seen or not a.get("title") or not a.get("seendate"):
                        continue
                    seen.add(key)
                    fh.write(json.dumps({"seen": seen_at(a), "title": a["title"], "domain": a.get("domain"),
                                         "language": a.get("language")}, ensure_ascii=False) + "\n")
                end = start
                time.sleep(SPACING_S)
        counts[group] = len(seen)
        print(f"{group}: {len(seen)} headlines")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=89)
    parser.add_argument("--out", type=Path, default=Path("data/fund_news"))
    args = parser.parse_args()
    print(json.dumps(collect(args.days, args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
