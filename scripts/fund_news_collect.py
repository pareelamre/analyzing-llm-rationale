#!/usr/bin/env python3
"""Collect timestamped headlines from GDELT for the fund's news-sentiment evaluation.

GDELT DOC 2.0 (free with attribution: "Data from the GDELT Project, gdeltproject.org")
serves article lists for the last three months. Each group's query is asked in 12-hour
windows so the 250-article cap per answer samples the whole day, one request every 6.5
seconds as GDELT asks, backing off (at most a minute) when it answers 429. Output: one
JSON line per headline in OUT_DIR/<group>.jsonl with its GDELT seen time (UTC), title and
domain, and OUT_DIR/_coverage.json saying how many windows each group got. Collection stops
at --deadline-minutes and keeps what it has, so a slow GDELT shrinks the sample (which the
evaluation's minimum-sample check accounts for) instead of losing the run. The groups take
turns window by window, newest first, so a deadline leaves every group with recent data.

Usage: fund_news_collect.py [--days 89] [--out data/fund_news] [--deadline-minutes 150]
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
MAX_RETRIES = 4
MAX_BACKOFF_S = 60.0


def fetch(query: str, start: datetime, end: datetime, *, session: requests.Session) -> list[dict]:
    url = (f"{API}?query={quote(query)}&mode=artlist&format=json&maxrecords=250&sort=datedesc"
           f"&startdatetime={start:%Y%m%d%H%M%S}&enddatetime={end:%Y%m%d%H%M%S}")
    for attempt in range(MAX_RETRIES):
        resp = session.get(url, timeout=60, headers={"User-Agent": "Foresea/fund-news-eval"})
        if resp.status_code == 429:
            time.sleep(min(MAX_BACKOFF_S, SPACING_S * (2 ** attempt)))
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


def collect(days: int, out_dir: Path, *, now: datetime | None = None, deadline_minutes: float = 150.0,
            fetcher=fetch, sleep=time.sleep, clock=time.monotonic) -> dict:
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    stop_at = clock() + deadline_minutes * 60
    windows = []
    end = now
    while end > now - timedelta(days=days):
        start = max(now - timedelta(days=days), end - timedelta(hours=WINDOW_HOURS))
        windows.append((start, end))
        end = start
    seen = {group: set() for group in GROUPS}
    done = {group: 0 for group in GROUPS}
    handles = {group: (out_dir / f"{group}.jsonl").open("w", encoding="utf-8") for group in GROUPS}
    deadline_hit = False
    try:
        for start, end in windows:  # newest first, groups taking turns
            if clock() > stop_at:
                deadline_hit = True
                break
            for group, query in GROUPS.items():
                try:
                    articles = fetcher(query, start, end, session=session)
                    done[group] += 1
                except (requests.RequestException, RuntimeError) as exc:
                    print(f"  {group} {start:%Y-%m-%d %H}h skipped: {exc}", file=sys.stderr, flush=True)
                    articles = []
                for a in articles:
                    key = a.get("url") or a.get("title")
                    if not key or key in seen[group] or not a.get("title") or not a.get("seendate"):
                        continue
                    seen[group].add(key)
                    handles[group].write(json.dumps({"seen": seen_at(a), "title": a["title"], "domain": a.get("domain"),
                                                     "language": a.get("language")}, ensure_ascii=False) + "\n")
                sleep(SPACING_S)
            print(f"{start:%Y-%m-%d %H}h: " + ", ".join(f"{g} {len(s)}" for g, s in seen.items()), flush=True)
    finally:
        for fh in handles.values():
            fh.close()
    coverage = {"windows_planned": len(windows), "windows_done": done, "deadline_hit": deadline_hit,
                "headlines": {g: len(s) for g, s in seen.items()},
                "oldest_window": windows[min(len(windows), max(done.values(), default=0)) - 1][0].isoformat()
                if max(done.values(), default=0) else None}
    (out_dir / "_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=89)
    parser.add_argument("--out", type=Path, default=Path("data/fund_news"))
    parser.add_argument("--deadline-minutes", type=float, default=150.0)
    args = parser.parse_args()
    print(json.dumps(collect(args.days, args.out, deadline_minutes=args.deadline_minutes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
