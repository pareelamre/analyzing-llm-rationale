#!/usr/bin/env python3
"""Fetch Kalshi resolved binary markets + daily candlestick history → DuckDB.

Uses the public Kalshi Trade API v2 (no auth required for market data).
Seeds from tickers already in track_record_store.duckdb, then paginates
settled markets for more coverage.

Usage:
    python scripts/kalshi_data.py
    python scripts/kalshi_data.py --max-markets 500 --resume

Output: data/kalshi.duckdb  (tables: markets, candlesticks)
"""
from __future__ import annotations

import argparse
import json
import time
import datetime
from pathlib import Path

import duckdb
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "kalshi.duckdb"
TRACK_DB = ROOT / "data" / "track_record_store.duckdb"

BASE = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"Accept": "application/json", "User-Agent": "research/1.0"}


def _get(path: str, params: dict = {}) -> dict:
    for attempt in range(4):
        try:
            r = requests.get(BASE + path, params=params, headers=HEADERS, timeout=20)
            if r.status_code in (429,):
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (404, 422):
                return {}
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            return {}
        except Exception:
            if attempt == 3:
                return {}
            time.sleep(2 ** attempt)
    return {}


def lookup_market(ticker: str) -> dict | None:
    """Look up a market by ticker; return market dict or None."""
    d = _get(f"/markets/{ticker}")
    return d.get("market") or (d if "ticker" in d else None)


def fetch_candlesticks(event_ticker: str, ticker: str,
                       start: datetime.datetime, end: datetime.datetime) -> list[dict]:
    """Fetch daily candlesticks for a market."""
    d = _get(
        f"/series/{event_ticker}/markets/{ticker}/candlesticks",
        {"start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()),
         "period_interval": 1440},
    )
    return d.get("candlesticks", [])


def parse_candles(ticker: str, sticks: list[dict]) -> list[dict]:
    rows = []
    for s in sticks:
        try:
            rows.append({
                "ticker": ticker,
                "ts": s["end_period_ts"],
                "open":   float(s["price"]["open_dollars"]),
                "high":   float(s["price"]["high_dollars"]),
                "low":    float(s["price"]["low_dollars"]),
                "close":  float(s["price"]["close_dollars"]),
                "yes_bid": float(s["yes_bid"]["close_dollars"]),
                "yes_ask": float(s["yes_ask"]["close_dollars"]),
                "volume":  float(s["volume_fp"]),
            })
        except Exception:
            continue
    return rows


def init_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            ticker       VARCHAR PRIMARY KEY,
            event_ticker VARCHAR,
            title        VARCHAR,
            category     VARCHAR,
            result       VARCHAR,
            open_time    VARCHAR,
            close_time   VARCHAR,
            volume_fp    DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS candlesticks (
            ticker   VARCHAR,
            ts       BIGINT,
            open     DOUBLE,
            high     DOUBLE,
            low      DOUBLE,
            close    DOUBLE,
            yes_bid  DOUBLE,
            yes_ask  DOUBLE,
            volume   DOUBLE,
            PRIMARY KEY (ticker, ts)
        )
    """)
    return con


def seed_tickers_from_track_record() -> list[str]:
    """Pull unique Kalshi tickers from the existing track record DB."""
    if not TRACK_DB.exists():
        return []
    try:
        con = duckdb.connect(str(TRACK_DB), read_only=True)
        rows = con.execute(
            "SELECT DISTINCT ident FROM forecast_snapshot WHERE platform='Kalshi'"
        ).df()["ident"].tolist()
        con.close()
        return rows
    except Exception:
        return []


def process_market(m: dict, con: duckdb.DuckDBPyConnection) -> int:
    """Fetch + store candlesticks for one market. Returns candle count."""
    ticker = m.get("ticker", "")
    event_ticker = m.get("event_ticker", "")
    if not ticker or not event_ticker:
        return 0

    open_t = m.get("open_time") or m.get("created_time", "")
    close_t = m.get("close_time") or m.get("expiration_time", "")
    if not open_t or not close_t:
        return 0

    try:
        start = datetime.datetime.fromisoformat(open_t.replace("Z", "+00:00")).replace(tzinfo=None)
        end   = datetime.datetime.fromisoformat(close_t.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return 0

    sticks = fetch_candlesticks(event_ticker, ticker, start, end)
    if not sticks:
        return 0

    rows = parse_candles(ticker, sticks)
    if not rows:
        return 0

    # Upsert market
    mdf = pd.DataFrame([{
        "ticker": ticker,
        "event_ticker": event_ticker,
        "title": m.get("title", ""),
        "category": m.get("category", ""),
        "result": m.get("result", ""),
        "open_time": open_t,
        "close_time": close_t,
        "volume_fp": float(m.get("volume_fp", 0) or 0),
    }])
    con.execute("DELETE FROM markets WHERE ticker = ?", [ticker])
    con.execute("INSERT INTO markets SELECT * FROM mdf")

    # Upsert candles
    cdf = pd.DataFrame(rows)
    con.execute("DELETE FROM candlesticks WHERE ticker = ?", [ticker])
    con.execute("INSERT OR IGNORE INTO candlesticks SELECT * FROM cdf")

    return len(rows)


def paginate_settled(max_markets: int, existing: set[str]) -> list[dict]:
    """Page through settled markets from the API."""
    markets = []
    cursor = None
    seen = set(existing)
    while len(markets) < max_markets:
        params = {"status": "settled", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        d = _get("/markets", params)
        batch = d.get("markets", [])
        if not batch:
            break
        for m in batch:
            t = m.get("ticker", "")
            if t and t not in seen:
                seen.add(t)
                markets.append(m)
        cursor = d.get("cursor")
        if not cursor:
            break
        time.sleep(0.1)
    return markets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-markets", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    db_path = Path(args.db)
    con = init_db(db_path)

    existing: set[str] = set()
    if args.resume:
        existing = set(con.execute("SELECT ticker FROM markets").df()["ticker"].tolist())
        print(f"Resuming: {len(existing)} markets already in DB")

    # 1. Seed from track record tickers
    seed = seed_tickers_from_track_record()
    print(f"Seeding {len(seed)} tickers from track record …")
    total_candles = 0
    for ticker in seed:
        if ticker in existing:
            continue
        m = lookup_market(ticker)
        if not m:
            continue
        n = process_market(m, con)
        if n:
            print(f"  {ticker}: {n} candles")
            total_candles += n
        time.sleep(0.1)

    # 2. Paginate settled markets for more coverage
    print(f"\nPaginating settled markets (target {args.max_markets}) …")
    settled = paginate_settled(args.max_markets, existing | {t for t in seed})
    print(f"Found {len(settled)} additional settled markets")
    done = 0
    for m in settled:
        n = process_market(m, con)
        total_candles += n
        done += 1
        if done % 50 == 0:
            n_mkt = con.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
            print(f"  {done}/{len(settled)} processed  ({n_mkt} markets in DB)")
        time.sleep(0.08)

    n_mkt = con.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    n_can = con.execute("SELECT COUNT(*) FROM candlesticks").fetchone()[0]
    con.close()
    print(f"\nDone. {n_mkt} markets, {n_can:,} candlesticks → {db_path}")


if __name__ == "__main__":
    main()
