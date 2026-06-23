#!/usr/bin/env python3
"""Fetch resolved Polymarket binary markets + daily price histories → DuckDB.

Pulls archived/closed binary markets from the Polymarket Gamma API and
CLOB price history API, filters to markets with sufficient volume and price
history, stores in data/polymarket.duckdb.

Usage:
    python scripts/pm_data.py
    python scripts/pm_data.py --max-markets 2000 --min-volume 5000
    python scripts/pm_data.py --resume   # skip markets already in DB

Output: data/polymarket.duckdb  (tables: markets, price_history)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "polymarket.duckdb"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
HEADERS = {"User-Agent": "research/1.0", "Accept": "application/json"}


def _gamma_get(path: str, params: dict) -> list | dict | None:
    """Returns None on 422/404 (end of results), raises on other errors."""
    url = GAMMA_BASE + path
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code in (404, 422):
                return None  # treat as end of pagination
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            return None
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return None


def _clob_price_history(token_id: str, fidelity: int = 1440) -> list[dict]:
    """Daily price history (fidelity=1440 min). Returns list of {t, p}."""
    url = CLOB_BASE + "/prices-history"
    params = {"market": token_id, "interval": "all", "fidelity": fidelity}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json().get("history", [])
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2 ** attempt)
    return []


def fetch_markets(max_markets: int, min_volume: float) -> list[dict]:
    """Page through the Gamma API to collect resolved binary markets."""
    markets: list[dict] = []
    limit = 100
    offset = 0
    seen_ids: set[str] = set()

    print(f"Fetching up to {max_markets} markets (min volume ${min_volume:,.0f}) …")
    while len(markets) < max_markets:
        batch = _gamma_get("/markets", {
            "limit": limit,
            "offset": offset,
            "closed": "true",
            "order": "volume",
            "ascending": "false",
        })
        if batch is None or not batch:
            break
        for m in batch:
            if m["id"] in seen_ids:
                continue
            seen_ids.add(m["id"])
            # Binary only: outcomes is ["Yes","No"]
            try:
                outcomes = json.loads(m.get("outcomes", "[]"))
            except Exception:
                outcomes = []
            if set(outcomes) != {"Yes", "No"}:
                continue
            vol = float(m.get("volumeNum", 0) or 0)
            if vol < min_volume:
                continue
            # Parse outcome prices
            try:
                prices = json.loads(m.get("outcomePrices", "[0,0]"))
                yes_price = float(prices[outcomes.index("Yes")])
            except Exception:
                yes_price = float("nan")
            # Resolution: use price convergence (markets settle near 0 or 1)
            if yes_price >= 0.95:
                outcome = "yes"
            elif yes_price <= 0.05:
                outcome = "no"
            else:
                outcome = "unresolved"
            markets.append({
                "id": m["id"],
                "question": m.get("question", ""),
                "category": m.get("category", ""),
                "volume": vol,
                "liquidity": float(m.get("liquidityNum", 0) or 0),
                "created_at": m.get("createdAt"),
                "end_date": m.get("endDateIso") or m.get("endDate"),
                "closed_time": m.get("closedTime"),
                "outcome": outcome,
                "yes_price_final": yes_price,
                "token_id_yes": None,
                "spread": float(m.get("spread", 0) or 0),
            })
            # Extract YES token ID for price history
            tok_str = m.get("clobTokenIds", "")
            if tok_str:
                try:
                    toks = json.loads(tok_str)
                    yes_idx = outcomes.index("Yes")
                    markets[-1]["token_id_yes"] = toks[yes_idx] if yes_idx < len(toks) else toks[0]
                except Exception:
                    pass

        offset += limit
        print(f"  {len(markets)} qualifying markets so far (offset={offset})")
        if len(batch) < limit:
            break
        time.sleep(0.2)

    return markets[:max_markets]


def fetch_price_histories(markets: list[dict], existing_ids: set[str]) -> list[dict]:
    """Fetch daily price history for each market."""
    rows: list[dict] = []
    total = len(markets)
    for i, m in enumerate(markets):
        if m["id"] in existing_ids:
            continue
        token = m.get("token_id_yes")
        if not token:
            continue
        hist = _clob_price_history(token)
        if len(hist) < 3:
            continue
        for pt in hist:
            rows.append({
                "market_id": m["id"],
                "ts": pt["t"],
                "price": pt["p"],
            })
        if (i + 1) % 50 == 0:
            print(f"  Price history: {i+1}/{total}")
        time.sleep(0.05)
    return rows


def save_to_db(markets: list[dict], price_rows: list[dict], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            id          VARCHAR PRIMARY KEY,
            question    VARCHAR,
            category    VARCHAR,
            volume      DOUBLE,
            liquidity   DOUBLE,
            created_at  VARCHAR,
            end_date    VARCHAR,
            closed_time VARCHAR,
            outcome     VARCHAR,
            yes_price_final DOUBLE,
            token_id_yes VARCHAR,
            spread      DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            market_id   VARCHAR,
            ts          BIGINT,
            price       DOUBLE
        )
    """)
    if markets:
        mdf = pd.DataFrame(markets)
        existing = set(con.execute("SELECT id FROM markets").df()["id"].tolist())
        new_markets = mdf[~mdf["id"].isin(existing)]
        if not new_markets.empty:
            con.execute("INSERT INTO markets SELECT * FROM new_markets")
    if price_rows:
        pdf = pd.DataFrame(price_rows)
        con.execute("INSERT INTO price_history SELECT * FROM pdf")
    n_markets = con.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    n_pts = con.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
    con.close()
    print(f"DB: {n_markets} markets, {n_pts:,} price points → {db_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-markets", type=int, default=3000)
    ap.add_argument("--min-volume", type=float, default=10_000,
                    help="Minimum market volume in USD")
    ap.add_argument("--resume", action="store_true",
                    help="Skip markets already in DB")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    db_path = Path(args.db)
    existing_ids: set[str] = set()
    if args.resume and db_path.exists():
        con = duckdb.connect(str(db_path), read_only=True)
        existing_ids = set(con.execute("SELECT id FROM markets").df()["id"].tolist())
        con.close()
        print(f"Resuming: {len(existing_ids)} markets already in DB")

    markets = fetch_markets(args.max_markets, args.min_volume)
    print(f"Fetched {len(markets)} qualifying markets")

    if not markets:
        print("No markets found.")
        return

    print("Fetching price histories …")
    price_rows = fetch_price_histories(markets, existing_ids)
    print(f"Fetched {len(price_rows):,} price points")

    save_to_db(markets, price_rows, db_path)


if __name__ == "__main__":
    main()
