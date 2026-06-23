#!/usr/bin/env python3
"""Fetch historical point-in-time fundamentals from SEC EDGAR XBRL API.

Each data point carries the actual `filed` date from the 10-Q/10-K —
no look-ahead bias. The ML pipeline uses `filed` as the "known-as-of" date
for walk-forward training/test splits.

Extracted metrics (quarterly + annual, 10-Q + 10-K):
  - EPS diluted          (EarningsPerShareDiluted)
  - Revenue              (Revenues / RevenueFromContractWithCustomer...)
  - Gross profit         (GrossProfit)
  - Net income           (NetIncomeLoss)
  - Operating income     (OperatingIncomeLoss)
  - Total assets         (Assets)
  - Total liabilities    (Liabilities)
  - Stockholders equity  (StockholdersEquity)
  - Long-term debt       (LongTermDebt)
  - Shares outstanding   (CommonStockSharesOutstanding)

Derived features computed at ML-feature-build time (not stored):
  - Trailing 12m EPS (sum last 4 quarterly EPS)
  - P/E = price / trailing_12m_EPS
  - Debt/equity = LongTermDebt / StockholdersEquity
  - ROE = NetIncomeLoss_trailing / StockholdersEquity
  - Revenue growth YoY (vs same quarter 1 year prior)
  - Gross margin = GrossProfit / Revenue

Usage:
    python scripts/stock_edgar.py
    python scripts/stock_edgar.py --resume
    python scripts/stock_edgar.py --tickers AAPL MSFT NVDA

Output: data/stock_edgar.duckdb  (table: facts)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

ROOT     = Path(__file__).parent.parent
PRICES_DB = str(ROOT / "data" / "stock_prices.duckdb")
EDGAR_DB  = ROOT / "data" / "stock_edgar.duckdb"

EDGAR_BASE      = "https://data.sec.gov"
EDGAR_WWW_BASE  = "https://www.sec.gov"
HEADERS         = {"User-Agent": "research pareel.amre@gmail.com", "Accept": "application/json"}

# XBRL concept → canonical name; try alternatives in order
CONCEPTS: dict[str, list[str]] = {
    "eps_diluted":     ["EarningsPerShareDiluted"],
    "revenue":         ["Revenues",
                        "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "SalesRevenueNet",
                        "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit":    ["GrossProfit"],
    "net_income":      ["NetIncomeLoss"],
    "operating_income":["OperatingIncomeLoss"],
    "assets":          ["Assets"],
    "liabilities":     ["Liabilities"],
    "equity":          ["StockholdersEquity",
                        "StockholdersEquityAttributableToParent"],
    "long_term_debt":  ["LongTermDebt", "LongTermDebtNoncurrent"],
    "shares_out":      ["CommonStockSharesOutstanding",
                        "EntityCommonStockSharesOutstanding"],
}


def _get(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2)
                continue
            if r.status_code in (404, 403):
                return {}
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                return {}
            time.sleep(2 ** attempt)
    return {}


def get_cik_map() -> dict[str, str]:
    """Return {ticker: CIK_padded} for all SEC-registered tickers."""
    d = _get(f"{EDGAR_WWW_BASE}/files/company_tickers.json")
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in d.values()}


def extract_concept(facts_gaap: dict, concept_names: list[str]) -> list[dict]:
    """
    Return raw filing records for the given concepts, merging all alternatives.
    Deduplicates on (period_end, fiscal_period, filed) so that revenue reported
    under different concept names in different eras doesn't double-count.
    """
    seen: set[tuple] = set()
    all_records: list[dict] = []
    matched_name = None
    for name in concept_names:
        concept = facts_gaap.get(name, {})
        for unit_key, records in concept.get("units", {}).items():
            filtered = [
                r for r in records
                if r.get("form") in ("10-Q", "10-K")
                and r.get("filed")
                and r.get("fp") in ("Q1", "Q2", "Q3", "Q4", "FY", "H1", "H2")
            ]
            for r in filtered:
                key = (r.get("end"), r.get("fp"), r.get("filed"))
                if key not in seen:
                    seen.add(key)
                    all_records.append(r)
                    matched_name = name
    return all_records, matched_name


def fetch_ticker_facts(cik: str) -> dict[str, list[dict]]:
    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    d = _get(url)
    facts_gaap = d.get("facts", {}).get("us-gaap", {})
    if not facts_gaap:
        return {}

    out = {}
    for metric, concept_names in CONCEPTS.items():
        records, matched = extract_concept(facts_gaap, concept_names)
        if records:
            out[metric] = records
    return out


def records_to_rows(ticker: str, metric: str, records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        try:
            rows.append({
                "ticker":      ticker,
                "metric":      metric,
                "filed":       r["filed"],           # point-in-time date
                "period_end":  r["end"],              # accounting period end
                "period_start": r.get("start"),
                "fiscal_period": r.get("fp"),
                "form":        r.get("form"),
                "value":       float(r["val"]),
            })
        except Exception:
            continue
    return rows


def init_db(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            ticker       VARCHAR,
            metric       VARCHAR,
            filed        DATE,
            period_end   DATE,
            period_start DATE,
            fiscal_period VARCHAR,
            form         VARCHAR,
            value        DOUBLE,
            PRIMARY KEY (ticker, metric, filed, period_end, fiscal_period)
        )
    """)
    return con


def get_tickers_from_prices() -> list[str]:
    con = duckdb.connect(PRICES_DB, read_only=True)
    tickers = con.execute("SELECT DISTINCT ticker FROM prices").df()["ticker"].tolist()
    con.close()
    return sorted(tickers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tickers", nargs="+", help="Specific tickers to fetch")
    ap.add_argument("--db", default=str(EDGAR_DB))
    args = ap.parse_args()

    db_path = Path(args.db)
    con = init_db(db_path)

    tickers = args.tickers or get_tickers_from_prices()
    tickers = [t.upper() for t in tickers]

    if args.resume:
        done = set(con.execute("SELECT DISTINCT ticker FROM facts").df()["ticker"].tolist())
        tickers = [t for t in tickers if t not in done]
        print(f"Resuming: {len(tickers)} tickers remaining")

    print(f"Fetching EDGAR CIK map …")
    cik_map = get_cik_map()
    missing = [t for t in tickers if t not in cik_map]
    if missing:
        print(f"  No CIK for: {missing}")
    tickers = [t for t in tickers if t in cik_map]
    print(f"Fetching {len(tickers)} tickers from EDGAR …")

    total_rows = 0
    for i, ticker in enumerate(tickers):
        cik = cik_map[ticker]
        facts = fetch_ticker_facts(cik)
        if not facts:
            print(f"  {ticker}: no data")
            time.sleep(0.15)
            continue

        rows = []
        for metric, records in facts.items():
            rows.extend(records_to_rows(ticker, metric, records))

        if rows:
            df = pd.DataFrame(rows).drop_duplicates(
                subset=["ticker", "metric", "filed", "period_end", "fiscal_period"]
            )
            con.execute("DELETE FROM facts WHERE ticker = ?", [ticker])
            con.execute("INSERT OR IGNORE INTO facts SELECT * FROM df")
            total_rows += len(df)

        if (i + 1) % 10 == 0 or len(rows) == 0:
            n_tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM facts").fetchone()[0]
            print(f"  {i+1}/{len(tickers)}  {ticker}: {len(rows)} rows  "
                  f"(DB: {n_tickers} tickers, {total_rows:,} rows this run)")

        time.sleep(0.12)  # EDGAR rate limit: ~10 req/s

    n_total = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    n_tix   = con.execute("SELECT COUNT(DISTINCT ticker) FROM facts").fetchone()[0]
    con.close()
    print(f"\nDone: {n_tix} tickers, {n_total:,} total facts → {db_path}")


if __name__ == "__main__":
    main()
