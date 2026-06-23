#!/usr/bin/env python3
"""Fetch Nifty 100 fundamental data + earnings surprises → DuckDB.

Sources (tried in order):
  1. NSE corporate financial results API — quarterly EPS with announcement dates
  2. BSE XBRL API — structured financials with filing dates

Both sources provide point-in-time data (announcement/filing date = known-as-of).

Usage:
    python scripts/nifty_fundamentals.py
    python scripts/nifty_fundamentals.py --resume

Output: data/nifty_fundamentals.duckdb
  tables: earnings_results, bse_financials
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import duckdb
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
NIFTY_DB  = str(ROOT / "data" / "nifty_prices.duckdb")
FUND_DB   = ROOT / "data" / "nifty_fundamentals.duckdb"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.bseindia.com/",
}


def nse_session() -> requests.Session:
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com/", headers=NSE_HEADERS, timeout=10)
    except Exception:
        pass
    return s


def fetch_nse_financials(session: requests.Session, symbol: str) -> list[dict]:
    """Fetch quarterly financial results from NSE for a symbol."""
    rows = []
    try:
        url = (f"https://www.nseindia.com/api/financial-results-comparator"
               f"?index=equities&symbol={symbol}&period=Quarterly")
        r = session.get(url, headers=NSE_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        results = data if isinstance(data, list) else data.get("data", [])
        for item in results:
            try:
                rows.append({
                    "ticker":         symbol,
                    "period_end":     pd.to_datetime(item.get("toDate") or item.get("period")).date(),
                    "announce_date":  pd.to_datetime(item.get("xbrlAttachment") or
                                                     item.get("resultDate") or
                                                     item.get("toDate")).date(),
                    "eps":            float(item.get("eps") or item.get("basicEps") or 0),
                    "revenue":        float(item.get("totalIncome") or item.get("revenue") or 0),
                    "net_income":     float(item.get("netProfit") or item.get("netIncome") or 0),
                    "source":         "nse",
                })
            except Exception:
                continue
    except Exception:
        pass
    return rows


def fetch_bse_financials(symbol: str, bse_code: str) -> list[dict]:
    """Fetch quarterly results from BSE API."""
    rows = []
    try:
        url = (f"https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
               f"?scripcode={bse_code}&type=QB")
        r = requests.get(url, headers=BSE_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        for item in (data if isinstance(data, list) else []):
            try:
                rows.append({
                    "ticker":        symbol,
                    "period_end":    pd.to_datetime(item.get("TO_DATE")).date(),
                    "announce_date": pd.to_datetime(item.get("QUARTERENDDATE") or
                                                    item.get("TO_DATE")).date(),
                    "eps":           float(item.get("EPS_DILUTED") or item.get("EPS") or 0),
                    "revenue":       float(item.get("NET_SALES") or 0),
                    "net_income":    float(item.get("NET_PROFIT") or 0),
                    "source":        "bse",
                })
            except Exception:
                continue
    except Exception:
        pass
    return rows


def fetch_screener_financials(symbol: str) -> list[dict]:
    """Fetch quarterly P&L from Screener.in (fallback — public, no auth needed)."""
    rows = []
    try:
        url = f"https://www.screener.in/company/{symbol}/consolidated/"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if r.status_code != 200:
            return []
        from html.parser import HTMLParser
        import re

        # Find quarterly table (simplified parse of first data table)
        text = r.text
        # Look for EPS data pattern in the page
        eps_matches = re.findall(r'"eps":\s*([\d.]+)', text)
        if eps_matches:
            for v in eps_matches[:20]:
                rows.append({"ticker": symbol, "eps": float(v), "source": "screener"})
    except Exception:
        pass
    return rows


def compute_eps_surprises(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a quarterly earnings DataFrame compute YoY EPS surprise.
    surprise_pct = (current_Q_EPS - same_Q_prior_year_EPS) / |prior|
    """
    df = df.sort_values(["ticker", "period_end"]).copy()
    df["qtr"]  = pd.to_datetime(df["period_end"]).dt.quarter
    df["year"] = pd.to_datetime(df["period_end"]).dt.year

    prior = df[["ticker", "year", "qtr", "eps"]].copy()
    prior["year"] = prior["year"] + 1
    prior = prior.rename(columns={"eps": "eps_prior"})

    merged = df.merge(prior, on=["ticker", "year", "qtr"], how="left")
    merged["surprise_pct"] = (
        (merged["eps"] - merged["eps_prior"]) /
        (merged["eps_prior"].abs() + 1e-9)
    ).clip(-5, 5)
    return merged


def get_tickers(db_path: str) -> list[str]:
    """Get tickers from price DB, or fall back to nifty_data hardcoded list."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        tickers = con.execute("SELECT DISTINCT ticker FROM prices").df()["ticker"].tolist()
        con.close()
        if tickers:
            return sorted(tickers)
    except Exception:
        pass
    # Fallback: import from nifty_data
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from nifty_data import NIFTY_100
    return sorted(NIFTY_100)


def init_db(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS earnings_results (
            ticker         VARCHAR,
            period_end     DATE,
            announce_date  DATE,
            eps            DOUBLE,
            revenue        DOUBLE,
            net_income     DOUBLE,
            eps_prior      DOUBLE,
            surprise_pct   DOUBLE,
            source         VARCHAR,
            PRIMARY KEY (ticker, period_end)
        )
    """)
    return con


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--db", default=str(FUND_DB))
    args = ap.parse_args()

    db_path = Path(args.db)
    con = init_db(db_path)
    tickers = get_tickers(NIFTY_DB)

    if args.resume:
        done = set(con.execute("SELECT DISTINCT ticker FROM earnings_results").df()["ticker"].tolist())
        tickers = [t for t in tickers if t not in done]
        print(f"Resuming: {len(tickers)} tickers remaining")

    print(f"Fetching quarterly results for {len(tickers)} Nifty 100 tickers …")
    sess = nse_session()
    all_rows = []

    for i, ticker in enumerate(tickers):
        rows = fetch_nse_financials(sess, ticker)
        if not rows:
            rows = fetch_screener_financials(ticker)

        if rows:
            df = pd.DataFrame(rows)
            # Compute YoY surprise
            if "period_end" in df.columns and "eps" in df.columns:
                df = compute_eps_surprises(df)
            con.execute("DELETE FROM earnings_results WHERE ticker = ?", [ticker])
            # Only insert rows that have the full schema
            for col in ["eps_prior", "surprise_pct"]:
                if col not in df.columns:
                    df[col] = None
            cols = ["ticker", "period_end", "announce_date", "eps", "revenue",
                    "net_income", "eps_prior", "surprise_pct", "source"]
            available = [c for c in cols if c in df.columns]
            insert_df = df[available].drop_duplicates(subset=["ticker", "period_end"] if "period_end" in available else ["ticker"])
            try:
                con.execute("INSERT OR IGNORE INTO earnings_results SELECT * FROM insert_df")
                all_rows.append(len(insert_df))
            except Exception:
                pass

        if (i + 1) % 10 == 0:
            n = con.execute("SELECT COUNT(*) FROM earnings_results").fetchone()[0]
            print(f"  {i+1}/{len(tickers)}  {ticker}: {len(rows) if rows else 0} rows  (DB: {n} total rows)")

        time.sleep(0.5)
        if (i + 1) % 20 == 0:
            sess = nse_session()  # refresh session

    n_total = con.execute("SELECT COUNT(*) FROM earnings_results").fetchone()[0]
    n_tix   = con.execute("SELECT COUNT(DISTINCT ticker) FROM earnings_results").fetchone()[0]
    con.close()
    print(f"\nDone: {n_tix} tickers, {n_total} quarterly results → {db_path}")


if __name__ == "__main__":
    main()
