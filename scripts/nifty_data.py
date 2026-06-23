#!/usr/bin/env python3
"""Fetch Nifty 100 constituents + historical prices → DuckDB.

Constituent list pulled from NSE India API (falls back to hardcoded list).
Prices via yfinance with .NS suffix (NSE).

Usage:
    python scripts/nifty_data.py
    python scripts/nifty_data.py --start 2010-01-01

Output: data/nifty_prices.duckdb  (table: prices)
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import duckdb
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "nifty_prices.duckdb"

# Nifty 100 = Nifty 50 + Nifty Next 50 (stable list, updated occasionally)
NIFTY_100 = [
    # Nifty 50
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFY","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","HCLTECH","BAJFINANCE","MARUTI","AXISBANK","TITAN","ASIANPAINT",
    "SUNPHARMA","ULTRACEMCO","NESTLEIND","WIPRO","ONGC","NTPC","POWERGRID","MM",
    "JSWSTEEL","TATASTEEL","TECHM","ADANIENT","ADANIPORTS","COALINDIA","TATAMOTORS",
    "BAJAJFINSV","BAJAJAUTO","DIVISLAB","CIPLA","DRREDDY","EICHERMOT","GRASIM",
    "HEROMOTOCO","INDUSINDBK","BRITANNIA","HDFCLIFE","SBILIFE","TATACONSUM",
    "APOLLOHOSP","BPCL","IOC","SHRIRAMFIN","LTIM",
    # Nifty Next 50
    "ADANIGREEN","ADANIPOWER","AMBUJACEM","ATGL","AWL","BAJAJHLDNG","BANKBARODA",
    "BEL","BERGEPAINT","BOSCHLTD","CANBK","CHOLAFIN","COLPAL","DABUR","DLF",
    "GAIL","GODREJCP","GODREJPROP","HAL","HAVELLS","HINDALCO","HINDPETRO","ICICIPRULI",
    "INDUSTOWER","IRCTC","JINDALSTEL","JSWENERGY","LICI","LUPIN","MARICO","MUTHOOTFIN",
    "NAUKRI","OBEROIRLTY","OFSS","PAGEIND","PIDILITIND","PNB","RECLTD","SAIL",
    "SIEMENS","SRF","TATAPOWER","TORNTPHARM","TRENT","TVSMOTOR","UPL","VEDL",
    "VOLTAS","ZOMATO","ZYDUSLIFE",
]


def fetch_nse_constituents() -> list[str]:
    """Try to fetch live Nifty 100 list from NSE API."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com/", headers=headers, timeout=10)
        r = s.get(
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100",
            headers=headers, timeout=15,
        )
        if r.status_code == 200:
            data = r.json().get("data", [])
            tickers = [d["symbol"] for d in data if d.get("symbol") and d["symbol"] != "NIFTY 100"]
            if len(tickers) >= 90:
                print(f"  Live NSE list: {len(tickers)} constituents")
                return tickers
    except Exception:
        pass
    print(f"  Using hardcoded list: {len(NIFTY_100)} constituents")
    return NIFTY_100


def init_db(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker  VARCHAR,
            date    DATE,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  BIGINT,
            PRIMARY KEY (ticker, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS constituents (
            ticker  VARCHAR PRIMARY KEY,
            ns_ticker VARCHAR,
            fetched_date DATE
        )
    """)
    return con


def fetch_ticker(symbol: str, start: str) -> pd.DataFrame:
    ns = symbol + ".NS"
    try:
        df = yf.download(ns, start=start, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if c[0] != "Ticker" else "ticker" for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "date"})
        df["ticker"] = symbol
        df["date"]   = pd.to_datetime(df["date"]).dt.date
        df["volume"] = df["volume"].fillna(0).astype(int)
        return df[["ticker", "date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    except Exception:
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    db_path = Path(args.db)
    con = init_db(db_path)

    print("Fetching Nifty 100 constituent list …")
    tickers = fetch_nse_constituents()

    if args.resume:
        done = set(con.execute("SELECT DISTINCT ticker FROM prices").df()["ticker"].tolist())
        tickers = [t for t in tickers if t not in done]
        print(f"  {len(tickers)} remaining after resume")

    total_rows = 0
    for i, ticker in enumerate(tickers):
        df = fetch_ticker(ticker, args.start)
        if df.empty:
            print(f"  {ticker}: no data")
        else:
            con.execute("DELETE FROM prices WHERE ticker = ?", [ticker])
            con.execute("INSERT OR IGNORE INTO prices SELECT * FROM df")
            total_rows += len(df)

        # Store constituent record
        ns_ticker = ticker + ".NS"
        cdf = pd.DataFrame([{"ticker": ticker, "ns_ticker": ns_ticker,
                              "fetched_date": pd.Timestamp.now().date()}])
        con.execute("DELETE FROM constituents WHERE ticker = ?", [ticker])
        con.execute("INSERT INTO constituents SELECT * FROM cdf")

        if (i + 1) % 10 == 0:
            n_tix = con.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
            print(f"  {i+1}/{len(tickers)}  {ticker}: {len(df)} rows  (DB: {n_tix} tickers)")

        time.sleep(0.3)

    n_tix = con.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
    n_rows = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    earliest = con.execute("SELECT MIN(date) FROM prices").fetchone()[0]
    con.close()
    print(f"\nDone: {n_tix} tickers, {n_rows:,} rows, from {earliest} → {db_path}")


if __name__ == "__main__":
    main()
