#!/usr/bin/env python3
"""Download S&P 100 OHLCV history via yfinance and store in DuckDB.

Usage:
    python scripts/stock_data.py
    python scripts/stock_data.py --tickers AAPL MSFT --start 2015-01-01
    python scripts/stock_data.py --universe sp100 --start 2010-01-01

Output: data/stock_prices.duckdb  (table: prices)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "stock_prices.duckdb"

SP100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B",
    "UNH", "XOM", "JPM", "LLY", "V", "AVGO", "MA", "PG", "JNJ", "COST", "MRK",
    "HD", "ABBV", "CVX", "BAC", "CRM", "NFLX", "KO", "AMD", "WMT", "ACN",
    "PEP", "TMO", "LIN", "MCD", "CSCO", "ORCL", "ABT", "GE", "CAT", "TXN",
    "NOW", "AMGN", "DIS", "IBM", "GS", "PM", "NEE", "RTX", "INTU", "SPGI",
    "BKNG", "HON", "QCOM", "LOW", "T", "UNP", "MS", "BMY", "AXP", "AMAT",
    "VZ", "GILD", "ELV", "BLK", "SYK", "DE", "MDT", "ISRG", "ZTS", "ADI",
    "SCHW", "VRTX", "REGN", "CI", "LRCX", "ETN", "MO", "PLD", "TJX", "CB",
    "ADP", "FISV", "SO", "SHW", "TGT", "CL", "CME", "ICE", "EOG", "SNPS",
    "CDNS", "WM", "GD", "PH", "MCO", "KLAC", "HUM", "MMC", "F", "GM",
]

SECTOR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC",
]


def fetch_and_store(tickers: list[str], start: str, end: str | None, db_path: Path) -> None:
    print(f"Downloading {len(tickers)} tickers from {start} …", flush=True)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=True,
        threads=True,
    )
    if raw.empty:
        print("No data returned.", file=sys.stderr)
        sys.exit(1)

    # Flatten multi-level columns → long format
    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        for ticker in raw["Close"].columns:
            try:
                t = pd.DataFrame({
                    "ticker": ticker,
                    "date": raw.index,
                    "open": raw["Open"][ticker].values,
                    "high": raw["High"][ticker].values,
                    "low": raw["Low"][ticker].values,
                    "close": raw["Close"][ticker].values,
                    "volume": raw["Volume"][ticker].values,
                })
                frames.append(t)
            except Exception as e:
                print(f"  Skip {ticker}: {e}")
        if not frames:
            print("All tickers failed.", file=sys.stderr)
            sys.exit(1)
        df = pd.concat(frames, ignore_index=True)
    else:
        # Single ticker
        df = raw[["Close", "Open", "High", "Low", "Volume"]].copy()
        df.columns = ["close", "open", "high", "low", "volume"]
        df.index.name = "date"
        df = df.reset_index()
        df["ticker"] = tickers[0]

    df = df.dropna(subset=["close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["ticker", "date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker  VARCHAR,
            date    DATE,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  BIGINT
        )
    """)
    # Upsert: delete then insert for the tickers we just downloaded
    ticker_list = "', '".join(df["ticker"].unique())
    con.execute(f"DELETE FROM prices WHERE ticker IN ('{ticker_list}')")
    con.execute("INSERT INTO prices SELECT * FROM df")
    n = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    tickers_n = con.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
    con.close()
    print(f"Stored {n:,} rows for {tickers_n} tickers → {db_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", help="explicit ticker list")
    ap.add_argument("--universe", choices=["sp100", "etfs", "both"], default="sp100")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    if args.tickers:
        tickers = args.tickers
    elif args.universe == "sp100":
        tickers = SP100
    elif args.universe == "etfs":
        tickers = SECTOR_ETFS
    else:
        tickers = SP100 + SECTOR_ETFS

    fetch_and_store(tickers, args.start, args.end, Path(args.db))


if __name__ == "__main__":
    main()
