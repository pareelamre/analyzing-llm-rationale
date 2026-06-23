#!/usr/bin/env python3
"""Fetch Nifty 100 fundamental data via yfinance (.NS suffix) → DuckDB.

Uses earnings_dates (EPS estimate vs actual = true analyst surprise),
quarterly_financials (revenue, margins), and info snapshot.

Usage:
    python scripts/nifty_yf_fundamentals.py
    python scripts/nifty_yf_fundamentals.py --resume

Output: data/nifty_fundamentals.duckdb
  tables: earnings_surprises, quarterly_financials, snapshot_ratios
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import duckdb
import pandas as pd
import yfinance as yf
from nifty_data import NIFTY_100

ROOT    = Path(__file__).parent.parent
FUND_DB = ROOT / "data" / "nifty_fundamentals.duckdb"


def init_db(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS earnings_surprises (
            ticker        VARCHAR,
            announce_date DATE,
            eps_estimate  DOUBLE,
            eps_actual    DOUBLE,
            surprise_pct  DOUBLE,
            PRIMARY KEY (ticker, announce_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS quarterly_financials (
            ticker        VARCHAR,
            period_end    DATE,
            revenue       DOUBLE,
            gross_profit  DOUBLE,
            net_income    DOUBLE,
            gross_margin  DOUBLE,
            net_margin    DOUBLE,
            PRIMARY KEY (ticker, period_end)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS snapshot_ratios (
            ticker         VARCHAR PRIMARY KEY,
            fetched_date   DATE,
            trailing_pe    DOUBLE,
            price_to_book  DOUBLE,
            roe            DOUBLE,
            revenue_growth DOUBLE,
            trailing_eps   DOUBLE,
            forward_eps    DOUBLE
        )
    """)
    return con


def fetch_earnings_surprises(ticker: str) -> pd.DataFrame:
    try:
        ed = yf.Ticker(ticker + ".NS").earnings_dates
        if ed is None or ed.empty:
            return pd.DataFrame()
        ed = ed.reset_index()
        date_col = ed.columns[0]
        ed = ed.rename(columns={
            date_col: "announce_date",
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
        })
        ed["announce_date"] = pd.to_datetime(ed["announce_date"]).dt.tz_localize(None).dt.date
        ed["eps_estimate"]  = pd.to_numeric(ed["eps_estimate"], errors="coerce")
        ed["eps_actual"]    = pd.to_numeric(ed["eps_actual"],   errors="coerce")
        ed = ed.dropna(subset=["eps_actual"])
        ed["surprise_pct"] = (
            (ed["eps_actual"] - ed["eps_estimate"]) /
            (ed["eps_estimate"].abs() + 1e-9)
        ).clip(-5, 5)
        ed["ticker"] = ticker
        return ed[["ticker", "announce_date", "eps_estimate", "eps_actual", "surprise_pct"]]
    except Exception:
        return pd.DataFrame()


def fetch_quarterly_financials(ticker: str) -> pd.DataFrame:
    rows = []
    try:
        qf = yf.Ticker(ticker + ".NS").quarterly_financials
        if qf is None or qf.empty:
            return pd.DataFrame()
        qf = qf.T
        for date, row in qf.iterrows():
            rev = row.get("Total Revenue") or row.get("Revenue")
            gp  = row.get("Gross Profit")
            ni  = row.get("Net Income")
            rows.append({
                "ticker":       ticker,
                "period_end":   pd.to_datetime(date).date(),
                "revenue":      float(rev) if rev is not None and pd.notna(rev) else None,
                "gross_profit": float(gp)  if gp  is not None and pd.notna(gp)  else None,
                "net_income":   float(ni)  if ni  is not None and pd.notna(ni)  else None,
                "gross_margin": float(gp / rev) if (rev and gp and float(rev) != 0
                                                     and pd.notna(rev) and pd.notna(gp)) else None,
                "net_margin":   float(ni / rev) if (rev and ni and float(rev) != 0
                                                     and pd.notna(rev) and pd.notna(ni)) else None,
            })
    except Exception:
        pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_snapshot(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker + ".NS").info
        return {
            "ticker":         ticker,
            "fetched_date":   pd.Timestamp.now().date(),
            "trailing_pe":    info.get("trailingPE"),
            "price_to_book":  info.get("priceToBook"),
            "roe":            info.get("returnOnEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "trailing_eps":   info.get("trailingEps"),
            "forward_eps":    info.get("forwardEps"),
        }
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--db", default=str(FUND_DB))
    args = ap.parse_args()

    db_path = Path(args.db)
    con = init_db(db_path)
    tickers = sorted(NIFTY_100)

    if args.resume:
        done = set(con.execute(
            "SELECT DISTINCT ticker FROM snapshot_ratios").df()["ticker"].tolist())
        tickers = [t for t in tickers if t not in done]
        print(f"Resuming: {len(tickers)} remaining")

    print(f"Fetching yfinance fundamentals for {len(tickers)} Nifty 100 tickers …")
    for i, ticker in enumerate(tickers):
        es = fetch_earnings_surprises(ticker)
        if not es.empty:
            con.execute("DELETE FROM earnings_surprises WHERE ticker = ?", [ticker])
            con.execute("INSERT OR IGNORE INTO earnings_surprises SELECT * FROM es")

        qf = fetch_quarterly_financials(ticker)
        if not qf.empty:
            con.execute("DELETE FROM quarterly_financials WHERE ticker = ?", [ticker])
            con.execute("INSERT OR IGNORE INTO quarterly_financials SELECT * FROM qf")

        snap = fetch_snapshot(ticker)
        if snap:
            sdf = pd.DataFrame([snap])
            con.execute("DELETE FROM snapshot_ratios WHERE ticker = ?", [ticker])
            con.execute("INSERT INTO snapshot_ratios SELECT * FROM sdf")

        if (i + 1) % 10 == 0:
            n_es = con.execute("SELECT COUNT(*) FROM earnings_surprises").fetchone()[0]
            print(f"  {i+1}/{len(tickers)}  {ticker}  ({n_es} surprise rows so far)")

        time.sleep(0.4)

    n_es  = con.execute("SELECT COUNT(*) FROM earnings_surprises").fetchone()[0]
    n_qf  = con.execute("SELECT COUNT(*) FROM quarterly_financials").fetchone()[0]
    n_sn  = con.execute("SELECT COUNT(*) FROM snapshot_ratios").fetchone()[0]
    con.close()
    print(f"\nDone: {n_es} earnings surprises, {n_qf} quarterly rows, {n_sn} snapshots → {db_path}")


if __name__ == "__main__":
    main()
