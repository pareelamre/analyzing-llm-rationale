#!/usr/bin/env python3
"""Fetch fundamental data for S&P 100 tickers via yfinance → DuckDB.

Stores point-in-time reconstructed fundamentals:
  - Quarterly EPS actuals + estimates → earnings surprise history
  - Trailing 12m EPS → historical P/E from price series
  - Revenue, gross margin, net margin from quarterly financials
  - Current-snapshot ratios: P/B, ROE, debt/equity (as-of-today)

Usage:
    python scripts/stock_fundamentals.py
    python scripts/stock_fundamentals.py --resume

Output: data/stock_fundamentals.duckdb
  tables: earnings_surprises, quarterly_financials, snapshot_ratios
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
PRICES_DB = str(ROOT / "data" / "stock_prices.duckdb")
FUND_DB   = ROOT / "data" / "stock_fundamentals.duckdb"


def init_db(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS earnings_surprises (
            ticker       VARCHAR,
            date         DATE,
            eps_actual   DOUBLE,
            eps_estimate DOUBLE,
            surprise_pct DOUBLE,   -- (actual - estimate) / |estimate|
            PRIMARY KEY (ticker, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS quarterly_financials (
            ticker         VARCHAR,
            date           DATE,
            revenue        DOUBLE,
            gross_profit   DOUBLE,
            net_income     DOUBLE,
            eps_diluted    DOUBLE,
            gross_margin   DOUBLE,   -- gross_profit / revenue
            net_margin     DOUBLE,   -- net_income / revenue
            PRIMARY KEY (ticker, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS snapshot_ratios (
            ticker         VARCHAR PRIMARY KEY,
            fetched_date   DATE,
            price_to_book  DOUBLE,
            roe            DOUBLE,
            debt_to_equity DOUBLE,
            revenue_growth DOUBLE,
            earnings_growth DOUBLE,
            forward_pe     DOUBLE
        )
    """)
    return con


def fetch_earnings_surprises(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return pd.DataFrame()
        ed = ed.reset_index()
        date_col = ed.columns[0]
        ed = ed.rename(columns={
            date_col: "date",
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
        })
        ed = ed[["date", "eps_estimate", "eps_actual"]].dropna(subset=["eps_actual"])
        ed["date"] = pd.to_datetime(ed["date"]).dt.date
        ed["eps_estimate"] = pd.to_numeric(ed["eps_estimate"], errors="coerce")
        ed["eps_actual"]   = pd.to_numeric(ed["eps_actual"],   errors="coerce")
        ed = ed.dropna(subset=["eps_actual"])
        ed["surprise_pct"] = (
            (ed["eps_actual"] - ed["eps_estimate"]) /
            (ed["eps_estimate"].abs() + 1e-9)
        ).clip(-5, 5)
        ed["ticker"] = ticker
        return ed[["ticker", "date", "eps_actual", "eps_estimate", "surprise_pct"]]
    except Exception:
        return pd.DataFrame()


def fetch_quarterly_financials(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    rows = []
    try:
        qf = t.quarterly_financials
        if qf is None or qf.empty:
            return pd.DataFrame()
        qf = qf.T
        for date, row in qf.iterrows():
            rev = row.get("Total Revenue") or row.get("Revenue")
            gp  = row.get("Gross Profit")
            ni  = row.get("Net Income")
            rows.append({
                "ticker": ticker,
                "date": pd.to_datetime(date).date(),
                "revenue":      float(rev) if rev is not None and pd.notna(rev) else None,
                "gross_profit": float(gp)  if gp  is not None and pd.notna(gp)  else None,
                "net_income":   float(ni)  if ni  is not None and pd.notna(ni)  else None,
                "eps_diluted":  None,
                "gross_margin": float(gp / rev) if (rev and gp and float(rev) != 0 and pd.notna(rev) and pd.notna(gp)) else None,
                "net_margin":   float(ni / rev) if (rev and ni and float(rev) != 0 and pd.notna(rev) and pd.notna(ni)) else None,
            })
    except Exception:
        pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_snapshot_ratios(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker":          ticker,
            "fetched_date":    pd.Timestamp.now().date(),
            "price_to_book":   info.get("priceToBook"),
            "roe":             info.get("returnOnEquity"),
            "debt_to_equity":  info.get("debtToEquity"),
            "revenue_growth":  info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "forward_pe":      info.get("forwardPE"),
        }
    except Exception:
        return None


def get_tickers(db_path: str) -> list[str]:
    con = duckdb.connect(db_path, read_only=True)
    tickers = con.execute("SELECT DISTINCT ticker FROM prices").df()["ticker"].tolist()
    con.close()
    return sorted(tickers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--db", default=str(FUND_DB))
    args = ap.parse_args()

    db_path = Path(args.db)
    con = init_db(db_path)

    tickers = get_tickers(PRICES_DB)
    print(f"Fetching fundamentals for {len(tickers)} tickers …")

    if args.resume:
        done = set(con.execute("SELECT DISTINCT ticker FROM snapshot_ratios").df()["ticker"].tolist())
        tickers = [t for t in tickers if t not in done]
        print(f"  {len(tickers)} remaining after resume")

    for i, ticker in enumerate(tickers):
        # Earnings surprises
        es = fetch_earnings_surprises(ticker)
        if not es.empty:
            con.execute("DELETE FROM earnings_surprises WHERE ticker = ?", [ticker])
            con.execute("INSERT OR IGNORE INTO earnings_surprises SELECT * FROM es")

        # Quarterly financials
        qf = fetch_quarterly_financials(ticker)
        if not qf.empty:
            con.execute("DELETE FROM quarterly_financials WHERE ticker = ?", [ticker])
            con.execute("INSERT OR IGNORE INTO quarterly_financials SELECT * FROM qf")

        # Snapshot ratios
        snap = fetch_snapshot_ratios(ticker)
        if snap:
            sdf = pd.DataFrame([snap])
            con.execute("DELETE FROM snapshot_ratios WHERE ticker = ?", [ticker])
            con.execute("INSERT INTO snapshot_ratios SELECT * FROM sdf")

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(tickers)} done")

        time.sleep(0.3)

    n_es  = con.execute("SELECT COUNT(*) FROM earnings_surprises").fetchone()[0]
    n_qf  = con.execute("SELECT COUNT(*) FROM quarterly_financials").fetchone()[0]
    n_sn  = con.execute("SELECT COUNT(*) FROM snapshot_ratios").fetchone()[0]
    con.close()
    print(f"\nDone: {n_es:,} earnings surprises, {n_qf:,} quarterly financials, {n_sn} snapshots → {db_path}")


if __name__ == "__main__":
    main()
