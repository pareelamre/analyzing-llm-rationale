#!/usr/bin/env python3
"""PEAD strategy on Nifty 100 using yfinance analyst consensus EPS surprise.

Signal: (actual_EPS - analyst_estimate) / |estimate|  — true analyst surprise,
        stronger than YoY comparison used for US stocks.
Entry:  next trading day after announce_date
Exit:   N trading days later
Long:   top-quartile beats; Short: bottom-quartile misses (or --long-only)

Usage:
    python scripts/nifty_pead.py
    python scripts/nifty_pead.py --holding 30 --long-only
    python scripts/nifty_pead.py --run-sweep

Output: results/nifty_pead/
"""
from __future__ import annotations

import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
PRICES_DB = str(ROOT / "data" / "nifty_prices.duckdb")
FUND_DB   = str(ROOT / "data" / "nifty_fundamentals.duckdb")
OUT_DIR   = ROOT / "results" / "nifty_pead"


def load_surprises() -> pd.DataFrame:
    con = duckdb.connect(FUND_DB, read_only=True)
    df = con.execute("""
        SELECT ticker, announce_date, eps_estimate, eps_actual, surprise_pct
        FROM earnings_surprises
        WHERE eps_actual IS NOT NULL
        ORDER BY ticker, announce_date
    """).df()
    con.close()
    df["announce_date"] = pd.to_datetime(df["announce_date"])
    df["year"] = df["announce_date"].dt.year
    return df


def load_prices() -> pd.DataFrame:
    con = duckdb.connect(PRICES_DB, read_only=True)
    df = con.execute("SELECT ticker, date, close FROM prices ORDER BY ticker, date").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def trade_return(prices_wide, ticker, entry_date, holding, direction, cost_bps):
    if ticker not in prices_wide.columns:
        return None
    col = prices_wide[ticker].dropna()
    # Entry: next trading day on or after announce_date + 1
    entry_rows = col[col.index > entry_date]
    if entry_rows.empty:
        return None
    p0 = entry_rows.iloc[0]
    exit_rows = col[col.index >= entry_rows.index[0] + pd.Timedelta(days=holding)]
    if exit_rows.empty:
        return None
    p1 = exit_rows.iloc[0]
    if p0 == 0:
        return None
    return direction * (p1 / p0 - 1) - cost_bps / 10_000


def walk_forward(surprises, prices_wide, holding, train_years,
                 top_pct, long_only, cost_bps):
    years = sorted(surprises["year"].unique())
    equity, folds, all_trades = pd.Series(dtype=float), [], []

    for test_year in years[train_years:]:
        train = surprises[surprises["year"] < test_year]
        test  = surprises[surprises["year"] == test_year]
        if len(train) < 30 or len(test) < 5:
            continue

        long_thresh  = train["surprise_pct"].quantile(1 - top_pct)
        short_thresh = train["surprise_pct"].quantile(top_pct)

        fold_trades = []
        for _, row in test.iterrows():
            s = row["surprise_pct"]
            direction = 1 if s >= long_thresh else (-1 if not long_only and s <= short_thresh else None)
            if direction is None:
                continue
            ret = trade_return(prices_wide, row["ticker"], row["announce_date"],
                               holding, direction, cost_bps)
            if ret is None:
                continue
            fold_trades.append({
                "year": int(test_year), "ticker": row["ticker"],
                "announce_date": row["announce_date"],
                "surprise_pct": round(float(row["surprise_pct"]), 4),
                "eps_actual": round(float(row["eps_actual"]), 4),
                "eps_estimate": round(float(row["eps_estimate"]), 4),
                "direction": "long" if direction == 1 else "short",
                "return": round(float(ret), 4),
            })

        if not fold_trades:
            continue

        fd   = pd.DataFrame(fold_trades)
        all_trades.append(fd)
        rets = fd["return"]
        ann_ret = rets.mean() * (252 / holding)
        vol     = rets.std() * np.sqrt(252 / holding)
        sharpe  = ann_ret / (vol + 1e-9)

        fold_eq = (1 + rets.reset_index(drop=True)).cumprod()
        equity  = fold_eq if equity.empty else pd.concat(
            [equity, equity.iloc[-1] * fold_eq], ignore_index=True)

        folds.append({
            "year": int(test_year), "n_trades": int(len(fd)),
            "win_rate": round(float((rets > 0).mean()), 3),
            "avg_return": round(float(rets.mean()), 4),
            "sharpe": round(float(sharpe), 3),
            "long_trades": int((fd["direction"] == "long").sum()),
            "short_trades": int((fd["direction"] == "short").sum()),
        })
        print(f"  {test_year}: Sharpe={sharpe:.2f}  AvgRet={rets.mean():.2%}"
              f"  n={len(fd)}  WinRate={(rets>0).mean():.1%}")

    return equity, folds, pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def run_sweep(surprises, prices_wide, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for holding, top_pct, long_only, cost_bps in product(
        [21, 30, 42, 63], [0.20, 0.25, 0.33], [True, False], [10, 20]
    ):
        _, folds, _ = walk_forward(surprises, prices_wide, holding=holding,
                                   train_years=2, top_pct=top_pct,
                                   long_only=long_only, cost_bps=cost_bps)
        if not folds:
            continue
        results.append({
            "holding": holding, "top_pct": top_pct,
            "long_only": long_only, "cost_bps": cost_bps,
            "avg_sharpe":   round(float(np.mean([f["sharpe"]      for f in folds])), 3),
            "avg_return":   round(float(np.mean([f["avg_return"]   for f in folds])), 4),
            "avg_win_rate": round(float(np.mean([f["win_rate"]     for f in folds])), 3),
            "n_folds": len(folds),
        })

    df = pd.DataFrame(results).sort_values("avg_sharpe", ascending=False)
    df.to_csv(out_dir / "sweep_summary.csv", index=False)
    print(f"\nSweep: {len(df)} combos")
    print(df.head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holding",     type=int,   default=30)
    ap.add_argument("--train-years", type=int,   default=2)
    ap.add_argument("--top-pct",     type=float, default=0.25)
    ap.add_argument("--cost-bps",    type=float, default=10.0)
    ap.add_argument("--long-only",   action="store_true")
    ap.add_argument("--run-sweep",   action="store_true")
    ap.add_argument("--out",         default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading earnings surprises …")
    surprises = load_surprises()
    print(f"  {len(surprises):,} events, {surprises['ticker'].nunique()} tickers, "
          f"{surprises['year'].min()}–{surprises['year'].max()}")

    print("Loading prices …")
    prices_wide = load_prices()

    if args.run_sweep:
        run_sweep(surprises, prices_wide, out_dir)
        return

    print(f"\nNifty PEAD  holding={args.holding}d  top={args.top_pct:.0%}  "
          f"long_only={args.long_only}  cost={args.cost_bps}bps")
    equity, folds, trades = walk_forward(
        surprises, prices_wide, holding=args.holding,
        train_years=args.train_years, top_pct=args.top_pct,
        long_only=args.long_only, cost_bps=args.cost_bps,
    )

    if not folds:
        print("No trades generated.")
        return

    rets = trades["return"]
    print(f"\nOverall: {len(trades)} trades  WinRate={(rets>0).mean():.1%}"
          f"  AvgRet={rets.mean():.2%}"
          f"  Sharpe={rets.mean()/(rets.std()+1e-9)*np.sqrt(252/args.holding):.2f}")

    (out_dir / "pead_summary.json").write_text(json.dumps(folds, indent=2))
    equity.to_csv(out_dir / "pead_equity.csv")
    trades.to_csv(out_dir / "pead_trades.csv", index=False)
    print(f"Results → {out_dir}/")


if __name__ == "__main__":
    main()
