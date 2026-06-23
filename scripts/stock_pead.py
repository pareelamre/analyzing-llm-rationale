#!/usr/bin/env python3
"""Post-Earnings Announcement Drift (PEAD) strategy.

Triggers trades on earnings filing dates using SEC EDGAR point-in-time data.
Signal: EPS surprise = (actual_Q_EPS - same_Q_prior_year_EPS) / |prior|
Entry:  buy on the day the 10-Q/10-K is filed (strictly after the filing)
Exit:   N trading days later
Long:   top-quartile surprise (biggest beats)
Short:  bottom-quartile surprise (biggest misses)  [optional, --long-only to skip]

Walk-forward OOS: train on past N years to estimate surprise thresholds,
test on next year.

Usage:
    python scripts/stock_pead.py
    python scripts/stock_pead.py --holding 30 --long-only
    python scripts/stock_pead.py --run-sweep

Output:
    results/stock_pead/pead_summary.json
    results/stock_pead/pead_equity.csv
    results/stock_pead/pead_trades.csv
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
PRICES_DB = str(ROOT / "data" / "stock_prices.duckdb")
EDGAR_DB  = str(ROOT / "data" / "stock_edgar.duckdb")
OUT_DIR   = ROOT / "results" / "stock_pead"


def load_eps_surprises() -> pd.DataFrame:
    """
    Build point-in-time EPS surprise for each (ticker, filing).
    Strictly uses same-quarter prior-year comparison.
    Returns rows with: ticker, filed, period_end, qtr, eps_actual, eps_prior, surprise_pct
    """
    con = duckdb.connect(EDGAR_DB, read_only=True)
    eps = con.execute("""
        SELECT ticker, filed, period_end, fiscal_period, value AS eps
        FROM facts
        WHERE metric = 'eps_diluted'
          AND fiscal_period IN ('Q1','Q2','Q3','Q4')
          AND form IN ('10-Q','10-K')
        ORDER BY ticker, period_end, filed
    """).df()
    con.close()

    eps["filed"]      = pd.to_datetime(eps["filed"])
    eps["period_end"] = pd.to_datetime(eps["period_end"])
    eps["qtr"]        = eps["period_end"].dt.quarter
    eps["year"]       = eps["period_end"].dt.year

    # Keep latest filing per (ticker, period_end) — handles amendments
    eps = (eps.sort_values(["ticker", "period_end", "filed"])
              .drop_duplicates(["ticker", "period_end"], keep="last"))

    # Match to same quarter prior year
    prior = eps[["ticker", "year", "qtr", "eps"]].copy()
    prior = prior.rename(columns={"eps": "eps_prior", "year": "prior_year"})
    prior["year"] = prior["prior_year"] + 1   # shift: prior_year Q → current_year Q

    merged = eps.merge(prior[["ticker", "year", "qtr", "eps_prior"]],
                       on=["ticker", "year", "qtr"], how="left")
    merged["surprise_pct"] = (
        (merged["eps"] - merged["eps_prior"]) /
        (merged["eps_prior"].abs() + 1e-9)
    ).clip(-5, 5)

    return merged.dropna(subset=["eps_prior"]).reset_index(drop=True)


def load_prices() -> pd.DataFrame:
    con = duckdb.connect(PRICES_DB, read_only=True)
    df = con.execute("SELECT ticker, date, close FROM prices ORDER BY ticker, date").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def compute_trade_return(
    prices_wide: pd.DataFrame,
    ticker: str,
    entry_date: pd.Timestamp,
    holding: int,
    direction: int,   # +1 long, -1 short
    cost_bps: float,
) -> float | None:
    if ticker not in prices_wide.columns:
        return None
    col = prices_wide[ticker]
    entry_rows = col[col.index >= entry_date].dropna()
    if entry_rows.empty:
        return None
    p0 = entry_rows.iloc[0]
    exit_rows = col[col.index >= entry_date + pd.Timedelta(days=holding)].dropna()
    if exit_rows.empty:
        return None
    p1 = exit_rows.iloc[0]
    if p0 == 0:
        return None
    gross = direction * (p1 / p0 - 1)
    return gross - cost_bps / 10_000


def walk_forward_pead(
    surprises: pd.DataFrame,
    prices_wide: pd.DataFrame,
    holding: int,
    train_years: int,
    top_pct: float,
    long_only: bool,
    cost_bps: float,
) -> tuple[pd.Series, list[dict], pd.DataFrame]:
    all_years = sorted(surprises["year"].unique())
    equity    = pd.Series(dtype=float)
    folds     = []
    all_trades = []

    for test_year in all_years[train_years:]:
        train_mask = surprises["year"] < test_year
        test_mask  = surprises["year"] == test_year
        train_df   = surprises[train_mask]
        test_df    = surprises[test_mask]

        if len(train_df) < 50 or len(test_df) < 10:
            continue

        # Thresholds from training distribution
        long_thresh  = train_df["surprise_pct"].quantile(1 - top_pct)
        short_thresh = train_df["surprise_pct"].quantile(top_pct)

        fold_trades = []
        for _, row in test_df.iterrows():
            s = row["surprise_pct"]
            direction = None
            if s >= long_thresh:
                direction = 1
            elif not long_only and s <= short_thresh:
                direction = -1
            if direction is None:
                continue

            ret = compute_trade_return(
                prices_wide, row["ticker"], row["filed"], holding, direction, cost_bps
            )
            if ret is None:
                continue

            fold_trades.append({
                "year":         test_year,
                "ticker":       row["ticker"],
                "filed":        row["filed"],
                "period_end":   row["period_end"],
                "surprise_pct": round(float(s), 4),
                "direction":    "long" if direction == 1 else "short",
                "return":       round(float(ret), 4),
            })

        if not fold_trades:
            continue

        fold_df  = pd.DataFrame(fold_trades)
        all_trades.append(fold_df)
        rets     = fold_df["return"]
        vol      = rets.std()
        ann_ret  = rets.mean() * (252 / holding)
        sharpe   = ann_ret / (vol * np.sqrt(252 / holding) + 1e-9)

        fold_eq = (1 + rets.sort_values(key=lambda x: fold_df.loc[x.index, "filed"])
                          .reset_index(drop=True)).cumprod()
        if equity.empty:
            equity = fold_eq
        else:
            equity = pd.concat([equity, equity.iloc[-1] * fold_eq], ignore_index=True)

        folds.append({
            "year":        int(test_year),
            "n_trades":    int(len(fold_df)),
            "win_rate":    round(float((rets > 0).mean()), 3),
            "avg_return":  round(float(rets.mean()), 4),
            "sharpe":      round(float(sharpe), 3),
            "long_trades": int((fold_df["direction"] == "long").sum()),
            "short_trades":int((fold_df["direction"] == "short").sum()),
        })
        print(f"  {test_year}: Sharpe={sharpe:.2f}  AvgRet={rets.mean():.2%}  "
              f"n={len(fold_df)}  WinRate={( rets>0).mean():.1%}")

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return equity, folds, trades_df


def run_sweep(surprises: pd.DataFrame, prices_wide: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for holding, top_pct, long_only, cost_bps in product(
        [21, 30, 42, 63],
        [0.20, 0.25, 0.33],
        [True, False],
        [5, 10],
    ):
        _, folds, trades = walk_forward_pead(
            surprises, prices_wide, holding=holding, train_years=4,
            top_pct=top_pct, long_only=long_only, cost_bps=cost_bps,
        )
        if not folds:
            continue
        avg_sharpe = np.mean([f["sharpe"] for f in folds])
        avg_ret    = np.mean([f["avg_return"] for f in folds])
        avg_wr     = np.mean([f["win_rate"] for f in folds])
        results.append({
            "holding": holding, "top_pct": top_pct,
            "long_only": long_only, "cost_bps": cost_bps,
            "avg_sharpe": round(float(avg_sharpe), 3),
            "avg_return": round(float(avg_ret), 4),
            "avg_win_rate": round(float(avg_wr), 3),
            "n_folds": len(folds),
        })

    df = pd.DataFrame(results).sort_values("avg_sharpe", ascending=False)
    out = out_dir / "sweep_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSweep done: {len(df)} combos → {out}")
    print("\nTop 10:")
    print(df.head(10).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holding",     type=int,   default=42)
    ap.add_argument("--train-years", type=int,   default=4)
    ap.add_argument("--top-pct",     type=float, default=0.25)
    ap.add_argument("--cost-bps",    type=float, default=5.0)
    ap.add_argument("--long-only",   action="store_true")
    ap.add_argument("--run-sweep",   action="store_true")
    ap.add_argument("--out",         default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading EPS surprises from EDGAR …")
    surprises = load_eps_surprises()
    print(f"  {len(surprises):,} earnings events, "
          f"{surprises['ticker'].nunique()} tickers, "
          f"{surprises['year'].min()}–{surprises['year'].max()}")

    print("Loading prices …")
    prices_wide = load_prices()

    if args.run_sweep:
        print("\nRunning parameter sweep …")
        run_sweep(surprises, prices_wide, out_dir)
        return

    print(f"\nPEAD walk-forward  holding={args.holding}d  "
          f"top_pct={args.top_pct:.0%}  long_only={args.long_only}")
    equity, folds, trades = walk_forward_pead(
        surprises, prices_wide,
        holding=args.holding, train_years=args.train_years,
        top_pct=args.top_pct, long_only=args.long_only, cost_bps=args.cost_bps,
    )

    if not folds:
        print("No trades generated.")
        return

    all_rets = trades["return"]
    print(f"\nOverall: {len(trades)} trades  "
          f"Win rate={( all_rets>0).mean():.1%}  "
          f"Avg ret={all_rets.mean():.2%}  "
          f"Sharpe={all_rets.mean() / (all_rets.std()+1e-9) * np.sqrt(252/args.holding):.2f}")
    print(f"Long: {(trades.direction=='long').sum()}  "
          f"Short: {(trades.direction=='short').sum()}")

    (out_dir / "pead_summary.json").write_text(json.dumps(folds, indent=2))
    equity.to_csv(out_dir / "pead_equity.csv")
    trades.to_csv(out_dir / "pead_trades.csv", index=False)
    print(f"\nResults → {out_dir}/")


if __name__ == "__main__":
    main()
