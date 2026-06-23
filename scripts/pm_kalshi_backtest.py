#!/usr/bin/env python3
"""Prediction market backtest on Kalshi data with LLM forecasts as signal.

Since these markets are still open (no oracle resolution), we simulate:
  - Entry: buy YES at market_price when model_prob - market_price > threshold
           buy NO  at (1 - market_price) when market_price - model_prob > threshold
  - Exit:  at the next available candlestick close (holding = N days)
  - Return: (exit_price - entry_price) / entry_price  (YES leg)
            or ((1-exit_price) - (1-entry_price)) / (1-entry_price)  (NO leg)

Also computes:
  - Edge distribution per model/market
  - Calibration: does higher model_prob predict higher future market price?
  - Spread-adjusted ROI (bid/ask spread as transaction cost)

Usage:
    python scripts/pm_kalshi_backtest.py
    python scripts/pm_kalshi_backtest.py --threshold 0.10 --holding 7
    python scripts/pm_kalshi_backtest.py --run-sweep
"""
from __future__ import annotations

import argparse
import datetime
import json
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
KALSHI_DB  = str(ROOT / "data" / "kalshi.duckdb")
TRACK_DB   = str(ROOT / "data" / "track_record_store.duckdb")
RESULTS_DIR = ROOT / "results" / "pm_kalshi"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    kcon = duckdb.connect(KALSHI_DB, read_only=True)
    candles = kcon.execute("""
        SELECT ticker, ts, open, high, low, close,
               yes_bid, yes_ask,
               COALESCE(yes_ask - yes_bid, 0.05) AS spread,
               volume
        FROM candlesticks ORDER BY ticker, ts
    """).df()
    candles["date"] = candles["ts"].apply(
        lambda t: datetime.datetime.utcfromtimestamp(t).date()
    )
    kcon.close()

    tcon = duckdb.connect(TRACK_DB, read_only=True)
    forecasts = tcon.execute("""
        SELECT ident AS ticker, model, snapshot_date,
               model_probability, market_probability,
               resolved, outcome, lead_time_days
        FROM forecast_snapshot
        WHERE platform = 'Kalshi'
    """).df()
    forecasts["snapshot_date"] = pd.to_datetime(forecasts["snapshot_date"]).dt.date
    tcon.close()

    return candles, forecasts


def compute_trades(
    candles: pd.DataFrame,
    forecasts: pd.DataFrame,
    threshold: float,
    holding: int,
    spread_cost: float,
) -> pd.DataFrame:
    trades = []
    candles_by_ticker = {t: g.sort_values("ts").reset_index(drop=True)
                         for t, g in candles.groupby("ticker")}

    for _, row in forecasts.iterrows():
        ticker = row["ticker"]
        hist = candles_by_ticker.get(ticker)
        if hist is None or hist.empty:
            continue

        snap_date = row["snapshot_date"]
        model_prob = row["model_probability"]
        market_prob = row["market_probability"]  # from track record snapshot
        edge = model_prob - market_prob

        if abs(edge) < threshold:
            continue

        # Find entry candle: first candle on or after snapshot_date
        entry_rows = hist[hist["date"] >= snap_date]
        if entry_rows.empty:
            continue
        entry = entry_rows.iloc[0]
        entry_price = float(entry["close"])
        entry_spread = float(entry["spread"]) if not pd.isna(entry["spread"]) else spread_cost

        # Find exit candle: N days later
        exit_rows = hist[hist["date"] >= snap_date + datetime.timedelta(days=holding)]
        if exit_rows.empty:
            exit_price = float(hist.iloc[-1]["close"])  # use last available
            hold_days = (hist.iloc[-1]["date"] - snap_date).days
        else:
            exit_price = float(exit_rows.iloc[0]["close"])
            hold_days = (exit_rows.iloc[0]["date"] - snap_date).days

        # Position and return
        cost = entry_spread / 2 + spread_cost  # half spread + fixed cost
        if edge > 0:  # model more bullish → buy YES
            position = "yes"
            ret = (exit_price - entry_price - cost)
        else:         # model more bearish → buy NO
            position = "no"
            ret = ((1 - exit_price) - (1 - entry_price) - cost)

        trades.append({
            "ticker": ticker,
            "model": row["model"],
            "snapshot_date": snap_date,
            "model_prob": round(model_prob, 4),
            "market_prob": round(market_prob, 4),
            "edge": round(edge, 4),
            "position": position,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "hold_days": hold_days,
            "return": round(float(ret), 4),
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def edge_analysis(candles: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    """Print edge distribution and calibration stats."""
    print("=== Edge distribution (model_prob - market_prob) ===")
    merged = forecasts.copy()
    merged["edge"] = merged["model_probability"] - merged["market_probability"]
    print(merged.groupby("model")["edge"].describe().round(3).to_string())

    print()
    print("=== Per-market summary ===")
    summary = (
        merged.groupby(["ticker", "model"])
        .agg(n=("edge","count"), mean_edge=("edge","mean"),
             mean_model=("model_probability","mean"),
             mean_market=("market_probability","mean"))
        .round(3)
    )
    print(summary.to_string())

    print()
    print("=== Calibration: does higher edge predict price movement? ===")
    by_ticker = {t: g.sort_values("ts").reset_index(drop=True)
                 for t, g in candles.groupby("ticker")}
    cal_rows = []
    for _, row in forecasts.iterrows():
        hist = by_ticker.get(row["ticker"])
        if hist is None: continue
        snap = row["snapshot_date"]
        fut = hist[hist["date"] >= snap + datetime.timedelta(days=7)]
        if fut.empty: continue
        now_rows = hist[hist["date"] >= snap]
        if now_rows.empty: continue
        price_now = float(now_rows.iloc[0]["close"])
        price_fut = float(fut.iloc[0]["close"])
        cal_rows.append({
            "model": row["model"],
            "edge": row["model_probability"] - row["market_probability"],
            "price_change_7d": price_fut - price_now,
            "edge_correct": int(
                (row["model_probability"] > row["market_probability"] and price_fut > price_now) or
                (row["model_probability"] < row["market_probability"] and price_fut < price_now)
            )
        })
    if cal_rows:
        cal = pd.DataFrame(cal_rows)
        print(cal.groupby("model")[["edge_correct","price_change_7d"]].mean().round(3).to_string())


def run_sweep(out_dir: Path) -> None:
    candles, forecasts = load_data()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for threshold, holding, spread_cost in product(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        [1, 3, 7, 14, 30],
        [0.01, 0.02, 0.05],
    ):
        trades = compute_trades(candles, forecasts, threshold, holding, spread_cost)
        if trades.empty or len(trades) < 3:
            continue
        rets = trades["return"]
        sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(len(rets))
        results.append({
            "threshold": threshold, "holding": holding, "spread_cost": spread_cost,
            "n_trades": len(trades),
            "win_rate": round(float((rets > 0).mean()), 3),
            "avg_return": round(float(rets.mean()), 4),
            "sharpe": round(float(sharpe), 3),
            "max_dd": round(float(((1+rets).cumprod() / (1+rets).cumprod().cummax() - 1).min()), 3),
        })

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    out = out_dir / "summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSweep: {len(df)} parameter combos → {out}")
    print("\nTop 10 by Sharpe (min 5 trades):")
    print(df[df.n_trades >= 5].head(10).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="Min |model_prob - market_prob| to enter")
    ap.add_argument("--holding", type=int, default=7, help="Holding period in days")
    ap.add_argument("--spread-cost", type=float, default=0.02,
                    help="Fixed spread cost per trade")
    ap.add_argument("--run-sweep", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    candles, forecasts = load_data()
    print(f"Loaded {len(candles)} candles ({candles['ticker'].nunique()} markets), "
          f"{len(forecasts)} LLM forecasts ({forecasts['model'].nunique()} models)")

    edge_analysis(candles, forecasts)

    if args.run_sweep:
        run_sweep(Path(args.out))
        return

    print(f"\n=== Backtest  threshold={args.threshold}  holding={args.holding}d ===")
    trades = compute_trades(candles, forecasts, args.threshold, args.holding, args.spread_cost)
    if trades.empty:
        print("No trades generated — try lower threshold.")
        return

    rets = trades["return"]
    print(f"Trades: {len(trades)}  Win rate: {(rets>0).mean():.1%}  "
          f"Avg return: {rets.mean():.1%}  Sharpe: {rets.mean()/(rets.std()+1e-9)*np.sqrt(len(rets)):.2f}")
    print()
    print("By model:")
    print(trades.groupby("model")["return"].agg(["mean","count"]).round(3).to_string())
    print()
    print("By ticker:")
    print(trades.groupby("ticker")[["return","edge"]].mean().round(3).to_string())
    print()
    print("All trades:")
    print(trades.sort_values("edge", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
