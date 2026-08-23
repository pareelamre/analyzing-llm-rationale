#!/usr/bin/env python3
"""Prediction market strategy backtester on Polymarket resolved markets.

Strategies:
  opening_fade   -- bet against overpriced early markets (mean reversion to fair value)
  momentum       -- follow price trend over a lookback window
  vol_signal     -- exploit volume-based mispricing
  time_decay     -- bet on/against resolution as deadline approaches (theta)
  calibration    -- exploit category-level systematic biases

For each resolved binary market, we simulate taking a position (YES or NO)
at day T and holding until resolution. Returns are computed assuming you
buy at the observed price (ask = price + spread/2) and receive 1.0 or 0.0
at resolution.

Usage:
    python scripts/pm_backtest.py
    python scripts/pm_backtest.py --strategy momentum --lookback 7
    python scripts/pm_backtest.py --run-sweep --workers 16 --out results/pm_sweep/

Output: results/pm_sweep/summary.csv
"""
from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DB_PATH = str(ROOT / "data" / "polymarket.duckdb")
RESULTS_DIR = ROOT / "results" / "pm_sweep"

StratType = Literal["opening_fade", "momentum", "time_decay", "calibration"]


@dataclass
class PMParams:
    strategy: StratType = "opening_fade"
    lookback: int = 7          # days for signal lookback
    entry_day: int = 3         # day from market open to enter (for opening_fade)
    min_days_left: int = 1     # minimum days remaining at entry
    max_days_left: int = 9999
    min_volume: float = 10_000
    yes_threshold: float = 0.6  # enter YES when signal says prob > threshold
    no_threshold: float = 0.4   # enter NO when signal says prob < threshold
    cost_bps: float = 50.0      # prediction market spread in bps (much higher than stocks)


@dataclass
class PMResult:
    sharpe: float
    annual_return: float
    max_drawdown: float
    win_rate: float
    n_bets: int
    avg_hold_days: float
    params: PMParams = field(repr=False, default_factory=PMParams)


def load_data(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(db_path, read_only=True)
    markets = con.execute("""
        SELECT id, question, category, volume, liquidity, created_at, end_date,
               closed_time, outcome, yes_price_final, spread
        FROM markets
        WHERE outcome IN ('yes', 'no')
          AND volume > 0
    """).df()
    prices = con.execute("""
        SELECT market_id, ts, price,
               to_timestamp(ts) AS dt
        FROM price_history
        ORDER BY market_id, ts
    """).df()
    con.close()
    return markets, prices


def _signal_opening_fade(price_history: pd.DataFrame, entry_day: int) -> float | None:
    """Signal: price on entry day. Fade extreme opening prices."""
    if len(price_history) < entry_day:
        return None
    return price_history.iloc[entry_day - 1]["price"]


def _signal_momentum(price_history: pd.DataFrame, lookback: int) -> float | None:
    """Signal: price change over lookback days. Positive = uptrend."""
    if len(price_history) < lookback + 1:
        return None
    recent = price_history.iloc[-1]["price"]
    past = price_history.iloc[-(lookback + 1)]["price"]
    return recent - past


def _signal_time_decay(price_history: pd.DataFrame, current_price: float,
                       days_left: float) -> float | None:
    """Signal: current price diverges from expected convergence rate."""
    if days_left <= 0:
        return None
    # Expected convergence: price should move toward 0 or 1 as days_left → 0
    # Signal: price is too far from extremes given time remaining
    return current_price  # raw price as signal; threshold determines direction


def compute_trade_returns(
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    p: PMParams,
) -> pd.DataFrame:
    """
    For each market, compute a simulated trade:
      - Compute signal at entry point
      - Determine position (YES/NO/none)
      - Compute return: (payout - entry_price) / entry_price
    Returns DataFrame of trades with columns: market_id, entry_date, position,
    entry_price, return, hold_days, category.
    """
    cost = p.cost_bps / 10_000
    trades: list[dict] = []

    prices_by_market = {mid: grp.reset_index(drop=True)
                        for mid, grp in prices.groupby("market_id")}

    for _, mkt in markets.iterrows():
        mid = mkt["id"]
        if mkt["volume"] < p.min_volume:
            continue
        hist = prices_by_market.get(mid)
        if hist is None or len(hist) < 3:
            continue

        hist = hist.sort_values("ts").reset_index(drop=True)
        hist["day"] = (hist["ts"] - hist["ts"].iloc[0]) / 86400

        outcome_val = 1.0 if mkt["outcome"] == "yes" else 0.0
        total_days = hist["day"].iloc[-1]

        if p.strategy == "opening_fade":
            # Enter at entry_day from open
            entry_rows = hist[hist["day"] >= p.entry_day]
            if entry_rows.empty:
                continue
            entry_row = entry_rows.iloc[0]
            entry_price = entry_row["price"]
            days_left = total_days - entry_row["day"]
            if not (p.min_days_left <= days_left <= p.max_days_left):
                continue
            signal = entry_price  # fade: high price → short YES

            if signal > p.yes_threshold:
                position = "no"  # fade high opening price
                entry_cost = (1 - entry_price) + cost
                payout = 1 - outcome_val
            elif signal < p.no_threshold:
                position = "yes"  # fade low opening price
                entry_cost = entry_price + cost
                payout = outcome_val
            else:
                continue

        elif p.strategy == "momentum":
            if len(hist) < p.lookback + 1:
                continue
            entry_row = hist.iloc[-1]
            entry_price = entry_row["price"]
            days_left = 0  # holding until resolution (it's resolved)
            sig = _signal_momentum(hist, p.lookback)
            if sig is None:
                continue
            if sig > 0.05:  # uptrend → long YES
                position = "yes"
                entry_cost = entry_price + cost
                payout = outcome_val
            elif sig < -0.05:  # downtrend → long NO
                position = "no"
                entry_cost = (1 - entry_price) + cost
                payout = 1 - outcome_val
            else:
                continue

        elif p.strategy == "time_decay":
            # Enter when price is "stuck" in middle with few days left
            entry_rows = hist[hist["day"] >= (total_days - p.lookback)]
            if entry_rows.empty:
                continue
            entry_row = entry_rows.iloc[0]
            entry_price = entry_row["price"]
            days_left = total_days - entry_row["day"]
            if days_left > p.max_days_left:
                continue
            # Near resolution: prices near 0.5 are "uncertain" — skip these
            if 0.35 < entry_price < 0.65:
                continue
            position = "yes" if entry_price > 0.5 else "no"
            if position == "yes":
                entry_cost = entry_price + cost
                payout = outcome_val
            else:
                entry_cost = (1 - entry_price) + cost
                payout = 1 - outcome_val

        elif p.strategy == "calibration":
            # Use opening price as signal; bet based on category-specific bias
            entry_row = hist.iloc[0]
            entry_price = entry_row["price"]
            days_left = total_days
            # Simply follow or fade the opening price
            if entry_price > p.yes_threshold:
                position = "yes"
                entry_cost = entry_price + cost
                payout = outcome_val
            elif entry_price < p.no_threshold:
                position = "no"
                entry_cost = (1 - entry_price) + cost
                payout = 1 - outcome_val
            else:
                continue
        else:
            continue

        ret = (payout - entry_cost) / max(entry_cost, 0.01)
        trades.append({
            "market_id": mid,
            "category": mkt["category"],
            "position": position,
            "entry_price": round(entry_price, 4),
            "payout": payout,
            "return": round(ret, 4),
            "hold_days": round(float(total_days - entry_row.get("day", 0.0)), 1),
            "volume": mkt["volume"],
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def compute_metrics(trades: pd.DataFrame) -> PMResult:
    if trades.empty:
        return PMResult(0, 0, 0, 0, 0, 0)
    rets = trades["return"]
    n = len(rets)
    win_rate = float((rets > 0).mean())
    # Approximate Sharpe (treat each trade as one period)
    mean_ret = rets.mean()
    std_ret = rets.std() if len(rets) > 1 else 1.0
    sharpe = mean_ret / (std_ret + 1e-9) * np.sqrt(n)  # information ratio
    # Equity curve (sorted by market close)
    equity = (1 + rets).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    avg_hold = trades["hold_days"].mean()
    return PMResult(
        sharpe=round(float(sharpe), 4),
        annual_return=round(float(mean_ret), 4),  # per-trade avg (not annualised)
        max_drawdown=round(float(max_dd), 4),
        win_rate=round(win_rate, 4),
        n_bets=n,
        avg_hold_days=round(float(avg_hold), 1),
    )


def build_param_grid() -> list[dict]:
    grid = []
    for strategy, lookback, entry_day, yes_thr, no_thr, min_days in product(
        ["opening_fade", "momentum", "time_decay", "calibration"],
        [3, 7, 14, 30],       # lookback
        [1, 3, 7, 14],        # entry_day
        [0.60, 0.65, 0.70],   # yes_threshold
        [0.30, 0.35, 0.40],   # no_threshold (symmetric with yes)
        [1, 7],                # min_days_left
    ):
        if yes_thr + no_thr > 1.05:
            continue  # degenerate
        grid.append({
            "strategy": strategy,
            "lookback": lookback,
            "entry_day": entry_day,
            "yes_threshold": yes_thr,
            "no_threshold": no_thr,
            "min_days_left": min_days,
        })
    return grid


def _run_one(args: tuple) -> dict:
    i, params, db_path = args
    markets, prices = load_data(db_path)
    p = PMParams(**params)
    trades = compute_trade_returns(markets, prices, p)
    r = compute_metrics(trades)
    return {"job_id": i, **params,
            "sharpe": r.sharpe, "avg_return": r.annual_return,
            "max_drawdown": r.max_drawdown, "win_rate": r.win_rate,
            "n_bets": r.n_bets, "avg_hold_days": r.avg_hold_days}


def run_sweep(db_path: str, workers: int, out_dir: Path) -> None:
    grid = build_param_grid()
    print(f"Running {len(grid)} parameter combinations with {workers} workers …")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, (i, p, db_path)): i for i, p in enumerate(grid)}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"Job {futures[fut]} failed: {e}")
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(grid)} done")

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    out = out_dir / "summary.csv"
    df.to_csv(out, index=False)
    print(f"\nResults → {out}")
    print("\nTop 10 by Sharpe:")
    print(df[df["n_bets"] >= 10].head(10).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--strategy", default="opening_fade")
    ap.add_argument("--lookback", type=int, default=7)
    ap.add_argument("--entry-day", type=int, default=3)
    ap.add_argument("--yes-threshold", type=float, default=0.65)
    ap.add_argument("--no-threshold", type=float, default=0.35)
    ap.add_argument("--min-days-left", type=int, default=1)
    ap.add_argument("--min-volume", type=float, default=10_000)
    ap.add_argument("--run-sweep", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    if args.run_sweep:
        run_sweep(args.db, args.workers, Path(args.out))
        return

    markets, prices = load_data(args.db)
    print(f"Loaded {len(markets)} markets, {len(prices):,} price points")

    p = PMParams(
        strategy=args.strategy,
        lookback=args.lookback,
        entry_day=args.entry_day,
        yes_threshold=args.yes_threshold,
        no_threshold=args.no_threshold,
        min_days_left=args.min_days_left,
        min_volume=args.min_volume,
    )
    trades = compute_trade_returns(markets, prices, p)
    if trades.empty:
        print("No trades generated.")
        return
    r = compute_metrics(trades)
    print(f"\nStrategy: {p.strategy}")
    print(f"  Bets: {r.n_bets}  Win rate: {r.win_rate:.1%}  Avg hold: {r.avg_hold_days:.0f}d")
    print(f"  Sharpe: {r.sharpe:.3f}  Avg return/trade: {r.annual_return:.1%}  Max DD: {r.max_drawdown:.1%}")
    if not trades.empty:
        print("\nBy category:")
        print(trades.groupby("category")["return"].agg(["mean", "count"]).sort_values("mean", ascending=False).to_string())


if __name__ == "__main__":
    main()
