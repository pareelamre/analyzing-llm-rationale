#!/usr/bin/env python3
"""Run one parameter combination for the stock strategy sweep.

In standalone mode, builds and saves the full parameter grid, then runs all jobs
sequentially. In SLURM mode, each array task runs a single combination.

Usage (local):
    python scripts/stock_sweep.py --build-grid
    python scripts/stock_sweep.py --job-id 0
    python scripts/stock_sweep.py --run-all --workers 8

Usage (SLURM):
    sbatch scripts/stock_sweep_slurm.sh   # submits array 0..N-1

Output: results/stock_sweep/result_{job_id}.json
        results/stock_sweep/summary.csv   (after --aggregate)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from stock_backtest import BacktestParams, load_prices_wide, run_backtest

DB_PATH = str(ROOT / "data" / "stock_prices.duckdb")
GRID_PATH = ROOT / "results" / "stock_sweep" / "param_grid.json"
RESULTS_DIR = ROOT / "results" / "stock_sweep"


def build_param_grid() -> list[dict]:
    """Cartesian product of all strategy parameters."""
    strategies = ["momentum", "vol_adj_mom", "ma_cross", "rsi_reversal"]
    lookbacks = [10, 21, 42, 63, 126, 252]        # 2w 1m 2m 3m 6m 12m
    holdings = [1, 5, 21, 63]                      # 1d 1w 1m 3m
    skip_recents = [0, 21]
    top_pcts = [0.10, 0.20, 0.30]
    directions = ["long_only", "long_short"]

    grid = []
    for strategy, lookback, holding, skip, top_pct, direction in product(
        strategies, lookbacks, holdings, skip_recents, top_pcts, directions
    ):
        # Skip invalid combinations
        if skip >= lookback:
            continue
        if strategy in ("ma_cross", "rsi_reversal") and skip > 0:
            continue  # skip_recent not meaningful for these
        if strategy in ("ma_cross", "rsi_reversal") and direction == "long_short":
            # These work better long-only or need separate short logic
            pass
        grid.append({
            "strategy": strategy,
            "lookback": lookback,
            "holding": holding,
            "skip_recent": skip,
            "top_pct": top_pct,
            "direction": direction,
        })

    return grid


def run_job(job_id: int, params: dict, train_end: str | None = None) -> dict:
    """Run a single backtest job and return the result dict."""
    p = BacktestParams(**params)
    prices = load_prices_wide(DB_PATH, start="2010-01-01", end=train_end)
    result = run_backtest(prices, p)
    return {
        "job_id": job_id,
        **params,
        "sharpe": result.sharpe,
        "annual_return": result.annual_return,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "n_trades": result.n_trades,
    }


def save_result(job_id: int, result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"result_{job_id:05d}.json"
    out.write_text(json.dumps(result))


def aggregate_results() -> None:
    import pandas as pd
    files = sorted(RESULTS_DIR.glob("result_*.json"))
    if not files:
        print("No result files found in", RESULTS_DIR)
        return
    rows = [json.loads(f.read_text()) for f in files]
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    out = RESULTS_DIR / "summary.csv"
    df.to_csv(out, index=False)
    print(f"Aggregated {len(df)} results → {out}")
    print("\nTop 10 by Sharpe:")
    print(df.head(10).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-grid", action="store_true", help="Build and save param grid")
    ap.add_argument("--job-id", type=int, default=None, help="Run single job by ID")
    ap.add_argument("--run-all", action="store_true", help="Run all jobs locally")
    ap.add_argument("--aggregate", action="store_true", help="Aggregate result files → summary.csv")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers for --run-all")
    ap.add_argument("--train-end", default=None, help="YYYY-MM-DD cutoff for OOS split")
    args = ap.parse_args()

    # Read job_id from SLURM env if not passed explicitly
    if args.job_id is None:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_id is not None:
            args.job_id = int(slurm_id)

    if args.build_grid or not GRID_PATH.exists():
        grid = build_param_grid()
        GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
        GRID_PATH.write_text(json.dumps(grid, indent=2))
        print(f"Grid: {len(grid)} combinations → {GRID_PATH}")
        if args.build_grid:
            return

    grid = json.loads(GRID_PATH.read_text())

    if args.aggregate:
        aggregate_results()
        return

    if args.job_id is not None:
        if args.job_id >= len(grid):
            print(f"job_id {args.job_id} out of range (grid has {len(grid)} entries)")
            sys.exit(1)
        result = run_job(args.job_id, grid[args.job_id], args.train_end)
        save_result(args.job_id, result)
        print(json.dumps(result, indent=2))
        return

    if args.run_all:
        print(f"Running {len(grid)} jobs with {args.workers} workers …")
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_job, i, params, args.train_end): i
                for i, params in enumerate(grid)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    result = fut.result()
                    save_result(i, result)
                except Exception as e:
                    print(f"Job {i} failed: {e}")
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(grid)} done")
        aggregate_results()
        return

    ap.print_help()


if __name__ == "__main__":
    main()
