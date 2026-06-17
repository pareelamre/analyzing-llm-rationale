#!/usr/bin/env python3
"""Walk-forward backtest for Foresea's 5-minute crypto quant/ML model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import crypto_5m  # noqa: E402


def _csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest the 5-minute crypto UP/DOWN model.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Crypto symbol, e.g. BTC, BTCUSDT, ETH.")
    parser.add_argument("--symbols", type=_csv_strings, default=None,
                        help="Comma-separated symbols for --benchmark, e.g. BTC,ETH,SOL.")
    parser.add_argument("--horizon-minutes", type=int, default=5, help="Forecast horizon, clamped to 1..15.")
    parser.add_argument("--lookback-minutes", type=int, default=240, help="Candles available to each forecast.")
    parser.add_argument("--market-probability", type=float, default=0.5,
                        help="Synthetic/observed UP contract price used for edge/PnL.")
    parser.add_argument("--target-price", type=float, default=None,
                        help="Contract target/strike price for --resolve.")
    parser.add_argument("--start-time-ms", type=int, default=None,
                        help="Market start timestamp in Unix milliseconds for --resolve.")
    parser.add_argument("--expiry-time-ms", type=int, default=None,
                        help="Market expiry timestamp in Unix milliseconds for --resolve.")
    parser.add_argument("--predicted-outcome", choices=["up", "down", "tie"], default=None,
                        help="Optional predicted outcome to score during --resolve.")
    parser.add_argument("--fee-bps", type=float, default=0.0, help="Per-trade fee in basis points.")
    parser.add_argument("--step-minutes", type=int, default=1, help="Spacing between walk-forward rows.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use only the most recent N walk-forward rows.")
    parser.add_argument("--days", type=float, default=None,
                        help="Fetch this many recent days of paginated 1m candles for --benchmark.")
    parser.add_argument("--max-candles", type=int, default=10000,
                        help="Maximum paginated candles per symbol when --days is set.")
    parser.add_argument("--ml-mode", choices=["fixed", "adaptive"], default="fixed",
                        help="Use fixed coefficients or fit an adaptive logistic overlay.")
    parser.add_argument("--training-window", type=int, default=500,
                        help="Max recent labelled examples for adaptive ML.")
    parser.add_argument("--edge-threshold", type=float, default=0.03,
                        help="Minimum model-vs-market edge before taking a trade.")
    parser.add_argument(
        "--strategy-mode",
        choices=["adaptive_model", "fade_5m_momentum", "follow_5m_momentum", "per_asset_regime_selector"],
                        default="adaptive_model",
                        help="Paper-trading strategy to record.")
    parser.add_argument("--momentum-threshold", type=float, default=0.002,
                        help="Absolute 5-minute momentum threshold for fade_5m_momentum.")
    parser.add_argument("--sweep", action="store_true",
                        help="Compare edge thresholds and ML modes instead of one backtest.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run sweep mode across multiple symbols and aggregate holdout evidence.")
    parser.add_argument("--ott-comparison", action="store_true",
                        help="Compare OTT follow/fade variants with the current per-asset selector.")
    parser.add_argument("--ott-length", type=int, default=2,
                        help="OTT VAR moving-average period for --ott-comparison.")
    parser.add_argument("--ott-percent", type=float, default=1.4,
                        help="OTT trailing percent for --ott-comparison.")
    parser.add_argument("--benchmark-log", type=Path, default=None,
                        help="When used with --benchmark, append a compact evidence record to this JSONL path.")
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve a completed 5-minute market from Binance candles.")
    parser.add_argument("--paper-signal", action="store_true",
                        help="Record a current 5-minute paper signal for later resolution.")
    parser.add_argument("--resolve-signal-log", action="store_true",
                        help="Resolve open paper signals in the JSONL signal log.")
    parser.add_argument("--import-signal-db", action="store_true",
                        help="Import JSONL paper signals into the SQLite signal database.")
    parser.add_argument("--signal-summary", action="store_true",
                        help="Summarize resolved paper-signal accuracy and PnL.")
    parser.add_argument("--paper-loop", action="store_true",
                        help="Continuously resolve expired signals and record new paper signals.")
    parser.add_argument("--backfill", action="store_true",
                        help="Reconstruct a long paper track record by replaying the live signal "
                             "logic over historical klines (--days) into the signal DB.")
    parser.add_argument("--step-minutes-backfill", type=int, default=None,
                        help="Spacing (minutes) between backfilled bars; defaults to the horizon.")
    parser.add_argument("--signal-log", type=Path, default=None,
                        help="JSONL path for --paper-signal and --resolve-signal-log.")
    parser.add_argument("--signal-db", type=Path, default=None,
                        help="SQLite path for queryable paper-signal storage.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of paper-loop iterations.")
    parser.add_argument("--sleep-seconds", type=float, default=60.0,
                        help="Seconds between paper-loop iterations.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview paper-loop signals without writing the signal log.")
    parser.add_argument("--edge-thresholds", type=_csv_floats, default=None,
                        help="Comma-separated thresholds for --sweep, e.g. 0,0.01,0.03,0.05.")
    parser.add_argument("--ml-modes", type=_csv_strings, default=None,
                        help="Comma-separated modes for --sweep, e.g. fixed,adaptive.")
    parser.add_argument("--min-trades", type=int, default=1,
                        help="Minimum trades required when selecting the best sweep threshold.")
    parser.add_argument("--min-resolved-trades", type=int, default=200,
                        help="Minimum resolved paper trades before --signal-summary says trade_ready.")
    parser.add_argument("--min-total-pnl", type=float, default=0.0,
                        help="Minimum total paper PnL per contract for --signal-summary trade_ready.")
    parser.add_argument("--min-hit-rate", type=float, default=0.53,
                        help="Minimum paper trade hit rate for --signal-summary trade_ready.")
    parser.add_argument("--selection-fraction", type=float, default=0.6,
                        help="Earlier-row fraction used to select sweep thresholds before holdout scoring.")
    parser.add_argument("--folds", type=int, default=1,
                        help="Chronological folds for repeated threshold-selection holdout scoring.")
    parser.add_argument("--include-rows", action="store_true", help="Include per-row forecasts and PnL.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output file.")
    args = parser.parse_args()

    klines = None
    if args.days is not None and not args.benchmark and not args.backfill:
        klines = crypto_5m.fetch_klines_history(
            args.symbol,
            days=args.days,
            max_candles=args.max_candles,
        )

    if args.paper_loop:
        result = crypto_5m.run_crypto_5m_paper_loop(
            args.symbols or [args.symbol],
            iterations=args.iterations,
            sleep_seconds=args.sleep_seconds,
            path=args.signal_log,
            market_probability=args.market_probability,
            fee_bps=args.fee_bps,
            horizon_minutes=args.horizon_minutes,
            lookback_minutes=args.lookback_minutes,
            ml_mode=args.ml_mode,
            training_window=args.training_window,
            edge_threshold=args.edge_threshold,
            strategy_mode=args.strategy_mode,
            momentum_threshold=args.momentum_threshold,
            db_path=args.signal_db,
            dry_run=args.dry_run,
        )
    elif args.backfill:
        backfill_days = args.days if args.days is not None else 30.0
        # Ensure we fetch enough 1-minute candles to cover the requested window
        # (~1440/day) even if --max-candles was left at its small default.
        needed_candles = int(backfill_days * 1440) + 240
        result = crypto_5m.backfill_crypto_5m_signals(
            args.symbols or [args.symbol],
            days=backfill_days,
            horizon_minutes=args.horizon_minutes,
            lookback_minutes=args.lookback_minutes,
            market_probability=args.market_probability,
            fee_bps=args.fee_bps,
            ml_mode=args.ml_mode,
            training_window=args.training_window,
            strategy_mode=args.strategy_mode,
            momentum_threshold=args.momentum_threshold,
            step_minutes=args.step_minutes_backfill,
            max_candles=max(args.max_candles, needed_candles),
            path=args.signal_log,
            db_path=args.signal_db,
        )
    elif args.paper_signal:
        result = crypto_5m.record_crypto_5m_signal(
            args.symbol,
            target_price=args.target_price,
            horizon_minutes=args.horizon_minutes,
            lookback_minutes=args.lookback_minutes,
            market_probability=args.market_probability,
            fee_bps=args.fee_bps,
            ml_mode=args.ml_mode,
            training_window=args.training_window,
            edge_threshold=args.edge_threshold,
            strategy_mode=args.strategy_mode,
            momentum_threshold=args.momentum_threshold,
            path=args.signal_log,
            db_path=args.signal_db,
        )
    elif args.resolve_signal_log:
        result = crypto_5m.resolve_crypto_5m_signal_log(path=args.signal_log, db_path=args.signal_db)
    elif args.import_signal_db:
        result = crypto_5m.import_crypto_5m_signal_log_to_db(
            path=args.signal_log,
            db_path=args.signal_db,
        )
    elif args.signal_summary:
        result = crypto_5m.summarize_crypto_5m_signal_log(
            path=args.signal_log,
            min_resolved_trades=args.min_resolved_trades,
            min_total_pnl=args.min_total_pnl,
            min_hit_rate=args.min_hit_rate,
        )
    elif args.resolve:
        if args.target_price is None:
            parser.error("--resolve requires --target-price")
        result = crypto_5m.resolve_crypto_5m_outcome(
            args.symbol,
            target_price=args.target_price,
            start_time_ms=args.start_time_ms,
            expiry_time_ms=args.expiry_time_ms,
            horizon_minutes=args.horizon_minutes,
            predicted_outcome=args.predicted_outcome,
        )
    elif args.benchmark:
        result = crypto_5m.benchmark_crypto_5m_strategy(
            args.symbols,
            history_days=args.days,
            max_candles=args.max_candles,
            horizon_minutes=args.horizon_minutes,
            lookback_minutes=args.lookback_minutes,
            market_probability=args.market_probability,
            fee_bps=args.fee_bps,
            step_minutes=args.step_minutes,
            max_rows=args.max_rows,
            ml_modes=args.ml_modes,
            training_window=args.training_window,
            edge_thresholds=args.edge_thresholds,
            min_trades=args.min_trades,
            selection_fraction=args.selection_fraction,
            folds=args.folds,
        )
        if args.benchmark_log:
            result["benchmark_log"] = crypto_5m.append_crypto_5m_benchmark_log(
                result,
                path=args.benchmark_log,
            )
    elif args.ott_comparison:
        result = crypto_5m.compare_ott_crypto_5m_strategies(
            args.symbols or [args.symbol],
            history_days=args.days,
            max_candles=args.max_candles,
            horizon_minutes=args.horizon_minutes,
            lookback_minutes=args.lookback_minutes,
            market_probability=args.market_probability,
            fee_bps=args.fee_bps,
            step_minutes=args.step_minutes,
            max_rows=args.max_rows,
            ml_mode=args.ml_mode,
            training_window=args.training_window,
            ott_length=args.ott_length,
            ott_percent=args.ott_percent,
        )
    elif args.sweep:
        result = crypto_5m.sweep_crypto_5m_thresholds(
            args.symbol,
            klines=klines,
            horizon_minutes=args.horizon_minutes,
            lookback_minutes=args.lookback_minutes,
            market_probability=args.market_probability,
            fee_bps=args.fee_bps,
            step_minutes=args.step_minutes,
            max_rows=args.max_rows,
            ml_modes=args.ml_modes,
            training_window=args.training_window,
            edge_thresholds=args.edge_thresholds,
            min_trades=args.min_trades,
            selection_fraction=args.selection_fraction,
            folds=args.folds,
        )
    else:
        result = crypto_5m.walk_forward_backtest(
            args.symbol,
            klines=klines,
            horizon_minutes=args.horizon_minutes,
            lookback_minutes=args.lookback_minutes,
            market_probability=args.market_probability,
            fee_bps=args.fee_bps,
            step_minutes=args.step_minutes,
            max_rows=args.max_rows,
            ml_mode=args.ml_mode,
            training_window=args.training_window,
            edge_threshold=args.edge_threshold,
            include_rows=args.include_rows,
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
