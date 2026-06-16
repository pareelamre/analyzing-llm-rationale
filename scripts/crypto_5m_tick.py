#!/usr/bin/env python3
"""Advance the 5-minute crypto paper dataset and public equity payload."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import crypto_5m  # noqa: E402


def _csv_symbols(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _prune_signal_log(path: Path, *, keep_days: float) -> dict[str, int]:
    records = crypto_5m._jsonl_records(path)
    if not records:
        return {"before": 0, "after": 0}
    latest = max((int(record.get("start_time_ms") or 0) for record in records), default=0)
    cutoff = latest - int(max(1.0, float(keep_days or 7.0)) * 24 * 60 * 60 * 1000)
    kept = [
        record for record in records
        if record.get("status") != "resolved" or int(record.get("start_time_ms") or 0) >= cutoff
    ]
    crypto_5m._write_jsonl(path, kept)
    return {"before": len(records), "after": len(kept)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Advance live 5-minute crypto paper rows.")
    parser.add_argument("--symbols", type=_csv_symbols, default=["BTC", "ETH", "SOL"])
    parser.add_argument("--signal-log", type=Path, default=crypto_5m.DEFAULT_SIGNAL_LOG_PATH)
    parser.add_argument("--signal-db", type=Path, default=crypto_5m.DEFAULT_SIGNAL_DB_PATH)
    parser.add_argument("--equity-out", type=Path, default=crypto_5m.DEFAULT_EQUITY_FALLBACK_PATH)
    parser.add_argument("--keep-days", type=float, default=7.0)
    parser.add_argument("--hours", type=float, default=72.0)
    parser.add_argument("--market-probability", type=float, default=0.50)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--horizon-minutes", type=int, default=5)
    parser.add_argument("--lookback-minutes", type=int, default=60)
    parser.add_argument("--ml-mode", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--training-window", type=int, default=120)
    parser.add_argument("--strategy-mode", default="per_asset_regime_selector")
    parser.add_argument("--momentum-threshold", type=float, default=0.0025)
    args = parser.parse_args()

    args.signal_log.parent.mkdir(parents=True, exist_ok=True)
    args.signal_db.parent.mkdir(parents=True, exist_ok=True)
    args.equity_out.parent.mkdir(parents=True, exist_ok=True)

    resolved = crypto_5m.resolve_crypto_5m_signal_log(
        path=args.signal_log,
        db_path=args.signal_db,
    )
    signals = []
    errors = []
    for symbol in args.symbols:
        try:
            signals.append(crypto_5m.record_crypto_5m_signal(
                symbol,
                path=args.signal_log,
                db_path=args.signal_db,
                horizon_minutes=args.horizon_minutes,
                lookback_minutes=args.lookback_minutes,
                market_probability=args.market_probability,
                fee_bps=args.fee_bps,
                ml_mode=args.ml_mode,
                training_window=args.training_window,
                strategy_mode=args.strategy_mode,
                momentum_threshold=args.momentum_threshold,
            ))
        except crypto_5m.CryptoModelError as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    prune = _prune_signal_log(args.signal_log, keep_days=args.keep_days)
    crypto_5m.import_crypto_5m_signal_log_to_db(path=args.signal_log, db_path=args.signal_db)
    equity = crypto_5m.crypto_5m_candidate_equity(db_path=args.signal_db, since_hours=args.hours)
    args.equity_out.write_text(json.dumps(equity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "resolved": resolved,
        "signals": len(signals),
        "errors": errors,
        "prune": prune,
        "equity_out": str(args.equity_out),
        "curves": [
            {
                "key": curve.get("key"),
                "trades": curve.get("trades"),
                "hit_rate": curve.get("hit_rate"),
                "pnl_per_contract": curve.get("pnl_per_contract"),
            }
            for curve in equity.get("curves", [])
        ],
    }, indent=2, sort_keys=True))
    return 1 if errors and not signals else 0


if __name__ == "__main__":
    raise SystemExit(main())
