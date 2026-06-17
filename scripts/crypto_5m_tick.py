#!/usr/bin/env python3
"""Advance the 5-minute crypto paper dataset and public equity payload."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import crypto_5m, crypto_kalshi  # noqa: E402


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
    parser.add_argument("--seed", type=Path, default=Path("data/crypto_5m_backfill_seed.jsonl.gz"),
                        help="Compact historical-backfill seed imported into the signal DB so the "
                             "equity curve keeps its long-term history (the live log is pruned).")
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
    parser.add_argument("--kalshi-log", type=Path, default=crypto_kalshi.DEFAULT_KALSHI_EDGE_LOG_PATH,
                        help="JSONL log of real Kalshi BTC model-vs-market paper trades.")
    parser.add_argument("--kalshi-out", type=Path, default=crypto_kalshi.DEFAULT_KALSHI_EDGE_PAYLOAD_PATH,
                        help="Public payload for the real-instrument calibration + equity curve.")
    parser.add_argument("--kalshi-edge-threshold", type=float, default=0.02,
                        help="Min net edge (after spread+fee) to take a Kalshi paper trade.")
    parser.add_argument("--kalshi-backfill-markets", type=int, default=6000,
                        help="How many recent settled KXBTCD markets to score for calibration each tick.")
    parser.add_argument("--no-kalshi", action="store_true", help="Skip the Kalshi real-instrument step.")
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
    # Seed the long-term backfill first (immutable, committed), then layer the
    # live log on top. The DB is rebuilt every run, so without the seed the curve
    # would only ever span the pruned live window.
    seed = None
    if args.seed and args.seed.exists():
        seed = crypto_5m.import_crypto_5m_signal_log_to_db(path=args.seed, db_path=args.signal_db)
    crypto_5m.import_crypto_5m_signal_log_to_db(path=args.signal_log, db_path=args.signal_db)
    equity = crypto_5m.crypto_5m_candidate_equity(db_path=args.signal_db, since_hours=args.hours)
    args.equity_out.write_text(json.dumps(equity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Real-instrument leg: snapshot/resolve actual Kalshi BTC markets so the public
    # chart shows a model-vs-market calibration + paper equity on a tradeable
    # contract, not the synthetic 50c toy. Best-effort; never fails the tick.
    kalshi = None
    if not args.no_kalshi:
        try:
            args.kalshi_log.parent.mkdir(parents=True, exist_ok=True)
            args.kalshi_out.parent.mkdir(parents=True, exist_ok=True)
            # Historic backfill (ephemeral log, regenerated from the API each tick):
            # estimates the Binance↔Kalshi index basis and scores already-settled
            # markets, so calibration is real immediately. Then the forward snapshot
            # (small committed log) reuses that basis for the eventual live ROI.
            import tempfile
            bf_log = Path(tempfile.gettempdir()) / "crypto_kalshi_backfill.jsonl"
            bf = crypto_kalshi.backfill_kalshi_btc(
                path=bf_log, max_markets=args.kalshi_backfill_markets, skip_quotes=True)
            basis = float(bf.get("basis_applied") or 0.0)
            cal = crypto_kalshi.kalshi_btc_equity(path=bf_log)
            snap = crypto_kalshi.snapshot_kalshi_btc_markets(
                path=args.kalshi_log, edge_threshold=args.kalshi_edge_threshold, basis=basis)
            kres = crypto_kalshi.resolve_kalshi_btc_log(path=args.kalshi_log)
            fwd = crypto_kalshi.kalshi_btc_equity(path=args.kalshi_log)
            kpayload = {
                "generated_at": cal["generated_at"],
                "instrument": cal["instrument"],
                "basis": basis,
                "n_resolved": cal["n_resolved"],   # historic calibration sample
                "n_open": fwd["n_open"],            # forward open paper trades
                "calibration": cal["calibration"],
                "paper_trades": fwd["paper_trades"],
                "note": cal["note"],
            }
            args.kalshi_out.write_text(json.dumps(kpayload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            kalshi = {"backfill": bf, "snapshot": snap, "resolve": kres,
                      "n_resolved": cal["n_resolved"], "n_open": fwd["n_open"]}
        except Exception as exc:  # noqa: BLE001 — never let the real-instrument leg break the tick
            kalshi = {"error": str(exc)}

    print(json.dumps({
        "resolved": resolved,
        "signals": len(signals),
        "errors": errors,
        "seed": seed,
        "kalshi": kalshi,
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
