#!/usr/bin/env python3
"""Analyze Foresea's 5-minute crypto paper-signal SQLite dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import crypto_5m  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the crypto 5m paper-signal SQLite dataset.")
    parser.add_argument("--signal-db", type=Path, default=None, help="SQLite signal DB path.")
    parser.add_argument("--min-group-signals", type=int, default=1, help="Hide regime groups smaller than this.")
    parser.add_argument("--regime-report", action="store_true",
                        help="Replay per-asset strategy experts and rank regime-selector candidates.")
    parser.add_argument("--min-trades", type=int, default=20,
                        help="Minimum replay trades for a per-asset selector candidate.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output file.")
    args = parser.parse_args()

    if args.regime_report:
        result = crypto_5m.crypto_5m_per_asset_regime_report(
            db_path=args.signal_db,
            min_trades=args.min_trades,
        )
    else:
        result = crypto_5m.analyze_crypto_5m_signal_db(
            db_path=args.signal_db,
            min_group_signals=args.min_group_signals,
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
