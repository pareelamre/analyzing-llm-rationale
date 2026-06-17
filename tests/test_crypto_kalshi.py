from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import crypto_5m, crypto_kalshi, market_data  # noqa: E402


def _synthetic_klines(n=200, base=65000.0):
    rows = []
    for i in range(n):
        price = base + (i % 7) * 5 - 15  # small wiggle so realized vol > 0
        rows.append([i * 60000, str(price), str(price * 1.0003), str(price * 0.9997), str(price), "10", i * 60000 + 59999])
    return rows


def _fake_events(close_time, *, strike=64000.0, bid=0.50, ask=0.52):
    return {"events": [{"markets": [{
        "ticker": "KXBTCD-TEST-T64000",
        "yes_sub_title": "$64,000 or above",
        "floor_strike": strike,
        "yes_bid_dollars": bid,
        "yes_ask_dollars": ask,
        "close_time": close_time,
    }]}]}


def test_snapshot_resolve_equity():
    # Manual monkeypatching (repo style) — no network.
    orig_fetch = crypto_5m._fetch_klines
    orig_json = market_data._get_json
    orig_resolve = market_data.resolve_kalshi
    orig_mtc = crypto_kalshi._minutes_to_close
    crypto_5m._fetch_klines = lambda symbol, **_: _synthetic_klines()
    try:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            log = Path(tmp) / "kal.jsonl"
            # Snapshot: market closes in the future so it is logged as an open paper trade.
            market_data._get_json = lambda url, params=None: _fake_events("2999-01-01T00:00:00Z")
            snap = crypto_kalshi.snapshot_kalshi_btc_markets(path=log, edge_threshold=0.0)
            assert snap["new_records"] == 1, snap
            assert snap["new_trades"] == 1, snap

            # Re-snapshot is idempotent (same ticker skipped).
            snap2 = crypto_kalshi.snapshot_kalshi_btc_markets(path=log, edge_threshold=0.0)
            assert snap2["new_records"] == 0, snap2

            # Resolve: pretend the market has closed and settled YES.
            crypto_kalshi._minutes_to_close = lambda ct: -1.0
            market_data.resolve_kalshi = lambda ticker: 1
            res = crypto_kalshi.resolve_kalshi_btc_log(path=log)
            assert res["resolved_now"] == 1, res

            eq = crypto_kalshi.kalshi_btc_equity(path=log)
            assert eq["n_resolved"] == 1, eq
            assert eq["calibration"]["n"] == 1, eq
            assert eq["paper_trades"]["n_trades"] == 1, eq
            assert eq["paper_trades"]["equity_curve"], eq
            assert len(eq["paper_trades"]["equity_curve"]) == len(eq["paper_trades"]["equity_curve_ts"])
            print("crypto_kalshi OK:", eq["paper_trades"]["pnl_per_contract"], eq["calibration"]["brier"])
    finally:
        crypto_5m._fetch_klines = orig_fetch
        market_data._get_json = orig_json
        market_data.resolve_kalshi = orig_resolve
        crypto_kalshi._minutes_to_close = orig_mtc


if __name__ == "__main__":
    test_snapshot_resolve_equity()
