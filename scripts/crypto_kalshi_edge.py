#!/usr/bin/env python3
"""Measure model-vs-market edge on real Kalshi BTC threshold markets (KXBTCD).

Unlike the synthetic 5-minute 50c crypto lab, this prices the *actual* tradeable
instrument — "BTC >= strike at <fixed close time>" — and compares a diffusion
model probability to the live bid/ask, net of the bid/ask spread and Kalshi's
fee. The output is a real-instrument edge, not a paper one.

Model: zero-drift lognormal diffusion from current BTC spot and realized
per-minute volatility (Black-Scholes digital):

    d2 = (ln(S0 / K) - 0.5 * sigma_T**2) / sigma_T ,  sigma_T = sigma_min * sqrt(T_min)
    P(S_T >= K) = Phi(d2)

Trade economics per contract (payout 1.0 on win):
    buy YES at ask:  net_edge = model_p       - ask        - fee(ask)
    buy NO  at (1-bid): net_edge = (1-model_p) - (1 - bid)  - fee(1-bid)
Kalshi fee ~= 0.07 * price * (1 - price) per contract.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import crypto_5m, market_data  # noqa: E402


def _minutes_to_close(close_time: str) -> float | None:
    try:
        dt = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (dt - datetime.now(timezone.utc)).total_seconds() / 60.0


def _kalshi_fee(price: float) -> float:
    """Kalshi trading fee per contract (~0.07 * p * (1-p))."""
    p = max(0.0, min(1.0, price))
    return 0.07 * p * (1.0 - p)


def _realized_sigma_per_min(closes: list[float]) -> float | None:
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(rets) < 30:
        return None
    return pstdev(rets)


def measure(*, series: str, symbol: str, vol_window: int, min_mid: float, max_mid: float) -> dict:
    raw = crypto_5m._fetch_klines(crypto_5m.normalize_symbol(symbol), limit=vol_window)
    closes = crypto_5m._parse_closes(raw)
    if len(closes) < 30:
        raise SystemExit("Not enough BTC candles to estimate volatility.")
    spot = closes[-1]
    sigma_min = _realized_sigma_per_min(closes)
    if not sigma_min:
        raise SystemExit("Could not estimate realized volatility.")

    data = market_data._get_json(market_data.KALSHI_EVENTS_URL, params={
        "series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 20,
    })
    events = data.get("events", []) if isinstance(data, dict) else []

    rows = []
    for event in events:
        for mk in event.get("markets", []) or []:
            q = market_data._kalshi_quote(mk)
            strike = q.get("floor_strike")
            bid, ask = q.get("yes_bid"), q.get("yes_ask")
            if strike is None or bid is None or ask is None or ask <= bid:
                continue
            mid = (bid + ask) / 2.0
            if not (min_mid <= mid <= max_mid):
                continue
            t_min = _minutes_to_close(q.get("close_time"))
            if not t_min or t_min <= 0:
                continue
            sigma_t = sigma_min * math.sqrt(t_min)
            if sigma_t <= 0:
                continue
            d2 = (math.log(spot / float(strike)) - 0.5 * sigma_t ** 2) / sigma_t
            model_p = crypto_5m._normal_cdf(d2)

            edge_yes = model_p - ask - _kalshi_fee(ask)
            edge_no = (1.0 - model_p) - (1.0 - bid) - _kalshi_fee(1.0 - bid)
            best_edge = max(edge_yes, edge_no)
            side = "BUY YES" if edge_yes >= edge_no else "BUY NO"
            rows.append({
                "ticker": mk.get("ticker"),
                "strike": float(strike),
                "close_in_h": round(t_min / 60.0, 1),
                "spot": round(spot, 1),
                "model_p": round(model_p, 4),
                "bid": bid, "ask": ask, "mid": round(mid, 4),
                "spread": round(ask - bid, 3),
                "raw_edge_vs_mid": round(model_p - mid, 4),
                "net_edge": round(best_edge, 4),
                "side": side,
                "tradeable": best_edge > 0,
            })

    rows.sort(key=lambda r: r["net_edge"], reverse=True)
    tradeable = [r for r in rows if r["tradeable"]]
    return {
        "series": series,
        "spot": round(spot, 1),
        "sigma_per_min": round(sigma_min, 6),
        "sigma_per_min_annualized_pct": round(sigma_min * math.sqrt(525600) * 100, 1),
        "vol_window_min": len(closes),
        "n_markets": len(rows),
        "n_tradeable_after_costs": len(tradeable),
        "median_spread": round(sorted(r["spread"] for r in rows)[len(rows) // 2], 3) if rows else None,
        "median_abs_raw_edge": round(sorted(abs(r["raw_edge_vs_mid"]) for r in rows)[len(rows) // 2], 4) if rows else None,
        "top": rows[:12],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Model-vs-market edge on real Kalshi BTC markets.")
    ap.add_argument("--series", default="KXBTCD", help="Kalshi series ticker (KXBTCD = BTC threshold).")
    ap.add_argument("--symbol", default="BTC", help="Spot/vol symbol on Binance.")
    ap.add_argument("--vol-window", type=int, default=1000, help="1-min candles for realized-vol estimate.")
    ap.add_argument("--min-mid", type=float, default=0.10, help="Only score markets with mid >= this.")
    ap.add_argument("--max-mid", type=float, default=0.90, help="Only score markets with mid <= this.")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = ap.parse_args()
    result = measure(series=args.series, symbol=args.symbol, vol_window=args.vol_window,
                     min_mid=args.min_mid, max_mid=args.max_mid)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
