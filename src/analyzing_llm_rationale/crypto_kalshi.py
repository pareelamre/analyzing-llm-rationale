"""Real-instrument model-vs-market edge + calibration for Kalshi BTC markets.

Unlike the synthetic 5-minute 50c crypto lab, this tracks the *actual* tradeable
KXBTCD threshold market ("BTC >= strike at a fixed close time"):

1. ``snapshot_kalshi_btc_markets`` records, once per market, the model's
   probability vs the live bid/ask, the chosen side, and the entry price net of
   spread + Kalshi fee (a paper trade is taken only when net edge > threshold).
2. ``resolve_kalshi_btc_log`` scores past-close records against the venue's own
   settlement (``market_data.resolve_kalshi``) — true ground truth.
3. ``kalshi_btc_equity`` builds a paper-equity curve for the +edge trades plus a
   calibration report (Brier + reliability buckets) of model_p vs realised
   outcome. That calibration is the real go/no-go the synthetic lab can't give.

Model: zero-drift lognormal diffusion (Black-Scholes digital) from current spot
and realised per-minute volatility.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Dict, List, Optional

from analyzing_llm_rationale import crypto_5m, market_data

DEFAULT_KALSHI_EDGE_LOG_PATH = Path("data/crypto_kalshi_edge_log.jsonl")
DEFAULT_KALSHI_EDGE_PAYLOAD_PATH = Path("static/crypto_kalshi_edge_payload.json")
DEFAULT_KALSHI_EDGE_REMOTE_URL = (
    "https://raw.githubusercontent.com/pareelamre/analyzing-llm-rationale/"
    "main/static/crypto_kalshi_edge_payload.json"
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime] = None) -> str:
    return (dt or _now()).isoformat()


def _minutes_to_close(close_time: Any) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (dt - _now()).total_seconds() / 60.0


def kalshi_fee(price: float) -> float:
    """Kalshi trading fee per contract (~0.07 * p * (1-p))."""
    p = max(0.0, min(1.0, float(price)))
    return round(0.07 * p * (1.0 - p), 4)


def realized_sigma_per_min(closes: List[float]) -> Optional[float]:
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(rets) < 30:
        return None
    return pstdev(rets)


def model_prob_above(spot: float, strike: float, sigma_min: float, minutes: float) -> Optional[float]:
    """P(S_T >= strike) under a zero-drift lognormal diffusion."""
    if spot <= 0 or strike <= 0 or sigma_min <= 0 or minutes <= 0:
        return None
    sigma_t = sigma_min * math.sqrt(minutes)
    if sigma_t <= 0:
        return None
    d2 = (math.log(spot / strike) - 0.5 * sigma_t ** 2) / sigma_t
    return crypto_5m._normal_cdf(d2)


def _spot_and_sigma(symbol: str, vol_window: int) -> tuple[float, float]:
    raw = crypto_5m._fetch_klines(crypto_5m.normalize_symbol(symbol), limit=vol_window)
    closes = crypto_5m._parse_closes(raw)
    if len(closes) < 30:
        raise crypto_5m.CryptoModelError("Not enough candles to estimate volatility.")
    sigma_min = realized_sigma_per_min(closes)
    if not sigma_min:
        raise crypto_5m.CryptoModelError("Could not estimate realized volatility.")
    return closes[-1], sigma_min


def snapshot_kalshi_btc_markets(
    *,
    series: str = "KXBTCD",
    symbol: str = "BTC",
    vol_window: int = 1000,
    min_mid: float = 0.10,
    max_mid: float = 0.90,
    edge_threshold: float = 0.02,
    basis: float = 0.0,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record one paper entry per *newly seen* KXBTCD market (model vs live quote).

    Each market is logged once, the first time we see it contested, and held to
    settlement — so re-running is cheap and idempotent (already-seen tickers are
    skipped). A paper trade ``is_trade`` is taken only when the net edge (after
    spread + fee) clears ``edge_threshold``.
    """
    log_path = path or DEFAULT_KALSHI_EDGE_LOG_PATH
    existing = crypto_5m._jsonl_records(log_path)
    seen = {r.get("ticker") for r in existing if r.get("type") == "kalshi_btc_edge"}

    spot, sigma_min = _spot_and_sigma(symbol, vol_window)
    data = market_data._get_json(market_data.KALSHI_EVENTS_URL, params={
        "series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 20,
    })
    events = data.get("events", []) if isinstance(data, dict) else []

    new_records: List[Dict[str, Any]] = []
    for event in events:
        for mk in event.get("markets", []) or []:
            ticker = mk.get("ticker")
            if not ticker or ticker in seen:
                continue
            q = market_data._kalshi_quote(mk)
            strike, bid, ask = q.get("floor_strike"), q.get("yes_bid"), q.get("yes_ask")
            if strike is None or bid is None or ask is None or ask <= bid:
                continue
            mid = (bid + ask) / 2.0
            if not (min_mid <= mid <= max_mid):
                continue
            t_min = _minutes_to_close(q.get("close_time"))
            if not t_min or t_min <= 0:
                continue
            model_p = model_prob_above(spot + basis, float(strike), sigma_min, t_min)
            if model_p is None:
                continue
            edge_yes = model_p - ask - kalshi_fee(ask)
            edge_no = (1.0 - model_p) - (1.0 - bid) - kalshi_fee(1.0 - bid)
            if edge_yes >= edge_no:
                side, entry, net_edge = "YES", ask, edge_yes
            else:
                side, entry, net_edge = "NO", 1.0 - bid, edge_no
            new_records.append({
                "type": "kalshi_btc_edge",
                "status": "open",
                "logged_at": _iso(),
                "ticker": ticker,
                "question": q.get("question"),
                "strike": float(strike),
                "close_time": q.get("close_time"),
                "minutes_to_close": round(t_min, 1),
                "snapshot_spot": round(spot, 2),
                "sigma_per_min": round(sigma_min, 6),
                "model_p": round(model_p, 4),
                "market_bid": bid,
                "market_ask": ask,
                "market_mid": round(mid, 4),
                "side": side,
                "entry_price": round(entry, 4),
                "fee": kalshi_fee(entry),
                "net_edge": round(net_edge, 4),
                "is_trade": net_edge > edge_threshold,
            })
            seen.add(ticker)

    if new_records:
        crypto_5m._write_jsonl(log_path, existing + new_records)
    return {
        "path": str(log_path),
        "spot": round(spot, 2),
        "sigma_per_min": round(sigma_min, 6),
        "seen_total": len(seen),
        "new_records": len(new_records),
        "new_trades": sum(1 for r in new_records if r["is_trade"]),
    }


def _kalshi_candlestick_quote(series: str, ticker: str, at_ts: int) -> Optional[tuple[float, float]]:
    """Historical (yes_bid, yes_ask) at or just before ``at_ts`` (unix seconds)."""
    base = market_data.KALSHI_API_URL.rsplit("/markets", 1)[0]
    url = f"{base}/series/{series}/markets/{ticker}/candlesticks"
    # Minute candles: these markets only trade in their final ~hour, so hourly
    # candles are empty/aggregated. Take the last *traded* minute at or before
    # the decision time (volume > 0), within a 90-minute lookback window.
    try:
        data = market_data._get_json(url, params={
            "start_ts": at_ts - 5400, "end_ts": at_ts, "period_interval": 1,
        })
    except Exception:
        return None
    candles = data.get("candlesticks") if isinstance(data, dict) else None
    if not candles:
        return None
    liquid = [c for c in candles
              if int(c.get("end_period_ts") or 0) <= at_ts and float(c.get("volume_fp") or 0) > 0]
    if not liquid:
        return None
    c = liquid[-1]
    bid = (c.get("yes_bid") or {}).get("close_dollars")
    ask = (c.get("yes_ask") or {}).get("close_dollars")
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= bid <= ask <= 1.0) or ask <= bid:
        return None
    return bid, ask


class _SpotIndex:
    """BTC 1-minute spot history indexed for spot/vol lookups at a past instant."""

    def __init__(self, raw: List[Any]):
        pts = []
        for row in raw:
            try:
                pts.append((int(row[0]), float(row[4])))
            except (TypeError, ValueError, IndexError):
                continue
        pts.sort()
        self.ms = [p[0] for p in pts]
        self.close = [p[1] for p in pts]

    def _idx(self, ms: int) -> int:
        import bisect
        return bisect.bisect_right(self.ms, ms) - 1

    def spot_at(self, ms: int) -> Optional[float]:
        i = self._idx(ms)
        return self.close[i] if i >= 0 else None

    def sigma_at(self, ms: int, window_min: int) -> Optional[float]:
        hi = self._idx(ms)
        if hi < 30:
            return None
        lo = max(1, hi - window_min)
        window = self.close[lo - 1: hi + 1]
        return realized_sigma_per_min(window)


def backfill_kalshi_btc(
    *,
    series: str = "KXBTCD",
    symbol: str = "BTC",
    max_markets: int = 300,
    decision_lead_min: int = 60,
    vol_window_min: int = 240,
    min_mid: float = 0.10,
    max_mid: float = 0.90,
    edge_threshold: float = 0.02,
    basis: Optional[float] = None,
    vol_mult: float = 1.0,
    skip_quotes: bool = False,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reconstruct calibration + paper trades from *already-settled* Kalshi BTC
    markets, instead of waiting for live ones to resolve.

    Calibration (model_p vs the venue's real result) needs only the strike, close
    time and Binance spot history — computed for every settled market. Paper
    trades additionally need the historical entry quote, pulled from the market's
    candlesticks at ``decision_lead_min`` before close.
    """
    log_path = path or DEFAULT_KALSHI_EDGE_LOG_PATH
    # 1. Page through settled markets.
    settled: List[Dict[str, Any]] = []
    cursor = None
    while len(settled) < max_markets:
        params = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = market_data._get_json(market_data.KALSHI_API_URL, params=params)
        mks = data.get("markets", []) if isinstance(data, dict) else []
        for x in mks:
            strike, result, ct = x.get("floor_strike"), x.get("result"), x.get("close_time")
            if strike is None or result not in ("yes", "no") or not ct:
                continue
            try:
                close_dt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
            except ValueError:
                continue
            settled.append({"ticker": x.get("ticker"), "strike": float(strike),
                            "result": 1 if result == "yes" else 0,
                            "close_ms": int(close_dt.timestamp() * 1000)})
            if len(settled) >= max_markets:
                break
        cursor = data.get("cursor")
        if not cursor or not mks:
            break
    if not settled:
        return {"error": "no settled markets returned"}

    # 2. One BTC spot-history fetch covering the whole window.
    min_decision = min(s["close_ms"] for s in settled) - (decision_lead_min + vol_window_min) * 60_000
    max_close = max(s["close_ms"] for s in settled)
    days = (max_close - min_decision) / 86_400_000 + 0.5
    raw = crypto_5m.fetch_klines_history(crypto_5m.normalize_symbol(symbol), days=days, max_candles=50_000)
    index = _SpotIndex(raw)
    spot_floor_ms = index.ms[0] if index.ms else max_close

    # Estimate the Binance↔Kalshi settlement-index basis from the settled strike
    # ladders themselves (no Kalshi index API needed): per event, the true
    # settlement sits between the highest YES strike and the lowest NO strike, so
    # (that midpoint − Binance spot at close) is the basis. The model uses Binance
    # spot, which runs above Kalshi's index, biasing P(BTC ≥ strike) upward.
    if basis is None:
        by_event: Dict[int, List[Dict[str, Any]]] = {}
        for s in settled:
            by_event.setdefault(s["close_ms"], []).append(s)
        basis_samples = []
        for close_ms, group in by_event.items():
            yes = [g["strike"] for g in group if g["result"] == 1]
            no = [g["strike"] for g in group if g["result"] == 0]
            close_spot = index.spot_at(close_ms)
            if yes and no and close_spot:
                basis_samples.append((max(yes) + min(no)) / 2.0 - close_spot)
        basis = round(median(basis_samples), 2) if basis_samples else 0.0

    # 3. Build a resolved record per market (calibration always; trade when quoted).
    records: List[Dict[str, Any]] = []
    n_trades = quoted = 0
    for s in settled:
        decision_ms = s["close_ms"] - decision_lead_min * 60_000
        if decision_ms < spot_floor_ms:
            continue  # outside our spot-history window
        spot = index.spot_at(decision_ms)
        sigma = index.sigma_at(decision_ms, vol_window_min)
        if not spot or not sigma:
            continue
        # Apply the estimated index basis to spot and the vol multiplier to sigma.
        model_p = model_prob_above(spot + basis, s["strike"], sigma * max(0.01, vol_mult), decision_lead_min)
        if model_p is None:
            continue
        outcome = s["result"]
        rec = {
            "type": "kalshi_btc_edge", "status": "resolved", "source": "backfill",
            "logged_at": _iso(), "resolved_at": _iso(),
            "ticker": s["ticker"], "strike": s["strike"],
            "close_time": datetime.fromtimestamp(s["close_ms"] / 1000, timezone.utc).isoformat(),
            "minutes_to_close": decision_lead_min,
            "snapshot_spot": round(spot, 2), "sigma_per_min": round(sigma, 6),
            "model_p": round(model_p, 4),
            "outcome": outcome,
            "model_correct": (model_p >= 0.5) == (outcome == 1),
            "model_brier": round((model_p - outcome) ** 2, 6),
            "is_trade": False,
        }
        # Only fetch a quote for near-the-money strikes (the rest never trade), so
        # the candlestick calls stay bounded.
        quote = None
        if not skip_quotes and 0.05 <= model_p <= 0.95:
            quote = _kalshi_candlestick_quote(series, s["ticker"], decision_ms // 1000)
        if quote:
            quoted += 1
            bid, ask = quote
            mid = (bid + ask) / 2.0
            if min_mid <= mid <= max_mid:
                edge_yes = model_p - ask - kalshi_fee(ask)
                edge_no = (1.0 - model_p) - (1.0 - bid) - kalshi_fee(1.0 - bid)
                if edge_yes >= edge_no:
                    side, entry, net_edge = "YES", ask, edge_yes
                else:
                    side, entry, net_edge = "NO", 1.0 - bid, edge_no
                rec.update({
                    "market_bid": bid, "market_ask": ask, "market_mid": round(mid, 4),
                    "side": side, "entry_price": round(entry, 4), "fee": kalshi_fee(entry),
                    "net_edge": round(net_edge, 4), "is_trade": net_edge > edge_threshold,
                })
                if rec["is_trade"]:
                    won = (outcome == 1) if side == "YES" else (outcome == 0)
                    rec["won"] = won
                    rec["pnl_per_contract"] = round((1.0 - entry if won else -entry) - kalshi_fee(entry), 4)
                    n_trades += 1
        records.append(rec)

    if records:
        existing = [r for r in crypto_5m._jsonl_records(log_path)
                    if r.get("type") == "kalshi_btc_edge" and r.get("ticker") not in {x["ticker"] for x in settled}]
        crypto_5m._write_jsonl(log_path, existing + records)
    return {
        "path": str(log_path),
        "settled_scanned": len(settled),
        "backfilled": len(records),
        "quoted": quoted,
        "trades": n_trades,
        "decision_lead_min": decision_lead_min,
        "spot_history_days": round(days, 1),
        "basis_applied": basis,
        "vol_mult": vol_mult,
    }


def resolve_kalshi_btc_log(
    *,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Score open records whose market has settled, via the venue's own result."""
    log_path = path or DEFAULT_KALSHI_EDGE_LOG_PATH
    records = crypto_5m._jsonl_records(log_path)
    resolved_now = 0
    for record in records:
        if record.get("type") != "kalshi_btc_edge" or record.get("status") == "resolved":
            continue
        # Only attempt once the close time has passed (cheap gate before the API call).
        t_min = _minutes_to_close(record.get("close_time"))
        if t_min is not None and t_min > 0:
            continue
        outcome = market_data.resolve_kalshi(record.get("ticker"))
        if outcome is None:
            continue  # not settled yet
        model_p = float(record.get("model_p") or 0.0)
        record["status"] = "resolved"
        record["resolved_at"] = _iso()
        record["outcome"] = int(outcome)
        record["model_correct"] = (model_p >= 0.5) == (outcome == 1)
        record["model_brier"] = round((model_p - outcome) ** 2, 6)
        if record.get("is_trade"):
            won = (outcome == 1) if record.get("side") == "YES" else (outcome == 0)
            entry = float(record.get("entry_price") or 0.0)
            fee = float(record.get("fee") or 0.0)
            record["won"] = won
            record["pnl_per_contract"] = round((1.0 - entry if won else -entry) - fee, 4)
        resolved_now += 1
    if resolved_now:
        crypto_5m._write_jsonl(log_path, records)
    return {"path": str(log_path), "records": len(records), "resolved_now": resolved_now}


def _reliability_buckets(resolved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        bucket = [r for r in resolved if lo <= float(r.get("model_p") or 0.0) < hi or (hi == 1.0 and float(r.get("model_p") or 0.0) == 1.0)]
        if not bucket:
            continue
        out.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": len(bucket),
            "mean_model_p": round(sum(float(r["model_p"]) for r in bucket) / len(bucket), 4),
            "empirical_rate": round(sum(int(r["outcome"]) for r in bucket) / len(bucket), 4),
        })
    return out


def kalshi_btc_equity(
    *,
    path: Optional[Path] = None,
    since_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Paper-equity curve for +edge trades, plus model calibration vs settlement."""
    log_path = path or DEFAULT_KALSHI_EDGE_LOG_PATH
    records = [r for r in crypto_5m._jsonl_records(log_path)
               if r.get("type") == "kalshi_btc_edge"]
    resolved = [r for r in records if r.get("status") == "resolved" and r.get("outcome") is not None]
    open_n = sum(1 for r in records if r.get("status") != "resolved")

    def _close_ms(r: Dict[str, Any]) -> int:
        try:
            return int(datetime.fromisoformat(str(r.get("close_time")).replace("Z", "+00:00")).timestamp() * 1000)
        except (TypeError, ValueError):
            return 0

    resolved.sort(key=_close_ms)
    if since_hours and since_hours > 0 and resolved:
        cutoff = _close_ms(resolved[-1]) - int(since_hours * 3600 * 1000)
        resolved = [r for r in resolved if _close_ms(r) >= cutoff]

    points: List[float] = []
    times: List[int] = []
    running = staked = wins = 0.0
    trades = 0
    for r in resolved:
        if not r.get("is_trade") or r.get("pnl_per_contract") is None:
            continue
        running += float(r["pnl_per_contract"])
        staked += float(r.get("entry_price") or 0.0)
        wins += 1 if r.get("won") else 0
        trades += 1
        points.append(round(running, 4))
        times.append(_close_ms(r))

    briers = [float(r["model_brier"]) for r in resolved if r.get("model_brier") is not None]
    correct = [1 for r in resolved if r.get("model_correct")]
    contested = [r for r in resolved if 0.15 <= float(r.get("model_p") or 0.0) <= 0.85]
    cbriers = [float(r["model_brier"]) for r in contested if r.get("model_brier") is not None]
    coverage_days = None
    if times:
        coverage_days = round((times[-1] - times[0]) / (24 * 3600 * 1000), 2) if len(times) > 1 else 0.0

    peak = dd = 0.0
    for v in points:
        peak = max(peak, v)
        dd = max(dd, peak - v)

    return {
        "generated_at": _iso(),
        "instrument": "Kalshi KXBTCD (BTC >= strike at close)",
        "n_resolved": len(resolved),
        "n_open": open_n,
        "calibration": {
            "n": len(resolved),
            "brier": round(sum(briers) / len(briers), 4) if briers else None,
            "directional_accuracy": round(len(correct) / len(resolved), 4) if resolved else None,
            # Trivial deep ITM/OTM strikes (model_p ~0/1) flatter overall Brier;
            # the contested band is where calibration actually matters. A perfectly
            # calibrated coinflip scores ~0.25 there, so contested_brier < 0.25 = skill.
            "contested_n": len(contested),
            "contested_brier": round(sum(cbriers) / len(cbriers), 4) if cbriers else None,
            "coinflip_baseline": 0.25,
            "reliability": _reliability_buckets(resolved),
        },
        "paper_trades": {
            "n_trades": trades,
            "hit_rate": round(wins / trades, 4) if trades else None,
            "pnl_per_contract": round(running, 4),
            "total_staked": round(staked, 4),
            "roi": round(running / staked, 4) if staked else None,
            "max_drawdown": round(dd, 4),
            "coverage_days": coverage_days,
            "equity_curve": points,
            "equity_curve_ts": times,
        },
        "note": "Real-instrument paper trades: model probability vs live Kalshi bid/ask, "
                "scored on the venue's own settlement. Net of spread + Kalshi fee; no slippage.",
    }
