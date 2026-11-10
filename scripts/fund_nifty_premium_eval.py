#!/usr/bin/env python3
"""The pre-registered Nifty option premium backtest (docs/fund.md, "Nifty option premium backtest").

Every 21 sessions since India VIX began, the index model is refit on the data up to that day
and the options analyst's own code (fund_options) picks strikes from its distribution. Options
are priced with Black-76 at that day's India VIX, flat across strikes, with a half-spread on
every leg and the analyst's 2% charges; each structure is held to the close 21 sessions later.

- Test A: the live rule (the best structure when the model values it 5% above its cost).
- Test B: the systematic sale (the iron condor every time).

Writes reports/fund_nifty_premium_eval.json. Closes are kept under --cache.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import NormalDist, fmean, pstdev
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import fund_options as fo  # noqa: E402
from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402
from analyzing_llm_rationale import index_forecast as ifc  # noqa: E402

NIFTY_KEY = "NSE_INDEX|Nifty 50"
H = 21
LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
GRID, SPAN = 50, 0.15
MIN_HALF_SPREAD, HALF_SPREAD_SHARE = 0.5, 0.01
MIN_TRADES, BOOTSTRAP, SEED = 60, 2000, 20261110
RISK_SHARES = (0.02, 0.10)


def black(forward: float, strike: float, sd: float, kind: str) -> float:
    """Black-76 with no discounting; ``sd`` is volatility times the square root of time."""
    if sd <= 0:
        return max(forward - strike, 0.0) if kind == "CE" else max(strike - forward, 0.0)
    d1 = math.log(forward / strike) / sd + sd / 2
    n = NormalDist().cdf
    call = forward * n(d1) - strike * n(d1 - sd)
    return call if kind == "CE" else call - forward + strike


def chain(spot: float) -> list[dict]:
    lo, hi = math.floor(spot * (1 - SPAN) / GRID) * GRID, math.ceil(spot * (1 + SPAN) / GRID) * GRID
    return [{"instrument_key": f"{kind}{k}", "instrument_type": kind, "strike_price": float(k), "lot_size": 1,
             "trading_symbol": f"NIFTY {k} {kind}"} for k in range(int(lo), int(hi) + 1, GRID) for kind in ("CE", "PE")]


def quotes(contracts: list[dict], forward: float, vix: float) -> dict[str, dict]:
    """Book-like quotes from a flat implied volatility, with the pre-registered half-spread."""
    sd = vix / 100 * math.sqrt(H / 252)
    out = {}
    for c in contracts:
        mid = black(forward, c["strike_price"], sd, c["instrument_type"])
        half = max(MIN_HALF_SPREAD, HALF_SPREAD_SHARE * mid)
        out[c["instrument_key"]] = {"bid": max(mid - half, 0.0), "ask": mid + half, "last": mid, "source": "book"}
    return out


def model_distribution(closes: list[tuple[date, float]], vix: list[tuple[date, float]]) -> Optional[fo.Distribution]:
    fit = ifc.fit_horizon(closes, vix, H)
    if fit is None:
        return None
    returns = [fit["sigma"] * ifc._z_quantile(fit["z"], p) for p in LEVELS]
    spot = closes[-1][1]
    return fo.Distribution(LEVELS, returns, spot, spot)


def priced(s: fo.Structure, raw: dict) -> Optional[dict]:
    if not all(fo._tradeable(raw.get(leg.key), leg.lot_size) for leg in s.legs):
        return None
    q = s.quote(raw)
    return q if q and q["ask"] > 0 else None


def decide(dist: fo.Distribution, spot: float, vix: float) -> dict[str, Any]:
    """Test A's pick (or why none) and Test B's condor, with their spreads, at one decision."""
    wanted = fo._wanted(chain(spot), spot, dist)
    raw = quotes(wanted, spot, vix)
    candidates = fo.candidate_structures(wanted, spot, dist)
    by_kind = {s.kind: s for s in candidates}
    out: dict[str, Any] = {"a": None, "b": None, "legs": {}}
    straddle = by_kind.get("straddle")
    market = priced(straddle, raw) if straddle else None
    if market:
        mid = (market["bid"] + market["ask"]) / 2
        if abs(dist.expect(straddle.payoff) / mid - 1) <= fo.MAX_VOL_GAP:
            best = None
            for s in candidates:
                q = priced(s, raw)
                if q is None:
                    continue
                edge = (dist.expect(s.payoff) - q["ask"]) / q["ask"] - orch.OPTION_COST
                if best is None or edge > best[0]:
                    best = (edge, s, q)
            if best and best[0] >= orch.MIN_EDGE:
                out["a"] = {"structure": best[1], "cost": best[2]["ask"], "edge": best[0]}
    for kind in ("condor", "put_spread", "call_spread"):
        s = by_kind.get(kind)
        q = priced(s, raw) if s else None
        if q is None:
            continue
        entry = {"structure": s, "cost": q["ask"], "edge": (dist.expect(s.payoff) - q["ask"]) / q["ask"] - orch.OPTION_COST}
        if kind == "condor":
            out["b"] = entry
        else:
            out["legs"][kind] = entry
    return out


def net_return(entry: dict, level: float) -> float:
    cost = entry["cost"]
    return (entry["structure"].payoff(level) - cost - orch.OPTION_COST * cost) / cost


def run(closes: list[tuple[date, float]], vix: list[tuple[date, float]]) -> dict[str, list[dict]]:
    vix_by_day = dict(vix)
    trades: dict[str, list[dict]] = {"a": [], "b": [], "put_spread": [], "call_spread": []}
    i = next((j for j in range(len(closes)) if j >= ifc.WARMUP_SESSIONS + 260 and closes[j][0] in vix_by_day), None)
    while i is not None and i + H < len(closes):
        day, spot = closes[i]
        v = vix_by_day.get(day)
        dist = model_distribution(closes[: i + 1], [x for x in vix if x[0] <= day]) if v else None
        if dist is not None:
            got = decide(dist, spot, v)
            level = closes[i + H][1]
            row = {"day": day.isoformat(), "spot": spot, "vix": v, "expiry_close": level}
            for key, entry in (("a", got["a"]), ("b", got["b"]), *got["legs"].items()):
                if entry:
                    trades[key].append({**row, "kind": entry["structure"].kind, "cost": round(entry["cost"], 2),
                                        "edge": round(entry["edge"], 4), "return": net_return(entry, level)})
        i += H
    return trades


def interval(values: list[float]) -> Optional[list[float]]:
    if len(values) < 10:
        return None
    rng = random.Random(SEED)
    draws = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP))
    return [round(draws[int(0.025 * BOOTSTRAP)], 5), round(draws[int(0.975 * BOOTSTRAP) - 1], 5)]


def fund_path(returns: list[float], share: float) -> dict:
    equity, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        equity *= 1 + share * r
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1)
    years = len(returns) * H / 252
    return {"annual_return": round(equity ** (1 / years) - 1, 4) if years > 0 and equity > 0 else None,
            "largest_fall": round(worst, 4)}


def summary(trades: list[dict]) -> dict:
    rets = [t["return"] for t in trades]
    if not rets:
        return {"trades": 0}
    cum, peak, fall = 0.0, 0.0, 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        fall = min(fall, cum - peak)
    by_year: dict[str, list[float]] = {}
    for t in trades:
        by_year.setdefault(t["day"][:4], []).append(t["return"])
    return {"trades": len(rets), "first": trades[0]["day"], "last": trades[-1]["day"],
            "mean_return": round(fmean(rets), 5), "mean_return_ci": interval(rets),
            "stdev": round(pstdev(rets), 4), "win_share": round(fmean(r > 0 for r in rets), 4),
            "worst": round(min(rets), 4), "best": round(max(rets), 4),
            "largest_fall_in_cumulative_return": round(fall, 4),
            "kinds": {k: sum(t["kind"] == k for t in trades) for k in sorted({t["kind"] for t in trades})},
            "years_positive": f"{sum(fmean(v) > 0 for v in by_year.values())}/{len(by_year)}",
            "by_year": {y: round(fmean(v), 4) for y, v in sorted(by_year.items())},
            **{f"fund_at_{int(s * 100)}pct_risk": fund_path(rets, s) for s in RISK_SHARES}}


def verdict(s: dict) -> str:
    ci = s.get("mean_return_ci")
    return "GO" if s.get("trades", 0) >= MIN_TRADES and ci and ci[0] > 0 else "NO-GO"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "nifty_premium_eval")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_nifty_premium_eval.json")
    ap.add_argument("--through", type=date.fromisoformat, default=None)
    args = ap.parse_args(argv)
    through = args.through or datetime.now(timezone.utc).date()
    args.cache.mkdir(parents=True, exist_ok=True)
    series = {}
    for name, key in (("nifty", NIFTY_KEY), ("vix", ifc.VIX_KEY)):
        path = args.cache / f"{name}.json"
        if not path.is_file():
            path.write_text(json.dumps([(d.isoformat(), c) for d, c in ifc.daily_closes(key, ifc.HISTORY_START, through)]))
        series[name] = [(date.fromisoformat(d), float(c)) for d, c in json.loads(path.read_text())]
    trades = run(series["nifty"], series["vix"])
    a, b = summary(trades["a"]), summary(trades["b"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "docs/fund.md, Nifty option premium backtest (pre-registered 2026-11-10)",
        "data": {"nifty": [series["nifty"][0][0].isoformat(), series["nifty"][-1][0].isoformat()],
                 "vix": [series["vix"][0][0].isoformat(), series["vix"][-1][0].isoformat()],
                 "pricing": "Black-76 at India VIX, flat across strikes; half-spread max(0.5 pt, 1%) per leg; 2% charges"},
        "test_a_live_rule": {"verdict": verdict(a), **a},
        "test_b_systematic_condor": {"verdict": verdict(b), **b},
        "condor_sides (description only)": {"put_spread": summary(trades["put_spread"]),
                                            "call_spread": summary(trades["call_spread"])},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for key in ("test_a_live_rule", "test_b_systematic_condor"):
        r = report[key]
        print(key, r["verdict"], {k: r.get(k) for k in ("trades", "mean_return", "mean_return_ci", "win_share", "worst",
                                                        "years_positive", "fund_at_2pct_risk", "fund_at_10pct_risk")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
