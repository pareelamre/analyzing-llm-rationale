"""Does the long-run "tide" flag on pair signals earn its place? (commodity_pairs.TIDE_Z)

    python scripts/pair_tide_backtest.py [--json out.json]

Replays the live pairs rule (commodity_pairs.history + simulate: hedged 60-session z past ENTRY_Z,
plain ratio confirming, exit at EXIT_Z or MAX_HOLD, costs) on FRED daily oil and gas since 1997.
Each trade is tagged with the pair's 10-year position at entry, using only months before the entry
month and the entry day's ratio (no look-ahead): "toward" if the trade bets the ratio moves toward
its 10-year average, "away" if it bets on a move further out, "neutral" if the ratio is within
the cut-off.

It then asks three questions:
  1. Do away trades do worse than toward trades, and is the gap bigger than chance (bootstrap)?
  2. Picking the cut-off on 2007-2016 only, does acting on it (skipping or halving away trades)
     improve 2017-2026, the years it never saw?
  3. Does the same hold on IMF monthly metals since 1992 (a rougher proxy: 2-month moves)?
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import commodity_pairs as cp  # noqa: E402
from analyzing_llm_rationale import pair_history as ph  # noqa: E402

CUTS = (0.0, 0.5, 1.0, 1.5)
SPLIT = "2017-01-01"
SEED = 7


def long_z(months: list[str], monthly: np.ndarray, day: date, now: float) -> float | None:
    """The day's log ratio against the 120 complete months before it."""
    key = day.isoformat()[:7]
    past = [v for m, v in zip(months, monthly) if m < key][-ph.WINDOW_MONTHS:]
    if len(past) < ph.WINDOW_MONTHS:
        return None
    w = np.array(past)
    sd = float(np.std(w, ddof=1))
    return (now - float(w.mean())) / sd if sd > 0 else None


def tag(side: int, lz: float, cut: float) -> str:
    if abs(lz) < cut:
        return "neutral"
    # side +1 buys the ratio (bets it rises): toward the average when the ratio sits below it.
    return "toward" if side * lz < 0 else "away"


def daily_trades() -> list[dict]:
    series = {k: dict(ph._fred_csv(v, http_get=requests.get, start=ph.DAILY_FROM))
              for k, v in cp.FRED_SERIES.items()}
    out = []
    for pair in cp.PAIRS:
        if pair["a"] not in series or pair["b"] not in series:
            continue
        a, b = series[pair["a"]], series[pair["b"]]
        days = [date.fromisoformat(d) for d in sorted(set(a) & set(b)) if a[d] > 0 and b[d] > 0]
        la = np.log([a[d.isoformat()] for d in days])
        lb = np.log([b[d.isoformat()] for d in days])
        by_month: dict[str, list[float]] = defaultdict(list)
        for d, x in zip(days, la - lb):
            by_month[d.isoformat()[:7]].append(float(x))
        months = sorted(by_month)
        monthly = np.array([np.mean(by_month[m]) for m in months])
        for t in cp.simulate(days, la, lb, cp.history(days, la, lb)):
            if t.get("status") not in ("reverted", "timed_out"):
                continue
            day = date.fromisoformat(t["entry_date"])
            lz = long_z(months, monthly, day, float(la[t["i"]] - lb[t["i"]]))
            if lz is None:
                continue
            out.append({"pair": pair["key"], "date": t["entry_date"], "side": t["side"], "lz": lz,
                        "ret": t["spread_return"]})
    return out


def monthly_trades() -> list[dict]:
    """Proxy on IMF monthly metals: a 2-month ratio move at least 2 sd out, bet back for one month."""
    monthly = ph.fetch_monthly(date.today())
    out = []
    for pair in cp.PAIRS:
        if pair["a"] not in ph.IMF_MONTHLY or pair["b"] not in ph.IMF_MONTHLY:
            continue
        months, r = ph.monthly_ratio(pair, monthly)
        lw = ph.WINDOW_MONTHS
        t = lw + 2
        while t < len(r) - 1:
            moves = r[t - lw + 1:t + 1] - r[t - lw - 1:t - 1]
            z = (r[t] - r[t - 2] - moves.mean()) / moves.std(ddof=1)
            if abs(z) < cp.ENTRY_Z:
                t += 1
                continue
            side = -1 if z > 0 else 1
            past = r[t - lw:t]
            lz = (r[t] - past.mean()) / past.std(ddof=1)
            out.append({"pair": pair["key"], "date": months[t] + "-01", "side": side, "lz": float(lz),
                        "ret": float(side * (r[t + 1] - r[t]) - 2 * cp.COST_PER_SIDE)})
            t += 2
    return out


def stats(rets: list[float], years: float) -> dict:
    if not rets:
        return {"n": 0}
    r = np.array(rets)
    curve = np.cumsum(r)
    drawdown = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], curve])) - np.concatenate([[0.0], curve])))
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    per_year = len(r) / years
    return {"n": len(r), "mean": round(float(r.mean()), 4), "won": round(float((r > 0).mean()), 3),
            "total": round(float(r.sum()), 4), "max_drawdown": round(drawdown, 4),
            "sharpe": round(float(r.mean()) / sd * np.sqrt(per_year), 2) if sd > 0 else None}


def variants(trades: list[dict], cut: float, years: float) -> dict:
    base = [t["ret"] for t in trades]
    skip = [t["ret"] for t in trades if tag(t["side"], t["lz"], cut) != "away"]
    half = [t["ret"] * (0.5 if tag(t["side"], t["lz"], cut) == "away" else 1.0) for t in trades]
    return {"all": stats(base, years), "skip_away": stats(skip, years), "half_away": stats(half, years)}


def gap(trades: list[dict], cut: float, n_boot: int = 10000) -> dict:
    toward = np.array([t["ret"] for t in trades if tag(t["side"], t["lz"], cut) == "toward"])
    away = np.array([t["ret"] for t in trades if tag(t["side"], t["lz"], cut) == "away"])
    if len(toward) < 3 or len(away) < 3:
        return {"toward": len(toward), "away": len(away)}
    rng = np.random.default_rng(SEED)
    diffs = np.array([rng.choice(toward, len(toward)).mean() - rng.choice(away, len(away)).mean()
                      for _ in range(n_boot)])
    return {"toward": stats(list(toward), 1)["n"], "away": len(away),
            "toward_mean": round(float(toward.mean()), 4), "away_mean": round(float(away.mean()), 4),
            "toward_won": round(float((toward > 0).mean()), 3), "away_won": round(float((away > 0).mean()), 3),
            "gap_mean": round(float(toward.mean() - away.mean()), 4),
            "gap_95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            "share_gap_positive": round(float((diffs > 0).mean()), 3)}


def span_years(trades: list[dict]) -> float:
    if not trades:
        return 1.0
    ds = sorted(date.fromisoformat(t["date"]) for t in trades)
    return max(1.0, (ds[-1] - ds[0]).days / 365.25)


def study(name: str, trades: list[dict]) -> dict:
    early = [t for t in trades if t["date"] < SPLIT]
    late = [t for t in trades if t["date"] >= SPLIT]
    # Pick the cut-off on the early years only: the one whose skip-away variant has the best Sharpe.
    picks = {c: variants(early, c, span_years(early))["skip_away"].get("sharpe") or -9 for c in CUTS}
    chosen = max(picks, key=lambda c: (picks[c], -c))
    return {
        "name": name, "trades": len(trades), "from": min(t["date"] for t in trades) if trades else None,
        "gap_by_cut": {str(c): gap(trades, c) for c in CUTS},
        "early_pick": {"by_cut": {str(c): v for c, v in picks.items()}, "chosen": chosen},
        "late_with_early_pick": variants(late, chosen, span_years(late)),
        "late_at_0_5": variants(late, 0.5, span_years(late)),
        "full_at_0_5": variants(trades, 0.5, span_years(trades)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args()
    results = [study("FRED daily oil and gas, live rule", daily_trades()),
               study("IMF monthly metals, 2-month proxy", monthly_trades())]
    text = json.dumps(results, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
