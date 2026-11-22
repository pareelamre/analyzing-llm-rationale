"""Relative value for the commodity pairs: does betting on a price ratio's return to its 10-year normal pay?

    python scripts/pair_rv_backtest.py [--pink CMO-Historical-Data-Monthly.xlsx] [--json out.json]

Owner, 2026-11-22: apply relative-value metrics to the pairs we monitor. This replays the rule in
pair_history (RV_*) on the World Bank's monthly prices (the Pink Sheet, CC BY 4.0), which carry
every leg of the nine price pairs. History starts in 1975, after gold, silver and oil stopped
being fixed or managed prices; with a 10-year window the first signal can come in 1985.

The rule, at each month end, using only the months before it:
  - z: the log price ratio against its trailing 120 months (mean and standard deviation);
  - enter when |z| >= K, betting the ratio moves back toward that normal (sell the ratio when it
    is high: sell the dear leg, buy the cheap one), optionally only once a month's move has
    already turned toward the normal (``turn``);
  - exit at the first month end with |z| < EXIT_Z, or after H months.
A trade's return is per dollar on both legs (equal money): side times the log ratio's move,
halved, less 0.1% a side on each leg. Prices are monthly averages, so entries and exits are
rougher than the daily tracker's.

Settings (K, turn, H) are picked on 1985-2004 alone, by the t-statistic of trade returns, then
scored on 2005-2026, which the pick never saw. The same settings are also run on the tracker's
own sources (IMF monthly metals and FRED oil and gas, 1992 on) as a check.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import commodity_pairs as cp  # noqa: E402
from analyzing_llm_rationale import pair_history as ph  # noqa: E402

START = "1975-01"
SPLIT = "2005-01"
KS = (1.5, 2.0, 2.5)
TURNS = (False, True)
HOLDS = (6, 12, 24)
EXIT_Z = 0.5
COST = 2 * 2 * cp.COST_PER_SIDE / 2  # entry and exit on each of two legs, per dollar on both legs
MIN_EARLY = 15
SEED = 7


def ratio_series(pair: dict, monthly: dict) -> tuple[list[str], np.ndarray]:
    months, r = ph.monthly_ratio(pair, monthly)
    keep = [i for i, m in enumerate(months) if m >= START]
    return [months[i] for i in keep], r[keep]


def z_at(r: np.ndarray, t: int) -> float | None:
    past = r[t - ph.WINDOW_MONTHS:t]
    if len(past) < ph.WINDOW_MONTHS:
        return None
    sd = float(np.std(past, ddof=1))
    return float((r[t] - past.mean()) / sd) if sd > 0 else None


def trades(pair: dict, months: list[str], r: np.ndarray, k: float, turn: bool, hold: int) -> list[dict]:
    out, t = [], ph.WINDOW_MONTHS
    while t < len(r) - 1:
        z = z_at(r, t)
        if z is None or abs(z) < k:
            t += 1
            continue
        side = -1 if z > 0 else 1
        if turn and side * (r[t] - r[t - 1]) <= 0:
            t += 1
            continue
        end = t
        for h in range(1, hold + 1):
            end = t + h
            ze = z_at(r, end)
            if end >= len(r) - 1 or (ze is not None and abs(ze) < EXIT_Z):
                break
        ret = side * float(r[end] - r[t]) / 2 - COST
        out.append({"pair": pair["key"], "entry": months[t], "exit": months[end], "z": round(z, 2),
                    "side": side, "months": end - t, "ret": ret})
        t = end + 1
    return out


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"trades": 0}
    x = np.array([t["ret"] for t in rows])
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {"trades": len(x), "won": round(float((x > 0).mean()), 3), "avg_return": round(float(x.mean()), 4),
            "median_return": round(float(np.median(x)), 4), "worst": round(float(x.min()), 4),
            "avg_months": round(float(np.mean([t["months"] for t in rows])), 1),
            "t_stat": round(float(x.mean()) / sd * np.sqrt(len(x)), 2) if sd > 0 else None}


def boot(rows: list[dict], n: int = 10000) -> list[float] | None:
    if len(rows) < 5:
        return None
    x = np.array([t["ret"] for t in rows])
    rng = np.random.default_rng(SEED)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def run_all(monthly: dict, k: float, turn: bool, hold: int) -> list[dict]:
    rows = []
    for pair in cp.PAIRS:
        if pair["source"] == "parity" or pair["a"] not in monthly or pair["b"] not in monthly:
            continue
        months, r = ratio_series(pair, monthly)
        rows += trades(pair, months, r, k, turn, hold)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pink", type=Path, help="a downloaded CMO-Historical-Data-Monthly.xlsx")
    parser.add_argument("--json", type=Path, help="write the results here")
    args = parser.parse_args()
    data = args.pink.read_bytes() if args.pink else \
        cp._get(ph.pink_sheet_url(), accept="application/octet-stream", http_get=requests.get).content
    pink = ph.parse_pink_sheet(data)

    grid = {}
    for k, turn, hold in itertools.product(KS, TURNS, HOLDS):
        rows = run_all(pink, k, turn, hold)
        early = [t for t in rows if t["entry"] < SPLIT]
        late = [t for t in rows if t["entry"] >= SPLIT]
        grid[(k, turn, hold)] = {"early": stats(early), "late": stats(late), "rows": rows}
    eligible = {key: g for key, g in grid.items() if g["early"]["trades"] >= MIN_EARLY}
    chosen = max(eligible, key=lambda key: (eligible[key]["early"].get("t_stat") or -99, -key[0]))
    k, turn, hold = chosen
    rows = grid[chosen]["rows"]
    late = [t for t in rows if t["entry"] >= SPLIT]
    by_pair = {}
    for pair in cp.PAIRS:
        mine = [t for t in rows if t["pair"] == pair["key"]]
        if mine:
            by_pair[pair["key"]] = {"all": stats(mine), "late": stats([t for t in mine if t["entry"] >= SPLIT])}

    # The tracker's own sources, 1992 on (oil and gas from 1997), same settings.
    live = ph.fetch_monthly(date.today())
    live = {c: v for c, v in live.items() if c not in ph.PINK_SHEET_LIVE}
    live_rows = run_all(live, k, turn, hold)

    result = {
        "rule": {"window_months": ph.WINDOW_MONTHS, "entry_z": k, "turn": turn, "max_months": hold,
                 "exit_z": EXIT_Z, "cost_per_side": cp.COST_PER_SIDE},
        "sample": f"World Bank Pink Sheet monthly prices, {START} on; picked on {START[:4]}-{int(SPLIT[:4]) - 1} "
                  f"(signals from 1985), scored on {SPLIT[:4]}-2026",
        "picked_on_early": grid[chosen]["early"],
        "unseen_late": stats(late) | {"avg_return_95": boot(late)},
        "all_years": stats(rows),
        "by_pair": by_pair,
        "tracker_sources_1992_on": stats(live_rows) | {"by_pair": {
            p: stats([t for t in live_rows if t["pair"] == p]) for p in sorted({t["pair"] for t in live_rows})}},
        "grid": [{"entry_z": key[0], "turn": key[1], "max_months": key[2], "early": g["early"], "late": g["late"]}
                 for key, g in grid.items()],
        "trades": rows,
    }
    print(f"chosen on early years: K={k} turn={turn} hold={hold} months")
    print("early ", grid[chosen]["early"])
    print("late  ", result["unseen_late"])
    print("all   ", result["all_years"])
    print("tracker sources 1992 on", {k2: v for k2, v in result["tracker_sources_1992_on"].items() if k2 != "by_pair"})
    for key, g in sorted(grid.items()):
        print(f"  K={key[0]} turn={str(key[1]):5s} H={key[2]:2d}  early n={g['early']['trades']:3d} "
              f"t={g['early'].get('t_stat')}  | late n={g['late']['trades']:3d} won={g['late'].get('won')} "
              f"avg={g['late'].get('avg_return')} t={g['late'].get('t_stat')}")
    for p, v in by_pair.items():
        print(f"  {p:16s} all {v['all']}  late {v['late'].get('trades')} won {v['late'].get('won')} avg {v['late'].get('avg_return')}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
