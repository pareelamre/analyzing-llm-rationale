#!/usr/bin/env python3
"""The pre-registered Bank Nifty put-spread test (docs/fund.md, "Bank Nifty put spreads").

The Nifty backtest's put side on Bank Nifty: every 21 sessions the index model is refit on Bank
Nifty with India VIX, fund_options picks the put credit spread from its distribution, and the
spread is priced with Black-76 at India VIX times the ratio of Bank Nifty's to Nifty's realized
volatility over the last 60 sessions, with half-spreads and the 2% charges, held to the close 21
sessions later. Nifty's own put spread on the same days, their correlation and the fund's path
with both are reported for description. Writes reports/fund_banknifty_put_eval.json.
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import index_forecast as ifc  # noqa: E402


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nifty = _module("fund_nifty_premium_eval")
ndx = _module("fund_ndx_put_eval")

BANK_KEY = "NSE_INDEX|Nifty Bank"
THROUGH = date(2026, 11, 13)
RATIO_SESSIONS = 60
BOTH_SHARES = (0.02, 0.05)


def vol_ratio(bank: list[tuple[date, float]], nif: list[tuple[date, float]], day: date) -> Optional[float]:
    """Bank Nifty's realized volatility over Nifty's, on the last 60 sessions both have to ``day``."""
    nif_by_day = dict(nif)
    common = [(d, c, nif_by_day[d]) for d, c in bank if d <= day and d in nif_by_day][-(RATIO_SESSIONS + 1):]
    if len(common) < RATIO_SESSIONS + 1:
        return None
    rb = [math.log(b / a) for (_, a, _), (_, b, _) in zip(common, common[1:])]
    rn = [math.log(b / a) for (_, _, a), (_, _, b) in zip(common, common[1:])]
    sn = pstdev(rn)
    return pstdev(rb) / sn if sn > 0 else None


def _close_on(series: list[tuple[date, float]], day: date) -> Optional[float]:
    """The close on ``day``, or the last one before it."""
    days = [d for d, _ in series]
    j = bisect.bisect_right(days, day) - 1
    return series[j][1] if j >= 0 else None


def _put(entries: dict) -> Optional[dict]:
    return entries.get("put_spread")


def run(bank: list[tuple[date, float]], nif: list[tuple[date, float]], vix: list[tuple[date, float]]) -> dict[str, list[dict]]:
    vix_by_day = dict(vix)
    H = nifty.H
    trades: dict[str, list[dict]] = {"put_spread": [], "unscaled": [], "nifty": []}
    i = next((j for j in range(len(bank)) if j >= ifc.WARMUP_SESSIONS + 260 and bank[j][0] in vix_by_day), None)
    while i is not None and i + H < len(bank):
        day, spot = bank[i]
        v = vix_by_day.get(day)
        ratio = vol_ratio(bank, nif, day) if v else None
        dist = nifty.model_distribution(bank[: i + 1], [x for x in vix if x[0] <= day]) if ratio else None
        if dist is not None:
            exit_day, level = bank[i + H]
            row = {"day": day.isoformat(), "spot": spot, "vix": v, "ratio": round(ratio, 4), "expiry_close": level}
            for key, vol in (("put_spread", v * ratio), ("unscaled", v)):
                entry = _put(ndx.decide(dist, spot, vol))
                if entry:
                    trades[key].append({**row, "vol": round(vol, 3), "kind": "put_spread", "cost": round(entry["cost"], 4),
                                        "edge": round(entry["edge"], 4), "return": nifty.net_return(entry, level)})
            n_spot, n_level = _close_on(nif, day), _close_on(nif, exit_day)
            n_dist = nifty.model_distribution([x for x in nif if x[0] <= day], [x for x in vix if x[0] <= day])
            entry = nifty.decide(n_dist, n_spot, v)["legs"].get("put_spread") if n_dist and n_spot and n_level else None
            if entry:
                trades["nifty"].append({"day": day.isoformat(), "spot": n_spot, "vix": v, "expiry_close": n_level,
                                        "kind": "put_spread", "cost": round(entry["cost"], 2),
                                        "edge": round(entry["edge"], 4), "return": nifty.net_return(entry, n_level)})
        i += H
    return trades


def correlation(a: list[float], b: list[float]) -> Optional[float]:
    if len(a) < 3:
        return None
    ma, mb = fmean(a), fmean(b)
    sa, sb = pstdev(a), pstdev(b)
    if sa == 0 or sb == 0:
        return None
    return fmean((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def both_sold(bank: list[dict], nif: list[dict]) -> dict:
    """ρ between the two spreads on days both traded, and the fund's path selling both at a share each."""
    by_day = {t["day"]: t["return"] for t in nif}
    pairs = [(t["return"], by_day[t["day"]]) for t in bank if t["day"] in by_day]
    rb, rn = [p[0] for p in pairs], [p[1] for p in pairs]
    rho = correlation(rb, rn)
    out = {"decisions": len(pairs), "rho": round(rho, 4) if rho is not None else None,
           "both_lost": sum(x < 0 and y < 0 for x, y in pairs),
           "bank_lost": sum(x < 0 for x in rb), "nifty_lost": sum(y < 0 for y in rn)}
    for share in BOTH_SHARES:
        # The pair is one combined bet each decision: a share of the fund at risk on each spread.
        combined = [(x + y) / 2 for x, y in pairs]
        out[f"fund_at_{int(share * 100)}pct_each"] = nifty.fund_path(combined, 2 * share)
        out[f"worst_decision_at_{int(share * 100)}pct_each"] = round(min((share * (x + y) for x, y in pairs), default=0.0), 4)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "banknifty_put_eval")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_banknifty_put_eval.json")
    ap.add_argument("--through", type=date.fromisoformat, default=THROUGH)
    args = ap.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    series = {}
    for name, key in (("banknifty", BANK_KEY), ("nifty", ifc.INDICES[0][1]), ("vix", ifc.VIX_KEY)):
        path = args.cache / f"{name}.json"
        if not path.is_file():
            path.write_text(json.dumps([(d.isoformat(), c) for d, c in ifc.daily_closes(key, ifc.HISTORY_START, args.through)]))
        series[name] = [(date.fromisoformat(d), float(c)) for d, c in json.loads(path.read_text())
                        if date.fromisoformat(d) <= args.through]
    trades = run(series["banknifty"], series["nifty"], series["vix"])
    put = nifty.summary(trades["put_spread"])
    ratios = sorted(t["ratio"] for t in trades["put_spread"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "docs/fund.md, Bank Nifty put spreads (pre-registered 2026-11-16)",
        "data": {"banknifty": [series["banknifty"][0][0].isoformat(), series["banknifty"][-1][0].isoformat()],
                 "pricing": "Black-76 at India VIX x Bank Nifty/Nifty 60-session realized vol, flat; "
                            "half-spread max(0.002% of index, 1%) per leg; 2% charges",
                 "vol_ratio": {"median": ratios[len(ratios) // 2], "min": ratios[0], "max": ratios[-1]} if ratios else None},
        "put_spread": {"verdict": nifty.verdict(put), **put},
        "description_only": {"unscaled_india_vix": nifty.summary(trades["unscaled"]),
                             "nifty_same_days": nifty.summary(trades["nifty"]),
                             "both_sold": both_sold(trades["put_spread"], trades["nifty"])},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    r = report["put_spread"]
    print("put_spread", r["verdict"], {k: r.get(k) for k in ("trades", "first", "last", "mean_return", "mean_return_ci",
                                                              "stdev", "win_share", "worst", "years_positive",
                                                              "fund_at_2pct_risk", "fund_at_10pct_risk")})
    print("vol_ratio", report["data"]["vol_ratio"])
    for k, v in report["description_only"].items():
        print(k, v if k == "both_sold" else {x: v.get(x) for x in ("trades", "mean_return", "mean_return_ci", "win_share",
                                                                     "years_positive", "fund_at_10pct_risk")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
