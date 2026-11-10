#!/usr/bin/env python3
"""The pre-registered Nasdaq-100 put-spread check (docs/fund.md, "Out-of-sample check: put
credit spreads on the Nasdaq-100").

The Nifty premium backtest's put side, run unchanged except for scale on an index and a
volatility index it has never seen: FRED's Nasdaq-100 and VXN. Every 21 sessions the index
model is refit on data up to that day, fund_options picks the put credit spread from its
distribution, and the spread is priced with Black-76 at that day's VXN, with half-spreads and
the 2% charges, and held to the close 21 sessions later. Writes reports/fund_ndx_put_eval.json.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import fund_options as fo  # noqa: E402

_spec = importlib.util.spec_from_file_location("fund_nifty_premium_eval", ROOT / "scripts" / "fund_nifty_premium_eval.py")
nifty = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nifty)

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
#: FRED's firewall drops agents containing "eval"; this one is plain.
USER_AGENT = "Foresea/fund-options (+https://foresea.ink)"
SERIES = {"ndx": "NASDAQ100", "vxn": "VXNCLS"}
GRID_SHARE = 0.002  # Nifty's 50 points at 25,000
MIN_HALF_SHARE = 0.00002  # Nifty's 0.5 points at 25,000
SPAN = 0.15


def nice_step(raw: float) -> float:
    """The nearest of 1, 2.5 and 5 times a power of ten."""
    power = 10 ** math.floor(math.log10(raw))
    return min((m * power for m in (1, 2.5, 5, 10)), key=lambda s: abs(s - raw))


def chain(spot: float) -> list[dict]:
    step = nice_step(GRID_SHARE * spot)
    lo, hi = math.floor(spot * (1 - SPAN) / step), math.ceil(spot * (1 + SPAN) / step)
    return [{"instrument_key": f"{kind}{k * step:g}", "instrument_type": kind, "strike_price": k * step, "lot_size": 1}
            for k in range(lo, hi + 1) for kind in ("CE", "PE")]


def quotes(contracts: list[dict], forward: float, vol: float) -> dict[str, dict]:
    sd = vol / 100 * math.sqrt(nifty.H / 252)
    out = {}
    for c in contracts:
        mid = nifty.black(forward, c["strike_price"], sd, c["instrument_type"])
        half = max(MIN_HALF_SHARE * forward, nifty.HALF_SPREAD_SHARE * mid)
        out[c["instrument_key"]] = {"bid": max(mid - half, 0.0), "ask": mid + half, "last": mid, "source": "book"}
    return out


def decide(dist: fo.Distribution, spot: float, vol: float) -> dict[str, dict]:
    """The put spread (the test), and the condor and call spread (description), at one decision."""
    wanted = fo._wanted(chain(spot), spot, dist)
    raw = quotes(wanted, spot, vol)
    out = {}
    for s in fo.candidate_structures(wanted, spot, dist):
        q = nifty.priced(s, raw) if s.kind != "straddle" else None
        if q:
            out[s.kind] = {"structure": s, "cost": q["ask"],
                           "edge": (dist.expect(s.payoff) - q["ask"]) / q["ask"] - nifty.orch.OPTION_COST}
    return out


def run(closes: list[tuple[date, float]], vol: list[tuple[date, float]]) -> dict[str, list[dict]]:
    vol_by_day = dict(vol)
    trades: dict[str, list[dict]] = {"put_spread": [], "condor": [], "call_spread": []}
    # Every close here has a VXN value (main keeps only those days), so the model's warm-up is a count.
    i = nifty.ifc.WARMUP_SESSIONS + 260
    while i + nifty.H < len(closes):
        day, spot = closes[i]
        v = vol_by_day.get(day)
        dist = nifty.model_distribution(closes[: i + 1], [x for x in vol if x[0] <= day]) if v else None
        if dist is not None:
            level = closes[i + nifty.H][1]
            for kind, entry in decide(dist, spot, v).items():
                trades[kind].append({"day": day.isoformat(), "spot": spot, "vol": v, "expiry_close": level,
                                     "kind": kind, "cost": round(entry["cost"], 4), "edge": round(entry["edge"], 4),
                                     "return": nifty.net_return(entry, level)})
        i += nifty.H
    return trades


def fred(series: str) -> list[tuple[date, float]]:
    resp = requests.get(FRED, params={"id": series}, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    out = []
    for row in csv.reader(io.StringIO(resp.text)):
        try:
            out.append((date.fromisoformat(row[0]), float(row[1])))
        except (IndexError, ValueError):
            continue
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "nifty_premium_eval")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_ndx_put_eval.json")
    ap.add_argument("--through", type=date.fromisoformat, default=None)
    args = ap.parse_args(argv)
    through = args.through or datetime.now(timezone.utc).date()
    args.cache.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, series in SERIES.items():
        path = args.cache / f"{name}.json"
        if not path.is_file():
            path.write_text(json.dumps([(d.isoformat(), v) for d, v in fred(series)]))
        data[name] = [(date.fromisoformat(d), float(v)) for d, v in json.loads(path.read_text())
                      if date.fromisoformat(d) <= through]
    # The model needs a session's close and volatility together; keep the days that have both.
    vxn_days = {d for d, _ in data["vxn"]}
    closes = [(d, v) for d, v in data["ndx"] if d in vxn_days]
    trades = run(closes, data["vxn"])
    put = nifty.summary(trades["put_spread"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "docs/fund.md, Out-of-sample check: put credit spreads on the Nasdaq-100 (pre-registered 2026-11-10)",
        "data": {"ndx": [closes[0][0].isoformat(), closes[-1][0].isoformat()],
                 "pricing": "Black-76 at VXN, flat; half-spread max(0.002% of index, 1%) per leg; 2% charges"},
        "put_spread": {"verdict": nifty.verdict(put), **put},
        "description_only": {"condor": nifty.summary(trades["condor"]),
                             "call_spread": nifty.summary(trades["call_spread"])},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    r = report["put_spread"]
    print("put_spread", r["verdict"], {k: r.get(k) for k in ("trades", "first", "last", "mean_return", "mean_return_ci",
                                                              "win_share", "worst", "years_positive",
                                                              "fund_at_2pct_risk", "fund_at_10pct_risk")})
    for k, v in report["description_only"].items():
        print(k, {x: v.get(x) for x in ("trades", "mean_return", "mean_return_ci", "win_share")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
