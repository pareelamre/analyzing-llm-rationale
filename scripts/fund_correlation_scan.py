#!/usr/bin/env python3
"""The pre-registered commodity correlation scan (docs/fund.md, "Commodity correlation scan").

Twelve predictors known before an NSE session opens (commodity returns, the energy pairs'
spread z-scores, US markets, the dollar, the rupee, India VIX) against what the fund could
trade on that session: Nifty 50 and Nifty Bank from open to close. The two energy spreads are
also tested against their own next 5 sessions. A test found in 2010-2018 (|t| >= 3) must hold
in 2019-2026 (same sign, t >= 2), and then trade at a profit after costs, to count.

Data: FRED (fredgraph CSV, no key) and Upstox's open daily candles. Writes
reports/fund_correlation_scan.json; downloads are kept under --cache.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import index_forecast as ifc  # noqa: E402

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
#: FRED's firewall drops agents containing "eval" (data sources note); this one is plain.
USER_AGENT = "Foresea/fund-correlations (+https://foresea.ink)"
FRED_SERIES = {"brent": "DCOILBRENTEU", "wti": "DCOILWTICO", "gas": "DHHNGSP", "dollar": "DTWEXBGS",
               "us10y": "DGS10", "vix": "VIXCLS", "nasdaq": "NASDAQ100", "rupee": "DEXINUS"}
UPSTOX = {"nifty": "NSE_INDEX|Nifty 50", "bank": "NSE_INDEX|Nifty Bank", "indiavix": ifc.VIX_KEY}
START, SPLIT = date(2010, 1, 1), date(2019, 1, 1)
HEDGE_WINDOW, Z_WINDOW = 250, 60
DISCOVERY_T, HOLDOUT_T = 3.0, 2.0
COSTS = {"index": 0.0005, "spread": 0.002}
SPREAD_HORIZON = 5
BOOTSTRAP, SEED = 2000, 20261110


def _cached(path: Path, fetch: Callable[[], Any]) -> Any:
    if path.is_file():
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def fred(series: str) -> list[tuple[str, float]]:
    resp = requests.get(FRED, params={"id": series}, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    out = []
    for row in csv.reader(io.StringIO(resp.text)):
        try:
            out.append((row[0], float(row[1])))
        except (IndexError, ValueError):
            continue
    return out


def upstox_ohlc(key: str, through: date) -> list[tuple[str, float, float]]:
    """(day, open, close) for every weekday session since START."""
    rows, to = {}, through
    while to >= START:
        frm = max(START, to - timedelta(days=ifc.CHUNK_DAYS))
        data = ifc.upstox_get(f"/v3/historical-candle/{quote(key, safe='')}/days/1/{to.isoformat()}/{frm.isoformat()}")
        for c in (data or {}).get("candles") or []:
            day = date.fromisoformat(str(c[0])[:10])
            if day.weekday() < 5 and float(c[1]) > 0 and float(c[4]) > 0:
                rows[day.isoformat()] = (float(c[1]), float(c[4]))
        to = frm - timedelta(days=1)
    return [(d, o, c) for d, (o, c) in sorted(rows.items())]


def log_change(series: Sequence[tuple[date, float]], lag: int = 1) -> dict[date, float]:
    return {series[i][0]: math.log(series[i][1] / series[i - lag][1]) for i in range(lag, len(series))
            if series[i][1] > 0 and series[i - lag][1] > 0}


def diff(series: Sequence[tuple[date, float]]) -> dict[date, float]:
    return {series[i][0]: series[i][1] - series[i - 1][1] for i in range(1, len(series))}


def spread_z(a: Sequence[tuple[date, float]], b: Sequence[tuple[date, float]]) -> tuple[dict[date, float], dict[date, float]]:
    """The tracker's rule: log a minus beta x log b, beta by least squares over HEDGE_WINDOW,
    z against the last Z_WINDOW spreads; and each day's spread (for the 5-session change)."""
    bd = dict(b)
    days = [d for d, _ in a if d in bd]
    la = {d: math.log(v) for d, v in a if d in bd and v > 0}
    lb = {d: math.log(bd[d]) for d in days if bd[d] > 0}
    days = [d for d in days if d in la and d in lb]
    z, spread = {}, {}
    for i in range(HEDGE_WINDOW, len(days)):
        win = days[i - HEDGE_WINDOW + 1: i + 1]
        xs, ys = [lb[d] for d in win], [la[d] for d in win]
        mx, my = fmean(xs), fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 1.0
        recent = [la[d] - beta * lb[d] for d in days[i - Z_WINDOW + 1: i + 1]]
        sd = pstdev(recent)
        if sd > 0:
            z[days[i]] = (recent[-1] - fmean(recent)) / sd
            spread[days[i]] = recent[-1]
    return z, spread


def known_before(values: dict[date, float], sessions: Sequence[date]) -> dict[date, float]:
    """For each NSE session, the latest value dated strictly before it."""
    days = sorted(values)
    out, j = {}, -1
    for s in sessions:
        while j + 1 < len(days) and days[j + 1] < s:
            j += 1
        if j >= 0 and (s - days[j]).days <= 5:
            out[s] = values[days[j]]
    return out


def corr_t(xs: Sequence[float], ys: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    n = len(xs)
    if n < 30:
        return None, None
    mx, my = fmean(xs), fmean(ys)
    sx, sy = pstdev(xs), pstdev(ys)
    if sx == 0 or sy == 0:
        return None, None
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n * sx * sy)
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, t


def interval(values: list[float]) -> Optional[list[float]]:
    if len(values) < 20:
        return None
    rng = random.Random(SEED)
    draws = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP))
    return [round(draws[int(0.025 * BOOTSTRAP)], 6), round(draws[int(0.975 * BOOTSTRAP) - 1], 6)]


def scan(tests: Sequence[tuple[str, dict[date, float], str, str, dict[date, float]]]) -> list[dict]:
    """Every pre-registered test: (predictor name, its values, target name, kind, target values),
    each pair of series keyed by the day the trade would be made."""
    out = []
    for pname, x, tname, kind, target in tests:
        days = sorted(d for d in target if d in x)
        disc = [d for d in days if d < SPLIT]
        hold = [d for d in days if d >= SPLIT]
        r1, t1 = corr_t([x[d] for d in disc], [target[d] for d in disc])
        r2, t2 = corr_t([x[d] for d in hold], [target[d] for d in hold])
        row = {"predictor": pname, "target": tname, "discovery": {"n": len(disc), "r": _r(r1), "t": _r(t1)},
               "holdout": {"n": len(hold), "r": _r(r2), "t": _r(t2)}}
        candidate = t1 is not None and abs(t1) >= DISCOVERY_T
        confirmed = candidate and t2 is not None and math.copysign(1, t2) == math.copysign(1, t1) and abs(t2) >= HOLDOUT_T
        row["candidate"], row["confirmed"] = candidate, confirmed
        if confirmed:
            mu, sd = fmean([x[d] for d in disc]), pstdev([x[d] for d in disc])
            direction = math.copysign(1, t1)
            rets = [direction * math.copysign(1, x[d] - mu) * target[d] - COSTS[kind]
                    for d in hold if abs(x[d] - mu) >= sd]
            ci = interval(rets)
            row["rule"] = {"trades": len(rets), "mean_net_return": _r(fmean(rets) if rets else None, 6),
                           "mean_net_return_ci": ci, "win_share": _r(fmean(r > 0 for r in rets) if rets else None),
                           "verdict": "GO" if ci and ci[0] > 0 else "NO-GO"}
        out.append(row)
    return out


def _r(v: Optional[float], places: int = 4) -> Optional[float]:
    return None if v is None else round(v, places)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "correlation_scan")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_correlation_scan.json")
    ap.add_argument("--through", type=date.fromisoformat, default=None)
    args = ap.parse_args(argv)
    through = args.through or datetime.now(timezone.utc).date()
    raw = {k: [(date.fromisoformat(d), v) for d, v in _cached(args.cache / f"{k}.json", lambda s=s: fred(s))
               if d >= "2009-01-01"] for k, s in FRED_SERIES.items()}
    nse = {k: [(date.fromisoformat(d), o, c) for d, o, c in _cached(args.cache / f"{k}.json", lambda key=key: upstox_ohlc(key, through))]
           for k, key in UPSTOX.items()}
    sessions = [d for d, _, _ in nse["nifty"] if START <= d <= through]
    brent_wti_z, brent_wti = spread_z(raw["brent"], raw["wti"])
    wti_gas_z, wti_gas = spread_z(raw["wti"], raw["gas"])
    vix_close = [(d, c) for d, _, c in nse["indiavix"]]
    us = {
        "brent_1d": log_change(raw["brent"]), "wti_1d": log_change(raw["wti"]), "gas_1d": log_change(raw["gas"]),
        "brent_wti_z": brent_wti_z, "wti_gas_z": wti_gas_z, "brent_5d": log_change(raw["brent"], 5),
        "nasdaq_1d": log_change(raw["nasdaq"]), "vix_change": diff(raw["vix"]), "dollar_1d": log_change(raw["dollar"]),
        "us10y_change": diff(raw["us10y"]), "rupee_1d": log_change(raw["rupee"]),
    }
    # Known before each NSE session opens: the latest US value dated before it.
    predictors = {k: known_before(v, sessions) for k, v in us.items()}
    predictors["indiavix_change"] = known_before(diff(vix_close), sessions)
    tests = []
    for name, key in (("nifty_open_to_close", "nifty"), ("bank_open_to_close", "bank")):
        target = {d: math.log(c / o) for d, o, c in nse[key] if START <= d <= through}
        tests += [(p, x, name, "index", target) for p, x in predictors.items()]
    # Each spread against its own next 5 sessions, from the close its z-score is read at.
    for zname, sp in (("brent_wti_z", brent_wti), ("wti_gas_z", wti_gas)):
        days = sorted(sp)
        ahead = {days[i]: sp[days[i + SPREAD_HORIZON]] - sp[days[i]] for i in range(len(days) - SPREAD_HORIZON)}
        tests.append((zname, us[zname], zname.replace("_z", "_next5"), "spread", ahead))
    rows = scan(tests)
    confirmed = [r for r in rows if r["confirmed"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "docs/fund.md, Commodity correlation scan (pre-registered 2026-11-10)",
        "discovery": [START.isoformat(), (SPLIT - timedelta(days=1)).isoformat()],
        "holdout": [SPLIT.isoformat(), through.isoformat()],
        "tests": len(rows), "candidates": sum(r["candidate"] for r in rows), "confirmed": len(confirmed),
        "go": [f"{r['predictor']} -> {r['target']}" for r in confirmed if r["rule"]["verdict"] == "GO"],
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("tests", "candidates", "confirmed", "go")}))
    for r in rows:
        flag = "CONFIRMED" if r["confirmed"] else ("candidate" if r["candidate"] else "")
        print(f"{r['predictor']:16s} -> {r['target']:20s} disc t {r['discovery']['t']!s:>7} hold t {r['holdout']['t']!s:>7} {flag}"
              + (f" rule {r['rule']}" if r.get("rule") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
