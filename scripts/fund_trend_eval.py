#!/usr/bin/env python3
"""The pre-registered trend-following test (docs/fund.md, "Trend-following test").

Time-series momentum on five markets the fund can trade: Nifty 50 and Nifty Bank (Upstox daily
closes), the Nasdaq-100 (FRED), and WTI crude and Henry Hub natural gas futures (EIA's daily NYMEX
contract-1 and contract-2 prices, rolled five sessions before expiry; EIA stopped publishing them
after 5 April 2024, so those two markets leave the portfolio then). Each market's position is
the average sign of its 63-, 126- and 252-session excess returns, scaled to 10% annual volatility
(capped at 3x), set weekly; costs are charged on trades and rolls. The portfolio is the equal-weight
average of the markets with a position.

Test A asks whether the portfolio's mean annual return after costs is above zero (bootstrap over
calendar years); Test B asks whether it made money in the 21-session stretches when the Nifty 50
fell 5% or more, when the fund's put-spread sleeve loses. Writes reports/fund_trend_eval.json.
Raw downloads are kept under --cache.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Optional, Sequence

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import index_forecast as ifc  # noqa: E402

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
EIA = {"RCLC1": "https://www.eia.gov/dnav/pet/hist/RCLC1D.htm", "RCLC2": "https://www.eia.gov/dnav/pet/hist/RCLC2D.htm",
       "RNGC1": "https://www.eia.gov/dnav/ng/hist/RNGC1D.htm", "RNGC2": "https://www.eia.gov/dnav/ng/hist/RNGC2D.htm"}
USER_AGENT = "Foresea/fund-trend (+https://foresea.ink)"
THROUGH = date(2026, 11, 13)

INDIA_CARRY = 0.05        # Indian cash rate minus dividends, a year (fixed; FRED has no current short rate)
NASDAQ_DIVIDEND = 0.008   # taken off the T-bill rate for the Nasdaq-100's carry
LOOKBACKS = (63, 126, 252)
VOL_WINDOW = 60
VOL_TARGET = 0.10
MAX_LEVERAGE = 3.0
ROLL_EARLY = 5
COST = {"index": 0.0005, "commodity": 0.0010}
BLOCK = 21
FALL = -0.05
MIN_BLOCKS = 15
BOOTSTRAP, SEED = 2000, 20261115
YEAR = 252

MARKETS = ("Nifty 50", "Nifty Bank", "Nasdaq-100", "WTI crude", "Natural gas")
LONG_HISTORY = ("Nasdaq-100", "WTI crude", "Natural gas")

Series = list[tuple[date, float]]


# Data -------------------------------------------------------------------------------------------

_WEEK = re.compile(r"<td class='B6'>(?:&nbsp;)*\s*(\d{4}) ([A-Za-z]{3})-\s*(\d+) to [^<]*</td>((?:\s*<td class='B3'>[^<]*</td>){5})")
_CELL = re.compile(r"<td class='B3'>([^<]*)</td>")
_MONTHS = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}


def eia_daily(html: str) -> Series:
    """EIA history tables: one row per week (Monday's date), Monday-Friday prices, blanks for no trade."""
    out = []
    for year, month, day, cells in _WEEK.findall(html):
        monday = date(int(year), _MONTHS[month], int(day))
        for k, value in enumerate(_CELL.findall(cells)):
            value = value.replace(",", "").strip()
            if value:
                out.append((monday + timedelta(days=k), float(value)))
    return sorted(out)


def fred(series: str) -> Series:
    resp = requests.get(FRED, params={"id": series}, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    out = []
    for row in csv.reader(io.StringIO(resp.text)):
        try:
            out.append((date.fromisoformat(row[0]), float(row[1])))
        except (IndexError, ValueError):
            continue
    return out


def _eia(code: str) -> Series:
    resp = requests.get(EIA[code], timeout=120, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return eia_daily(resp.text)


def cached(path: Path, fetch: Callable[[], Series]) -> Series:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([(d.isoformat(), v) for d, v in fetch()]))
    return [(date.fromisoformat(d), float(v)) for d, v in json.loads(path.read_text())]


# Returns ----------------------------------------------------------------------------------------

def wti_expiries(sessions: Sequence[date]) -> list[date]:
    """Last trading day of each WTI contract: three sessions before the 25th of the month before delivery,
    or three sessions before the last session before the 25th when the 25th is not a session."""
    out, seen = [], set(sessions)
    months = sorted({(d.year, d.month) for d in sessions})
    for y, m in months:
        d25 = date(y, m, 25)
        if sessions[-1] < d25:  # the data ends before this month's 25th
            continue
        before = [d for d in sessions if d < d25]
        anchor_idx = len(before) if d25 in seen else len(before) - 1
        if anchor_idx - 3 >= 0:
            out.append(sessions[anchor_idx - 3])
    return out


def ng_expiries(sessions: Sequence[date]) -> list[date]:
    """Last trading day of each natural gas contract: the third session before the first of the delivery month."""
    out = []
    months = sorted({(d.year, d.month) for d in sessions})
    for y, m in months:
        first = date(y, m, 1)
        before = [d for d in sessions if d < first]
        if len(before) >= 3 and sessions[-1] >= first and before[-1] >= first - timedelta(days=10):
            out.append(before[-3])
    return out


def futures_returns(c1: Series, c2: Series, expiries: Sequence[date]) -> tuple[list[date], list[float], list[bool]]:
    """Daily return of the contract held: contract 1, switching to contract 2 ROLL_EARLY sessions before
    contract 1 expires. Also flags each session whose close is a roll (contract 1 sold, contract 2 bought)."""
    p1, p2 = dict(c1), dict(c2)
    days = sorted(set(p1) & set(p2))
    index = {d: k for k, d in enumerate(days)}
    expiry_idx = sorted(index[e] for e in expiries if e in index)
    held = ["C1"] * len(days)
    for e in expiry_idx:
        for k in range(max(0, e - ROLL_EARLY), e + 1):
            held[k] = "C2"
    expired = set(expiry_idx)
    dates, rets, rolls = [], [], []
    for k in range(1, len(days)):
        prev, cur = days[k - 1], days[k]
        before = p1[prev] if held[k - 1] == "C1" else p2[prev]
        # The contract held at the previous close: contract 2 becomes contract 1 once contract 1 expires.
        after = p1[cur] if held[k - 1] == "C1" or (k - 1) in expired else p2[cur]
        if before <= 0:
            continue
        dates.append(cur)
        rets.append(after / before - 1.0)
        rolls.append(held[k] == "C2" and held[k - 1] == "C1")
    return dates, rets, rolls


def index_excess(closes: Series, carry: Callable[[date], float]) -> tuple[list[date], list[float], list[bool]]:
    """Daily price change less the year's carry spread over the sessions; a roll on each month's first session."""
    dates, rets, rolls = [], [], []
    for (d0, p0), (d1, p1) in zip(closes, closes[1:]):
        dates.append(d1)
        rets.append(p1 / p0 - 1.0 - carry(d1) / YEAR)
        rolls.append(d1.month != d0.month)
    return dates, rets, rolls


def _week_ends(dates: Sequence[date]) -> set[int]:
    """Indices of each ISO week's last session."""
    return {k for k in range(len(dates)) if k == len(dates) - 1 or dates[k + 1].isocalendar()[:2] != dates[k].isocalendar()[:2]}


def strategy(dates: Sequence[date], rets: Sequence[float], rolls: Sequence[bool], cost: float) -> list[tuple[date, float, float]]:
    """(date, return after costs, position held into that date) for each session after the warm-up."""
    week_end = _week_ends(dates)
    logs = [math.log1p(r) for r in rets]
    out, w = [], None
    for k in range(len(dates)):
        if w is not None:
            pnl = w * rets[k] - (cost * abs(w) if rolls[k] else 0.0)
            out.append((dates[k], pnl, w))
        if k in week_end and k + 1 >= max(LOOKBACKS):
            signal = fmean([(s > 0) - (s < 0) for s in (sum(logs[k + 1 - n:k + 1]) for n in LOOKBACKS)])
            vol = pstdev(rets[k + 1 - VOL_WINDOW:k + 1]) * math.sqrt(YEAR)
            new = max(-MAX_LEVERAGE, min(MAX_LEVERAGE, signal * VOL_TARGET / vol)) if vol > 0 else 0.0
            trade = abs(new - (w or 0.0))
            if out and trade:
                d, pnl, held = out[-1]
                out[-1] = (d, pnl - cost * trade, held)
            elif not out and trade:
                out.append((dates[k], -cost * trade, 0.0))
            w = new
    return out


def portfolio(by_market: dict[str, list[tuple[date, float, float]]], names: Sequence[str]) -> tuple[list[date], list[float], date]:
    """Equal-weight daily return over the markets with a position; and the first day all of them have one.

    A market has a position from its first set position until its data ends (EIA's futures stop in 2024).
    """
    started = {m: by_market[m][0][0] for m in names}
    ended = {m: by_market[m][-1][0] for m in names}
    pnl = {m: {d: p for d, p, _ in by_market[m]} for m in names}
    days = sorted(set().union(*(pnl[m] for m in names)))
    out_d, out_r = [], []
    for d in days:
        active = [m for m in names if started[m] <= d <= ended[m]]
        if active:
            out_d.append(d)
            out_r.append(sum(pnl[m].get(d, 0.0) for m in active) / len(active))
    return out_d, out_r, max(started.values())


# Statistics -------------------------------------------------------------------------------------

def annualized(rets: Sequence[float]) -> Optional[float]:
    return fmean(rets) * YEAR if rets else None


def sharpe(rets: Sequence[float]) -> Optional[float]:
    if len(rets) < 2:
        return None
    sd = pstdev(rets)
    return fmean(rets) / sd * math.sqrt(YEAR) if sd > 0 else None


def max_drawdown(rets: Sequence[float]) -> float:
    peak = level = worst = 0.0
    for r in rets:
        level += r
        peak = max(peak, level)
        worst = min(worst, level - peak)
    return worst


def year_bootstrap(days: Sequence[date], rets: Sequence[float]) -> Optional[list[float]]:
    """95% interval of the annualized mean, resampling calendar years."""
    by_year: dict[int, list[float]] = {}
    for d, r in zip(days, rets):
        by_year.setdefault(d.year, []).append(r)
    years = sorted(by_year)
    if len(years) < 5:
        return None
    rng = random.Random(SEED)
    draws = sorted(annualized([r for y in rng.choices(years, k=len(years)) for r in by_year[y]]) for _ in range(BOOTSTRAP))
    return [round(draws[int(0.025 * BOOTSTRAP)], 5), round(draws[int(0.975 * BOOTSTRAP) - 1], 5)]


def summary(days: Sequence[date], rets: Sequence[float]) -> dict[str, Any]:
    by_year: dict[int, float] = {}
    for d, r in zip(days, rets):
        by_year[d.year] = by_year.get(d.year, 0.0) + r
    ann = annualized(rets)
    sr = sharpe(rets)
    return {"first": days[0].isoformat() if days else None, "last": days[-1].isoformat() if days else None,
            "sessions": len(rets), "annual_return": None if ann is None else round(ann, 5),
            "annual_return_ci": year_bootstrap(days, rets), "sharpe": None if sr is None else round(sr, 3),
            "annual_vol": round(pstdev(rets) * math.sqrt(YEAR), 4) if len(rets) > 1 else None,
            "max_drawdown": round(max_drawdown(rets), 4),
            "worst_year": min(by_year.items(), key=lambda kv: kv[1]) if by_year else None,
            "best_year": max(by_year.items(), key=lambda kv: kv[1]) if by_year else None,
            "by_year": {str(y): round(v, 4) for y, v in sorted(by_year.items())}}


def fall_blocks(index_closes: Series, port_days: Sequence[date], port_rets: Sequence[float],
                start: date, end: date) -> list[dict[str, Any]]:
    """Back-to-back BLOCK-session blocks of the index in [start, end]: the index's move and the portfolio's return."""
    closes = [(d, p) for d, p in index_closes if start <= d <= end]
    by_day = dict(zip(port_days, port_rets))
    out = []
    for k in range(0, len(closes) - BLOCK, BLOCK):
        (d0, p0), (d1, p1) = closes[k], closes[k + BLOCK]
        trend = sum(r for d, r in by_day.items() if d0 < d <= d1)
        out.append({"from": d0.isoformat(), "to": d1.isoformat(), "index_move": round(p1 / p0 - 1.0, 5), "trend": round(trend, 5)})
    return out


def block_test(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    falls = [b for b in blocks if b["index_move"] <= FALL]
    values = [b["trend"] for b in falls]
    ci = None
    if len(values) >= 2:
        rng = random.Random(SEED)
        draws = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP))
        ci = [round(draws[int(0.025 * BOOTSTRAP)], 5), round(draws[int(0.975 * BOOTSTRAP) - 1], 5)]
    return {"blocks": len(blocks), "fall_blocks": len(falls), "mean_in_falls": round(fmean(values), 5) if values else None,
            "mean_in_falls_ci": ci, "share_positive": round(fmean(v > 0 for v in values), 3) if values else None,
            "mean_all_blocks": round(fmean(b["trend"] for b in blocks), 5) if blocks else None,
            "falls": falls}


# Run --------------------------------------------------------------------------------------------

def load(cache: Path) -> dict[str, Any]:
    nifty = cached(cache / "nifty50.json", lambda: ifc.daily_closes("NSE_INDEX|Nifty 50", ifc.HISTORY_START, THROUGH))
    bank = cached(cache / "niftybank.json", lambda: ifc.daily_closes("NSE_INDEX|Nifty Bank", ifc.HISTORY_START, THROUGH))
    ndx = cached(cache / "nasdaq100.json", lambda: fred("NASDAQ100"))
    tbill = cached(cache / "dtb3.json", lambda: fred("DTB3"))
    eia = {code: cached(cache / f"{code}.json", lambda code=code: _eia(code)) for code in EIA}
    return {"nifty": [x for x in nifty if x[0] <= THROUGH], "bank": [x for x in bank if x[0] <= THROUGH],
            "ndx": [x for x in ndx if x[0] <= THROUGH], "tbill": tbill,
            "eia": {k: [x for x in v if x[0] <= THROUGH] for k, v in eia.items()}}


def market_returns(data: dict[str, Any]) -> dict[str, tuple[list[date], list[float], list[bool], float]]:
    bills = sorted(data["tbill"])

    def us_carry(d: date, _cache: dict = {}) -> float:
        # The latest T-bill rate on or before the day.
        if d not in _cache:
            lo, hi, best = 0, len(bills) - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                if bills[mid][0] <= d:
                    best, lo = bills[mid][1], mid + 1
                else:
                    hi = mid - 1
            _cache[d] = (best or 0.0) / 100.0 - NASDAQ_DIVIDEND
        return _cache[d]
    eia = data["eia"]
    wti_days = sorted(set(dict(eia["RCLC1"])) & set(dict(eia["RCLC2"])))
    ng_days = sorted(set(dict(eia["RNGC1"])) & set(dict(eia["RNGC2"])))
    return {
        "Nifty 50": (*index_excess(data["nifty"], lambda d: INDIA_CARRY), COST["index"]),
        "Nifty Bank": (*index_excess(data["bank"], lambda d: INDIA_CARRY), COST["index"]),
        "Nasdaq-100": (*index_excess(data["ndx"], us_carry), COST["index"]),
        "WTI crude": (*futures_returns(eia["RCLC1"], eia["RCLC2"], wti_expiries(wti_days)), COST["commodity"]),
        "Natural gas": (*futures_returns(eia["RNGC1"], eia["RNGC2"], ng_expiries(ng_days)), COST["commodity"]),
    }


def report(data: dict[str, Any]) -> dict[str, Any]:
    rets = market_returns(data)
    by_market = {m: strategy(d, r, rl, c) for m, (d, r, rl, c) in rets.items()}
    days, port, all_in = portfolio(by_market, MARKETS)
    last = max(d for d in days if d <= THROUGH)
    window = [(d, r) for d, r in zip(days, port) if all_in <= d <= last]
    w_days, w_rets = [d for d, _ in window], [r for _, r in window]
    test_a = summary(w_days, w_rets)
    ci = test_a["annual_return_ci"]
    a_go = bool(ci and ci[0] > 0)
    blocks = fall_blocks(data["nifty"], w_days, w_rets, all_in, last)
    test_b = block_test(blocks)
    bci = test_b["mean_in_falls_ci"]
    b_go = bool(test_b["fall_blocks"] >= MIN_BLOCKS and bci and bci[0] > 0)
    verdict = "GO" if a_go else ("GO as a hedge" if b_go and (test_a["annual_return"] or 0.0) >= 0 else "NO-GO")
    l_days, l_port, l_all = portfolio(by_market, LONG_HISTORY)
    long_window = [(d, r) for d, r in zip(l_days, l_port) if l_all <= d <= last]
    ndx_blocks = block_test(fall_blocks(data["ndx"], w_days, w_rets, all_in, last))
    return {
        "verdict": verdict, "rule": "docs/fund.md, Trend-following test (pre-registered 2026-11-15)",
        "test_a": {"go": a_go, **test_a},
        "test_b": {"go": b_go, **{k: v for k, v in test_b.items() if k != "falls"}, "falls": test_b["falls"]},
        "settings": {"lookbacks": LOOKBACKS, "vol_window": VOL_WINDOW, "vol_target": VOL_TARGET, "max_leverage": MAX_LEVERAGE,
                     "roll_early": ROLL_EARLY, "cost": COST, "india_carry": INDIA_CARRY, "nasdaq_dividend": NASDAQ_DIVIDEND,
                     "block": BLOCK, "fall": FALL, "min_blocks": MIN_BLOCKS, "through": THROUGH.isoformat()},
        "sources": {"Nifty 50, Nifty Bank": "Upstox daily closes", "Nasdaq-100": "FRED NASDAQ100, carry from FRED DTB3",
                    "WTI crude": "EIA RCLC1/RCLC2", "Natural gas": "EIA RNGC1/RNGC2"},
        "description_only": {
            "by_market": {m: summary([d for d, _, _ in s], [p for _, p, _ in s]) for m, s in by_market.items()},
            "long_history_portfolio": {"markets": LONG_HISTORY, **summary([d for d, _ in long_window], [r for _, r in long_window])},
            "nasdaq_fall_blocks": {k: v for k, v in ndx_blocks.items() if k != "falls"},
            "average_gross_position": {m: round(fmean(abs(w) for _, _, w in s), 3) for m, s in by_market.items()},
            "data_ends": {m: s[-1][0].isoformat() for m, s in by_market.items()},
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "trend_eval")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_trend_eval.json")
    args = ap.parse_args(argv)
    out = report(load(args.cache))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str) + "\n")
    a, b = out["test_a"], out["test_b"]
    print(f"verdict {out['verdict']}: Test A {a['annual_return']} a year {a['annual_return_ci']} (Sharpe {a['sharpe']}, "
          f"{a['first']} to {a['last']}); Test B {b['fall_blocks']} fall blocks, mean {b['mean_in_falls']} {b['mean_in_falls_ci']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
