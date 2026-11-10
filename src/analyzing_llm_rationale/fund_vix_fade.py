"""The US VIX fade forward shadow test (docs/fund.md, "US VIX fade: forward shadow test").

The commodity correlation scan found one test that held from 2010-2018 into 2019-2026: a
large one-day change in US VIX before an NSE session points the way Bank Nifty moves from that
session's open to its close. All of it came from the sell side (a sharp fall in VIX, India opens
high, Bank Nifty gives some back), and much of it from 2020, so it is recorded forward before
anything trades on it.

For every NSE session from FORWARD_FROM, once FRED has the VIX closes dated before it (FRED posts
a day late), this records the trade the fixed rule would have made and grades it with that
session's Bank Nifty open and close from Upstox's open candles. It never trades.
"""
from __future__ import annotations

import csv
import io
import json
import math
import random
from datetime import date, datetime, time, timedelta, timezone
from statistics import fmean
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote

import requests

ANALYST = "vix_fade"
FRED_VIX = "https://fred.stlouisfed.org/graph/fredgraph.csv"
#: FRED's firewall drops agents containing "eval"; this one is plain.
USER_AGENT = "Foresea/fund-vix (+https://foresea.ink)"
BANK_NIFTY = "NSE_INDEX|Nifty Bank"
#: Upstox's candle routes are open data; index_forecast uses them too, but it needs numpy, which
#: the fund's job does not install.
UPSTOX_BASE_URL = "https://api.upstox.com"
IST = timezone(timedelta(hours=5, minutes=30))
FORWARD_FROM = date(2026, 11, 10)
#: Fixed by the scan's discovery years, 2010-2018 (reports/fund_correlation_scan.json).
DISCOVERY_MEAN, DISCOVERY_SD = 0.006, 1.651
COST = 0.0005
MAX_GAP_DAYS = 5
MIN_SHORTS, BOOTSTRAP, SEED = 50, 2000, 20261110

HttpGet = Callable[..., Any]


def vix_closes(*, http_get: HttpGet = requests.get) -> list[tuple[date, float]]:
    resp = http_get(FRED_VIX, params={"id": "VIXCLS"}, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    out = []
    for row in csv.reader(io.StringIO(resp.text)):
        try:
            out.append((date.fromisoformat(row[0]), float(row[1])))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def sessions(start: date, through: date, *, http_get: HttpGet = requests.get) -> list[tuple[date, float, float]]:
    """(day, open, close) for each finished Bank Nifty session from ``start`` through ``through``."""
    path = f"/v3/historical-candle/{quote(BANK_NIFTY, safe='')}/days/1/{through.isoformat()}/{start.isoformat()}"
    resp = http_get(f"{UPSTOX_BASE_URL}{path}", timeout=30, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    resp.raise_for_status()
    candles = ((resp.json() or {}).get("data") or {}).get("candles") or []
    out = {}
    for c in candles:
        day = date.fromisoformat(str(c[0])[:10])
        if day.weekday() < 5 and float(c[1]) > 0 and float(c[4]) > 0:
            out[day] = (float(c[1]), float(c[4]))
    return [(d, o, c) for d, (o, c) in sorted(out.items())]


def signal(vix: Sequence[tuple[date, float]], session: date) -> Optional[dict]:
    """The rule's call for ``session`` from the latest VIX change dated before it, or None while
    FRED may still be missing that close: only a value dated on or after the session proves the
    days before it are all in (FRED posts a day late)."""
    if not any(d >= session for d, _ in vix):
        return None
    before = [(d, v) for d, v in vix if d < session]
    if len(before) < 2 or (session - before[-1][0]).days > MAX_GAP_DAYS:
        return None
    change = before[-1][1] - before[-2][1]
    if change - DISCOVERY_MEAN <= -DISCOVERY_SD:
        side = "SHORT"
    elif change - DISCOVERY_MEAN >= DISCOVERY_SD:
        side = "LONG"
    else:
        side = "NONE"
    return {"side": side, "vix_date": before[-1][0].isoformat(), "vix_change": round(change, 3)}


def finished_through(now: datetime) -> date:
    """The last session day whose close is final (15:45 IST)."""
    local = now.astimezone(IST)
    return local.date() if local.time() >= time(15, 45) else local.date() - timedelta(days=1)


def record(conn, fund_id: str, now: datetime, vix: Sequence[tuple[date, float]],
           bank: Sequence[tuple[date, float, float]]) -> int:
    """Write each finished session once its VIX signal is known: the trade and its result."""
    done = {r[0] for r in conn.execute("SELECT key FROM shadow_fills WHERE fund_id = ? AND analyst = ?",
                                       (fund_id, ANALYST))}
    n = 0
    for day, open_, close in bank:
        if day < FORWARD_FROM or day.isoformat() in done:
            continue
        call = signal(vix, day)
        if call is None:
            continue  # FRED has not posted the close yet; try again next run
        sign = {"SHORT": -1.0, "LONG": 1.0, "NONE": 0.0}[call["side"]]
        net = sign * math.log(close / open_) - (COST if sign else 0.0)
        detail = {**call, "open": open_, "close": close, "net_return": round(net, 6)}
        conn.execute(
            "INSERT INTO shadow_fills (fund_id, analyst, key, instrument_id, label, side, ts, shares, cost_usd, fee_usd,"
            " detail_json, outcome, settled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fund_id, ANALYST, day.isoformat(), "nse_index:BANKNIFTY", f"Bank Nifty open to close, {day}",
             call["side"], now.isoformat(), 1.0 if sign else 0.0, open_, COST if sign else 0.0,
             json.dumps(detail, sort_keys=True), (1 if net > 0 else 0) if sign else None, now.isoformat()))
        n += 1
    return n


def _interval(values: list[float]) -> Optional[list[float]]:
    if len(values) < 5:
        return None
    rng = random.Random(SEED)
    draws = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP))
    return [round(draws[int(0.025 * BOOTSTRAP)], 6), round(draws[int(0.975 * BOOTSTRAP) - 1], 6)]


def summary(conn, fund_id: str) -> dict:
    rows = [(key, side, json.loads(detail)) for key, side, detail in conn.execute(
        "SELECT key, side, detail_json FROM shadow_fills WHERE fund_id = ? AND analyst = ? ORDER BY key",
        (fund_id, ANALYST))]
    by_side = {s: [d["net_return"] for _key, side, d in rows if side == s] for s in ("SHORT", "LONG")}
    shorts = by_side["SHORT"]
    ci = _interval(shorts)
    due = len(shorts) >= MIN_SHORTS
    return {
        "mode": "shadow", "kind": "vixfade", "since": FORWARD_FROM.isoformat(), "sessions": len(rows),
        "shorts": len(shorts), "longs": len(by_side["LONG"]),
        "short_mean_return": round(fmean(shorts), 6) if shorts else None, "short_mean_return_ci": ci,
        "long_mean_return": round(fmean(by_side["LONG"]), 6) if by_side["LONG"] else None,
        "verdict": ("GO" if ci and ci[0] > 0 else "NO-GO") if due else "pending",
        "recent": [{"session": key, "side": side, "vix_change": d["vix_change"], "net_return": d["net_return"]}
                   for key, side, d in rows if side != "NONE"][-5:],
    }


def run(conn, fund_id: str, now: datetime, *, http_get: HttpGet = requests.get) -> dict:
    through = finished_through(now)
    added = 0
    if through >= FORWARD_FROM:
        added = record(conn, fund_id, now, vix_closes(http_get=http_get),
                       sessions(FORWARD_FROM, through, http_get=http_get))
    return {**summary(conn, fund_id), "added_this_run": added}
