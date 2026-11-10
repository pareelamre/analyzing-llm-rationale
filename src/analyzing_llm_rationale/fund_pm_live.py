"""The live Polymarket decided-ranges test (docs/fund.md, "Live Polymarket decided-ranges test").

A day's high can only rise, and Polymarket's daily highest-temperature markets settle on the
airport's own routine METARs. So once those readings put a range out of reach, its outcome is
known. Price history could not show whether such a range could still be bought (most of its
prices predated the deciding reading), so at every fund run this module records what the order
book actually offered:

1. reads the open temperature events on the METAR rule and the stations' latest METARs
   (aviationweather.gov, routine reports only, T-group tenths where present);
2. for every range the running high has decided, walks the asks of the side that has won, at
   99 cents or less where a dollar still pays more than the price plus the taker fee, up to
   FILL_CAP_USD, and records the fill (once per range) or that nothing was for sale;
3. grades fills at settlement from Gamma's outcome prices.

It records and does not trade; the verdict is due once 100 fills have settled.
"""
from __future__ import annotations

import json
import math
import random
import re
from datetime import date, datetime, time, timedelta, timezone
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import requests

from . import fund_weather as fw

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_BOOK = "https://clob.polymarket.com/book"
METAR_API = "https://aviationweather.gov/api/data/metar"
RULE_MARK = "hourly temperature remark"
ANALYST = "decided_ranges"
HORIZON_DAYS = 3
METAR_HOURS = 30
MAX_PRICE = 0.99
FILL_CAP_USD = 100.0
MIN_FILLS, MIN_DAYS, BOOTSTRAP, SEED = 100, 7, 2000, 20261110

#: Each station's time zone (local midnight starts its day), from Open-Meteo, 2026-11-10.
STATION_TZ = {
    "CYYZ": "America/Toronto", "EDDM": "Europe/Berlin", "EFHK": "Europe/Helsinki",
    "EGLC": "Europe/London", "EHAM": "Europe/Amsterdam", "EPWA": "Europe/Warsaw",
    "FACT": "Africa/Johannesburg", "KATL": "America/New_York", "KAUS": "America/Chicago",
    "KDAL": "America/Chicago", "KDEN": "America/Denver", "KHOU": "America/Chicago",
    "KLAX": "America/Los_Angeles", "KLGA": "America/New_York", "KMIA": "America/New_York",
    "KORD": "America/Chicago", "KSEA": "America/Los_Angeles", "KSFO": "America/Los_Angeles",
    "LEMD": "Europe/Madrid", "LFPB": "Europe/Paris", "LIMC": "Europe/Rome",
    "LLBG": "Asia/Jerusalem", "LTAC": "Europe/Istanbul", "LTFM": "Europe/Istanbul",
    "MMMX": "America/Mexico_City", "MPMG": "America/Panama", "NZWN": "Pacific/Auckland",
    "OEJN": "Asia/Riyadh", "OPKC": "Asia/Karachi", "RCSS": "Asia/Taipei",
    "RJTT": "Asia/Tokyo", "RKPK": "Asia/Seoul", "RKSI": "Asia/Seoul",
    "RPLL": "Asia/Manila", "SAEZ": "America/Argentina/Buenos_Aires", "SBGR": "America/Sao_Paulo",
    "UUWW": "Europe/Moscow", "VILK": "Asia/Kolkata", "WMKK": "Asia/Kuala_Lumpur",
    "WSSS": "Asia/Singapore", "ZBAA": "Asia/Shanghai", "ZGGG": "Asia/Shanghai",
    "ZGSZ": "Asia/Shanghai", "ZHHH": "Asia/Shanghai", "ZSPD": "Asia/Shanghai",
    "ZSQD": "Asia/Shanghai", "ZUCK": "Asia/Shanghai", "ZUUU": "Asia/Shanghai",
}

HttpGet = Callable[..., Any]

_RANGE = (
    (re.compile(r"^(-?\d+)\s*°([CF]) or below$"), lambda a, b: (-math.inf, a + 0.5)),
    (re.compile(r"^(-?\d+)\s*°([CF]) or higher$"), lambda a, b: (a - 0.5, math.inf)),
    (re.compile(r"^(-?\d+)\s*[-–]\s*(-?\d+)\s*°([CF])$"), lambda a, b: (a - 0.5, b + 0.5)),
    (re.compile(r"^(-?\d+)\s*°([CF])$"), lambda a, b: (a - 0.5, a + 0.5)),
)
_T_GROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
_BODY_TEMP = re.compile(r"\s(M?\d{1,2})/(M?\d{1,2})?\s")


def bounds_of(title: str) -> Optional[tuple[str, tuple[float, float]]]:
    """("C" or "F", half-degree bounds) of a range title such as "29°C" or "60-61°F"."""
    title = (title or "").strip()
    for pattern, make in _RANGE:
        m = pattern.match(title)
        if m:
            nums = [int(g) for g in m.groups() if g not in ("C", "F")]
            return m.groups()[-1], make(nums[0], nums[-1])
    return None


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def settle_value(celsius: float, unit: str) -> int:
    """The market's whole-degree value for a reading (Fahrenheit converts first)."""
    return round_half_up(celsius * 9 / 5 + 32 if unit == "F" else celsius)


def decided(bounds: tuple[float, float], high: int) -> Optional[str]:
    """No once the rounded high reaches the range's top whole degree plus one; Yes for an
    "or higher" range once it reaches the range's bottom whole degree plus one."""
    lo, hi = bounds
    if math.isfinite(hi) and high >= (hi - 0.5) + 1:
        return "NO"
    if not math.isfinite(hi) and math.isfinite(lo) and high >= (lo + 0.5) + 1:
        return "YES"
    return None


def metar_celsius(raw: str) -> Optional[float]:
    """A report's temperature as the market reads it: T-group tenths, else the body's degrees."""
    t = _T_GROUP.search(raw or "")
    if t:
        return (-1 if t.group(1) == "1" else 1) * int(t.group(2)) / 10
    body = _BODY_TEMP.search(f" {raw} ")
    if body:
        v = body.group(1)
        return -float(v[1:]) if v.startswith("M") else float(v)
    return None


def readings(stations: Iterable[str], *, http_get: HttpGet = requests.get) -> dict[str, list[tuple[datetime, float]]]:
    """Routine METAR temperatures per station over the last METAR_HOURS."""
    out: dict[str, list[tuple[datetime, float]]] = {}
    stations = sorted(set(stations))
    for i in range(0, len(stations), 40):
        rows = fw._get(METAR_API, {"ids": ",".join(stations[i:i + 40]), "hours": METAR_HOURS, "format": "json"},
                       http_get=http_get) or []
        for r in rows:
            raw = str(r.get("rawOb") or "")
            if r.get("metarType") == "SPECI" or raw.startswith("SPECI"):
                continue
            c = metar_celsius(raw)
            if c is None or r.get("obsTime") is None:
                continue
            out.setdefault(str(r.get("icaoId")), []).append((datetime.fromtimestamp(int(r["obsTime"]), timezone.utc), c))
    return {k: sorted(v) for k, v in out.items()}


def open_events(now: datetime, *, http_get: HttpGet = requests.get) -> list[dict]:
    """Open temperature events on the METAR rule ending within HORIZON_DAYS, parsed."""
    params = {"tag_slug": "weather", "closed": "false", "limit": 100,
              "end_date_min": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "end_date_max": (now + timedelta(days=HORIZON_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    out = []
    for offset in range(0, 2000, 100):
        page = fw._get(GAMMA_EVENTS, {**params, "offset": offset}, http_get=http_get) or []
        out += [e for e in (parse_open_event(ev) for ev in page) if e]
        if len(page) < 100:
            break
    return out


def parse_open_event(ev: Mapping[str, Any]) -> Optional[dict]:
    markets = ev.get("markets") or []
    if not str(ev.get("title", "")).startswith("Highest temperature in") or not markets:
        return None
    desc = str(markets[0].get("description") or "")
    station = re.search(r"\(([A-Z]{4})\)", desc)
    day = re.search(r"on (\w+ \d{1,2})\??$", str(ev.get("title")))
    if RULE_MARK not in desc or not station or not day:
        return None
    end = datetime.fromisoformat(str(ev["endDate"]).replace("Z", "+00:00"))
    try:
        local_day = datetime.strptime(f"{day.group(1)} {end.year}", "%B %d %Y").date()
    except ValueError:
        return None
    ranges, unit = [], None
    for m in markets:
        parsed = bounds_of(m.get("groupItemTitle"))
        try:
            tokens = json.loads(m.get("clobTokenIds") or "[]")
        except ValueError:
            tokens = []
        if parsed is None or len(tokens) != 2:
            return None
        unit = unit or parsed[0]
        ranges.append({"title": m.get("groupItemTitle"), "bounds": parsed[1], "yes_token": tokens[0],
                       "no_token": tokens[1]})
    return {"id": str(ev["id"]), "title": ev["title"], "station": station.group(1), "unit": unit, "day": local_day,
            "end": end, "ranges": ranges}


def asks(token: str, *, http_get: HttpGet = requests.get) -> list[tuple[float, float]]:
    """(price, shares) offered for ``token``, cheapest first."""
    book = fw._get(CLOB_BOOK, {"token_id": token}, http_get=http_get) or {}
    return sorted((float(a["price"]), float(a["size"])) for a in book.get("asks") or [])


def walk(levels: Sequence[tuple[float, float]], cap_usd: float = FILL_CAP_USD) -> Optional[dict]:
    """Buy up the asks at MAX_PRICE or less while a dollar pays more than price plus fee."""
    shares = cost = fee = 0.0
    for price, size in levels:
        if price > MAX_PRICE or 1.0 - price - fw.polymarket_fee(price) <= 0:
            break
        room = cap_usd - cost - fee
        if room <= 0:
            break
        take = min(size, room / (price + fw.polymarket_fee(price)))
        shares += take
        cost += take * price
        fee += take * fw.polymarket_fee(price)
    if shares <= 0:
        return None
    return {"shares": shares, "cost": cost, "fee": fee, "avg_price": cost / shares,
            "best_ask": levels[0][0], "best_size": levels[0][1]}


def running_high(obs: Sequence[tuple[datetime, float]], day: date, tz: str, unit: str, at: datetime) -> Optional[int]:
    start = datetime.combine(day, time(0), ZoneInfo(tz))
    values = [c for when, c in obs if start <= when <= at]
    return settle_value(max(values), unit) if values else None


def record(conn, fund_id: str, now: datetime, events: Sequence[dict], obs: Mapping[str, list],
           *, book: Callable[[str], list] = asks) -> dict:
    """Note every decided range (once) and fill it the first time something is for sale."""
    seen = filled = 0
    stamp = now.isoformat()
    for ev in events:
        tz = STATION_TZ.get(ev["station"])
        high = running_high(obs.get(ev["station"], []), ev["day"], tz, ev["unit"], now) if tz else None
        if high is None:
            continue
        for r in ev["ranges"]:
            side = decided(r["bounds"], high)
            if side is None:
                continue
            token = r["yes_token"] if side == "YES" else r["no_token"]
            row = conn.execute("SELECT shares FROM shadow_fills WHERE fund_id = ? AND analyst = ? AND key = ?",
                               (fund_id, ANALYST, token)).fetchone()
            if row is not None and row[0] > 0:
                continue
            seen += row is None
            try:
                fill = walk(book(token))
            except (requests.RequestException, ValueError, KeyError):
                continue
            detail = {"station": ev["station"], "day": ev["day"].isoformat(), "high_so_far": high, "unit": ev["unit"],
                      "range": r["title"], **({k: round(v, 6) for k, v in fill.items()} if fill else {})}
            if row is None:
                conn.execute(
                    "INSERT INTO shadow_fills (fund_id, analyst, key, instrument_id, label, side, ts, shares, cost_usd,"
                    " fee_usd, detail_json, event_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (fund_id, ANALYST, token, f"polymarket:{token}", f"{ev['title']} {r['title']}", side, stamp,
                     fill["shares"] if fill else 0.0, fill["cost"] if fill else 0.0, fill["fee"] if fill else 0.0,
                     json.dumps(detail, sort_keys=True), ev["id"]))
            elif fill:
                conn.execute("UPDATE shadow_fills SET ts = ?, shares = ?, cost_usd = ?, fee_usd = ?, detail_json = ?"
                             " WHERE fund_id = ? AND analyst = ? AND key = ?",
                             (stamp, fill["shares"], fill["cost"], fill["fee"], json.dumps(detail, sort_keys=True),
                              fund_id, ANALYST, token))
            filled += bool(fill)
    return {"decided_new": seen, "filled_now": filled}


def grade(conn, fund_id: str, now: datetime, *, http_get: HttpGet = requests.get) -> int:
    """Settle fills whose events have resolved (Gamma's outcome prices)."""
    pending = conn.execute("SELECT DISTINCT event_id FROM shadow_fills WHERE fund_id = ? AND analyst = ?"
                           " AND shares > 0 AND outcome IS NULL", (fund_id, ANALYST)).fetchall()
    n = 0
    for (event_id,) in pending:
        try:
            ev = fw._get(f"{GAMMA_EVENTS}/{event_id}", {}, http_get=http_get) or {}
        except (requests.RequestException, ValueError):
            continue
        for m in ev.get("markets") or []:
            try:
                prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
                tokens = json.loads(m.get("clobTokenIds") or "[]")
            except ValueError:
                continue
            if not m.get("closed") or len(prices) != 2 or prices[0] not in (0.0, 1.0) or len(tokens) != 2:
                continue
            for token, won in ((tokens[0], prices[0] == 1.0), (tokens[1], prices[1] == 1.0)):
                n += conn.execute("UPDATE shadow_fills SET outcome = ?, settled_at = ? WHERE fund_id = ? AND analyst = ?"
                                  " AND key = ? AND outcome IS NULL AND shares > 0",
                                  (1 if won else 0, now.isoformat(), fund_id, ANALYST, token)).rowcount
    return n


def net_return(row: Mapping[str, Any]) -> float:
    spent = row["cost_usd"] + row["fee_usd"]
    return (row["shares"] * row["outcome"] - spent) / spent


def _interval(rows: list[dict]) -> Optional[list[float]]:
    by_day: dict[str, list[float]] = {}
    for r in rows:
        by_day.setdefault(r["day"], []).append(net_return(r))
    days = sorted(by_day)
    if len(days) < 2:
        return None
    rng = random.Random(SEED)
    draws = sorted(fmean([x for d in rng.choices(days, k=len(days)) for x in by_day[d]]) for _ in range(BOOTSTRAP))
    return [round(draws[int(0.025 * BOOTSTRAP)], 5), round(draws[int(0.975 * BOOTSTRAP) - 1], 5)]


def summary(conn, fund_id: str) -> dict:
    """The shadow record for the fund page, and the verdict once it is due."""
    rows = [dict(zip(("label", "side", "ts", "shares", "cost_usd", "fee_usd", "detail_json", "outcome"), r))
            for r in conn.execute("SELECT label, side, ts, shares, cost_usd, fee_usd, detail_json, outcome"
                                  " FROM shadow_fills WHERE fund_id = ? AND analyst = ? ORDER BY ts DESC",
                                  (fund_id, ANALYST))]
    for r in rows:
        r["day"] = json.loads(r["detail_json"]).get("day")
    fills = [r for r in rows if r["shares"] > 0]
    settled = [r for r in fills if r["outcome"] is not None]
    days = len({r["day"] for r in settled})
    ci = _interval(settled)
    due = len(settled) >= MIN_FILLS and days >= MIN_DAYS
    return {
        "mode": "shadow", "kind": "live", "since": rows[-1]["ts"] if rows else None,
        "decided_seen": len(rows), "fills": len(fills), "settled": len(settled), "settled_days": days,
        "losses": sum(r["outcome"] == 0 for r in settled),
        "mean_price": round(sum(r["cost_usd"] for r in fills) / sum(r["shares"] for r in fills), 4) if fills else None,
        "dollars_filled": round(sum(r["cost_usd"] + r["fee_usd"] for r in fills), 2),
        "mean_return": round(fmean(net_return(r) for r in settled), 5) if settled else None,
        "mean_return_ci": ci,
        "verdict": ("GO" if ci and ci[0] > 0 else "NO-GO") if due else "pending",
        "recent": [{"label": r["label"], "side": r["side"], "price": round(r["cost_usd"] / r["shares"], 4),
                    "dollars": round(r["cost_usd"] + r["fee_usd"], 2), "outcome": r["outcome"]} for r in fills[:8]],
    }


def run(conn, fund_id: str, now: datetime, *, http_get: HttpGet = requests.get) -> dict:
    """One live pass: record decided ranges, grade settled fills, and summarise."""
    events = open_events(now, http_get=http_get)
    obs = readings({e["station"] for e in events}, http_get=http_get) if events else {}
    out = record(conn, fund_id, now, events, obs, book=lambda token: asks(token, http_get=http_get))
    out["graded"] = grade(conn, fund_id, now, http_get=http_get)
    return {**summary(conn, fund_id), "last_pass": {"events": len(events), **out}}
