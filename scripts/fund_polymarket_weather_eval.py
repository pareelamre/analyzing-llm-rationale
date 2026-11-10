#!/usr/bin/env python3
"""The pre-registered Polymarket weather evaluation (docs/fund.md, "Polymarket weather evaluation").

Polymarket's daily highest-temperature markets settle on an airport's routine METAR readings,
rounded to whole degrees. From a seeded random sample of 2,000 settled events (2026-04-01 to
2026-11-08) this runs two tests at the fund's 08:00 and 20:00 UTC cycles:

1. the weather analyst's forecast model (fund_weather), unchanged, in each market's unit;
2. buying ranges the airport's own readings have already decided.

Prices are Polymarket's hourly price history plus a 1-cent half-spread and its weather taker
fee. Readings come from the Iowa Environmental Mesonet, forecasts from Open-Meteo's Previous
Runs API, station coordinates from OurAirports. Raw downloads are kept under --cache. Writes
reports/fund_polymarket_weather_eval.json.
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
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import fmean
from time import sleep
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import fund_weather as fw  # noqa: E402
from analyzing_llm_rationale.fund_pm_live import bounds_of, decided, settle_value  # noqa: E402,F401

GAMMA = "https://gamma-api.polymarket.com/events"
PRICES = "https://clob.polymarket.com/prices-history"
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
AIRPORTS = "https://davidmegginson.github.io/ourairports-data/airports.csv"
RULE_MARK = "hourly temperature remark"
FIRST, LAST = date(2026, 4, 1), date(2026, 11, 8)
SAMPLE, SEED = 2000, 20261110
CYCLES_UTC = (8, 20)
HALF_SPREAD, SENSITIVITY = 0.01, (0.005, 0.02)
MAX_AGE_S = 3 * 3600
MIN_SIGMA = {"C": 0.5, "F": 1.0}
LAST_ENTRY = time(9, 0)
LOOKBACK_H = 36
MAX_LEAD = 3
MIN_TRADES, BOOTSTRAP = 100, 2000
IEM_GAP_S = 5

def _cached(path: Path, fetch: Callable[[], Any], *, text: bool = False) -> Any:
    if path.is_file():
        return path.read_text() if text else json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if text else json.dumps(data))
    return data


def _slim(ev: dict) -> dict:
    """The fields the test reads (the full records run to hundreds of megabytes)."""
    markets = ev.get("markets") or []
    return {"id": ev["id"], "title": ev.get("title"), "endDate": ev.get("endDate"), "startDate": ev.get("startDate"),
            "markets": [{"groupItemTitle": m.get("groupItemTitle"), "outcomePrices": m.get("outcomePrices"),
                         "clobTokenIds": m.get("clobTokenIds"),
                         "description": m.get("description") if i == 0 else None} for i, m in enumerate(markets)]}


def all_events(session: requests.Session) -> list[dict]:
    """Every settled highest-temperature event ending FIRST..LAST, a week of end dates at a time."""
    found: dict[str, dict] = {}
    end = LAST + timedelta(days=1)
    while end > FIRST:
        start = max(FIRST, end - timedelta(days=7))
        for offset in range(0, 2000, 100):
            page = fw._get(GAMMA, {"tag_slug": "weather", "closed": "true", "limit": 100, "offset": offset,
                                   "end_date_min": f"{start}T00:00:00Z", "end_date_max": f"{end}T00:00:00Z"},
                           http_get=session.get)
            for ev in page or []:
                if str(ev.get("title", "")).startswith("Highest temperature in"):
                    found[str(ev["id"])] = _slim(ev)
            if not page or len(page) < 100:
                break
        end = start
    return sorted(found.values(), key=lambda e: str(e["id"]))


def parse_event(ev: dict) -> Optional[dict]:
    """The event's station, unit, local day and ranges, if it settles on the METAR rule."""
    markets = ev.get("markets") or []
    desc = " ".join(str(m.get("description") or "") for m in markets[:1])
    station = re.search(r"\(([A-Z]{4})\)", desc)
    if RULE_MARK not in desc or not station or not markets:
        return None
    ranges, unit = [], None
    for m in markets:
        parsed = bounds_of(m.get("groupItemTitle"))
        try:
            prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            token = json.loads(m.get("clobTokenIds") or "[]")[0]
        except (ValueError, IndexError, TypeError):
            return None
        if parsed is None or len(prices) != 2 or prices[0] not in (0.0, 1.0):
            return None
        unit = unit or parsed[0]
        ranges.append({"title": m.get("groupItemTitle"), "bounds": parsed[1], "token": token, "yes": prices[0] == 1.0})
    if sum(r["yes"] for r in ranges) != 1:
        return None
    day = re.search(r"on (\w+ \d{1,2})\??$", ev["title"])
    end = datetime.fromisoformat(str(ev["endDate"]).replace("Z", "+00:00"))
    try:
        local_day = datetime.strptime(f"{day.group(1)} {end.year}", "%B %d %Y").date() if day else None
    except ValueError:
        local_day = None
    if local_day is None or local_day > end.date():
        return None
    opened = datetime.fromisoformat(str(ev.get("startDate") or ev["endDate"]).replace("Z", "+00:00"))
    return {"id": str(ev["id"]), "title": ev["title"], "station": station.group(1), "unit": unit, "day": local_day,
            "opened": opened, "ranges": ranges}


def airports(cache: Path, session: requests.Session) -> dict[str, tuple[float, float]]:
    def fetch() -> str:
        resp = session.get(AIRPORTS, timeout=120, headers={"User-Agent": fw.USER_AGENT})
        resp.raise_for_status()
        return resp.text
    rows = csv.DictReader(io.StringIO(_cached(cache / "airports.csv", fetch, text=True)))
    return {r["ident"]: (float(r["latitude_deg"]), float(r["longitude_deg"])) for r in rows if len(r["ident"]) == 4}


def metars(station: str, cache: Path, session: requests.Session) -> list[tuple[datetime, float]]:
    """(UTC time, degrees C to the tenth) for every routine METAR with a temperature."""
    start, end = FIRST - timedelta(days=50), LAST + timedelta(days=2)
    params = [("station", station), ("data", "tmpf"), ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
              ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3"),
              ("year1", start.year), ("month1", start.month), ("day1", start.day),
              ("year2", end.year), ("month2", end.month), ("day2", end.day)]

    def fetch() -> str:
        # The archive serves one large request at a time politely; it answers bursts with 429.
        for attempt in range(6):
            resp = session.get(IEM_ASOS, params=params, timeout=300, headers={"User-Agent": fw.USER_AGENT})
            if resp.status_code != 429:
                break
            sleep(30 * (attempt + 1))
        resp.raise_for_status()
        sleep(IEM_GAP_S)
        return resp.text
    out = []
    for row in csv.DictReader(io.StringIO(_cached(cache / "metars" / f"{station}.csv", fetch, text=True))):
        try:
            when = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            out.append((when, round((float(row["tmpf"]) - 32) * 5 / 9, 1)))
        except (KeyError, ValueError):
            continue
    return sorted(out)


def forecasts(station: str, lat: float, lon: float, unit: str, cache: Path, session: requests.Session) -> dict:
    variables = [f"temperature_2m_previous_day{k}" for k in range(1, MAX_LEAD + 1)]
    raw = _cached(cache / "forecasts" / f"{station}.json", lambda: fw._get(fw.PREVIOUS_RUNS_API, {
        "latitude": lat, "longitude": lon, "models": ",".join(fw.MODELS), "hourly": ",".join(variables),
        "temperature_unit": "fahrenheit" if unit == "F" else "celsius", "timezone": "auto",
        "start_date": (FIRST - timedelta(days=50)).isoformat(), "end_date": (LAST + timedelta(days=1)).isoformat()},
        http_get=session.get))
    return {"tz": raw.get("timezone") or "UTC", "by_lead": {k: fw.daily_highs(raw, v) for k, v in enumerate(variables, 1)}}


def daily_settlement(obs: list[tuple[datetime, float]], tz: str, unit: str) -> dict[date, int]:
    zone, out = ZoneInfo(tz), {}
    for when, c in obs:
        d = when.astimezone(zone).date()
        v = settle_value(c, unit)
        out[d] = max(out.get(d, v), v)
    return out


def price_history(token: str, day: date, cache: Path) -> list[dict]:
    start = int(datetime.combine(day - timedelta(days=2), time(0), timezone.utc).timestamp())
    return _cached(cache / "prices" / f"{token}.json", lambda: (fw._get(PRICES, {
        "market": token, "startTs": start, "endTs": start + 4 * 86400, "fidelity": 60}, http_get=requests.get)
        .get("history") or []))


def point_at(history: list[dict], at: datetime) -> Optional[dict]:
    ts = at.timestamp()
    usable = [h for h in history if ts - MAX_AGE_S <= h["t"] <= ts]
    return max(usable, key=lambda h: h["t"]) if usable else None


def price_at(history: list[dict], at: datetime) -> Optional[float]:
    point = point_at(history, at)
    return float(point["p"]) if point else None


def cycles(ev: dict, tz: str) -> list[datetime]:
    """The fund's cycle times while the event is open, from two days before its day to its end."""
    out = []
    for back in (2, 1, 0):
        d = ev["day"] - timedelta(days=back)
        for h in CYCLES_UTC:
            at = datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc)
            if at >= ev["opened"]:
                out.append(at)
    end_local = datetime.combine(ev["day"] + timedelta(days=1), time(0), ZoneInfo(tz))
    return [t for t in out if t < end_local]


def lead_for(ev: dict, tz: str, at: datetime) -> Optional[int]:
    """The freshest previous_dayK whose runs, for every hour of the local day, are 6 h older than ``at``."""
    end_local = datetime.combine(ev["day"] + timedelta(days=1), time(0), ZoneInfo(tz))
    k = max(1, math.ceil((end_local - at + timedelta(hours=6)).total_seconds() / 86400))
    return k if k <= MAX_LEAD else None


def evaluate(events: list[dict], stations: dict[str, dict], half_spread: float) -> tuple[list[dict], list[dict], list[dict]]:
    """Brackets priced and trades for test 1, and trades for test 2."""
    rows = {(unit, k): [] for unit in ("C", "F") for k in range(1, MAX_LEAD + 1)}
    for code, st in stations.items():
        for k, highs in st["forecasts"]["by_lead"].items():
            rows[(st["unit"], k)] += fw.calibration_rows(fw.City(code, code, code, code, 0, 0, st["tz"]), highs,
                                                         {d: float(v) for d, v in st["settled"].items()})
    fits: dict[tuple, Optional[fw.Calibration]] = {}
    brackets, trades1, trades2 = [], [], []
    for ev in events:
        st = stations.get(ev["station"])
        if st is None:
            continue
        tz, unit = st["tz"], ev["unit"]
        local_midnight = datetime.combine(ev["day"], time(0), ZoneInfo(tz))
        for at in cycles(ev, tz):
            prices = {r["token"]: price_at(st["prices"].get(r["token"], []), at) for r in ev["ranges"]}
            # Test 1: the forecast, before the day's morning.
            k = lead_for(ev, tz, at)
            if k is not None and local_midnight + timedelta(hours=LAST_ENTRY.hour) - timedelta(hours=LOOKBACK_H) <= at \
                    < local_midnight + timedelta(hours=LAST_ENTRY.hour):
                known = at.date() - timedelta(days=1)
                key = (unit, k, known)
                if key not in fits:
                    fits[key] = fw.fit(rows[(unit, k)], known, MIN_SIGMA[unit])
                cal, e = fits[key], fw.ensemble(st["forecasts"]["by_lead"][k], ev["day"])
                if cal is not None and e is not None:
                    centre, sigma = e[0] + cal.bias.get(ev["station"], 0.0), cal.sigma(e[1])
                    for r in ev["ranges"]:
                        p_mkt = prices[r["token"]]
                        if p_mkt is None:
                            continue
                        p = fw.chance(centre, sigma, r["bounds"])
                        brackets.append({"day": ev["day"].isoformat(), "station": ev["station"], "p": p,
                                         "market": p_mkt, "yes": r["yes"]})
                        hit = fw.rule(p, p_mkt - half_spread, p_mkt + half_spread, fw.polymarket_fee)
                        if hit:
                            side, price, edge = hit
                            won = (side == "YES") == r["yes"]
                            trades1.append({"day": ev["day"].isoformat(), "station": ev["station"], "side": side,
                                            "price": price, "edge": edge, "won": won,
                                            "return": ((1.0 if won else 0.0) - price - fw.polymarket_fee(price)) / price})
            # Test 2: ranges the readings have already decided, while the day is under way.
            if at < local_midnight:
                continue
            so_far = [(when, c) for when, c in st["obs"] if local_midnight <= when <= at]
            if not so_far:
                continue
            high = settle_value(max(c for _, c in so_far), unit)
            for r in ev["ranges"]:
                side = decided(r["bounds"], high)
                p_mkt = prices[r["token"]]
                if side is None or p_mkt is None:
                    continue
                price = (p_mkt if side == "YES" else 1.0 - p_mkt) + half_spread
                if price > 0.99 or 1.0 - price - fw.polymarket_fee(price) <= 0:
                    continue
                won = (side == "YES") == r["yes"]
                # Diagnostic added after the first run: was the price set before the reading that
                # decided the range? Such a price was no longer on offer once the outcome was known.
                point = point_at(st["prices"].get(r["token"], []), at)
                first = decided_at(so_far, r["bounds"], unit, side)
                trades2.append({"day": ev["day"].isoformat(), "station": ev["station"], "side": side, "price": price,
                                "won": won, "return": ((1.0 if won else 0.0) - price - fw.polymarket_fee(price)) / price,
                                "priced_before_decision": bool(point and first and point["t"] < first.timestamp())})
    return brackets, trades1, trades2


def decided_at(readings: list[tuple[datetime, float]], bounds: tuple[float, float], unit: str,
               side: str) -> Optional[datetime]:
    """The time of the reading that first decided the range."""
    running = None
    for when, c in sorted(readings):
        v = settle_value(c, unit)
        running = v if running is None else max(running, v)
        if decided(bounds, running) == side:
            return when
    return None


def interval(items: list[dict], stat: Callable[[list[dict]], Optional[float]]) -> Optional[list[float]]:
    by_day: dict[str, list[dict]] = {}
    for it in items:
        by_day.setdefault(it["day"], []).append(it)
    days = sorted(by_day)
    if len(days) < 5:
        return None
    rng = random.Random(SEED)
    draws = sorted(v for v in (stat([it for d in rng.choices(days, k=len(days)) for it in by_day[d]])
                               for _ in range(BOOTSTRAP)) if v is not None)
    return [round(draws[int(0.025 * len(draws))], 5), round(draws[int(0.975 * len(draws)) - 1], 5)]


def mean_return(trades: list[dict]) -> Optional[float]:
    return fmean(t["return"] for t in trades) if trades else None


def brier_skill(brackets: list[dict]) -> Optional[float]:
    if not brackets:
        return None
    model = fmean((b["p"] - b["yes"]) ** 2 for b in brackets)
    market = fmean((b["market"] - b["yes"]) ** 2 for b in brackets)
    return 1 - model / market if market > 0 else None


def trade_summary(trades: list[dict]) -> dict:
    ret = mean_return(trades)
    return {"trades": len(trades), "days": len({t["day"] for t in trades}), "losses": sum(not t["won"] for t in trades),
            "mean_return": None if ret is None else round(ret, 5), "mean_return_ci": interval(trades, mean_return),
            "mean_price": round(fmean(t["price"] for t in trades), 4) if trades else None}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "polymarket_weather_eval")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_polymarket_weather_eval.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    session = requests.Session()
    raw = _cached(args.cache / "events.json", lambda: all_events(session))
    parsed = [e for e in (parse_event(ev) for ev in raw) if e and FIRST <= e["day"] <= LAST]
    sample = random.Random(SEED).sample(parsed, min(SAMPLE, len(parsed)))
    print(f"{len(raw)} events, {len(parsed)} on the METAR rule, {len(sample)} sampled", flush=True)
    coords = airports(args.cache, session)
    stations: dict[str, dict] = {}
    for code in sorted({e["station"] for e in sample}):
        if code not in coords:
            print(f"{code}: no coordinates, skipped", flush=True)
            continue
        unit = next(e["unit"] for e in sample if e["station"] == code)
        fc = forecasts(code, *coords[code], unit, args.cache, session)
        obs = metars(code, args.cache, session)
        stations[code] = {"unit": unit, "tz": fc["tz"], "forecasts": fc, "obs": obs,
                          "settled": daily_settlement(obs, fc["tz"], unit), "prices": {}}
        print(f"{code}: {len(obs)} readings, tz {fc['tz']}", flush=True)
    jobs = [(r["token"], e) for e in sample if e["station"] in stations for r in e["ranges"]]

    def load(job: tuple[str, dict]) -> tuple[str, str, list]:
        token, ev = job
        try:
            return ev["station"], token, price_history(token, ev["day"], args.cache)
        except (requests.RequestException, ValueError):
            return ev["station"], token, []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (code, token, hist) in enumerate(pool.map(load, jobs), 1):
            stations[code]["prices"][token] = hist
            if i % 2000 == 0:
                print(f"prices {i}/{len(jobs)}", flush=True)
    agree = [stations[e["station"]]["settled"].get(e["day"]) for e in sample if e["station"] in stations]
    rebuilt = sum(1 for e in sample if e["station"] in stations and any(
        r["yes"] and r["bounds"][0] < (stations[e["station"]]["settled"].get(e["day"]) or -999) < r["bounds"][1]
        for r in e["ranges"]))
    brackets, t1, t2 = evaluate(sample, stations, HALF_SPREAD)
    s1, s2 = trade_summary(t1), trade_summary(t2)
    skill = brier_skill(brackets)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "docs/fund.md, Polymarket weather evaluation (pre-registered 2026-11-10)",
        "sample": {"events_on_rule": len(parsed), "sampled": len(sample), "stations": len(stations),
                   "settlement_rebuilt_matches_winner": f"{rebuilt}/{sum(v is not None for v in agree)}"},
        "test1_forecast": {"verdict": "GO" if s1["mean_return_ci"] and s1["mean_return_ci"][0] > 0 else "SHADOW",
                           "brackets": len(brackets), "brier_skill": None if skill is None else round(skill, 5),
                           "brier_skill_ci": interval(brackets, brier_skill), **s1},
        "test2_decided": {"verdict": "GO" if s2["trades"] >= MIN_TRADES and s2["mean_return_ci"]
                          and s2["mean_return_ci"][0] > 0 else "NO-GO", **s2,
                          "priced_before_the_deciding_reading": sum(t["priced_before_decision"] for t in t2),
                          "note": "Not acted on: most prices predate the reading that decided the range (docs/fund.md)."},
        "sensitivity (description only)": {f"half_spread_{h}": {"test1": trade_summary(r[1]), "test2": trade_summary(r[2])}
                                           for h, r in ((h, evaluate(sample, stations, h)) for h in SENSITIVITY)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("sample", "test1_forecast", "test2_decided")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
