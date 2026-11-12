#!/usr/bin/env python3
"""The pre-registered Kalshi hourly temperature test (docs/fund.md, "Kalshi hourly temperature test").

For every settled hourly event in New York, Coastal Los Angeles, Chicago Metro and Miami that
settles on the Kalshi Weather Index, forecasts the index 50 minutes before the hour as the
previous hour's settled value plus HRRR's change over the hour, prices each "above" market from
the city's recent errors at that time of day, applies the fund's event rule at the order book's
best bid and ask, and writes reports/fund_hourly_temp_eval.json with the verdict. Raw downloads
are kept under --cache, so a re-run reads them instead of asking the APIs again.

``--open`` runs the opening-prices amendment instead (docs/fund.md, "Amendment (2026-11-12): opening
prices"): ladder mismatches in each hour's opening book, and the model with only what is known at
the open (the reading for H-2, HRRR's change from H-2 to H) traded at the opening bid and ask. It
writes reports/fund_hourly_open_eval.json.

HRRR comes from NOAA's archive on AWS as GRIB2, so the first run needs ``pip install eccodes``
(not a project dependency; only the download step imports it).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import fund_weather as fw  # noqa: E402

HRRR_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{run:%Y%m%d}/conus/hrrr.t{run:%H}z.wrfsfcf{lead:02d}.grib2"
HRRR_FIELD = ("TMP", "2 m above ground")
FIRST_HOUR = datetime(2026, 9, 10, 15, tzinfo=timezone.utc)
LAST_HOUR = datetime(2026, 11, 11, 23, tzinfo=timezone.utc)
DECISION_LEAD = timedelta(minutes=50)
RUN_READY_AFTER = timedelta(hours=2)
WINDOW = timedelta(days=21)
HOUR_BAND = 2
MIN_ERRORS = 40
TRADER_GAP = 0.10
BOOTSTRAP, SEED = 2000, 20261112
MIN_TRADES = 100
KALSHI_GAP_S, HRRR_THREADS = 0.06, 6
BANDS = (("00-06", 0, 6), ("06-12", 6, 12), ("12-18", 12, 18), ("18-24", 18, 24))


@dataclass(frozen=True)
class City:
    series: str
    name: str
    tz: str
    stations: tuple[tuple[str, float, float], ...]

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(s[0] for s in self.stations)


# Member stations from each index's CFTC certification (Appendix B); coordinates from aviationweather.gov.
CITIES = (
    City("KXTEMPNYCHS", "New York", "America/New_York", (
        ("KLGA", 40.77945, -73.88027), ("KEWR", 40.68277, -74.16924), ("KTEB", 40.85898, -74.05617),
        ("KCDW", 40.87645, -74.28285), ("KHPN", 41.06236, -73.70455), ("KFRG", 40.73442, -73.41637),
        ("KSMQ", 40.62407, -74.66900), ("KISP", 40.79389, -73.10183))),
    City("KXTEMPLAXHS", "Coastal Los Angeles", "America/Los_Angeles", (
        ("KHHR", 33.92352, -118.33291), ("KLAX", 33.93817, -118.38660), ("KSMO", 34.02099, -118.44713),
        ("KLGB", 33.81177, -118.14717), ("KSNA", 33.67976, -117.86745))),
    City("KXTEMPCHIHS", "Chicago Metro", "America/Chicago", (
        ("KMDW", 41.78412, -87.75515), ("KORD", 41.96017, -87.93161), ("KPWK", 42.12076, -87.90480),
        ("KDPA", 41.89641, -88.25119), ("KARR", 41.77133, -88.48146))),
    City("KXTEMPMIAH", "Miami", "America/New_York", (
        ("KMIA", 25.78806, -80.31692), ("KOPF", 25.91017, -80.28283), ("KFLL", 26.07872, -80.16223),
        ("KFXE", 26.19904, -80.17763), ("KPMP", 26.24636, -80.11104))),
)


def _cached(path: Path, fetch: Callable[[], Any]) -> Any:
    if path.is_file():
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def event_hour(ev: dict) -> Optional[datetime]:
    """The settlement hour H of an hourly event (its strike date, else its markets' close time)."""
    if ev.get("strike_date"):
        return _ts(ev["strike_date"])
    closes = {m.get("close_time") for m in ev.get("markets") or []} - {None}
    return _ts(closes.pop()) if len(closes) == 1 else None


def on_index(ev: dict) -> bool:
    return any("Synoptic Data" in (m.get("rules_primary") or "") for m in ev.get("markets") or [])


def settled_values(events: list[dict]) -> dict[datetime, float]:
    """Each hour's settled index value, from its markets' expiration values."""
    out = {}
    for ev in events:
        hour = event_hour(ev)
        values = {fw._price(m.get("expiration_value")) for m in ev.get("markets") or []} - {None}
        if hour and len(values) == 1:
            out[hour] = values.pop()
    return out


def run_for(hour: datetime) -> datetime:
    """The newest hourly HRRR run initialised at least RUN_READY_AFTER before the decision."""
    return (hour - DECISION_LEAD - RUN_READY_AFTER).replace(minute=0, second=0, microsecond=0)


def city_forecasts(city: City, stations: dict[tuple[datetime, int], dict[str, float]]) -> dict[datetime, dict[datetime, float]]:
    """Per run, the mean over the city's stations of each forecast hour, as {run: {valid hour: degrees F}}."""
    out: dict[datetime, dict[datetime, float]] = {}
    for (run, lead), temps in stations.items():
        if all(s in temps for s in city.members):
            out.setdefault(run, {})[run + timedelta(hours=lead)] = fmean(temps[s] for s in city.members)
    return out


def hrrr_change(runs: dict[datetime, dict[datetime, float]], hour: datetime, gap: int = 1) -> Optional[float]:
    """The run's forecast for the hour minus its forecast for ``gap`` hours before."""
    run = runs.get(run_for(hour)) or {}
    now, before = run.get(hour), run.get(hour - timedelta(hours=gap))
    return None if now is None or before is None else now - before


def centres(values: dict[datetime, float], runs: dict[datetime, dict[datetime, float]],
            hours: list[datetime], *, use_hrrr: bool = True, gap: int = 1) -> dict[datetime, float]:
    """f = the settled value ``gap`` hours before, plus HRRR's change since then."""
    out = {}
    for hour in hours:
        last = values.get(hour - timedelta(hours=gap))
        if last is None:
            continue
        if not use_hrrr:
            out[hour] = last
        elif (delta := hrrr_change(runs, hour, gap)) is not None:
            out[hour] = last + delta
    return out


def _local_hour(hour: datetime, tz: str) -> int:
    return hour.astimezone(ZoneInfo(tz)).hour


def recent_errors(hour: datetime, tz: str, values: dict[datetime, float], f: dict[datetime, float],
                  gap: int = 1) -> list[float]:
    """Errors of the same city's earlier hours, settled by the decision (``gap`` hours back), at the same local hour +-2."""
    target = _local_hour(hour, tz)
    out = []
    for h, centre in f.items():
        if not hour - WINDOW <= h <= hour - timedelta(hours=gap) or h not in values:
            continue
        apart = abs(_local_hour(h, tz) - target)
        if min(apart, 24 - apart) <= HOUR_BAND:
            out.append(values[h] - centre)
    return out


def chance_above(centre: float, errors: list[float], strike: float) -> float:
    return (sum(centre + e > strike for e in errors) + 0.5) / (len(errors) + 1)


def quote_at(rows: Optional[list], at: datetime) -> tuple[Optional[float], Optional[float]]:
    ts = at.timestamp()
    usable = [r for r in rows or [] if r.get("end_period_ts", ts + 1) <= ts]
    if not usable:
        return None, None
    last = max(usable, key=lambda r: r["end_period_ts"])
    return (fw._price((last.get("yes_bid") or {}).get("close_dollars")),
            fw._price((last.get("yes_ask") or {}).get("close_dollars")))


def opening_quote(rows: Optional[list], opened: datetime, field: str = "open_dollars") -> tuple[Optional[float], Optional[float]]:
    """The book in the first minute after the market opens: its first values, or with ``close_dollars`` its last."""
    end = int(opened.timestamp()) + 60
    first = next((r for r in rows or [] if r.get("end_period_ts") == end), None)
    if first is None:
        return None, None
    return fw._price((first.get("yes_bid") or {}).get(field)), fw._price((first.get("yes_ask") or {}).get(field))


def ladder_mismatch(rungs: list[tuple[float, Optional[float], Optional[float], bool]]) -> Optional[dict]:
    """The best pair of rungs k1 < k2 with ask(k1) below bid(k2): Yes on k1 plus No on k2 pays at least $1.

    ``rungs`` holds (strike, bid, ask, settled yes). Counts only when $1 beats the cost plus both fees.
    """
    best = None
    for k1, _, ask1, yes1 in rungs:
        for k2, bid2, _, yes2 in rungs:
            if k1 >= k2 or ask1 is None or bid2 is None or not 0 < ask1 < 1 or not 0 < bid2 < 1:
                continue
            cost = ask1 + (1.0 - bid2)
            fees = fw.kalshi_fee(ask1) + fw.kalshi_fee(1.0 - bid2)
            sure = 1.0 - cost - fees
            if sure > 0 and (best is None or sure > best["sure_profit"]):
                payoff = (1.0 if yes1 else 0.0) + (0.0 if yes2 else 1.0)
                best = {"low": k1, "high": k2, "ask_low": ask1, "bid_high": bid2, "cost": round(cost, 4),
                        "sure_profit": round(sure, 4), "payoff": payoff, "return": (payoff - cost - fees) / cost}
    return best


def trader_rule(p: float, bid: Optional[float], ask: Optional[float]) -> Optional[tuple[str, float]]:
    """The article's rule: trade whenever the model beats the price by TRADER_GAP."""
    best = None
    if ask is not None and 0 < ask < 1 and p - ask >= TRADER_GAP:
        best = ("YES", ask, p - ask)
    if bid is not None and 0 < bid < 1 and (1 - p) - (1 - bid) >= TRADER_GAP and (best is None or bid - p > best[2]):
        best = ("NO", 1 - bid, bid - p)
    return None if best is None else best[:2]


def _return(side: str, price: float, yes: bool) -> float:
    payoff = 1.0 if (side == "YES") == yes else 0.0
    return (payoff - price - fw.kalshi_fee(price)) / price


def decide(city: City, events: list[dict], runs: dict[datetime, dict[datetime, float]],
           candles: dict[str, dict[str, list]], *, use_hrrr: bool = True, at_open: bool = False,
           field: str = "open_dollars") -> tuple[list[dict], list[dict], list[dict], dict]:
    """Every priced market, the rule's trades and the trader rule's trades for one city.

    By default the decision is 50 minutes before the hour on the last hour's reading. With ``at_open``
    it is at the market's open (1 hour before), on the reading for H-2, at the opening book (``field``).
    """
    gap = 2 if at_open else 1
    values = settled_values(events)
    by_hour = {h: ev for ev in events if (h := event_hour(ev))}
    f = centres(values, runs, sorted(by_hour), use_hrrr=use_hrrr, gap=gap)
    brackets, trades, trader = [], [], []
    coverage = {"events": len(by_hour), "with_centre": 0, "forecast": 0}
    for hour, ev in sorted(by_hour.items()):
        if not FIRST_HOUR <= hour <= LAST_HOUR or hour not in f:
            continue
        coverage["with_centre"] += 1
        errors = recent_errors(hour, city.tz, values, f, gap)
        if len(errors) < MIN_ERRORS:
            continue
        coverage["forecast"] += 1
        at = hour - (timedelta(hours=1) if at_open else DECISION_LEAD)
        local = _local_hour(hour, city.tz)
        for m in ev.get("markets") or []:
            strike = fw._price(m.get("floor_strike"))
            if m.get("strike_type") != "greater" or strike is None or m.get("result") not in ("yes", "no"):
                continue
            p = chance_above(f[hour], errors, strike)
            rows = candles.get(ev["event_ticker"], {}).get(m["ticker"])
            bid, ask = opening_quote(rows, at, field) if at_open else quote_at(rows, at)
            yes = m["result"] == "yes"
            row = {"city": city.series, "day": hour.date().isoformat(), "hour": hour.isoformat(), "local_hour": local,
                   "ticker": m["ticker"], "p": round(p, 4), "bid": bid, "ask": ask, "yes": yes}
            brackets.append(row)
            if hit := fw.rule(p, bid, ask):
                side, price, edge = hit
                trades.append({**row, "side": side, "price": price, "edge": round(edge, 4), "return": _return(side, price, yes)})
            if hit := trader_rule(p, bid, ask):
                side, price = hit
                trader.append({**row, "side": side, "price": price, "return": _return(side, price, yes)})
    return brackets, trades, trader, coverage


def _two_sided(b: dict) -> bool:
    return b["bid"] is not None and b["ask"] is not None and 0 < b["bid"] <= b["ask"] < 1


def brier_skill(brackets: list[dict]) -> Optional[float]:
    rows = [b for b in brackets if _two_sided(b)]
    if not rows:
        return None
    model = fmean((b["p"] - b["yes"]) ** 2 for b in rows)
    market = fmean(((b["bid"] + b["ask"]) / 2 - b["yes"]) ** 2 for b in rows)
    return 1 - model / market if market > 0 else None


def mean_return(trades: list[dict]) -> Optional[float]:
    return fmean(t["return"] for t in trades) if trades else None


def bootstrap(items: list[dict], stat: Callable[[list[dict]], Optional[float]]) -> Optional[list[float]]:
    """95% interval from resampling UTC dates (an hour's markets, and a day's hours, move together)."""
    by_day: dict[str, list[dict]] = {}
    for it in items:
        by_day.setdefault(it["day"], []).append(it)
    days = sorted(by_day)
    if len(days) < 5:
        return None
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        v = stat([it for d in rng.choices(days, k=len(days)) for it in by_day[d]])
        if v is not None:
            draws.append(v)
    if not draws:
        return None
    draws.sort()
    return [round(draws[int(0.025 * len(draws))], 5), round(draws[int(0.975 * len(draws)) - 1], 5)]


def summary(brackets: list[dict], trades: list[dict]) -> dict:
    ret, skill = mean_return(trades), brier_skill(brackets)
    return {"markets": len(brackets), "two_sided": sum(_two_sided(b) for b in brackets),
            "brier_skill": None if skill is None else round(skill, 5), "brier_skill_ci": bootstrap(brackets, brier_skill),
            "trades": len(trades), "days_traded": len({t["day"] for t in trades}),
            "mean_return": None if ret is None else round(ret, 5), "mean_return_ci": bootstrap(trades, mean_return),
            "win_rate": round(fmean((t["return"] > 0) for t in trades), 4) if trades else None,
            "mean_price": round(fmean(t["price"] for t in trades), 4) if trades else None}


def reliability(brackets: list[dict]) -> list[dict]:
    edges = (0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01)
    bins = []
    for lo, hi in zip(edges, edges[1:]):
        rows = [b for b in brackets if lo <= b["p"] < hi]
        if rows:
            two = [b for b in rows if _two_sided(b)]
            bins.append({"model": f"{lo:.2f}-{min(hi, 1):.2f}", "n": len(rows),
                         "mean_forecast": round(fmean(b["p"] for b in rows), 4),
                         "observed": round(fmean(b["yes"] for b in rows), 4),
                         "mean_market_mid": round(fmean((b["bid"] + b["ask"]) / 2 for b in two), 4) if two else None})
    return bins


def report(results: dict[str, dict], *, generated: Optional[str] = None) -> dict:
    brackets = [b for r in results.values() for b in r["brackets"]]
    trades = [t for r in results.values() for t in r["trades"]]
    pooled = summary(brackets, trades)
    ci = pooled["mean_return_ci"]
    verdict = "GO" if pooled["trades"] >= MIN_TRADES and ci and ci[0] > 0 else "NO-GO"
    days = sorted({b["day"] for b in brackets})

    def band(rows: list[dict], lo: int, hi: int) -> list[dict]:
        return [r for r in rows if lo <= r["local_hour"] < hi]
    no_hrrr = [b for r in results.values() for b in r["no_hrrr"]["brackets"]]
    no_hrrr_trades = [t for r in results.values() for t in r["no_hrrr"]["trades"]]
    trader = [t for r in results.values() for t in r["trader"]]
    return {
        "generated_at": generated or datetime.now(timezone.utc).isoformat(), "verdict": verdict,
        "rule": "docs/fund.md, Kalshi hourly temperature test (pre-registered 2026-11-12)",
        "window": {"first": days[0] if days else None, "last": days[-1] if days else None,
                   "hours": [FIRST_HOUR.isoformat(), LAST_HOUR.isoformat()]},
        "sources": {"markets": "Kalshi public API (events, one-minute event candlesticks)",
                    "forecast": "NOAA HRRR on AWS (noaa-hrrr-bdp-pds), 2 m temperature, mean over each city's stations",
                    "settlement": "Kalshi Weather Index, via each event's expiration value"},
        "coverage": {c: r["coverage"] for c, r in results.items()},
        "pooled": pooled,
        "by_city": {c.series: {"city": c.name, **summary(results[c.series]["brackets"], results[c.series]["trades"])}
                    for c in CITIES if c.series in results},
        "by_local_time (description only)": {name: summary(band(brackets, lo, hi), band(trades, lo, hi))
                                             for name, lo, hi in BANDS},
        "without_hrrr (description only)": summary(no_hrrr, no_hrrr_trades),
        "trader_rule (description only)": {"gap": TRADER_GAP, **summary([], trader)},
        "reliability": reliability(brackets),
    }


def ladder_hours(city: City, events: list[dict], candles: dict[str, dict[str, list]]) -> tuple[int, list[dict]]:
    """Hours priced at the open (two or more rungs with an opening quote), and each hour's best mismatch."""
    priced, found = 0, []
    for ev in events:
        hour = event_hour(ev)
        if not hour or not FIRST_HOUR <= hour <= LAST_HOUR:
            continue
        rungs = []
        for m in ev.get("markets") or []:
            strike = fw._price(m.get("floor_strike"))
            if m.get("strike_type") != "greater" or strike is None or m.get("result") not in ("yes", "no"):
                continue
            bid, ask = opening_quote(candles.get(ev["event_ticker"], {}).get(m["ticker"]), hour - timedelta(hours=1))
            if bid is not None or ask is not None:
                rungs.append((strike, bid, ask, m["result"] == "yes"))
        if len(rungs) < 2:
            continue
        priced += 1
        if hit := ladder_mismatch(rungs):
            found.append({"city": city.series, "day": hour.date().isoformat(), "hour": hour.isoformat(), **hit})
    return priced, found


def open_report(results: dict[str, dict], *, generated: Optional[str] = None) -> dict:
    """Both opening-price verdicts, with the end-of-minute book and the 10-minute test for description."""
    found = [h for r in results.values() for h in r["ladder"]]
    priced = sum(r["priced"] for r in results.values())
    brackets = [b for r in results.values() for b in r["brackets"]]
    trades = [t for r in results.values() for t in r["trades"]]
    model = summary(brackets, trades)
    ci = model["mean_return_ci"]
    test1 = "GO" if len(found) >= MIN_TRADES else "NO-GO"
    test2 = "GO" if model["trades"] >= MIN_TRADES and ci and ci[0] > 0 else "NO-GO"
    spreads = [b["ask"] - b["bid"] for b in brackets if _two_sided(b)]
    late = [b for r in results.values() for b in r["minute_close"]["brackets"]]
    late_trades = [t for r in results.values() for t in r["minute_close"]["trades"]]

    def band(rows: list[dict], lo: int, hi: int) -> list[dict]:
        return [x for x in rows if lo <= x["local_hour"] < hi]
    return {
        "generated_at": generated or datetime.now(timezone.utc).isoformat(),
        "verdict": {"test1_ladder_mismatches": test1, "test2_model_at_open": test2},
        "rule": "docs/fund.md, Amendment (2026-11-12): opening prices",
        "hours": [FIRST_HOUR.isoformat(), LAST_HOUR.isoformat()],
        "test1_ladder_mismatches": {
            "hours_priced_at_open": priced, "hours_with_mismatch": len(found),
            "by_city": {c: len(r["ladder"]) for c, r in results.items()},
            "payoff_below_1": sum(h["payoff"] < 1 for h in found),
            "mean_sure_profit": round(fmean(h["sure_profit"] for h in found), 4) if found else None,
            "mean_return": round(mean_return(found), 5) if found else None,
            "mean_return_ci": bootstrap(found, mean_return), "examples": found[:10]},
        "test2_model_at_open": model,
        "test2_coverage": {c: r["coverage"] for c, r in results.items()},
        "test2_by_city": {c.series: {"city": c.name, **summary(results[c.series]["brackets"], results[c.series]["trades"])}
                          for c in CITIES if c.series in results},
        "test2_by_local_time (description only)": {name: summary(band(brackets, lo, hi), band(trades, lo, hi))
                                                   for name, lo, hi in BANDS},
        "test2_end_of_first_minute (description only)": summary(late, late_trades),
        "opening_spread (description only)": {"two_sided": len(spreads),
                                              "mean": round(fmean(spreads), 4) if spreads else None},
        "reliability": reliability(brackets),
    }


def collect(city: City, cache: Path, session: requests.Session) -> dict:
    """One city's settled index events and one-minute candles per event, up to each decision."""
    raw = _cached(cache / city.series / "events.json",
                  lambda: fw.events(city.series, "settled", http_get=session.get, pages=60))
    events = [ev for ev in raw if on_index(ev) and (h := event_hour(ev)) and h <= LAST_HOUR]
    candles = {}
    for ev in events:
        hour = event_hour(ev)
        if not FIRST_HOUR <= hour <= LAST_HOUR:
            continue
        ticker = ev["event_ticker"]
        lo, hi = int((hour - timedelta(hours=1)).timestamp()), int((hour - DECISION_LEAD).timestamp())
        body = _cached(cache / city.series / "candles" / f"{ticker}.json", lambda ticker=ticker, lo=lo, hi=hi: _kalshi(
            session, f"{fw.KALSHI_API}/series/{city.series}/events/{ticker}/candlesticks",
            {"start_ts": lo, "end_ts": hi, "period_interval": 1}))
        candles[ticker] = dict(zip(body.get("market_tickers") or [], body.get("market_candlesticks") or []))
        if len(candles) % 500 == 0:
            print(f"{city.name}: candles for {len(candles)} events", file=sys.stderr, flush=True)
    return {"events": events, "candles": candles}


def _kalshi(session: requests.Session, url: str, params: dict) -> Any:
    out = fw._get(url, params, http_get=session.get)
    time.sleep(KALSHI_GAP_S)
    return out


def needed_files(hours: list[datetime], gap: int = 1) -> list[tuple[datetime, int]]:
    """The (run, lead) files that give each hour, and the hour ``gap`` before it, from the hour's run."""
    out = set()
    for hour in hours:
        run = run_for(hour)
        lead = int((hour - run) / timedelta(hours=1))
        out |= {(run, lead), (run, lead - gap)}
    return sorted(out)


def _hrrr_bytes(session: requests.Session, run: datetime, lead: int) -> Optional[bytes]:
    """The 2 m temperature message of one HRRR file, by byte range from its index, or None if missing."""
    url = HRRR_URL.format(run=run, lead=lead)
    for attempt in range(4):
        try:
            idx = session.get(url + ".idx", timeout=60)
            if idx.status_code == 404:
                return None
            idx.raise_for_status()
            lines = idx.text.splitlines()
            for i, line in enumerate(lines):
                parts = line.split(":")
                if (parts[3], parts[4]) == HRRR_FIELD:
                    end = lines[i + 1].split(":")[1] if i + 1 < len(lines) else ""
                    resp = session.get(url, headers={"Range": f"bytes={parts[1]}-{int(end) - 1 if end else ''}"}, timeout=120)
                    resp.raise_for_status()
                    return resp.content
            return None
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    return None


def station_forecasts(files: list[tuple[datetime, int]], cache: Path, session: requests.Session) -> dict[tuple[datetime, int], dict[str, float]]:
    """Every station's HRRR 2 m temperature (degrees F) per (run, lead), cached as one small JSON per file."""
    stations = {s[0]: (s[1], s[2]) for c in CITIES for s in c.stations}
    path = lambda run, lead: cache / "hrrr" / f"{run:%Y%m%dT%H}f{lead:02d}.json"  # noqa: E731
    out = {(run, lead): json.loads(path(run, lead).read_text()) for run, lead in files if path(run, lead).is_file()}
    todo = [k for k in files if k not in out]
    if not todo:
        return out
    import eccodes  # only the download step needs GRIB decoding
    nearest: dict[str, int] = {}
    with ThreadPoolExecutor(HRRR_THREADS) as pool:
        for n, ((run, lead), data) in enumerate(zip(todo, pool.map(lambda k: _hrrr_bytes(session, *k), todo)), 1):
            temps: dict[str, float] = {}
            if data:
                gid = eccodes.codes_new_from_message(data)
                try:
                    if not nearest:
                        nearest.update({s: eccodes.codes_grib_find_nearest(gid, la, lo)[0].index
                                        for s, (la, lo) in stations.items()})
                    values = eccodes.codes_get_values(gid)
                    temps = {s: round((float(values[i]) - 273.15) * 9 / 5 + 32, 3) for s, i in nearest.items()}
                finally:
                    eccodes.codes_release(gid)
            path(run, lead).parent.mkdir(parents=True, exist_ok=True)
            path(run, lead).write_text(json.dumps(temps))
            out[(run, lead)] = temps
            if n % 200 == 0:
                print(f"HRRR {n}/{len(todo)} files", file=sys.stderr, flush=True)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "hourly_temp_eval")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--open", action="store_true", help="run the opening-prices amendment")
    args = ap.parse_args(argv)
    session = requests.Session()
    data = {city.series: collect(city, args.cache, session) for city in CITIES}
    hours = sorted({h for d in data.values() for ev in d["events"] if (h := event_hour(ev)) and h >= FIRST_HOUR})
    stations = station_forecasts(needed_files(hours, gap=2 if args.open else 1), args.cache, session)
    if args.open:
        return _main_open(data, stations, args.out or ROOT / "reports" / "fund_hourly_open_eval.json")
    args.out = args.out or ROOT / "reports" / "fund_hourly_temp_eval.json"
    results = {}
    for city in CITIES:
        d, runs = data[city.series], city_forecasts(city, stations)
        brackets, trades, trader, coverage = decide(city, d["events"], runs, d["candles"])
        nb, nt, _, _ = decide(city, d["events"], runs, d["candles"], use_hrrr=False)
        results[city.series] = {"brackets": brackets, "trades": trades, "trader": trader, "coverage": coverage,
                                "no_hrrr": {"brackets": nb, "trades": nt}}
        print(f"{city.name}: {coverage}, {len(trades)} trades", flush=True)
    out = report(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    p = out["pooled"]
    print(f"verdict {out['verdict']}: {p['trades']} trades, mean return {p['mean_return']} {p['mean_return_ci']}, "
          f"Brier skill {p['brier_skill']} {p['brier_skill_ci']}")
    return 0


def _main_open(data: dict[str, dict], stations: dict, out_path: Path) -> int:
    results = {}
    for city in CITIES:
        d, runs = data[city.series], city_forecasts(city, stations)
        priced, ladder = ladder_hours(city, d["events"], d["candles"])
        brackets, trades, _, coverage = decide(city, d["events"], runs, d["candles"], at_open=True)
        lb, lt, _, _ = decide(city, d["events"], runs, d["candles"], at_open=True, field="close_dollars")
        results[city.series] = {"priced": priced, "ladder": ladder, "brackets": brackets, "trades": trades,
                                "coverage": coverage, "minute_close": {"brackets": lb, "trades": lt}}
        print(f"{city.name}: {priced} hours priced at the open, {len(ladder)} mismatches, {coverage}, "
              f"{len(trades)} trades", flush=True)
    out = open_report(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    t1, t2 = out["test1_ladder_mismatches"], out["test2_model_at_open"]
    print(f"verdict {out['verdict']}: {t1['hours_with_mismatch']} of {t1['hours_priced_at_open']} hours with a mismatch; "
          f"model {t2['trades']} trades, mean return {t2['mean_return']} {t2['mean_return_ci']}, "
          f"Brier skill {t2['brier_skill']} {t2['brier_skill_ci']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
