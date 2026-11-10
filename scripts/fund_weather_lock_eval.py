#!/usr/bin/env python3
"""The pre-registered decided-ranges evaluation (docs/fund.md, "Decided-ranges evaluation").

A day's high can only rise, so once the airport thermometer has passed a Kalshi range by a
degree, that range has lost (and an "or above" range below the reading has won). This buys
the side that has already won at the fund's 20:00 UTC cycle, on every settled day of the 24
cities, and asks whether that made money after Kalshi's fee, given that settlement uses The
Weather Company's reading rather than the airport's. Writes reports/fund_weather_lock_eval.json.

Readings: every routine and special METAR at the settlement station, from the Iowa
Environmental Mesonet archive. Prices: Kalshi's hourly event candlesticks. Raw downloads are
kept under --cache.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import fund_weather as fw  # noqa: E402

IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DECISION_HOUR = 20
HOURS = range(18, 24)
MARGIN_F = 1.0
MAX_PRICE = 0.99
CANDLE_MAX_AGE_S = 3 * 3600
MIN_TRADES = 100
BOOTSTRAP, SEED = 2000, 20261110


def _cached(path: Path, fetch: Callable[[], Any], *, text: bool = False) -> Any:
    if path.is_file():
        return path.read_text() if text else json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if text else json.dumps(data))
    return data


def readings(csv_text: str) -> list[tuple[datetime, float]]:
    """(UTC time, degrees F) for every METAR with a temperature."""
    out = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        try:
            out.append((datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc),
                        float(row["tmpf"])))
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(out)


def running_high(obs: list[tuple[datetime, float]], day: date, tz: str, at: datetime) -> Optional[float]:
    """The largest reading from local midnight of ``day`` up to ``at``."""
    start = datetime.combine(day, time(0), ZoneInfo(tz))
    values = [t for when, t in obs if start <= when <= at]
    return max(values) if values else None


def decided(bounds: tuple[float, float], high_so_far: float, margin: float = MARGIN_F) -> Optional[str]:
    """The side a range has already won by, with ``margin`` for the airport-settlement gap."""
    lo, hi = bounds
    if hi != float("inf") and high_so_far >= hi + margin:
        return "NO"
    if hi == float("inf") and lo != float("-inf") and high_so_far >= lo + margin:
        return "YES"
    return None


def trade(side: str, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    """The price of the decided side, if it leaves a gain after Kalshi's fee."""
    price = (1.0 - bid) if side == "NO" and bid is not None and bid > 0 else (ask if side == "YES" else None)
    if price is None or not 0 < price <= MAX_PRICE or 1.0 - price - fw.kalshi_fee(price) <= 0:
        return None
    return price


def quote_at(rows: list, at: datetime) -> tuple[Optional[float], Optional[float]]:
    ts = at.timestamp()
    usable = [r for r in rows or [] if ts - CANDLE_MAX_AGE_S <= r["end_period_ts"] <= ts]
    if not usable:
        return None, None
    last = max(usable, key=lambda r: r["end_period_ts"])
    return (fw._price((last.get("yes_bid") or {}).get("close_dollars")),
            fw._price((last.get("yes_ask") or {}).get("close_dollars")))


def collect(city: fw.City, cache: Path, session: requests.Session, through: date) -> dict:
    events = _cached(cache / city.series / "events.json", lambda: fw.events(city.series, "settled", http_get=session.get))
    events = [ev for ev in events if (d := fw.event_day(ev.get("event_ticker", ""))) and
              fw.SETTLES_ON_TWC_FROM <= d <= through]
    start, end = fw.SETTLES_ON_TWC_FROM - timedelta(days=1), through + timedelta(days=2)
    params = [("station", city.station[1:]), ("data", "tmpf"), ("tz", "Etc/UTC"), ("format", "onlycomma"),
              ("latlon", "no"), ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3"),
              ("report_type", "4"), ("year1", start.year), ("month1", start.month), ("day1", start.day),
              ("year2", end.year), ("month2", end.month), ("day2", end.day)]

    def fetch_metars() -> str:
        resp = session.get(IEM_ASOS, params=params, timeout=120, headers={"User-Agent": fw.USER_AGENT})
        resp.raise_for_status()
        return resp.text
    obs = readings(_cached(cache / city.series / "metars.csv", fetch_metars, text=True))
    candles = {}
    for ev in events:
        ticker = ev["event_ticker"]
        day = fw.event_day(ticker)
        lo = int(datetime(day.year, day.month, day.day, 16, tzinfo=timezone.utc).timestamp())
        body = _cached(cache / city.series / "candles_pm" / f"{ticker}.json", lambda ticker=ticker, lo=lo: fw._get(
            f"{fw.KALSHI_API}/series/{city.series}/events/{ticker}/candlesticks",
            {"start_ts": lo, "end_ts": lo + 14 * 3600, "period_interval": 60}, http_get=session.get))
        candles[ticker] = dict(zip(body.get("market_tickers") or [], body.get("market_candlesticks") or []))
    return {"events": events, "obs": obs, "candles": candles}


def trades_at(city: fw.City, data: dict, hour: int) -> list[dict]:
    out = []
    for ev in data["events"]:
        day = fw.event_day(ev["event_ticker"])
        at = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
        if at >= datetime.combine(day + timedelta(days=1), time(0), ZoneInfo(city.tz)):
            continue  # the local day is over
        high = running_high(data["obs"], day, city.tz, at)
        if high is None:
            continue
        for m in ev.get("markets") or []:
            bounds = fw.bracket_bounds(m)
            side = decided(bounds, high) if bounds else None
            if side is None or m.get("result") not in ("yes", "no"):
                continue
            bid, ask = quote_at(data["candles"].get(ev["event_ticker"], {}).get(m["ticker"]), at)
            price = trade(side, bid, ask)
            if price is None:
                continue
            won = (m["result"] == "yes") == (side == "YES")
            out.append({"city": city.series, "day": day.isoformat(), "hour": hour, "ticker": m["ticker"],
                        "side": side, "price": price, "high_so_far": high, "settled": fw._price(m.get("expiration_value")),
                        "won": won, "return": ((1.0 if won else 0.0) - price - fw.kalshi_fee(price)) / price})
    return out


def mean_return(trades: list[dict]) -> Optional[float]:
    return fmean(t["return"] for t in trades) if trades else None


def bootstrap(trades: list[dict]) -> Optional[list[float]]:
    by_day: dict[str, list[dict]] = {}
    for t in trades:
        by_day.setdefault(t["day"], []).append(t)
    days = sorted(by_day)
    if len(days) < 5:
        return None
    rng = random.Random(SEED)
    draws = sorted(mean_return([t for d in rng.choices(days, k=len(days)) for t in by_day[d]]) for _ in range(BOOTSTRAP))
    return [round(draws[int(0.025 * BOOTSTRAP)], 6), round(draws[int(0.975 * BOOTSTRAP) - 1], 6)]


def summary(trades: list[dict]) -> dict:
    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t["day"]] = by_day.get(t["day"], 0.0) + t["return"]
    losses = [t for t in trades if not t["won"]]
    ret = mean_return(trades)
    return {"trades": len(trades), "days": len(by_day), "losses": len(losses),
            "mean_return": None if ret is None else round(ret, 6), "mean_return_ci": bootstrap(trades),
            "mean_price": round(fmean(t["price"] for t in trades), 4) if trades else None,
            "worst_day": min(by_day.items(), key=lambda kv: kv[1]) if by_day else None,
            "loss_examples": [{k: t[k] for k in ("ticker", "side", "price", "high_so_far", "settled")} for t in losses[:10]]}


def report(primary: list[dict], by_hour: dict[int, list[dict]], *, through: date) -> dict:
    pooled = summary(primary)
    ci = pooled["mean_return_ci"]
    verdict = "GO" if pooled["trades"] >= MIN_TRADES and ci and ci[0] > 0 else "NO-GO"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "verdict": verdict,
        "rule": "docs/fund.md, Decided-ranges evaluation (pre-registered 2026-11-10)",
        "settled_through": through.isoformat(), "decision": f"{DECISION_HOUR:02d}:00 UTC", "margin_f": MARGIN_F,
        "sources": {"prices": "Kalshi public API (hourly event candlesticks)",
                    "readings": "Iowa Environmental Mesonet ASOS archive (routine and special METARs)",
                    "settlement": "The Weather Company, via each event's expiration value"},
        "pooled": pooled,
        "by_side": {s: summary([t for t in primary if t["side"] == s]) for s in ("NO", "YES")},
        "by_city": {c.series: {"city": c.name, **summary([t for t in primary if t["city"] == c.series])}
                    for c in fw.CITIES},
        "by_hour_utc (description only)": {f"{h:02d}": summary(ts) for h, ts in by_hour.items()},
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "weather_eval")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_weather_lock_eval.json")
    ap.add_argument("--through", type=date.fromisoformat, default=None)
    args = ap.parse_args(argv)
    through = args.through or (datetime.now(timezone.utc).date() - timedelta(days=1))
    session = requests.Session()
    by_hour: dict[int, list[dict]] = {h: [] for h in HOURS}
    for city in fw.CITIES:
        data = collect(city, args.cache, session, through)
        for h in HOURS:
            by_hour[h] += trades_at(city, data, h)
        print(f"{city.name}: {len(data['events'])} days, {len(data['obs'])} readings", flush=True)
    out = report(by_hour[DECISION_HOUR], by_hour, through=through)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    p = out["pooled"]
    print(f"verdict {out['verdict']}: {p['trades']} trades, {p['losses']} lost, mean return {p['mean_return']} "
          f"{p['mean_return_ci']}, mean price {p['mean_price']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
