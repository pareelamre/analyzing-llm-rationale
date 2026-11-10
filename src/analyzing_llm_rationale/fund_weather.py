"""Kalshi daily high temperatures for the Foresea Fund (docs/fund.md, "Weather evaluation").

The weather analyst prices each bracket of Kalshi's daily high-temperature markets (24 US
cities, settled on The Weather Company's reading since 2026-09-02) from eight weather models:

- each model's high is its largest hourly 2 m temperature over the city's local day, at the
  settlement station's NWS coordinates (Open-Meteo, no account);
- the centre is the models' mean plus the city's recent bias (observed minus forecast, over
  its last BIAS_DAYS settled days);
- the spread grows with the models' disagreement: sigma squared = c + d x (their spread)
  squared, fitted on every city's last SPREAD_DAYS days;
- the high is Normal(centre, sigma), and a bracket's chance counts whole degrees, as the
  settlement does: "61 to 62" pays from 60.5 to 62.5.

The trading rule is the fund's event rule with Kalshi's own fee, 0.07 x price x (1 - price)
per contract. scripts/fund_weather_eval.py tests the same code on every settled market before
it may trade; the verdict decides whether the analyst trades or records in shadow mode.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from statistics import NormalDist, fmean
from time import sleep as time_sleep
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import requests

from . import fund_orchestrator as orch

USER_AGENT = "Foresea/fund-weather (+https://foresea.ink)"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
#: Kalshi moved these contracts to The Weather Company's reading on 2026-09-02.
SETTLES_ON_TWC_FROM = date(2026, 9, 3)


@dataclass(frozen=True)
class City:
    series: str
    name: str
    cli: str  # the climate report the contract names
    station: str
    lat: float
    lon: float
    tz: str


#: Every daily high series that settles on The Weather Company, with the NWS coordinates and
#: time zone of the station its contract names (api.weather.gov/stations, 2026-11-10).
CITIES: tuple[City, ...] = tuple(City(*row) for row in (
    ("KXHIGHNY", "New York City", "CLINYC", "KNYC", 40.7833, -73.9667, "America/New_York"),
    ("KXHIGHLAX", "Los Angeles", "CLILAX", "KLAX", 33.9381, -118.3889, "America/Los_Angeles"),
    ("KXHIGHMIA", "Miami", "CLIMIA", "KMIA", 25.7906, -80.3164, "America/New_York"),
    ("KXHIGHCHI", "Chicago", "CLIMDW", "KMDW", 41.7842, -87.7553, "America/Chicago"),
    ("KXHIGHAUS", "Austin", "CLIAUS", "KAUS", 30.1830, -97.6799, "America/Chicago"),
    ("KXHIGHPHIL", "Philadelphia", "CLIPHL", "KPHL", 39.8733, -75.2268, "America/New_York"),
    ("KXHIGHTSFO", "San Francisco", "CLISFO", "KSFO", 37.6196, -122.3656, "America/Los_Angeles"),
    ("KXHIGHDEN", "Denver", "CLIDEN", "KDEN", 39.8466, -104.6562, "America/Denver"),
    ("KXHIGHTPHX", "Phoenix", "CLIPHX", "KPHX", 33.4278, -112.0035, "America/Phoenix"),
    ("KXHIGHTATL", "Atlanta", "CLIATL", "KATL", 33.6403, -84.4269, "America/New_York"),
    ("KXHIGHTDAL", "Dallas", "CLIDFW", "KDFW", 32.8974, -97.0220, "America/Chicago"),
    ("KXHIGHTMIN", "Minneapolis", "CLIMSP", "KMSP", 44.8831, -93.2289, "America/Chicago"),
    ("KXHIGHTSEA", "Seattle", "CLISEA", "KSEA", 47.4447, -122.3136, "America/Los_Angeles"),
    ("KXHIGHTHOU", "Houston", "CLIHOU", "KHOU", 29.6375, -95.2825, "America/Chicago"),
    ("KXHIGHTBOS", "Boston", "CLIBOS", "KBOS", 42.3606, -71.0106, "America/New_York"),
    ("KXHIGHTDC", "Washington DC", "CLIDCA", "KDCA", 38.8483, -77.0342, "America/New_York"),
    ("KXHIGHTOKC", "Oklahoma City", "CLIOKC", "KOKC", 35.3886, -97.6003, "America/Chicago"),
    ("KXHIGHTLV", "Las Vegas", "CLILAS", "KLAS", 36.0719, -115.1634, "America/Los_Angeles"),
    ("KXHIGHTSATX", "San Antonio", "CLISAT", "KSAT", 29.5328, -98.4636, "America/Chicago"),
    ("KXHIGHTNOLA", "New Orleans", "CLIMSY", "KMSY", 29.9928, -90.2508, "America/Chicago"),
    ("KXHIGHTSAN", "San Diego", "CLISAN", "KSAN", 32.7336, -117.1831, "America/Los_Angeles"),
    ("KXHIGHTSDF", "Louisville", "CLISDF", "KSDF", 38.1741, -85.7365, "America/Kentucky/Louisville"),
    ("KXHIGHTEWR", "Newark", "CLIEWR", "KEWR", 40.6825, -74.1694, "America/New_York"),
    ("KXHIGHTTTN", "Trenton", "CLITTN", "KTTN", 40.2764, -74.8164, "America/New_York"),
))
BY_SERIES = {c.series: c for c in CITIES}

MODELS = ("gfs_seamless", "ecmwf_ifs025", "ecmwf_aifs025_single", "icon_seamless", "gem_seamless",
          "jma_seamless", "ukmo_seamless", "meteofrance_seamless")
#: A model's day counts only with most of its hours.
MIN_HOURS = 20
BIAS_DAYS, MIN_BIAS_DAYS = 30, 7
SPREAD_DAYS, MIN_SPREAD_DAYS = 45, 21
MIN_SIGMA = 1.0
KALSHI_FEE_RATE = 0.07
#: A day's market is priced from forecasts only before its morning: later, what the
#: thermometer has already read matters more than any model.
LAST_ENTRY_LOCAL = time(9, 0)
#: All of one city-day's brackets are one bet on one number, so together they get at most
#: this share of the fund.
EVENT_CAP = 0.02

HttpGet = Callable[..., Any]


#: Kalshi and Open-Meteo answer a burst with 429; a short wait clears it.
RETRIES, RETRY_WAIT_S = 4, 3.0


def _get(url: str, params: Mapping[str, Any], *, http_get: HttpGet, sleep: Callable[[float], None] = time_sleep) -> Any:
    for attempt in range(RETRIES):
        resp = http_get(url, params=dict(params), timeout=60, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        if getattr(resp, "status_code", 200) in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
            sleep(RETRY_WAIT_S * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise requests.RequestException(f"{url} kept refusing")


# Forecasts ------------------------------------------------------------------------------------

def daily_highs(payload: Mapping[str, Any], variable: str = "temperature_2m") -> dict[str, dict[date, float]]:
    """Each model's high per local day from an Open-Meteo multi-model hourly payload (requested
    with the city's time zone, so its times are local)."""
    hourly = payload.get("hourly") or {}
    times = [str(t)[:10] for t in hourly.get("time") or []]
    out: dict[str, dict[date, float]] = {}
    for model in MODELS:
        values = hourly.get(f"{variable}_{model}")
        if not values:
            continue
        by_day: dict[str, list[float]] = {}
        for day, v in zip(times, values):
            if v is not None:
                by_day.setdefault(day, []).append(float(v))
        out[model] = {date.fromisoformat(d): max(vs) for d, vs in by_day.items() if len(vs) >= MIN_HOURS}
    return out


def ensemble(highs: Mapping[str, Mapping[date, float]], day: date) -> Optional[tuple[float, float, int]]:
    """The models' mean and spread (standard deviation) for ``day``, and how many had it."""
    values = [h[day] for h in highs.values() if day in h]
    if len(values) < 3:
        return None
    mean = fmean(values)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)), len(values)


def live_forecast(city: City, *, http_get: HttpGet = requests.get) -> dict[str, dict[date, float]]:
    params = {"latitude": city.lat, "longitude": city.lon, "hourly": "temperature_2m", "models": ",".join(MODELS),
              "temperature_unit": "fahrenheit", "timezone": city.tz, "forecast_days": 3}
    return daily_highs(_get(FORECAST_API, params, http_get=http_get))


def past_forecasts(city: City, start: date, end: date, lead_days: int = 1, *,
                   http_get: HttpGet = requests.get) -> dict[str, dict[date, float]]:
    """What each model forecast for each day, from its run ``lead_days`` earlier."""
    variable = f"temperature_2m_previous_day{lead_days}"
    params = {"latitude": city.lat, "longitude": city.lon, "hourly": variable, "models": ",".join(MODELS),
              "temperature_unit": "fahrenheit", "timezone": city.tz, "start_date": start.isoformat(),
              "end_date": end.isoformat()}
    return daily_highs(_get(PREVIOUS_RUNS_API, params, http_get=http_get), variable)


# Markets ---------------------------------------------------------------------------------------

_EVENT_DAY = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})$")
_MONTHS = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                       "NOV", "DEC"), start=1)}


def event_day(event_ticker: str) -> Optional[date]:
    """KXHIGHNY-26NOV08 -> 2026-11-08."""
    m = _EVENT_DAY.search(event_ticker)
    if not m or m.group(2) not in _MONTHS:
        return None
    return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


def bracket_bounds(market: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    """The high temperatures a market pays Yes on, counting whole degrees as the settlement
    does: less than 61 is below 60.5, greater than 68 is 68.5 and up, 61 to 62 is 60.5-62.5."""
    kind, floor, cap = market.get("strike_type"), market.get("floor_strike"), market.get("cap_strike")
    try:
        if kind == "less" and cap is not None:
            return -math.inf, float(cap) - 0.5
        if kind == "greater" and floor is not None:
            return float(floor) + 0.5, math.inf
        if kind == "between" and floor is not None and cap is not None:
            return float(floor) - 0.5, float(cap) + 0.5
    except (TypeError, ValueError):
        return None
    return None


def _price(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def events(series: str, status: str, *, http_get: HttpGet = requests.get, pages: int = 20) -> list[dict]:
    """A series' events with their markets (status "open" or "settled"), newest first."""
    out, cursor = [], None
    for _ in range(pages):
        params = {"series_ticker": series, "status": status, "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        body = _get(f"{KALSHI_API}/events", params, http_get=http_get)
        out.extend(body.get("events") or [])
        cursor = body.get("cursor")
        if not cursor or not body.get("events"):
            break
    return out


def observed_highs(settled: Iterable[Mapping[str, Any]]) -> dict[date, float]:
    """The settlement's high per day, from settled events' expiration values."""
    out = {}
    for ev in settled:
        day = event_day(str(ev.get("event_ticker") or ""))
        values = {_price(m.get("expiration_value")) for m in ev.get("markets") or []} - {None}
        if day and day >= SETTLES_ON_TWC_FROM and len(values) == 1:
            out[day] = values.pop()
    return out


# Calibration ----------------------------------------------------------------------------------

@dataclass
class Row:
    city: str
    day: date
    mean: float
    spread: float
    observed: float


def calibration_rows(city: City, highs: Mapping[str, Mapping[date, float]], observed: Mapping[date, float]) -> list[Row]:
    rows = []
    for day, obs in sorted(observed.items()):
        e = ensemble(highs, day)
        if e is not None:
            rows.append(Row(city.series, day, e[0], e[1], obs))
    return rows


@dataclass
class Calibration:
    bias: dict[str, float]
    c: float
    d: float
    days: int
    known_through: date

    def sigma(self, spread: float) -> float:
        return max(MIN_SIGMA, math.sqrt(max(self.c + self.d * spread * spread, 0.0)))


def fit(rows: Sequence[Row], known_through: date) -> Optional[Calibration]:
    """Each city's bias over its last BIAS_DAYS known days, and the spread fit over every city's
    last SPREAD_DAYS; None with fewer than MIN_SPREAD_DAYS days of history."""
    known = [r for r in rows if r.day <= known_through]
    bias: dict[str, float] = {}
    for city in {r.city for r in known}:
        recent = sorted((r for r in known if r.city == city), key=lambda r: r.day)[-BIAS_DAYS:]
        bias[city] = fmean(r.observed - r.mean for r in recent) if len(recent) >= MIN_BIAS_DAYS else 0.0
    window = [r for r in known if r.day > known_through - timedelta(days=SPREAD_DAYS)]
    if len({r.day for r in window}) < MIN_SPREAD_DAYS:
        return None
    x = [r.spread ** 2 for r in window]
    y = [(r.observed - r.mean - bias.get(r.city, 0.0)) ** 2 for r in window]
    mx, my = fmean(x), fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    d = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx if sxx > 0 else 0.0
    c = my - d * mx
    if d < 0 or c < 0:  # a fit that shrinks with disagreement is noise: fall back to one spread
        c, d = my, 0.0
    return Calibration(bias, c, d, len({r.day for r in window}), known_through)


def chance(centre: float, sigma: float, bounds: tuple[float, float]) -> float:
    dist = NormalDist(centre, sigma)
    lo, hi = bounds
    return (dist.cdf(hi) if math.isfinite(hi) else 1.0) - (dist.cdf(lo) if math.isfinite(lo) else 0.0)


# The rule ---------------------------------------------------------------------------------------

def kalshi_fee(price: float) -> float:
    """Kalshi's taker fee per contract (rounding to the cent ignored for paper)."""
    return KALSHI_FEE_RATE * price * (1.0 - price)


def rule(p_yes: float, bid: Optional[float], ask: Optional[float]) -> Optional[tuple[str, float, float]]:
    """The fund's event rule with Kalshi's fee: (side, price, edge per dollar) or None."""
    if bid is None or ask is None or not 0 <= bid <= ask <= 1:
        return None
    if abs(p_yes - (bid + ask) / 2) > orch.MAX_DISAGREEMENT:
        return None
    best = None
    for side, prob, price in (("YES", p_yes, ask), ("NO", 1.0 - p_yes, 1.0 - bid)):
        if not orch.PRICE_BAND[0] <= price <= orch.PRICE_BAND[1] or prob - price < orch.MIN_GAP:
            continue
        edge = (prob - price - kalshi_fee(price)) / price
        if edge >= orch.MIN_EDGE and (best is None or edge > best[2]):
            best = (side, price, edge)
    return best


def tradeable_day(day: date, city: City, at: datetime) -> bool:
    """Forecasts alone price a day until its morning."""
    return at < datetime.combine(day, LAST_ENTRY_LOCAL, ZoneInfo(city.tz))


def _fmt(x: float) -> str:
    return f"{x:.1f}°F"


def weather_signals(now: datetime, *, trust: float = orch.DEFAULT_TRUST, analyst: str = "weather_model",
                    http_get: HttpGet = requests.get, history_days: int = 50) -> dict[str, Any]:
    """The weather analyst's cycle: signals on open markets whose day has not begun, every
    priced bracket (for the shadow record), and the cities skipped, with why."""
    signals: list[orch.Signal] = []
    priced: list[dict] = []
    skipped: list[dict] = []
    today = now.date()
    for city in CITIES:
        try:
            open_events = [ev for ev in events(city.series, "open", http_get=http_get, pages=1)
                           if (d := event_day(str(ev.get("event_ticker") or ""))) and tradeable_day(d, city, now)]
            if not open_events:
                continue
            observed = observed_highs(events(city.series, "settled", http_get=http_get, pages=2))
            history = past_forecasts(city, today - timedelta(days=history_days), today, 1, http_get=http_get)
            forecast = live_forecast(city, http_get=http_get)
        except (requests.RequestException, ValueError, KeyError) as exc:
            skipped.append({"instrument": f"kalshi:{city.series}", "label": f"{city.name} high temperature",
                            "reason": f"no forecast or market data: {type(exc).__name__}"})
            continue
        known = max(observed) if observed else today
        cal = fit(calibration_rows(city, history, observed), known)
        if cal is None:
            skipped.append({"instrument": f"kalshi:{city.series}", "label": f"{city.name} high temperature",
                            "reason": "too little settled history to calibrate"})
            continue
        for ev in open_events:
            day = event_day(ev["event_ticker"])
            e = ensemble(forecast, day)
            if e is None:
                continue
            centre, sigma = e[0] + cal.bias.get(city.series, 0.0), cal.sigma(e[1])
            picks = []
            for m in ev.get("markets") or []:
                bounds = bracket_bounds(m)
                if bounds is None:
                    continue
                p = chance(centre, sigma, bounds)
                bid, ask = _price(m.get("yes_bid_dollars")), _price(m.get("yes_ask_dollars"))
                priced.append({"platform": "kalshi", "ident": m["ticker"], "question": _question(city, day, m),
                               "model_probability": round(p, 4), "market_bid": bid, "market_ask": ask})
                hit = rule(p, bid, ask)
                if hit:
                    picks.append((m, p, hit))
            total = sum(orch.KELLY_FRACTION * trust * _kelly(p, side, price) for _m, p, (side, price, _e) in picks)
            scale = min(1.0, EVENT_CAP / total) if total > 0 else 1.0
            for m, p, (side, price, edge) in picks:
                prob = p if side == "YES" else 1.0 - p
                source = f"{len(MODELS)} weather models via Open-Meteo, {now:%Y-%m-%d %H:%M} UTC"
                signals.append(orch.Signal(
                    analyst=analyst, side=side, price=price, edge=edge, trust=trust,
                    instrument=orch.event_instrument({"platform": "kalshi", "ident": m["ticker"],
                                                      "question": _question(city, day, m)}),
                    kelly=_kelly(p, side, price) * scale,
                    facts=[orch._fact("question", _question(city, day, m), "Kalshi"),
                           orch._fact("forecast high", f"{_fmt(centre)}, give or take {_fmt(sigma)}", source),
                           orch._fact("models agreeing within", _fmt(e[1]), source),
                           orch._fact(f"{city.name} bias corrected", _fmt(cal.bias.get(city.series, 0.0)),
                                      f"last {BIAS_DAYS} settled days"),
                           orch._fact(f"model chance of {side}", orch._pct(prob), "weather analyst"),
                           orch._fact(f"price to buy {side}", orch._cents(price), "Kalshi order book"),
                           orch._fact("expected profit per dollar after Kalshi's fee", orch._pct(edge),
                                      "orchestrator")]))
    return {"signals": signals, "priced": priced, "skipped": skipped}


def _kelly(p_yes: float, side: str, price: float) -> float:
    prob = p_yes if side == "YES" else 1.0 - p_yes
    return max(0.0, (prob - price) / (1.0 - price))


def _question(city: City, day: date, market: Mapping[str, Any]) -> str:
    what = market.get("yes_sub_title") or market.get("subtitle") or market.get("ticker")
    return f"{city.name} high on {day.day} {day:%b}: {what}"
