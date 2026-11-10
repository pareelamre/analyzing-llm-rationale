#!/usr/bin/env python3
"""The pre-registered weather evaluation (docs/fund.md, "Weather evaluation").

Downloads every settled Kalshi daily-high market that settles on The Weather Company, the
eight models' past forecasts for those days and the markets' hourly prices; prices each
bracket at the fund's two decision points with fund_weather, applies the fund's rule, and
writes reports/fund_weather_eval.json with the verdict. Raw downloads are kept under --cache,
so a re-run reads them instead of asking the APIs again.

Decision points (forecasts no newer than a real cycle would have had):
- A: 08:00 UTC on day D, that day's markets, previous_day1 forecasts, highs known to D-1;
- B: 20:00 UTC on day D-1, the next day's markets, previous_day2 forecasts, highs known to D-2.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import fund_weather as fw  # noqa: E402

DECISIONS = {"A": {"lead": 1, "day_offset": 0, "hour": 8, "known_lag": 1},
             "B": {"lead": 2, "day_offset": -1, "hour": 20, "known_lag": 2}}
CANDLE_MAX_AGE_S = 3 * 3600
BOOTSTRAP, SEED = 2000, 20261110
REQUEST_GAP_S = 0.06


def _cached(path: Path, fetch: Callable[[], Any]) -> Any:
    if path.is_file():
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _get(session: requests.Session, url: str, params: dict) -> Any:
    for attempt in range(4):
        resp = session.get(url, params=params, timeout=60, headers={"User-Agent": fw.USER_AGENT})
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_GAP_S)
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def collect(city: fw.City, cache: Path, session: requests.Session, through: date) -> dict:
    """One city's settled events, past forecasts at both leads, and hourly prices per event."""
    settled = _cached(cache / city.series / "events.json",
                      lambda: fw.events(city.series, "settled", http_get=session.get))
    settled = [ev for ev in settled if (d := fw.event_day(ev.get("event_ticker", ""))) and
               fw.SETTLES_ON_TWC_FROM <= d <= through]
    start = fw.SETTLES_ON_TWC_FROM - timedelta(days=2)
    raw = _cached(cache / city.series / "forecasts.json", lambda: _get(session, fw.PREVIOUS_RUNS_API, {
        "latitude": city.lat, "longitude": city.lon, "models": ",".join(fw.MODELS),
        "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2", "temperature_unit": "fahrenheit",
        "timezone": city.tz, "start_date": start.isoformat(), "end_date": through.isoformat()}))
    forecasts = {lead: fw.daily_highs(raw, f"temperature_2m_previous_day{lead}") for lead in (1, 2)}
    candles = {}
    for ev in settled:
        ticker = ev["event_ticker"]
        day = fw.event_day(ticker)
        lo = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) - 6 * 3600
        hi = lo + 6 * 3600 + 10 * 3600
        body = _cached(cache / city.series / "candles" / f"{ticker}.json", lambda ticker=ticker, lo=lo, hi=hi: _get(
            session, f"{fw.KALSHI_API}/series/{city.series}/events/{ticker}/candlesticks",
            {"start_ts": lo, "end_ts": hi, "period_interval": 60}))
        candles[ticker] = {t: rows for t, rows in zip(body.get("market_tickers") or [],
                                                      body.get("market_candlesticks") or [])}
    return {"events": settled, "forecasts": forecasts, "candles": candles}


def quote_at(rows: list, at: datetime) -> tuple[Optional[float], Optional[float]]:
    ts = at.timestamp()
    usable = [r for r in rows or [] if ts - CANDLE_MAX_AGE_S <= r["end_period_ts"] <= ts]
    if not usable:
        return None, None
    last = max(usable, key=lambda r: r["end_period_ts"])
    bid = fw._price((last.get("yes_bid") or {}).get("close_dollars"))
    ask = fw._price((last.get("yes_ask") or {}).get("close_dollars"))
    return bid, ask


def decisions(data: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Every priced bracket and every trade the rule makes, at both decision points."""
    rows = {lead: [r for city in fw.CITIES if city.series in data
                   for r in fw.calibration_rows(city, data[city.series]["forecasts"][lead],
                                                fw.observed_highs(data[city.series]["events"]))]
            for lead in (1, 2)}
    fits: dict[tuple[int, date], Optional[fw.Calibration]] = {}
    brackets, trades = [], []
    for city in fw.CITIES:
        if city.series not in data:
            continue
        d = data[city.series]
        for ev in d["events"]:
            day = fw.event_day(ev["event_ticker"])
            for name, spec in DECISIONS.items():
                lead = spec["lead"]
                known = day - timedelta(days=spec["known_lag"])
                if (lead, known) not in fits:
                    fits[(lead, known)] = fw.fit(rows[lead], known)
                cal = fits[(lead, known)]
                e = fw.ensemble(d["forecasts"][lead], day)
                if cal is None or e is None:
                    continue
                at = datetime(day.year, day.month, day.day, spec["hour"], tzinfo=timezone.utc) + \
                    timedelta(days=spec["day_offset"])
                centre, sigma = e[0] + cal.bias.get(city.series, 0.0), cal.sigma(e[1])
                for m in ev.get("markets") or []:
                    bounds = fw.bracket_bounds(m)
                    if bounds is None or m.get("result") not in ("yes", "no"):
                        continue
                    p = fw.chance(centre, sigma, bounds)
                    bid, ask = quote_at(d["candles"].get(ev["event_ticker"], {}).get(m["ticker"]), at)
                    yes = m["result"] == "yes"
                    brackets.append({"decision": name, "city": city.series, "day": day.isoformat(),
                                     "ticker": m["ticker"], "p": p, "bid": bid, "ask": ask, "yes": yes})
                    hit = fw.rule(p, bid, ask)
                    if hit:
                        side, price, edge = hit
                        payoff = 1.0 if (side == "YES") == yes else 0.0
                        trades.append({"decision": name, "city": city.series, "day": day.isoformat(),
                                       "ticker": m["ticker"], "side": side, "price": price, "edge": edge,
                                       "return": (payoff - price - fw.kalshi_fee(price)) / price})
    return brackets, trades


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
    """95% interval from resampling settlement days (a day's brackets move together)."""
    by_day: dict[str, list[dict]] = {}
    for it in items:
        by_day.setdefault(it["day"], []).append(it)
    days = sorted(by_day)
    if len(days) < 5:
        return None
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        sample = [it for d in rng.choices(days, k=len(days)) for it in by_day[d]]
        v = stat(sample)
        if v is not None:
            draws.append(v)
    draws.sort()
    return [round(draws[int(0.025 * len(draws))], 5), round(draws[int(0.975 * len(draws)) - 1], 5)]


def summary(brackets: list[dict], trades: list[dict]) -> dict:
    ret = mean_return(trades)
    skill = brier_skill(brackets)
    return {"brackets": len(brackets), "two_sided": sum(_two_sided(b) for b in brackets),
            "brier_skill": None if skill is None else round(skill, 5), "brier_skill_ci": bootstrap(brackets, brier_skill),
            "trades": len(trades), "days_traded": len({t["day"] for t in trades}),
            "mean_return": None if ret is None else round(ret, 5), "mean_return_ci": bootstrap(trades, mean_return),
            "win_rate": round(fmean((t["return"] > 0) for t in trades), 4) if trades else None}


def reliability(brackets: list[dict]) -> list[dict]:
    bins = []
    for lo in (0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95):
        hi = {0.0: 0.05, 0.05: 0.15, 0.15: 0.3, 0.3: 0.5, 0.5: 0.7, 0.7: 0.85, 0.85: 0.95, 0.95: 1.01}[lo]
        rows = [b for b in brackets if lo <= b["p"] < hi]
        if rows:
            bins.append({"model": f"{lo:.2f}-{min(hi, 1):.2f}", "n": len(rows),
                         "mean_forecast": round(fmean(b["p"] for b in rows), 4),
                         "observed": round(fmean(b["yes"] for b in rows), 4)})
    return bins


def report(brackets: list[dict], trades: list[dict], *, through: date) -> dict:
    pooled = summary(brackets, trades)
    ci = pooled["mean_return_ci"]
    verdict = "GO" if ci and ci[0] > 0 else "SHADOW"
    days = sorted({b["day"] for b in brackets})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "verdict": verdict,
        "rule": "docs/fund.md, Weather evaluation (pre-registered 2026-11-10)",
        "window": {"first": days[0] if days else None, "last": days[-1] if days else None, "settled_through": through.isoformat()},
        "sources": {"markets": "Kalshi public API (events, hourly event candlesticks)",
                    "forecasts": f"Open-Meteo Previous Runs API: {', '.join(fw.MODELS)}",
                    "settlement": "The Weather Company, via each event's expiration value"},
        "pooled": pooled,
        "by_decision": {k: summary([b for b in brackets if b["decision"] == k], [t for t in trades if t["decision"] == k])
                        for k in DECISIONS},
        "by_city": {c.series: {"city": c.name, **summary([b for b in brackets if b["city"] == c.series],
                                                         [t for t in trades if t["city"] == c.series])}
                    for c in fw.CITIES},
        "reliability": reliability(brackets),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "weather_eval")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_weather_eval.json")
    ap.add_argument("--through", type=date.fromisoformat, default=None, help="last settled day (default: yesterday UTC)")
    args = ap.parse_args(argv)
    through = args.through or (datetime.now(timezone.utc).date() - timedelta(days=1))
    session = requests.Session()
    data = {}
    for city in fw.CITIES:
        try:
            data[city.series] = collect(city, args.cache, session, through)
            print(f"{city.name}: {len(data[city.series]['events'])} settled days", flush=True)
        except (requests.RequestException, ValueError) as exc:
            print(f"{city.name}: skipped ({exc})", flush=True)
    brackets, trades = decisions(data)
    out = report(brackets, trades, through=through)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    p = out["pooled"]
    print(f"verdict {out['verdict']}: {p['trades']} trades, mean return {p['mean_return']} {p['mean_return_ci']}, "
          f"Brier skill {p['brier_skill']} {p['brier_skill_ci']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
