"""Statistical-arbitrage tracker for ten commodity pairs, from MCX futures and benchmark spot prices.

Each pair's spread is the log price of one leg minus a hedge ratio times the log price of the
other, with the hedge ratio refit every day by least squares on up to 250 sessions. The spread's
z-score against its last 60 sessions says how stretched the pair is. A pair is "stretched" at
|z| >= 2, and the tracker records that as a signal, then follows it until |z| falls below 0.5
(reverted) or 20 sessions pass (timed out).

Prices: MCX front-month futures from Upstox's open candle route, stitched across rolls and
ratio-adjusted so a roll is not read as a move, then stored daily (data/commodity_prices.jsonl)
because expired contracts leave the route. Brent, WTI and Henry Hub gas are daily spot prices
from FRED. Gold, platinum and the rupee are a daily snapshot from the open fawazahmed0 currency
API, also stored. The gold import-parity pair compares MCX gold at 11:00 IST, when that snapshot
is taken, with the snapshot's gold price landed in India (rupees plus import duty).

This is a tracker, not a strategy with an edge: the same rule on 2010-2026 US futures history
mostly lost money or broke even after costs (see BACKTEST). Live signals are recorded only on
the newest session at the time of a run, so the ledger never adds a signal after the fact.
"""
from __future__ import annotations

import gzip
import json
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote

import numpy as np
import requests

#: key, label, first leg, second leg, price source, why traders watch it.
#: international: benchmark USD spot price (FRED). india: MCX front-month futures in INR.
#: parity: MCX gold at 11:00 IST against the import-parity (landed) cost then, hedge fixed at 1.
PAIRS: tuple[dict[str, str], ...] = (
    {"key": "gold-silver", "a": "gold", "b": "silver", "source": "india", "label": "Gold / Silver (MCX)",
     "why": "The gold-silver ratio, the most watched precious-metals spread."},
    {"key": "brent-wti", "a": "brentcrude", "b": "crudeoil", "source": "international", "label": "Brent / WTI",
     "why": "The benchmark crude spread between the North Sea and US grades."},
    {"key": "wti-natgas", "a": "crudeoil", "b": "naturalgas", "source": "international", "label": "WTI / Natural gas",
     "why": "The oil-gas ratio: energy priced as liquid fuel against gas."},
    {"key": "copper-gold", "a": "copper", "b": "gold", "source": "india", "label": "Copper / Gold (MCX)",
     "why": "The copper-gold ratio, read as growth against safety."},
    {"key": "gold-platinum", "a": "gold", "b": "platinum", "source": "international", "label": "Gold / Platinum",
     "why": "Monetary against industrial precious metal."},
    {"key": "copper-aluminium", "a": "copper", "b": "aluminium", "source": "india", "label": "Copper / Aluminium (MCX)",
     "why": "The two largest base metals, substitutes in wiring and cable."},
    {"key": "zinc-lead", "a": "zinc", "b": "lead", "source": "india", "label": "Zinc / Lead (MCX)",
     "why": "Mined together and traded as a classic base-metal pair."},
    {"key": "copper-zinc", "a": "copper", "b": "zinc", "source": "india", "label": "Copper / Zinc (MCX)",
     "why": "Construction and manufacturing demand on both legs."},
    {"key": "nickel-copper", "a": "nickel", "b": "copper", "source": "india", "label": "Nickel / Copper (MCX)",
     "why": "Battery and steel demand against the broad industrial metal."},
    {"key": "gold-parity", "a": "gold", "b": "gold", "source": "parity", "label": "MCX Gold / import parity",
     "why": "India's gold price against the landed cost of imported gold: the domestic premium."},
)
HEDGE_WINDOW = 250
Z_WINDOW = 60
MIN_SESSIONS = 80
FULL_HISTORY = HEDGE_WINDOW + Z_WINDOW
ENTRY_Z = 2.0
EXIT_Z = 0.5
MAX_HOLD = 20
COST_PER_SIDE = 0.001
#: The first session the live tracker evaluated. Signals before it come only from the replay.
LIVE_SINCE = date(2026, 11, 6)

#: This module's rule walked over 2010-2026 daily US futures closes (Yahoo Finance continuous
#: front months, so roll gaps add noise) for the pairs with a US contract on both legs, 0.1%
#: cost per entry and exit. Return is the spread move per unit of gross exposure.
BACKTEST: dict[str, Any] = {
    "window": "2011-2026",
    "pairs": {
        "gold-silver": {"signals": 77, "reverted": 0.494, "avg_return": -0.00727, "total_return": -0.5595, "win_rate": 0.506, "years_up": 5, "years": 16},
        "brent-wti": {"signals": 99, "reverted": 0.697, "avg_return": 0.00352, "total_return": 0.3481, "win_rate": 0.697, "years_up": 9, "years": 16},
        "wti-natgas": {"signals": 92, "reverted": 0.489, "avg_return": 0.00752, "total_return": 0.692, "win_rate": 0.543, "years_up": 10, "years": 16},
        "copper-gold": {"signals": 82, "reverted": 0.5, "avg_return": -0.00595, "total_return": -0.4875, "win_rate": 0.561, "years_up": 7, "years": 16},
        "gold-platinum": {"signals": 89, "reverted": 0.494, "avg_return": -0.00273, "total_return": -0.2429, "win_rate": 0.551, "years_up": 5, "years": 16},
        "copper-aluminium": {"signals": 62, "reverted": 0.532, "avg_return": 0.00302, "total_return": 0.1874, "win_rate": 0.629, "years_up": 7, "years": 12},
    },
}

HttpGet = Callable[..., Any]
IST = timezone(timedelta(hours=5, minutes=30))
USER_AGENT = "Foresea/commodity-pairs (+https://foresea.ink)"

#: Upstox's candle route and instrument list are open data: no account or token.
UPSTOX_BASE_URL = "https://api.upstox.com"
MCX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz"
#: Commodity and its full-size MCX futures symbol.
MCX_SYMBOLS = {"gold": "GOLD", "silver": "SILVER", "copper": "COPPER", "aluminium": "ALUMINIUM", "zinc": "ZINC", "lead": "LEAD", "nickel": "NICKEL"}
#: MCX quotes gold per 10 g and the other metals per kg. Gold is put per kg so the gold-silver
#: ratio reads in its usual form.
MCX_PER_KG = {"gold": 100.0}
#: The series moves to the next contract this many calendar days before the front one expires.
ROLL_DAYS = 7
#: How far back the first run looks; contracts listed now rarely go further.
MCX_BACKFILL_DAYS = 400
#: Commodities also stored at 11:00 IST ("india_am"): the close of the 30-minute candle from
#: 10:30, when the free gold snapshot below is taken. Matching the two cut the parity premium's
#: day-to-day noise from 1.17% (MCX close against the next snapshot) to 0.58% over 60 sessions.
MORNING_COMMODITIES = ("gold",)
MORNING_CANDLE = "10:30"
#: Upstox serves at most a quarter of 30-minute candles per request.
MORNING_CHUNK_DAYS = 90
#: Daily spot prices on FRED (from the US EIA); the CSV download needs no key.
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES = {"brentcrude": "DCOILBRENTEU", "crudeoil": "DCOILWTICO", "naturalgas": "DHHNGSP"}
#: Enough FRED history for a full hedge window plus the z-score window.
FRED_DAYS = 600
#: Gold and platinum in USD and the rupee from the fawazahmed0 currency API (CC0, no key or rate
#: limit), one snapshot a day at about 05:30 UTC. Its gold-platinum score tracked one from US
#: futures closes at 0.94 correlation over 2025-26. Cloudflare Pages is its fallback host.
METALS_URLS = (
    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{day}/v1/currencies/usd.json",
    "https://{day}.currency-api.pages.dev/v1/currencies/usd.json",
)
#: Stored commodity and its code in the snapshot, which quotes units per USD: the metals are
#: inverted to USD per troy ounce, the rupee kept as rupees per USD.
SNAPSHOT_CODES = {"gold": ("xau", True), "platinum": ("xpt", True), "usdinr": ("inr", False)}
#: How far back the first run looks: a full hedge window plus the z-score window.
METALS_BACKFILL_DAYS = 450
TROY_OUNCE_G = 31.1034768
#: India's import duty on gold (basic customs duty plus AIDC) from each date: cut from 15% to 6%
#: on 23 July 2024, raised back to 15% on 13 May 2026.
GOLD_IMPORT_DUTY = ((date(2024, 7, 23), 0.06), (date(2026, 5, 13), 0.15))
DUTY_BEFORE = 0.15
#: The (commodity, source) series some feed supplies. A pair needing any other is paused.
FEEDS = frozenset(
    {(c, "international") for c in (*FRED_SERIES, "gold", "platinum")} | {(c, "india") for c in MCX_SYMBOLS}
    | {("gold", "india_am"), ("gold", "landed")}
)


class FeedError(RuntimeError):
    """A price feed request failed."""


def _get(url: str, *, http_get: HttpGet, params: Optional[dict] = None, accept: str = "application/json") -> Any:
    resp = http_get(url, params=params, timeout=30, headers={"Accept": accept, "User-Agent": USER_AGENT})
    status = getattr(resp, "status_code", None)
    if status != 200:
        raise FeedError(f"{url} returned HTTP {status}")
    return resp


def fetch_fred(today: date, *, http_get: HttpGet = requests.get) -> dict[tuple[str, str], list[tuple[date, float]]]:
    """Weekday spot prices of each FRED series over the last FRED_DAYS, complete sessions only."""
    start = (today - timedelta(days=FRED_DAYS)).isoformat()
    out = {}
    for commodity, series_id in FRED_SERIES.items():
        text = _get(FRED_CSV_URL, params={"id": series_id, "cosd": start}, accept="text/csv", http_get=http_get).text
        prices = {}
        for line in text.splitlines()[1:]:
            day, _, value = line.partition(",")
            try:
                d, v = date.fromisoformat(day.strip()), float(value)
            except ValueError:
                continue  # FRED leaves market holidays blank
            if d.weekday() < 5 and d < today and math.isfinite(v) and v > 0:
                prices[d] = v
        out[(commodity, "international")] = sorted(prices.items())
    return out


def mcx_contracts(*, http_get: HttpGet = requests.get) -> dict[str, list[dict]]:
    """The listed futures of each MCX commodity, nearest expiry first."""
    rows = json.loads(gzip.decompress(_get(MCX_INSTRUMENTS_URL, accept="application/gzip", http_get=http_get).content))
    commodity_of = {symbol: commodity for commodity, symbol in MCX_SYMBOLS.items()}
    out: dict[str, list[dict]] = {commodity: [] for commodity in MCX_SYMBOLS}
    for row in rows:
        commodity = commodity_of.get(row.get("underlying_symbol"))
        if commodity and row.get("instrument_type") == "FUT" and row.get("expiry") and row.get("instrument_key"):
            out[commodity].append({
                "key": row["instrument_key"], "contract": row.get("trading_symbol") or row["instrument_key"],
                "expiry": datetime.fromtimestamp(row["expiry"] / 1000, IST).date(),
            })
    return {commodity: sorted(listed, key=lambda c: c["expiry"]) for commodity, listed in out.items()}


def contract_candles(contract: dict, start: date, through: date, *, http_get: HttpGet = requests.get,
                     morning: bool = False) -> dict[date, tuple[float, float]]:
    """Weekday (close, volume) of one MCX contract from ``start`` through ``through``.

    With ``morning``, the close is the 30-minute candle from MORNING_CANDLE IST instead of the day's.
    """
    out, to = {}, through
    while to >= start:
        frm = max(start, to - timedelta(days=MORNING_CHUNK_DAYS - 1)) if morning else start
        unit = "minutes/30" if morning else "days/1"
        path = f"/v3/historical-candle/{quote(contract['key'], safe='')}/{unit}/{to.isoformat()}/{frm.isoformat()}"
        try:
            body = _get(f"{UPSTOX_BASE_URL}{path}", http_get=http_get).json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or body.get("status") != "success":
            raise FeedError(f"Upstox {path} returned no candles")
        for candle in (body.get("data") or {}).get("candles") or []:
            stamp = str(candle[0]) if candle else ""
            if morning and stamp[11:16] != MORNING_CANDLE:
                continue
            try:
                day, close, volume = date.fromisoformat(stamp[:10]), float(candle[4]), float(candle[5] or 0)
            except (IndexError, TypeError, ValueError):
                continue
            if day.weekday() < 5 and start <= day <= through and math.isfinite(close) and close > 0:
                out[day] = (close, volume)
        to = frm - timedelta(days=1)
    return out


def front_rows(commodity: str, listed: Sequence[dict], candles: dict[str, dict], start: date, through: date,
               after: Optional[dict] = None, source: str = "india") -> list[dict]:
    """One stored row per session: the front contract's close and the next contract's that day.

    The front is the nearest contract with trades that session that is more than ROLL_DAYS from
    expiry (the furthest one with trades, when none is). A session no contract traded is skipped:
    a far month's candle without volume repeats a stale price. The front never steps back to an
    earlier month than the last stored row's, so the series rolls forward only.
    """
    rows, floor = [], date.fromisoformat(after["expiry"]) if after else date.min
    day = start
    while day <= through:
        traded = [c for c in listed if c["expiry"] >= floor and day in candles.get(c["contract"], {})]
        active = [c for c in traded if candles[c["contract"]][day][1] > 0]
        if active:
            front = next((c for c in active if (c["expiry"] - day).days > ROLL_DAYS), active[-1])
            later = [c for c in traded if c["expiry"] > front["expiry"]]
            row = {"date": day.isoformat(), "commodity": commodity, "source": source, "contract": front["contract"],
                   "expiry": front["expiry"].isoformat(), "close": candles[front["contract"]][day][0]}
            if later:
                row.update({"next_contract": later[0]["contract"], "next_close": candles[later[0]["contract"]][day][0]})
            rows.append(row)
            floor = front["expiry"]
        day += timedelta(days=1)
    return rows


def fetch_mcx(stored: Sequence[dict], today: date, *, http_get: HttpGet = requests.get) -> list[dict]:
    """New stored rows for each MCX series, for the complete sessions after its last stored one.

    The series are each commodity's daily close ("india") and, for MORNING_COMMODITIES, its
    11:00 IST price ("india_am"). A series whose candles fail to load is skipped; the next run
    fetches it from the same point.
    """
    contracts = mcx_contracts(http_get=http_get)
    wanted = [(c, "india") for c in MCX_SYMBOLS] + [(c, "india_am") for c in MORNING_COMMODITIES]
    last: dict[tuple[str, str], dict] = {}
    for row in stored:
        key = (row["commodity"], row.get("source", "india"))
        if key in wanted and (key not in last or row["date"] > last[key]["date"]):
            last[key] = row
    through, new = today - timedelta(days=1), []
    for commodity, source in wanted:
        after = last.get((commodity, source))
        start = date.fromisoformat(after["date"]) + timedelta(days=1) if after else today - timedelta(days=MCX_BACKFILL_DAYS)
        listed = [c for c in contracts.get(commodity, []) if not after or c["expiry"] >= date.fromisoformat(after["expiry"])][:3]
        if start > through or not listed:
            continue
        try:
            candles = {c["contract"]: contract_candles(c, start, through, http_get=http_get, morning=source == "india_am")
                       for c in listed}
        except FeedError:
            continue
        new.extend(front_rows(commodity, listed, candles, start, through, after, source))
    return new


def _metals_snapshot(day: date, *, http_get: HttpGet) -> Optional[dict]:
    for template in METALS_URLS:
        try:
            body = _get(template.format(day=day.isoformat()), http_get=http_get).json()
        except (FeedError, ValueError, requests.RequestException):
            continue
        if isinstance(body, dict) and isinstance(body.get("usd"), dict):
            return body["usd"]
    return None


def fetch_metals(stored: Sequence[dict], today: date, *, http_get: HttpGet = requests.get) -> list[dict]:
    """New snapshot rows (SNAPSHOT_CODES) for the weekdays after each one's last stored day, before today.

    A day's snapshot is used from the next IST day on, like an MCX session, so the newest metals
    date never runs ahead of the newest MCX one. An old day missing from both hosts is skipped;
    a missing one from the last three days stops the run there, and the next run retries it.
    """
    last: dict[str, str] = {}
    for row in stored:
        if row.get("source") == "international" and row["commodity"] in SNAPSHOT_CODES:
            last[row["commodity"]] = max(last.get(row["commodity"], ""), row["date"])
    backfill = today - timedelta(days=METALS_BACKFILL_DAYS)
    starts = {c: date.fromisoformat(last[c]) + timedelta(days=1) if c in last else backfill for c in SNAPSHOT_CODES}
    day, rows = min(starts.values()), []
    while day < today:
        due = [c for c in SNAPSHOT_CODES if starts[c] <= day]
        if day.weekday() < 5 and due:
            rates = _metals_snapshot(day, http_get=http_get)
            if rates is None and (today - day).days <= 3:
                break
            for commodity in due:
                code, invert = SNAPSHOT_CODES[commodity]
                rate = (rates or {}).get(code)
                if isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0:
                    rows.append({"date": day.isoformat(), "commodity": commodity, "source": "international",
                                 "close": round(1 / rate, 4) if invert else round(rate, 6)})
        day += timedelta(days=1)
    return rows


def _roll_ratio(earlier: Sequence[dict], new_contract: str) -> float:
    """The new contract's price over the old one's on the last session both traded."""
    old = earlier[-1].get("contract")
    for row in reversed(earlier):
        if row.get("contract") != old:
            break
        if row.get("next_contract") == new_contract and row.get("next_close"):
            return row["next_close"] / row["close"]
    return 1.0


def continuous(rows: Sequence[dict]) -> list[tuple[date, float]]:
    """One commodity's stored front-month closes, ratio-adjusted back from the newest contract.

    At each roll the earlier prices are scaled by the new contract's price over the old one's, so
    the step between contract months is not read as a move. The newest prices are the real ones.
    """
    rows = sorted(rows, key=lambda r: r["date"])
    out, factor = [], 1.0
    for i in range(len(rows) - 1, -1, -1):
        out.append((date.fromisoformat(rows[i]["date"]), rows[i]["close"] * factor))
        if i and rows[i - 1].get("contract") != rows[i].get("contract"):
            factor *= _roll_ratio(rows[:i], rows[i]["contract"])
    return out[::-1]


def import_duty(day: date) -> float:
    rate = DUTY_BEFORE
    for since, duty in GOLD_IMPORT_DUTY:
        if day >= since:
            rate = duty
    return rate


def build_series(stored: Sequence[dict], fred: dict) -> dict[tuple[str, str], list[tuple[date, float]]]:
    """Every (commodity, source) price series the pairs read, plus gold's landed cost in India."""
    series = dict(fred)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in stored:
        grouped.setdefault((row["commodity"], row.get("source", "india")), []).append(row)
    for (commodity, source), rows in grouped.items():
        scale = MCX_PER_KG.get(commodity, 1.0) if source in ("india", "india_am") else 1.0
        series[(commodity, source)] = [(day, price * scale) for day, price in continuous(rows)]
    # The world gold price in rupees per kg with import duty, from one snapshot: what MCX gold
    # (put per kg) would cost to import that morning.
    inr = dict(series.get(("usdinr", "international"), []))
    series[("gold", "landed")] = [
        (day, usd * inr[day] * 1000 / TROY_OUNCE_G * (1 + import_duty(day)))
        for day, usd in series.get(("gold", "international"), []) if day in inr
    ]
    return series


def _needs(pair: dict[str, str]) -> tuple[tuple[str, str], tuple[str, str]]:
    if pair["source"] == "parity":
        return (pair["a"], "india_am"), (pair["b"], "landed")
    return (pair["a"], pair["source"]), (pair["b"], pair["source"])


def aligned(pair: dict[str, str], series: dict) -> tuple[list[date], np.ndarray, np.ndarray]:
    key_a, key_b = _needs(pair)
    a, b = series.get(key_a, []), series.get(key_b, [])
    b_by_day = dict(b)
    days = [d for d, _ in a if d in b_by_day]
    a_by_day = dict(a)
    return days, np.log([a_by_day[d] for d in days]), np.log([b_by_day[d] for d in days])


def _hedge(la: np.ndarray, lb: np.ndarray, fixed: bool) -> tuple[float, float]:
    """Least-squares hedge ratio, or the plain price ratio (1) when the fit is not positive.

    A negative fit means the legs moved against each other over the window, so a
    "hedged" spread would add the two exposures instead of offsetting them.
    """
    if not fixed:
        beta, alpha = np.polyfit(lb, la, 1)
        if beta > 0:
            return float(beta), float(alpha)
    return 1.0, float(np.mean(la - lb))


def history(days: Sequence[date], la: np.ndarray, lb: np.ndarray, *, fixed: bool = False) -> list[dict]:
    """Per-session hedge ratio and z-score, each from data up to that session only."""
    out = []
    for t in range(MIN_SESSIONS - 1, len(days)):
        lo = max(0, t + 1 - HEDGE_WINDOW)
        beta, _ = _hedge(la[lo:t + 1], lb[lo:t + 1], fixed)
        window = la[t + 1 - Z_WINDOW:t + 1] - beta * lb[t + 1 - Z_WINDOW:t + 1]
        sd = float(np.std(window, ddof=1))
        if not sd > 0:
            continue
        out.append({"date": days[t], "i": t, "beta": beta, "spread": float(window[-1]), "mean": float(window.mean()),
                    "z": float((window[-1] - window.mean()) / sd)})
    return out


def diagnostics(la: np.ndarray, lb: np.ndarray, *, fixed: bool = False) -> dict[str, Any]:
    """Half-life and Engle-Granger cointegration test on the latest hedge window."""
    la, lb = la[-HEDGE_WINDOW:], lb[-HEDGE_WINDOW:]
    beta, alpha = _hedge(la, lb, fixed)
    resid = la - alpha - beta * lb
    lag, diff = resid[:-1], np.diff(resid)
    # Engle-Granger: Dickey-Fuller regression of the residual change on its lagged level.
    rho = float(lag @ diff / (lag @ lag))
    err = diff - rho * lag
    se = math.sqrt(float(err @ err) / (len(diff) - 1) / float(lag @ lag))
    n = len(resid)
    # MacKinnon (2010) 5% critical value, two variables with a constant.
    crit = -3.3377 - 5.967 / n - 8.98 / n ** 2
    slope = float(np.polyfit(lag, diff, 1)[0])
    half_life = -math.log(2) / math.log(1 + slope) if -1 < slope < 0 else None
    return {
        "eg_t": round(rho / se, 3), "eg_crit_5pct": round(crit, 3), "cointegrated": rho / se < crit,
        "half_life": round(half_life, 1) if half_life is not None else None,
    }


def simulate(days: Sequence[date], la: np.ndarray, lb: np.ndarray, hist: list[dict], *, start: int = 0) -> list[dict]:
    """Replay the signal rule over a pair's history from session index ``start``."""
    trades, open_trade = [], None
    for row in hist:
        t = row["i"]
        if t < start:
            continue
        if open_trade is not None:
            held = t - open_trade["i"]
            move = (la[t] - open_trade["beta"] * lb[t]) - open_trade["entry_spread"]
            if abs(row["z"]) < EXIT_Z or held >= MAX_HOLD:
                open_trade.update({
                    "exit_date": days[t].isoformat(), "sessions": held,
                    "status": "reverted" if abs(row["z"]) < EXIT_Z else "timed_out",
                    "spread_return": round(open_trade["side"] * move / (1 + abs(open_trade["beta"])) - 2 * COST_PER_SIDE, 6),
                })
                trades.append(open_trade)
                open_trade = None
            continue
        if abs(row["z"]) >= ENTRY_Z:
            open_trade = {
                "i": t, "entry_date": days[t].isoformat(), "entry_z": round(row["z"], 3), "beta": row["beta"],
                "entry_spread": row["spread"], "side": -1 if row["z"] > 0 else 1,
            }
    if open_trade is not None:
        open_trade["status"] = "open"
        trades.append(open_trade)
    return trades


def summarize(trades: list[dict]) -> dict[str, Any]:
    done = [t for t in trades if t.get("status") in ("reverted", "timed_out")]
    if not done:
        return {"signals": len(trades), "closed": 0}
    returns = np.array([t["spread_return"] for t in done])
    return {
        "signals": len(trades), "closed": len(done),
        "reverted": round(sum(t["status"] == "reverted" for t in done) / len(done), 3),
        "avg_return": round(float(returns.mean()), 5),
        "total_return": round(float(returns.sum()), 4),
        "win_rate": round(float((returns > 0).mean()), 3),
    }


def _legs(pair: dict[str, str]) -> tuple[str, str]:
    if pair["source"] == "parity":
        return "MCX gold", "Import cost"
    first, second = pair["label"].replace(" (MCX)", "").split(" / ")
    return first, second


def _state(pair: dict[str, str], z: float) -> dict[str, Any]:
    if abs(z) < ENTRY_Z:
        return {"stretched": False, "rich": None, "cheap": None}
    first, second = _legs(pair)
    rich, cheap = (first, second) if z > 0 else (second, first)
    return {"stretched": True, "rich": rich, "cheap": cheap}


def track_pair(pair: dict[str, str], series: dict) -> Optional[dict[str, Any]]:
    """The tracker row for one pair: today's hedge, z-score, diagnostics and a replay."""
    fixed = pair["source"] == "parity"
    if any(need not in FEEDS for need in _needs(pair)):
        return {"key": pair["key"], "label": pair["label"], "why": pair["why"], "source": pair["source"], "sessions": 0,
                "paused": True, "note": "Paused until a price feed for this pair is connected."}
    days, la, lb = aligned(pair, series)
    if len(days) < MIN_SESSIONS:
        return {"key": pair["key"], "label": pair["label"], "why": pair["why"], "source": pair["source"],
                "sessions": len(days), "note": f"{len(days)} sessions of history; tracking starts at {MIN_SESSIONS}."}
    hist = history(days, la, lb, fixed=fixed)
    last = hist[-1]
    # The replay covers the sessions before live tracking began; from then on the ledger records signals.
    replay = [t for t in simulate(days, la, lb, hist) if t["entry_date"] < LIVE_SINCE.isoformat()]
    # How far the spread sits from its window average, and the move in either leg alone that closes it.
    gap = last["spread"] - last["mean"]
    ratio = math.exp(la[-1] - lb[-1])
    return {
        "key": pair["key"], "label": pair["label"], "legs": list(_legs(pair)), "why": pair["why"], "source": pair["source"],
        "sessions": len(days), "data_from": days[0].isoformat(), "date": last["date"].isoformat(),
        "short_history": len(days) < FULL_HISTORY,
        "ratio": round(ratio, 6),
        "usual_ratio": round(ratio * math.exp(-gap), 6),
        "to_usual": {"a": round(math.expm1(-gap), 5), "b": round(math.expm1(gap / last["beta"]), 5)},
        "beta": round(last["beta"], 4), "hedge": "fixed" if fixed else ("ratio" if last["beta"] == 1.0 else "regression"),
        "z": round(last["z"], 3),
        "z_recent": [round(row["z"], 3) for row in hist[-Z_WINDOW:]],
        **_state(pair, last["z"]),
        **diagnostics(la, lb, fixed=fixed),
        "replay": summarize(replay),
        "replay_signals": [
            {k: t.get(k) for k in ("entry_date", "entry_z", "status", "exit_date", "sessions", "spread_return")} for t in replay
        ],
        "_days": days, "_la": la, "_lb": lb, "_hist": hist,
    }


def update_ledger(ledger: list[dict], rows: list[dict], now: datetime) -> int:
    """Advance open live signals through new sessions, then open new ones on the newest session."""
    added = 0
    by_key = {row["key"]: row for row in rows if "_hist" in row}
    for event in ledger:
        if event["status"] != "open" or event["pair"] not in by_key:
            continue
        row = by_key[event["pair"]]
        la, lb, days = row["_la"], row["_lb"], row["_days"]
        entry_i = days.index(date.fromisoformat(event["entry_date"])) if date.fromisoformat(event["entry_date"]) in days else None
        if entry_i is None:
            continue
        # Measured on today's series: a roll rescales earlier MCX prices, so both ends of the
        # trade must share one basis. Without a roll this equals the recorded entry spread.
        entry_spread = la[entry_i] - event["beta"] * lb[entry_i]
        for h in row["_hist"]:
            if h["i"] <= entry_i:
                continue
            held = h["i"] - entry_i
            move = (la[h["i"]] - event["beta"] * lb[h["i"]]) - entry_spread
            if abs(h["z"]) < EXIT_Z or held >= MAX_HOLD:
                event.update({
                    "status": "reverted" if abs(h["z"]) < EXIT_Z else "timed_out", "exit_date": h["date"].isoformat(),
                    "exit_z": round(h["z"], 3), "sessions": held,
                    "spread_return": round(event["side"] * move / (1 + abs(event["beta"])) - 2 * COST_PER_SIDE, 6),
                })
                break
    open_pairs = {e["pair"] for e in ledger if e["status"] == "open"}
    seen = {(e["pair"], e["entry_date"]) for e in ledger}
    for key, row in by_key.items():
        last = row["_hist"][-1]
        if key in open_pairs or abs(last["z"]) < ENTRY_Z or (key, last["date"].isoformat()) in seen:
            continue
        ledger.append({
            "pair": key, "entry_date": last["date"].isoformat(), "issued_at": now.isoformat(),
            "entry_z": round(last["z"], 3), "beta": round(last["beta"], 6), "entry_spread": round(last["spread"], 8),
            "side": -1 if last["z"] > 0 else 1, "status": "open",
        })
        added += 1
    return added


def run(series: dict, ledger: list[dict], now: datetime) -> tuple[dict, list[dict]]:
    rows = [track_pair(pair, series) for pair in PAIRS]
    update_ledger(ledger, rows, now)
    live = {}
    for pair in PAIRS:
        events = [e for e in ledger if e["pair"] == pair["key"]]
        live[pair["key"]] = summarize(events) | {"open": sum(e["status"] == "open" for e in events)}
    public = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    dated = [row["date"] for row in public if row.get("date")]
    payload = {
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "data_through": max(dated) if dated else None,
        "source": "Upstox (MCX front-month futures), FRED (Brent, WTI and Henry Hub spot) and the fawazahmed0 currency API (gold, platinum and the rupee)",
        "rule": {
            "hedge_window": HEDGE_WINDOW, "z_window": Z_WINDOW, "entry_z": ENTRY_Z, "exit_z": EXIT_Z,
            "max_hold": MAX_HOLD, "cost_per_side": COST_PER_SIDE, "min_sessions": MIN_SESSIONS,
        },
        "backtest": BACKTEST,
        "pairs": public,
        "track_record": {
            "by_pair": live,
            "live_since": LIVE_SINCE.isoformat(),
            "first_signal": min((e["entry_date"] for e in ledger), default=None),
            "recent": [
                {k: e.get(k) for k in ("pair", "entry_date", "entry_z", "status", "exit_date", "sessions", "spread_return")}
                for e in sorted(ledger, key=lambda e: e["entry_date"])[-30:]
            ],
        },
    }
    return payload, ledger
