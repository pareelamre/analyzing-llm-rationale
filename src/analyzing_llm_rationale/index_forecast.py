"""Calibrated range forecasts for ten NSE indices, from Upstox daily closes and India VIX.

Each index gets a forecast of its close 1, 5 and 21 sessions ahead: quantiles, the chance it
closes higher, and the chance it closes above round levels. Up/down calls have no skill on
these indices (a 2018-2026 walk-forward test put every model at a coin flip), so the forecast
is a calibrated range, not a direction.

The model is a HAR-style variance regression: realized variance over the horizon is regressed
on a fast EWMA, a slow EWMA and India VIX, refit every day: on the last 750 sessions for the
1-session horizon and on every session since India VIX began in 2009 for the longer two (see
TRAINING_SESSIONS). Its residuals, standardized by the fitted volatility, give the shape of the
distribution, so fat tails come from the data rather than a normal assumption. For the
21-session horizon the ten
indices' residuals are pooled into one shape (see POOLED_SHAPE_HORIZONS). A plain EWMA normal
forecast is recorded alongside as the baseline the live track record is scored against.

Forecasts are only issued between the close they are made from and the next session's open,
so a late run can never record a forecast for a close that has already happened.
"""
from __future__ import annotations

import bisect
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Sequence
from urllib.parse import quote

import numpy as np
import requests

#: Upstox's candle and market-holiday routes are open data endpoints: no account or token.
UPSTOX_BASE_URL = "https://api.upstox.com"
USER_AGENT = "Foresea/index-forecast (+https://foresea.ink)"
IST = timezone(timedelta(hours=5, minutes=30))
NSE_OPEN = time(9, 15)
#: A day's candle is only used once the closing session has fixed its close.
CLOSE_SETTLED = time(15, 45)

#: Display name and Upstox instrument key.
INDICES: tuple[tuple[str, str], ...] = (
    ("Nifty 50", "NSE_INDEX|Nifty 50"),
    ("Nifty Bank", "NSE_INDEX|Nifty Bank"),
    ("Nifty Financial Services", "NSE_INDEX|Nifty Fin Service"),
    ("Nifty Midcap Select", "NSE_INDEX|NIFTY MID SELECT"),
    ("Nifty Next 50", "NSE_INDEX|Nifty Next 50"),
    ("Nifty IT", "NSE_INDEX|Nifty IT"),
    ("Nifty Auto", "NSE_INDEX|Nifty Auto"),
    ("Nifty Pharma", "NSE_INDEX|Nifty Pharma"),
    ("Nifty FMCG", "NSE_INDEX|Nifty FMCG"),
    ("Nifty Metal", "NSE_INDEX|Nifty Metal"),
)
VIX_KEY = "NSE_INDEX|India VIX"
#: A year of warm-up before India VIX begins (March 2009), which every training row needs.
HISTORY_START = date(2008, 1, 1)
#: Upstox serves at most a decade of daily candles per request.
CHUNK_DAYS = 3600
HORIZONS = (1, 5, 21)
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
#: The tails' own levels, for pricing options on the range (fund_options); not drawn on the page.
TAIL_QUANTILES = (0.01, 0.99)
_NORMAL_Z = (-1.6448536, -1.2815516, -0.6744898, 0.0, 0.6744898, 1.2815516, 1.6448536)
FAST_ALPHA = 0.06  # RiskMetrics lambda 0.94
SLOW_ALPHA = 0.01
WARMUP_SESSIONS = 250
#: Sessions each horizon's fit trains on; None trains on every session with India VIX. In the
#: 2016-2026 walk-forward on Upstox closes, the whole history beat the last 750 sessions on
#: quantile loss at 5 and 21 sessions in both halves (21 sessions: 152.3 to 150.0 in 2016-20,
#: 119.1 to 117.7 in 2021-26; Brier on "closes higher" 0.2410 to 0.2374 and 0.2452 to 0.2430).
#: At 1 session it tied on quantile loss but its 80% range held only 78.4% of closes against
#: 79.8%, so that horizon keeps the recent window.
TRAINING_SESSIONS: dict[int, Optional[int]] = {1: 750, 5: None, 21: None}

#: Horizons whose residual shape pools every index forecast that day. 750 sessions hold only
#: about 35 separate 21-session outcomes per index, so one index's own tails are noisy. In the
#: 2016-2026 walk-forward, pooling the ten indices beat each index's own shape at 21 sessions on
#: quantile loss, Brier on "closes higher" (0.2434 to 0.2409 in 2016-20, 0.2435 to 0.2417 in
#: 2021-26) and Brier on the round levels, in both halves. At 1 and 5 sessions it did worse.
POOLED_SHAPE_HORIZONS = (21,)

#: This module walked forward over Upstox closes of the ten indices (Midcap Select from 2024,
#: once two years past its 2022 launch): a forecast every fifth Nifty 50 session from 2016, each
#: index fit on its history through that day and the 21-session shape pooled as run() does,
#: scored like the live ledger.
BACKTEST = {
    "window": "2016-2026",
    "indices": 10,
    "by_horizon": [
        {"h": 1, "n": 4943, "coverage_50": 0.500, "coverage_80": 0.798, "coverage_90": 0.901, "skill_vs_baseline": 0.004},
        {"h": 5, "n": 4933, "coverage_50": 0.494, "coverage_80": 0.787, "coverage_90": 0.887, "skill_vs_baseline": 0.012},
        {"h": 21, "n": 4903, "coverage_50": 0.505, "coverage_80": 0.804, "coverage_90": 0.890, "skill_vs_baseline": 0.025},
    ],
    "direction": "Up/down calls from momentum, trend, volatility and VIX features had no skill (AUC 0.47-0.52).",
}


class UpstoxError(RuntimeError):
    """An Upstox request failed."""


HttpGet = Callable[..., Any]


def upstox_get(path: str, *, http_get: HttpGet = requests.get) -> Any:
    """GET one Upstox route and return its ``data``."""
    resp = http_get(f"{UPSTOX_BASE_URL}{path}", timeout=30, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    status = getattr(resp, "status_code", None)
    try:
        body = resp.json()
    except ValueError:
        body = None
    if status != 200 or not isinstance(body, dict) or body.get("status") != "success":
        raise UpstoxError(f"Upstox {path} returned HTTP {status}")
    return body.get("data")


def _closes(candles: Iterable[Sequence[Any]]) -> list[tuple[date, float]]:
    """Weekday closes from Upstox candles ``[timestamp, open, high, low, close, ...]``; later rows win.

    Weekend special sessions (Muhurat trading, a Sunday budget) are dropped: forecasts count
    sessions as weekdays less holidays (next_sessions), and the ledger scores by that count.
    """
    out = {}
    for candle in candles:
        try:
            day, close = date.fromisoformat(str(candle[0])[:10]), float(candle[4])
        except (IndexError, TypeError, ValueError):
            continue
        if day.weekday() < 5 and math.isfinite(close) and close > 0:
            out[day] = close
    return sorted(out.items())


def daily_closes(key: str, start: date, through: date, *, http_get: HttpGet = requests.get) -> list[tuple[date, float]]:
    """Every weekday close of one instrument from ``start`` through ``through``."""
    path_key = quote(key, safe="")
    # The historical route can lag the evening close; the intraday route carries today's candle.
    candles = list((upstox_get(f"/v3/historical-candle/intraday/{path_key}/days/1", http_get=http_get) or {}).get("candles") or [])
    to = through
    while to >= start:
        frm = max(start, to - timedelta(days=CHUNK_DAYS))
        chunk = upstox_get(f"/v3/historical-candle/{path_key}/days/1/{to.isoformat()}/{frm.isoformat()}", http_get=http_get)
        candles.extend((chunk or {}).get("candles") or [])
        to = frm - timedelta(days=1)
    return [(day, close) for day, close in _closes(candles) if start <= day <= through]


def fetch_holidays(today: date, *, http_get: HttpGet = requests.get) -> list[date]:
    """NSE full-day trading holidays this year and, near the year end, early next year."""
    rows = list(upstox_get("/v2/market/holidays", http_get=http_get) or [])
    # The list covers the current year only; next year's dates are looked up one by one.
    day, last = date(today.year + 1, 1, 1), today + timedelta(days=60)
    while day <= last:
        if day.weekday() < 5:
            rows.extend(upstox_get(f"/v2/market/holidays/{day.isoformat()}", http_get=http_get) or [])
        day += timedelta(days=1)
    return sorted({
        date.fromisoformat(row["date"])
        for row in rows
        if isinstance(row, dict) and row.get("date") and row.get("holiday_type") == "TRADING_HOLIDAY"
        and "NSE" in (row.get("closed_exchanges") or [])
    })


def fetch_market_data(now: datetime, *, http_get: HttpGet = requests.get) -> dict[str, Any]:
    """Fetch every index's closes since HISTORY_START, India VIX and NSE holidays for one run."""
    local = now.astimezone(IST)
    through = local.date() if local.time() >= CLOSE_SETTLED else local.date() - timedelta(days=1)
    levels = {}
    for name, key in INDICES:
        try:
            levels[name] = daily_closes(key, HISTORY_START, through, http_get=http_get)
        except UpstoxError:
            levels[name] = []
    vix = daily_closes(VIX_KEY, HISTORY_START, through, http_get=http_get)
    return {"levels": levels, "vix": vix, "holidays": fetch_holidays(local.date(), http_get=http_get)}


def _ewma_var(r: np.ndarray, alpha: float) -> np.ndarray:
    """Adjusted EWMA of squared returns (pandas ``ewm(alpha=..., adjust=True)``)."""
    out = np.empty_like(r)
    num = den = 0.0
    for i, x in enumerate(r):
        num = x * x + (1 - alpha) * num
        den = 1 + (1 - alpha) * den
        out[i] = num / den
    return out


def _features(closes: Sequence[tuple[date, float]], vix: Sequence[tuple[date, float]]):
    days = [d for d, _ in closes]
    logc = np.log(np.array([c for _, c in closes]))
    r = np.diff(logc)
    fast, slow = _ewma_var(r, FAST_ALPHA), _ewma_var(r, SLOW_ALPHA)
    vix_days = [d for d, _ in vix]
    vix_vals = [v for _, v in vix]
    vix_var = np.full(len(r), np.nan)
    for i, day in enumerate(days[1:]):
        j = bisect.bisect_right(vix_days, day) - 1
        if j >= 0:
            vix_var[i] = (vix_vals[j] / 100) ** 2 / 252
    # Row i describes the close on days[i + 1].
    return days[1:], logc[1:], np.column_stack([fast, slow, vix_var])


def fit_horizon(closes: Sequence[tuple[date, float]], vix: Sequence[tuple[date, float]], h: int) -> Optional[dict[str, Any]]:
    """Fit the variance regression and residual shape for one index and horizon."""
    days, logc, X = _features(closes, vix)
    n = len(logc)
    usable = np.arange(WARMUP_SESSIONS, n - h)
    usable = usable[np.isfinite(X[usable]).all(axis=1)]
    if TRAINING_SESSIONS.get(h):
        usable = usable[-TRAINING_SESSIONS[h]:]
    if len(usable) < 250 or not np.isfinite(X[-1]).all():
        return None
    fwd = logc[usable + h] - logc[usable]
    coef, *_ = np.linalg.lstsq(X[usable], fwd ** 2 / h, rcond=None)
    coef = np.clip(coef, 0, None)
    if coef.sum() <= 0:
        coef = np.array([1.0, 0.0, 0.0])
    fitted = X[usable] @ coef
    keep = fitted > 0
    z = np.sort(fwd[keep] / np.sqrt(fitted[keep] * h))
    var_now = float(X[-1] @ coef)
    if not var_now > 0:
        return None
    return {
        "as_of": days[-1], "close": float(math.exp(logc[-1])), "sigma": math.sqrt(var_now * h),
        "baseline_sigma": math.sqrt(float(X[-1, 0]) * h), "z": z, "coef": coef.tolist(),
    }


def _z_quantile(z: np.ndarray, p: float) -> float:
    return float(np.quantile(z, p))


def _z_cdf(z: np.ndarray, x: float) -> float:
    """Empirical CDF of the standardized residuals, linearly interpolated between order statistics."""
    n = len(z)
    if x <= z[0]:
        return 0.5 / n
    if x >= z[-1]:
        return 1 - 0.5 / n
    i = int(np.searchsorted(z, x)) - 1
    frac = float((x - z[i]) / (z[i + 1] - z[i])) if z[i + 1] > z[i] else 0.0
    return (i + 0.5 + frac) / n


def _nice_step(raw: float) -> float:
    """The 1, 2, 2.5 or 5 x 10^k step nearest ``raw`` on a log scale."""
    exp = 10 ** math.floor(math.log10(raw))
    return min((m * exp for m in (1, 2, 2.5, 5, 10)), key=lambda step: abs(math.log(step / raw)))


def round_levels(close: float, sigma: float) -> list[float]:
    """Six round levels about half a standard deviation apart, so most fall inside the range."""
    step = _nice_step(close * (math.exp(sigma) - 1) / 2)
    base = math.floor(close / step) * step
    # In a crash the step can exceed half the close; a level at or below zero has no meaning.
    return [base + k * step for k in range(-2, 4) if base + k * step > 0]


def next_sessions(after: date, count: int, holidays: Iterable[date]) -> list[date]:
    closed = set(holidays)
    out, day = [], after
    while len(out) < count:
        day += timedelta(days=1)
        if day.weekday() < 5 and day not in closed:
            out.append(day)
    return out


def fit_index(closes, vix) -> Optional[dict[int, dict[str, Any]]]:
    """Every horizon's fit for one index, or None without enough history."""
    if len(closes) < WARMUP_SESSIONS + 300:
        return None
    fits = {h: fit_horizon(closes, vix, h) for h in HORIZONS}
    return None if any(fit is None for fit in fits.values()) else fits


def pooled_shapes(all_fits: Iterable[dict[int, dict[str, Any]]]) -> dict[int, np.ndarray]:
    """The standardized residuals of every fitted index together, for POOLED_SHAPE_HORIZONS."""
    all_fits = list(all_fits)
    if not all_fits:
        return {}
    return {h: np.sort(np.concatenate([fits[h]["z"] for fits in all_fits])) for h in POOLED_SHAPE_HORIZONS}


def forecast_index(name: str, closes, vix, holidays) -> Optional[dict[str, Any]]:
    """Today's forecast for one index at every horizon from its own residuals alone."""
    fits = fit_index(closes, vix)
    return build_forecast(name, fits, holidays) if fits else None


def build_forecast(name: str, fits: dict[int, dict[str, Any]], holidays, shapes: Optional[dict[int, np.ndarray]] = None) -> dict[str, Any]:
    """The forecast from one index's fits. ``shapes`` replaces its own residuals at those horizons."""
    as_of, close = fits[1]["as_of"], fits[1]["close"]
    sessions = next_sessions(as_of, max(HORIZONS), holidays)
    horizons = []
    for h, fit in fits.items():
        sigma, z = fit["sigma"], (shapes or {}).get(h, fit["z"])
        horizons.append({
            "h": h,
            "target_date": sessions[h - 1].isoformat(),
            "sigma": round(sigma, 6),
            "quantiles": [round(close * math.exp(sigma * _z_quantile(z, p)), 2) for p in QUANTILES],
            "tail_quantiles": {str(p): round(close * math.exp(sigma * _z_quantile(z, p)), 2) for p in TAIL_QUANTILES},
            "baseline_quantiles": [round(close * math.exp(fit["baseline_sigma"] * q), 2) for q in _NORMAL_Z],
            "p_up": round(1 - _z_cdf(z, 0.0), 4),
            "levels": [
                {"level": level, "p_above": round(1 - _z_cdf(z, math.log(level / close) / sigma), 4)}
                for level in round_levels(close, sigma)
            ],
        })
    return {
        "name": name, "as_of": as_of.isoformat(), "close": round(close, 2),
        "vol_annual_pct": round(fits[1]["sigma"] * math.sqrt(252) * 100, 2),
        "horizons": horizons,
    }


def issue_window_open(as_of: date, now: datetime, holidays: Iterable[date]) -> bool:
    """True while ``now`` is after the as_of close and before the next session opens."""
    next_open = datetime.combine(next_sessions(as_of, 1, holidays)[0], NSE_OPEN, tzinfo=IST)
    close = datetime.combine(as_of, time(15, 30), tzinfo=IST)
    return close <= now.astimezone(IST) < next_open


def _quantile_score(quantiles: Sequence[float], close: float, realized: float) -> float:
    """Mean pinball loss over the stored quantiles, in log-return basis points."""
    y = math.log(realized / close)
    total = 0.0
    for p, q in zip(QUANTILES, quantiles):
        diff = y - math.log(q / close)
        total += max(p * diff, (p - 1) * diff)
    return total / len(QUANTILES) * 1e4


def score_ledger(ledger: list[dict], levels: dict[str, Sequence[tuple[date, float]]]) -> int:
    """Score every pending ledger row whose target session has closed. Returns rows scored."""
    scored = 0
    for row in ledger:
        if row.get("realized") is not None:
            continue
        series = levels.get(row["name"]) or []
        days = [d for d, _ in series]
        start = bisect.bisect_right(days, date.fromisoformat(row["as_of"]))
        target = start + row["h"] - 1
        if target >= len(series):
            continue
        realized = series[target][1]
        q = row["quantiles"]
        row.update({
            "realized": round(realized, 2),
            "realized_date": series[target][0].isoformat(),
            "in_50": q[2] <= realized <= q[4],
            "in_80": q[1] <= realized <= q[5],
            "in_90": q[0] <= realized <= q[6],
            "up": realized > row["close"],
            "qs": round(_quantile_score(q, row["close"], realized), 4),
            "qs_baseline": round(_quantile_score(row["baseline_quantiles"], row["close"], realized), 4),
        })
        scored += 1
    return scored


def summarize_ledger(ledger: list[dict]) -> list[dict]:
    out = []
    for h in HORIZONS:
        done = [row for row in ledger if row["h"] == h and row.get("realized") is not None]
        if not done:
            out.append({"h": h, "n": 0})
            continue
        qs, qs_base = sum(r["qs"] for r in done), sum(r["qs_baseline"] for r in done)
        out.append({
            "h": h,
            "n": len(done),
            "coverage_50": round(sum(r["in_50"] for r in done) / len(done), 4),
            "coverage_80": round(sum(r["in_80"] for r in done) / len(done), 4),
            "coverage_90": round(sum(r["in_90"] for r in done) / len(done), 4),
            "skill_vs_baseline": round(1 - qs / qs_base, 4) if qs_base > 0 else None,
            "brier_up": round(sum((r["p_up"] - r["up"]) ** 2 for r in done) / len(done), 4),
        })
    return out


def ledger_rows(forecast: dict, issued_at: datetime) -> list[dict]:
    return [
        {
            "name": forecast["name"], "as_of": forecast["as_of"], "issued_at": issued_at.isoformat(),
            "close": forecast["close"], "h": hz["h"], "target_date": hz["target_date"],
            "quantiles": hz["quantiles"], "baseline_quantiles": hz["baseline_quantiles"], "p_up": hz["p_up"],
        }
        for hz in forecast["horizons"]
    ]


def run(data: dict[str, Any], ledger: list[dict], now: datetime) -> tuple[dict, list[dict]]:
    """Score the ledger, add today's forecasts when the issue window is open, build the payload."""
    holidays = [date.fromisoformat(d) if isinstance(d, str) else d for d in data["holidays"]]
    score_ledger(ledger, data["levels"])
    known = {(row["name"], row["as_of"]) for row in ledger}
    forecasts, notes = [], []
    fitted = {name: fit_index(data["levels"].get(name) or [], data["vix"]) for name, _ in INDICES}
    shapes = pooled_shapes(fits for fits in fitted.values() if fits)
    for name, _ in INDICES:
        if fitted[name] is None:
            # Left out rather than carried over: an old forecast would read as today's.
            notes.append(f"{name}: not enough history to forecast.")
            continue
        forecast = build_forecast(name, fitted[name], holidays, shapes)
        if (name, forecast["as_of"]) not in known:
            if issue_window_open(date.fromisoformat(forecast["as_of"]), now, holidays):
                ledger.extend(ledger_rows(forecast, now))
                known.add((name, forecast["as_of"]))
            else:
                # Built after the next session opened: shown, but kept out of the track record.
                forecast["recorded"] = False
        forecasts.append(forecast)
    scored = sorted((row for row in ledger if row.get("realized") is not None), key=lambda r: (r["realized_date"], r["name"], r["h"]))
    payload = {
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "as_of": max((f["as_of"] for f in forecasts), default=None),
        "source": "Upstox (NSE index closes and India VIX)",
        "model": {
            "name": "HAR volatility with empirical residuals",
            "inputs": ["fast EWMA (0.94)", "slow EWMA (0.99)", "India VIX"],
            "training_sessions": {str(h): TRAINING_SESSIONS.get(h) for h in HORIZONS},
            "quantiles": list(QUANTILES),
            "horizons": list(HORIZONS),
            "baseline": "EWMA (0.94) volatility, normal distribution",
            "pooled_shape_horizons": list(POOLED_SHAPE_HORIZONS),
        },
        "backtest": BACKTEST,
        "indices": forecasts,
        "track_record": {
            "by_horizon": summarize_ledger(ledger),
            "first_forecast": min((row["as_of"] for row in ledger), default=None),
            "pending": sum(1 for row in ledger if row.get("realized") is None),
            "recent": [
                {k: row[k] for k in ("name", "h", "as_of", "realized_date", "close", "realized", "in_80", "quantiles")}
                for row in scored[-30:]
            ],
        },
        "notes": notes,
    }
    return payload, ledger
