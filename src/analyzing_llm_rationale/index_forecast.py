"""Calibrated range forecasts for ten NSE indices, from Tapetide daily levels and India VIX.

Each index gets a forecast of its close 1, 5 and 21 sessions ahead: quantiles, the chance it
closes higher, and the chance it closes above round levels. Up/down calls have no skill on
these indices (a 2018-2026 walk-forward test put every model at a coin flip), so the forecast
is a calibrated range, not a direction.

The model is a HAR-style variance regression: realized variance over the horizon is regressed
on a fast EWMA, a slow EWMA and India VIX, refit every day on the last 750 sessions. Its
residuals, standardized by the fitted volatility, give the shape of the distribution, so fat
tails come from the data rather than a normal assumption. A plain EWMA normal forecast is
recorded alongside as the baseline the live track record is scored against.

Forecasts are only issued between the close they are made from and the next session's open,
so a late run can never record a forecast for a close that has already happened.
"""
from __future__ import annotations

import bisect
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import requests

TAPETIDE_BASE_URL = "https://api.tapetide.com"
IST = timezone(timedelta(hours=5, minutes=30))
NSE_OPEN = time(9, 15)

#: Display name, as Tapetide's index catalogue lists it, and the slug used when the catalogue
#: lookup is unavailable.
INDICES: tuple[tuple[str, str], ...] = (
    ("Nifty 50", "nifty-50"),
    ("Nifty Bank", "nifty-bank"),
    ("Nifty Financial Services", "nifty-financial-services"),
    ("Nifty Midcap Select", "nifty-midcap-select"),
    ("Nifty Next 50", "nifty-next-50"),
    ("Nifty IT", "nifty-it"),
    ("Nifty Auto", "nifty-auto"),
    ("Nifty Pharma", "nifty-pharma"),
    ("Nifty FMCG", "nifty-fmcg"),
    ("Nifty Metal", "nifty-metal"),
)
HORIZONS = (1, 5, 21)
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
_NORMAL_Z = (-1.6448536, -1.2815516, -0.6744898, 0.0, 0.6744898, 1.2815516, 1.6448536)
FAST_ALPHA = 0.06  # RiskMetrics lambda 0.94
SLOW_ALPHA = 0.01
WARMUP_SESSIONS = 250
TRAINING_SESSIONS = 750
FETCH_SESSIONS = 1000

#: This module walked forward over Yahoo Finance copies of the ten indices (Nifty Midcap 50
#: standing in for Midcap Select, whose history Yahoo lacks): a forecast every fifth session
#: from 2016, each fit on the 1,000 sessions before it, scored like the live ledger.
BACKTEST = {
    "window": "2016-2026",
    "indices": 10,
    "by_horizon": [
        {"h": 1, "n": 5335, "coverage_50": 0.496, "coverage_80": 0.803, "coverage_90": 0.898, "skill_vs_baseline": 0.013},
        {"h": 5, "n": 5326, "coverage_50": 0.506, "coverage_80": 0.797, "coverage_90": 0.895, "skill_vs_baseline": 0.012},
        {"h": 21, "n": 5295, "coverage_50": 0.487, "coverage_80": 0.784, "coverage_90": 0.885, "skill_vs_baseline": 0.011},
    ],
    "direction": "Up/down calls from momentum, trend, volatility and VIX features had no skill (AUC 0.47-0.52).",
}


class TapetideError(RuntimeError):
    """A Tapetide request failed or returned no data."""


HttpGet = Callable[..., Any]


def tapetide_get(path: str, token: str, params: Optional[dict] = None, *, http_get: HttpGet = requests.get) -> list[dict]:
    """GET one Tapetide list route and return its ``data`` rows, following cursors."""
    rows: list[dict] = []
    params = dict(params or {})
    for _ in range(10):
        resp = http_get(
            f"{TAPETIDE_BASE_URL}{path}", params=params, timeout=30,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "Foresea/index-forecast"},
        )
        status = getattr(resp, "status_code", None)
        try:
            body = resp.json()
        except ValueError:
            body = None
        if status != 200 or not isinstance(body, dict):
            code = ((body or {}).get("error") or {}).get("code") if isinstance(body, dict) else None
            raise TapetideError(f"Tapetide {path} returned HTTP {status}{f' ({code})' if code else ''}")
        if body.get("data") is None:
            notice = body.get("notice") or {}
            raise TapetideError(f"Tapetide {path} returned no data ({notice.get('code') or 'no notice'})")
        rows.extend(row for row in body["data"] if isinstance(row, dict))
        cursor = (body.get("meta") or {}).get("next_cursor")
        if not cursor:
            return rows
        params["cursor"] = cursor
    raise TapetideError(f"Tapetide {path} paged more than 10 times")


def resolve_slugs(token: str, *, http_get: HttpGet = requests.get) -> dict[str, str]:
    """Map each forecast index's name to its Tapetide slug from the index catalogue."""
    catalogue = {str(row.get("name", "")).strip().lower(): row.get("slug") for row in tapetide_get("/v1/reference/indices", token, {"limit": 200}, http_get=http_get)}
    return {name: catalogue.get(name.lower()) or fallback for name, fallback in INDICES}


def _closes(rows: Iterable[dict]) -> list[tuple[date, float]]:
    out = {}
    for row in rows:
        try:
            day, close = date.fromisoformat(str(row["date"])), float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(close) and close > 0:
            out[day] = close
    return sorted(out.items())


def fetch_market_data(token: str, today: date, *, http_get: HttpGet = requests.get) -> dict[str, Any]:
    """Fetch the levels, India VIX and NSE holidays one daily run needs (13 requests)."""
    start = (today - timedelta(days=int(FETCH_SESSIONS * 1.45))).isoformat()
    window = {"from": start, "to": today.isoformat(), "limit": FETCH_SESSIONS, "order": "desc"}
    slugs = resolve_slugs(token, http_get=http_get)
    levels = {}
    for name, _ in INDICES:
        try:
            levels[name] = _closes(tapetide_get(f"/v1/indices/{slugs[name]}/levels", token, window, http_get=http_get))
        except TapetideError:
            levels[name] = []
    vix = _closes(tapetide_get("/v1/market/vix/history", token, window, http_get=http_get))
    holidays = {
        date.fromisoformat(row["date"])
        for row in tapetide_get("/v1/market/holidays", token, {"from": (today - timedelta(days=40)).isoformat(), "to": (today + timedelta(days=60)).isoformat(), "limit": 500}, http_get=http_get)
        if row.get("exchange", "NSE") == "NSE" and (row.get("type") or "FULL") == "FULL" and row.get("date")
    }
    return {"slugs": slugs, "levels": levels, "vix": vix, "holidays": sorted(holidays)}


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
    usable = usable[np.isfinite(X[usable]).all(axis=1)][-TRAINING_SESSIONS:]
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


def forecast_index(name: str, closes, vix, holidays) -> Optional[dict[str, Any]]:
    """Today's forecast for one index at every horizon, or None without enough history."""
    if len(closes) < WARMUP_SESSIONS + 300:
        return None
    fits = {h: fit_horizon(closes, vix, h) for h in HORIZONS}
    if any(fit is None for fit in fits.values()):
        return None
    as_of, close = fits[1]["as_of"], fits[1]["close"]
    sessions = next_sessions(as_of, max(HORIZONS), holidays)
    horizons = []
    for h, fit in fits.items():
        sigma, z = fit["sigma"], fit["z"]
        horizons.append({
            "h": h,
            "target_date": sessions[h - 1].isoformat(),
            "sigma": round(sigma, 6),
            "quantiles": [round(close * math.exp(sigma * _z_quantile(z, p)), 2) for p in QUANTILES],
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
    for name, _ in INDICES:
        forecast = forecast_index(name, data["levels"].get(name) or [], data["vix"], holidays)
        if forecast is None:
            # Left out rather than carried over: an old forecast would read as today's.
            notes.append(f"{name}: not enough Tapetide history to forecast.")
            continue
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
        "source": "Tapetide (NSE official index levels and India VIX)",
        "model": {
            "name": "HAR volatility with empirical residuals",
            "inputs": ["fast EWMA (0.94)", "slow EWMA (0.99)", "India VIX"],
            "training_sessions": TRAINING_SESSIONS,
            "quantiles": list(QUANTILES),
            "horizons": list(HORIZONS),
            "baseline": "EWMA (0.94) volatility, normal distribution",
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
