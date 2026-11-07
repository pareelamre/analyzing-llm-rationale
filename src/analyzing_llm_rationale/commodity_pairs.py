"""Statistical-arbitrage tracker for ten commodity pairs, from Tapetide daily commodity prices.

Each pair's spread is the log price of one leg minus a hedge ratio times the log price of the
other, with the hedge ratio refit every day by least squares on up to 250 sessions. The spread's
z-score against its last 60 sessions says how stretched the pair is. A pair is "stretched" at
|z| >= 2, and the tracker records that as a signal, then follows it until |z| falls below 0.5
(reverted) or 20 sessions pass (timed out).

This is a tracker, not a strategy with an edge: the same rule on 2010-2026 US futures history
mostly lost money or broke even after costs (see BACKTEST). Live signals are recorded only on
the newest session at the time of a run, so the ledger never adds a signal after the fact.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import requests

from .index_forecast import tapetide_get

#: key, label, first leg, second leg, price source, why traders watch it.
#: international: benchmark USD price. india: domestic exchange (MCX) price in INR.
#: parity: the domestic exchange price against the import-parity (landed) cost, hedge fixed at 1.
PAIRS: tuple[dict[str, str], ...] = (
    {"key": "gold-silver", "a": "gold", "b": "silver", "source": "international", "label": "Gold / Silver",
     "why": "The gold-silver ratio, the most watched precious-metals spread."},
    {"key": "brent-wti", "a": "brentcrude", "b": "crudeoil", "source": "international", "label": "Brent / WTI",
     "why": "The benchmark crude spread between the North Sea and US grades."},
    {"key": "wti-natgas", "a": "crudeoil", "b": "naturalgas", "source": "international", "label": "WTI / Natural gas",
     "why": "The oil-gas ratio: energy priced as liquid fuel against gas."},
    {"key": "copper-gold", "a": "copper", "b": "gold", "source": "international", "label": "Copper / Gold",
     "why": "The copper-gold ratio, read as growth against safety."},
    {"key": "gold-platinum", "a": "gold", "b": "platinum", "source": "international", "label": "Gold / Platinum",
     "why": "Monetary against industrial precious metal."},
    {"key": "copper-aluminium", "a": "copper", "b": "aluminium", "source": "international", "label": "Copper / Aluminium",
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
FIRST_DATE = date(2026, 4, 1)

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


def fetch_prices(token: str, today: date, *, http_get: HttpGet = requests.get) -> list[dict]:
    """Every daily commodity row from Tapetide since prices began (about 11 rows a day)."""
    start = max(FIRST_DATE, today - timedelta(days=400))
    return tapetide_get(
        "/v1/macro/commodities", token,
        {"from": start.isoformat(), "to": today.isoformat(), "limit": 1000, "order": "asc"},
        http_get=http_get,
    )


def price_series(rows: Iterable[dict]) -> dict[tuple[str, str], list[tuple[date, float]]]:
    """Weekday prices per (commodity, source). Weekend rows repeat Friday and are dropped."""
    fields = {"international": "international_price_usd", "india": "india_market_price_inr", "landed": "india_landed_price_inr"}
    out: dict[tuple[str, str], dict[date, float]] = {}
    for row in rows:
        try:
            day = date.fromisoformat(str(row["date"]))
        except (KeyError, ValueError):
            continue
        if day.weekday() >= 5:
            continue
        for source, field in fields.items():
            if source != "india" and row.get("international_stale"):
                continue
            value = row.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                out.setdefault((str(row.get("commodity")), source), {})[day] = float(value)
    return {key: sorted(values.items()) for key, values in out.items()}


def aligned(pair: dict[str, str], series: dict) -> tuple[list[date], np.ndarray, np.ndarray]:
    if pair["source"] == "parity":
        a, b = series.get((pair["a"], "india"), []), series.get((pair["b"], "landed"), [])
    else:
        a, b = series.get((pair["a"], pair["source"]), []), series.get((pair["b"], pair["source"]), [])
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
        out.append({"date": days[t], "i": t, "beta": beta, "spread": float(window[-1]), "z": float((window[-1] - window.mean()) / sd)})
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


def _state(pair: dict[str, str], z: float) -> dict[str, Any]:
    if abs(z) < ENTRY_Z:
        return {"stretched": False, "rich": None, "cheap": None}
    first, second = pair["label"].replace(" (MCX)", "").split(" / ")
    rich, cheap = (first, second) if z > 0 else (second, first)
    return {"stretched": True, "rich": rich, "cheap": cheap}


def track_pair(pair: dict[str, str], series: dict) -> Optional[dict[str, Any]]:
    """The tracker row for one pair: today's hedge, z-score, diagnostics and a replay."""
    fixed = pair["source"] == "parity"
    days, la, lb = aligned(pair, series)
    if len(days) < MIN_SESSIONS:
        return {"key": pair["key"], "label": pair["label"], "why": pair["why"], "source": pair["source"],
                "sessions": len(days), "note": f"{len(days)} sessions of history; tracking starts at {MIN_SESSIONS}."}
    hist = history(days, la, lb, fixed=fixed)
    last = hist[-1]
    replay = simulate(days, la, lb, hist)
    return {
        "key": pair["key"], "label": pair["label"], "why": pair["why"], "source": pair["source"],
        "sessions": len(days), "data_from": days[0].isoformat(), "date": last["date"].isoformat(),
        "short_history": len(days) < FULL_HISTORY,
        "ratio": round(math.exp(la[-1] - lb[-1]), 6),
        "beta": round(last["beta"], 4), "hedge": "fixed" if fixed else ("ratio" if last["beta"] == 1.0 else "regression"),
        "z": round(last["z"], 3),
        "z_recent": [round(row["z"], 3) for row in hist[-Z_WINDOW:]],
        **_state(pair, last["z"]),
        **diagnostics(la, lb, fixed=fixed),
        "replay": summarize(replay),
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
        for h in row["_hist"]:
            if h["i"] <= entry_i:
                continue
            held = h["i"] - entry_i
            move = (la[h["i"]] - event["beta"] * lb[h["i"]]) - event["entry_spread"]
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


def run(rows_in: list[dict], ledger: list[dict], now: datetime) -> tuple[dict, list[dict]]:
    series = price_series(rows_in)
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
        "source": "Tapetide (international benchmark and Indian exchange commodity prices)",
        "rule": {
            "hedge_window": HEDGE_WINDOW, "z_window": Z_WINDOW, "entry_z": ENTRY_Z, "exit_z": EXIT_Z,
            "max_hold": MAX_HOLD, "cost_per_side": COST_PER_SIDE, "min_sessions": MIN_SESSIONS,
        },
        "backtest": BACKTEST,
        "pairs": public,
        "track_record": {
            "by_pair": live,
            "first_signal": min((e["entry_date"] for e in ledger), default=None),
            "recent": [
                {k: e.get(k) for k in ("pair", "entry_date", "entry_z", "status", "exit_date", "sessions", "spread_return")}
                for e in sorted(ledger, key=lambda e: e["entry_date"])[-30:]
            ],
        },
    }
    return payload, ledger
