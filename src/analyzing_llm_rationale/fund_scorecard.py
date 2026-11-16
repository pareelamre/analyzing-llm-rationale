"""The fund's scorecard and allocator (docs/fund.md, "Scorecard and allocator").

Every strategy is scored the same way, as return per dollar at risk per position after costs,
pooling an out-of-sample test reported in that unit with the fund's own closed positions. The
allocator then sizes each analyst from its score:

- proven (enough evidence, low end of the 95% range above zero): a quarter of the Kelly bet at
  that low end, as a multiple of the starting size between 0.5x and 2.5x;
- on trial (anything else still running): half the starting size;
- failed (enough evidence, high end below zero): nothing.

The multiple becomes the analyst's trust (``DEFAULT_TRUST`` x multiple), which is what the
orchestrator already sizes by, so the fund's other limits are untouched.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from analyzing_llm_rationale import fund_orchestrator as orch

Z95 = 1.96
#: A test run on another market than the one the fund trades counts as this share of its trades.
PROXY_WEIGHT = 0.5
#: Fewer counted positions than this decide nothing: the strategy stays on trial.
MIN_POSITIONS = 20
TRIAL_SCALE = 0.5
MIN_PROVEN_SCALE = 0.5
MAX_SCALE = 2.5
#: Share of the fund at risk per position at the starting size (trust DEFAULT_TRUST). Prediction
#: markets size by Kelly per market; 2% stands for a typical starting position there.
STARTING_SIZE = {"option": orch.OPTION_RISK, "pair": 2 * orch.PAIR_RISK, "event": 0.02}

#: The strategies the fund runs, and how each is sized.
STRATEGIES = {
    "put_premium": {"label": "Put premium (Nifty and Bank Nifty put spreads)", "kind": "option",
                    "market": "Index options"},
    "pairs_model": {"label": "Commodity pairs", "kind": "pair", "market": "Commodities"},
    "blend_model": {"label": "Blend analyst (market and model)", "kind": "event", "market": "Prediction markets"},
}

#: The Bank Nifty put-spread test (docs/fund.md, "Bank Nifty put spreads").
BANK_NIFTY_REPORT = "fund_banknifty_put_eval.json"

OPENING = {"open", "add", "transfer"}
CLOSING = {"close", "settle", "expire"}

#: Every idea tested since 2026-11-09 that got no money, with its result (docs/fund.md).
TESTED = (
    {"idea": "On-device model's raw odds", "market": "Prediction markets", "date": "2026-11-09",
     "verdict": "Retired", "analyst": "event_model",
     "result": "Brier 0.0076 worse than the market (95% range 0.0012 to 0.0146) on 116 markets"},
    {"idea": "News sentiment (FinBERT)", "market": "Indices and oil", "date": "2026-11-09", "verdict": "NO-GO",
     "result": "rank correlation with the next day's return 0.005 (range -0.056 to 0.068)"},
    {"idea": "Commodities leading Indian indices", "market": "Indices", "date": "2026-11-10", "verdict": "NO-GO",
     "result": "nothing held up in the 2019-2026 holdout"},
    {"idea": "US VIX fade on Bank Nifty", "market": "Indices", "date": "2026-11-10", "verdict": "Retired",
     "analyst": "vix_fade",
     "result": "about 12 signals a year put a verdict years away, and no same-night price exists to trade it"},
    {"idea": "Nifty straddles and condors", "market": "Index options", "date": "2026-11-10", "verdict": "Retired",
     "analyst": "options_model", "result": "+4.0% per rupee at risk (range -5.7% to +13.6%), short of the bar"},
    {"idea": "Weather forecasts, Kalshi daily highs", "market": "Prediction markets", "date": "2026-11-10",
     "verdict": "Retired", "analyst": "weather_model",
     "result": "-25.2% per dollar after fees (range -32.5% to -17.5%)"},
    {"idea": "Ranges already decided by airport readings", "market": "Prediction markets", "date": "2026-11-10",
     "verdict": "Retired", "analyst": "decided_ranges",
     "result": "1,808 of 1,818 decided ranges were already priced at a full dollar"},
    {"idea": "Weather forecasts, Polymarket daily highs", "market": "Prediction markets", "date": "2026-11-10",
     "verdict": "NO-GO", "result": "-14.0% per dollar after fees (range -21.6% to -4.6%)"},
    {"idea": "NVIDIA Kumo forecasts", "market": "Indices and oil", "date": "2026-11-11", "verdict": "NO-GO",
     "result": "36% larger errors than assuming no change on the indices"},
    {"idea": "HRRR forecasts, Kalshi hourly temperatures", "market": "Prediction markets", "date": "2026-11-12",
     "verdict": "NO-GO", "result": "-44.5% per dollar after fees (range -49.9% to -38.9%)"},
    {"idea": "Market makers' opening prices, hourly markets", "market": "Prediction markets", "date": "2026-11-12",
     "verdict": "NO-GO", "result": "14 loose ladders in 5,816 hours; the model at the open lost 58.9% per dollar"},
    {"idea": "Trend following, indices and commodities", "market": "Indices and commodities", "date": "2026-11-15",
     "verdict": "NO-GO", "result": "+0.3% a year (range -1.9% to +2.7%), Sharpe 0.06"},
)


@dataclasses.dataclass
class Evidence:
    label: str
    trades: int
    mean: float
    sd: float
    weight: float = 1.0  # the share of its trades that counts
    source: str = ""

    @property
    def counted(self) -> float:
        return self.trades * self.weight


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _put_test(root: Path, report: str, label: str) -> Optional[Evidence]:
    test = _load(root / "reports" / report).get("put_spread") or {}
    if not (test.get("trades") and test.get("stdev") is not None and test.get("mean_return") is not None):
        return None
    years = f"{str(test.get('first', ''))[:4]}-{str(test.get('last', ''))[:4]}"
    return Evidence(f"{label}, {years}", int(test["trades"]), float(test["mean_return"]), float(test["stdev"]),
                    PROXY_WEIGHT, f"reports/{report}")


def bank_nifty_on(root: Path) -> bool:
    """Bank Nifty's put spreads trade, and count, only while their own test says GO (docs/fund.md)."""
    return (_load(root / "reports" / BANK_NIFTY_REPORT).get("put_spread") or {}).get("verdict") == "GO"


def together(analyst: str, root: Path) -> Optional[float]:
    """For a strategy that holds positions which move together, their correlation (ρ): the put
    premium sleeve's Nifty and Bank Nifty spreads, from the Bank Nifty test."""
    if analyst != "put_premium" or not bank_nifty_on(root):
        return None
    rho = ((_load(root / "reports" / BANK_NIFTY_REPORT).get("description_only") or {}).get("both_sold") or {}).get("rho")
    return float(rho) if rho is not None and 0 <= float(rho) <= 1 else 1.0


def test_evidence(analyst: str, root: Path) -> tuple[list[Evidence], Optional[str]]:
    """An analyst's out-of-sample tests in return per dollar at risk, and a line of context for
    evidence that does not count (an in-sample backtest, or a test scored in Brier)."""
    if analyst == "put_premium":
        ndx = _put_test(root, "fund_ndx_put_eval.json", "Out-of-sample test on the Nasdaq-100")
        if ndx is None:
            return [], "Its out-of-sample report is missing."
        context = "The Nifty backtest it was designed on is in-sample, so it is not counted."
        bank = _put_test(root, BANK_NIFTY_REPORT, "Bank Nifty test") if bank_nifty_on(root) else None
        if bank is None:
            return [ndx], context
        return [ndx, bank], (f"{context} Bank Nifty counts at half weight: it shares the Nifty backtest's months "
                             f"and much of its movement.")
    if analyst == "pairs_model":
        return [], ("No out-of-sample test yet: the US backtest (2011-2026) chose which pairs to trade, "
                    "so it is not counted.")
    if analyst == "blend_model":
        t = _load(root / "configs" / "fund_blend.json").get("time_split_test") or {}
        if t.get("test_markets"):
            gap = " (95% range of the gap {:+.3f} to {:+.3f})".format(*t["blend_minus_market_ci95"]) \
                if len(t.get("blend_minus_market_ci95") or []) == 2 else ""
            return [], (f"Its out-of-sample test was scored in Brier, not returns: {t.get('blend_brier')} against "
                        f"the market's {t.get('market_brier')} on {t['test_markets']} markets, a tie{gap}.")
        return [], "No out-of-sample test yet."
    return [], None


def closed_positions(conn: sqlite3.Connection, fund_id: str) -> list[dict]:
    """The fund's closed positions, one per position (both legs of a pair are one), with the
    return per dollar at risk after fees and the analyst with the largest attribution weight."""
    rows = conn.execute(
        "SELECT ts, cycle_id, instrument_id, side, action, purpose, pair_key, notional_usd, fee_usd,"
        " realized_pnl_usd, attribution_json FROM fund_trades WHERE fund_id = ? ORDER BY ts, rowid", (fund_id,))
    open_: dict[tuple[str, str], dict] = {}
    done: list[dict] = []
    for ts, cycle, iid, side, action, purpose, pair_key, notional, fee, realized, attribution in rows:
        if purpose == "hedge":
            continue
        book = open_.setdefault((iid, side), {"realized": 0.0, "at_risk": 0.0})
        if action in OPENING:
            book["realized"] += float(realized or 0.0)  # the opening fee, booked as a negative
            continue
        book["realized"] += float(realized or 0.0)
        book["at_risk"] += float(notional or 0.0) - float(realized or 0.0) - float(fee or 0.0)
        if action in CLOSING:
            weights = json.loads(attribution or "{}")
            done.append({"analyst": max(weights, key=weights.get) if weights else "unattributed",
                         "pair_key": pair_key, "closed_at": ts, "cycle_id": cycle,
                         "realized": book["realized"], "at_risk": book["at_risk"]})
            del open_[(iid, side)]
    merged: dict[tuple, dict] = {}
    out = []
    for p in done:
        if not p["pair_key"]:
            out.append(p)
            continue
        key = (p["analyst"], p["pair_key"], p["cycle_id"])
        if key in merged:
            merged[key]["realized"] += p["realized"]
            merged[key]["at_risk"] += p["at_risk"]
        else:
            merged[key] = dict(p)
            out.append(merged[key])
    for p in out:
        p["return"] = p["realized"] / p["at_risk"] if p["at_risk"] > 1e-9 else 0.0
    return [p for p in out if p["at_risk"] > 1e-9]


def pool(evidence: Sequence[Evidence], live: Sequence[float]) -> dict:
    """Pooled mean, standard deviation and 95% range over counted trades."""
    parts = [(e.counted, e.mean, e.sd) for e in evidence if e.counted > 0] + [(1.0, r, 0.0) for r in live]
    n = sum(c for c, _m, _s in parts)
    if n <= 0:
        return {"counted": 0.0, "mean": None, "sd": None, "low": None, "high": None}
    mean = sum(c * m for c, m, _s in parts) / n
    var = sum(c * (s * s + (m - mean) ** 2) for c, m, s in parts) / n
    se = math.sqrt(var / n) if var > 0 else 0.0
    out = {"counted": round(n, 1), "mean": round(mean, 5), "sd": round(math.sqrt(var), 5)}
    if n >= 2 and var > 0:
        out.update(low=round(mean - Z95 * se, 5), high=round(mean + Z95 * se, 5))
    else:
        out.update(low=None, high=None)
    return out


def _size_text(kind: str, scale: float) -> str:
    if scale <= 0:
        return "no money"
    if kind == "option":
        return f"{orch.OPTION_RISK * scale * 100:.1f}% of the fund at risk per spread"
    if kind == "pair":
        return f"{orch.PAIR_RISK * scale * 100:.1f}% of the fund per leg"
    share = orch.KELLY_FRACTION * orch.DEFAULT_TRUST * scale
    return f"1/{round(1 / share)} of the Kelly bet per market" if share > 0 else "no money"


def allocate(score: Mapping[str, Any], kind: str, rho: Optional[float] = None) -> tuple[str, float, Optional[float]]:
    """Status, size multiple and (for a proven strategy) the quarter-Kelly share at the low end.

    ``rho`` is the correlation of two positions the strategy holds at once. For two bets with the
    same edge and risk, the Kelly bet on each is the single bet's divided by (1 + rho)."""
    low, high, sd = score.get("low"), score.get("high"), score.get("sd")
    enough = float(score.get("counted") or 0.0) >= MIN_POSITIONS
    if enough and high is not None and high < 0:
        return "failed", 0.0, None
    if enough and low is not None and low > 0 and sd:
        kelly = orch.KELLY_FRACTION * low / (sd * sd)
        each = kelly / (1 + rho) if rho is not None else kelly
        return "proven", round(min(MAX_SCALE, max(MIN_PROVEN_SCALE, each / STARTING_SIZE[kind])), 4), round(kelly, 5)
    return "trial", TRIAL_SCALE, None


def _why(status: str, score: Mapping[str, Any], kelly: Optional[float], scale: float,
         rho: Optional[float] = None) -> str:
    n = score.get("counted") or 0
    if status == "proven":
        capped = f", above the {MAX_SCALE:g}x limit" if scale >= MAX_SCALE else ""
        if rho is not None:
            return (f"The low end of its range is above zero. A quarter-Kelly bet at that low end is "
                    f"{kelly * 100:.1f}% of the fund at risk. Its two spreads move together (correlation "
                    f"{rho:.2f}), so each gets that divided by {1 + rho:.2f}: "
                    f"{kelly / (1 + rho) * 100:.1f}%{capped}.")
        return (f"The low end of its range is above zero. A quarter-Kelly bet at that low end is "
                f"{kelly * 100:.1f}% of the fund at risk{capped}.")
    if status == "failed":
        return "Even the high end of its range is below zero."
    if n == 0:
        return f"No record yet: it needs {MIN_POSITIONS} positions before its record can decide."
    if n < MIN_POSITIONS:
        return f"{n:g} of the {MIN_POSITIONS} positions it needs before its record can decide."
    return "Its range still spans zero."


def scorecard(conn: Optional[sqlite3.Connection], fund_id: str, root: Path,
              analysts: Iterable[str] = tuple(STRATEGIES)) -> dict:
    """The scorecard: one row per running strategy, its status and size, and the ideas not funded."""
    closed = closed_positions(conn, fund_id) if conn is not None else []
    rows = []
    for analyst in analysts:
        spec = STRATEGIES[analyst]
        tests, context = test_evidence(analyst, root)
        live = [p["return"] for p in closed if p["analyst"] == analyst]
        score = pool(tests, live)
        rho = together(analyst, root)
        status, scale, kelly = allocate(score, spec["kind"], rho)
        live_score = pool([], live)
        rows.append({
            "analyst": analyst, "label": spec["label"], "market": spec["market"], "kind": spec["kind"],
            "status": status, "scale": scale, "trust": round(orch.DEFAULT_TRUST * scale, 4),
            "size": _size_text(spec["kind"], scale), "starting_size": _size_text(spec["kind"], 1.0),
            "kelly_at_low": kelly, "together": rho, "score": score, "why": _why(status, score, kelly, scale, rho),
            "context": context,
            "tests": [{"label": e.label, "trades": e.trades, "counted": e.counted, "mean": e.mean, "sd": e.sd,
                       "source": e.source} for e in tests],
            "live": {"positions": len(live), "mean": live_score["mean"], "low": live_score["low"],
                     "high": live_score["high"],
                     "realized_usd": round(sum(p["realized"] for p in closed if p["analyst"] == analyst), 2)},
        })
    others = {}
    for p in closed:
        if p["analyst"] not in STRATEGIES:
            o = others.setdefault(p["analyst"], {"analyst": p["analyst"], "positions": 0, "realized_usd": 0.0})
            o["positions"] += 1
            o["realized_usd"] = round(o["realized_usd"] + p["realized"], 2)
    return {
        "unit": "return per dollar at risk per position, after costs",
        "rules": {"min_positions": MIN_POSITIONS, "proxy_weight": PROXY_WEIGHT, "trial_scale": TRIAL_SCALE,
                  "scale_range": [MIN_PROVEN_SCALE, MAX_SCALE], "kelly_fraction": orch.KELLY_FRACTION},
        "strategies": rows, "retired_records": sorted(others.values(), key=lambda o: o["analyst"]),
        "tested": [dict(t) for t in TESTED],
    }


def apply(signals: Sequence[orch.Signal], card: Mapping[str, Any]) -> tuple[list[orch.Signal], list[dict]]:
    """Signals resized by the scorecard (an analyst it does not list keeps its trust), and the
    signals of analysts it gives no money, with why."""
    rows = {r["analyst"]: r for r in card.get("strategies") or []}
    out, skipped = [], []
    for s in signals:
        row = rows.get(s.analyst)
        if row is None:
            out.append(s)
        elif row["scale"] <= 0:
            skipped.append({"instrument": s.instrument.get("id"), "label": s.instrument.get("label"),
                            "reason": f"the scorecard gives {row['label']} no money: {row['why']}"})
        else:
            out.append(dataclasses.replace(s, trust=row["trust"]))
    return out, skipped
