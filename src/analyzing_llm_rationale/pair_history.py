"""Where each commodity pair's price ratio sits in its long history, and the story behind its extremes.

Owner, 2026-11-17: apply context, including the narrative, to the extremes seen in each pair's
relative value, for better forecasts. The daily tracker (commodity_pairs) scores a pair against
its last 60 sessions. This module adds the long view, per pair:

- **Position:** today's price ratio against its trailing 10 years of monthly averages (a z-score,
  a percentile, the range). Monthly prices are the IMF's metals benchmarks and FRED's daily oil
  and gas spot prices averaged by month, both from FRED and free of keys. Today's ratio is the
  latest monthly one moved by the tracker's own daily ratio since then, so MCX and LME price
  levels never mix.
- **Analogues:** past months on the same side at least EXTREME_Z out, and how often the ratio
  moved back toward its 10-year average within 3 and 12 months. Over 1992-2026 that happened
  60% of the time across six pairs, against 52-56% for all months.
- **Episodes and their stories:** runs of months at least EPISODE_Z out, joined to the stories in
  data/pair_episodes.json: what drove each one and how it ended. Each story's driver is
  "temporary" (an outage, a storm, a squeeze, a crisis, a bottleneck) or "lasting" (a change in how
  the commodity is produced or used). Ended episodes are tallied by driver: on 1992-2026 prices
  temporary causes came back to normal within two years far more often than lasting ones. A
  driver is judged from what was known at the time, but the stories were written afterwards,
  so the tally is context, not a tested rule.

Nothing here changes a trade: the fund shows it on each pair trade as evidence.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import requests

from . import commodity_pairs as cp

#: IMF monthly benchmark prices on FRED (USD a tonne), from 1992.
IMF_MONTHLY = {"copper": "PCOPPUSDM", "aluminium": "PALUMUSDM", "zinc": "PZINCUSDM", "lead": "PLEADUSDM",
               "nickel": "PNICKUSDM"}
#: The oil and gas legs use the tracker's own FRED daily series, averaged by month, from this date.
DAILY_FROM = "1997-01-01"
WINDOW_MONTHS = 120
EXTREME_Z = 1.5
EPISODE_Z = 2.0
#: Months below EPISODE_Z that still count as one episode.
EPISODE_GAP = 3
BACK_WITHIN = 24
EPISODES_PATH = Path(__file__).resolve().parents[2] / "data" / "pair_episodes.json"

HttpGet = Callable[..., Any]


def _fred_csv(series_id: str, *, http_get: HttpGet, start: Optional[str] = None) -> list[tuple[str, float]]:
    params = {"id": series_id, **({"cosd": start} if start else {})}
    text = cp._get(cp.FRED_CSV_URL, params=params, accept="text/csv", http_get=http_get).text
    rows = []
    for line in text.splitlines()[1:]:
        day, _, value = line.partition(",")
        try:
            v = float(value)
        except ValueError:
            continue
        if math.isfinite(v) and v > 0:
            rows.append((day.strip(), v))
    return rows


def fetch_monthly(today: date, *, http_get: HttpGet = requests.get) -> dict[str, list[tuple[str, float]]]:
    """Commodity -> [(YYYY-MM, monthly average price)], complete months only."""
    current = today.isoformat()[:7]
    out: dict[str, list[tuple[str, float]]] = {}
    for commodity, series_id in IMF_MONTHLY.items():
        out[commodity] = [(day[:7], v) for day, v in _fred_csv(series_id, http_get=http_get) if day[:7] < current]
    for commodity, series_id in cp.FRED_SERIES.items():
        months: dict[str, list[float]] = defaultdict(list)
        for day, v in _fred_csv(series_id, http_get=http_get, start=DAILY_FROM):
            months[day[:7]].append(v)
        out[commodity] = sorted((m, float(np.mean(v))) for m, v in months.items() if m < current)
    return out


def monthly_ratio(pair: Mapping[str, str], monthly: Mapping[str, Sequence[tuple[str, float]]]
                  ) -> tuple[list[str], np.ndarray]:
    a, b = dict(monthly.get(pair["a"]) or []), dict(monthly.get(pair["b"]) or [])
    months = sorted(set(a) & set(b))
    return months, np.log([a[m] / b[m] for m in months]) if months else np.array([])


def _z(r: np.ndarray, t: int) -> float:
    w = r[t + 1 - WINDOW_MONTHS:t + 1]
    sd = float(np.std(w, ddof=1))
    return float((r[t] - w.mean()) / sd) if sd > 0 else 0.0


def now_log_ratio(months: Sequence[str], r: np.ndarray, days: Sequence[date], daily: np.ndarray) -> Optional[float]:
    """Today's log ratio: the newest month with daily prices, moved by the daily ratio since then."""
    by_month: dict[str, list[float]] = defaultdict(list)
    for d, x in zip(days, daily):
        by_month[d.isoformat()[:7]].append(float(x))
    for i in range(len(months) - 1, -1, -1):
        if months[i] in by_month:
            return float(r[i] + daily[-1] - np.mean(by_month[months[i]]))
    return None


def detect_episodes(months: Sequence[str], r: np.ndarray) -> list[dict[str, Any]]:
    """Runs of months at least EPISODE_Z from the trailing 10-year average, with what followed each peak."""
    found, cur = [], None
    for t in range(WINDOW_MONTHS - 1, len(r)):
        z = _z(r, t)
        if abs(z) < EPISODE_Z:
            continue
        side = 1 if z > 0 else -1
        if cur and cur["side"] == side and t - cur["end"] <= EPISODE_GAP:
            cur["end"] = t
            if abs(z) > abs(cur["z"]):
                cur.update(peak=t, z=z)
        else:
            cur = {"side": side, "start": t, "end": t, "peak": t, "z": z}
            found.append(cur)
    out = []
    for e in found:
        p = e["peak"]
        w = r[p + 1 - WINDOW_MONTHS:p + 1]
        mean, sd = float(w.mean()), float(np.std(w, ddof=1))
        back = next((k for k in range(1, len(r) - p) if abs(r[p + k] - mean) <= sd), None)
        after12 = float(math.expm1(r[p + 12] - r[p])) if p + 12 < len(r) else None
        out.append({
            "side": "high" if e["side"] > 0 else "low", "from": months[e["start"]], "to": months[e["end"]],
            "peak": months[p], "ratio": round(math.exp(r[p]), 4), "usual": round(math.exp(mean), 4),
            "z": round(e["z"], 2), "after_12m": None if after12 is None else round(after12, 4),
            "months_to_normal": back, "settled": p + BACK_WITHIN < len(r),
        })
    return out


def analogues(months: Sequence[str], r: np.ndarray, z_now: float) -> Optional[dict[str, Any]]:
    """How the ratio moved after past months this far out on the same side."""
    if abs(z_now) < EXTREME_Z:
        return None
    side = 1 if z_now > 0 else -1
    moves: dict[int, list[float]] = {3: [], 12: []}
    count = 0
    for t in range(WINDOW_MONTHS - 1, len(r)):
        z = _z(r, t)
        if abs(z) < EXTREME_Z or (z > 0) != (side > 0):
            continue
        count += 1
        for h in moves:
            if t + h < len(r):
                moves[h].append(-side * float(r[t + h] - r[t]))
    out: dict[str, Any] = {"months": count}
    for h, m in moves.items():
        out[f"back_{h}m"] = round(float(np.mean(np.array(m) > 0)), 3) if m else None
        out[f"median_{h}m"] = round(float(math.expm1(np.median(m))), 4) if m else None
    return out


def load_stories(path: Path = EPISODES_PATH) -> dict[str, list[dict[str, Any]]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("pairs") or {}
    except (OSError, ValueError):
        return {}


def _story_for(episode: Mapping[str, Any], stories: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for s in stories:
        if s["side"] == episode["side"] and s["from"] <= episode["peak"] and (s["to"] is None or episode["peak"] <= s["to"]):
            return s
    return None


def join_stories(episodes: Sequence[dict[str, Any]], stories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per story (its strongest month), plus any episode no story covers yet."""
    rows: dict[Any, dict[str, Any]] = {}
    for e in episodes:
        story = _story_for(e, stories)
        key = (story["side"], story["from"]) if story else (e["side"], e["from"])
        if key not in rows or abs(e["z"]) > abs(rows[key]["z"]):
            rows[key] = {**e, "story": dict(story) if story else None}
    return sorted(rows.values(), key=lambda e: e["peak"])


def driver_record(by_pair: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, int]]:
    """Across pairs: how many ended episodes of each driver were back to normal within BACK_WITHIN months."""
    out: dict[str, dict[str, int]] = {}
    for episodes in by_pair.values():
        for e in episodes:
            story = e.get("story")
            if not story or not e["settled"]:
                continue
            row = out.setdefault(story["driver"], {"episodes": 0, "back": 0})
            row["episodes"] += 1
            row["back"] += int(e["months_to_normal"] is not None and e["months_to_normal"] <= BACK_WITHIN)
    return out


def pair_context(pair: Mapping[str, str], monthly: Mapping[str, Sequence[tuple[str, float]]],
                 days: Sequence[date], daily: np.ndarray, stories: Sequence[Mapping[str, Any]]
                 ) -> Optional[dict[str, Any]]:
    months, r = monthly_ratio(pair, monthly)
    if len(r) < WINDOW_MONTHS + 1 or len(daily) == 0:
        return None
    now = now_log_ratio(months, r, days, daily)
    if now is None:
        return None
    w = r[-WINDOW_MONTHS:]
    mean, sd = float(w.mean()), float(np.std(w, ddof=1))
    z = (now - mean) / sd if sd > 0 else 0.0
    episodes = join_stories(detect_episodes(months, r), stories)
    side = "high" if z > 0 else "low"
    current = next((s for s in stories if s["to"] is None and s["side"] == side), None) if abs(z) >= EXTREME_Z else None
    return {
        "source": "IMF monthly prices (FRED)" if pair["a"] in IMF_MONTHLY else "FRED daily spot prices, monthly averages",
        "data_from": months[0], "data_to": months[-1], "window_months": WINDOW_MONTHS,
        "ratio": round(math.exp(now), 4), "usual": round(math.exp(mean), 4),
        "low": round(math.exp(float(w.min())), 4), "high": round(math.exp(float(w.max())), 4),
        "z": round(z, 3), "percentile": round(float((w < now).mean()), 3),
        "extreme": abs(z) >= EXTREME_Z, "analogues": analogues(months, r, z),
        "story": dict(current) if current else None, "episodes": episodes,
    }


def contexts(series: Mapping, monthly: Mapping[str, Sequence[tuple[str, float]]], *,
             stories: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None) -> dict[str, Any]:
    """Long-run context for every pair with a long history, and the record of each kind of cause."""
    stories = load_stories() if stories is None else stories
    out: dict[str, Any] = {}
    for pair in cp.PAIRS:
        if pair["source"] == "parity" or pair["a"] not in monthly or pair["b"] not in monthly:
            continue
        days, la, lb = cp.aligned(pair, series)
        ctx = pair_context(pair, monthly, days, la - lb, stories.get(pair["key"]) or [])
        if ctx is not None:
            out[pair["key"]] = ctx
    record = driver_record({k: v["episodes"] for k, v in out.items()})
    return {"pairs": out, "driver_record": record}

