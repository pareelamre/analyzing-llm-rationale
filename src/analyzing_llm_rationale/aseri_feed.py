"""Daily geopolitical signals from aseri, as one evidence item for questions on their topics.

aseri (https://exdsgift.github.io/aseri/) publishes, once a day and as static JSON with no key:
ships through the Strait of Hormuz, Bab el-Mandeb, the Suez Canal and the Taiwan Strait (IMF
PortWatch), the Caldara-Iacoviello geopolitical risk index, and Ukraine's air-raid alert
hours. For each it gives the recent values and a probabilistic forecast (Timestral-209M),
scored in public. A question about Iran, the Red Sea, Taiwan, Ukraine or war risk gets the
matching series, so the forecast reasons from what ships and alerts are doing now rather
than from headlines alone.

Terms: aseri's forecasts and scores are CC BY-NC 4.0, and PortWatch data needs the IMF's
permission for commercial use. Foresea is non-commercial (a paper fund that earns nothing),
so both may be used with credit; every item carries aseri's own attribution. Revisit if
Foresea ever charges or runs ads.

Left out on purpose: aseri's Brent forecast (no better than repeating the last price on its
own backtest, and its ranges held 71% of the time against 80% promised; FRED has the price)
and "World War III" Wikipedia views (attention, not events).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from datetime import date
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

API = "https://exdsgift.github.io/aseri/api/v1"
SITE = "https://exdsgift.github.io/aseri/"
USER_AGENT = "Foresea/evidence (+https://foresea.ink)"
TIMEOUT_S = 5
#: aseri updates once a day and asks clients to poll at most once an hour.
CACHE_TTL_S = 3600
FAIL_TTL_S = 600
#: Stop fetching more series once a question has spent this long on aseri.
BUDGET_S = 8.0

#: Series in priority order, each with the question words that call for it. Hints are whole
#: words or phrases of the lower-cased question; "gulf" alone is left out (Gulf of Mexico).
SERIES_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hormuz_transits", ("hormuz", "iran", "iranian", "iran's", "tehran", "khamenei", "persian gulf",
                         "tanker", "tankers")),
    ("bab_el_mandeb_transits", ("houthi", "houthis", "yemen", "red sea", "bab el mandeb", "mandeb")),
    ("suez_transits", ("suez", "red sea", "houthi", "houthis")),
    ("taiwan_strait_transits", ("taiwan", "taiwanese", "taipei", "taiwan strait")),
    ("ua_air_raid_hours", ("ukraine", "ukrainian", "ukraine's", "kyiv", "zelensky", "zelenskyy",
                           "russia", "russian", "russia's", "putin")),
    ("gpr_daily", ("war", "invade", "invades", "invasion", "airstrike", "airstrikes", "missile", "missiles",
                   "nuclear", "military", "geopolitical", "ceasefire", "troops", "iran", "israel", "hormuz",
                   "taiwan", "ukraine", "russia", "houthi", "houthis", "nato", "conflict")),
)
MAX_SERIES = 4
#: The lower and upper quantiles of the range shown: 8 in 10 days should land inside.
RANGE = ("0.1", "0.9")

HttpGet = Callable[..., Any]
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def series_for(question: str) -> list[str]:
    """The aseri series a question touches, most specific first, at most MAX_SERIES."""
    text = " " + " ".join(re.findall(r"[a-z0-9']+", (question or "").lower().replace("-", " "))) + " "
    found = [sid for sid, hints in SERIES_HINTS if any(f" {h} " in text for h in hints)]
    return found[:MAX_SERIES]


def _get_json(path: str, http_get: HttpGet) -> Any:
    """GET one aseri file, cached for an hour; a failure is remembered for FAIL_TTL_S so a
    site outage costs one timeout, not one per question."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(path)
    if hit and isinstance(hit[1], Exception) and now - hit[0] < FAIL_TTL_S:
        raise hit[1]
    if hit and not isinstance(hit[1], Exception) and now - hit[0] < CACHE_TTL_S:
        return hit[1]
    try:
        resp = http_get(f"{API}/{path}", timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT,
                                                                     "Accept": "application/json"})
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        with _cache_lock:
            _cache[path] = (now, exc)
        raise
    with _cache_lock:
        _cache[path] = (now, body)
    return body


def _fmt(x: float) -> str:
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.0f}"
    return f"{x:.1f}".rstrip("0").rstrip(".") if x else "0"


def _day(iso: str) -> str:
    d = date.fromisoformat(iso[:10])
    return f"{d:%b} {d.day}"


def _vs_normal(avg: float, base: float) -> str:
    if not base:
        return ""
    pct = (avg / base - 1) * 100
    if abs(pct) < 5:
        return "about normal"
    return f"{abs(pct):.0f}% {'above' if pct > 0 else 'below'} normal"


def describe(meta: dict, latest: dict, actuals: dict, scores: Optional[dict]) -> str:
    """One plain paragraph for one series: where it is, against normal, and aseri's range."""
    obs = [(d, v) for d, v in zip(actuals.get("date") or [], actuals.get("observation") or []) if v is not None]
    if not obs:
        return ""
    unit = meta.get("unit") or latest.get("unit") or ""
    last_day, last = obs[-1]
    week = [v for _, v in obs[-7:]]
    month = [v for _, v in obs[-30:]]
    avg = sum(week) / len(week)
    base = float(meta.get("baseline") or 0)
    source = (meta.get("source") or {}).get("title") or ""
    level = f"{_fmt(last)} on {_day(last_day)}" if unit.startswith("index") else f"{_fmt(last)} {unit} on {_day(last_day)}"
    parts = [f"{meta.get('title', meta.get('series_id'))} ({source}): {level}; "
             f"7-day average {_fmt(avg)}, {_vs_normal(avg, base)} (normal {_fmt(base)}, "
             f"{meta.get('baseline_label', 'its usual level')}). Last 30 days ranged {_fmt(min(month))}-{_fmt(max(month))}."]
    if meta.get("stale_since") or meta.get("source_status") not in (None, "ok"):
        parts.append(f"Source status {meta.get('source_status')}, stale since {meta.get('stale_since')}; "
                     "aseri holds the last good values.")
    q = latest.get("quantiles") or {}
    dates = latest.get("target_end_date") or []
    lo, mid, hi = q.get(RANGE[0]) or [], q.get("0.5") or [], q.get(RANGE[1]) or []
    steps = [s for s in (meta.get("interval_steps") or []) if 0 < s <= min(len(lo), len(mid), len(hi), len(dates))]
    if steps:
        ahead = "; ".join(f"{_day(dates[s - 1])}: likely {_fmt(lo[s - 1])}-{_fmt(hi[s - 1])}, "
                          f"middle {_fmt(mid[s - 1])}" for s in steps)
        parts.append(f"aseri forecast from data to {_day(latest.get('reference_date') or last_day)} (8 in 10 days should land in the range): "
                     f"{ahead}.")
    if scores:
        live = (scores.get("live") or {}).get("n") or 0
        need = scores.get("min_n_to_display") or 30
        retro = ((scores.get("retrospective") or {}).get("buckets") or {}).get("1-7") or {}
        held = (retro.get("coverage_80") or {}).get("rate")
        record = f"live record {live} scored forecasts" + (f" (shown from {need})" if live < need else "")
        if held is not None:
            record += f"; on its backtest the range held {held:.0%} of the time at 1-7 days ahead"
        parts.append(f"Track record: {record}.")
    return " ".join(parts)


def brief(meta: dict, latest: dict, actuals: dict) -> str:
    """A one-line version for short snippets: the week's level against normal and the far range."""
    obs = [v for v in (actuals.get("observation") or []) if v is not None]
    if not obs:
        return ""
    avg = sum(obs[-7:]) / len(obs[-7:])
    unit = meta.get("unit") or ""
    line = (f"{meta.get('title', meta.get('series_id'))} {_fmt(avg)}"
            + ("" if unit.startswith("index") else f" {unit.split(',')[0]}")
            + f" (7-day avg), {_vs_normal(avg, float(meta.get('baseline') or 0))}")
    q, dates = latest.get("quantiles") or {}, latest.get("target_end_date") or []
    lo, hi = q.get(RANGE[0]) or [], q.get(RANGE[1]) or []
    steps = [s for s in (meta.get("interval_steps") or []) if 0 < s <= min(len(lo), len(hi), len(dates))]
    if steps:
        s = steps[-1]
        line += f"; likely {_fmt(lo[s - 1])}-{_fmt(hi[s - 1])} by {_day(dates[s - 1])}"
    return line


def evidence(question: str, *, http_get: Optional[HttpGet] = None) -> list[dict]:
    """One evidence item with every aseri series the question touches, or [] (never raises)."""
    wanted = series_for(question)
    if not wanted:
        return []
    if http_get is None:
        import requests
        http_get = requests.get
    started = time.monotonic()
    try:
        catalog = _get_json("series.json", http_get)
    except Exception:
        logger.warning("aseri catalog unavailable; continuing without it", exc_info=True)
        return []
    by_id = {s.get("series_id"): s for s in catalog.get("series") or []}
    paragraphs, titles, briefs, newest, credits = [], [], [], "", []
    for sid in wanted:
        meta = by_id.get(sid)
        if not meta or time.monotonic() - started > BUDGET_S:
            continue
        links = meta.get("links") or {}
        try:
            latest = _get_json(links.get("latest") or f"series/{sid}/latest.json", http_get)
            actuals = _get_json(links.get("actuals") or f"series/{sid}/actuals.json", http_get)
        except Exception:
            logger.warning("aseri series %s unavailable; skipping it", sid, exc_info=True)
            continue
        try:
            scores = _get_json(links.get("scores") or f"series/{sid}/scores.json", http_get)
        except Exception:
            scores = None
        text = describe(meta, latest, actuals, scores)
        if not text:
            continue
        paragraphs.append(text)
        titles.append(meta.get("title") or sid)
        briefs.append(brief(meta, latest, actuals))
        newest = max(newest, str(latest.get("reference_date") or meta.get("last_observation_date") or ""))
        credit = (meta.get("license") or {}).get("attribution")
        if credit and credit not in credits:
            credits.append(credit)
    if not paragraphs:
        return []
    licence = (catalog.get("license") or {}).get("name") or "CC BY-NC 4.0"
    model_credit = (catalog.get("license") or {}).get("attribution") or "Forecasts: aseri."
    text = ("Daily geopolitical signals from aseri, measured data with model ranges:\n"
            + "\n".join(f"- {p}" for p in paragraphs)
            + f"\nCredit: {' '.join(credits)} {model_credit} Forecasts and scores {licence}; used non-commercially.")
    return [{
        "title": f"aseri daily signals: {', '.join(titles)}",
        "url": SITE,
        "text": text,
        "summary": "aseri: " + "; ".join(b for b in briefs if b) + ".",
        "source": "aseri",
        "source_channel": "aseri",
        "publish_date": newest,
    }]


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


__all__: Sequence[str] = ("SERIES_HINTS", "series_for", "describe", "brief", "evidence", "clear_cache")
