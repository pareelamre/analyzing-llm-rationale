#!/usr/bin/env python3
"""End-to-end evaluation: does headline sentiment earn a place in the Foresea Fund?

Implements the rule pre-registered in docs/fund.md ("News sentiment evaluation", with its
Amendment 1) without changing it: headlines (fund_news_corpus.py) are scored by each
sentiment model; for each
instrument and trading day, the signal is the mean P(positive) - P(negative) over headlines
seen in the 24 hours before that instrument's close (at least 5 headlines); the target is
the next close-to-close return. The first half of the dates only sets the threshold; the
second half decides GO or NO-GO, with 97.5% intervals bootstrapped over dates.

Usage: fund_finbert_eval.py [--news data/fund_news] [--out reports/fund_finbert_eval]
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import sys
import time as clock
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = {
    "finbert": "ProsusAI/finbert",
    "distilroberta": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
}
MIN_HEADLINES = 5
WINDOW = timedelta(hours=24)
MIN_TEST_DAYS = 150
CI = 97.5
BOOTSTRAPS = 4000
#: instrument -> (headline group, close time UTC, round-trip cost). Gold is judged on the
#: forward record instead (Amendment 1: no historical gold series we may fetch).
INSTRUMENTS = {
    "Nifty 50": ("india_equity", time(10, 0), 0.0005),
    "Nifty Bank": ("india_equity", time(10, 0), 0.0005),
    "S&P 500": ("us_equity", time(20, 0), 0.0005),
    "WTI crude": ("crude", time(18, 30), 0.0010),
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_ATTEMPTS = 4

Scorer = Callable[[Sequence[str]], list[float]]


def fred_closes(series: str, start: date, *, http_get=requests.get, sleep=clock.sleep) -> list[tuple[date, float]]:
    for attempt in range(FRED_ATTEMPTS):
        try:
            resp = http_get(FRED_URL.format(series=series), timeout=120, headers={"User-Agent": "Foresea/fund-eval"})
            resp.raise_for_status()
            text = resp.text
            break
        except requests.RequestException:
            if attempt == FRED_ATTEMPTS - 1:
                raise
            sleep(15 * (attempt + 1))
    out = []
    for row in csv.reader(io.StringIO(text)):
        try:
            day, value = date.fromisoformat(row[0]), float(row[1])
        except (ValueError, IndexError):
            continue
        if day >= start:
            out.append((day, value))
    return out


def load_prices(start: date, end: date) -> dict[str, list[tuple[date, float]]]:
    from analyzing_llm_rationale import index_forecast

    india_start = max(start, index_forecast.HISTORY_START)
    return {
        "Nifty 50": index_forecast.daily_closes("NSE_INDEX|Nifty 50", india_start, end),
        "Nifty Bank": index_forecast.daily_closes("NSE_INDEX|Nifty Bank", india_start, end),
        "S&P 500": fred_closes("SP500", start),
        "WTI crude": fred_closes("DCOILWTICO", start),
    }


def load_headlines(news_dir: Path) -> dict[str, list[tuple[datetime, str]]]:
    out = {}
    for path in sorted(news_dir.glob("[!_]*.jsonl")):
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        out[path.stem] = sorted((datetime.fromisoformat(r["seen"]), r["title"]) for r in rows)
    return out


def _windows(prices: dict[str, list[tuple[date, float]]], headlines: dict[str, list[tuple[datetime, str]]]):
    """(instrument, group, day t, close t, next close, headlines seen in the 24 hours before close t)."""
    for name, (group, close_at, _) in INSTRUMENTS.items():
        series = sorted(prices.get(name) or [])
        items = headlines.get(group) or []
        stamps = [ts for ts, _ in items]
        for (day, close), (_, next_close) in zip(series, series[1:]):
            cutoff = datetime.combine(day, close_at, tzinfo=timezone.utc)
            lo = bisect.bisect_right(stamps, cutoff - WINDOW)
            hi = bisect.bisect_right(stamps, cutoff)
            yield name, group, day, close, next_close, [title for _, title in items[lo:hi]]


def needed_titles(prices: dict[str, list[tuple[date, float]]],
                  headlines: dict[str, list[tuple[datetime, str]]]) -> dict[str, set[str]]:
    """Per group, the headlines that fall in some instrument's window; only these need scoring."""
    need: dict[str, set[str]] = {}
    for _, group, _, _, _, titles in _windows(prices, headlines):
        need.setdefault(group, set()).update(titles)
    return need


def daily_rows(prices: dict[str, list[tuple[date, float]]], headlines: dict[str, list[tuple[datetime, str]]],
               scores: dict[str, dict[str, float]]) -> list[dict]:
    """One row per (instrument, day t) with at least MIN_HEADLINES headlines in the 24 hours
    before close t, the mean sentiment then, and the return from close t to the next close."""
    rows = []
    for name, group, day, close, next_close, titles in _windows(prices, headlines):
        group_scores = scores.get(group) or {}
        window = [group_scores[title] for title in titles if title in group_scores]
        if len(window) < MIN_HEADLINES or close <= 0:
            continue
        rows.append({"instrument": name, "day": day, "signal": float(np.mean(window)), "headlines": len(window),
                     "ret": next_close / close - 1.0, "cost": INSTRUMENTS[name][2]})
    return rows


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate(rows: list[dict], *, seed: int = 0) -> dict:
    """The pre-registered test on one model's rows."""
    days = sorted({r["day"] for r in rows})
    if len(days) < 4:
        return {"decision": "NO-GO", "reason": "too few days", "test_instrument_days": 0}
    split = days[len(days) // 2]
    calib = [r for r in rows if r["day"] < split]
    test = [r for r in rows if r["day"] >= split]
    tau = float(np.median([abs(r["signal"]) for r in calib])) if calib else 0.0

    def stats(sample: list[dict]) -> tuple[float, float, float, int]:
        s = np.array([r["signal"] for r in sample])
        y = np.array([r["ret"] for r in sample])
        pos = np.where(s > tau, 1.0, np.where(s < -tau, -1.0, 0.0))
        traded = pos != 0
        costs = np.array([r["cost"] for r in sample])
        net = pos * y - costs
        long_net = y - costs
        trade_mean = float(net[traded].mean()) if traded.any() else float("nan")
        return _spearman(s, y), trade_mean, float(long_net[traded].mean()) if traded.any() else float("nan"), \
            int(traded.sum())

    ic, trade_mean, long_mean, n_trades = stats(test)
    by_day: dict[date, list[dict]] = {}
    for r in test:
        by_day.setdefault(r["day"], []).append(r)
    test_days = list(by_day)
    rng = np.random.default_rng(seed)
    ics, trades = [], []
    for _ in range(BOOTSTRAPS):
        pick = [r for d in rng.choice(len(test_days), len(test_days)) for r in by_day[test_days[d]]]
        b_ic, b_trade, _, _ = stats(pick)
        ics.append(b_ic)
        trades.append(b_trade)
    lo_q, hi_q = (100 - CI) / 2, 100 - (100 - CI) / 2
    ic_ci = [float(np.nanpercentile(ics, lo_q)), float(np.nanpercentile(ics, hi_q))]
    trade_ci = [float(np.nanpercentile(trades, lo_q)), float(np.nanpercentile(trades, hi_q))]
    checks = {
        "enough_test_days": len(test) >= MIN_TEST_DAYS,
        "correlation_above_zero": ic_ci[0] > 0,
        "rule_net_above_zero": trade_ci[0] > 0,
        "rule_beats_always_long": bool(np.isfinite(trade_mean) and trade_mean > long_mean),
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO-GO", "checks": checks,
        "calibration_days": len({r["day"] for r in calib}), "test_days": len(test_days),
        "test_instrument_days": len(test), "threshold": round(tau, 4),
        "spearman": round(ic, 4), "spearman_ci": [round(v, 4) for v in ic_ci],
        "trades": n_trades, "rule_net_mean": round(trade_mean, 5), "rule_net_ci": [round(v, 5) for v in trade_ci],
        "always_long_net_mean": round(long_mean, 5),
        "by_instrument": {name: {"days": sum(1 for r in test if r["instrument"] == name),
                                 "spearman": round(_spearman(np.array([r["signal"] for r in test if r["instrument"] == name]),
                                                             np.array([r["ret"] for r in test if r["instrument"] == name])), 4)}
                          for name in INSTRUMENTS},
    }


def hf_scorer(model_id: str) -> Scorer:
    from transformers import pipeline

    clf = pipeline("text-classification", model=model_id, top_k=None, truncation=True, device=-1)

    def score(texts: Sequence[str]) -> list[float]:
        out = []
        for i in range(0, len(texts), 64):
            for labels in clf(list(texts[i:i + 64])):
                p = {d["label"].lower(): d["score"] for d in labels}
                out.append(p.get("positive", 0.0) - p.get("negative", 0.0))
            if i and i % 12_800 == 0:
                print(f"  {model_id}: {i}/{len(texts)} scored", flush=True)
        return out
    return score


def run(news_dir: Path, out_base: Path, scorers: dict[str, Scorer],
        prices: Optional[dict] = None) -> dict:
    headlines = load_headlines(news_dir)
    coverage_path = news_dir / "_coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.is_file() else None
    first = min((items[0][0] for items in headlines.values() if items), default=datetime.now(timezone.utc))
    prices = prices if prices is not None else load_prices(first.date(), datetime.now(timezone.utc).date())
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rule": "docs/fund.md, News sentiment evaluation (pre-registered 2026-11-09)",
              "headlines": {g: len(v) for g, v in headlines.items()},
              "collection": coverage,
              "price_days": {k: len(v) for k, v in prices.items()},
              "price_range": {k: [str(min(v)[0]), str(max(v)[0])] if v else None for k, v in prices.items()},
              "source": "Headlines: idleengine/financial-news-multisource (non-commercial research use); "
                        "prices: Upstox (Nifty) and FRED (S&P 500, WTI)",
              "models": {}}
    need = needed_titles(prices, headlines)
    titles = sorted(set().union(*need.values()))
    report["scored_headlines"] = len(titles)
    print(f"{len(titles)} distinct headlines fall in some window", flush=True)
    for name, scorer in scorers.items():
        by_title = dict(zip(titles, scorer(titles)))
        scores = {group: {t: by_title[t] for t in group_titles} for group, group_titles in need.items()}
        result = evaluate(daily_rows(prices, headlines, scores))
        report["models"][name] = {"model": MODELS.get(name, name), **result}
        print(f"{name}: {result['decision']} | {json.dumps({k: result.get(k) for k in ('test_instrument_days', 'spearman', 'spearman_ci', 'rule_net_mean', 'rule_net_ci', 'always_long_net_mean')})}")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--news", type=Path, default=ROOT / "data" / "fund_news")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_finbert_eval")
    args = parser.parse_args()
    run(args.news, args.out, {name: hf_scorer(model_id) for name, model_id in MODELS.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
