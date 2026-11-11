#!/usr/bin/env python3
"""The pre-registered NVIDIA Kumo Forecast test (docs/fund.md, "NVIDIA Kumo Forecast test").

Kumo Forecast (nvidia/Kumo-Forecast, Apache 2.0) runs zero-shot, as released: the default plain
mode, a 512-session context and the package's own preprocessing.

- Test A: every 21 sessions it forecasts the Nasdaq-100's and Nifty's log close 21 sessions
  ahead; the baseline is no change.
- Test B: every 20 sessions it forecasts log Brent minus log WTI 20 sessions ahead; the
  baselines are no change and the pair rule's mean reversion.

The score is |Kumo's error| minus |baseline's error|, bootstrapped over decisions. Writes
reports/fund_kumo_eval.json.

Kumo is not a Foresea dependency: pass its checkout with --kumo-dir
(github.com/NVIDIA/Kumo-TS, the Kumo-Forecast folder). Its weights download from Hugging Face on
the first call.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Callable, Optional, Sequence

import requests

ROOT = Path(__file__).resolve().parents[1]
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
#: FRED's firewall drops agents containing "eval"; this one is plain.
USER_AGENT = "Foresea/fund-kumo (+https://foresea.ink)"
SEQ_LEN = 512
INDEX_H = 21
PAIR_H = 20
HEDGE_WINDOW = 250
MEAN_WINDOW = 60
BOOTSTRAP, SEED = 2000, 20261110

#: (dates, values) -> the forecast path, PAIR_H or INDEX_H values long
Forecaster = Callable[[Sequence[date], Sequence[float], int], Sequence[float]]


def decisions(n: int, horizon: int) -> list[int]:
    """Decision days: the first with SEQ_LEN sessions up to and including it, then every horizon
    sessions, while the outcome horizon sessions later is known. They do not overlap."""
    return list(range(SEQ_LEN - 1, n - horizon, horizon))


def mean_reversion(series: Sequence[float], i: int, horizon: int = PAIR_H) -> Optional[float]:
    """The pair rule's bet as a forecast: an OLS fit, on the trailing HEDGE_WINDOW sessions, of the
    horizon-session change on the gap to the MEAN_WINDOW-session mean, applied at day i. Only
    changes complete by day i enter the fit."""
    def gap(j: int) -> float:
        return series[j] - fmean(series[j - MEAN_WINDOW + 1: j + 1])
    xs, ys = [], []
    for j in range(max(MEAN_WINDOW - 1, i - HEDGE_WINDOW + 1), i - horizon + 1):
        xs.append(gap(j))
        ys.append(series[j + horizon] - series[j])
    if len(xs) < 30:
        return None
    mx, my = fmean(xs), fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
    return my + slope * (gap(i) - mx)


def interval(values: Sequence[float]) -> Optional[list[float]]:
    if len(values) < 5:
        return None
    rng = random.Random(SEED)
    draws = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP))
    return [round(draws[int(0.025 * BOOTSTRAP)], 6), round(draws[int(0.975 * BOOTSTRAP) - 1], 6)]


def score(rows: Sequence[dict], model: str, baseline: str) -> dict:
    """Mean of |model error| - |baseline error| (negative favours the model), its interval, and
    both mean absolute errors, over rows that have both."""
    pairs = [(r["actual"], r[model], r[baseline]) for r in rows if r.get(model) is not None and r.get(baseline) is not None]
    diffs = [abs(a - m) - abs(a - b) for a, m, b in pairs]
    return {
        "decisions": len(diffs),
        "mean_abs_diff": round(fmean(diffs), 6) if diffs else None,
        "mean_abs_diff_ci": interval(diffs),
        f"mae_{model}": round(fmean(abs(a - m) for a, m, _ in pairs), 6) if pairs else None,
        f"mae_{baseline}": round(fmean(abs(a - b) for a, _, b in pairs), 6) if pairs else None,
    }


def hit_rate(rows: Sequence[dict], model: str) -> Optional[float]:
    calls = [(r["actual"], r[model]) for r in rows if r.get(model) not in (None, 0.0) and r["actual"] != 0.0]
    return round(sum((a > 0) == (m > 0) for a, m in calls) / len(calls), 4) if calls else None


def run_series(name: str, days: Sequence[date], values: Sequence[float], horizon: int,
               forecast: Forecaster, darr: Optional[Forecaster] = None, pair: bool = False) -> list[dict]:
    rows = []
    for i in decisions(len(values), horizon):
        window_days, window = days[i - SEQ_LEN + 1: i + 1], values[i - SEQ_LEN + 1: i + 1]
        last = values[i]
        row = {"series": name, "day": days[i].isoformat(), "actual": values[i + horizon] - last,
               "kumo": forecast(window_days, window, horizon)[horizon - 1] - last, "no_change": 0.0,
               "drift": (window[-1] - window[0]) / (SEQ_LEN - 1) * horizon}
        if pair:
            row["mean_reversion"] = mean_reversion(values, i, horizon)
        if darr is not None:
            try:
                row["kumo_darr"] = darr(days[: i + 1], values[: i + 1], horizon)[horizon - 1] - last
            except Exception as exc:  # DARR is description only; a failure leaves it out
                row["kumo_darr"] = None
                row["darr_error"] = type(exc).__name__
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"  {name}: {len(rows)} decisions, through {row['day']}", file=sys.stderr, flush=True)
    return rows


def kumo_forecasters(kumo_dir: Path, stub_dir: Path) -> tuple[Forecaster, Forecaster]:
    """Kumo's plain and DARR modes as forecasters, at the package defaults."""
    sys.path[:0] = [str(kumo_dir)]
    try:
        import datasetsforecast  # noqa: F401
    except ImportError:
        # Only Kumo's benchmark-dataset loader imports this at module level; inference never
        # calls it, so a stub stands in rather than installing the package. It is written to
        # disk because DARR's data-loader workers import Kumo afresh in their own processes.
        package = stub_dir / "datasetsforecast"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
        (package / "long_horizon.py").write_text(
            '"""Stub for Kumo-TS inference: only the benchmark-dataset loader uses this."""\n\n\n'
            "class LongHorizon:\n    pass\n")
        sys.path[:0] = [str(stub_dir)]
    import pandas as pd
    from sdk.forecasting import ForecastingConfig, perform_forecasting

    def frame(days: Sequence[date], values: Sequence[float]) -> "pd.DataFrame":
        return pd.DataFrame({"timestamp": pd.to_datetime(list(days)), "target": list(values)})

    def config(horizon: int) -> "ForecastingConfig":
        # num_workers=0 loads data in this process. It changes no number, only speed: with the
        # default 2 workers, every DARR call spawned two processes that re-imported torch.
        return ForecastingConfig(seq_len=SEQ_LEN, forecast_horizon=horizon, num_workers=0)

    def plain(days, values, horizon):
        out = perform_forecasting(df=frame(days, values), config=config(horizon))
        return out["target_forecast"].tolist()

    def darr(days, values, horizon):
        df = frame(days, values)
        out = perform_forecasting(df=df.iloc[-SEQ_LEN:], config=config(horizon), context_df=df)
        return out["target_forecast"].tolist()

    return plain, darr


def fred(series: str) -> list[tuple[date, float]]:
    resp = requests.get(FRED, params={"id": series}, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    out = []
    for row in csv.reader(io.StringIO(resp.text)):
        try:
            out.append((date.fromisoformat(row[0]), float(row[1])))
        except (IndexError, ValueError):
            continue
    return out


def cached(path: Path, fetch: Callable[[], list]) -> list[tuple[date, float]]:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([(d.isoformat(), v) for d, v in fetch()]))
    return [(date.fromisoformat(d), float(v)) for d, v in json.loads(path.read_text())]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kumo-dir", type=Path, required=True, help="the Kumo-Forecast folder of a Kumo-TS checkout")
    ap.add_argument("--index-cache", type=Path, default=ROOT / "data" / "nifty_premium_eval",
                    help="ndx.json and nifty.json from the options backtests")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "kumo_eval")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fund_kumo_eval.json")
    ap.add_argument("--through", type=date.fromisoformat, default=date(2026, 11, 10))
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    plain, darr = kumo_forecasters(args.kumo_dir.resolve(), args.cache.resolve() / "stubs")

    def logs(rows):
        rows = [(d, v) for d, v in rows if d <= args.through and v > 0]
        return [d for d, _ in rows], [math.log(v) for _, v in rows]

    index_rows = []
    for name, file in (("Nasdaq-100", "ndx.json"), ("Nifty 50", "nifty.json")):
        days, values = logs([(date.fromisoformat(d), float(v)) for d, v in json.loads((args.index_cache / file).read_text())])
        index_rows += run_series(name, days, values, INDEX_H, plain, darr)
        print(name, "decisions", sum(r["series"] == name for r in index_rows), flush=True)

    brent = dict(cached(args.cache / "brent.json", lambda: fred("DCOILBRENTEU")))
    wti = dict(cached(args.cache / "wti.json", lambda: fred("DCOILWTICO")))
    # Both prices on the day and both positive (WTI settled below zero on 2020-04-20; a log needs a
    # positive price, so that day cannot enter the spread).
    common = sorted(d for d in set(brent) & set(wti) if d <= args.through and brent[d] > 0 and wti[d] > 0)
    spread = [math.log(brent[d]) - math.log(wti[d]) for d in common]
    pair_rows = run_series("Brent / WTI", common, spread, PAIR_H, plain, darr, pair=True)
    print("Brent / WTI decisions", len(pair_rows), flush=True)

    a = score(index_rows, "kumo", "no_change")
    per_index = {name: score([r for r in index_rows if r["series"] == name], "kumo", "no_change")
                 for name in ("Nasdaq-100", "Nifty 50")}
    a_go = bool(a["mean_abs_diff_ci"] and a["mean_abs_diff_ci"][1] < 0
                and all(v["mean_abs_diff"] is not None and v["mean_abs_diff"] < 0 for v in per_index.values()))
    b = {base: score(pair_rows, "kumo", base) for base in ("no_change", "mean_reversion")}
    b_go = all(v["mean_abs_diff_ci"] and v["mean_abs_diff_ci"][1] < 0 for v in b.values())

    def described(rows):
        return {"hit_rate_kumo": hit_rate(rows, "kumo"), "hit_rate_kumo_darr": hit_rate(rows, "kumo_darr"),
                "hit_rate_drift": hit_rate(rows, "drift"),
                "drift_vs_no_change": score(rows, "drift", "no_change"),
                "kumo_darr_vs_no_change": score(rows, "kumo_darr", "no_change"),
                "darr_failures": sum(1 for r in rows if r.get("darr_error"))}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "docs/fund.md, NVIDIA Kumo Forecast test (pre-registered 2026-11-11)",
        "model": {"repo": "nvidia/Kumo-Forecast", "checkpoint": "run8_best_model_cr.pt (package default)",
                  "mode": "plain, zero-shot", "seq_len": SEQ_LEN, "num_workers": "0 (data loading only)"},
        "data": {name: [next(r["day"] for r in index_rows if r["series"] == name),
                        [r["day"] for r in index_rows if r["series"] == name][-1]] for name in ("Nasdaq-100", "Nifty 50")}
                | {"Brent / WTI": [pair_rows[0]["day"], pair_rows[-1]["day"]]},
        "test_a_indices": {"verdict": "GO" if a_go else "NO-GO", "pooled": a, "per_index": per_index},
        "test_b_pair": {"verdict": "GO" if b_go else "NO-GO", **b},
        "description_only": {"indices": described(index_rows), "pair": described(pair_rows),
                             "pair_mean_reversion_vs_no_change": score(pair_rows, "mean_reversion", "no_change")},
        "decisions": index_rows + pair_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("A", report["test_a_indices"]["verdict"], json.dumps(a), json.dumps(per_index))
    print("B", report["test_b_pair"]["verdict"], json.dumps(b))
    print("description", json.dumps(report["description_only"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
