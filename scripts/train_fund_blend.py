#!/usr/bin/env python3
"""Train the Foresea Fund's blend analyst on Foresea's own graded forecasts (docs/fund.md).

The blend is a logistic regression on three numbers, logit(market price), logit(model
forecast) and their gap, fitted on every resolved model snapshot in the track-record store.
Each market counts once (its snapshots are weighted 1/n), because a market's daily snapshots
are not independent.

It is tested before it is saved: fit on markets resolved before --cutoff, scored on markets
resolved after, per market, against the market price alone, with a bootstrap over markets. The
2026-11-09 run on 698 markets: the blend's Brier was 0.0011 lower than the market's, 95% CI
[-0.0061, +0.0041], so not yet distinguishable from the market; raw model forecasts were worse
than the market (+0.0076, CI excludes 0); boosted trees overfit (+0.0143) and were dropped.
That is why the blend joins the fund in shadow mode: its predictions are recorded and graded,
not traded, until its live record earns trust.

Usage: python scripts/train_fund_blend.py STORE.duckdb [--cutoff 2026-10-01] [--out configs/fund_blend.json]
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("logit_market", "logit_model", "gap")
EPS = 0.01
BOOTSTRAPS = 2000


def logit(x: float) -> float:
    x = min(max(float(x), EPS), 1 - EPS)
    return math.log(x / (1 - x))


def load(store: Path) -> list[dict]:
    con = duckdb.connect(str(store), read_only=True)
    rows = con.execute("""
        SELECT platform || ':' || ident, model_probability, market_probability, outcome, resolved_ts
        FROM forecast_snapshot
        WHERE resolved AND outcome IS NOT NULL AND key NOT LIKE '%_backfill'
          AND model IS NOT NULL AND model != 'crowd-follow'
          AND model_probability IS NOT NULL AND market_probability IS NOT NULL
    """).fetchall()
    con.close()
    return [{"market": r[0], "p": float(r[1]), "m": float(r[2]), "y": int(r[3]), "resolved": str(r[4])} for r in rows]


def design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["market"]] = counts.get(r["market"], 0) + 1
    x = np.array([[logit(r["m"]), logit(r["p"]), logit(r["p"]) - logit(r["m"])] for r in rows])
    y = np.array([r["y"] for r in rows])
    w = np.array([1.0 / counts[r["market"]] for r in rows])
    return x, y, w


def fit(rows: list[dict]) -> LogisticRegression:
    x, y, w = design(rows)
    return LogisticRegression(C=1.0, max_iter=1000).fit(x, y, sample_weight=w)


def evaluate(train: list[dict], test: list[dict]) -> dict:
    model = fit(train)
    x, y, _ = design(test)
    blend = model.predict_proba(x)[:, 1]
    market = np.array([r["m"] for r in test])
    raw = np.array([r["p"] for r in test])
    by_market: dict[str, list[int]] = {}
    for i, r in enumerate(test):
        by_market.setdefault(r["market"], []).append(i)

    def per_market(p: np.ndarray) -> np.ndarray:
        return np.array([np.mean((p[idx] - y[idx]) ** 2) for idx in by_market.values()])

    base = per_market(market)
    rng = np.random.default_rng(0)
    out = {"test_markets": len(by_market), "test_snapshots": len(test), "market_brier": round(float(base.mean()), 4)}
    for name, p in (("blend", blend), ("model", raw)):
        diff = per_market(p) - base
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOTSTRAPS)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[f"{name}_brier"] = round(float(per_market(p).mean()), 4)
        out[f"{name}_minus_market"] = round(float(diff.mean()), 4)
        out[f"{name}_minus_market_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", type=Path)
    parser.add_argument("--cutoff", default="2026-10-01")
    parser.add_argument("--out", type=Path, default=ROOT / "configs" / "fund_blend.json")
    args = parser.parse_args()

    rows = load(args.store)
    train = [r for r in rows if r["resolved"] < args.cutoff]
    test = [r for r in rows if r["resolved"] >= args.cutoff]
    test_result = evaluate(train, test)
    final = fit(rows)  # the saved weights use every resolved market
    payload = {
        "analyst": "blend_model",
        "kind": "logistic regression on logit(market), logit(model) and their gap; markets weighted equally",
        "features": list(FEATURES),
        "coef": [round(float(c), 6) for c in final.coef_[0]],
        "intercept": round(float(final.intercept_[0]), 6),
        "trained_on": {"markets": len({r["market"] for r in rows}), "snapshots": len(rows),
                       "resolved_through": max(r["resolved"] for r in rows)[:10]},
        "time_split_test": {"cutoff": args.cutoff, **test_result},
        "mode": "shadow",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
