"""Walk the index forecast forward over 2016-2026 and score it overall and per index.

    python scripts/index_forecast_backtest.py [--cache closes.json] [--json out.json]

A forecast every fifth Nifty 50 session from 2016: each index is fit on its own closes through
that day (index_forecast.fit_index), the month-ahead shape is pooled across the indices fit
that day as run() does, and each horizon is scored on the close h sessions later exactly like
the live ledger (score_ledger / summarize_ledger). Prints index_forecast.BACKTEST's overall rows
and a row per index, so the published per-index record (BACKTEST["by_index"]) can be checked.
Data: Upstox daily closes and India VIX (index_forecast.fetch_market_data).
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import index_forecast as ixf  # noqa: E402

START = date(2016, 1, 1)
STEP = 5


def load(cache: Path | None) -> dict:
    if cache and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        data = ixf.fetch_market_data(datetime.now(timezone.utc))
        raw = {"levels": {k: [[d.isoformat(), c] for d, c in v] for k, v in data["levels"].items()},
               "vix": [[d.isoformat(), c] for d, c in data["vix"]]}
        if cache:
            cache.write_text(json.dumps(raw), encoding="utf-8")
    return {"levels": {k: [(date.fromisoformat(d), c) for d, c in v] for k, v in raw["levels"].items()},
            "vix": [(date.fromisoformat(d), c) for d, c in raw["vix"]]}


def walk(data: dict) -> list[dict]:
    levels, vix = data["levels"], data["vix"]
    days = {name: [d for d, _ in series] for name, series in levels.items()}
    anchor = [d for d in days["Nifty 50"] if d >= START][::STEP]
    rows = []
    for t in anchor:
        fits = {}
        for name, series in levels.items():
            cut = bisect.bisect_right(days[name], t)
            if cut == 0 or days[name][cut - 1] != t:
                continue
            fit = ixf.fit_index(series[:cut], vix[:bisect.bisect_right([d for d, _ in vix], t)])
            if fit:
                fits[name] = (fit, cut)
        shapes = ixf.pooled_shapes(f for f, _ in fits.values())
        for name, (fit, cut) in fits.items():
            forecast = ixf.build_forecast(name, fit, [], shapes)
            for row in ixf.ledger_rows(forecast, datetime.now(timezone.utc)):
                target = cut + row["h"] - 1
                if target >= len(levels[name]):
                    continue
                rows.append(row)
    ixf.score_ledger(rows, levels)
    return [r for r in rows if r.get("realized") is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, help="read closes from here, or save them here after fetching")
    parser.add_argument("--json", type=Path, help="write the summary here")
    args = parser.parse_args()
    rows = walk(load(args.cache))
    names = [name for name, _ in ixf.INDICES]
    out = {
        "window": f"{START.year}-{max(r['realized_date'] for r in rows)[:4]}",
        "by_horizon": ixf.summarize_ledger(rows),
        "by_index": {name: ixf.summarize_ledger([r for r in rows if r["name"] == name]) for name in names},
        "first": {name: min((r["as_of"] for r in rows if r["name"] == name), default=None) for name in names},
    }
    text = json.dumps(out, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
