#!/usr/bin/env python3
"""Export resolved track-record snapshots as an autoresearch benchmark.

The evolution loop's learning signal: real prediction-market questions that have
since resolved become a `{id, question, answer}` dataset the autoresearch harness
can optimize a candidate prompt against (`run_autoresearch_once`). One record per
resolved market for the primary model, with the market context attached.

Guardrail — time split: pass ``--until YYYY-MM-DD`` to hold out everything that
resolved on/after a date, so a prompt is never optimized and scored on the same
markets. Promotion stays manual/gated (autoresearch only promotes on a measured
improvement); this script just produces the benchmark.

Usage:
  python scripts/export_benchmark.py [--store data/track_record_store.json]
      [--model gpt-oss-120b] [--out analysis/autoresearch/benchmark_resolved.json]
      [--until 2026-06-01]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.trackrec_store import FileStore  # noqa: E402


def _snapshot_date(e: dict) -> str:
    return str(e.get("snapshot_date") or "")


def export(store_path: Path, model: str, out_path: Path, until: str | None) -> int:
    store = FileStore(store_path, autosave=False)
    by_market: dict[tuple, dict] = {}
    for (kind, _id), e in store.items():
        if kind != "ForecastSnapshot":
            continue
        if not e.get("resolved") or e.get("outcome") is None:
            continue
        if (e.get("model") or model) != model:
            continue
        if until and _snapshot_date(e) >= until:
            continue  # time-split holdout
        # One record per market (snapshots of a market share the outcome).
        by_market.setdefault((e.get("platform"), e.get("ident")), e)

    records = []
    for i, e in enumerate(sorted(by_market.values(), key=_snapshot_date)):
        records.append({
            "id": i,
            "question": e.get("question") or "",
            "answer": "yes" if int(e["outcome"]) == 1 else "no",
            "market_platform": e.get("platform"),
            "market_url": e.get("market_url"),
            "market_probability": e.get("market_probability"),
            "news_articles": [],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {len(records)} resolved records to {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=ROOT / "data" / "track_record_store.json")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "autoresearch" / "benchmark_resolved.json")
    ap.add_argument("--until", default=None, help="Hold out markets resolved on/after this YYYY-MM-DD.")
    args = ap.parse_args()
    if not args.store.exists():
        print(f"store not found: {args.store}", file=sys.stderr)
        return 1
    return export(args.store, args.model, args.out, args.until)


if __name__ == "__main__":
    raise SystemExit(main())
