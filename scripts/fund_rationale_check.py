#!/usr/bin/env python3
"""Check the on-device model's forecasts against their own rationales (docs/fund.md, shadow).

On 2026-11-08 four of the on-device model's forecasts were inverted: each rationale argued
"No" while the number said 98% Yes. This asks a second, independent model whether each
rationale concludes the event is likely, and flags a forecast whose number points the other
way. The checker is Open-Jev 2B (ZefanCai/Open-Jev-2B: an Apache-2.0 LoRA and decision head
on Qwen/Qwen3.5-2B, MIT code) which returns a calibrated yes/no probability without
generating text. It is not a forecaster and is never asked about the event itself.

It runs in shadow: verdicts go to static/fund_rationale_checks.json, which the fund records
and publishes, and nothing is blocked until the flags prove worth trusting.

Usage:
  fund_rationale_check.py --store STORE.duckdb --list-only   # prints pending=N
  fund_rationale_check.py --store STORE.duckdb               # checks pending forecasts
  fund_rationale_check.py --validate CASES.json              # scores the checker on known cases
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "static" / "fund_rationale_checks.json"
FORECAST_MODEL = "qwen3.5-9b-selfhosted"
CHECKER_REPO = "ZefanCai/Open-Jev-2B"
CHECKER_REVISION = "0c7aa498b1627be8da4acf34c863ff0ee0a92785"
#: Forecasts this recent are checked; older ones have been replaced by the nightly pass.
SINCE_HOURS = 48
#: The published file keeps this many checks, newest first.
KEEP = 300
#: Long rationales are cut here; the conclusion is almost always early or explicit.
MAX_RATIONALE_CHARS = 3000
#: A side counts as argued when the checker gives it this much probability.
CLEAR = 0.70
QUESTION = ("Read the forecast question and the analyst's rationale. Does the rationale conclude "
            "that the event in the question is more likely than not to happen?")


def pending_forecasts(store: Path, done: set[str], *, model: str = FORECAST_MODEL,
                      since_hours: float = SINCE_HOURS, now: Optional[datetime] = None) -> list[dict]:
    """The latest snapshot per market from ``model`` with a rationale, not yet checked."""
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(hours=since_hours)).isoformat()
    con = duckdb.connect(str(store), read_only=True)
    rows = con.execute("""
        SELECT key, platform, ident, question, model_probability, snapshot_ts, rationale
        FROM forecast_snapshot
        WHERE model = ? AND rationale IS NOT NULL AND length(trim(rationale)) > 0
          AND model_probability IS NOT NULL AND CAST(snapshot_ts AS VARCHAR) >= ?
        QUALIFY row_number() OVER (PARTITION BY platform, ident ORDER BY snapshot_ts DESC) = 1
    """, [model, since]).fetchall()
    con.close()
    out = []
    for key, platform, ident, question, p, ts, rationale in rows:
        if key in done:
            continue
        out.append({"key": key, "instrument_id": f"{str(platform or 'kalshi').lower()}:{ident}",
                    "question": question or ident, "forecast": float(p), "forecasted_at": str(ts),
                    "rationale": str(rationale)[:MAX_RATIONALE_CHARS]})
    return sorted(out, key=lambda r: r["forecasted_at"])


def verdict(forecast: float, rationale_yes: float) -> str:
    """agrees, contradicts, or unclear (the rationale argues neither side clearly)."""
    if rationale_yes >= CLEAR:
        return "contradicts" if forecast < 0.5 else "agrees"
    if rationale_yes <= 1 - CLEAR:
        return "contradicts" if forecast > 0.5 else "agrees"
    return "unclear"


def open_jev_checker(repo: str = CHECKER_REPO, revision: str = CHECKER_REVISION,
                     device: str = "cpu") -> Callable[[str, str], float]:
    """P(the rationale concludes Yes), from Open-Jev's calibrated noul head."""
    import torch
    from huggingface_hub import snapshot_download
    from jev.api import compile_request, format_response
    from jev.metrics import softmax
    from jev.model import DecisionModel

    checkpoint = Path(snapshot_download(repo, revision=revision, allow_patterns=["package/checkpoint/*",
                                                                                  "package/checkpoint/**"]))
    checkpoint = checkpoint / "package" / "checkpoint"
    temperature = json.loads((checkpoint / "temperature.json").read_text())["temperature"]
    model = DecisionModel.load(checkpoint, device=device)

    def ask(question: str, rationale: str) -> float:
        records = compile_request({"forecast_question": question, "rationale": rationale},
                                  {"concludes_yes": {"type": "noul", "instructions": QUESTION}})
        with torch.inference_mode():
            logits = model(records)
        probs = [softmax(row.float().cpu().tolist(), temperature=temperature) for row in logits]
        return float(format_response(records, probs)["answers"]["concludes_yes"]["noul"])
    return ask


def run(store: Path, out_path: Path, ask: Callable[[str, str], float], *, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    previous = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
    checks = list(previous.get("checks") or [])
    todo = pending_forecasts(store, {c["key"] for c in checks}, now=now)
    for item in todo:
        p_yes = ask(item["question"], item["rationale"])
        checks.append({**{k: item[k] for k in ("key", "instrument_id", "question", "forecast", "forecasted_at")},
                       "rationale_yes": round(p_yes, 4), "verdict": verdict(item["forecast"], p_yes),
                       "checked_at": now.isoformat()})
        print(f"{checks[-1]['verdict']:11} forecast {item['forecast']:.2f} rationale-yes {p_yes:.2f} | "
              f"{item['question'][:70]}")
    checks = sorted(checks, key=lambda c: c["forecasted_at"], reverse=True)[:KEEP]
    payload = {
        "generated_at": now.isoformat(), "forecast_model": FORECAST_MODEL, "mode": "shadow",
        "checker": {"model": CHECKER_REPO, "revision": CHECKER_REVISION, "base": "Qwen/Qwen3.5-2B",
                    "question": QUESTION, "clear_threshold": CLEAR},
        "checks": checks,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"checked": len(todo), "kept": len(checks)}


def validate(cases_path: Path, ask: Callable[[str, str], float]) -> dict:
    """Run the checker on known forecasts (tests/fixtures/rationale_check_cases.json: the
    2026-11-08 inversions and correct forecasts from the same run) and count how many it gets."""
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    results = []
    for case in cases:
        p_yes = ask(case["question"], case["rationale"][:MAX_RATIONALE_CHARS])
        got = verdict(float(case["forecast"]), p_yes)
        results.append({"ident": case["ident"], "forecast": case["forecast"], "rationale_yes": round(p_yes, 4),
                        "expected": case["expected"], "got": got, "right": got == case["expected"]})
        print(f"{'OK ' if got == case['expected'] else 'MISS'} expected {case['expected']:11} got {got:11} "
              f"forecast {float(case['forecast']):.2f} rationale-yes {p_yes:.2f} | {case['ident'][:50]}")
    caught = sum(1 for r in results if r["expected"] == "contradicts" and r["got"] == "contradicts")
    inverted = sum(1 for r in results if r["expected"] == "contradicts")
    false_flags = sum(1 for r in results if r["expected"] == "agrees" and r["got"] == "contradicts")
    summary = {"cases": len(results), "right": sum(r["right"] for r in results),
               "inversions_caught": f"{caught}/{inverted}", "false_flags": false_flags}
    print(json.dumps(summary))
    return {"summary": summary, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--list-only", action="store_true", help="print pending=N and stop (no model needed)")
    parser.add_argument("--validate", type=Path, help="score the checker on known cases and stop")
    args = parser.parse_args()
    if args.validate:
        validate(args.validate, open_jev_checker(device=os.environ.get("JEV_DEVICE", "cpu")))
        return 0
    if not args.store:
        parser.error("--store is required")
    if args.list_only:
        previous = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {}
        n = len(pending_forecasts(args.store, {c["key"] for c in previous.get("checks") or []}))
        print(f"pending={n}")
        return 0
    result = run(args.store, args.out, open_jev_checker(device=os.environ.get("JEV_DEVICE", "cpu")))
    print(json.dumps(result), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
