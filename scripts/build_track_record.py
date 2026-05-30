"""
Precompute Foresea's public track record from REAL resolved forecasts.

A forecast is "resolved" only when the ground-truth outcome is known. The
Metaculus dataset carries that ground truth, and results/ carries the model's
actual predictions — so the join of the two is a genuine, verifiable record of
forecasts that were right or wrong.

This writes `static/track_record.json`, served by the app and the
`/track-record` API route. Re-run whenever results are refreshed:

    python scripts/build_track_record.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.metrics import (  # noqa: E402
    accuracy,
    brier_score,
    ece,
    iter_examples,
    load_targets,
    normalize_answer,
    normalize_confidence,
)

# The deployed app serves gpt-oss-120b / variant0 / temperature 0.0 — the track
# record must reflect exactly that configuration.
MODEL_LABEL = "GPT-OSS-120B"
MODEL_KEY = "gpt-oss-120b"
VARIANT = "variant0_neutral_baseline"
TEMPERATURE_DIR = "temperature_00"
MAX_LOG_ROWS = 300  # embedded sample; aggregate stats use the full resolved set


def _dataset_path() -> Path:
    return next(ROOT.glob("forecasting_qa_news_metaculus_*.json"))


def _results_path() -> Path:
    return (
        ROOT / "results" / MODEL_LABEL / TEMPERATURE_DIR
        / f"results_{VARIANT}.json"
    )


def main() -> int:
    dataset_path = _dataset_path()
    results_path = _results_path()
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return 1

    targets = load_targets(dataset_path)
    dataset = {int(r["id"]): r for r in json.loads(dataset_path.read_text(encoding="utf-8"))}
    rows = json.loads(results_path.read_text(encoding="utf-8"))

    examples, missing = iter_examples(rows, targets)
    if not examples:
        print("No resolved examples found.")
        return 1

    # ── Aggregate calibration over the FULL resolved set ──────────────────────
    acc = accuracy(examples)
    brier = brier_score(examples)
    calib_err = ece(examples, bins=10)

    # ── Reliability curve: bin by predicted P(yes), observed frequency of yes ──
    bins = 10
    bucket = [{"n": 0, "p_sum": 0.0, "yes": 0} for _ in range(bins)]
    for ex in examples:
        idx = min(int(ex.p_yes * bins), bins - 1)
        bucket[idx]["n"] += 1
        bucket[idx]["p_sum"] += ex.p_yes
        bucket[idx]["yes"] += ex.target
    calibration = []
    for i, b in enumerate(bucket):
        if b["n"] == 0:
            continue
        calibration.append({
            "bin": f"{i*10}-{(i+1)*10}%",
            "n": b["n"],
            "avg_predicted": round(b["p_sum"] / b["n"], 4),
            "observed_yes_rate": round(b["yes"] / b["n"], 4),
        })

    # ── Per-question resolved log ─────────────────────────────────────────────
    log = []
    for row in rows:
        rid = row.get("id")
        if rid not in targets:
            continue
        ans = normalize_answer(row.get("predicted_answer"))
        conf = normalize_confidence(row.get("confidence"))
        if ans is None or conf is None:
            continue
        rec = dataset.get(int(rid), {})
        actual = "Yes" if targets[rid] == 1 else "No"
        predicted = ans.capitalize()
        cats = rec.get("categories") or []
        log.append({
            "id": rid,
            "question": rec.get("question") or "",
            "predicted": predicted,
            "confidence": round(conf, 3),
            "actual": actual,
            "correct": predicted.lower() == actual.lower(),
            "category": (cats[0] if isinstance(cats, list) and cats else None),
            "resolve_time": rec.get("resolve_time"),
        })

    # Most recently resolved first; keep an embedded sample for page weight
    log.sort(key=lambda r: (r.get("resolve_time") or ""), reverse=True)
    sample = log[:MAX_LOG_ROWS]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_KEY,
        "variant": VARIANT,
        "temperature": 0.0,
        "methodology": (
            "Backtest on resolved Metaculus questions: each forecast was produced by "
            f"{MODEL_KEY} and scored against the published real-world outcome. "
            "Live user forecasts are not included until their questions resolve."
        ),
        "n_resolved": len(examples),
        "n_missing": missing,
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "ece": round(calib_err, 4),
        "calibration": calibration,
        "log_sample_size": len(sample),
        "log": sample,
    }

    out = ROOT / "static" / "track_record.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Wrote {out.relative_to(ROOT)} — {len(examples)} resolved, "
        f"acc={acc:.3f} brier={brier:.3f} ece={calib_err:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
