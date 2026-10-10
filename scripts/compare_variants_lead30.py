"""Structured-prompt comparison under a defensible pre-event cutoff (lead 30d).

Computes accuracy / Brier / ECE for V0-V8 at temperature 0.0 on the records
common to all conditions (oracle full-evidence vs 30-day pre-event cutoff),
so each metric is a within-question comparison where only the evidence
cutoff and prompt structure vary.

Writes analysis/lead30_variant_comparison.csv and prints a summary table.

Covers GPT-OSS-120B plus the two local Qwen models.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.metrics import (  # noqa: E402
    Example,
    accuracy,
    brier_score,
    ece,
    load_targets,
    normalize_answer,
    normalize_confidence,
)

DATASET = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
VARIANTS = [
    "variant0_neutral_baseline",
    "variant1_predicted_event",
    "variant2_key_attribute",
    "variant3_reasoning_type",
    "variant4_credibility",
    "variant5_key_conditions",
    "variant6_step_by_step_reasoning",
    "variant7_uncertainty_language",
    "variant8_temporal_anchors",
]
# (model_dir, temp_dir) — the temperature each model's oracle sweep lives under.
MODELS = [
    ("GPT-OSS-120B", "temperature_00"),
    ("Qwen2.5-7b-instruct", "temperature_000"),
    ("Qwen3-32B", "temperature_0"),
]


def load_examples(path: Path, targets: dict) -> dict[int, Example]:
    rows = {}
    for r in json.loads(path.read_text(encoding="utf-8")):
        rid = r.get("id")
        ans = normalize_answer(r.get("predicted_answer"))
        conf = normalize_confidence(r.get("confidence"))
        if rid in targets and ans and conf is not None:
            rows[rid] = Example(ans, conf, targets[rid])
    return rows


def main() -> None:
    targets = load_targets(DATASET)
    out_rows: list[dict] = []

    for model, temp in MODELS:
        base = ROOT / "results" / model / temp
        oracle: dict[str, dict] = {}
        cutoff: dict[str, dict] = {}
        missing = []
        for v in VARIANTS:
            o_path = base / f"results_{v}.json"
            c_path = base / "lead_30d" / f"results_{v}.json"
            if not o_path.exists() or not c_path.exists():
                missing.append(v)
                continue
            oracle[v] = load_examples(o_path, targets)
            cutoff[v] = load_examples(c_path, targets)
        if missing:
            print(f"\n### {model}: SKIPPED — missing {len(missing)} variant files: {missing}")
            continue

        # Records present in BOTH conditions for every variant (strict pairing).
        common = set.intersection(*(set(oracle[v]) & set(cutoff[v]) for v in VARIANTS))
        n = len(common)
        print(f"\n### {model} ({temp}) — common records across 9 variants x 2 conditions: {n}")
        print(f"{'variant':<38} {'acc_oracle':>10} {'acc_30d':>8} {'brier_or':>9} {'brier_30d':>9} {'ece_or':>7} {'ece_30d':>8}")
        for v in VARIANTS:
            ex_o = [oracle[v][i] for i in common]
            ex_c = [cutoff[v][i] for i in common]
            acc_o, acc_c = accuracy(ex_o), accuracy(ex_c)
            br_o, br_c = brier_score(ex_o), brier_score(ex_c)
            ec_o, ec_c = ece(ex_o, 10), ece(ex_c, 10)
            print(f"{v:<38} {acc_o:>10.4f} {acc_c:>8.4f} {br_o:>9.4f} {br_c:>9.4f} {ec_o:>7.4f} {ec_c:>8.4f}")
            out_rows.append({
                "model": model,
                "variant": v,
                "n": n,
                "acc_oracle": round(acc_o, 4),
                "acc_lead30": round(acc_c, 4),
                "brier_oracle": round(br_o, 4),
                "brier_lead30": round(br_c, 4),
                "ece_oracle": round(ec_o, 4),
                "ece_lead30": round(ec_c, 4),
            })

    if not out_rows:
        raise SystemExit("No results computed — check that the sweep results exist.")

    out = ROOT / "analysis" / "lead30_variant_comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
