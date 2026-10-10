"""One-sided paired bootstrap: P(bootstrap delta <= 0) = p-value that the
variant is NOT worse than V0 on Brier. Vectorized with numpy.

Covers GPT-OSS-120B plus the two local Qwen models, for both the lead-30d
(pre-event cutoff) and oracle (full evidence) conditions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.metrics import (
    Example,
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
MODELS = [
    ("GPT-OSS-120B", "temperature_00"),
    ("Qwen2.5-7b-instruct", "temperature_000"),
    ("Qwen3-32B", "temperature_0"),
]
N_BOOT = 10_000


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
    rng = np.random.default_rng(42)

    for model, temp in MODELS:
        base = ROOT / "results" / model / temp
        for cond, sub in (("lead30 (30d pre-event cutoff)", "lead_30d"), ("oracle (full evidence)", "")):
            paths = {v: base / sub / f"results_{v}.json" for v in VARIANTS}
            if not all(p.exists() for p in paths.values()):
                print(f"\n### {model} / {cond}: SKIPPED (missing variant files)")
                continue
            ex = {v: load_examples(p, targets) for v, p in paths.items()}
            common = sorted(set.intersection(*(set(ex[v]) for v in VARIANTS)))
            n = len(common)
            print(f"\n=== {model} — {cond}, n={n} — one-sided paired bootstrap (V0 vs variant) ===")
            print(f"{'variant':<38} {'brier_V0':>9} {'brier_V':>9} {'delta':>8} {'p(V not worse)':>15}")
            v0 = [ex[VARIANTS[0]][i] for i in common]
            b0 = np.array([(x.p_yes - x.target) ** 2 for x in v0])
            for v in VARIANTS[1:]:
                other = [ex[v][i] for i in common]
                bv = np.array([(x.p_yes - x.target) ** 2 for x in other])
                db = bv - b0
                idx = rng.integers(0, n, size=(N_BOOT, n))
                boot = db[idx].mean(axis=1)
                p = float((boot <= 0).mean())
                print(f"{v:<38} {b0.mean():>9.4f} {bv.mean():>9.4f} {db.mean():>+8.4f} {p:>15.4f}")


if __name__ == "__main__":
    main()
