"""In-depth comparison of the structured-prompt sweep under a 30-day pre-event cutoff.

Produces a full analysis package for the lead-30d condition across all three
models (GPT-OSS-120B, Qwen2.5-7b-instruct, Qwen3-32B):

  1. Per-variant metrics (accuracy, Brier, ECE, log loss, Brier skill vs base rate)
  2. Oracle -> 30d degradation per variant
  3. Paired bootstrap significance vs V0 (one-sided, 10k resamples)
  4. Evidence-filtering statistics (how much evidence the cutoff removed)
  5. Confidence/calibration breakdown by bin
  6. Answer-agreement matrix between variants
  7. Stratified metrics by evidence availability (0 articles kept vs >=1)

Outputs (all under analysis/lead30/):
  - variant_metrics.csv          per model x variant x condition metrics
  - degradation.csv              oracle -> 30d deltas
  - significance.csv             paired bootstrap vs V0
  - evidence_filtering.csv       cutoff evidence statistics
  - calibration_bins.csv         confidence-bin calibration
  - agreement_matrix.csv         pairwise answer agreement
  - evidence_strata.csv          metrics stratified by evidence kept
  - SUMMARY.md                   human-readable write-up
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.metrics import (  # noqa: E402
    Example,
    load_targets,
    normalize_answer,
    normalize_confidence,
)

DATASET = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
OUT = ROOT / "analysis" / "lead30"
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
SHORT = {v: v.split("_", 1)[0].replace("variant", "V") for v in VARIANTS}
MODELS = [
    ("GPT-OSS-120B", "temperature_00"),
    ("Qwen2.5-7b-instruct", "temperature_000"),
    ("Qwen3-32B", "temperature_0"),
]
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_examples(rows: list[dict], targets: dict) -> dict[int, Example]:
    out = {}
    for r in rows:
        rid = r.get("id")
        ans = normalize_answer(r.get("predicted_answer"))
        conf = normalize_confidence(r.get("confidence"))
        if rid in targets and ans and conf is not None:
            out[rid] = Example(ans, conf, targets[rid])
    return out


def log_loss(examples: list[Example]) -> float:
    eps = 1e-12
    total = 0.0
    for ex in examples:
        p = min(max(ex.p_yes, eps), 1 - eps)
        total += -(math.log(p) if ex.target == 1 else math.log(1 - p))
    return total / len(examples)


def brier_skill(examples: list[Example], base_rate: float) -> float:
    """1 - Brier/Brier_ref, where the reference is the constant base-rate forecast."""
    bs = sum((ex.p_yes - ex.target) ** 2 for ex in examples) / len(examples)
    ref = sum((base_rate - ex.target) ** 2 for ex in examples) / len(examples)
    return 1 - bs / ref if ref else float("nan")


def ece_bins(examples: list[Example], bins: int = 10) -> list[dict]:
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [e for e in examples if (lo <= e.confidence < hi) or (b == bins - 1 and e.confidence == 1.0)]
        if not sel:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "mean_conf": "", "accuracy": "", "gap": ""})
            continue
        mc = sum(e.confidence for e in sel) / len(sel)
        ac = sum(e.correct for e in sel) / len(sel)
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": len(sel),
            "mean_conf": round(mc, 4),
            "accuracy": round(ac, 4),
            "gap": round(mc - ac, 4),
        })
    return rows


def paired_bootstrap(a: list[Example], b: list[Example]) -> tuple[float, float]:
    """One-sided p that variant b is NOT worse than a on Brier (p = P(delta <= 0))."""
    n = len(a)
    db = np.array([(y.p_yes - y.target) ** 2 - (x.p_yes - x.target) ** 2 for x, y in zip(a, b)])
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boot = db[idx].mean(axis=1)
    return float(db.mean()), float((boot <= 0).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = load_targets(DATASET)
    base_rate = sum(targets.values()) / len(targets)

    metrics_rows: list[dict] = []
    degrad_rows: list[dict] = []
    signif_rows: list[dict] = []
    evid_rows: list[dict] = []
    calib_rows: list[dict] = []
    agree_rows: list[dict] = []
    strata_rows: list[dict] = []

    for model, temp in MODELS:
        base = ROOT / "results" / model / temp
        oracle, cutoff, raw_cutoff = {}, {}, {}
        for v in VARIANTS:
            oracle[v] = to_examples(load_rows(base / f"results_{v}.json"), targets)
            raw_cutoff[v] = load_rows(base / "lead_30d" / f"results_{v}.json")
            cutoff[v] = to_examples(raw_cutoff[v], targets)

        common = sorted(set.intersection(*(set(oracle[v]) & set(cutoff[v]) for v in VARIANTS)))
        n = len(common)

        # ---- 1. per-variant metrics ----
        for v in VARIANTS:
            for cond, src in (("oracle", oracle), ("lead30", cutoff)):
                ex = [src[v][i] for i in common]
                metrics_rows.append({
                    "model": model,
                    "variant": SHORT[v],
                    "variant_full": v,
                    "condition": cond,
                    "n": n,
                    "accuracy": round(sum(e.correct for e in ex) / n, 4),
                    "brier": round(sum((e.p_yes - e.target) ** 2 for e in ex) / n, 4),
                    "log_loss": round(log_loss(ex), 4),
                    "ece": round(
                        sum(
                            abs(
                                (sum(e.confidence for e in sel) / len(sel))
                                - (sum(e.correct for e in sel) / len(sel))
                            ) * len(sel) / n
                            for sel in [
                                [e for e in ex if (b / 10 <= e.confidence < (b + 1) / 10)
                                 or (b == 9 and e.confidence == 1.0)]
                                for b in range(10)
                            ]
                            if sel
                        ),
                        4,
                    ),
                    "brier_skill_vs_base": round(brier_skill(ex, base_rate), 4),
                    "mean_confidence": round(sum(e.confidence for e in ex) / n, 4),
                    "pct_yes": round(sum(1 for e in ex if e.predicted_label == 1) / n, 4),
                })

        # ---- 2. degradation ----
        for v in VARIANTS:
            o = [oracle[v][i] for i in common]
            c = [cutoff[v][i] for i in common]
            degrad_rows.append({
                "model": model,
                "variant": SHORT[v],
                "acc_delta": round(
                    (sum(e.correct for e in c) / n) - (sum(e.correct for e in o) / n), 4
                ),
                "brier_delta": round(
                    (sum((e.p_yes - e.target) ** 2 for e in c) / n)
                    - (sum((e.p_yes - e.target) ** 2 for e in o) / n),
                    4,
                ),
                "logloss_delta": round(log_loss(c) - log_loss(o), 4),
            })

        # ---- 3. significance vs V0 ----
        v0 = [cutoff[VARIANTS[0]][i] for i in common]
        for v in VARIANTS[1:]:
            other = [cutoff[v][i] for i in common]
            delta, p = paired_bootstrap(v0, other)
            signif_rows.append({
                "model": model,
                "condition": "lead30",
                "variant": SHORT[v],
                "brier_delta_vs_V0": round(delta, 4),
                "p_variant_not_worse": round(p, 4),
                "significant_at_05": "yes" if p < 0.05 else "no",
            })
        v0o = [oracle[VARIANTS[0]][i] for i in common]
        for v in VARIANTS[1:]:
            other = [oracle[v][i] for i in common]
            delta, p = paired_bootstrap(v0o, other)
            signif_rows.append({
                "model": model,
                "condition": "oracle",
                "variant": SHORT[v],
                "brier_delta_vs_V0": round(delta, 4),
                "p_variant_not_worse": round(p, 4),
                "significant_at_05": "yes" if p < 0.05 else "no",
            })

        # ---- 4. evidence filtering (from V0 records; cutoff is variant-independent) ----
        kept, total, fracs = [], [], []
        for r in raw_cutoff[VARIANTS[0]]:
            fc = r.get("forecast_cutoff") or {}
            if fc.get("n_articles_total") is None:
                continue
            nt, nk = fc["n_articles_total"], fc.get("n_articles_kept", 0)
            total.append(nt)
            kept.append(nk)
            if nt:
                fracs.append(nk / nt)
        if kept:
            evid_rows.append({
                "model": model,
                "n_records": len(kept),
                "mean_articles_total": round(float(np.mean(total)), 2),
                "mean_articles_kept": round(float(np.mean(kept)), 2),
                "mean_kept_frac": round(float(np.mean(fracs)), 4) if fracs else "",
                "pct_with_zero_kept": round(100 * sum(1 for k in kept if k == 0) / len(kept), 2),
                "pct_with_any_kept": round(100 * sum(1 for k in kept if k > 0) / len(kept), 2),
                "median_kept": float(np.median(kept)),
                "p25_kept": float(np.percentile(kept, 25)),
                "p75_kept": float(np.percentile(kept, 75)),
            })

        # ---- 5. calibration bins (lead30, V0) ----
        for v in VARIANTS:
            ex = [cutoff[v][i] for i in common]
            for row in ece_bins(ex):
                calib_rows.append({"model": model, "variant": SHORT[v], **row})

        # ---- 6. agreement matrix (lead30) ----
        for i, va in enumerate(VARIANTS):
            for vb in VARIANTS[i + 1:]:
                same = sum(
                    1 for rid in common
                    if cutoff[va][rid].predicted_label == cutoff[vb][rid].predicted_label
                )
                agree_rows.append({
                    "model": model,
                    "variant_a": SHORT[va],
                    "variant_b": SHORT[vb],
                    "agreement": round(same / n, 4),
                })

        # ---- 7. stratified by evidence availability ----
        zero_ids, some_ids = [], []
        for r in raw_cutoff[VARIANTS[0]]:
            fc = r.get("forecast_cutoff") or {}
            if fc.get("n_articles_total") is None:
                continue
            rid = r.get("id")
            if rid not in common:
                continue
            (zero_ids if fc.get("n_articles_kept", 0) == 0 else some_ids).append(rid)
        for v in VARIANTS:
            for label, ids in (("zero_kept", zero_ids), ("some_kept", some_ids)):
                if not ids:
                    continue
                ex = [cutoff[v][i] for i in ids]
                strata_rows.append({
                    "model": model,
                    "variant": SHORT[v],
                    "stratum": label,
                    "n": len(ex),
                    "accuracy": round(sum(e.correct for e in ex) / len(ex), 4),
                    "brier": round(sum((e.p_yes - e.target) ** 2 for e in ex) / len(ex), 4),
                    "mean_confidence": round(sum(e.confidence for e in ex) / len(ex), 4),
                })

    def write(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        p = OUT / name
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p}  ({len(rows)} rows)")

    write("variant_metrics.csv", metrics_rows)
    write("degradation.csv", degrad_rows)
    write("significance.csv", signif_rows)
    write("evidence_filtering.csv", evid_rows)
    write("calibration_bins.csv", calib_rows)
    write("agreement_matrix.csv", agree_rows)
    write("evidence_strata.csv", strata_rows)

    # ---- SUMMARY.md ----
    lines = [
        "# Structured-prompt comparison under a 30-day pre-event cutoff",
        "",
        "All variants (V0-V8) evaluated with evidence restricted to articles published",
        "at least 30 days before each question's knowable time (`--cutoff-reference event_end",
        "--forecast-lead-days 30`), at temperature 0.0. Compared against the retrospective",
        "oracle condition (full evidence up to resolution).",
        "",
        f"Dataset base rate (P(yes)): {base_rate:.4f}",
        "",
    ]
    for model, _ in MODELS:
        mrows = [r for r in metrics_rows if r["model"] == model and r["condition"] == "lead30"]
        orows = {r["variant"]: r for r in metrics_rows if r["model"] == model and r["condition"] == "oracle"}
        if not mrows:
            continue
        n = mrows[0]["n"]
        lines += [
            f"## {model}  (n={n})",
            "",
            "| Variant | Acc (30d) | Brier (30d) | ECE (30d) | Brier (oracle) | Brier delta |",
            "|---|---|---|---|---|---|",
        ]
        for r in mrows:
            o = orows[r["variant"]]
            lines.append(
                f"| {r['variant']} | {r['accuracy']:.4f} | {r['brier']:.4f} | {r['ece']:.4f} "
                f"| {o['brier']:.4f} | {r['brier'] - o['brier']:+.4f} |"
            )
        ev = next((e for e in evid_rows if e["model"] == model), None)
        if ev:
            lines += [
                "",
                f"Evidence filtering: {ev['mean_articles_kept']:.2f} of "
                f"{ev['mean_articles_total']:.2f} articles kept on average "
                f"({ev['mean_kept_frac']*100:.1f}%); "
                f"{ev['pct_with_zero_kept']:.1f}% of questions had **zero** evidence kept.",
                "",
            ]
        sig = [s for s in signif_rows if s["model"] == model and s["condition"] == "lead30"]
        worse = [s for s in sig if s["significant_at_05"] == "yes" and s["brier_delta_vs_V0"] > 0]
        better = [s for s in sig if s["significant_at_05"] == "yes" and s["brier_delta_vs_V0"] < 0]
        lines.append(
            f"vs V0 under the cutoff: {len(worse)} variant(s) significantly worse, "
            f"{len(better)} significantly better (p<0.05, paired bootstrap)."
        )
        if worse:
            lines.append(f"  - worse: {', '.join(s['variant'] for s in worse)}")
        if better:
            lines.append(f"  - better: {', '.join(s['variant'] for s in better)}")
        lines.append("")

    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
