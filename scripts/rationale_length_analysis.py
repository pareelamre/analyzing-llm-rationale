#!/usr/bin/env python3
"""
Rationale length analysis — addresses reviewer concern that structured prompts
produce longer rationales which LLM judges may reward independently of quality.

Outputs (analysis/length_analysis/):
  length_by_variant.csv         — mean/median word count per model × variant
  length_by_model.csv           — mean word count per model (across variants)
  length_judge_correlations.csv — Spearman ρ between length and each judge score
  partial_effects.csv           — OLS variant effects on judge scores before/after
                                  controlling for length (β_variant raw vs partial)
  length_analysis_report.md     — human-readable summary
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS_ROOT = ROOT / "results"
GEMMA_JUDGE_DIR = ROOT / "analysis" / "llm_judge_rationale_eval_gemma" / "gemma-4-31b-it"
KIMI_JUDGE_DIR = ROOT / "analysis" / "llm_judge_rationale_eval_kimi" / "kimi-k2.5"
OUTPUT_DIR = ROOT / "analysis" / "length_analysis"

JUDGE_ATTRIBUTES = [
    "plausibility", "completeness", "source_consistency",
    "non_hallucination", "informativeness", "conciseness",
]

# model label → (results_dir, temperature_subdir, judge_file_stem)
MODEL_MAP = {
    "Qwen2.5-7b-instruct": ("Qwen2.5-7b-instruct", "temperature_000", "Qwen2.5-7b-instruct__temperature_000"),
    "Qwen3-32B":           ("Qwen3-32B",            "temperature_0",   "Qwen3-32B__temperature_0"),
    "GPT-OSS-120B":        ("GPT-OSS-120B",          "temperature_025", "GPT-OSS-120B__temperature_025"),
}

VARIANTS_ORDERED = [
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


# ---------- helpers ----------

def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def char_count(text: str) -> int:
    return len(text) if text else 0


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    denom = n * (n ** 2 - 1)
    return 1 - 6 * d2 / denom if denom else float("nan")


def _ranks(vals: list[float]) -> list[float]:
    indexed = sorted(enumerate(vals), key=lambda t: t[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def ols_simple(x: list[float], y: list[float]) -> tuple[float, float]:
    """Returns (slope, intercept) for y ~ x."""
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    mx, my = mean(x), mean(y)
    ss_xx = sum((xi - mx) ** 2 for xi in x)
    ss_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if ss_xx == 0:
        return float("nan"), float("nan")
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    return slope, intercept


def partial_corr_controlling_length(
    length_vals: list[float],
    variant_indicator: list[float],
    score_vals: list[float],
) -> float:
    """
    Partial effect of variant on score after removing linear effect of length.
    Returns the OLS coefficient for variant_indicator in:
        score ~ intercept + β1*length + β2*variant_indicator
    using the Frisch-Waugh theorem (residualise both on length).
    """
    n = len(score_vals)
    if n < 4:
        return float("nan")
    slope_l_s, int_l_s = ols_simple(length_vals, score_vals)
    slope_l_v, int_l_v = ols_simple(length_vals, variant_indicator)
    if math.isnan(slope_l_s) or math.isnan(slope_l_v):
        return float("nan")
    resid_s = [s - (slope_l_s * l + int_l_s) for s, l in zip(score_vals, length_vals)]
    resid_v = [v - (slope_l_v * l + int_l_v) for v, l in zip(variant_indicator, length_vals)]
    slope, _ = ols_simple(resid_v, resid_s)
    return slope


# ---------- data loading ----------

def load_results(model_dir: str, temp_dir: str) -> dict[int, dict[str, dict]]:
    """Returns {id: {variant: result_record}}"""
    base = RESULTS_ROOT / model_dir / temp_dir
    out: dict[int, dict[str, dict]] = {}
    for variant in VARIANTS_ORDERED:
        path = base / f"results_{variant}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("results", [])
        for rec in records:
            qid = rec.get("id")
            if qid is None:
                continue
            out.setdefault(qid, {})[variant] = rec
    return out


def load_judge(judge_dir: Path, stem: str) -> dict[int, dict[str, dict[str, float]]]:
    """Returns {id: {variant: {attr: score}}}"""
    path = judge_dir / f"{stem}.jsonl"
    out: dict[int, dict[str, dict[str, float]]] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            rows = parsed if isinstance(parsed, list) else [parsed]
            for row in rows:
                if isinstance(row, dict) and "id" in row:
                    out[row["id"]] = row.get("variant_scores", {})
    return out


# ---------- main ----------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all records: (model, variant, qid, word_count, char_count, judge_scores)
    all_rows: list[dict] = []

    for model_label, (model_dir, temp_dir, judge_stem) in MODEL_MAP.items():
        print(f"Loading {model_label}...")
        results = load_results(model_dir, temp_dir)
        gemma = load_judge(GEMMA_JUDGE_DIR, judge_stem)
        kimi = load_judge(KIMI_JUDGE_DIR, judge_stem)

        for qid, variant_results in results.items():
            g_scores = gemma.get(qid, {})
            k_scores = kimi.get(qid, {})
            for variant, rec in variant_results.items():
                rationale = rec.get("rationale") or ""
                wc = word_count(rationale)
                cc = char_count(rationale)
                # average Gemma + Kimi scores per attribute
                combined: dict[str, float] = {}
                for attr in JUDGE_ATTRIBUTES:
                    gv = g_scores.get(variant, {}).get(attr)
                    kv = k_scores.get(variant, {}).get(attr)
                    vals = [v for v in [gv, kv] if v is not None]
                    if vals:
                        combined[attr] = sum(vals) / len(vals)
                all_rows.append({
                    "model": model_label,
                    "variant": variant,
                    "id": qid,
                    "word_count": wc,
                    "char_count": cc,
                    **{f"gemma_{a}": g_scores.get(variant, {}).get(a) for a in JUDGE_ATTRIBUTES},
                    **{f"kimi_{a}": k_scores.get(variant, {}).get(a) for a in JUDGE_ATTRIBUTES},
                    **{f"combined_{a}": combined.get(a) for a in JUDGE_ATTRIBUTES},
                })

    print(f"Total rows: {len(all_rows)}")

    # ---- 1. Length by variant (averaged across models) ----
    from collections import defaultdict
    length_by_mv: dict[tuple, list[int]] = defaultdict(list)
    for row in all_rows:
        length_by_mv[(row["model"], row["variant"])].append(row["word_count"])

    length_rows = []
    for (model, variant), wcs in sorted(length_by_mv.items()):
        length_rows.append({
            "model": model,
            "variant": variant,
            "n": len(wcs),
            "mean_words": round(mean(wcs), 1),
            "median_words": round(median(wcs), 1),
            "max_words": max(wcs),
        })

    with open(OUTPUT_DIR / "length_by_variant.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "variant", "n", "mean_words", "median_words", "max_words"])
        writer.writeheader()
        writer.writerows(length_rows)

    # ---- 2. Length by model ----
    length_by_model: dict[str, list[int]] = defaultdict(list)
    for row in all_rows:
        length_by_model[row["model"]].append(row["word_count"])

    model_length_rows = [
        {"model": m, "n": len(wcs), "mean_words": round(mean(wcs), 1), "median_words": round(median(wcs), 1)}
        for m, wcs in sorted(length_by_model.items())
    ]
    with open(OUTPUT_DIR / "length_by_model.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "n", "mean_words", "median_words"])
        writer.writeheader()
        writer.writerows(model_length_rows)

    # ---- 3. Correlation: length vs judge scores ----
    corr_rows = []
    for model_label in MODEL_MAP:
        model_rows = [r for r in all_rows if r["model"] == model_label]
        wcs = [r["word_count"] for r in model_rows]
        for attr in JUDGE_ATTRIBUTES:
            for judge_prefix in ("gemma", "kimi", "combined"):
                scores = [r[f"{judge_prefix}_{attr}"] for r in model_rows]
                paired = [(w, s) for w, s in zip(wcs, scores) if s is not None]
                if len(paired) < 10:
                    continue
                wx, sx = zip(*paired)
                corr_rows.append({
                    "model": model_label,
                    "judge": judge_prefix,
                    "attribute": attr,
                    "n": len(paired),
                    "spearman_rho": round(spearman(list(wx), list(sx)), 4),
                })

    with open(OUTPUT_DIR / "length_judge_correlations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "judge", "attribute", "n", "spearman_rho"])
        writer.writeheader()
        writer.writerows(corr_rows)

    # ---- 4. Partial effects: variant → judge score, before and after controlling for length ----
    partial_rows = []
    for model_label in MODEL_MAP:
        model_rows = [r for r in all_rows if r["model"] == model_label]
        for variant in VARIANTS_ORDERED[1:]:  # v0 is reference
            # indicator: 1 if this variant, 0 if variant0
            indicator_rows = [
                r for r in model_rows
                if r["variant"] in (variant, "variant0_neutral_baseline")
            ]
            if not indicator_rows:
                continue
            ind = [1.0 if r["variant"] == variant else 0.0 for r in indicator_rows]
            wcs = [r["word_count"] for r in indicator_rows]
            for attr in JUDGE_ATTRIBUTES:
                scores = [r[f"combined_{attr}"] for r in indicator_rows]
                paired = [(i, w, s) for i, w, s in zip(ind, wcs, scores) if s is not None]
                if len(paired) < 20:
                    continue
                pi, pw, ps = zip(*paired)
                # raw OLS coefficient for variant indicator
                raw_slope, _ = ols_simple(list(pi), list(ps))
                # partial coefficient controlling for word_count
                partial_slope = partial_corr_controlling_length(list(pw), list(pi), list(ps))
                partial_rows.append({
                    "model": model_label,
                    "variant": variant,
                    "attribute": attr,
                    "n": len(paired),
                    "raw_beta": round(raw_slope, 4) if not math.isnan(raw_slope) else "",
                    "partial_beta_controlling_length": round(partial_slope, 4) if not math.isnan(partial_slope) else "",
                    "attenuation": round(raw_slope - partial_slope, 4) if not (math.isnan(raw_slope) or math.isnan(partial_slope)) else "",
                })

    with open(OUTPUT_DIR / "partial_effects.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "variant", "attribute", "n",
                                                "raw_beta", "partial_beta_controlling_length", "attenuation"])
        writer.writeheader()
        writer.writerows(partial_rows)

    # ---- 5. Report ----
    _write_report(length_rows, model_length_rows, corr_rows, partial_rows)
    print(f"Wrote all outputs to {OUTPUT_DIR}/")

    # ---- Quick console summary ----
    print("\n=== Mean word count by variant (averaged across models) ===")
    variant_mean: dict[str, list[float]] = defaultdict(list)
    for r in length_rows:
        variant_mean[r["variant"]].append(r["mean_words"])
    for v in VARIANTS_ORDERED:
        vals = variant_mean.get(v, [])
        print(f"  {v:42s}  {mean(vals):5.1f} words")

    print("\n=== Length–judge correlations (combined, Qwen2.5-7B) ===")
    q25_corrs = [r for r in corr_rows if r["model"] == "Qwen2.5-7b-instruct" and r["judge"] == "combined"]
    for r in sorted(q25_corrs, key=lambda x: -abs(x["spearman_rho"])):
        print(f"  {r['attribute']:25s}  ρ={r['spearman_rho']:+.3f}")

    print("\n=== Top attenuations (raw β → partial β) for completeness ===")
    comp = [r for r in partial_rows if r["attribute"] == "completeness" and r["model"] == "Qwen2.5-7b-instruct"]
    for r in sorted(comp, key=lambda x: abs(float(x["attenuation"])) if x["attenuation"] != "" else 0, reverse=True):
        if r["attenuation"] == "":
            continue
        print(f"  {r['variant']:42s}  raw={r['raw_beta']:+.4f}  partial={r['partial_beta_controlling_length']:+.4f}  Δ={r['attenuation']:+.4f}")


def _write_report(length_rows, model_length_rows, corr_rows, partial_rows) -> None:
    from collections import defaultdict

    lines = [
        "# Rationale Length Analysis",
        "",
        "Addresses reviewer concern: do structured prompts produce longer rationales",
        "that LLM judges reward for length rather than quality?",
        "",
        "---",
        "",
        "## 1. Mean rationale length by model",
        "",
        "| Model | N | Mean words | Median words |",
        "|-------|---|-----------|-------------|",
    ]
    for r in model_length_rows:
        lines.append(f"| {r['model']} | {r['n']} | {r['mean_words']:.1f} | {r['median_words']:.1f} |")

    lines += ["", "## 2. Mean rationale length by variant (across models)", "", "| Variant | Mean words |", "|---------|-----------|"]
    variant_mean: dict[str, list[float]] = defaultdict(list)
    for r in length_rows:
        variant_mean[r["variant"]].append(r["mean_words"])
    for v in [
        "variant0_neutral_baseline",
        "variant1_predicted_event",
        "variant2_key_attribute",
        "variant3_reasoning_type",
        "variant4_credibility",
        "variant5_key_conditions",
        "variant6_step_by_step_reasoning",
        "variant7_uncertainty_language",
        "variant8_temporal_anchors",
    ]:
        vals = variant_mean.get(v, [])
        if vals:
            lines.append(f"| {v} | {mean(vals):.1f} |")

    lines += [
        "",
        "## 3. Spearman ρ: rationale word count vs judge scores (combined, all models)",
        "",
        "| Model | Attribute | ρ (length vs score) |",
        "|-------|-----------|-------------------|",
    ]
    combined_corrs = [r for r in corr_rows if r["judge"] == "combined"]
    for r in sorted(combined_corrs, key=lambda x: (x["model"], x["attribute"])):
        lines.append(f"| {r['model']} | {r['attribute']} | {r['spearman_rho']:+.3f} |")

    lines += [
        "",
        "## 4. Variant effects before and after controlling for length",
        "",
        "β_raw = OLS slope for variant indicator vs judge score.",
        "β_partial = same after partialling out word count (Frisch-Waugh).",
        "Attenuation = β_raw − β_partial; positive means length explains part of the effect.",
        "",
        "| Model | Variant | Attribute | β_raw | β_partial | Attenuation |",
        "|-------|---------|-----------|-------|-----------|-------------|",
    ]
    for r in partial_rows:
        if r["attenuation"] == "":
            continue
        lines.append(
            f"| {r['model']} | {r['variant']} | {r['attribute']} | "
            f"{r['raw_beta']} | {r['partial_beta_controlling_length']} | {r['attenuation']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "- Positive length–score correlations indicate judges reward longer rationales.",
        "- If β_partial ≈ β_raw, the variant effect is independent of length.",
        "- Large positive attenuation (β_raw >> β_partial) means the apparent variant benefit",
        "  is partly or fully explained by the longer rationale it elicits.",
        "- `conciseness` should show negative length correlation by design.",
    ]

    (OUTPUT_DIR / "length_analysis_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
