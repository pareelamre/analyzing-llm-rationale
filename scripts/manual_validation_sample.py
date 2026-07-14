#!/usr/bin/env python3
"""
Third-judge validation sample for thesis Section on LLM-judge reliability.

Samples 100 rationales (stratified correct/incorrect) from the Qwen2.5-7B
temperature_000 variant0 run, annotates them with GPT-OSS-120B using 5
criteria that deliberately differ from the existing 6 (Gemma/Kimi):

  criterion_alignment   — rationale addresses the specific resolution criteria
  evidence_consistency  — claims are supported by the provided evidence
  hallucination         — no unsupported facts (1 = clean, 0 = hallucinated)
  usefulness            — would help a human forecaster understand the forecast
  probability_support   — rationale justifies the specific confidence value,
                          not just the direction (faithfulness-relevant)

Outputs
-------
analysis/manual_validation/
  sample.jsonl                  — the 100 annotated records
  agreement_stats.json          — Spearman ρ and Cohen κ tables
  agreement_report.md           — human-readable summary
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyzing_llm_rationale.config import load_model_configs
from analyzing_llm_rationale.providers import OpenAICompatibleProvider
from analyzing_llm_rationale.cli import resolve_api_key

DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
RESULTS_PATH = ROOT / "results" / "Qwen2.5-7b-instruct" / "temperature_000" / "results_variant0_neutral_baseline.json"
GEMMA_JUDGE_PATH = ROOT / "analysis" / "llm_judge_rationale_eval_gemma" / "gemma-4-31b-it" / "Qwen2.5-7b-instruct__temperature_000.jsonl"
KIMI_JUDGE_PATH = ROOT / "analysis" / "llm_judge_rationale_eval_kimi" / "kimi-k2.5" / "Qwen2.5-7b-instruct__temperature_000.jsonl"
OUTPUT_DIR = ROOT / "analysis" / "manual_validation"

SAMPLE_N = 100
RANDOM_SEED = 42
VARIANT = "variant0_neutral_baseline"
JUDGE_MODEL = "gemma-4-31b-it"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 128
JUDGE_TIMEOUT = 90.0
MAX_RETRIES = 3

# Criteria for the third judge — deliberately more targeted than the existing 6
THIRD_JUDGE_CRITERIA = [
    "criterion_alignment",
    "evidence_consistency",
    "hallucination",
    "usefulness",
    "probability_support",
]

# Mapping from third-judge criteria to closest existing judge criterion
# (for cross-judge correlation reporting)
CRITERIA_OVERLAP = {
    "criterion_alignment": ["completeness", "source_consistency"],
    "evidence_consistency": ["source_consistency"],
    "hallucination": ["non_hallucination"],
    "usefulness": ["informativeness"],
    "probability_support": [],  # novel — no direct analogue
}

SYSTEM_PROMPT = """\
You are a meticulous evaluator of AI-generated forecasting rationales.

You will receive a forecasting question with:
- The question text and resolution criteria
- A brief evidence digest (summaries of news articles)
- The model's predicted answer (Yes/No) and confidence (0–1)
- The model's rationale

Score the rationale on five criteria, each from 0.0 to 1.0:

criterion_alignment (0–1):
  Does the rationale directly address what the resolution criteria requires?
  1.0 = squarely targets the resolution condition; 0.0 = ignores it.

evidence_consistency (0–1):
  Are the specific claims in the rationale supported by the evidence digest?
  1.0 = every claim traceable to the evidence; 0.0 = claims contradict or ignore evidence.

hallucination (0–1):
  Does the rationale avoid adding facts, numbers, or events NOT present in the evidence?
  1.0 = no hallucinations; 0.0 = introduces unsupported or fabricated content.

usefulness (0–1):
  Would a human forecaster, reading only this rationale, gain meaningful insight
  into why this probability was assigned? 1.0 = highly useful; 0.0 = generic or unhelpful.

probability_support (0–1):
  Does the rationale justify the SPECIFIC confidence value (not just Yes/No direction)?
  E.g. if confidence is 0.9, does the rationale convey strong certainty?
  If confidence is 0.55, does it reflect genuine uncertainty?
  1.0 = confidence is well-grounded in the rationale; 0.0 = mismatch or no calibration signal.

Rules:
- Evaluate based solely on the provided context. Do not use outside knowledge.
- Be strict: a fluent, well-structured rationale that fails criteria should score low.
- Return ONLY a JSON object with exactly these five keys and float values 0.0–1.0.
  No markdown, no prose outside the JSON.
"""

USER_TEMPLATE = """\
QUESTION: {question}

RESOLUTION CRITERIA: {resolution_criteria}

EVIDENCE DIGEST:
{evidence_digest}

MODEL PREDICTION: {predicted_answer} (confidence: {confidence})
MODEL RATIONALE: {rationale}

Return JSON with keys: criterion_alignment, evidence_consistency, hallucination, usefulness, probability_support.
"""


def load_dataset() -> dict[int, dict]:
    with open(DATASET_PATH) as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


def load_results() -> dict[int, dict]:
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("results", [])
    return {r["id"]: r for r in records if r.get("id") is not None}


def load_judge(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            variant_scores = row.get("variant_scores", {})
            if VARIANT in variant_scores:
                out[row["id"]] = variant_scores[VARIANT]
    return out


def evidence_digest(record: dict, max_articles: int = 3) -> str:
    articles = record.get("news_articles", [])[:max_articles]
    parts = []
    for i, art in enumerate(articles, 1):
        summary = art.get("summary_llm") or art.get("summary") or art.get("text", "")[:300]
        parts.append(f"[{i}] {summary[:400]}")
    return "\n".join(parts) if parts else "(no evidence)"


def stratified_sample(
    result_ids: list[int],
    dataset: dict,
    results: dict,
    n: int,
    seed: int,
) -> list[int]:
    correct, incorrect = [], []
    for qid in result_ids:
        rec = dataset.get(qid)
        res = results.get(qid)
        if not rec or not res:
            continue
        gt = rec.get("answer", "").lower().strip()
        pred = (res.get("predicted_answer") or "").lower().strip()
        if gt == pred:
            correct.append(qid)
        else:
            incorrect.append(qid)

    rng = random.Random(seed)
    n_correct = min(n // 2, len(correct))
    n_incorrect = min(n - n_correct, len(incorrect))
    sampled = rng.sample(correct, n_correct) + rng.sample(incorrect, n_incorrect)
    rng.shuffle(sampled)
    return sampled


def _extract_json(text: str) -> dict:
    if text is None:
        raise ValueError("empty response")
    text = text.strip()
    # strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    # find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in: {text[:100]!r}")
    return json.loads(text[start : end + 1])


def call_judge(provider: OpenAICompatibleProvider, user_text: str) -> dict[str, float] | None:
    import requests as _requests
    session = _requests.Session()
    for attempt in range(MAX_RETRIES):
        try:
            payload = {
                "model": provider.model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                "temperature": JUDGE_TEMPERATURE,
                "max_tokens": JUDGE_MAX_TOKENS,
            }
            resp = session.post(
                provider.base_url,
                headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=provider.request_timeout_s,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            # gpt-oss-120b (reasoning model) may put output in reasoning_content when content is null
            text = msg.get("content") or msg.get("reasoning_content") or ""
            scores = _extract_json(text)
            if all(k in scores for k in THIRD_JUDGE_CRITERIA):
                return {k: float(scores[k]) for k in THIRD_JUDGE_CRITERIA}
            raise ValueError(f"missing keys; got {list(scores.keys())}")
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  attempt {attempt+1} failed: {exc}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


# ---------- agreement statistics ----------

def spearman_rho(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    rx = _ranks(x)
    ry = _ranks(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def _ranks(vals: list[float]) -> list[float]:
    indexed = sorted(enumerate(vals), key=lambda t: t[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def cohen_kappa(x: list[float], y: list[float], threshold: float = 0.5) -> float:
    bx = [1 if v >= threshold else 0 for v in x]
    by = [1 if v >= threshold else 0 for v in y]
    n = len(bx)
    if n == 0:
        return float("nan")
    agree = sum(a == b for a, b in zip(bx, by)) / n
    p1 = sum(bx) / n
    p2 = sum(by) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe >= 1.0:
        return float("nan")
    return (agree - pe) / (1 - pe)


def mean_absolute_diff(x: list[float], y: list[float]) -> float:
    if not x:
        return float("nan")
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x)


def compute_agreement(
    a_scores: dict[int, dict[str, float]],
    b_scores: dict[int, dict[str, float]],
    criteria: list[str],
    sample_ids: list[int],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for crit in criteria:
        pairs = [
            (a_scores[qid][crit], b_scores[qid][crit])
            for qid in sample_ids
            if qid in a_scores and qid in b_scores
            and crit in a_scores[qid] and crit in b_scores[qid]
        ]
        if not pairs:
            continue
        x, y = zip(*pairs)
        result[crit] = {
            "n": len(x),
            "spearman_rho": round(spearman_rho(list(x), list(y)), 4),
            "cohen_kappa": round(cohen_kappa(list(x), list(y)), 4),
            "mean_abs_diff": round(mean_absolute_diff(list(x), list(y)), 4),
            "mean_a": round(sum(x) / len(x), 4),
            "mean_b": round(sum(y) / len(y), 4),
        }
    return result


def render_report(
    agreement_stats: dict,
    sample_n: int,
    n_correct: int,
    n_incorrect: int,
) -> str:
    lines = [
        "# Manual Validation: Inter-Judge Agreement Report",
        "",
        f"**Sample**: {sample_n} rationales from Qwen2.5-7B, temperature=0, variant0_neutral_baseline",
        f"  - Correct predictions: {n_correct}",
        f"  - Incorrect predictions: {n_incorrect}",
        "",
        "**Third judge**: Gemma-4-31B-it with 5 targeted criteria (criterion_alignment,",
        "evidence_consistency, hallucination, usefulness, probability_support).",
        "",
        "**Existing judges**: Gemma-4-31B-it and Kimi-K2.5 with 6 criteria",
        "(plausibility, completeness, source_consistency, non_hallucination,",
        "informativeness, conciseness).",
        "",
        "---",
        "",
    ]

    for pair_key, pair_data in agreement_stats.items():
        lines.append(f"## {pair_key}")
        lines.append("")
        lines.append(f"| Criterion | n | Spearman ρ | Cohen κ | Mean |diff| | Mean A | Mean B |")
        lines.append(f"|-----------|---|-----------|---------|----------|--------|--------|")
        for crit, stats in pair_data.items():
            if not stats:
                lines.append(f"| {crit} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {crit} | {stats['n']} | {stats['spearman_rho']:.3f} | "
                f"{stats['cohen_kappa']:.3f} | {stats['mean_abs_diff']:.3f} | "
                f"{stats['mean_a']:.3f} | {stats['mean_b']:.3f} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Interpretation notes",
        "",
        "- **Spearman ρ > 0.6**: moderate-to-strong ordinal agreement",
        "- **Cohen κ > 0.4**: moderate agreement on binarised scores (threshold 0.5)",
        "- **Mean |diff| < 0.15**: acceptable absolute score spread",
        "- `probability_support` has no direct analogue in the Gemma/Kimi criteria;",
        "  it is reported only for the GPT-OSS-120B third-judge output.",
        "- `hallucination` (third judge) corresponds to `non_hallucination` (existing judges);",
        "  the scale is inverted — high hallucination score = *clean* rationale.",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-n", type=int, default=SAMPLE_N)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--models-config", type=Path, default=ROOT / "configs" / "models.yaml")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--dry-run", action="store_true", help="Sample only; skip LLM calls")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset and results...")
    dataset = load_dataset()
    results = load_results()
    gemma_scores = load_judge(GEMMA_JUDGE_PATH)
    kimi_scores = load_judge(KIMI_JUDGE_PATH)

    eligible = [qid for qid in results if qid in dataset and qid in gemma_scores and qid in kimi_scores]
    print(f"Eligible IDs (have results + both judge scores): {len(eligible)}")

    sample_ids = stratified_sample(eligible, dataset, results, args.sample_n, args.seed)
    n_correct = sum(
        1 for qid in sample_ids
        if dataset[qid].get("answer", "").lower() == (results[qid].get("predicted_answer") or "").lower()
    )
    n_incorrect = len(sample_ids) - n_correct
    print(f"Sampled {len(sample_ids)}: {n_correct} correct, {n_incorrect} incorrect")

    # ---- third-judge annotation ----
    third_judge_scores: dict[int, dict[str, float]] = {}
    sample_out_path = args.output_dir / "sample.jsonl"

    if sample_out_path.exists():
        print("Loading existing third-judge annotations from sample.jsonl...")
        with open(sample_out_path) as f:
            for line in f:
                row = json.loads(line)
                if "third_judge_scores" in row and row["third_judge_scores"]:
                    third_judge_scores[row["id"]] = row["third_judge_scores"]
        print(f"  Loaded {len(third_judge_scores)} existing annotations")

    to_annotate = [qid for qid in sample_ids if qid not in third_judge_scores]
    print(f"Need to annotate: {len(to_annotate)}")

    if not args.dry_run and to_annotate:
        model_configs = load_model_configs(args.models_config)
        cfg = model_configs[args.judge_model]
        import types
        fake_args = types.SimpleNamespace(
            provider="openai-compatible",
            api_key_env_var=cfg.api_key_env_var,
            api_key_file=cfg.api_key_file,
        )
        api_key = resolve_api_key(fake_args)
        provider = OpenAICompatibleProvider(
            model_name=cfg.local_model_name,
            api_key=api_key,
            base_url=cfg.api_base_url,
            request_timeout_s=JUDGE_TIMEOUT,
        )

        for i, qid in enumerate(to_annotate, 1):
            rec = dataset[qid]
            res = results[qid]
            user_text = USER_TEMPLATE.format(
                question=rec.get("question", ""),
                resolution_criteria=rec.get("resolution_criteria", ""),
                evidence_digest=evidence_digest(rec),
                predicted_answer=res.get("predicted_answer", ""),
                confidence=res.get("confidence", ""),
                rationale=res.get("rationale", ""),
            )
            print(f"  [{i}/{len(to_annotate)}] annotating qid={qid}...", end=" ", flush=True)
            scores = call_judge(provider, user_text)
            if scores:
                third_judge_scores[qid] = scores
                print("ok")
            else:
                print("FAILED")

    # ---- write sample.jsonl ----
    with open(sample_out_path, "w") as f:
        for qid in sample_ids:
            rec = dataset[qid]
            res = results[qid]
            row = {
                "id": qid,
                "question": rec.get("question", ""),
                "ground_truth": rec.get("answer", ""),
                "predicted_answer": res.get("predicted_answer", ""),
                "confidence": res.get("confidence"),
                "rationale": res.get("rationale", ""),
                "correct": rec.get("answer", "").lower() == (res.get("predicted_answer") or "").lower(),
                "gemma_scores": gemma_scores.get(qid),
                "kimi_scores": kimi_scores.get(qid),
                "third_judge_scores": third_judge_scores.get(qid),
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(sample_ids)} rows to {sample_out_path}")

    # ---- inter-judge agreement ----
    annotated_ids = [qid for qid in sample_ids if qid in third_judge_scores]
    print(f"\nComputing agreement over {len(annotated_ids)} annotated records...")

    # Gemma vs Kimi (original 6 criteria)
    shared_criteria_6 = ["plausibility", "completeness", "source_consistency",
                         "non_hallucination", "informativeness", "conciseness"]
    gemma_kimi_agreement = compute_agreement(
        gemma_scores, kimi_scores, shared_criteria_6, sample_ids
    )

    # Map third-judge criteria to overlapping existing criteria for cross-judge stats
    # third hallucination ↔ non_hallucination; third evidence_consistency ↔ source_consistency;
    # third usefulness ↔ informativeness
    def cross_agree(third_crit: str, existing_crit: str, existing_judge: dict) -> dict[str, float]:
        third_vals, exist_vals = [], []
        for qid in annotated_ids:
            if qid not in existing_judge or existing_crit not in existing_judge[qid]:
                continue
            tv = third_judge_scores[qid].get(third_crit)
            ev = existing_judge[qid].get(existing_crit)
            if tv is None or ev is None:
                continue
            third_vals.append(tv)
            exist_vals.append(ev)
        if not third_vals:
            return {}
        return {
            "n": len(third_vals),
            "spearman_rho": round(spearman_rho(third_vals, exist_vals), 4),
            "cohen_kappa": round(cohen_kappa(third_vals, exist_vals), 4),
            "mean_abs_diff": round(mean_absolute_diff(third_vals, exist_vals), 4),
            "mean_a": round(sum(third_vals) / len(third_vals), 4),
            "mean_b": round(sum(exist_vals) / len(exist_vals), 4),
        }

    # Build cross-judge tables
    overlap_pairs = [
        ("hallucination", "non_hallucination"),
        ("evidence_consistency", "source_consistency"),
        ("usefulness", "informativeness"),
    ]
    third_vs_gemma = {
        f"{tc} ↔ {ec}": cross_agree(tc, ec, gemma_scores)
        for tc, ec in overlap_pairs
    }
    third_vs_kimi = {
        f"{tc} ↔ {ec}": cross_agree(tc, ec, kimi_scores)
        for tc, ec in overlap_pairs
    }

    # Third-judge internal stats (all 5 criteria, vs correct/incorrect)
    def criterion_by_outcome(crit: str) -> dict:
        correct_vals = [third_judge_scores[qid][crit] for qid in annotated_ids
                        if dataset[qid].get("answer","").lower() == (results[qid].get("predicted_answer") or "").lower()
                        and crit in third_judge_scores.get(qid, {})]
        incorrect_vals = [third_judge_scores[qid][crit] for qid in annotated_ids
                          if dataset[qid].get("answer","").lower() != (results[qid].get("predicted_answer") or "").lower()
                          and crit in third_judge_scores.get(qid, {})]
        mc = sum(correct_vals) / len(correct_vals) if correct_vals else float("nan")
        mi = sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else float("nan")
        return {"mean_correct": round(mc, 4), "mean_incorrect": round(mi, 4),
                "diff": round(mc - mi, 4) if not (math.isnan(mc) or math.isnan(mi)) else float("nan")}

    third_judge_by_outcome = {c: criterion_by_outcome(c) for c in THIRD_JUDGE_CRITERIA}

    agreement_stats = {
        "Gemma-4-31B vs Kimi-K2.5 (existing 6 criteria)": gemma_kimi_agreement,
        "Gemma-4-31B (third judge, 5 criteria) vs Gemma-4-31B (primary, 6 criteria)": third_vs_gemma,
        "Gemma-4-31B (third judge, 5 criteria) vs Kimi-K2.5 (overlapping concepts)": third_vs_kimi,
    }

    stats_path = args.output_dir / "agreement_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "agreement": agreement_stats,
            "third_judge_by_outcome": third_judge_by_outcome,
            "sample_n": len(sample_ids),
            "annotated_n": len(annotated_ids),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
        }, f, indent=2)
    print(f"Wrote {stats_path}")

    report = render_report(agreement_stats, len(sample_ids), n_correct, n_incorrect)
    report_path = args.output_dir / "agreement_report.md"
    report_path.write_text(report)
    print(f"Wrote {report_path}")

    # Print quick summary
    print("\n=== Quick summary ===")
    print("\nGemma vs Kimi (shared 6 criteria):")
    for crit, s in gemma_kimi_agreement.items():
        print(f"  {crit:25s}  ρ={s['spearman_rho']:+.3f}  κ={s['cohen_kappa']:+.3f}  |diff|={s['mean_abs_diff']:.3f}")

    if annotated_ids:
        print(f"\nThird-judge (n={len(annotated_ids)}) outcome gap (correct - incorrect):")
        for crit, s in third_judge_by_outcome.items():
            print(f"  {crit:25s}  Δ={s['diff']:+.4f}  (correct={s['mean_correct']:.3f}, incorrect={s['mean_incorrect']:.3f})")


if __name__ == "__main__":
    main()
