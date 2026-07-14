#!/usr/bin/env python3
"""Question-level uncertainty analysis for prompt-variant forecast metrics.

The comparisons are paired by question ID. For each model/temperature, every
main prompt variant is compared against variant0_neutral_baseline using only
questions scored in both conditions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from analyzing_llm_rationale.metrics import (
    load_targets,
    normalize_answer,
    normalize_confidence,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
RESULTS_ROOT = ROOT / "results"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "uncertainty_qwen_gpt"
DEFAULT_MODELS = ["GPT-OSS-120B", "Qwen2.5-7b-instruct", "Qwen3-32B"]
BASELINE_VARIANT = "variant0_neutral_baseline"
MAIN_VARIANT_RE = re.compile(r"variant[0-8]_")
N_BOOTSTRAPS = 10000
N_PERMUTATIONS = 10000


@dataclass(frozen=True)
class ScoredQuestion:
    question_id: int
    correct: float
    brier: float
    log_loss: float


@dataclass(frozen=True)
class RunScores:
    scored: dict[int, ScoredQuestion]
    n_expected: int
    n_missing: int

    @property
    def coverage(self) -> float:
        return len(self.scored) / self.n_expected if self.n_expected else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_json_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("results", [])
        return [row for row in rows if isinstance(row, dict)]
    return []


def _log_loss(p_yes: float, target: int) -> float:
    eps = 1e-15
    p = min(max(p_yes, eps), 1.0 - eps)
    return -(target * math.log(p) + (1 - target) * math.log(1.0 - p))


def _score_rows(path: Path, targets: dict[int, int]) -> RunScores:
    scored: dict[int, ScoredQuestion] = {}
    n_expected = len(targets)
    for row in _load_json_rows(path):
        rid = row.get("id")
        if rid not in targets:
            continue
        answer = normalize_answer(row.get("predicted_answer"))
        confidence = normalize_confidence(row.get("confidence"))
        if answer is None or confidence is None:
            continue
        pred = 1 if answer == "yes" else 0
        p_yes = confidence if pred == 1 else 1.0 - confidence
        target = targets[int(rid)]
        scored[int(rid)] = ScoredQuestion(
            question_id=int(rid),
            correct=float(pred == target),
            brier=(p_yes - target) ** 2,
            log_loss=_log_loss(p_yes, target),
        )
    return RunScores(
        scored=scored,
        n_expected=n_expected,
        n_missing=max(n_expected - len(scored), 0),
    )


def _variant_name(path: Path) -> str:
    return path.stem.removeprefix("results_")


def iter_run_files(results_root: Path, models: Iterable[str]) -> Iterable[tuple[str, str, str, Path]]:
    for model in models:
        model_dir = results_root / model
        if not model_dir.exists():
            continue
        for temp_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for path in sorted(temp_dir.glob("results_variant*.json")):
                variant = _variant_name(path)
                if variant == BASELINE_VARIANT or MAIN_VARIANT_RE.match(variant):
                    yield model, temp_dir.name, variant, path


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstraps: int) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    idx = rng.integers(0, len(values), size=(n_bootstraps, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def permutation_pvalue(
    diffs: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    if len(diffs) == 0:
        return math.nan
    observed = abs(float(diffs.mean()))
    if observed == 0.0:
        return 1.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (n_permutations + 1))


def paired_effect_size(diffs: np.ndarray) -> float:
    if len(diffs) < 2:
        return math.nan
    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        return 0.0 if float(diffs.mean()) == 0.0 else math.copysign(math.inf, float(diffs.mean()))
    return float(diffs.mean() / sd)


def bh_adjust(p_values: list[float]) -> list[float]:
    indexed = [(i, p) for i, p in enumerate(p_values) if not math.isnan(p)]
    adjusted = [math.nan] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 1.0
    for rank, (idx, p) in reversed(list(enumerate(indexed, start=1))):
        running = min(running, p * m / rank)
        adjusted[idx] = running
    return adjusted


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_summary(path: Path, run_rows: list[dict], comparison_rows: list[dict]) -> None:
    sig_acc = [r for r in comparison_rows if r["accuracy_q_bh"] < 0.05]
    sig_brier = [r for r in comparison_rows if r["brier_q_bh"] < 0.05]
    sig_log_loss = [r for r in comparison_rows if r["log_loss_q_bh"] < 0.05]
    robust_brier = [
        r for r in comparison_rows
        if r["brier_q_bh"] < 0.05
        and r["brier_delta_ci_low"] * r["brier_delta_ci_high"] > 0
    ]
    robust_log_loss = [
        r for r in comparison_rows
        if r["log_loss_q_bh"] < 0.05
        and r["log_loss_delta_ci_low"] * r["log_loss_delta_ci_high"] > 0
    ]
    brier_improvements = [r for r in robust_brier if r["brier_delta"] < 0]
    brier_degradations = [r for r in robust_brier if r["brier_delta"] > 0]

    lines = [
        "# Paired Probabilistic Forecast Analysis",
        "",
        "Comparisons are paired by question ID against `variant0_neutral_baseline`",
        "within the same forecast model and temperature. Confidence intervals use",
        "a paired bootstrap over question IDs. P-values use a paired sign-flip",
        "permutation test over question-level metric differences, with",
        "Benjamini-Hochberg correction across all variant-vs-baseline comparisons",
        "within each metric.",
        "",
        "Metric priority follows probabilistic forecasting practice: Brier score is",
        "the primary metric, log loss is a secondary proper scoring rule, ECE is",
        "reported elsewhere only as a calibration diagnostic, accuracy is auxiliary,",
        "and output coverage is part of system reliability.",
        "",
        f"- Run-level estimates: `{len(run_rows)}` model/temperature/variant runs",
        f"- Paired comparisons: `{len(comparison_rows)}`",
        f"- Brier comparisons with BH q < 0.05: `{len(sig_brier)}`",
        f"- Robust Brier differences (q < 0.05 and CI excludes 0): `{len(robust_brier)}`",
        f"- Robust Brier improvements: `{len(brier_improvements)}`",
        f"- Robust Brier degradations: `{len(brier_degradations)}`",
        f"- Log-loss comparisons with BH q < 0.05: `{len(sig_log_loss)}`",
        f"- Robust log-loss differences (q < 0.05 and CI excludes 0): `{len(robust_log_loss)}`",
        f"- Accuracy comparisons with BH q < 0.05: `{len(sig_acc)}`",
        "",
        "Small metric gaps should not be interpreted unless the paired interval",
        "excludes zero and the multiple-comparison-adjusted q-value remains small.",
        "A lower ECE should not be described as an improvement when Brier/log loss",
        "or output coverage worsens.",
        "",
        "## Robust Brier Differences",
        "",
        "| Model | Temp | Variant | Δ Brier | 95% CI | q | dz | Δ log loss | Coverage Δ | Direction |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(robust_brier, key=lambda r: abs(r["brier_delta"]), reverse=True)[:20]:
        direction = "improves Brier" if row["brier_delta"] < 0 else "worsens Brier"
        lines.append(
            f"| {row['model']} | {row['temperature_dir']} | {row['variant']} | "
            f"{row['brier_delta']:.3f} | "
            f"[{row['brier_delta_ci_low']:.3f}, {row['brier_delta_ci_high']:.3f}] | "
            f"{_fmt(row['brier_q_bh'])} | {row['brier_effect_dz']:.3f} | "
            f"{row['log_loss_delta']:.3f} | {row['coverage_delta']:.3f} | {direction} |"
        )
    lines += [
        "",
        "## Largest Accuracy Differences (Auxiliary)",
        "",
        "| Model | Temp | Variant | Δ accuracy | 95% CI | q | dz | Δ Brier | Brier q | Coverage Δ |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    top = sorted(comparison_rows, key=lambda r: abs(r["accuracy_delta"]), reverse=True)[:15]
    for row in top:
        lines.append(
            f"| {row['model']} | {row['temperature_dir']} | {row['variant']} | "
            f"{row['accuracy_delta']:.3f} | "
            f"[{row['accuracy_delta_ci_low']:.3f}, {row['accuracy_delta_ci_high']:.3f}] | "
            f"{_fmt(row['accuracy_q_bh'])} | {row['accuracy_effect_dz']:.3f} | "
            f"{row['brier_delta']:.3f} | {_fmt(row['brier_q_bh'])} | "
            f"{row['coverage_delta']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)
    targets = load_targets(args.dataset)

    runs: dict[tuple[str, str, str], RunScores] = {}
    for model, temp, variant, path in iter_run_files(args.results_root, args.models):
        runs[(model, temp, variant)] = _score_rows(path, targets)

    run_rows: list[dict] = []
    for (model, temp, variant), run in sorted(runs.items()):
        correct = np.array([row.correct for row in run.scored.values()], dtype=float)
        brier = np.array([row.brier for row in run.scored.values()], dtype=float)
        log_loss = np.array([row.log_loss for row in run.scored.values()], dtype=float)
        acc_lo, acc_hi = bootstrap_ci(correct, rng, args.n_bootstraps)
        brier_lo, brier_hi = bootstrap_ci(brier, rng, args.n_bootstraps)
        log_loss_lo, log_loss_hi = bootstrap_ci(log_loss, rng, args.n_bootstraps)
        run_rows.append({
            "model": model,
            "temperature_dir": temp,
            "variant": variant,
            "n_expected": run.n_expected,
            "n_scored": len(run.scored),
            "n_missing": run.n_missing,
            "coverage": run.coverage,
            "output_validity_rate": run.coverage,
            "log_loss": float(log_loss.mean()) if len(log_loss) else math.nan,
            "log_loss_ci_low": log_loss_lo,
            "log_loss_ci_high": log_loss_hi,
            "brier_score": float(brier.mean()) if len(brier) else math.nan,
            "brier_ci_low": brier_lo,
            "brier_ci_high": brier_hi,
            "accuracy": float(correct.mean()) if len(correct) else math.nan,
            "accuracy_ci_low": acc_lo,
            "accuracy_ci_high": acc_hi,
        })

    comparison_rows: list[dict] = []
    for model in args.models:
        temps = sorted({temp for m, temp, _ in runs if m == model})
        for temp in temps:
            baseline = runs.get((model, temp, BASELINE_VARIANT))
            if not baseline:
                continue
            for key, variant_rows in sorted(runs.items()):
                key_model, key_temp, variant = key
                if key_model != model or key_temp != temp or variant == BASELINE_VARIANT:
                    continue
                common_ids = sorted(set(baseline.scored) & set(variant_rows.scored))
                if not common_ids:
                    continue
                acc_diff = np.array(
                    [
                        variant_rows.scored[rid].correct - baseline.scored[rid].correct
                        for rid in common_ids
                    ],
                    dtype=float,
                )
                brier_diff = np.array(
                    [
                        variant_rows.scored[rid].brier - baseline.scored[rid].brier
                        for rid in common_ids
                    ],
                    dtype=float,
                )
                log_loss_diff = np.array(
                    [
                        variant_rows.scored[rid].log_loss - baseline.scored[rid].log_loss
                        for rid in common_ids
                    ],
                    dtype=float,
                )
                acc_lo, acc_hi = bootstrap_ci(acc_diff, rng, args.n_bootstraps)
                brier_lo, brier_hi = bootstrap_ci(brier_diff, rng, args.n_bootstraps)
                log_loss_lo, log_loss_hi = bootstrap_ci(log_loss_diff, rng, args.n_bootstraps)
                comparison_rows.append({
                    "model": model,
                    "temperature_dir": temp,
                    "variant": variant,
                    "baseline_variant": BASELINE_VARIANT,
                    "n_paired_questions": len(common_ids),
                    "baseline_n_scored": len(baseline.scored),
                    "variant_n_scored": len(variant_rows.scored),
                    "baseline_n_missing": baseline.n_missing,
                    "variant_n_missing": variant_rows.n_missing,
                    "baseline_coverage": baseline.coverage,
                    "variant_coverage": variant_rows.coverage,
                    "coverage_delta": variant_rows.coverage - baseline.coverage,
                    "brier_delta": float(brier_diff.mean()),
                    "brier_delta_ci_low": brier_lo,
                    "brier_delta_ci_high": brier_hi,
                    "brier_p_permutation": permutation_pvalue(brier_diff, rng, args.n_permutations),
                    "brier_effect_dz": paired_effect_size(brier_diff),
                    "log_loss_delta": float(log_loss_diff.mean()),
                    "log_loss_delta_ci_low": log_loss_lo,
                    "log_loss_delta_ci_high": log_loss_hi,
                    "log_loss_p_permutation": permutation_pvalue(
                        log_loss_diff, rng, args.n_permutations
                    ),
                    "log_loss_effect_dz": paired_effect_size(log_loss_diff),
                    "accuracy_delta": float(acc_diff.mean()),
                    "accuracy_delta_ci_low": acc_lo,
                    "accuracy_delta_ci_high": acc_hi,
                    "accuracy_p_permutation": permutation_pvalue(acc_diff, rng, args.n_permutations),
                    "accuracy_effect_dz": paired_effect_size(acc_diff),
                })

    brier_q = bh_adjust([row["brier_p_permutation"] for row in comparison_rows])
    log_loss_q = bh_adjust([row["log_loss_p_permutation"] for row in comparison_rows])
    acc_q = bh_adjust([row["accuracy_p_permutation"] for row in comparison_rows])
    for row, bq, lq, aq in zip(comparison_rows, brier_q, log_loss_q, acc_q):
        row["brier_q_bh"] = bq
        row["log_loss_q_bh"] = lq
        row["accuracy_q_bh"] = aq

    write_csv(
        args.output_dir / "run_metric_confidence_intervals.csv",
        run_rows,
        [
            "model", "temperature_dir", "variant",
            "n_expected", "n_scored", "n_missing", "coverage", "output_validity_rate",
            "brier_score", "brier_ci_low", "brier_ci_high",
            "log_loss", "log_loss_ci_low", "log_loss_ci_high",
            "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        ],
    )
    write_csv(
        args.output_dir / "paired_variant_vs_baseline.csv",
        comparison_rows,
        [
            "model", "temperature_dir", "variant", "baseline_variant", "n_paired_questions",
            "baseline_n_scored", "variant_n_scored", "baseline_n_missing", "variant_n_missing",
            "baseline_coverage", "variant_coverage", "coverage_delta",
            "brier_delta", "brier_delta_ci_low", "brier_delta_ci_high",
            "brier_p_permutation", "brier_q_bh", "brier_effect_dz",
            "log_loss_delta", "log_loss_delta_ci_low", "log_loss_delta_ci_high",
            "log_loss_p_permutation", "log_loss_q_bh", "log_loss_effect_dz",
            "accuracy_delta", "accuracy_delta_ci_low", "accuracy_delta_ci_high",
            "accuracy_p_permutation", "accuracy_q_bh", "accuracy_effect_dz",
        ],
    )
    write_summary(args.output_dir / "summary.md", run_rows, comparison_rows)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
