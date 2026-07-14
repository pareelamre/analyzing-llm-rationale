#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
METRICS_PATH = ROOT / "analysis" / "metrics_by_model_temperature_variant.csv"
DEFAULT_OUTPUT_CSV = ROOT / "analysis" / "simple_baselines.csv"
DEFAULT_OUTPUT_MD = ROOT / "analysis" / "simple_baselines.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate non-LLM forecasting baselines on the benchmark dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--llm-metrics-csv", type=Path, default=METRICS_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--category-prior-weight",
        type=float,
        default=5.0,
        help="Pseudo-count weight for smoothing category rates toward the global base rate.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in payload:
        answer = str(row.get("answer", "")).strip().lower()
        if answer not in {"yes", "no"}:
            continue
        rows.append({
            "id": int(row["id"]),
            "target": 1 if answer == "yes" else 0,
            "categories": categories_for(row),
        })
    return rows


def categories_for(row: dict[str, Any]) -> list[str]:
    categories = row.get("categories") or []
    if isinstance(categories, str):
        return [categories]
    return [str(c) for c in categories if c]


def hard_prediction(p_yes: float) -> int | None:
    if math.isclose(p_yes, 0.5):
        return None
    return 1 if p_yes > 0.5 else 0


def brier_score(probs: list[float], targets: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, targets)) / len(targets)


def hard_accuracy(probs: list[float], targets: list[int]) -> float:
    scored = [(hard_prediction(p), y) for p, y in zip(probs, targets)]
    scored = [(pred, y) for pred, y in scored if pred is not None]
    if not scored:
        return float("nan")
    return sum(int(pred == y) for pred, y in scored) / len(scored)


def confidence_ece(probs: list[float], targets: list[int], bins: int) -> float:
    total = len(targets)
    total_error = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        idx: list[int] = []
        for j, p_yes in enumerate(probs):
            prediction = hard_prediction(p_yes)
            if prediction is None:
                confidence = 0.5
            else:
                confidence = p_yes if prediction == 1 else 1.0 - p_yes
            if confidence >= lo and (confidence < hi or i == bins - 1):
                idx.append(j)
        if not idx:
            continue
        avg_confidence = sum(max(probs[j], 1.0 - probs[j]) for j in idx) / len(idx)
        correct = 0
        for j in idx:
            prediction = hard_prediction(probs[j])
            if prediction is None:
                # A 50% baseline is exactly uncommitted; count expected correctness.
                correct += 0.5
            else:
                correct += int(prediction == targets[j])
        avg_accuracy = correct / len(idx)
        total_error += (len(idx) / total) * abs(avg_accuracy - avg_confidence)
    return total_error


def probability_ece(probs: list[float], targets: list[int], bins: int) -> float:
    total = len(targets)
    total_error = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        idx = [
            j for j, p in enumerate(probs)
            if p >= lo and (p < hi or i == bins - 1)
        ]
        if not idx:
            continue
        avg_prob = sum(probs[j] for j in idx) / len(idx)
        avg_target = sum(targets[j] for j in idx) / len(idx)
        total_error += (len(idx) / total) * abs(avg_target - avg_prob)
    return total_error


def metric_row(
    name: str,
    probs: list[float],
    targets: list[int],
    bins: int,
    notes: str,
) -> dict[str, object]:
    return {
        "baseline": name,
        "n": len(targets),
        "accuracy": hard_accuracy(probs, targets),
        "brier_score": brier_score(probs, targets),
        "confidence_ece": confidence_ece(probs, targets, bins),
        "probability_ece": probability_ece(probs, targets, bins),
        "mean_p_yes": sum(probs) / len(probs),
        "notes": notes,
    }


def category_probs(
    rows: list[dict[str, Any]],
    *,
    smoothed: bool,
    prior_weight: float,
) -> list[float]:
    global_rate = sum(row["target"] for row in rows) / len(rows)
    counts: Counter[str] = Counter()
    yes_counts: Counter[str] = Counter()
    for row in rows:
        for category in row["categories"]:
            counts[category] += 1
            yes_counts[category] += int(row["target"])

    probs: list[float] = []
    for row in rows:
        rates: list[float] = []
        for category in row["categories"]:
            n = counts[category] - 1
            y = yes_counts[category] - int(row["target"])
            if smoothed:
                rates.append((y + prior_weight * global_rate) / (n + prior_weight))
            elif n > 0:
                rates.append(y / n)
        probs.append(sum(rates) / len(rates) if rates else global_rate)
    return probs


def load_llm_comparison(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    rows = list(csv.DictReader(path.open()))
    neutral = [row for row in rows if row.get("variant") == "variant0_neutral_baseline"]
    models = sorted({row["model"] for row in neutral})
    best_neutral: list[dict[str, str]] = []
    best_overall: list[dict[str, str]] = []
    for model in models:
        neutral_rows = [row for row in neutral if row["model"] == model]
        model_rows = [row for row in rows if row["model"] == model]
        best_neutral.append(min(neutral_rows, key=lambda r: float(r["conditional_brier_score"])))
        best_overall.append(min(model_rows, key=lambda r: float(r["conditional_brier_score"])))
    return best_neutral, best_overall


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline",
        "n",
        "accuracy",
        "brier_score",
        "confidence_ece",
        "probability_ece",
        "mean_p_yes",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    baseline_rows: list[dict[str, object]],
    llm_neutral_rows: list[dict[str, str]],
    llm_best_rows: list[dict[str, str]],
    yes_rate: float,
) -> None:
    lines = [
        "# Simple Forecasting Baselines",
        "",
        f"Dataset Yes base rate: `{yes_rate:.3f}`.",
        "",
        "## Non-LLM Baselines",
        "",
        "| Baseline | Accuracy | Brier score | Confidence ECE | Probability ECE | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in baseline_rows:
        lines.append(
            f"| {row['baseline']} | {fmt(row['accuracy'])} | "
            f"{fmt(row['brier_score'])} | {fmt(row['confidence_ece'])} | "
            f"{fmt(row['probability_ece'])} | {row['notes']} |"
        )

    lines += [
        "",
        "## Neutral LLM Baselines",
        "",
        "| Model | Temperature | Accuracy | Brier score | ECE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in llm_neutral_rows:
        lines.append(
            f"| {row['model']} | {row['temperature_dir']} | "
            f"{float(row['conditional_accuracy']):.3f} | "
            f"{float(row['conditional_brier_score']):.3f} | "
            f"{float(row['conditional_ece']):.3f} |"
        )

    lines += [
        "",
        "## Best Observed LLM Runs By Brier Score",
        "",
        "| Model | Temperature | Variant | Accuracy | Brier score | ECE |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in llm_best_rows:
        lines.append(
            f"| {row['model']} | {row['temperature_dir']} | {row['variant']} | "
            f"{float(row['conditional_accuracy']):.3f} | "
            f"{float(row['conditional_brier_score']):.3f} | "
            f"{float(row['conditional_ece']):.3f} |"
        )

    lines += [
        "",
        "Metaculus community prediction is the preferred external crowd baseline,",
        "but the current API token does not expose aggregation histories for these",
        "benchmark questions. Add it when Metaculus grants the required data tier.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_dataset(args.dataset)
    targets = [int(row["target"]) for row in rows]
    yes_rate = sum(targets) / len(targets)

    baseline_rows = [
        metric_row(
            "50% probability",
            [0.5] * len(rows),
            targets,
            args.bins,
            "Uninformative probabilistic baseline; hard accuracy is undefined at exactly 50%.",
        ),
        metric_row(
            "Global base rate",
            [yes_rate] * len(rows),
            targets,
            args.bins,
            "Always predicts the benchmark Yes base rate.",
        ),
        metric_row(
            "Category base rate (leave-one-out)",
            category_probs(rows, smoothed=False, prior_weight=args.category_prior_weight),
            targets,
            args.bins,
            "Uses category-specific resolved frequencies excluding the current question.",
        ),
        metric_row(
            "Category base rate (smoothed leave-one-out)",
            category_probs(rows, smoothed=True, prior_weight=args.category_prior_weight),
            targets,
            args.bins,
            "Shrinks leave-one-out category rates toward the global base rate.",
        ),
    ]

    write_csv(args.output_csv, baseline_rows)
    llm_neutral_rows, llm_best_rows = load_llm_comparison(args.llm_metrics_csv)
    write_markdown(args.output_md, baseline_rows, llm_neutral_rows, llm_best_rows, yes_rate)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
