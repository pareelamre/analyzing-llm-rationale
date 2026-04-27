#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
RESULTS_ROOT = ROOT / "results"
DEFAULT_MODELS = [
    "Qwen2.5-7b-instruct",
    "Qwen3-32B",
    "GPT-OSS-120B",
]


@dataclass(frozen=True)
class Example:
    predicted_answer: str
    confidence: float
    target: int

    @property
    def predicted_label(self) -> int:
        return 1 if self.predicted_answer == "yes" else 0

    @property
    def p_yes(self) -> float:
        return self.confidence if self.predicted_label == 1 else 1.0 - self.confidence

    @property
    def correct(self) -> int:
        return int(self.predicted_label == self.target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy, Brier score, and ECE for result JSON files."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model directory names under results/.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the benchmark dataset with ground-truth answers.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory containing model result folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "analysis" / "metrics_by_model_temperature_variant.csv",
        help="Where to write the detailed metrics CSV.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equal-width bins for ECE.",
    )
    return parser.parse_args()


def load_targets(dataset_path: Path) -> dict[int, int]:
    payload = json.loads(dataset_path.read_text())
    targets: dict[int, int] = {}
    for row in payload:
        answer = str(row["answer"]).strip().lower()
        if answer not in {"yes", "no"}:
            continue
        targets[int(row["id"])] = 1 if answer == "yes" else 0
    return targets


def normalize_answer(value: object) -> str | None:
    if value is None:
        return None
    answer = str(value).strip().lower()
    if answer in {"yes", "no"}:
        return answer
    return None


def normalize_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(confidence) or math.isinf(confidence):
        return None
    if not (0.0 <= confidence <= 1.0):
        return None
    return confidence


def iter_examples(rows: Iterable[dict], targets: dict[int, int]) -> tuple[list[Example], int]:
    examples: list[Example] = []
    missing = 0
    for row in rows:
        rid = row.get("id")
        if rid not in targets:
            missing += 1
            continue
        answer = normalize_answer(row.get("predicted_answer"))
        confidence = normalize_confidence(row.get("confidence"))
        if answer is None or confidence is None:
            missing += 1
            continue
        examples.append(Example(answer, confidence, targets[rid]))
    return examples, missing


def accuracy(examples: list[Example]) -> float:
    return sum(ex.correct for ex in examples) / len(examples)


def brier_score(examples: list[Example]) -> float:
    return sum((ex.p_yes - ex.target) ** 2 for ex in examples) / len(examples)


def ece(examples: list[Example], bins: int) -> float:
    total = len(examples)
    if total == 0:
        return float("nan")

    bin_counts = [0] * bins
    bin_confidence = [0.0] * bins
    bin_accuracy = [0.0] * bins

    for ex in examples:
        # ECE uses confidence in the predicted label vs empirical correctness.
        idx = min(int(ex.confidence * bins), bins - 1)
        bin_counts[idx] += 1
        bin_confidence[idx] += ex.confidence
        bin_accuracy[idx] += ex.correct

    total_error = 0.0
    for count, conf_sum, acc_sum in zip(bin_counts, bin_confidence, bin_accuracy):
        if count == 0:
            continue
        avg_conf = conf_sum / count
        avg_acc = acc_sum / count
        total_error += (count / total) * abs(avg_acc - avg_conf)
    return total_error


def parse_variant(filename: str) -> str:
    return filename.removeprefix("results_").removesuffix(".json")


def parse_temperature(dirname: str) -> float:
    raw = dirname.removeprefix("temperature_")
    if raw in {"0", "00", "000"}:
        return 0.0
    if len(raw) == 3 and raw.isdigit():
        return int(raw) / 1000.0
    return float(raw)


def main() -> None:
    args = parse_args()
    targets = load_targets(args.dataset)
    rows_out: list[dict[str, object]] = []

    for model in args.models:
        model_dir = args.results_root / model
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        for temp_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for result_path in sorted(temp_dir.glob("results_variant*.json")):
                payload = json.loads(result_path.read_text())
                examples, missing = iter_examples(payload, targets)
                if not examples:
                    continue

                rows_out.append(
                    {
                        "model": model,
                        "temperature_dir": temp_dir.name,
                        "temperature": parse_temperature(temp_dir.name),
                        "variant": parse_variant(result_path.name),
                        "n_scored": len(examples),
                        "n_missing": missing,
                        "accuracy": accuracy(examples),
                        "brier_score": brier_score(examples),
                        "ece": ece(examples, args.bins),
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "temperature_dir",
        "temperature",
        "variant",
        "n_scored",
        "n_missing",
        "accuracy",
        "brier_score",
        "ece",
    ]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
