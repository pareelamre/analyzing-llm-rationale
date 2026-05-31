#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyzing_llm_rationale.metrics import (
    Example,
    accuracy,
    brier_score,
    ece,
    iter_examples,
    load_targets,
    normalize_answer,
    normalize_confidence,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
RESULTS_ROOT = ROOT / "results"
DEFAULT_MODELS = [
    "Qwen2.5-7b-instruct",
    "Qwen3-32B",
    "GPT-OSS-120B",
]


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


def parse_variant(filename: str) -> str:
    return filename.removeprefix("results_").removesuffix(".json")


def parse_temperature(dirname: str) -> float:
    raw = dirname.removeprefix("temperature_")
    if raw in {"0", "00", "000"}:
        return 0.0
    if raw.isdigit() and raw.startswith("00"):
        return int(raw) / 100.0
    if len(raw) == 3 and raw.isdigit():
        return int(raw) / 100.0
    if len(raw) == 2 and raw.isdigit() and raw.startswith("0"):
        return int(raw) / 10.0
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
