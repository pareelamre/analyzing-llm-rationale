#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyzing_llm_rationale.metrics import (
    Example,
    accuracy,
    brier_score,
    ece,
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


def iter_result_rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("results", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def target_id(value: object) -> int | None:
    try:
        rid = int(value)
    except (TypeError, ValueError):
        return None
    return rid


def score_result_rows(
    rows: list[dict],
    targets: dict[int, int],
    bins: int,
) -> dict[str, float | int]:
    rows_by_id: dict[int, dict] = {}
    n_extra_rows = 0
    for row in rows:
        rid = target_id(row.get("id"))
        if rid not in targets:
            n_extra_rows += 1
            continue
        rows_by_id[rid] = row

    examples: list[Example] = []
    n_invalid = 0
    for rid, row in rows_by_id.items():
        answer = normalize_answer(row.get("predicted_answer"))
        confidence = normalize_confidence(row.get("confidence"))
        if answer is None or confidence is None:
            n_invalid += 1
            continue
        examples.append(Example(answer, confidence, targets[rid]))

    n_expected = len(targets)
    n_valid = len(examples)
    n_absent = n_expected - len(rows_by_id)
    n_invalid_or_missing = n_invalid + n_absent
    coverage = n_valid / n_expected if n_expected else float("nan")

    conditional_accuracy = accuracy(examples) if examples else float("nan")
    conditional_brier = brier_score(examples) if examples else float("nan")
    conditional_ece = ece(examples, bins) if examples else float("nan")

    # Invalid or absent forecasts are failures. For Brier, the worst possible
    # binary score is 1.0, so malformed outputs receive that penalty.
    valid_brier_total = sum((ex.p_yes - ex.target) ** 2 for ex in examples)
    valid_correct_total = sum(ex.correct for ex in examples)
    coverage_penalized_accuracy = valid_correct_total / n_expected if n_expected else float("nan")
    coverage_penalized_brier = (
        (valid_brier_total + n_invalid_or_missing) / n_expected
        if n_expected
        else float("nan")
    )

    return {
        "n_expected": n_expected,
        "n_valid": n_valid,
        "n_scored": n_valid,
        "n_invalid": n_invalid,
        "n_absent": n_absent,
        "n_missing": n_invalid_or_missing,
        "n_invalid_or_missing": n_invalid_or_missing,
        "n_extra_rows": n_extra_rows,
        "coverage": coverage,
        "conditional_accuracy": conditional_accuracy,
        "conditional_brier_score": conditional_brier,
        "conditional_ece": conditional_ece,
        "coverage_penalized_accuracy": coverage_penalized_accuracy,
        "coverage_penalized_brier_score": coverage_penalized_brier,
        # Backward-compatible aliases: these remain valid-output conditional.
        "accuracy": conditional_accuracy,
        "brier_score": conditional_brier,
        "ece": conditional_ece,
    }


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
                metrics = score_result_rows(iter_result_rows(payload), targets, args.bins)
                if metrics["n_valid"] == 0:
                    continue

                rows_out.append(
                    {
                        "model": model,
                        "temperature_dir": temp_dir.name,
                        "temperature": parse_temperature(temp_dir.name),
                        "variant": parse_variant(result_path.name),
                        **metrics,
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "temperature_dir",
        "temperature",
        "variant",
        "n_expected",
        "n_valid",
        "n_scored",
        "n_invalid",
        "n_absent",
        "n_missing",
        "n_invalid_or_missing",
        "n_extra_rows",
        "coverage",
        "conditional_accuracy",
        "conditional_brier_score",
        "conditional_ece",
        "coverage_penalized_accuracy",
        "coverage_penalized_brier_score",
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
