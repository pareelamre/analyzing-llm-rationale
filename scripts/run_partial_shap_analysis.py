#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
RESULTS_ROOT = ROOT / "results"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "partial_shap_analysis"
DEFAULT_JUDGE_OUTPUT_DIRS = {
    "gemma-4-31b-it": ROOT / "analysis" / "llm_judge_rationale_eval_gemma" / "gemma-4-31b-it",
    "kimi-k2.5": ROOT / "analysis" / "llm_judge_rationale_eval_kimi" / "kimi-k2.5",
}
ATTRIBUTES = [
    "plausibility",
    "completeness",
    "source_consistency",
    "non_hallucination",
    "informativeness",
    "conciseness",
]
VARIANT_ALIASES = {
    "variant7_uncertain_language": "variant7_uncertainty_language",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SHAP analysis on the currently available LLM-judge rationale scores. "
            "This works on partial judge outputs too."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--judge-dirs",
        nargs="*",
        default=[
            f"{judge}={path}"
            for judge, path in DEFAULT_JUDGE_OUTPUT_DIRS.items()
        ],
        help="Mappings like judge_name=/abs/path/to/jsonl_dir",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    return parser.parse_args()


def load_dataset_answers(dataset_path: Path) -> dict[int, str]:
    rows = json.loads(dataset_path.read_text())
    return {int(row["id"]): str(row["answer"]).strip().lower() for row in rows}


def parse_filename(path: Path) -> tuple[str, str]:
    model_label, temperature_dir = path.stem.split("__", maxsplit=1)
    return model_label, temperature_dir


def load_variant_predictions(
    results_root: Path,
    model_label: str,
    temperature_dir: str,
    dataset_answers: dict[int, str],
) -> dict[tuple[int, str], dict[str, Any]]:
    variant_predictions: dict[tuple[int, str], dict[str, Any]] = {}
    variant_files = sorted((results_root / model_label / temperature_dir).glob("results_variant*.json"))
    for path in variant_files:
        variant = path.stem.removeprefix("results_")
        rows = json.loads(path.read_text())
        for row in rows:
            rid = int(row["id"])
            predicted_answer = str(row.get("predicted_answer") or "").strip().lower()
            answer = dataset_answers.get(rid)
            if answer not in {"yes", "no"} or predicted_answer not in {"yes", "no"}:
                continue
            variant_predictions[(rid, variant)] = {
                "forecast_correct": int(predicted_answer == answer),
                "confidence": row.get("confidence"),
            }
    return variant_predictions


def normalize_variant_name(variant: str) -> str:
    return VARIANT_ALIASES.get(variant, variant)


def iter_payloads(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def load_judge_rows(
    judge_name: str,
    judge_dir: Path,
    results_root: Path,
    dataset_answers: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(judge_dir.glob("*.jsonl")):
        model_label, temperature_dir = parse_filename(path)
        variant_predictions = load_variant_predictions(
            results_root,
            model_label,
            temperature_dir,
            dataset_answers,
        )
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                for item in iter_payloads(payload):
                    rid = int(item["id"])
                    variant_scores = item.get("variant_scores", {})
                    for raw_variant, score_row in variant_scores.items():
                        if not isinstance(score_row, dict):
                            continue
                        variant = normalize_variant_name(raw_variant)
                        prediction_row = variant_predictions.get((rid, variant))
                        if prediction_row is None:
                            continue
                        out_row = {
                            "judge": judge_name,
                            "model": model_label,
                            "temperature_dir": temperature_dir,
                            "variant": variant,
                            "id": rid,
                            "forecast_correct": prediction_row["forecast_correct"],
                        }
                        for attribute in ATTRIBUTES:
                            value = score_row.get(attribute)
                            if value is None:
                                break
                            out_row[attribute] = float(value)
                        else:
                            rows.append(out_row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shap_matrix_for_positive_class(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1], dtype=float)
    shap_values_array = np.asarray(shap_values, dtype=float)
    if shap_values_array.ndim == 3:
        return shap_values_array[:, :, -1]
    return shap_values_array


def fit_and_explain(
    rows: list[dict[str, Any]],
    *,
    random_state: int,
    n_estimators: int,
    max_depth: int,
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]]]:
    features = np.asarray([[row[attribute] for attribute in ATTRIBUTES] for row in rows], dtype=float)
    target = np.asarray([row["forecast_correct"] for row in rows], dtype=int)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    probabilities = cross_val_predict(
        model,
        features,
        target,
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "n_rows": float(len(rows)),
        "positive_rate": float(target.mean()),
        "cv_roc_auc": float(roc_auc_score(target, probabilities)),
        "cv_accuracy": float(accuracy_score(target, predictions)),
    }

    model.fit(features, target)
    shap_values = shap_matrix_for_positive_class(model, features)

    feature_rows: list[dict[str, Any]] = []
    mean_abs_values = np.abs(shap_values).mean(axis=0)
    for index, attribute in enumerate(ATTRIBUTES):
        attribute_values = features[:, index]
        shap_column = shap_values[:, index]
        direction = float(np.corrcoef(attribute_values, shap_column)[0, 1])
        if math.isnan(direction):
            direction = 0.0
        feature_rows.append(
            {
                "feature": attribute,
                "mean_abs_shap": float(mean_abs_values[index]),
                "mean_value": float(attribute_values.mean()),
                "mean_value_correct": float(attribute_values[target == 1].mean()),
                "mean_value_incorrect": float(attribute_values[target == 0].mean()),
                "correct_minus_incorrect": float(attribute_values[target == 1].mean() - attribute_values[target == 0].mean()),
                "value_shap_correlation": direction,
            }
        )
    feature_rows.sort(key=lambda row: float(row["mean_abs_shap"]), reverse=True)

    detail_rows: list[dict[str, Any]] = []
    for row, shap_row in zip(rows, shap_values):
        detail_row = {
            "judge": row["judge"],
            "model": row["model"],
            "temperature_dir": row["temperature_dir"],
            "variant": row["variant"],
            "id": row["id"],
            "forecast_correct": row["forecast_correct"],
        }
        for index, attribute in enumerate(ATTRIBUTES):
            detail_row[attribute] = row[attribute]
            detail_row[f"shap_{attribute}"] = float(shap_row[index])
        detail_rows.append(detail_row)

    return feature_rows, metrics, detail_rows


def build_combined_rows(judge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: defaultdict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in judge_rows:
        key = (row["model"], row["temperature_dir"], row["variant"], row["id"])
        buckets[key].append(row)

    combined_rows: list[dict[str, Any]] = []
    for (model, temperature_dir, variant, rid), rows in buckets.items():
        if len(rows) < 2:
            continue
        combined_row = {
            "judge": "combined_mean",
            "model": model,
            "temperature_dir": temperature_dir,
            "variant": variant,
            "id": rid,
            "forecast_correct": rows[0]["forecast_correct"],
        }
        for attribute in ATTRIBUTES:
            combined_row[attribute] = float(np.mean([row[attribute] for row in rows]))
        combined_rows.append(combined_row)
    return combined_rows


def write_summary_md(
    path: Path,
    metrics_by_dataset: dict[str, dict[str, float]],
    feature_rows_by_dataset: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Partial SHAP Analysis",
        "",
        "This analysis uses the currently available LLM-judge outputs and predicts `forecast_correct` from the judged rationale attributes.",
        "",
    ]
    for dataset_name in sorted(metrics_by_dataset):
        metrics = metrics_by_dataset[dataset_name]
        feature_rows = feature_rows_by_dataset[dataset_name]
        lines.extend(
            [
                f"## {dataset_name}",
                "",
                f"- Rows: `{int(metrics['n_rows'])}`",
                f"- Positive rate: `{metrics['positive_rate']:.3f}`",
                f"- CV ROC-AUC: `{metrics['cv_roc_auc']:.3f}`",
                f"- CV Accuracy: `{metrics['cv_accuracy']:.3f}`",
                "",
                "| Feature | Mean | Correct - Incorrect | Mean |SHAP| | Value-SHAP Corr |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in feature_rows:
            lines.append(
                f"| {row['feature']} | {row['mean_value']:.3f} | "
                f"{row['correct_minus_incorrect']:.3f} | {row['mean_abs_shap']:.5f} | "
                f"{row['value_shap_correlation']:.3f} |"
            )
        lines.append("")
        if feature_rows:
            top = feature_rows[0]
            lines.append(
                f"Top SHAP feature: `{top['feature']}` with mean |SHAP| `{top['mean_abs_shap']:.5f}`."
            )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_answers = load_dataset_answers(args.dataset)

    judge_dirs: dict[str, Path] = {}
    for item in args.judge_dirs:
        judge_name, raw_path = item.split("=", maxsplit=1)
        judge_dirs[judge_name] = Path(raw_path)

    all_rows: list[dict[str, Any]] = []
    for judge_name, judge_dir in judge_dirs.items():
        if not judge_dir.exists():
            continue
        all_rows.extend(
            load_judge_rows(
                judge_name,
                judge_dir,
                args.results_root,
                dataset_answers,
            )
        )

    if not all_rows:
        raise SystemExit("No judged rows found for SHAP analysis.")

    datasets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        datasets[row["judge"]].append(row)

    combined_rows = build_combined_rows(all_rows)
    if combined_rows:
        datasets["combined_mean"] = combined_rows

    metrics_summary_rows: list[dict[str, Any]] = []
    feature_summary_rows: list[dict[str, Any]] = []
    metrics_by_dataset: dict[str, dict[str, float]] = {}
    feature_rows_by_dataset: dict[str, list[dict[str, Any]]] = {}

    for dataset_name, rows in sorted(datasets.items()):
        feature_rows, metrics, detail_rows = fit_and_explain(
            rows,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        metrics_by_dataset[dataset_name] = metrics
        feature_rows_by_dataset[dataset_name] = feature_rows

        metrics_summary_rows.append(
            {
                "dataset": dataset_name,
                **metrics,
            }
        )
        for feature_row in feature_rows:
            feature_summary_rows.append(
                {
                    "dataset": dataset_name,
                    **feature_row,
                }
            )

        detail_path = args.output_dir / f"{dataset_name}_details.csv"
        detail_fieldnames = [
            "judge",
            "model",
            "temperature_dir",
            "variant",
            "id",
            "forecast_correct",
            *ATTRIBUTES,
            *[f"shap_{attribute}" for attribute in ATTRIBUTES],
        ]
        write_csv(detail_path, detail_rows, detail_fieldnames)

    write_csv(
        args.output_dir / "metrics_summary.csv",
        metrics_summary_rows,
        ["dataset", "n_rows", "positive_rate", "cv_roc_auc", "cv_accuracy"],
    )
    write_csv(
        args.output_dir / "feature_importance.csv",
        feature_summary_rows,
        [
            "dataset",
            "feature",
            "mean_abs_shap",
            "mean_value",
            "mean_value_correct",
            "mean_value_incorrect",
            "correct_minus_incorrect",
            "value_shap_correlation",
        ],
    )
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_by_dataset, indent=2),
        encoding="utf-8",
    )
    write_summary_md(
        args.output_dir / "summary.md",
        metrics_by_dataset,
        feature_rows_by_dataset,
    )


if __name__ == "__main__":
    main()
