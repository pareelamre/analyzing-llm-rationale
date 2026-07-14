#!/usr/bin/env python3
"""
SHAP analysis — revised to address reviewer concerns:

  1. Grouped cross-validation: all rows for the same question ID land in the
     same fold (StratifiedGroupKFold), preventing leakage of question difficulty.
     A fold audit is written to prove there is zero train/test question overlap.

  2. Control features added alongside the 6 judge attributes:
       - model (one-hot)
       - prompt variant (one-hot)
       - temperature (float)
       - rationale word count
       - evidence word count
       - output validity flag (predicted_answer is yes/no)
       - question category flags derived from the observed dataset labels

  Two models are fit per judge dataset:
    * judge_only  — 6 judge attributes, grouped CV
    * full        — judge attributes + all controls, grouped CV

  The difference in ROC-AUC between the two reveals how much control
  variables explain beyond raw judge scores.

  SHAP is computed on the full model to show true marginal attribution.
  By default, SHAP values are computed on a reproducible row sample so grouped
  CV metrics remain exact while explanation tables stay tractable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
RESULTS_ROOT = ROOT / "results"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "shap_analysis"
DEFAULT_JUDGE_OUTPUT_DIRS = {
    "gemma-4-31b-it": ROOT / "analysis" / "llm_judge_rationale_eval_gemma" / "gemma-4-31b-it",
    "kimi-k2.5": ROOT / "analysis" / "llm_judge_rationale_eval_kimi" / "kimi-k2.5",
}
DEFAULT_FORECAST_MODELS = [
    "GPT-OSS-120B",
    "Qwen2.5-7b-instruct",
    "Qwen3-32B",
]
JUDGE_ATTRIBUTES = [
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
N_CV_SPLITS = 5
DEFAULT_TOP_N_CATEGORIES = 12
DEFAULT_SHAP_SAMPLE_SIZE = 12000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--judge-dirs",
        nargs="*",
        default=[f"{j}={p}" for j, p in DEFAULT_JUDGE_OUTPUT_DIRS.items()],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--cv-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP value computation and write RandomForest impurity importances instead.",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=DEFAULT_SHAP_SAMPLE_SIZE,
        help="Rows sampled for SHAP explanations after fitting on all rows; use 0 for all rows.",
    )
    parser.add_argument(
        "--top-n-categories",
        type=int,
        default=DEFAULT_TOP_N_CATEGORIES,
        help="Number of observed question category labels to encode as multi-hot controls.",
    )
    parser.add_argument(
        "--forecast-models",
        nargs="*",
        default=None,
        help="Optional forecast model labels to include from judge JSONL files.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only fit the pooled combined_mean judge dataset.",
    )
    return parser.parse_args()


# ---------- data loading ----------

def load_dataset(path: Path) -> dict[int, dict]:
    rows = json.loads(path.read_text())
    return {int(r["id"]): r for r in rows}


def _word_count(text: str | None) -> int:
    return len(text.split()) if text else 0


def _evidence_word_count(record: dict) -> int:
    total = 0
    for art in record.get("news_articles", []):
        total += _word_count(art.get("summary_llm") or art.get("summary") or art.get("text", ""))
    return total


def _temperature_float(temperature_dir: str) -> float:
    raw = temperature_dir.removeprefix("temperature_")
    if raw in {"0", "00", "000"}:
        return 0.0
    if raw.isdigit():
        if raw.startswith("00"):
            return int(raw) / 100.0
        if len(raw) == 3:
            return int(raw) / 100.0
        if len(raw) == 2 and raw.startswith("0"):
            return int(raw) / 10.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def normalize_variant(v: str) -> str:
    return VARIANT_ALIASES.get(v, v)


def iter_payloads(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def load_variant_predictions(
    results_root: Path,
    model_label: str,
    temperature_dir: str,
    dataset_answers: dict[int, str],
) -> dict[tuple[int, str], dict]:
    out: dict[tuple[int, str], dict] = {}
    base = results_root / model_label / temperature_dir
    if not base.exists():
        return out
    for path in sorted(base.glob("results_variant*.json")):
        variant = normalize_variant(path.stem.removeprefix("results_"))
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = rows.get("results", [])
        for row in rows:
            rid = int(row["id"])
            pred = str(row.get("predicted_answer") or "").strip().lower()
            answer = dataset_answers.get(rid)
            if answer not in {"yes", "no"}:
                continue
            out[(rid, variant)] = {
                "forecast_correct": int(pred == answer),
                "output_valid": int(pred in {"yes", "no"}),
                "rationale_words": _word_count(row.get("rationale")),
            }
    return out


def load_judge_rows(
    judge_name: str,
    judge_dir: Path,
    results_root: Path,
    dataset: dict[int, dict],
    forecast_models: set[str] | None = None,
) -> list[dict]:
    dataset_answers = {rid: str(r["answer"]).strip().lower() for rid, r in dataset.items()}
    rows: list[dict] = []

    # Collect all unique models and variants for one-hot encoding later
    for path in sorted(judge_dir.glob("*.jsonl")):
        model_label, temperature_dir = path.stem.split("__", maxsplit=1)
        if forecast_models is not None and model_label not in forecast_models:
            continue
        variant_preds = load_variant_predictions(
            results_root, model_label, temperature_dir, dataset_answers
        )
        temperature = _temperature_float(temperature_dir)

        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                for item in iter_payloads(json.loads(line)):
                    rid = int(item["id"])
                    ds_rec = dataset.get(rid)
                    if ds_rec is None:
                        continue
                    evidence_words = _evidence_word_count(ds_rec)
                    categories = ds_rec.get("categories") or []
                    if isinstance(categories, str):
                        categories = [categories]

                    for raw_variant, score_row in item.get("variant_scores", {}).items():
                        if not isinstance(score_row, dict):
                            continue
                        variant = normalize_variant(raw_variant)
                        pred_row = variant_preds.get((rid, variant))
                        if pred_row is None:
                            continue

                        out_row: dict = {
                            "judge": judge_name,
                            "model": model_label,
                            "temperature_dir": temperature_dir,
                            "temperature": temperature,
                            "variant": variant,
                            "id": rid,
                            "forecast_correct": pred_row["forecast_correct"],
                            "output_valid": pred_row["output_valid"],
                            "rationale_words": pred_row["rationale_words"],
                            "evidence_words": evidence_words,
                            "categories": categories,
                        }
                        for attr in JUDGE_ATTRIBUTES:
                            v = score_row.get(attr)
                            if v is None:
                                break
                            out_row[attr] = float(v)
                        else:
                            rows.append(out_row)
    return rows


def build_combined_rows(all_rows: list[dict]) -> list[dict]:
    buckets: defaultdict[tuple, list[dict]] = defaultdict(list)
    for row in all_rows:
        key = (row["model"], row["temperature_dir"], row["variant"], row["id"])
        buckets[key].append(row)
    combined: list[dict] = []
    for group in buckets.values():
        if len(group) < 2:
            continue
        base = {k: v for k, v in group[0].items() if k not in JUDGE_ATTRIBUTES}
        base["judge"] = "combined_mean"
        for attr in JUDGE_ATTRIBUTES:
            base[attr] = float(np.mean([r[attr] for r in group]))
        combined.append(base)
    return combined


# ---------- feature engineering ----------

def build_feature_matrix(
    rows: list[dict],
    feature_names_out: list[str],
    *,
    judge_only: bool = False,
    top_n_categories: int = DEFAULT_TOP_N_CATEGORIES,
) -> np.ndarray:
    """
    Build a numeric feature matrix from rows.
    Modifies feature_names_out in place to record the column order.
    """
    feature_names_out.clear()

    # 1. Judge attributes (always included)
    feature_names_out.extend(JUDGE_ATTRIBUTES)

    if judge_only:
        return np.array([[row[a] for a in JUDGE_ATTRIBUTES] for row in rows], dtype=float)

    # 2. Collect all unique models and variants for one-hot encoding
    all_models = sorted({row["model"] for row in rows})
    all_variants = sorted({row["variant"] for row in rows})
    category_counts: Counter[str] = Counter()
    for row in rows:
        category_counts.update(str(c) for c in row.get("categories", []) if c)
    top_categories = [cat for cat, _ in category_counts.most_common(top_n_categories)]
    # drop first level (reference) to avoid multicollinearity
    model_dummies = all_models[1:]
    variant_dummies = all_variants[1:]

    feature_names_out.append("temperature")
    feature_names_out.append("rationale_words")
    feature_names_out.append("evidence_words")
    feature_names_out.append("output_valid")
    for m in model_dummies:
        feature_names_out.append(f"model_{m}")
    for v in variant_dummies:
        feature_names_out.append(f"variant_{v}")
    for cat in top_categories:
        feature_names_out.append(f"cat_{cat}")
    feature_names_out.append("cat_other")

    matrix = []
    for row in rows:
        vec: list[float] = [row[a] for a in JUDGE_ATTRIBUTES]
        vec.append(row["temperature"])
        vec.append(float(row["rationale_words"]))
        vec.append(float(row["evidence_words"]))
        vec.append(float(row["output_valid"]))
        for m in model_dummies:
            vec.append(1.0 if row["model"] == m else 0.0)
        for v in variant_dummies:
            vec.append(1.0 if row["variant"] == v else 0.0)
        row_cats = row.get("categories") or []
        matched_known_category = False
        for cat in top_categories:
            has_category = cat in row_cats
            matched_known_category = matched_known_category or has_category
            vec.append(1.0 if cat in row_cats else 0.0)
        vec.append(0.0 if matched_known_category else 1.0)
        matrix.append(vec)
    return np.array(matrix, dtype=float)


def make_grouped_cv_splits(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    splitter = StratifiedGroupKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(splitter.split(features, target, groups))
    audit_rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        audit_rows.append({
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_questions": int(len(train_groups)),
            "test_questions": int(len(test_groups)),
            "overlap_questions": int(len(overlap)),
            "test_positive_rate": float(target[test_idx].mean()),
        })
    return splits, audit_rows


# ---------- model fitting ----------

def fit_and_explain(
    rows: list[dict],
    *,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    cv_n_jobs: int,
    shap_sample_size: int,
    skip_shap: bool,
    judge_only: bool = False,
    top_n_categories: int = DEFAULT_TOP_N_CATEGORIES,
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    feature_names: list[str] = []
    features = build_feature_matrix(
        rows,
        feature_names,
        judge_only=judge_only,
        top_n_categories=top_n_categories,
    )
    target = np.array([row["forecast_correct"] for row in rows], dtype=int)
    groups = np.array([row["id"] for row in rows], dtype=int)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    # Grouped CV: all rows for the same question ID stay together.
    cv_splits, fold_audit_rows = make_grouped_cv_splits(
        features,
        target,
        groups,
        random_state=random_state,
    )
    probabilities = cross_val_predict(
        clf, features, target, groups=groups,
        cv=cv_splits, method="predict_proba", n_jobs=cv_n_jobs,
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "n_rows": float(len(rows)),
        "n_questions": float(len(set(groups))),
        "positive_rate": float(target.mean()),
        "cv_roc_auc": float(roc_auc_score(target, probabilities)),
        "cv_accuracy": float(accuracy_score(target, predictions)),
        "judge_only": judge_only,
        "cv_folds": float(N_CV_SPLITS),
        "max_fold_question_overlap": float(max(r["overlap_questions"] for r in fold_audit_rows)),
        "explanation_kind": "shap",
    }

    clf.fit(features, target)

    if skip_shap:
        metrics["shap_n_rows"] = 0.0
        metrics["explanation_kind"] = "rf_impurity_importance"
        feature_rows: list[dict] = []
        importances = clf.feature_importances_
        for i, fname in enumerate(feature_names):
            col = features[:, i]
            feature_rows.append({
                "feature": fname,
                "mean_abs_shap": float(importances[i]),
                "mean_value": float(col.mean()),
                "mean_value_correct": float(col[target == 1].mean()),
                "mean_value_incorrect": float(col[target == 0].mean()),
                "correct_minus_incorrect": float(col[target == 1].mean() - col[target == 0].mean()),
                "value_shap_correlation": 0.0,
            })
        feature_rows.sort(key=lambda r: float(r["mean_abs_shap"]), reverse=True)
        return feature_rows, metrics, [], fold_audit_rows

    import shap

    if shap_sample_size and shap_sample_size > 0 and len(rows) > shap_sample_size:
        rng = np.random.default_rng(random_state)
        shap_idx = np.sort(rng.choice(len(rows), size=shap_sample_size, replace=False))
    else:
        shap_idx = np.arange(len(rows))
    shap_features = features[shap_idx]
    shap_rows = [rows[int(i)] for i in shap_idx]
    metrics["shap_n_rows"] = float(len(shap_idx))

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(shap_features)
    if isinstance(shap_vals, list):
        shap_mat = np.asarray(shap_vals[-1], dtype=float)
    else:
        shap_mat = np.asarray(shap_vals, dtype=float)
        if shap_mat.ndim == 3:
            shap_mat = shap_mat[:, :, -1]

    mean_abs = np.abs(shap_mat).mean(axis=0)
    feature_rows: list[dict] = []
    for i, fname in enumerate(feature_names):
        col = features[:, i]
        shap_col_values = shap_features[:, i]
        shap_col = shap_mat[:, i]
        corr = float(np.corrcoef(shap_col_values, shap_col)[0, 1]) if shap_col_values.std() > 0 else 0.0
        feature_rows.append({
            "feature": fname,
            "mean_abs_shap": float(mean_abs[i]),
            "mean_value": float(col.mean()),
            "mean_value_correct": float(col[target == 1].mean()),
            "mean_value_incorrect": float(col[target == 0].mean()),
            "correct_minus_incorrect": float(col[target == 1].mean() - col[target == 0].mean()),
            "value_shap_correlation": corr if not math.isnan(corr) else 0.0,
        })
    feature_rows.sort(key=lambda r: float(r["mean_abs_shap"]), reverse=True)

    detail_rows: list[dict] = []
    for row, shap_row in zip(shap_rows, shap_mat):
        dr: dict = {
            "judge": row["judge"], "model": row["model"],
            "temperature_dir": row["temperature_dir"], "variant": row["variant"],
            "id": row["id"], "forecast_correct": row["forecast_correct"],
        }
        for attr in JUDGE_ATTRIBUTES:
            dr[attr] = row[attr]
        for i, fname in enumerate(feature_names):
            dr[f"shap_{fname}"] = float(shap_row[i])
        detail_rows.append(dr)

    return feature_rows, metrics, detail_rows, fold_audit_rows


# ---------- output ----------

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(
    path: Path,
    metrics_by_key: dict[str, dict],
    feature_rows_by_key: dict[str, list[dict]],
) -> None:
    lines = [
        "# SHAP Analysis (revised — grouped CV + controls)",
        "",
        "Cross-validation is grouped by question ID so no question appears in both",
        "train and test. Two models are fit per judge dataset:",
        "- **judge_only**: 6 judge attributes only (grouped CV)",
        "- **full**: judge attributes + model, variant, temperature, rationale length,",
        "  evidence length, output validity, category flags (grouped CV)",
        "- Question ID is controlled through the grouped split rather than used as a",
        "  feature, because held-out questions have unseen IDs.",
        "- SHAP rows are judged forecast attempts. Use",
        "  `analysis/metrics_by_model_temperature_variant.csv` for explicit",
        "  valid-output conditional and coverage-aware forecasting metrics.",
        "",
        "ROC-AUC difference between judge_only and full reveals confound magnitude.",
        "When run with `--skip-shap`, feature rankings are RandomForest impurity",
        "importances rather than SHAP values.",
        "",
    ]
    for key in sorted(metrics_by_key):
        metrics = metrics_by_key[key]
        feature_rows = feature_rows_by_key[key]
        label = "judge attributes only" if metrics["judge_only"] else "full model with controls"
        importance_label = "Mean |SHAP|" if metrics.get("explanation_kind") == "shap" else "RF importance"
        corr_label = "Value-SHAP Corr" if metrics.get("explanation_kind") == "shap" else "Value-Importance Corr"
        lines += [
            f"## {key}  ({label})",
            "",
            f"- Rows: `{int(metrics['n_rows'])}` across `{int(metrics['n_questions'])}` questions",
            f"- SHAP explanation rows: `{int(metrics['shap_n_rows'])}`",
            f"- Positive rate: `{metrics['positive_rate']:.3f}`",
            f"- CV ROC-AUC (grouped): `{metrics['cv_roc_auc']:.3f}`",
            f"- CV Accuracy (grouped): `{metrics['cv_accuracy']:.3f}`",
            f"- Max train/test question overlap across folds: `{int(metrics['max_fold_question_overlap'])}`",
            "",
            f"| Feature | Mean | Correct − Incorrect | {importance_label} | {corr_label} |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in feature_rows[:15]:  # top 15 features
            lines.append(
                f"| {row['feature']} | {row['mean_value']:.3f} | "
                f"{row['correct_minus_incorrect']:.3f} | {row['mean_abs_shap']:.5f} | "
                f"{row['value_shap_correlation']:.3f} |"
            )
        lines.append("")
        if feature_rows:
            top = feature_rows[0]
            lines.append(f"Top feature: `{top['feature']}` ({importance_label} = `{top['mean_abs_shap']:.5f}`).")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)

    judge_dirs: dict[str, Path] = {}
    for item in args.judge_dirs:
        name, raw_path = item.split("=", maxsplit=1)
        judge_dirs[name] = Path(raw_path)

    all_rows: list[dict] = []
    forecast_models = set(args.forecast_models) if args.forecast_models else None
    for judge_name, judge_dir in judge_dirs.items():
        if not judge_dir.exists():
            continue
        rows = load_judge_rows(judge_name, judge_dir, args.results_root, dataset, forecast_models)
        print(f"Loaded {len(rows)} rows for judge {judge_name}")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No judged rows found.")

    datasets: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        datasets[row["judge"]].append(row)
    combined = build_combined_rows(all_rows)
    if args.combined_only:
        datasets = {"combined_mean": combined}
    elif combined:
        datasets["combined_mean"] = combined

    metrics_summary: list[dict] = []
    feature_summary: list[dict] = []
    fold_audit_summary: list[dict] = []
    metrics_by_key: dict[str, dict] = {}
    feature_rows_by_key: dict[str, list[dict]] = {}

    for ds_name, rows in sorted(datasets.items()):
        for judge_only in (True, False):
            key = f"{ds_name}__{'judge_only' if judge_only else 'full'}"
            print(f"Fitting {key} ({len(rows)} rows)...")
            feat_rows, metrics, detail_rows, fold_audit_rows = fit_and_explain(
                rows,
                random_state=args.random_state,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                cv_n_jobs=args.cv_n_jobs,
                shap_sample_size=args.shap_sample_size,
                skip_shap=args.skip_shap,
                judge_only=judge_only,
                top_n_categories=args.top_n_categories,
            )
            metrics_by_key[key] = metrics
            feature_rows_by_key[key] = feat_rows

            metrics_summary.append({"dataset": key, **metrics})
            for audit_row in fold_audit_rows:
                fold_audit_summary.append({"dataset": key, **audit_row})
            for fr in feat_rows:
                feature_summary.append({"dataset": key, **fr})

            if detail_rows:
                detail_fieldnames = (
                    ["judge", "model", "temperature_dir", "variant", "id", "forecast_correct"]
                    + JUDGE_ATTRIBUTES
                    + [f"shap_{f}" for f in ([a for a in JUDGE_ATTRIBUTES] if judge_only
                       else [r["feature"] for r in feat_rows])]
                )
                write_csv(args.output_dir / f"{key}_details.csv", detail_rows,
                          list(dict.fromkeys(detail_fieldnames)))

    write_csv(
        args.output_dir / "metrics_summary.csv",
        metrics_summary,
        [
            "dataset",
            "n_rows",
            "n_questions",
            "positive_rate",
            "cv_roc_auc",
            "cv_accuracy",
            "judge_only",
            "cv_folds",
            "max_fold_question_overlap",
            "shap_n_rows",
            "explanation_kind",
        ],
    )
    write_csv(
        args.output_dir / "fold_audit.csv",
        fold_audit_summary,
        [
            "dataset",
            "fold",
            "train_rows",
            "test_rows",
            "train_questions",
            "test_questions",
            "overlap_questions",
            "test_positive_rate",
        ],
    )
    write_csv(
        args.output_dir / "feature_importance.csv",
        feature_summary,
        ["dataset", "feature", "mean_abs_shap", "mean_value",
         "mean_value_correct", "mean_value_incorrect",
         "correct_minus_incorrect", "value_shap_correlation"],
    )
    # Keep metrics_summary.json for backward compat (combined_mean judge_only ≈ old behaviour)
    compat = {k: v for k, v in metrics_by_key.items() if "judge_only" in k}
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps({k.replace("__judge_only", ""): v for k, v in compat.items()}, indent=2),
        encoding="utf-8",
    )
    write_summary_md(args.output_dir / "summary.md", metrics_by_key, feature_rows_by_key)
    print("Done.")


if __name__ == "__main__":
    main()
