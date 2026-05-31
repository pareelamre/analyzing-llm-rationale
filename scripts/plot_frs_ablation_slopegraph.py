#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
RESULTS_ROOT = ROOT / "results"
OUT_CSV = ROOT / "analysis" / "frs_ablation_metrics.csv"
OUT_PNG = ROOT / "paper" / "frs_ablation_slopegraph.png"
OUT_PDF = ROOT / "paper" / "frs_ablation_slopegraph.pdf"

MODEL_RUNS = [
    ("Qwen2.5-7b-instruct", "temperature_000"),
    ("Qwen3-32B", "temperature_0"),
    ("GPT-OSS-120B", "temperature_00"),
]

MODEL_COLORS = {
    "Qwen2.5-7b-instruct": "#4C78A8",
    "Qwen3-32B": "#F58518",
    "GPT-OSS-120B": "#54A24B",
}

PANELS = [
    {
        "title": "Accuracy",
        "metric": "accuracy",
        "higher_is_better": True,
        "fmt": "{:.3f}",
    },
    {
        "title": "Brier score",
        "metric": "brier_score",
        "higher_is_better": False,
        "fmt": "{:.3f}",
    },
    {
        "title": "ECE",
        "metric": "ece",
        "higher_is_better": False,
        "fmt": "{:.3f}",
    },
]


def load_targets() -> dict[int, int]:
    rows = json.loads(DATASET_PATH.read_text())
    return {
        int(row["id"]): 1 if str(row["answer"]).strip().lower() == "yes" else 0
        for row in rows
        if str(row.get("answer")).strip().lower() in {"yes", "no"}
    }


def normalize_answer(value: object) -> str | None:
    answer = str(value or "").strip().lower()
    return answer if answer in {"yes", "no"} else None


def normalize_confidence(value: object) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
        return None
    return confidence


def expected_ids_for_path(path: Path) -> set[int]:
    return {int(row["id"]) for row in json.loads(path.read_text()) if "id" in row}


def compute_metrics(path: Path, targets: dict[int, int]) -> dict[str, float]:
    rows = json.loads(path.read_text())
    examples: list[tuple[int, int, float, float]] = []
    malformed = 0
    for row in rows:
        row_id = int(row.get("id"))
        if row_id not in targets:
            malformed += 1
            continue
        answer = normalize_answer(row.get("predicted_answer"))
        confidence = normalize_confidence(row.get("confidence"))
        if answer is None or confidence is None:
            malformed += 1
            continue
        predicted_label = 1 if answer == "yes" else 0
        p_yes = confidence if predicted_label == 1 else 1.0 - confidence
        examples.append((predicted_label, targets[row_id], p_yes, confidence))

    if not examples:
        raise ValueError(f"No valid prediction rows in {path}")

    n = len(examples)
    accuracy = sum(predicted == target for predicted, target, _p_yes, _confidence in examples) / n
    brier = sum((p_yes - target) ** 2 for _predicted, target, p_yes, _confidence in examples) / n

    bins = 10
    bin_counts = [0] * bins
    bin_confidence = [0.0] * bins
    bin_accuracy = [0.0] * bins
    for predicted, target, _p_yes, confidence in examples:
        idx = min(int(confidence * bins), bins - 1)
        bin_counts[idx] += 1
        bin_confidence[idx] += confidence
        bin_accuracy[idx] += int(predicted == target)

    ece = 0.0
    for count, conf_sum, acc_sum in zip(bin_counts, bin_confidence, bin_accuracy):
        if count:
            ece += (count / n) * abs((acc_sum / count) - (conf_sum / count))

    return {
        "n_scored": float(n),
        "malformed_rows": float(malformed),
        "accuracy": accuracy,
        "brier_score": brier,
        "ece": ece,
    }


def build_metrics() -> pd.DataFrame:
    targets = load_targets()
    rows = []
    for model, with_temp in MODEL_RUNS:
        paths = {
            "with-FRS": RESULTS_ROOT
            / model
            / with_temp
            / "results_variant0_neutral_baseline.json",
            "without-FRS": RESULTS_ROOT
            / model
            / "without_frs"
            / "temperature_0"
            / "results_variant0_neutral_baseline.json",
        }
        expected_ids = set.union(*(expected_ids_for_path(path) for path in paths.values()))
        for condition, path in paths.items():
            metrics = compute_metrics(path, targets)
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "temperature_dir": with_temp if condition == "with-FRS" else "without_frs/temperature_0",
                    "coverage_rate": metrics["n_scored"] / len(expected_ids),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def is_beneficial(before: float, after: float, higher_is_better: bool) -> bool:
    return after > before if higher_is_better else after < before


def main() -> None:
    metrics = build_metrics()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_CSV, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(PANELS),
        figsize=(7.35, 2.45),
        constrained_layout=False,
    )

    x_positions = [0, 1]
    x_labels = ["with-FRS", "without-FRS"]
    beneficial_color = "#0F766E"
    harmful_color = "#B91C1C"

    for ax, panel in zip(axes, PANELS):
        metric = str(panel["metric"])
        values = metrics[metric]
        pad = max((values.max() - values.min()) * 0.22, 0.01)
        ax.set_ylim(values.min() - pad, values.max() + pad)
        ax.set_title(str(panel["title"]), fontweight="bold", pad=5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.45, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for model, _temp in MODEL_RUNS:
            model_rows = metrics[metrics["model"] == model].set_index("condition")
            y_with = float(model_rows.loc["with-FRS", metric])
            y_without = float(model_rows.loc["without-FRS", metric])
            beneficial = is_beneficial(y_with, y_without, bool(panel["higher_is_better"]))
            shift_color = beneficial_color if beneficial else harmful_color
            ax.plot(
                x_positions,
                [y_with, y_without],
                color=shift_color,
                linewidth=1.45,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(
                x_positions,
                [y_with, y_without],
                s=24,
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

        if not bool(panel["higher_is_better"]):
            ax.text(
                0.02,
                0.93,
                "lower is better",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=5.8,
                color="#475569",
            )
        else:
            ax.text(
                0.02,
                0.93,
                "higher is better",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=5.8,
                color="#475569",
            )

    model_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=model, markersize=5)
        for model, color in MODEL_COLORS.items()
    ]
    shift_handles = [
        plt.Line2D([0, 1], [0, 0], color=beneficial_color, linewidth=1.8, label="beneficial shift"),
        plt.Line2D([0, 1], [0, 0], color=harmful_color, linewidth=1.8, label="harmful shift"),
    ]
    fig.legend(
        handles=model_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.58, 0.065),
        fontsize=6.6,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.legend(
        handles=shift_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.58, 0.015),
        fontsize=6.3,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    fig.text(
        0.015,
        0.035,
        "Line color marks whether moving from with-FRS to without-FRS improves the metric.",
        ha="left",
        va="center",
        fontsize=6.2,
        color="#374151",
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.31, wspace=0.38)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
