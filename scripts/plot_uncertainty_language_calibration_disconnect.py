#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
RESULTS_ROOT = ROOT / "results"
OUT_CSV = ROOT / "analysis" / "uncertainty_language_calibration.csv"
OUT_PNG = ROOT / "paper" / "uncertainty_language_calibration_disconnect.png"
OUT_PDF = ROOT / "paper" / "uncertainty_language_calibration_disconnect.pdf"

MODEL_RUNS = [
    ("Qwen2.5-7b-instruct", "temperature_000"),
    ("Qwen3-32B", "temperature_0"),
    ("GPT-OSS-120B", "temperature_025"),
]

MODEL_LABELS = {
    "Qwen2.5-7b-instruct": "Qwen2.5",
    "Qwen3-32B": "Qwen3",
    "GPT-OSS-120B": "GPT-OSS",
}

MODEL_COLORS = {
    "Qwen2.5-7b-instruct": "#4C78A8",
    "Qwen3-32B": "#F58518",
    "GPT-OSS-120B": "#54A24B",
}

VARIANTS = {
    "V0": "variant0_neutral_baseline",
    "V7": "variant7_uncertainty_language",
}

UNCERTAINTY_RE = re.compile(
    r"\b("
    r"likely|unlikely|possible|possibly|probable|probably|uncertain|uncertainty|"
    r"may|might|could|suggests?|appears?|seems?|risk|risks|chance|chances"
    r")\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\b\w+\b")


def load_targets() -> dict[int, int]:
    rows = json.loads(DATASET_PATH.read_text())
    return {
        int(row["id"]): 1 if str(row["answer"]).strip().lower() == "yes" else 0
        for row in rows
        if str(row.get("answer")).strip().lower() in {"yes", "no"}
    }


def load_results(model: str, temperature_dir: str, variant: str) -> list[dict[str, object]]:
    path = RESULTS_ROOT / model / temperature_dir / f"results_{variant}.json"
    return json.loads(path.read_text())


def lexical_frequency(rows: list[dict[str, object]]) -> tuple[float, float, int, int]:
    hits = 0
    tokens = 0
    rows_with_hit = 0
    for row in rows:
        rationale = str(row.get("rationale") or "")
        row_hits = len(UNCERTAINTY_RE.findall(rationale))
        hits += row_hits
        tokens += len(WORD_RE.findall(rationale))
        rows_with_hit += int(row_hits > 0)
    per_100_words = 100.0 * hits / tokens if tokens else float("nan")
    share_rows = rows_with_hit / len(rows) if rows else float("nan")
    return per_100_words, share_rows, hits, tokens


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


def calibration_metrics(rows: list[dict[str, object]], targets: dict[int, int]) -> tuple[float, float, int]:
    examples: list[tuple[int, int, float, float]] = []
    for row in rows:
        row_id = int(row.get("id"))
        if row_id not in targets:
            continue
        answer = normalize_answer(row.get("predicted_answer"))
        confidence = normalize_confidence(row.get("confidence"))
        if answer is None or confidence is None:
            continue
        predicted = 1 if answer == "yes" else 0
        p_yes = confidence if predicted == 1 else 1.0 - confidence
        examples.append((predicted, targets[row_id], p_yes, confidence))

    if not examples:
        raise ValueError("No valid prediction rows available.")

    n = len(examples)
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
    return brier, ece, n


def build_rows() -> list[dict[str, object]]:
    targets = load_targets()
    rows_out = []
    for model, temperature_dir in MODEL_RUNS:
        variant_rows = {
            label: load_results(model, temperature_dir, variant)
            for label, variant in VARIANTS.items()
        }
        metrics = {}
        for label, rows in variant_rows.items():
            per_100, share_rows, hits, tokens = lexical_frequency(rows)
            brier, ece, n_scored = calibration_metrics(rows, targets)
            metrics[label] = {
                "uncertainty_per_100_words": per_100,
                "uncertainty_row_share": share_rows,
                "uncertainty_hits": hits,
                "rationale_tokens": tokens,
                "brier_score": brier,
                "ece": ece,
                "n_scored": n_scored,
            }
            rows_out.append(
                {
                    "model": model,
                    "temperature_dir": temperature_dir,
                    "variant": label,
                    **metrics[label],
                }
            )
        rows_out.append(
            {
                "model": model,
                "temperature_dir": temperature_dir,
                "variant": "V7_minus_V0",
                "uncertainty_per_100_words": metrics["V7"]["uncertainty_per_100_words"]
                - metrics["V0"]["uncertainty_per_100_words"],
                "uncertainty_row_share": metrics["V7"]["uncertainty_row_share"]
                - metrics["V0"]["uncertainty_row_share"],
                "uncertainty_hits": metrics["V7"]["uncertainty_hits"]
                - metrics["V0"]["uncertainty_hits"],
                "rationale_tokens": metrics["V7"]["rationale_tokens"]
                - metrics["V0"]["rationale_tokens"],
                "brier_score": metrics["V7"]["brier_score"] - metrics["V0"]["brier_score"],
                "ece": metrics["V7"]["ece"] - metrics["V0"]["ece"],
                "n_scored": metrics["V7"]["n_scored"],
            }
        )
    return rows_out


def write_csv(rows: list[dict[str, object]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "temperature_dir",
        "variant",
        "uncertainty_per_100_words",
        "uncertainty_row_share",
        "uncertainty_hits",
        "rationale_tokens",
        "brier_score",
        "ece",
        "n_scored",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    by_model_variant = {
        (str(row["model"]), str(row["variant"])): row
        for row in rows
    }

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

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(7.3, 2.75),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
        constrained_layout=False,
    )

    x = np.arange(len(MODEL_RUNS))
    width = 0.34
    v0_values = [
        float(by_model_variant[(model, "V0")]["uncertainty_per_100_words"])
        for model, _temp in MODEL_RUNS
    ]
    v7_values = [
        float(by_model_variant[(model, "V7")]["uncertainty_per_100_words"])
        for model, _temp in MODEL_RUNS
    ]
    ax_left.bar(
        x - width / 2,
        v0_values,
        width,
        color="#CBD5E1",
        edgecolor="#94A3B8",
        linewidth=0.4,
        label="V0 neutral",
    )
    ax_left.bar(
        x + width / 2,
        v7_values,
        width,
        color="#64748B",
        edgecolor="#334155",
        linewidth=0.4,
        label="V7 uncertainty language",
    )
    for index, (model, _temp) in enumerate(MODEL_RUNS):
        ax_left.plot(
            [index - width / 2, index + width / 2],
            [v0_values[index], v7_values[index]],
            color=MODEL_COLORS[model],
            linewidth=1.0,
            alpha=0.9,
        )
    ax_left.set_title("Lexical uncertainty", fontweight="bold", pad=5)
    ax_left.set_ylabel("Uncertainty terms per 100 words")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([MODEL_LABELS[model] for model, _temp in MODEL_RUNS], rotation=0)
    ax_left.grid(axis="y", color="#CBD5E1", linewidth=0.45, alpha=0.8)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.legend(frameon=False, fontsize=6.1, loc="upper left")

    ax_right.axvline(0, color="#111827", linewidth=0.7, zorder=1)
    ax_right.axvspan(0, 0.04, color="#F7E4E4", alpha=0.85, zorder=0)
    y_positions = np.arange(len(MODEL_RUNS))
    offsets = {"Brier score": -0.12, "ECE": 0.12}
    metric_styles = {
        "Brier score": {"key": "brier_score", "marker": "o", "color": "#BE123C"},
        "ECE": {"key": "ece", "marker": "s", "color": "#7C2D12"},
    }
    max_delta = 0.0
    for model, _temp in MODEL_RUNS:
        delta_row = by_model_variant[(model, "V7_minus_V0")]
        max_delta = max(
            max_delta,
            abs(float(delta_row["brier_score"])),
            abs(float(delta_row["ece"])),
        )
    xlim = max(0.04, math.ceil(max_delta * 120) / 100)

    for row_index, (model, _temp) in enumerate(MODEL_RUNS):
        delta_row = by_model_variant[(model, "V7_minus_V0")]
        for metric_name, style in metric_styles.items():
            value = float(delta_row[str(style["key"])])
            y = row_index + offsets[metric_name]
            ax_right.hlines(
                y,
                xmin=min(0, value),
                xmax=max(0, value),
                color=str(style["color"]),
                linewidth=1.25,
                alpha=0.9,
            )
            ax_right.scatter(
                value,
                y,
                marker=str(style["marker"]),
                s=25,
                color=str(style["color"]),
                edgecolor="white",
                linewidth=0.45,
                label=metric_name if row_index == 0 else None,
                zorder=3,
            )

    ax_right.set_title("Calibration change: V7 - V0", fontweight="bold", pad=5)
    ax_right.set_xlabel("Delta metric (positive = worse)")
    ax_right.set_yticks(y_positions)
    ax_right.set_yticklabels([MODEL_LABELS[model] for model, _temp in MODEL_RUNS])
    ax_right.set_xlim(-xlim * 0.08, xlim)
    ax_right.grid(axis="x", color="#CBD5E1", linewidth=0.45, alpha=0.8)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.legend(frameon=False, fontsize=6.1, loc="lower right")
    ax_right.text(
        0.98,
        0.93,
        "all shifts harmful",
        transform=ax_right.transAxes,
        ha="right",
        va="center",
        fontsize=6.0,
        color="#B91C1C",
    )

    fig.text(
        0.015,
        0.035,
        "V7 substantially increases cautious wording, but Brier score and ECE worsen for the matched runs.",
        ha="left",
        va="center",
        fontsize=6.3,
        color="#374151",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.25, wspace=0.32)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
