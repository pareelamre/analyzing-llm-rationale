#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMPORTANCE_PATH = ROOT / "analysis" / "shap_analysis" / "feature_importance.csv"
OUT_PNG = ROOT / "paper" / "shap_importance_attribute_gaps.png"
OUT_PDF = ROOT / "paper" / "shap_importance_attribute_gaps.pdf"

DATASET = "combined_mean"
ATTRIBUTES = [
    "plausibility",
    "completeness",
    "source_consistency",
    "non_hallucination",
    "informativeness",
    "conciseness",
]

LABELS = {
    "plausibility": "Plausibility",
    "completeness": "Completeness",
    "source_consistency": "Source consistency",
    "non_hallucination": "Non-hallucination",
    "informativeness": "Informativeness",
    "conciseness": "Conciseness",
}

HIGHLIGHT = {
    "completeness": "#2F6FAD",
    "plausibility": "#D55E00",
}
NEUTRAL = "#9AA4B2"
TEXT = "#1F2937"


def load_rows() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with IMPORTANCE_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != DATASET or row["feature"] not in ATTRIBUTES:
                continue
            rows[row["feature"]] = {
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "mean_value_correct": float(row["mean_value_correct"]),
                "mean_value_incorrect": float(row["mean_value_incorrect"]),
                "correct_minus_incorrect": float(row["correct_minus_incorrect"]),
            }
    missing = sorted(set(ATTRIBUTES) - set(rows))
    if missing:
        raise ValueError(f"Missing feature importance rows for: {missing}")
    return rows


def color_for(attribute: str) -> str:
    return HIGHLIGHT.get(attribute, NEUTRAL)


def main() -> None:
    rows = load_rows()
    order = sorted(ATTRIBUTES, key=lambda name: rows[name]["mean_abs_shap"])
    y = np.arange(len(order))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.6,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_imp, ax_gap) = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.75),
        gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.42},
    )

    labels = [LABELS[name] for name in order]
    colors = [color_for(name) for name in order]
    importance = [rows[name]["mean_abs_shap"] for name in order]
    gaps = [rows[name]["correct_minus_incorrect"] for name in order]
    correct = [rows[name]["mean_value_correct"] for name in order]
    incorrect = [rows[name]["mean_value_incorrect"] for name in order]

    ax_imp.barh(y, importance, height=0.58, color=colors, edgecolor="white", linewidth=0.5)
    ax_imp.set_yticks(y)
    ax_imp.set_yticklabels(labels)
    ax_imp.set_xlabel(r"Mean |SHAP value|")
    ax_imp.set_title("Diagnostic importance")
    ax_imp.set_xlim(0, max(importance) * 1.28)
    ax_imp.grid(axis="x", color="#E5E7EB", linewidth=0.45)
    ax_imp.set_axisbelow(True)
    ax_imp.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax_imp.spines[spine].set_visible(False)
    for yi, attr, value in zip(y, order, importance):
        weight = "bold" if attr in HIGHLIGHT else "normal"
        ax_imp.text(
            value + max(importance) * 0.025,
            yi,
            f"{value:.3f}",
            va="center",
            ha="left",
            color=color_for(attr),
            fontsize=6.6,
            fontweight=weight,
        )

    ax_gap.axvline(0, color="#6B7280", linewidth=0.7)
    for yi, attr, lo, hi, gap in zip(y, order, incorrect, correct, gaps):
        color = color_for(attr)
        ax_gap.plot([lo, hi], [yi, yi], color=color, linewidth=1.7, solid_capstyle="round")
        ax_gap.scatter([lo], [yi], s=20, color="white", edgecolor=color, linewidth=0.9, zorder=3)
        ax_gap.scatter([hi], [yi], s=24, color=color, edgecolor="white", linewidth=0.45, zorder=4)
        weight = "bold" if attr in HIGHLIGHT else "normal"
        ax_gap.text(
            hi + 0.018,
            yi,
            f"+{gap:.3f}",
            va="center",
            ha="left",
            color=color,
            fontsize=6.5,
            fontweight=weight,
        )

    ax_gap.set_yticks(y)
    ax_gap.set_yticklabels([])
    ax_gap.set_xlabel("Judge score: incorrect \u2192 correct forecasts")
    ax_gap.set_title("Correct-minus-incorrect gap")
    ax_gap.set_xlim(0.48, 0.96)
    ax_gap.grid(axis="x", color="#E5E7EB", linewidth=0.45)
    ax_gap.set_axisbelow(True)
    ax_gap.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax_gap.spines[spine].set_visible(False)

    fig.text(
        0.02,
        0.02,
        "Combined mean of Gemma-4-31B-it and Kimi-K2.5 judge attributes. Open circles show incorrect forecasts; filled circles show correct forecasts.",
        ha="left",
        va="bottom",
        fontsize=5.7,
        color="#4B5563",
    )

    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.24, top=0.86)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
