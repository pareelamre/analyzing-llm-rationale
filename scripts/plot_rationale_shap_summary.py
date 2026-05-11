#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
DETAIL_PATH = ROOT / "analysis" / "partial_shap_analysis" / "combined_mean_details.csv"
IMPORTANCE_PATH = ROOT / "analysis" / "partial_shap_analysis" / "feature_importance.csv"
OUT_PNG = ROOT / "paper" / "rationale_shap_summary.png"
OUT_PDF = ROOT / "paper" / "rationale_shap_summary.pdf"

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


def load_importance() -> dict[str, float]:
    values: dict[str, float] = {}
    with IMPORTANCE_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "combined_mean":
                values[row["feature"]] = float(row["mean_abs_shap"])
    return values


def load_detail_rows() -> list[dict[str, str]]:
    with DETAIL_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    importance = load_importance()
    rows = load_detail_rows()
    order = sorted(ATTRIBUTES, key=lambda name: importance[name], reverse=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.8,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 6.0,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.4, 2.35), constrained_layout=False)
    grid = GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1.0, 0.035, 0.36, 1.0],
        wspace=0.02,
    )
    ax_swarm = fig.add_subplot(grid[0, 0])
    ax_cbar = fig.add_subplot(grid[0, 1])
    ax_bar = fig.add_subplot(grid[0, 3])

    rng = np.random.default_rng(11)
    cmap = plt.get_cmap("coolwarm")
    scatter = None
    max_points_per_feature = 6000

    for y_index, attribute in enumerate(order):
        shap_values = np.asarray(
            [float(row[f"shap_{attribute}"]) for row in rows], dtype=float
        )
        judge_scores = np.asarray([float(row[attribute]) for row in rows], dtype=float)

        if len(shap_values) > max_points_per_feature:
            sample = rng.choice(
                len(shap_values), size=max_points_per_feature, replace=False
            )
            shap_values = shap_values[sample]
            judge_scores = judge_scores[sample]

        jitter = rng.normal(0.0, 0.065, size=len(shap_values))
        scatter = ax_swarm.scatter(
            shap_values,
            y_index + jitter,
            c=judge_scores,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=3.6,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )

    ax_swarm.axvline(0, color="#7a7a7a", linewidth=0.55, alpha=0.9)
    ax_swarm.set_yticks(np.arange(len(order)))
    ax_swarm.set_yticklabels([LABELS[attribute] for attribute in order])
    ax_swarm.invert_yaxis()
    ax_swarm.set_xlim(-0.215, 0.205)
    ax_swarm.set_xlabel("SHAP value (impact on model output)")
    ax_swarm.grid(axis="x", color="#e0e0e0", linewidth=0.35, alpha=0.85)
    ax_swarm.tick_params(axis="y", length=0, pad=1.6)
    ax_swarm.spines["top"].set_visible(False)
    ax_swarm.spines["right"].set_visible(False)
    ax_swarm.spines["left"].set_visible(False)

    if scatter is None:
        raise RuntimeError("No rows were available for plotting.")

    cbar = fig.colorbar(scatter, cax=ax_cbar)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(which="both", labelsize=5.6, length=0, pad=1.5)
    cbar.set_label("Feature value", fontsize=5.2, labelpad=1.0)

    bar_values = [importance[attribute] for attribute in order]
    y_positions = np.arange(len(order))
    ax_bar.barh(
        y_positions,
        bar_values,
        height=0.55,
        color="#ff0052",
        edgecolor="none",
        alpha=0.96,
    )
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels([LABELS[attribute] for attribute in order], fontsize=5.4)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, max(bar_values) * 1.18)
    ax_bar.set_xlabel(r"mean(|SHAP value|)")
    ax_bar.grid(axis="x", color="#e0e0e0", linewidth=0.35, alpha=0.85)
    ax_bar.tick_params(axis="y", which="both", left=False, labelleft=True, pad=1.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)

    for y_index, value in enumerate(bar_values):
        ax_bar.text(
            value + max(bar_values) * 0.018,
            y_index,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=5.6,
            color="#ff0052",
        )

    fig.text(0.26, 0.035, "(a) SHAP beeswarm plot", ha="center", va="center", fontsize=6.2)
    fig.text(0.775, 0.035, "(b) Mean absolute SHAP value plot", ha="center", va="center", fontsize=6.2)

    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.245, top=0.985)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
