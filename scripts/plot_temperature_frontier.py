#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "analysis" / "metrics_by_model_temperature.csv"
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
OUT_PNG = ROOT / "paper" / "temperature_frontier_metrics.png"
OUT_PDF = ROOT / "paper" / "temperature_frontier_metrics.pdf"

MODEL_ORDER = [
    "Qwen2.5-7b-instruct",
    "Qwen3-32B",
    "GPT-OSS-120B",
]

MODEL_COLORS = {
    "Qwen2.5-7b-instruct": "#4C78A8",
    "Qwen3-32B": "#F58518",
    "GPT-OSS-120B": "#54A24B",
}

PANELS = [
    {
        "title": "Accuracy",
        "column": "mean_accuracy",
        "ylabel": "Mean accuracy",
        "higher_is_better": True,
    },
    {
        "title": "Brier score",
        "column": "mean_brier_score",
        "ylabel": "Mean Brier score",
        "higher_is_better": False,
    },
    {
        "title": "Missing-output rate",
        "column": "missing_rate",
        "ylabel": "Missing outputs",
        "higher_is_better": False,
    },
]


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_PATH)
    dataset_size = len(json.loads(DATASET_PATH.read_text()))
    df["missing_rate"] = df["total_missing"] / (df["runs"] * dataset_size)
    missing_models = sorted(set(MODEL_ORDER) - set(df["model"]))
    if missing_models:
        raise ValueError(f"Missing model rows: {missing_models}")
    return df


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.45, alpha=0.8)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.35, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(axis="both", color="#94A3B8", width=0.45)


def main() -> None:
    df = load_metrics()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(PANELS),
        figsize=(7.4, 2.25),
        sharex=True,
        constrained_layout=False,
    )

    for ax, panel in zip(axes, PANELS):
        ax.axvspan(
            -0.03,
            0.28,
            color="#DFF3EA",
            alpha=0.85,
            zorder=0,
        )
        for model in MODEL_ORDER:
            model_df = df[df["model"] == model].sort_values("temperature")
            ax.plot(
                model_df["temperature"],
                model_df[str(panel["column"])],
                marker="o",
                markersize=3.4,
                linewidth=1.35,
                color=MODEL_COLORS[model],
                label=model,
                zorder=3,
            )

        ax.set_title(str(panel["title"]), fontweight="bold", pad=5)
        ax.set_ylabel(str(panel["ylabel"]))
        ax.set_xlabel("Temperature")
        ax.set_xlim(-0.05, 2.05)
        ax.set_xticks([0, 0.25, 0.75, 1.25, 1.75, 2.0])
        ax.set_xticklabels(["0", "0.25", "0.75", "1.25", "1.75", "2"])
        style_axes(ax)

        if not bool(panel["higher_is_better"]):
            ax.text(
                0.98,
                0.92,
                "lower is better",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=5.9,
                color="#475569",
            )
        else:
            ax.text(
                0.98,
                0.92,
                "higher is better",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=5.9,
                color="#475569",
            )

    axes[0].text(
        0.16,
        0.83,
        "Pareto-preferred\nlow-T region",
        transform=axes[0].transAxes,
        ha="center",
        va="center",
        fontsize=5.8,
        color="#0F766E",
    )
    axes[2].yaxis.set_major_formatter(lambda y, _pos: f"{100 * y:.2f}%")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.58, 0.015),
        fontsize=6.6,
        handletextpad=0.45,
        columnspacing=1.3,
    )
    fig.text(
        0.015,
        0.035,
        "Shaded region marks temperatures 0-0.25, where accuracy/Brier remain strong without elevated missingness.",
        ha="left",
        va="center",
        fontsize=6.2,
        color="#374151",
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.85, bottom=0.28, wspace=0.28)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
