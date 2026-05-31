#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "analysis" / "metrics_by_model_variant.csv"
OUT_PNG = ROOT / "paper" / "variant_delta_from_v0.png"
OUT_PDF = ROOT / "paper" / "variant_delta_from_v0.pdf"

MODEL_ORDER = [
    "Qwen2.5-7b-instruct",
    "Qwen3-32B",
    "GPT-OSS-120B",
]

VARIANT_ORDER = [
    "variant1_predicted_event",
    "variant2_key_attribute",
    "variant3_reasoning_type",
    "variant4_credibility",
    "variant5_key_conditions",
    "variant6_step_by_step_reasoning",
    "variant7_uncertainty_language",
    "variant8_temporal_anchors",
]

VARIANT_LABELS = {
    "variant1_predicted_event": "V1 Event",
    "variant2_key_attribute": "V2 Attribute",
    "variant3_reasoning_type": "V3 Reasoning",
    "variant4_credibility": "V4 Credibility",
    "variant5_key_conditions": "V5 Conditions",
    "variant6_step_by_step_reasoning": "V6 Steps",
    "variant7_uncertainty_language": "V7 Uncertainty",
    "variant8_temporal_anchors": "V8 Temporal",
}

MODEL_COLORS = {
    "Qwen2.5-7b-instruct": "#4C78A8",
    "Qwen3-32B": "#F58518",
    "GPT-OSS-120B": "#54A24B",
}

PANELS = [
    {
        "title": "Accuracy",
        "metric": "mean_accuracy",
        "baseline_transform": 1.0,
        "xlabel": "Delta accuracy vs. V0",
    },
    {
        "title": "Brier score",
        "metric": "mean_brier_score",
        "baseline_transform": -1.0,
        "xlabel": "Improvement vs. V0 (-Delta Brier)",
    },
    {
        "title": "ECE",
        "metric": "mean_ece",
        "baseline_transform": -1.0,
        "xlabel": "Improvement vs. V0 (-Delta ECE)",
    },
]


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_PATH)
    needed_variants = set(VARIANT_ORDER) | {"variant0_neutral_baseline"}
    missing_models = sorted(set(MODEL_ORDER) - set(df["model"]))
    missing_variants = sorted(needed_variants - set(df["variant"]))
    if missing_models or missing_variants:
        raise ValueError(
            f"Missing metrics for models={missing_models} variants={missing_variants}"
        )
    return df


def deltas_for_panel(df: pd.DataFrame, panel: dict[str, object]) -> pd.DataFrame:
    metric = str(panel["metric"])
    sign = float(panel["baseline_transform"])
    rows = []
    for model in MODEL_ORDER:
        model_df = df[df["model"] == model].set_index("variant")
        baseline = float(model_df.loc["variant0_neutral_baseline", metric])
        for variant in VARIANT_ORDER:
            value = float(model_df.loc[variant, metric])
            rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "delta": sign * (value - baseline),
                }
            )
    return pd.DataFrame(rows)


def symmetric_limit(panel_frames: list[pd.DataFrame]) -> float:
    max_abs = max(float(frame["delta"].abs().max()) for frame in panel_frames)
    return float(np.ceil((max_abs * 1.1) / 0.01) * 0.01)


def main() -> None:
    df = load_metrics()
    panel_frames = [deltas_for_panel(df, panel) for panel in PANELS]
    xlim = symmetric_limit(panel_frames)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.2,
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
        figsize=(7.45, 3.15),
        sharey=True,
        constrained_layout=False,
    )
    y_positions = np.arange(len(VARIANT_ORDER))
    offsets = {
        "Qwen2.5-7b-instruct": -0.18,
        "Qwen3-32B": 0.0,
        "GPT-OSS-120B": 0.18,
    }

    for ax, panel, frame in zip(axes, PANELS, panel_frames):
        ax.axvline(0, color="#111827", linewidth=0.75, zorder=1)
        ax.axvspan(0, xlim, color="#DFF3EA", alpha=0.55, zorder=0)
        ax.axvspan(-xlim, 0, color="#F7E4E4", alpha=0.55, zorder=0)

        for model in MODEL_ORDER:
            model_frame = frame[frame["model"] == model].set_index("variant")
            x = [float(model_frame.loc[variant, "delta"]) for variant in VARIANT_ORDER]
            y = y_positions + offsets[model]
            ax.hlines(
                y,
                xmin=np.minimum(0, x),
                xmax=np.maximum(0, x),
                color=MODEL_COLORS[model],
                linewidth=1.2,
                alpha=0.75,
                zorder=2,
            )
            ax.scatter(
                x,
                y,
                s=17,
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.45,
                label=model,
                zorder=3,
            )

        ax.set_title(str(panel["title"]), fontweight="bold", pad=5)
        ax.set_xlabel(str(panel["xlabel"]), labelpad=4)
        ax.set_xlim(-xlim, xlim)
        ax.set_xticks(np.linspace(-xlim, xlim, 5))
        ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:+.2f}")
        ax.grid(axis="x", color="#CBD5E1", linewidth=0.4, alpha=0.8)
        ax.tick_params(axis="y", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER])
    axes[0].invert_yaxis()

    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.015),
        fontsize=6.6,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.text(
        0.015,
        0.025,
        "Positive values are beneficial relative to V0.",
        ha="left",
        va="center",
        fontsize=6.4,
        color="#374151",
    )
    fig.subplots_adjust(left=0.2, right=0.995, top=0.91, bottom=0.2, wspace=0.17)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
