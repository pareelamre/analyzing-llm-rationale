#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "analysis" / "metrics_by_model_variant.csv"
OUT_DIR = ROOT / "paper"

MODEL_ORDER = [
    "Qwen2.5-7b-instruct",
    "Qwen3-32B",
    "GPT-OSS-120B",
]

VARIANT_ORDER = [
    "variant0_neutral_baseline",
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
    "variant0_neutral_baseline": "V0",
    "variant1_predicted_event": "V1",
    "variant2_key_attribute": "V2",
    "variant3_reasoning_type": "V3",
    "variant4_credibility": "V4",
    "variant5_key_conditions": "V5",
    "variant6_step_by_step_reasoning": "V6",
    "variant7_uncertainty_language": "V7",
    "variant8_temporal_anchors": "V8",
}

VARIANT_FOOTNOTE = (
    "V0 base; V1 event; V2 attribute; V3 reasoning; V4 credibility; "
    "V5 conditions; V6 steps; V7 uncertainty; V8 temporal anchors"
)

PANELS = [
    {
        "title": "Accuracy",
        "column": "mean_accuracy",
        "higher_is_better": True,
        "fmt": "{:.3f}",
        "slug": "accuracy",
    },
    {
        "title": "Brier score",
        "column": "mean_brier_score",
        "higher_is_better": False,
        "fmt": "{:.3f}",
        "slug": "brier_score",
    },
    {
        "title": "ECE",
        "column": "mean_ece",
        "higher_is_better": False,
        "fmt": "{:.3f}",
        "slug": "ece",
    },
]


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_PATH)
    missing_models = sorted(set(MODEL_ORDER) - set(df["model"]))
    missing_variants = sorted(set(VARIANT_ORDER) - set(df["variant"]))
    if missing_models or missing_variants:
        raise ValueError(
            f"Missing metrics for models={missing_models} variants={missing_variants}"
        )
    return df


def matrix_for(df: pd.DataFrame, metric: str) -> np.ndarray:
    pivot = df.pivot(index="model", columns="variant", values=metric)
    return pivot.loc[MODEL_ORDER, VARIANT_ORDER].to_numpy(dtype=float)


def utility_matrix(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo = float(finite.min())
    hi = float(finite.max())
    if np.isclose(lo, hi):
        return np.ones_like(values) * 0.5
    utility = (values - lo) / (hi - lo)
    if not higher_is_better:
        utility = 1.0 - utility
    return utility


def best_col_for_row(values: np.ndarray, row: int, higher_is_better: bool) -> int:
    row_values = values[row]
    if higher_is_better:
        return int(np.nanargmax(row_values))
    return int(np.nanargmin(row_values))


def text_color(score: float) -> str:
    return "#FFFFFF" if score >= 0.58 else "#111827"


def render_panel(
    df: pd.DataFrame,
    panel: dict[str, object],
    cmap: LinearSegmentedColormap,
) -> tuple[Path, Path]:
    values = matrix_for(df, str(panel["column"]))
    utility = utility_matrix(values, bool(panel["higher_is_better"]))

    fig, ax = plt.subplots(figsize=(6.6, 2.55), constrained_layout=False)
    ax.imshow(utility, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_title(str(panel["title"]), fontweight="bold", pad=6)
    ax.set_xticks(np.arange(len(VARIANT_ORDER)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER])
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_xticks(np.arange(-0.5, len(VARIANT_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index, _model in enumerate(MODEL_ORDER):
        best_col = best_col_for_row(
            values,
            row_index,
            bool(panel["higher_is_better"]),
        )
        for col_index, _variant in enumerate(VARIANT_ORDER):
            score = float(utility[row_index, col_index])
            value = float(values[row_index, col_index])
            is_best = col_index == best_col
            ax.text(
                col_index,
                row_index,
                str(panel["fmt"]).format(value),
                ha="center",
                va="center",
                color=text_color(score),
                fontsize=7.4,
                fontweight="bold" if is_best else "normal",
            )
        ax.add_patch(
            plt.Rectangle(
                (best_col - 0.5, row_index - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#111827",
                linewidth=1.25,
                zorder=5,
            )
        )
        ax.text(
            best_col + 0.34,
            row_index - 0.31,
            "*",
            ha="center",
            va="center",
            color="#111827" if utility[row_index, best_col] < 0.58 else "#FFFFFF",
            fontsize=8.0,
            fontweight="bold",
            zorder=6,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.02,
        0.1,
        VARIANT_FOOTNOTE,
        ha="left",
        va="center",
        fontsize=6.2,
        color="#374151",
    )
    fig.text(
        0.02,
        0.04,
        "Darker cells are better within this metric. * best variant within model.",
        ha="left",
        va="center",
        fontsize=6.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.2, right=0.995, bottom=0.34, top=0.82)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"model_variant_{panel['slug']}_heatmap.png"
    pdf_path = OUT_DIR / f"model_variant_{panel['slug']}_heatmap.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    df = load_metrics()

    cmap = LinearSegmentedColormap.from_list(
        "forecast_metric_utility",
        ["#F8FAFC", "#C7EAE5", "#5AAE9D", "#075B5A"],
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    for panel in PANELS:
        png_path, pdf_path = render_panel(df, panel, cmap)
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
