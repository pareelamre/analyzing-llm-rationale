from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "grouped_controlled_top_shap.csv"
OUTPUT = BASE_DIR / "grouped_controlled_shap_top_features.pdf"

MODEL_SOURCES = {
    "GPT-OSS-120B": BASE_DIR / "gpt_oss_120b" / "feature_importance.csv",
    "Qwen2.5-7B-Instruct": BASE_DIR / "qwen25_7b" / "feature_importance.csv",
    "Qwen3-32B": BASE_DIR / "qwen3_32b" / "feature_importance.csv",
}
MODEL_ORDER = ["GPT-OSS-120B", "Qwen2.5-7B-Instruct", "Qwen3-32B"]
RATIONALE_FEATURES = {
    "plausibility",
    "completeness",
    "source_consistency",
    "non_hallucination",
    "informativeness",
    "conciseness",
}
FEATURE_LABELS = {
    "completeness": "Completeness",
    "plausibility": "Plausibility",
    "informativeness": "Informativeness",
    "evidence_words": "Evidence words",
    "rationale_words": "Rationale words",
    "conciseness": "Conciseness",
    "source_consistency": "Source consistency",
    "non_hallucination": "Non-hallucination",
}
COLORS = {
    "rationale_quality": "#4C78A8",
    "control": "#F58518",
}
TOP_N = 6


def build_source_csv() -> pd.DataFrame:
    rows = []
    for model, path in MODEL_SOURCES.items():
        data = pd.read_csv(path)
        subset = (
            data[data["dataset"] == "combined_mean__full"]
            .sort_values("mean_abs_shap", ascending=False)
            .head(TOP_N)
            .reset_index(drop=True)
        )
        for rank, row in enumerate(subset.itertuples(index=False), start=1):
            feature = row.feature
            rows.append(
                {
                    "forecast_model": model,
                    "rank": rank,
                    "feature": feature,
                    "feature_type": (
                        "rationale_quality"
                        if feature in RATIONALE_FEATURES
                        else "control"
                    ),
                    "mean_abs_shap": row.mean_abs_shap,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE, index=False)
    return out


def main() -> None:
    data = build_source_csv()
    max_value = float(data["mean_abs_shap"].max())

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(7.2, 5.2), sharex=True)

    for axis, model in zip(axes, MODEL_ORDER):
        subset = data[data["forecast_model"] == model].sort_values("rank")
        positions = list(range(len(subset)))
        labels = [
            FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
            for feature in subset["feature"]
        ]
        bar_colors = [
            COLORS.get(feature_type, "#777777")
            for feature_type in subset["feature_type"]
        ]

        axis.barh(
            positions,
            subset["mean_abs_shap"],
            color=bar_colors,
            edgecolor="white",
            linewidth=0.7,
        )
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_title(model, loc="left", fontweight="bold", pad=3)
        axis.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_xlim(0, max_value * 1.18)

        for position, value in zip(positions, subset["mean_abs_shap"]):
            axis.text(
                float(value) + max_value * 0.018,
                position,
                f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=8,
            )

    axes[-1].set_xlabel("Mean absolute SHAP value")
    figure.suptitle(
        "Top grouped controlled SHAP features by forecast model",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles=[
            Patch(facecolor=COLORS["rationale_quality"], label="Rationale quality"),
            Patch(facecolor=COLORS["control"], label="Control"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.955), h_pad=1.0)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
