"""Figures for the lead-30d structured-prompt comparison.

Produces:
  analysis/lead30/brier_oracle_vs_30d.png   grouped bars, oracle vs cutoff per variant
  analysis/lead30/degradation_by_model.png  Brier degradation per variant per model
  analysis/lead30/evidence_strata.png       accuracy by evidence availability
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "lead30"
MODELS = ["GPT-OSS-120B", "Qwen2.5-7b-instruct", "Qwen3-32B"]
VARIANTS = [f"V{i}" for i in range(9)]

matplotlib.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
})


def read(name: str) -> list[dict]:
    return list(csv.DictReader((OUT / name).open(encoding="utf-8")))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#e2e8f0")
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors="#475569")


def fig_oracle_vs_cutoff() -> None:
    metrics = read("variant_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        o = {r["variant"]: float(r["brier"]) for r in metrics
             if r["model"] == model and r["condition"] == "oracle"}
        c = {r["variant"]: float(r["brier"]) for r in metrics
             if r["model"] == model and r["condition"] == "lead30"}
        x = range(len(VARIANTS))
        ax.bar([i - 0.2 for i in x], [o[v] for v in VARIANTS], 0.4,
               label="oracle (full evidence)", color="#94a3b8")
        ax.bar([i + 0.2 for i in x], [c[v] for v in VARIANTS], 0.4,
               label="30d pre-event cutoff", color="#3b82f6")
        ax.set_xticks(list(x))
        ax.set_xticklabels(VARIANTS)
        ax.set_title(model, fontsize=11, color="#1e293b")
        style(ax)
    axes[0].set_ylabel("Brier score (lower is better)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Structured-prompt variants: retrospective vs ex-ante evidence",
                 fontsize=13, color="#0f172a")
    fig.tight_layout()
    fig.savefig(OUT / "brier_oracle_vs_30d.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("wrote brier_oracle_vs_30d.png")


def fig_degradation() -> None:
    deg = read("degradation.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = ["#3b82f6", "#f97316", "#10b981"]
    width = 0.26
    for k, model in enumerate(MODELS):
        vals = {r["variant"]: float(r["brier_delta"]) for r in deg if r["model"] == model}
        xs = [i + (k - 1) * width for i in range(len(VARIANTS))]
        ax.bar(xs, [vals[v] for v in VARIANTS], width, label=model, color=colors[k])
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("Brier degradation (30d - oracle)")
    ax.set_title("How much the evidence cutoff costs each prompt variant",
                 fontsize=12, color="#0f172a")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "degradation_by_model.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("wrote degradation_by_model.png")


def fig_strata() -> None:
    strata = read("evidence_strata.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, model in zip(axes, MODELS):
        z = {r["variant"]: float(r["accuracy"]) for r in strata
             if r["model"] == model and r["stratum"] == "zero_kept"}
        s = {r["variant"]: float(r["accuracy"]) for r in strata
             if r["model"] == model and r["stratum"] == "some_kept"}
        x = range(len(VARIANTS))
        ax.bar([i - 0.2 for i in x], [z[v] for v in VARIANTS], 0.4,
               label="0 articles kept", color="#ef4444")
        ax.bar([i + 0.2 for i in x], [s[v] for v in VARIANTS], 0.4,
               label="\u22651 article kept", color="#10b981")
        ax.set_xticks(list(x))
        ax.set_xticklabels(VARIANTS)
        ax.set_title(model, fontsize=11, color="#1e293b")
        style(ax)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Accuracy by evidence availability under the 30-day cutoff",
                 fontsize=13, color="#0f172a")
    fig.tight_layout()
    fig.savefig(OUT / "evidence_strata.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("wrote evidence_strata.png")


if __name__ == "__main__":
    fig_oracle_vs_cutoff()
    fig_degradation()
    fig_strata()
