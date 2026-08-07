#!/usr/bin/env python3
"""Ex-ante forecast-time cutoff sweep: accuracy/Brier/ECE vs forecast horizon.

Compares each model's forecasts at a sequence of forecast-time cutoffs (`none` =
oracle evidence up to resolution; then 7/30/90 days before the event end) for a
single variant, and emits a thesis-ready CSV + figure.

Metrics are computed on the records **common to all conditions** of a given model,
so each curve is a within-question comparison where only the evidence cutoff changes.

Reads the `run-batch --cutoff-reference event_end` layout:
    results/<model>/<temp>/results_<variant>.json            # none (oracle)
    results/<model>/<temp>/lead_<NN>d/results_<variant>.json  # cutoff at NN days

Usage:
    python scripts/plot_lead_time_cutoff.py            # default 3 thesis models
    python scripts/plot_lead_time_cutoff.py --models "GPT-OSS-120B:temperature_025"
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.metrics import (  # noqa: E402
    Example,
    accuracy,
    brier_score,
    ece,
    load_targets,
    normalize_answer,
    normalize_confidence,
)
import json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
LEADS = (7, 30, 90)
# (model_dir, temp_dir) at each model's best-Brier temperature; see project-thesis-scope.
DEFAULT_MODELS = [
    ("GPT-OSS-120B", "temperature_025"),
    ("Qwen2.5-7b-instruct", "temperature_000"),
    ("Qwen3-32B", "temperature_0"),
]


def load_examples(path: Path, targets: dict) -> tuple[dict[int, Example], dict]:
    """Return (examples, evidence_stats).

    evidence_stats keys (all empty-string when no cutoff data present):
      evidence_kept_frac   – mean(kept/total) over questions that had articles
      evidence_n_total     – mean total articles available per question
      evidence_n_kept      – mean articles kept after cutoff
      evidence_frac_any    – fraction of questions with ≥1 article kept
      evidence_n_kept_p50  – median articles kept (distribution centre)
      evidence_n_kept_p25  – 25th percentile (how sparse the low end is)
    """
    rows: dict[int, Example] = {}
    n_total_all: list[int] = []
    n_kept_all: list[int] = []
    fracs: list[float] = []
    for r in json.loads(path.read_text(encoding="utf-8")):
        rid = r.get("id")
        ans = normalize_answer(r.get("predicted_answer"))
        conf = normalize_confidence(r.get("confidence"))
        if rid in targets and ans and conf is not None:
            rows[rid] = Example(ans, conf, targets[rid])
        fc = r.get("forecast_cutoff")
        if isinstance(fc, dict) and fc.get("n_articles_total") is not None:
            nt = fc["n_articles_total"]
            nk = fc.get("n_articles_kept", 0)
            n_total_all.append(nt)
            n_kept_all.append(nk)
            if nt:
                fracs.append(nk / nt)
    if not n_kept_all:
        return rows, {}
    sorted_kept = sorted(n_kept_all)
    n = len(sorted_kept)
    p25 = sorted_kept[n // 4]
    p50 = statistics.median(sorted_kept)
    ev: dict = {
        "evidence_kept_frac":  round(sum(fracs) / len(fracs), 4) if fracs else "",
        "evidence_n_fetched":  round(statistics.mean(n_total_all), 2),
        "evidence_n_kept":     round(statistics.mean(n_kept_all), 2),
        "evidence_frac_any":   round(sum(1 for k in n_kept_all if k > 0) / n, 4),
        "evidence_n_kept_p50": p50,
        "evidence_n_kept_p25": p25,
    }
    return rows, ev


_EV_FIELDS = [
    "evidence_kept_frac", "evidence_n_fetched", "evidence_n_kept",
    "evidence_frac_any", "evidence_n_kept_p50", "evidence_n_kept_p25",
]


def model_curve(model: str, temp_dir: str, variant: str, targets: dict) -> list[dict]:
    base = ROOT / "results" / model / temp_dir
    conds = {"none": base / f"results_{variant}.json"}
    for lead in LEADS:
        conds[str(lead)] = base / f"lead_{lead}d" / f"results_{variant}.json"
    examples: dict[str, dict] = {}
    ev_stats: dict[str, dict] = {}
    for label, path in conds.items():
        if not path.exists():
            print(f"  WARN missing: {path}", file=sys.stderr)
            return []
        examples[label], ev_stats[label] = load_examples(path, targets)
    common = set.intersection(*(set(examples[c]) for c in conds))
    out = []
    for label in ["none", *map(str, LEADS)]:
        ex = [examples[label][i] for i in common]
        row: dict = {
            "model": model,
            "variant": variant,
            "lead_days": label,
            "lead_sort": -1 if label == "none" else int(label),
            "n": len(ex),
            "accuracy": round(accuracy(ex), 4),
            "brier": round(brier_score(ex), 4),
            "ece": round(ece(ex, 10), 4),
        }
        ev = ev_stats[label]
        for f in _EV_FIELDS:
            row[f] = ev.get(f, "")
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="variant0_neutral_baseline")
    ap.add_argument(
        "--models",
        default=None,
        help='Override as "Model:temp_dir,Model:temp_dir". Default: the 3 thesis models at best-Brier temps.',
    )
    ap.add_argument("--out-csv", type=Path, default=ROOT / "analysis" / "lead_time_cutoff.csv")
    ap.add_argument("--out-fig", type=Path, default=ROOT / "analysis" / "lead_time_cutoff.png")
    args = ap.parse_args()

    models = (
        [tuple(m.split(":", 1)) for m in args.models.split(",")] if args.models else DEFAULT_MODELS
    )
    targets = load_targets(DATASET)

    rows: list[dict] = []
    for model, temp_dir in models:
        rows.extend(model_curve(model, temp_dir, args.variant, targets))
    if not rows:
        raise SystemExit("No curves computed — check that the sweep results exist.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "variant", "lead_days", "n", "accuracy", "brier", "ece"] + _EV_FIELDS
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out_csv}")

    # Figure: accuracy vs forecast horizon — clean web aesthetics.
    import matplotlib
    import matplotlib.lines as mlines

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PALETTE = ["#3b82f6", "#f97316", "#10b981"]
    matplotlib.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#e2e8f0")
        ax.spines[spine].set_linewidth(0.8)

    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.9, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=0, colors="#64748b", labelsize=10.5)

    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    for i, (model, mrows) in enumerate(by_model.items()):
        color = PALETTE[i % len(PALETTE)]
        mrows.sort(key=lambda r: r["lead_sort"])
        full_ev = next(r for r in mrows if r["lead_days"] == "none")
        cuts = [r for r in mrows if r["lead_days"] != "none"]
        xs = [int(r["lead_days"]) for r in cuts]
        ys = [r["accuracy"] for r in cuts]
        ax.axhline(full_ev["accuracy"], color=color, ls=(0, (4, 4)), lw=0.9, alpha=0.45, zorder=1)
        ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=8,
                markerfacecolor="white", markeredgewidth=2.0, markeredgecolor=color,
                label=model, zorder=3, solid_capstyle="round", solid_joinstyle="round")

    ax.set_xlabel("Forecast cutoff  ·  days before event end", color="#64748b",
                  fontsize=10.5, labelpad=10)
    ax.set_ylabel("Accuracy", color="#64748b", fontsize=10.5)
    ax.set_title("Ex-ante forecasting accuracy vs forecast horizon",
                 color="#0f172a", fontsize=14, fontweight="600", pad=16, loc="left")
    ax.set_xticks(list(LEADS))
    ax.invert_xaxis()

    fe_handle = mlines.Line2D([], [], color="#94a3b8", ls=(0, (4, 4)), lw=0.9,
                              label="Full-evidence upper bound (no cutoff)")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles=handles + [fe_handle],
        labels=labels + ["Full-evidence upper bound (no cutoff)"],
        fontsize=9.5, frameon=True, loc="upper right",
        edgecolor="#e2e8f0", framealpha=1.0, borderpad=0.8,
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out_fig}")

    def _style_panel(panel: "plt.Axes") -> None:
        panel.set_facecolor("white")
        for sp in ("top", "right"):
            panel.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            panel.spines[sp].set_color("#e2e8f0")
            panel.spines[sp].set_linewidth(0.8)
        panel.yaxis.grid(True, color="#e2e8f0", linewidth=0.9, zorder=0)
        panel.xaxis.grid(False)
        panel.set_axisbelow(True)
        panel.tick_params(axis="both", which="both", length=0, colors="#64748b", labelsize=10.5)

    _marker_kw = dict(linewidth=2.0, marker="o", markersize=8, markerfacecolor="white",
                      markeredgewidth=2.0, zorder=3,
                      solid_capstyle="round", solid_joinstyle="round")

    # ── Chart 2: Brier score vs forecast horizon ─────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(9, 5.5))
    fig_b.patch.set_facecolor("white")
    _style_panel(ax_b)

    for i, (model, mrows) in enumerate(by_model.items()):
        color = PALETTE[i % len(PALETTE)]
        mrows.sort(key=lambda r: r["lead_sort"])
        full_ev = next(r for r in mrows if r["lead_days"] == "none")
        cuts = [r for r in mrows if r["lead_days"] != "none"]
        xs = [int(r["lead_days"]) for r in cuts]
        ax_b.axhline(full_ev["brier"], color=color, ls=(0, (4, 4)), lw=0.9, alpha=0.45, zorder=1)
        ax_b.plot(xs, [r["brier"] for r in cuts], color=color, markeredgecolor=color,
                  label=model, **_marker_kw)

    ax_b.set_xlabel("Forecast cutoff  ·  days before event end", color="#64748b",
                    fontsize=10.5, labelpad=10)
    ax_b.set_ylabel("Brier score  ↓  lower is better", color="#64748b", fontsize=10.5)
    ax_b.set_title("Brier score vs forecast horizon",
                   color="#0f172a", fontsize=14, fontweight="600", pad=16, loc="left")
    ax_b.set_xticks(list(LEADS))
    ax_b.invert_xaxis()

    fe_b = mlines.Line2D([], [], color="#94a3b8", ls=(0, (4, 4)), lw=0.9,
                         label="Full-evidence lower bound (no cutoff)")
    h_b, l_b = ax_b.get_legend_handles_labels()
    leg_b = ax_b.legend(handles=h_b + [fe_b], labels=l_b + ["Full-evidence lower bound (no cutoff)"],
                        fontsize=9.5, frameon=True, loc="upper right",
                        edgecolor="#e2e8f0", framealpha=1.0, borderpad=0.8)
    leg_b.get_frame().set_linewidth(0.8)

    fig_b.tight_layout()
    out_brier = args.out_fig.with_name(
        args.out_fig.stem.replace("cutoff", "brier") + args.out_fig.suffix
    )
    fig_b.savefig(out_brier, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_brier}")

    # ── Chart 3: Accuracy (x) vs Brier score (y) — colour = model, shape = lead time ──
    # Y-axis is inverted so both axes point "better" toward the upper-right corner.
    LEAD_ORDER  = ["90", "30", "7", "none"]
    LEAD_MARKER = {"90": "s", "30": "^", "7": "o", "none": "D"}
    LEAD_SIZE   = {"90": 95,  "30": 95,  "7": 95,  "none": 120}
    LEAD_LABEL  = {"90": "90d  (long lead)",  "30": "30d",
                   "7":  "7d   (short lead)", "none": "Full evidence (no cutoff)"}
    # nudge (dx, dy) for each model's annotation near the full-evidence diamond
    ANNOT_NUDGE = {
        "GPT-OSS-120B":        (-10,  6, "right"),
        "Qwen2.5-7b-instruct": ( 10,  4, "left"),
        "Qwen3-32B":           ( 10, -6, "left"),
    }

    fig_c, ax_c = plt.subplots(figsize=(9, 6.5))
    fig_c.patch.set_facecolor("white")
    _style_panel(ax_c)
    ax_c.yaxis.grid(True,  color="#e2e8f0", linewidth=0.9, zorder=0)
    ax_c.xaxis.grid(True,  color="#e2e8f0", linewidth=0.9, zorder=0)

    # Trajectory lines connecting each model's points from worst → best evidence
    for i, (model, mrows) in enumerate(by_model.items()):
        color = PALETTE[i % len(PALETTE)]
        ordered = sorted(mrows, key=lambda r: LEAD_ORDER.index(r["lead_days"]))
        ax_c.plot([r["accuracy"] for r in ordered],
                  [r["brier"]    for r in ordered],
                  color=color, linewidth=1.2, alpha=0.28, zorder=1,
                  solid_capstyle="round", solid_joinstyle="round")

    # Scatter points: iterate lead times outermost so same-lead markers are drawn together
    for lead in LEAD_ORDER:
        for i, (model, mrows) in enumerate(by_model.items()):
            color = PALETTE[i % len(PALETTE)]
            row = next(r for r in mrows if r["lead_days"] == lead)
            ax_c.scatter(
                row["accuracy"], row["brier"],
                marker=LEAD_MARKER[lead], s=LEAD_SIZE[lead],
                color=color, edgecolors="white", linewidths=1.6,
                zorder=4, clip_on=False,
            )

    # Model name annotations anchored to each full-evidence diamond
    for i, (model, mrows) in enumerate(by_model.items()):
        color = PALETTE[i % len(PALETTE)]
        fe = next(r for r in mrows if r["lead_days"] == "none")
        dx, dy, ha = ANNOT_NUDGE.get(model, (10, 0, "left"))
        ax_c.annotate(
            model,
            xy=(fe["accuracy"], fe["brier"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, color=color, fontweight="600",
            va="center", ha=ha,
        )

    ax_c.invert_yaxis()   # lower Brier → top of chart; upper-right = best on both metrics

    ax_c.set_xlabel("Accuracy  →  higher is better", color="#64748b", fontsize=10.5, labelpad=10)
    ax_c.set_ylabel("Brier score  ↑  lower is better", color="#64748b", fontsize=10.5)
    ax_c.set_title("Forecast quality  ·  accuracy vs Brier score",
                   color="#0f172a", fontsize=14, fontweight="600", pad=16, loc="left")

    # Subtle "ideal corner" hint
    ax_c.text(0.985, 0.015, "← ideal", transform=ax_c.transAxes,
              fontsize=8, color="#cbd5e1", ha="right", va="bottom")

    # Two-section legend: model colours (top) + lead-time shapes (bottom)
    model_handles = [
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=PALETTE[i], markersize=9, label=model)
        for i, model in enumerate(by_model.keys())
    ]
    lead_handles = [
        mlines.Line2D([0], [0], marker=LEAD_MARKER[ld], color="w",
                      markerfacecolor="#64748b", markersize=8, label=LEAD_LABEL[ld])
        for ld in LEAD_ORDER
    ]

    leg_models = ax_c.legend(
        handles=model_handles, title="Model", title_fontsize=8.5,
        fontsize=9, frameon=True, loc="lower left",
        edgecolor="#e2e8f0", framealpha=1.0, borderpad=0.8,
    )
    leg_models.get_frame().set_linewidth(0.8)
    ax_c.add_artist(leg_models)

    leg_leads = ax_c.legend(
        handles=lead_handles, title="Lead time", title_fontsize=8.5,
        fontsize=9, frameon=True, loc="lower right",
        edgecolor="#e2e8f0", framealpha=1.0, borderpad=0.8,
    )
    leg_leads.get_frame().set_linewidth(0.8)

    fig_c.tight_layout()
    out_combined = args.out_fig.with_name(
        args.out_fig.stem.replace("cutoff", "combined") + args.out_fig.suffix
    )
    fig_c.savefig(out_combined, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_combined}")

    # Replace the legacy combined chart with compact model small multiples.
    # Model identity is carried by panel titles; the shared legend only explains lead time.
    LEAD_ORDER_COMPACT = ["90", "30", "7", "none"]
    LEAD_MARKER_COMPACT = {"90": "s", "30": "^", "7": "o", "none": "D"}
    LEAD_COLOR_COMPACT = {
        "90": "#2563eb",
        "30": "#0f766e",
        "7": "#d97706",
        "none": "#334155",
    }
    LEAD_LABEL_COMPACT = {"90": "90d", "30": "30d", "7": "7d", "none": "Full evidence"}

    model_order = list(by_model.keys())
    fig_compact, axes = plt.subplots(
        1, len(model_order), figsize=(9.2, 3.7), sharex=True, sharey=True
    )
    fig_compact.patch.set_facecolor("white")
    if len(model_order) == 1:
        axes = [axes]

    for ax_compact, model in zip(axes, model_order):
        mrows = by_model[model]
        ordered = sorted(mrows, key=lambda r: LEAD_ORDER_COMPACT.index(r["lead_days"]))
        _style_panel(ax_compact)
        ax_compact.xaxis.grid(True, color="#e2e8f0", linewidth=0.8, zorder=0)
        ax_compact.set_title(model, color="#0f172a", fontsize=10.5, fontweight="600", pad=9)
        ax_compact.set_xlim(0.56, 0.84)
        ax_compact.set_ylim(0.355, 0.145)
        ax_compact.set_xticks([0.60, 0.70, 0.80])
        ax_compact.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35])
        ax_compact.tick_params(axis="both", which="both", length=0, colors="#64748b", labelsize=8.5)
        ax_compact.plot(
            [r["accuracy"] for r in ordered],
            [r["brier"] for r in ordered],
            color="#94a3b8", linewidth=1.2, alpha=0.75, zorder=1,
            solid_capstyle="round", solid_joinstyle="round",
        )
        for row in ordered:
            lead = row["lead_days"]
            ax_compact.scatter(
                row["accuracy"], row["brier"],
                marker=LEAD_MARKER_COMPACT[lead],
                s=58 if lead != "none" else 72,
                color=LEAD_COLOR_COMPACT[lead], edgecolors="white", linewidths=1.2,
                zorder=3, clip_on=False,
            )
    axes[0].set_ylabel("Brier score (lower is better)", color="#64748b", fontsize=9.5, labelpad=8)
    fig_compact.supxlabel("Accuracy (higher is better)", color="#64748b", fontsize=9.5, y=0.035)
    fig_compact.suptitle(
        "Forecast quality across evidence cutoffs",
        color="#0f172a", fontsize=13.5, fontweight="600", x=0.08, ha="left", y=0.99,
    )
    lead_handles_compact = [
        mlines.Line2D(
            [0], [0], marker=LEAD_MARKER_COMPACT[lead], color="w",
            markerfacecolor=LEAD_COLOR_COMPACT[lead], markeredgecolor="white",
            markeredgewidth=0.8, markersize=7, label=LEAD_LABEL_COMPACT[lead],
        )
        for lead in LEAD_ORDER_COMPACT
    ]
    fig_compact.legend(
        handles=lead_handles_compact, title="Evidence available at forecast time",
        title_fontsize=8.5, fontsize=8.5, frameon=False, ncol=4,
        loc="upper right", bbox_to_anchor=(0.98, 0.995), columnspacing=1.0,
        handletextpad=0.3,
    )
    fig_compact.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.19, wspace=0.08)
    fig_compact.savefig(out_combined, dpi=150, bbox_inches="tight")
    fig_compact.savefig(out_combined.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote polished {out_combined}")

    # Final combined view: one shared accuracy/Brier coordinate system for direct comparison.
    MODEL_COLORS = {
        "GPT-OSS-120B": "#2563eb",
        "Qwen2.5-7b-instruct": "#ea580c",
        "Qwen3-32B": "#059669",
    }
    fig_single, ax_single = plt.subplots(figsize=(8.8, 5.2))
    fig_single.patch.set_facecolor("white")
    _style_panel(ax_single)
    ax_single.xaxis.grid(True, color="#e2e8f0", linewidth=0.85, zorder=0)
    ax_single.yaxis.grid(True, color="#e2e8f0", linewidth=0.85, zorder=0)
    ax_single.set_xlim(0.56, 0.84)
    ax_single.set_ylim(0.355, 0.145)
    ax_single.set_xticks([0.60, 0.70, 0.80])
    ax_single.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35])
    ax_single.tick_params(axis="both", which="both", length=0, colors="#64748b", labelsize=9.5)

    for model in model_order:
        mrows = by_model[model]
        ordered = sorted(mrows, key=lambda r: LEAD_ORDER_COMPACT.index(r["lead_days"]))
        color = MODEL_COLORS.get(model, "#475569")
        ax_single.plot(
            [r["accuracy"] for r in ordered],
            [r["brier"] for r in ordered],
            color=color, linewidth=1.4, alpha=0.34, zorder=1,
            solid_capstyle="round", solid_joinstyle="round",
        )
        for row in ordered:
            lead = row["lead_days"]
            ax_single.scatter(
                row["accuracy"], row["brier"],
                marker=LEAD_MARKER_COMPACT[lead],
                s=62 if lead != "none" else 82,
                color=color, edgecolors="white", linewidths=1.4,
                zorder=3, clip_on=False,
            )

    ax_single.set_xlabel("Accuracy (higher is better)", color="#64748b", fontsize=10, labelpad=10)
    ax_single.set_ylabel("Brier score (lower is better)", color="#64748b", fontsize=10, labelpad=10)
    ax_single.set_title(
        "Forecast quality across evidence cutoffs",
        color="#0f172a", fontsize=14, fontweight="600", pad=15, loc="left",
    )

    model_handles_single = [
        mlines.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS.get(model, "#475569"),
            markeredgecolor="white", markeredgewidth=0.8, markersize=7, label=model,
        )
        for model in model_order
    ]
    lead_handles_single = [
        mlines.Line2D(
            [0], [0], marker=LEAD_MARKER_COMPACT[lead], color="w",
            markerfacecolor="#64748b", markeredgecolor="white", markeredgewidth=0.8,
            markersize=7, label=LEAD_LABEL_COMPACT[lead],
        )
        for lead in LEAD_ORDER_COMPACT
    ]
    legend_models = fig_single.legend(
        handles=model_handles_single, title="Model", title_fontsize=8.5, fontsize=8.5,
        frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.08, 0.995),
        columnspacing=1.0, handletextpad=0.3,
    )
    fig_single.legend(
        handles=lead_handles_single, title="Evidence available at forecast time",
        title_fontsize=8.5, fontsize=8.5, frameon=False, ncol=4,
        loc="upper right", bbox_to_anchor=(0.98, 0.995), columnspacing=0.9,
        handletextpad=0.3,
    )
    fig_single.subplots_adjust(left=0.11, right=0.98, top=0.76, bottom=0.16)
    fig_single.savefig(out_combined, dpi=150, bbox_inches="tight")
    fig_single.savefig(out_combined.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote final shared-scale {out_combined}")


if __name__ == "__main__":
    main()
