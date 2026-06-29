#!/usr/bin/env python3
"""Plot cumulative paper PnL equity curves for all tracked models, including crowd-follow.

crowd-follow has zero edge by definition (model_prob = market_prob), so the standard
paper_pnl code (which requires edge >= 0.05) filters all its bets out. Here we use
min_edge=0 for all models so crowd-follow appears as the zero-expectancy baseline.
"""
# ruff: noqa: E402,I001
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from analyzing_llm_rationale.trackrec_store import DuckDBStore

STORE_PATH = ROOT / "data" / "track_record_store.duckdb"
OUT_PATH = ROOT / "analysis" / "equity_curves.png"

# Tailwind-inspired palette
COLORS = {
    "council":             "#111827",   # gray-900
    "gpt-oss-120b":        "#3b82f6",   # blue-500
    "kimi-k2.6":           "#f97316",   # orange-500
    "gemma-4-31b-it":      "#10b981",   # emerald-500
    "crowd-follow":        "#94a3b8",   # slate-400  (baseline)
}
LABELS = {
    "council":        "Council",
    "gpt-oss-120b":   "GPT-OSS-120B",
    "kimi-k2.6":      "Kimi K2.6",
    "gemma-4-31b-it": "Gemma 4-31B",
    "crowd-follow":   "Crowd (baseline)",
}
# Linestyle per model — Kimi/Gemma may overlap, so use distinct dashes
LINESTYLES = {
    "council":        "-",
    "gpt-oss-120b":   "-",
    "kimi-k2.6":      (0, (4, 2)),   # long dash
    "gemma-4-31b-it": "-",
    "crowd-follow":   (0, (2, 2)),   # dotted
}
MODEL_ORDER = ["council", "gpt-oss-120b", "gemma-4-31b-it", "kimi-k2.6", "crowd-follow"]


def _edge(model_p: float, market_p: float) -> float:
    return abs(model_p - market_p)


def _bet_fee(platform: str | None, stake: float, p_side: float) -> float:
    """Kalshi price-dependent taker fee; Polymarket fee-free."""
    if (platform or "").lower() == "kalshi":
        fee_rate = 0.07 if p_side <= 0.5 else (0.07 * (1 - p_side) / p_side)
        return stake * min(fee_rate, 0.07)
    return 0.0


def equity_curve(rows: list[dict], *, min_edge: float = 0.0) -> tuple[list[str], list[float]]:
    """Cumulative flat-stake PnL over resolved snapshots, ordered by resolution."""
    bets: list[tuple[str, float]] = []
    for r in sorted(rows, key=lambda x: x.get("resolved_ts") or x.get("snapshot_ts") or ""):
        model_p = float(r["model_probability"])
        market_p = float(r["market_probability"])
        edge = _edge(model_p, market_p)
        if edge < min_edge:
            continue
        if model_p == 0.5:
            continue
        side_yes = model_p > 0.5
        p_side = market_p if side_yes else (1.0 - market_p)
        if not (0.0 < p_side < 1.0):
            continue
        outcome = int(r["outcome"])
        win = (outcome == 1) == side_yes
        payout = (1.0 - p_side) / p_side
        fee = _bet_fee(r.get("platform"), 1.0, p_side)
        profit = 1.0 * (payout if win else -1.0) - fee
        ts = str(r.get("resolved_ts") or r.get("snapshot_ts") or "")[:10]
        bets.append((ts, profit))
    dates = [b[0] for b in bets]
    curve = list(np.cumsum([b[1] for b in bets]))
    return dates, curve


def main() -> None:
    store = DuckDBStore(STORE_PATH)
    rows_raw = store._con.execute("""
        SELECT model, ident, snapshot_ts, model_probability, market_probability,
               outcome, platform, resolved_ts
        FROM forecast_snapshot
        WHERE resolved = true
          AND outcome IS NOT NULL
          AND key NOT LIKE '%_backfill'
        ORDER BY resolved_ts
    """).fetchall()

    by_model: dict[str, list[dict]] = defaultdict(list)
    for model, ident, snapshot_ts, model_p, market_p, outcome, platform, resolved_ts in rows_raw:
        rec = {
            "model": model,
            "ident": ident,
            "snapshot_ts": str(snapshot_ts),
            "model_probability": model_p,
            "market_probability": market_p,
            "outcome": outcome,
            "platform": platform,
            "resolved_ts": str(resolved_ts),
        }
        by_model[model or "council"].append(rec)

    llm_groups = {m: rows for m, rows in by_model.items() if m != "crowd-follow"}
    if llm_groups:
        _label, reference_rows = max(llm_groups.items(), key=lambda item: (len(item[1]), item[0]))
        by_model["crowd-follow"] = [
            {**r, "model": "crowd-follow", "model_probability": r["market_probability"]}
            for r in reference_rows
        ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=10, colors="#64748b")
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)

    # Draw zero line
    ax.axhline(0, color="#94a3b8", linewidth=1.0, linestyle=":")

    for model_key in MODEL_ORDER:
        rows = by_model.get(model_key, [])
        if not rows:
            continue
        dates, curve = equity_curve(rows, min_edge=0.0)
        if not curve:
            continue
        curve_disp = curve
        color = COLORS[model_key]
        lw = 1.5 if model_key == "crowd-follow" else 2.2
        ls = LINESTYLES[model_key]
        alpha = 0.65 if model_key == "crowd-follow" else 1.0
        xs = list(range(len(curve_disp)))
        ax.plot(xs, curve_disp, color=color, linewidth=lw, linestyle=ls, alpha=alpha, zorder=3)
        # End label (use unshifted final for the number)
        final_true = curve[-1]
        final_disp = curve_disp[-1]
        sign = "+" if final_true >= 0 else ""
        ax.annotate(
            f"{LABELS[model_key]}\n{sign}{final_true:.1f}u",
            xy=(xs[-1], final_disp),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            va="center",
            fontweight="bold" if model_key != "crowd-follow" else "normal",
        )
        # Dots on each resolved bet
        ax.scatter(xs, curve_disp, color=color, s=22, zorder=4, alpha=0.6)

    n_bets = max(len(equity_curve(by_model.get(m, []), min_edge=0.0)[0]) for m in MODEL_ORDER if by_model.get(m))
    ax.set_xlim(-1, n_bets + 12)
    ax.set_xlabel("Resolved bets (chronological)", fontsize=10, color="#64748b", labelpad=8)
    ax.set_ylabel("Cumulative PnL (flat $1 stake)", fontsize=10, color="#64748b", labelpad=8)
    ax.set_title(
        "Paper equity curves — all models vs crowd-follow baseline\n"
        "Flat $1 stake, min edge = 0  ·  crowd-follow always bets the market's own direction",
        fontsize=11, fontweight="bold", color="#1e293b", pad=10,
        linespacing=1.5,
    )

    legend_handles = [
        mpatches.Patch(color=COLORS[m], label=LABELS[m])
        for m in MODEL_ORDER if by_model.get(m)
    ]
    ax.legend(handles=legend_handles, fontsize=9, frameon=False,
              loc="upper left", labelcolor="#1e293b")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
