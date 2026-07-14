#!/usr/bin/env python3
"""
Score a completed human-annotation file against the LLM judges.

Reads the human labels produced via ``scripts/build_human_annotation_sheet.py``
(``analysis/manual_validation/human_annotations.csv``) and the LLM-judge scores
already attached to ``analysis/manual_validation/sample.jsonl`` (gemma, kimi,
and the GPT/Gemma "third judge"), then reports:

  * human vs each LLM judge, per criterion (Spearman ρ, Cohen κ, % agreement);
  * judge vs judge (Gemma vs Kimi);
  * the human pass-rate per criterion.

These are the numbers a reviewer asks for when checking whether LLM judges can
stand in for a human. This script only consumes human labels — it never
generates them. Rows left blank in the CSV are skipped, so a partially completed
annotation export can still be scored.

Outputs:
  analysis/manual_validation/agreement_human_vs_judges.json
  analysis/manual_validation/agreement_human_vs_judges.md
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MV = ROOT / "analysis" / "manual_validation"
SAMPLE_PATH = MV / "sample.jsonl"
HUMAN_PATH = MV / "human_annotations.csv"

HUMAN_CRITERIA = [
    "criterion_alignment", "evidence_consistency", "hallucination",
    "usefulness", "probability_support",
]

# The human criteria mirror the third judge exactly. Against the original two
# judges (6 different criteria) only three concepts line up cleanly; the other
# two have no honest analogue, so we do not force a comparison.
HUMAN_TO_EXISTING = {
    "hallucination": "non_hallucination",
    "evidence_consistency": "source_consistency",
    "usefulness": "informativeness",
}
EXISTING_PAIR_CRITERIA = [
    "plausibility", "completeness", "source_consistency",
    "non_hallucination", "informativeness", "conciseness",
]


# ── stats (kept identical to scripts/manual_validation_sample.py) ──────────────
def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def cohen_kappa(x: list[float], y: list[float], threshold: float = 0.5) -> float:
    bx = [1 if v >= threshold else 0 for v in x]
    by = [1 if v >= threshold else 0 for v in y]
    n = len(bx)
    if not n:
        return float("nan")
    po = sum(1 for a, b in zip(bx, by) if a == b) / n
    px1, py1 = sum(bx) / n, sum(by) / n
    pe = px1 * py1 + (1 - px1) * (1 - py1)
    return (po - pe) / (1 - pe) if (1 - pe) else float("nan")


def pct_agreement(x: list[float], y: list[float], threshold: float = 0.5) -> float:
    bx = [1 if v >= threshold else 0 for v in x]
    by = [1 if v >= threshold else 0 for v in y]
    return sum(1 for a, b in zip(bx, by) if a == b) / len(bx) if bx else float("nan")


def mean_abs_diff(x: list[float], y: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x) if x else float("nan")


# ── data loading ──────────────────────────────────────────────────────────────
def load_sample() -> dict[int, dict]:
    out = {}
    for line in SAMPLE_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[int(r["id"])] = r
    return out


def load_human() -> dict[int, dict]:
    out = {}
    with HUMAN_PATH.open() as fh:
        for row in csv.DictReader(fh):
            try:
                rid = int(row["id"])
            except (KeyError, ValueError):
                continue
            scores = {}
            for c in HUMAN_CRITERIA:
                v = (row.get(c) or "").strip()
                if v != "":
                    try:
                        scores[c] = float(v)
                    except ValueError:
                        pass
            if len(scores) == len(HUMAN_CRITERIA):  # only fully-scored rows
                out[rid] = scores
    return out


def _paired(ids, a: dict, b: dict, ka: str, kb: str):
    xs, ys = [], []
    for i in ids:
        va = a.get(i, {}).get(ka)
        vb = b.get(i, {}).get(kb)
        if va is not None and vb is not None:
            xs.append(float(va))
            ys.append(float(vb))
    return xs, ys


def _stat_row(xs, ys) -> dict:
    return {
        "n": len(xs),
        "spearman_rho": round(spearman_rho(xs, ys), 4),
        "cohen_kappa": round(cohen_kappa(xs, ys), 4),
        "pct_agreement": round(pct_agreement(xs, ys), 4),
        "mean_abs_diff": round(mean_abs_diff(xs, ys), 4),
        "mean_a": round(sum(xs) / len(xs), 4) if xs else None,
        "mean_b": round(sum(ys) / len(ys), 4) if ys else None,
    }


def main() -> None:
    if not HUMAN_PATH.exists():
        raise SystemExit(
            f"missing {HUMAN_PATH}\n"
            "Annotate via analysis/manual_validation/human_annotation.html, then "
            "save the exported CSV there."
        )
    sample = load_sample()
    human = load_human()
    if not human:
        raise SystemExit("no fully-scored human rows found in the CSV")

    gemma = {i: r.get("gemma_scores", {}) for i, r in sample.items()}
    kimi = {i: r.get("kimi_scores", {}) for i, r in sample.items()}
    third = {i: r.get("third_judge_scores", {}) for i, r in sample.items()}
    ids = sorted(human)

    report: dict = {"n_human_scored": len(human), "sample_size": len(sample), "sections": {}}

    # human vs third judge (identical 5 criteria)
    sec = {}
    for c in HUMAN_CRITERIA:
        sec[c] = _stat_row(*_paired(ids, human, third, c, c))
    report["sections"]["human_vs_third_judge"] = sec

    # human vs gemma / kimi (three mappable concepts)
    for name, judge in (("human_vs_gemma", gemma), ("human_vs_kimi", kimi)):
        sec = {}
        for hk, ek in HUMAN_TO_EXISTING.items():
            sec[f"{hk} ↔ {ek}"] = _stat_row(*_paired(ids, human, judge, hk, ek))
        report["sections"][name] = sec

    # judge vs judge (gemma vs kimi) on the full sample, for reference
    all_ids = sorted(sample)
    sec = {}
    for c in EXISTING_PAIR_CRITERIA:
        sec[c] = _stat_row(*_paired(all_ids, gemma, kimi, c, c))
    report["sections"]["gemma_vs_kimi"] = sec

    # human pass-rate per criterion (sanity / base rates)
    report["human_pass_rate"] = {
        c: round(sum(1 for i in ids if human[i][c] >= 0.5) / len(ids), 4)
        for c in HUMAN_CRITERIA
    }

    (MV / "agreement_human_vs_judges.json").write_text(json.dumps(report, indent=2))
    _write_md(report)
    print(f"scored {len(human)} human-annotated rationales")
    print("  → analysis/manual_validation/agreement_human_vs_judges.{json,md}")
    hv = report["sections"]["human_vs_third_judge"]
    for c in HUMAN_CRITERIA:
        s = hv[c]
        print(f"  human vs third  {c:22s} κ={s['cohen_kappa']:+.3f}  ρ={s['spearman_rho']:+.3f}  agree={s['pct_agreement']:.0%}")


def _tbl(sec: dict) -> list[str]:
    out = ["| Criterion | n | Spearman ρ | Cohen κ | % agree | Mean \\|diff\\| | Human | Judge |",
           "|-----------|---|-----------|---------|---------|------------|-------|-------|"]
    for crit, s in sec.items():
        out.append(
            f"| {crit} | {s['n']} | {s['spearman_rho']:.3f} | {s['cohen_kappa']:.3f} | "
            f"{s['pct_agreement']:.0%} | {s['mean_abs_diff']:.3f} | {s['mean_a']} | {s['mean_b']} |"
        )
    return out


def _write_md(report: dict) -> None:
    L = [
        "# Human validation: human labels vs LLM judges",
        "",
        f"Human-annotated rationales: **{report['n_human_scored']}** "
        f"(of {report['sample_size']} in the sample).",
        "",
        "Cohen κ binarises scores at 0.5; % agree is the binary match rate; "
        "Spearman ρ uses the raw 0/0.5/1 scale. Higher = the LLM judge tracks "
        "the human better.",
        "",
        "## Human vs GPT/Gemma third judge (same 5 criteria)",
        "",
        *_tbl(report["sections"]["human_vs_third_judge"]),
        "",
        "## Human vs Gemma judge (mappable criteria)",
        "",
        *_tbl(report["sections"]["human_vs_gemma"]),
        "",
        "## Human vs Kimi judge (mappable criteria)",
        "",
        *_tbl(report["sections"]["human_vs_kimi"]),
        "",
        "## Gemma vs Kimi (the two judges, full sample)",
        "",
        *_tbl(report["sections"]["gemma_vs_kimi"]),
        "",
        "## Human pass-rate per criterion",
        "",
        "| Criterion | Pass-rate (score ≥ 0.5) |",
        "|-----------|--------------------------|",
        *[f"| {c} | {r:.0%} |" for c, r in report["human_pass_rate"].items()],
        "",
        "## Reading these numbers",
        "",
        "- κ ≥ 0.6 = substantial, 0.4–0.6 = moderate, < 0.4 = weak human↔judge "
        "agreement. Weak agreement on a criterion is evidence the LLM judge is "
        "*not* a safe stand-in for a human on that dimension.",
        "- `criterion_alignment` and `probability_support` have no analogue in "
        "the Gemma/Kimi 6-criteria rubric, so they are only validated against "
        "the third judge.",
    ]
    (MV / "agreement_human_vs_judges.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
