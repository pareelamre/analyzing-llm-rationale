#!/usr/bin/env python3
"""Compare baseline forecasts with faithfulness perturbation forecasts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "analysis" / "faithfulness" / "faithfulness_perturbation_input.json"
DEFAULT_BASELINE_CANDIDATES = (
    ROOT / "results" / "Qwen2.5-7b-instruct" / "temperature_000" / "results_variant0_neutral_baseline.json",
    ROOT / "results" / "Qwen2.5-7b-instruct" / "temperature_0" / "results_variant0_neutral_baseline.json",
)
DEFAULT_BASELINE = next((path for path in DEFAULT_BASELINE_CANDIDATES if path.exists()), DEFAULT_BASELINE_CANDIDATES[0])
DEFAULT_RESULTS = ROOT / "analysis" / "faithfulness" / "results_variant0_neutral_baseline.json"
DEFAULT_OUT_CSV = ROOT / "analysis" / "faithfulness" / "faithfulness_perturbation_pairs.csv"
DEFAULT_OUT_MD = ROOT / "analysis" / "faithfulness" / "faithfulness_perturbation_summary.md"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_answer(value: object) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in {"yes", "no"} else None


def normalize_confidence(value: object) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(confidence) or math.isinf(confidence):
        return None
    if 0.0 <= confidence <= 1.0:
        return confidence
    return None


def p_yes(row: dict) -> float | None:
    answer = normalize_answer(row.get("predicted_answer"))
    confidence = normalize_confidence(row.get("confidence"))
    if answer is None or confidence is None:
        return None
    return confidence if answer == "yes" else 1.0 - confidence


def paired_rows(perturbation_input: list[dict], baseline: list[dict], perturbation_results: list[dict]) -> list[dict]:
    baseline_by_id = {int(row["id"]): row for row in baseline if row.get("id") is not None}
    perturbed_by_id = {int(row["id"]): row for row in perturbation_results if row.get("id") is not None}
    metadata_by_id = {int(row["id"]): row for row in perturbation_input if row.get("id") is not None}

    rows: list[dict] = []
    for perturbation_id, metadata in sorted(metadata_by_id.items()):
        original_id = int(metadata["original_id"])
        baseline_row = baseline_by_id.get(original_id)
        perturbed_row = perturbed_by_id.get(perturbation_id)
        if baseline_row is None or perturbed_row is None:
            continue
        baseline_p = p_yes(baseline_row)
        perturbed_p = p_yes(perturbed_row)
        if baseline_p is None or perturbed_p is None:
            continue
        baseline_answer = normalize_answer(baseline_row.get("predicted_answer"))
        perturbed_answer = normalize_answer(perturbed_row.get("predicted_answer"))
        rows.append(
            {
                "original_id": original_id,
                "perturbation_id": perturbation_id,
                "perturbation_type": metadata.get("perturbation_type"),
                "baseline_answer": baseline_answer,
                "perturbed_answer": perturbed_answer,
                "baseline_confidence": baseline_row.get("confidence"),
                "perturbed_confidence": perturbed_row.get("confidence"),
                "baseline_p_yes": baseline_p,
                "perturbed_p_yes": perturbed_p,
                "delta_p_yes": perturbed_p - baseline_p,
                "abs_delta_p_yes": abs(perturbed_p - baseline_p),
                "answer_flipped": int(baseline_answer != perturbed_answer),
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["perturbation_type"]), []).append(row)

    summary_rows: list[dict] = []
    for test_name, group in sorted(groups.items()):
        abs_deltas = [float(row["abs_delta_p_yes"]) for row in group]
        signed_deltas = [float(row["delta_p_yes"]) for row in group]
        flips = [int(row["answer_flipped"]) for row in group]
        summary_rows.append(
            {
                "perturbation_type": test_name,
                "n": len(group),
                "mean_delta_p_yes": mean(signed_deltas),
                "mean_abs_delta_p_yes": mean(abs_deltas),
                "median_abs_delta_p_yes": median(abs_deltas),
                "share_abs_delta_ge_0_05": sum(delta >= 0.05 for delta in abs_deltas) / len(abs_deltas),
                "share_abs_delta_ge_0_10": sum(delta >= 0.10 for delta in abs_deltas) / len(abs_deltas),
                "answer_flip_rate": sum(flips) / len(flips),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(path: Path, summary_rows: list[dict], pair_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Rationale Faithfulness Perturbation Summary",
        "",
        f"Paired perturbation forecasts analyzed: {pair_count}",
        "",
        "| Perturbation | n | Mean dP(Yes) | Mean abs dP(Yes) | Median abs dP(Yes) | >=5pp | >=10pp | Answer flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {perturbation_type} | {n} | {mean_delta_p_yes} | {mean_abs_delta_p_yes} | "
            "{median_abs_delta_p_yes} | {share_abs_delta_ge_0_05} | "
            "{share_abs_delta_ge_0_10} | {answer_flip_rate} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: low movement under order perturbation alone is not sufficient evidence of direct rationale faithfulness. "
            "Evidence masking, contradiction injection, actor/date swaps, and criterion swaps test whether forecasts move when "
            "the support, stated rationale, target entity/time, or resolution rule is perturbed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturbation-input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline-results-path", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--perturbation-results-path", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    perturbation_input = load_json(args.perturbation_input_path)
    baseline = load_json(args.baseline_results_path)
    perturbation_results = load_json(args.perturbation_results_path)
    if not all(isinstance(payload, list) for payload in (perturbation_input, baseline, perturbation_results)):
        raise SystemExit("all inputs must be JSON lists")
    pairs = paired_rows(perturbation_input, baseline, perturbation_results)
    summary_rows = summarize(pairs)
    write_csv(args.out_csv, pairs)
    write_markdown(args.out_md, summary_rows, len(pairs))
    print(json.dumps({"pairs": len(pairs), "out_csv": str(args.out_csv), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
