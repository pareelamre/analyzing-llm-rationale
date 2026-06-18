#!/usr/bin/env python3
"""Accuracy / Brier / ECE stratified by forecast-time lead.

Answers the methodological question "is this ex-ante forecasting or retrospective
evidence-conditioned resolution?" by comparing the same model/variant run at several
forecast cutoffs. Reads the directory layout produced by `run-batch`:

    results/<model>/<temp>/results_<variant>.json            # lead = none (oracle evidence)
    results/<model>/<temp>/lead_<NN>d/results_<variant>.json  # lead = NN days

Flat accuracy across leads => the task was retrospectively (near-)trivial.
Accuracy that degrades as the lead grows => genuine forecasting signal at horizon.

Usage:
    python scripts/lead_time_accuracy.py --model GPT-OSS-120B
    python scripts/lead_time_accuracy.py --model GPT-OSS-120B --temperature-dir temperature_070 \
        --variant variant0_neutral_baseline --output-csv analysis/lead_time_accuracy.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Optional

from analyzing_llm_rationale.metrics import (
    accuracy,
    brier_score,
    ece,
    iter_examples,
    load_targets,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
RESULTS_ROOT = ROOT / "results"

# Sorts "none" (oracle) first, then ascending lead. Larger lead == longer horizon.
_NONE_LEAD = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model directory name under results/.")
    parser.add_argument(
        "--temperature-dir",
        default=None,
        help="Restrict to one temperature dir (e.g. temperature_070). Default: all.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Restrict to one variant (e.g. variant0_neutral_baseline). Default: all.",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--bins", type=int, default=10, help="Equal-width bins for ECE.")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def lead_from_dir(dirname: str) -> Optional[int]:
    """Return the lead in days for a `lead_<NN>d` dir, or None if it is not one."""
    match = re.fullmatch(r"lead_(\d+)d", dirname)
    return int(match.group(1)) if match else None


def variant_from_filename(name: str) -> str:
    return name.removeprefix("results_").removesuffix(".json")


def mean_kept_fraction(rows: list[dict]) -> Optional[float]:
    fracs = []
    for row in rows:
        prov = row.get("forecast_cutoff")
        if isinstance(prov, dict) and prov.get("n_articles_total"):
            fracs.append(prov["n_articles_kept"] / prov["n_articles_total"])
    return mean(fracs) if fracs else None


def collect_result_files(model_dir: Path, temperature_dir: Optional[str]) -> list[tuple[str, int, Path]]:
    """Yield (temperature_dir, lead_days, result_path). lead_days == _NONE_LEAD for oracle."""
    found: list[tuple[str, int, Path]] = []
    temp_dirs = (
        [model_dir / temperature_dir]
        if temperature_dir
        else sorted(p for p in model_dir.iterdir() if p.is_dir())
    )
    for temp_dir in temp_dirs:
        if not temp_dir.is_dir():
            continue
        # Oracle (no-cutoff) runs live directly in the temperature dir.
        for result_path in sorted(temp_dir.glob("results_variant*.json")):
            found.append((temp_dir.name, _NONE_LEAD, result_path))
        # Cutoff runs live in lead_<NN>d/ subdirs.
        for sub in sorted(p for p in temp_dir.iterdir() if p.is_dir()):
            lead = lead_from_dir(sub.name)
            if lead is None:
                continue
            for result_path in sorted(sub.glob("results_variant*.json")):
                found.append((temp_dir.name, lead, result_path))
    return found


def main() -> None:
    args = parse_args()
    model_dir = args.results_root / args.model
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")
    targets = load_targets(args.dataset)

    rows_out: list[dict[str, object]] = []
    for temp_name, lead, result_path in collect_result_files(model_dir, args.temperature_dir):
        variant = variant_from_filename(result_path.name)
        if args.variant and variant != args.variant:
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        examples, missing = iter_examples(payload, targets)
        if not examples:
            continue
        rows_out.append(
            {
                "model": args.model,
                "temperature_dir": temp_name,
                "variant": variant,
                "lead_days": "none" if lead == _NONE_LEAD else lead,
                "_lead_sort": lead,
                "n_scored": len(examples),
                "n_missing": missing,
                "accuracy": round(accuracy(examples), 4),
                "brier_score": round(brier_score(examples), 4),
                "ece": round(ece(examples, args.bins), 4),
                "mean_kept_frac": (
                    round(mk, 4) if (mk := mean_kept_fraction(payload)) is not None else ""
                ),
            }
        )

    rows_out.sort(key=lambda r: (r["variant"], r["temperature_dir"], r["_lead_sort"]))
    for row in rows_out:
        row.pop("_lead_sort", None)

    if not rows_out:
        raise SystemExit("No scorable result files found for the given filters.")

    # Pretty table to stdout.
    cols = ["variant", "temperature_dir", "lead_days", "n_scored", "accuracy", "brier_score", "ece", "mean_kept_frac"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows_out)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows_out:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in cols))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols + ["n_missing", "model"])
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nWrote {len(rows_out)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
