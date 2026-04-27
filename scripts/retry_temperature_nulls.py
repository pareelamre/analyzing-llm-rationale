#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

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


def needs_reprocess(row: object) -> bool:
    if not isinstance(row, dict):
        return True
    answer = row.get("predicted_answer")
    normalized_answer = str(answer).strip().lower() if answer is not None else None
    if normalized_answer not in {"yes", "no"}:
        return True
    confidence = row.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return True
    return not (0.0 <= confidence <= 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retry null predictions for a model/temperature, variant by variant.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir-name", default=None)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--temperature-tag", required=True)
    parser.add_argument("--max-attempts", default="6")
    parser.add_argument("--request-timeout-s", default=None)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--variants", nargs="*", default=VARIANT_ORDER)
    return parser


def load_expected_ids() -> set[int]:
    dataset_path = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
    return {row["id"] for row in json.loads(dataset_path.read_text())}


def summarize_results(results_dir_name: str, temp_tag: str, variant: str, expected_ids: set[int]) -> tuple[int, int, int]:
    path = ROOT / "results" / results_dir_name / temp_tag / f"results_{variant}.json"
    records = json.loads(path.read_text())
    seen = {}
    for row in records:
        seen[row.get("id")] = row
    missing = expected_ids - set(seen)
    nulls = {rid for rid, row in seen.items() if rid in expected_ids and needs_reprocess(row)}
    complete = len(expected_ids) - len(missing) - len(nulls)
    return complete, len(nulls), len(missing)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    expected_ids = load_expected_ids()
    results_dir_name = args.results_dir_name or args.model

    for variant in args.variants:
        complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, variant, expected_ids)
        log(
            f"BEFORE {args.temperature_tag} {variant} complete={complete} nulls={nulls} missing={missing}"
        )
        if nulls == 0:
            continue

        for round_idx in range(1, args.max_rounds + 1):
            complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, variant, expected_ids)
            log(
                f"ROUND {round_idx} {args.temperature_tag} {variant} complete={complete} nulls={nulls} missing={missing}"
            )
            if nulls == 0:
                break

            cmd = [
                sys.executable,
                "scripts/run_variant.py",
                "--variant",
                variant,
                "--model",
                args.model,
                "--temperature",
                args.temperature,
                "--temperature-tag",
                args.temperature_tag,
                "--max-attempts",
                str(args.max_attempts),
                "--reprocess-nulls",
            ]
            if args.request_timeout_s is not None:
                cmd.extend(["--request-timeout-s", str(args.request_timeout_s)])
            log(f"RUN NULL {args.temperature_tag} {variant}")
            proc = subprocess.run(cmd, cwd=ROOT, text=True)
            log(f"EXIT {proc.returncode} {args.temperature_tag} {variant}")
            if proc.returncode != 0:
                return proc.returncode

        complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, variant, expected_ids)
        log(
            f"FINAL {args.temperature_tag} {variant} complete={complete} nulls={nulls} missing={missing}"
        )

    log(f"DONE {args.temperature_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
