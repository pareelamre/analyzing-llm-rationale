#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover one variant by looping over nulls and missing ids.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir-name", default=None)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--temperature-tag", required=True)
    parser.add_argument("--max-attempts", default="6")
    parser.add_argument("--request-timeout-s", default=None)
    parser.add_argument("--max-null-rounds", type=int, default=3)
    parser.add_argument("--max-missing-rounds", type=int, default=4)
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
    nulls = {rid for rid, row in seen.items() if row.get("predicted_answer") is None}
    complete = len(expected_ids) - len(missing) - len(nulls)
    return complete, len(nulls), len(missing)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run_variant_once(
    variant: str,
    model: str,
    temperature: str,
    temperature_tag: str,
    max_attempts: str,
    request_timeout_s: str | None,
    reprocess_nulls: bool,
) -> int:
    cmd = [
        sys.executable,
        "scripts/run_variant.py",
        "--variant",
        variant,
        "--model",
        model,
        "--temperature",
        temperature,
        "--temperature-tag",
        temperature_tag,
        "--max-attempts",
        str(max_attempts),
    ]
    if request_timeout_s is not None:
        cmd.extend(["--request-timeout-s", str(request_timeout_s)])
    if reprocess_nulls:
        cmd.append("--reprocess-nulls")
    mode = "NULL" if reprocess_nulls else "MISSING"
    log(f"RUN {mode} {temperature_tag} {variant}")
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    log(f"EXIT {proc.returncode} {temperature_tag} {variant}")
    return proc.returncode


def main() -> int:
    args = build_parser().parse_args()
    results_dir_name = args.results_dir_name or args.model
    expected_ids = load_expected_ids()

    complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, args.variant, expected_ids)
    log(f"START {args.temperature_tag} {args.variant} complete={complete} nulls={nulls} missing={missing}")

    for round_idx in range(1, args.max_null_rounds + 1):
        complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, args.variant, expected_ids)
        log(f"BEFORE_NULL round={round_idx} {args.temperature_tag} {args.variant} complete={complete} nulls={nulls} missing={missing}")
        if nulls == 0:
            break
        rc = run_variant_once(
            args.variant,
            args.model,
            args.temperature,
            args.temperature_tag,
            args.max_attempts,
            args.request_timeout_s,
            reprocess_nulls=True,
        )
        if rc != 0:
            return rc

    for round_idx in range(1, args.max_missing_rounds + 1):
        complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, args.variant, expected_ids)
        log(f"BEFORE_MISSING round={round_idx} {args.temperature_tag} {args.variant} complete={complete} nulls={nulls} missing={missing}")
        if missing == 0:
            break
        rc = run_variant_once(
            args.variant,
            args.model,
            args.temperature,
            args.temperature_tag,
            args.max_attempts,
            args.request_timeout_s,
            reprocess_nulls=False,
        )
        if rc != 0:
            return rc
        complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, args.variant, expected_ids)
        if nulls > 0:
            log(f"CLEANUP_NULL_AFTER_MISSING {args.temperature_tag} {args.variant} complete={complete} nulls={nulls} missing={missing}")
            rc = run_variant_once(
                args.variant,
                args.model,
                args.temperature,
                args.temperature_tag,
                args.max_attempts,
                args.request_timeout_s,
                reprocess_nulls=True,
            )
            if rc != 0:
                return rc

    complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, args.variant, expected_ids)
    log(f"FINAL {args.temperature_tag} {args.variant} complete={complete} nulls={nulls} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
