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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover one temperature in parallel until aggregate nulls stop decreasing."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir-name", default=None)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--temperature-tag", required=True)
    parser.add_argument("--max-attempts", default="6")
    parser.add_argument("--request-timeout-s", default=None)
    parser.add_argument("--max-null-rounds", type=int, default=3)
    parser.add_argument("--max-missing-rounds", type=int, default=4)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--variants", nargs="*", default=VARIANT_ORDER)
    parser.add_argument("--child-log-dir", default="logs")
    return parser


def load_expected_ids() -> set[int]:
    dataset_path = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
    return {row["id"] for row in json.loads(dataset_path.read_text())}


def summarize_variant(results_dir_name: str, temp_tag: str, variant: str, expected_ids: set[int]) -> tuple[int, int, int]:
    path = ROOT / "results" / results_dir_name / temp_tag / f"results_{variant}.json"
    records = json.loads(path.read_text())
    seen = {}
    for row in records:
        seen[row.get("id")] = row
    missing = expected_ids - set(seen)
    nulls = {rid for rid, row in seen.items() if row.get("predicted_answer") is None}
    complete = len(expected_ids) - len(missing) - len(nulls)
    return complete, len(nulls), len(missing)


def summarize_temperature(
    results_dir_name: str,
    temp_tag: str,
    variants: list[str],
    expected_ids: set[int],
) -> tuple[int, int, int, dict[str, tuple[int, int, int]]]:
    per_variant: dict[str, tuple[int, int, int]] = {}
    total_complete = 0
    total_nulls = 0
    total_missing = 0
    for variant in variants:
        complete, nulls, missing = summarize_variant(results_dir_name, temp_tag, variant, expected_ids)
        per_variant[variant] = (complete, nulls, missing)
        total_complete += complete
        total_nulls += nulls
        total_missing += missing
    return total_complete, total_nulls, total_missing, per_variant


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def spawn_recovery_worker(
    variant: str,
    args: argparse.Namespace,
    sweep_idx: int,
) -> tuple[subprocess.Popen[str], Path]:
    child_log_dir = ROOT / args.child_log_dir
    child_log_dir.mkdir(parents=True, exist_ok=True)
    child_log_path = child_log_dir / (
        f"{args.model}_{args.temperature_tag}_{variant}_sweep{sweep_idx:02d}.log"
    )
    cmd = [
        sys.executable,
        "scripts/recover_variant.py",
        "--variant",
        variant,
        "--model",
        args.model,
        "--results-dir-name",
        args.results_dir_name or args.model,
        "--temperature",
        args.temperature,
        "--temperature-tag",
        args.temperature_tag,
        "--max-attempts",
        str(args.max_attempts),
        "--max-null-rounds",
        str(args.max_null_rounds),
        "--max-missing-rounds",
        str(args.max_missing_rounds),
    ]
    if args.request_timeout_s is not None:
        cmd.extend(["--request-timeout-s", str(args.request_timeout_s)])
    log(f"SPAWN sweep={sweep_idx} {variant} log={child_log_path}")
    log_handle = child_log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._codex_log_handle = log_handle  # type: ignore[attr-defined]
    return proc, child_log_path


def wait_for_workers(
    workers: dict[str, tuple[subprocess.Popen[str], Path]],
    sweep_idx: int,
) -> int:
    for variant, (proc, child_log_path) in workers.items():
        rc = proc.wait()
        log(f"EXIT sweep={sweep_idx} {variant} rc={rc} log={child_log_path}")
        log_handle = getattr(proc, "_codex_log_handle", None)
        if log_handle is not None:
            log_handle.close()
        if rc != 0:
            return rc
    return 0


def main() -> int:
    args = build_parser().parse_args()
    results_dir_name = args.results_dir_name or args.model
    expected_ids = load_expected_ids()

    prev_complete, prev_nulls, prev_missing, _ = summarize_temperature(
        results_dir_name,
        args.temperature_tag,
        list(args.variants),
        expected_ids,
    )
    log(
        f"START {args.temperature_tag} complete={prev_complete} nulls={prev_nulls} missing={prev_missing}"
    )

    if prev_nulls == 0 and prev_missing == 0:
        log(f"ALREADY_COMPLETE {args.temperature_tag}")
        return 0

    for sweep_idx in range(1, args.max_sweeps + 1):
        log(
            f"BEFORE_SWEEP sweep={sweep_idx} {args.temperature_tag} "
            f"complete={prev_complete} nulls={prev_nulls} missing={prev_missing}"
        )
        workers = {
            variant: spawn_recovery_worker(variant, args, sweep_idx)
            for variant in args.variants
        }
        rc = wait_for_workers(workers, sweep_idx)
        if rc != 0:
            return rc

        new_complete, new_nulls, new_missing, per_variant = summarize_temperature(
            results_dir_name,
            args.temperature_tag,
            list(args.variants),
            expected_ids,
        )
        log(
            f"AFTER_SWEEP sweep={sweep_idx} {args.temperature_tag} "
            f"complete={new_complete} nulls={new_nulls} missing={new_missing}"
        )
        for variant in args.variants:
            complete, nulls, missing = per_variant[variant]
            log(
                f"VARIANT sweep={sweep_idx} {variant} "
                f"complete={complete} nulls={nulls} missing={missing}"
            )

        if new_nulls == 0 and new_missing == 0:
            log(f"COMPLETE {args.temperature_tag}")
            return 0
        if new_nulls >= prev_nulls:
            log(
                f"STOP_NO_NULL_IMPROVEMENT sweep={sweep_idx} {args.temperature_tag} "
                f"prev_nulls={prev_nulls} new_nulls={new_nulls}"
            )
            return 0

        prev_complete, prev_nulls, prev_missing = new_complete, new_nulls, new_missing

    log(f"STOP_MAX_SWEEPS {args.temperature_tag} sweeps={args.max_sweeps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
