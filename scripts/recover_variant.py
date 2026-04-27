#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


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
    parser.add_argument("--pass-max-records", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--stable-rounds-to-stop", type=int, default=4)
    parser.add_argument("--sleep-between-rounds-s", type=float, default=2.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def load_expected_ids() -> set[int]:
    dataset_path = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
    return {row["id"] for row in json.loads(dataset_path.read_text())}


def filter_expected_ids(expected_ids: set[int], shard_count: int, shard_index: int) -> set[int]:
    if shard_count <= 1:
        return expected_ids
    return {rid for rid in expected_ids if rid % shard_count == shard_index}


def summarize_results(results_dir_name: str, temp_tag: str, variant: str, expected_ids: set[int]) -> tuple[int, int, int]:
    path = ROOT / "results" / results_dir_name / temp_tag / f"results_{variant}.json"
    records = json.loads(path.read_text())
    seen = {}
    for row in records:
        seen[row.get("id")] = row
    missing = expected_ids - set(seen)
    nulls = {
        rid
        for rid, row in seen.items()
        if rid in expected_ids and needs_reprocess(row)
    }
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
    max_records: int,
    progress_every: int,
    shard_count: int,
    shard_index: int,
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
    if max_records > 0:
        cmd.extend(["--max-records", str(max_records)])
    if progress_every > 0:
        cmd.extend(["--progress-every", str(progress_every)])
    if request_timeout_s is not None:
        cmd.extend(["--request-timeout-s", str(request_timeout_s)])
    if reprocess_nulls:
        cmd.append("--reprocess-nulls")
    if shard_count > 1:
        cmd.extend(["--shard-count", str(shard_count), "--shard-index", str(shard_index)])
    mode = "NULL" if reprocess_nulls else "MISSING"
    log(f"RUN {mode} {temperature_tag} {variant}")
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    log(f"EXIT {proc.returncode} {temperature_tag} {variant}")
    return proc.returncode


def main() -> int:
    args = build_parser().parse_args()
    results_dir_name = args.results_dir_name or args.model
    expected_ids = filter_expected_ids(load_expected_ids(), args.shard_count, args.shard_index)

    complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, args.variant, expected_ids)
    log(f"START {args.temperature_tag} {args.variant} complete={complete} nulls={nulls} missing={missing}")

    null_round = 0
    missing_round = 0
    stable_rounds = 0
    consecutive_failures = 0

    while True:
        complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, args.variant, expected_ids)
        if nulls == 0 and missing == 0:
            break
        if stable_rounds >= args.stable_rounds_to_stop:
            log(
                f"STOP_STABLE {args.temperature_tag} {args.variant} "
                f"stable_rounds={stable_rounds} complete={complete} nulls={nulls} missing={missing}"
            )
            break

        run_nulls = nulls > 0 and (null_round < args.max_null_rounds or missing == 0)
        if run_nulls:
            null_round += 1
            mode = "NULL"
            log(
                f"BEFORE_NULL round={null_round} {args.temperature_tag} {args.variant} "
                f"complete={complete} nulls={nulls} missing={missing}"
            )
        else:
            if missing == 0 or missing_round >= args.max_missing_rounds:
                log(
                    f"STOP_LIMIT {args.temperature_tag} {args.variant} "
                    f"null_rounds={null_round} missing_rounds={missing_round} "
                    f"complete={complete} nulls={nulls} missing={missing}"
                )
                break
            missing_round += 1
            mode = "MISSING"
            log(
                f"BEFORE_MISSING round={missing_round} {args.temperature_tag} {args.variant} "
                f"complete={complete} nulls={nulls} missing={missing}"
            )

        before = (complete, nulls, missing)
        rc = run_variant_once(
            args.variant,
            args.model,
            args.temperature,
            args.temperature_tag,
            args.max_attempts,
            args.request_timeout_s,
            reprocess_nulls=(mode == "NULL"),
            max_records=args.pass_max_records,
            progress_every=args.progress_every,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
        if rc != 0:
            consecutive_failures += 1
            log(
                f"CHILD_FAIL {args.temperature_tag} {args.variant} mode={mode} rc={rc} "
                f"consecutive_failures={consecutive_failures}"
            )
            if consecutive_failures >= args.max_consecutive_failures:
                return rc
            time.sleep(args.sleep_between_rounds_s)
            continue

        consecutive_failures = 0
        complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, args.variant, expected_ids)
        after = (complete, nulls, missing)
        progressed = after != before and (after[0] > before[0] or after[1] < before[1] or after[2] < before[2])
        if progressed:
            stable_rounds = 0
            log(
                f"AFTER_{mode} {args.temperature_tag} {args.variant} "
                f"complete={complete} nulls={nulls} missing={missing}"
            )
        else:
            stable_rounds += 1
            log(
                f"NO_PROGRESS {args.temperature_tag} {args.variant} mode={mode} "
                f"stable_rounds={stable_rounds} complete={complete} nulls={nulls} missing={missing}"
            )
        if args.sleep_between_rounds_s > 0:
            time.sleep(args.sleep_between_rounds_s)

    complete, nulls, missing = summarize_results(results_dir_name, args.temperature_tag, args.variant, expected_ids)
    log(f"FINAL {args.temperature_tag} {args.variant} complete={complete} nulls={nulls} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
