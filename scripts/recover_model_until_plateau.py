#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_metrics import accuracy, brier_score, ece, iter_examples, load_targets

DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a model's null predictions temperature by temperature, "
            "re-scoring after each sweep until nulls and performance plateau."
        )
    )
    parser.add_argument("--model", required=True, help="Model key from configs/models.yaml.")
    parser.add_argument(
        "--results-dir-name",
        required=True,
        help="Directory name under results/ for this model, e.g. GPT-OSS-120B.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="*",
        default=None,
        help="Temperature directory tags to target. Defaults to only temperatures with gaps.",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--max-attempts", default="6")
    parser.add_argument("--request-timeout-s", default=None)
    parser.add_argument("--max-null-rounds", type=int, default=3)
    parser.add_argument("--max-missing-rounds", type=int, default=4)
    parser.add_argument("--temperature-max-sweeps", type=int, default=1)
    parser.add_argument("--max-model-sweeps", type=int, default=12)
    parser.add_argument("--plateau-rounds", type=int, default=2)
    parser.add_argument("--accuracy-tol", type=float, default=1e-6)
    parser.add_argument("--brier-tol", type=float, default=1e-6)
    parser.add_argument("--ece-tol", type=float, default=1e-6)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--child-log-dir", default="logs")
    return parser


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def summarize_result_file(result_path: Path, expected_ids: set[int]) -> tuple[int, int, int]:
    records = json.loads(result_path.read_text())
    seen = {}
    for row in records:
        seen[row.get("id")] = row
    missing = expected_ids - set(seen)
    nulls = {
        rid
        for rid, row in seen.items()
        if rid in expected_ids and (
            row.get("predicted_answer") is None or row.get("confidence") is None
        )
    }
    complete = len(expected_ids) - len(missing) - len(nulls)
    return complete, len(nulls), len(missing)


def summarize_temperature(temp_dir: Path, expected_ids: set[int]) -> dict[str, int]:
    total_complete = 0
    total_nulls = 0
    total_missing = 0
    file_count = 0
    for result_path in sorted(temp_dir.glob("results_variant*.json")):
        complete, nulls, missing = summarize_result_file(result_path, expected_ids)
        total_complete += complete
        total_nulls += nulls
        total_missing += missing
        file_count += 1
    return {
        "complete": total_complete,
        "nulls": total_nulls,
        "missing": total_missing,
        "files": file_count,
    }


def discover_target_temperatures(results_dir: Path, expected_ids: set[int]) -> list[str]:
    selected: list[str] = []
    for temp_dir in sorted((p for p in results_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        summary = summarize_temperature(temp_dir, expected_ids)
        if summary["nulls"] > 0 or summary["missing"] > 0:
            selected.append(temp_dir.name)
    return selected


def evaluate_temperatures(
    results_dir: Path,
    temperatures: list[str],
    targets: dict[int, int],
    bins: int,
) -> dict[str, float | int]:
    runs = 0
    total_scored = 0
    total_missing = 0
    acc_sum = 0.0
    brier_sum = 0.0
    ece_sum = 0.0

    for temp_tag in temperatures:
        temp_dir = results_dir / temp_tag
        for result_path in sorted(temp_dir.glob("results_variant*.json")):
            payload = json.loads(result_path.read_text())
            examples, missing = iter_examples(payload, targets)
            if not examples:
                continue
            runs += 1
            total_scored += len(examples)
            total_missing += missing
            acc_sum += accuracy(examples)
            brier_sum += brier_score(examples)
            ece_sum += ece(examples, bins)

    if runs == 0:
        return {
            "runs": 0,
            "total_scored": 0,
            "total_missing": total_missing,
            "mean_accuracy": 0.0,
            "mean_brier_score": 0.0,
            "mean_ece": 0.0,
        }

    return {
        "runs": runs,
        "total_scored": total_scored,
        "total_missing": total_missing,
        "mean_accuracy": acc_sum / runs,
        "mean_brier_score": brier_sum / runs,
        "mean_ece": ece_sum / runs,
    }


def summarize_temperatures(
    results_dir: Path,
    temperatures: list[str],
    expected_ids: set[int],
) -> tuple[dict[str, dict[str, int]], int, int]:
    per_temp: dict[str, dict[str, int]] = {}
    total_nulls = 0
    total_missing = 0
    for temp_tag in temperatures:
        summary = summarize_temperature(results_dir / temp_tag, expected_ids)
        per_temp[temp_tag] = summary
        total_nulls += summary["nulls"]
        total_missing += summary["missing"]
    return per_temp, total_nulls, total_missing


def run_temperature_recovery(args: argparse.Namespace, temp_tag: str) -> int:
    temperature_value = temp_tag.removeprefix("temperature_")
    if temperature_value in {"0", "00", "000"}:
        temperature_arg = "0.0"
    elif len(temperature_value) == 3 and temperature_value.isdigit():
        temperature_arg = str(int(temperature_value) / 1000.0)
    else:
        temperature_arg = temperature_value

    cmd = [
        sys.executable,
        "scripts/recover_temperature_parallel.py",
        "--model",
        args.model,
        "--results-dir-name",
        args.results_dir_name,
        "--temperature",
        temperature_arg,
        "--temperature-tag",
        temp_tag,
        "--max-attempts",
        str(args.max_attempts),
        "--max-null-rounds",
        str(args.max_null_rounds),
        "--max-missing-rounds",
        str(args.max_missing_rounds),
        "--max-sweeps",
        str(args.temperature_max_sweeps),
        "--child-log-dir",
        args.child_log_dir,
    ]
    if args.request_timeout_s is not None:
        cmd.extend(["--request-timeout-s", str(args.request_timeout_s)])

    log(f"RUN_TEMPERATURE {temp_tag}")
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    log(f"EXIT_TEMPERATURE {temp_tag} rc={proc.returncode}")
    return proc.returncode


def main() -> int:
    args = build_parser().parse_args()
    results_dir = ROOT / "results" / args.results_dir_name
    expected_ids = set(load_targets(args.dataset).keys())
    targets = load_targets(args.dataset)

    if args.temperatures:
        temperatures = list(args.temperatures)
    else:
        temperatures = discover_target_temperatures(results_dir, expected_ids)

    if not temperatures:
        log("No temperatures with nulls or missing rows were found.")
        return 0

    log(f"TARGET_TEMPERATURES {' '.join(temperatures)}")

    per_temp, prev_nulls, prev_missing = summarize_temperatures(results_dir, temperatures, expected_ids)
    metrics = evaluate_temperatures(results_dir, temperatures, targets, args.bins)
    log(
        "START "
        f"nulls={prev_nulls} missing={prev_missing} "
        f"mean_accuracy={metrics['mean_accuracy']:.6f} "
        f"mean_brier={metrics['mean_brier_score']:.6f} "
        f"mean_ece={metrics['mean_ece']:.6f}"
    )
    for temp_tag in temperatures:
        summary = per_temp[temp_tag]
        log(
            f"TEMP {temp_tag} complete={summary['complete']} "
            f"nulls={summary['nulls']} missing={summary['missing']} files={summary['files']}"
        )

    best_accuracy = float(metrics["mean_accuracy"])
    best_brier = float(metrics["mean_brier_score"])
    best_ece = float(metrics["mean_ece"])
    plateau_rounds = 0

    for sweep_idx in range(1, args.max_model_sweeps + 1):
        outstanding = [temp for temp in temperatures if per_temp[temp]["nulls"] > 0 or per_temp[temp]["missing"] > 0]
        if not outstanding:
            log("All targeted temperatures are complete.")
            return 0

        log(f"MODEL_SWEEP {sweep_idx} outstanding={' '.join(outstanding)}")
        for temp_tag in outstanding:
            rc = run_temperature_recovery(args, temp_tag)
            if rc != 0:
                return rc

        per_temp, new_nulls, new_missing = summarize_temperatures(results_dir, temperatures, expected_ids)
        metrics = evaluate_temperatures(results_dir, temperatures, targets, args.bins)
        log(
            "AFTER_MODEL_SWEEP "
            f"sweep={sweep_idx} nulls={new_nulls} missing={new_missing} "
            f"mean_accuracy={metrics['mean_accuracy']:.6f} "
            f"mean_brier={metrics['mean_brier_score']:.6f} "
            f"mean_ece={metrics['mean_ece']:.6f}"
        )
        for temp_tag in temperatures:
            summary = per_temp[temp_tag]
            log(
                f"TEMP {temp_tag} complete={summary['complete']} "
                f"nulls={summary['nulls']} missing={summary['missing']} files={summary['files']}"
            )

        improved_nulls = new_nulls < prev_nulls or new_missing < prev_missing
        improved_accuracy = float(metrics["mean_accuracy"]) > best_accuracy + args.accuracy_tol
        improved_brier = float(metrics["mean_brier_score"]) < best_brier - args.brier_tol
        improved_ece = float(metrics["mean_ece"]) < best_ece - args.ece_tol

        if improved_accuracy:
            best_accuracy = float(metrics["mean_accuracy"])
        if improved_brier:
            best_brier = float(metrics["mean_brier_score"])
        if improved_ece:
            best_ece = float(metrics["mean_ece"])

        if improved_nulls or improved_accuracy or improved_brier or improved_ece:
            plateau_rounds = 0
            log(
                f"PROGRESS sweep={sweep_idx} "
                f"improved_nulls={improved_nulls} "
                f"improved_accuracy={improved_accuracy} "
                f"improved_brier={improved_brier} "
                f"improved_ece={improved_ece}"
            )
        else:
            plateau_rounds += 1
            log(f"PLATEAU sweep={sweep_idx} plateau_rounds={plateau_rounds}")

        prev_nulls, prev_missing = new_nulls, new_missing

        if new_nulls == 0 and new_missing == 0:
            log("Recovery complete: no nulls or missing rows remain.")
            return 0
        if plateau_rounds >= args.plateau_rounds:
            log("Stopping because recovery performance plateaued.")
            return 0

    log(f"Stopping because max model sweeps ({args.max_model_sweeps}) was reached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
