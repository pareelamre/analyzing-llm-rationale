"""Cron-friendly watchdog for variant runs.

Design goals:
- No heredocs, no ~ expansion, no shell-sensitive quoting.
- Idempotent: if a worker for a variant is already running, do nothing.
- Chunked: runs at most one small chunk per variant per invocation.
- Checks for completion: output has all dataset ids and no null predicted_answer.

Usage examples:
  python scripts\\cron_watchdog.py --variants 3 4 --chunk-size 10
  python scripts\\cron_watchdog.py --variants 5 6 --chunk-size 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(r"C:\Users\paree\Documents\Analyzing Rationale of LLMs")
DATASET = BASE / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
OUT_DIR = BASE / "results" / "Qwen2.5-7b-instruct" / "temperature_00"

VARIANT_CFG = {
    3: {
        "out": OUT_DIR / "results_variant3_reasoning_type.json",
        "gen": BASE / "scripts" / "run_qwen_all_minimal_v3.py",
        "fix": BASE / "scripts" / "rerun_null_ids_drop_text_v3.py",
        "required": {"id", "predicted_answer", "confidence", "rationale", "reasoning_type"},
    },
    4: {
        "out": OUT_DIR / "results_variant4_credibility.json",
        "gen": BASE / "scripts" / "run_qwen_all_minimal_v4.py",
        "fix": BASE / "scripts" / "rerun_null_ids_drop_text_v4.py",
        "required": {"id", "predicted_answer", "confidence", "rationale"},
    },
    5: {
        "out": OUT_DIR / "results_variant5_key_conditions.json",
        "gen": BASE / "scripts" / "run_qwen_all_minimal_v5.py",
        "fix": BASE / "scripts" / "rerun_null_ids_drop_text_v5.py",
        "required": {"id", "predicted_answer", "confidence", "rationale", "key_conditions"},
    },
    6: {
        "out": OUT_DIR / "results_variant6_chain_of_thought.json",
        "gen": BASE / "scripts" / "run_qwen_all_minimal_v6.py",
        "fix": BASE / "scripts" / "rerun_null_ids_drop_text_v6.py",
        "required": {"id", "predicted_answer", "confidence", "rationale", "steps"},
    },
}


def load_json_retry(path: Path, *, tries: int = 3, sleep_s: float = 0.5):
    last = None
    for _ in range(tries):
        try:
            return json.load(path.open("r", encoding="utf-8"))
        except Exception as e:
            last = e
            time.sleep(sleep_s)
    raise last  # type: ignore[misc]


def dataset_goal_ids() -> set[int]:
    data = load_json_retry(DATASET)
    ids: set[int] = set()
    if isinstance(data, list):
        for r in data:
            if isinstance(r, dict) and isinstance(r.get("id"), int):
                ids.add(r["id"])
    return ids


def output_stats(path: Path, required: set[str], goal_ids: set[int]):
    if not path.exists():
        return {
            "exists": False,
            "written": 0,
            "unique_ids": 0,
            "null_pred": 0,
            "missing_required": 0,
            "missing_goal": len(goal_ids),
        }

    data = load_json_retry(path)
    if not isinstance(data, list):
        return {
            "exists": True,
            "written": 0,
            "unique_ids": 0,
            "null_pred": 0,
            "missing_required": 0,
            "missing_goal": len(goal_ids),
        }

    ids = []
    null_pred = 0
    missing_required = 0

    for r in data:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        if isinstance(rid, int):
            ids.append(rid)
        if r.get("predicted_answer") is None:
            null_pred += 1
        if not required.issubset(r.keys()):
            missing_required += 1

    id_set = set(ids)
    missing_goal = len(goal_ids - id_set)

    return {
        "exists": True,
        "written": len(data),
        "unique_ids": len(id_set),
        "null_pred": null_pred,
        "missing_required": missing_required,
        "missing_goal": missing_goal,
    }


def is_worker_running(variant: int) -> bool:
    """Check for any python process mentioning the variant's gen/fix script.

    We use WMIC because it's available on this Windows setup and avoids PowerShell quoting.
    """

    cfg = VARIANT_CFG[variant]
    needles = [cfg["gen"].name, cfg["fix"].name]
    # Also consider wrapper names if someone launched them manually.
    needles += [f"run_full_then_nullfix_v{variant}.py"]

    # Note: wmic output includes our own command line; that's fine.
    try:
        out = subprocess.check_output(
            ["cmd", "/c", "wmic process where \"name='python.exe'\" get CommandLine"],
            cwd=str(BASE),
            text=True,
            errors="ignore",
        )
    except Exception:
        return False

    low = out.lower()
    return any(n.lower() in low for n in needles)


def run_one_chunk(variant: int, chunk_size: int) -> tuple[int, int]:
    cfg = VARIANT_CFG[variant]

    env = dict(os.environ)
    env["MAX_RECORDS"] = str(chunk_size)

    # 1) generate chunk
    r1 = subprocess.run([sys.executable, str(cfg["gen"])], cwd=str(BASE), env=env)
    # 2) null-fix pass
    r2 = subprocess.run([sys.executable, str(cfg["fix"])], cwd=str(BASE))
    return r1.returncode, r2.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=int(os.environ.get("CHUNK_SIZE", "10")))
    args = ap.parse_args()

    goal_ids = dataset_goal_ids()
    goal_n = len(goal_ids)
    print(f"goal={goal_n}")

    any_incomplete = False

    for v in args.variants:
        if v not in VARIANT_CFG:
            print(f"v{v}: unknown")
            continue

        cfg = VARIANT_CFG[v]
        st = output_stats(cfg["out"], cfg["required"], goal_ids)

        complete = (
            st["exists"]
            and st["missing_goal"] == 0
            and st["null_pred"] == 0
            and st["missing_required"] == 0
        )

        print(
            f"v{v}: written={st['written']} unique_ids={st['unique_ids']} "
            f"missing_goal={st['missing_goal']} null_predicted_answer={st['null_pred']} "
            f"missing_required_fields={st['missing_required']}"
        )

        if complete:
            continue

        any_incomplete = True

        if is_worker_running(v):
            print(f"v{v}: worker_running -> skip start")
            continue

        rc1, rc2 = run_one_chunk(v, args.chunk_size)
        print(f"v{v}: ran_one_chunk chunk_size={args.chunk_size} rc_gen={rc1} rc_fix={rc2}")

    if not any_incomplete:
        print("all_complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
