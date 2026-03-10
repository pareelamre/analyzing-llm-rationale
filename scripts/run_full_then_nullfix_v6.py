import json
import os
import subprocess
import sys
import time
from pathlib import Path

base = Path(r"C:\Users\paree\Documents\Analyzing Rationale of LLMs")
python = sys.executable

full_script = base / "scripts" / "run_qwen_all_minimal_v6.py"
nullfix_script = base / "scripts" / "rerun_null_ids_drop_text_v6.py"
output_path = base / "results" / "Qwen2.5-7b-instruct" / "temperature_00" / "results_variant6_chain_of_thought.json"
input_path = base / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"

# Chunk size to reduce random network/timeout failures.
CHUNK = int(os.environ.get("CHUNK_SIZE", os.environ.get("MAX_RECORDS", "10")) or "10")
SLEEP_OK = float(os.environ.get("SLEEP_OK_S", "1.0"))
SLEEP_ERR = float(os.environ.get("SLEEP_ERR_S", "5.0"))


def expected_total() -> int:
    try:
        data = json.load(input_path.open("r", encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def stats():
    """Return (total_written, null_predicted_answer)."""
    if not output_path.exists():
        return 0, 0
    for _ in range(3):
        try:
            data = json.load(output_path.open("r", encoding="utf-8"))
            if not isinstance(data, list):
                return 0, 0
            total = len(data)
            nulls = sum(1 for r in data if isinstance(r, dict) and r.get("predicted_answer") is None)
            return total, nulls
        except Exception:
            time.sleep(0.5)
    return 0, 0


goal = expected_total()
print(f"Goal records: {goal} (from {input_path})")

ONE_SHOT = os.environ.get("ONE_CHUNK_ONLY", "").strip().lower() in {"1", "true", "yes"}

# In ONE_SHOT mode, run exactly one chunk (full + nullfix once) and exit.
if ONE_SHOT:
    env = dict(**os.environ)
    env["MAX_RECORDS"] = str(CHUNK)

    res = subprocess.run([python, str(full_script)], cwd=str(base), env=env)
    if res.returncode != 0:
        print("One-shot: full pass failed")
        raise SystemExit(res.returncode)

    res2 = subprocess.run([python, str(nullfix_script)], cwd=str(base))
    if res2.returncode != 0:
        print("One-shot: null-fix pass failed")
        raise SystemExit(res2.returncode)

    print("One-shot: completed one chunk (full + null-fix)")
    raise SystemExit(0)

err_streak = 0
while True:
    total, nulls = stats()
    if goal and total >= goal and nulls == 0:
        break

    env = dict(**os.environ)
    env["MAX_RECORDS"] = str(CHUNK)

    res = subprocess.run([python, str(full_script)], cwd=str(base), env=env)
    if res.returncode != 0:
        err_streak += 1
        time.sleep(min(60.0, SLEEP_ERR * (2 ** min(err_streak, 4))))
        continue

    res2 = subprocess.run([python, str(nullfix_script)], cwd=str(base))
    if res2.returncode != 0:
        err_streak += 1
        time.sleep(min(60.0, SLEEP_ERR * (2 ** min(err_streak, 4))))
        continue

    err_streak = 0
    time.sleep(SLEEP_OK)

print("Completed: all ids processed with no nulls")
