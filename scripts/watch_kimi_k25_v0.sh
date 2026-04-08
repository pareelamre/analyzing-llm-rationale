#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/horse/ws/paam844f-codabench/analyzing-llm-rationale"
LOG_DIR="$ROOT/logs"
RUN_CMD=(python -u "$ROOT/scripts/run_variant.py" --variant variant0_neutral_baseline --model kimi-k2.5 --temperature 0.0 --temperature-tag temperature_0)

mkdir -p "$LOG_DIR"

while true; do
  if ! pgrep -f "run_variant.py .*--model kimi-k2.5 .*--variant variant0_neutral_baseline" >/dev/null; then
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="$LOG_DIR/kimi_k25_v0_t0_${TS}.log"
    : > "$LOG"
    nohup env PYTHONUNBUFFERED=1 "${RUN_CMD[@]}" >> "$LOG" 2>&1 &
  fi
  sleep 60
done
