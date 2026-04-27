#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_sweep_local.sh --model MODEL_CONFIG --input-path PATH --model-label LABEL [options]

Options:
  --model MODEL_CONFIG         Required model key from configs/models.yaml
  --input-path PATH            Required dataset path
  --model-label LABEL          Required results directory under results/
  --variants "0 1 ... 8"       Variant indices to run (default: 0 1 2 3 4 5 6 7 8)
  --temperatures "..."         Temperature values to run (default: 0)
  --request-timeout-s VALUE    Request timeout override
  -h, --help                   Show this help text
EOF
}

MODEL_CONFIG=""
INPUT_PATH=""
MODEL_LABEL=""
VARIANTS=(0 1 2 3 4 5 6 7 8)
TEMPERATURES=(0)
REQUEST_TIMEOUT_S=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_CONFIG="${2:-}"
      shift 2
      ;;
    --input-path)
      INPUT_PATH="${2:-}"
      shift 2
      ;;
    --model-label)
      MODEL_LABEL="${2:-}"
      shift 2
      ;;
    --variants)
      read -r -a VARIANTS <<< "${2:-}"
      shift 2
      ;;
    --temperatures)
      read -r -a TEMPERATURES <<< "${2:-}"
      shift 2
      ;;
    --request-timeout-s)
      REQUEST_TIMEOUT_S="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_CONFIG}" || -z "${INPUT_PATH}" || -z "${MODEL_LABEL}" ]]; then
  usage >&2
  exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/envs/py310/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/envs/py310/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

temperature_tag() {
  "${PYTHON_BIN}" - <<PY
value = float("${1}")
normalized = f"{value:.3f}".rstrip("0").rstrip(".")
if not normalized:
    normalized = "0"
print(f"temperature_{normalized.replace('.', '')}")
PY
}

variant_name() {
  case "$1" in
    0) echo "variant0_neutral_baseline" ;;
    1) echo "variant1_predicted_event" ;;
    2) echo "variant2_key_attribute" ;;
    3) echo "variant3_reasoning_type" ;;
    4) echo "variant4_credibility" ;;
    5) echo "variant5_key_conditions" ;;
    6) echo "variant6_step_by_step_reasoning" ;;
    7) echo "variant7_uncertainty_language" ;;
    8) echo "variant8_temporal_anchors" ;;
    *)
      echo "Unsupported variant index: $1" >&2
      exit 1
      ;;
  esac
}

echo "Local sweep start: $(date)"
echo "Model: ${MODEL_CONFIG}"
echo "Input: ${INPUT_PATH}"
echo "Model label: ${MODEL_LABEL}"

for temperature in "${TEMPERATURES[@]}"; do
  temp_tag=$(temperature_tag "${temperature}")
  for variant in "${VARIANTS[@]}"; do
    variant_full=$(variant_name "${variant}")
    cmd=(
      "${PYTHON_BIN}" "scripts/run_variant.py"
      "--variant" "${variant_full}"
      "--model" "${MODEL_CONFIG}"
      "--temperature" "${temperature}"
      "--temperature-tag" "${temp_tag}"
      "--input-path" "${INPUT_PATH}"
      "--model-label" "${MODEL_LABEL}"
    )
    if [[ -n "${REQUEST_TIMEOUT_S}" ]]; then
      cmd+=("--request-timeout-s" "${REQUEST_TIMEOUT_S}")
    fi

    echo "=== $(date) ${variant_full} ${temp_tag} ==="
    "${cmd[@]}"
  done
done

echo "Local sweep done: $(date)"
