#!/bin/bash

set -euo pipefail

if [[ -z "${VARIANT_NAME:-}" ]]; then
  echo "ERROR: VARIANT_NAME is not set"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

mkdir -p logs

WORKSPACE_CACHE_ROOT=${WORKSPACE_CACHE_ROOT:-${REPO_ROOT}/.cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${WORKSPACE_CACHE_ROOT}/xdg}
export HF_HOME=${HF_HOME:-${WORKSPACE_CACHE_ROOT}/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export HF_ASSETS_CACHE=${HF_ASSETS_CACHE:-${HF_HOME}/assets}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME}/modules}
export TORCH_HOME=${TORCH_HOME:-${WORKSPACE_CACHE_ROOT}/torch}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${WORKSPACE_CACHE_ROOT}/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${WORKSPACE_CACHE_ROOT}/triton}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${WORKSPACE_CACHE_ROOT}/pip}
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_ASSETS_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${TORCH_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${PIP_CACHE_DIR}"

PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/envs/py310/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/envs/py310/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

MODEL_CONFIG=${MODEL_CONFIG:-qwen2.5-7b-instruct}
# Provider defaults to the value declared in configs/models.yaml for the selected model.
# Set RUN_PROVIDER explicitly only when you need to override the model config.
RUN_PROVIDER=${RUN_PROVIDER:-}
MODEL_LABEL=${MODEL_LABEL:-}
LOCAL_MODEL_NAME=${LOCAL_MODEL_NAME:-}
ROUTER_MODEL_NAME=${ROUTER_MODEL_NAME:-}
MODEL_DEVICE=${MODEL_DEVICE:-cuda}
TEMPERATURE=${TEMPERATURE:-0.0}
TEMPERATURE_TAG=${TEMPERATURE_TAG:-$("${PYTHON_BIN}" - <<'PY'
import os
value = os.environ.get("TEMPERATURE", "0.0").strip()
print(f"temperature_{value.replace('.', '')}")
PY
)}
MAX_RECORDS=${MAX_RECORDS:-${CHUNK_SIZE:-0}}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-${RETRY_MAX:-3}}
RETRY_BASE_SLEEP_S=${RETRY_BASE_SLEEP_S:-1.5}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-120}
PROMPT_PATH=${PROMPT_PATH:-}
INPUT_PATH=${INPUT_PATH:-forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json}
SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH:-prompts/system.txt}
OUTPUT_PATH=${OUTPUT_PATH:-}
ERROR_LOG_PATH=${ERROR_LOG_PATH:-}
MODELS_CONFIG=${MODELS_CONFIG:-configs/models.yaml}
VARIANTS_CONFIG=${VARIANTS_CONFIG:-configs/variants.yaml}
AUTO_RERUN_NULLS=${AUTO_RERUN_NULLS:-1}
VERIFY_RESULTS=${VERIFY_RESULTS:-1}
FAIL_ON_VERIFY=${FAIL_ON_VERIFY:-1}
REPROCESS_NULLS=${REPROCESS_NULLS:-0}

# Resolve the effective provider from the model config when not overridden.
if [[ -z "${RUN_PROVIDER}" ]]; then
  RUN_PROVIDER=$("${PYTHON_BIN}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from analyzing_llm_rationale.config import load_model_configs
models = load_model_configs(Path("${MODELS_CONFIG}"))
print(models["${MODEL_CONFIG}"].provider)
PY
)
fi

if [[ "${RUN_PROVIDER}" == "hf-router" && -z "${HF_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}" ]]; then
  echo "ERROR: HF_TOKEN or HUGGINGFACEHUB_API_TOKEN must be set for hf-router"
  exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-unknown} - $(date)"
echo "Variant: ${VARIANT_NAME}"
echo "Provider: ${RUN_PROVIDER}"
echo "Model key: ${MODEL_CONFIG}"
echo "Cache root: ${WORKSPACE_CACHE_ROOT}"
if [[ -n "${MODEL_LABEL}" ]]; then
  echo "Model label override: ${MODEL_LABEL}"
fi
echo "Temperature: ${TEMPERATURE} (${TEMPERATURE_TAG})"
if [[ -n "${OUTPUT_PATH}" ]]; then
  echo "Output override: ${OUTPUT_PATH}"
fi

if [[ "${RUN_PROVIDER}" == "local-qwen" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  fi
  "${PYTHON_BIN}" - <<'PY'
try:
    import torch
    print(f"CUDA: {torch.cuda.is_available()}")
except Exception as exc:
    print(f"CUDA check failed: {exc}")
PY
fi

append_arg() {
  local -n args_ref="$1"
  local flag="$2"
  local value="${3:-}"
  if [[ -n "${value}" ]]; then
    args_ref+=("${flag}" "${value}")
  fi
}

build_run_args() {
  local -n args_ref="$1"
  args_ref=(
    --variant "${VARIANT_NAME}"
    --model "${MODEL_CONFIG}"
    --temperature "${TEMPERATURE}"
    --temperature-tag "${TEMPERATURE_TAG}"
    --max-tokens "${MAX_TOKENS}"
    --max-records "${MAX_RECORDS}"
    --max-attempts "${MAX_ATTEMPTS}"
    --retry-base-sleep-s "${RETRY_BASE_SLEEP_S}"
    --request-timeout-s "${REQUEST_TIMEOUT_S}"
    --device "${MODEL_DEVICE}"
    --input-path "${INPUT_PATH}"
    --system-prompt-path "${SYSTEM_PROMPT_PATH}"
    --variants-config "${VARIANTS_CONFIG}"
    --models-config "${MODELS_CONFIG}"
  )

  append_arg args_ref --provider "${RUN_PROVIDER}"
  append_arg args_ref --model-label "${MODEL_LABEL}"
  append_arg args_ref --local-model-name "${LOCAL_MODEL_NAME}"
  append_arg args_ref --router-model-name "${ROUTER_MODEL_NAME}"
  append_arg args_ref --user-prompt-path "${PROMPT_PATH}"
  append_arg args_ref --output-path "${OUTPUT_PATH}"
  append_arg args_ref --error-log-path "${ERROR_LOG_PATH}"

  if [[ "${DROP_ARTICLE_TEXT:-0}" == "1" ]]; then
    args_ref+=(--drop-article-text)
  fi
}

build_verify_args() {
  local -n args_ref="$1"
  args_ref=(
    --variant "${VARIANT_NAME}"
    --model "${MODEL_CONFIG}"
    --temperature "${TEMPERATURE}"
    --temperature-tag "${TEMPERATURE_TAG}"
    --input-path "${INPUT_PATH}"
    --variants-config "${VARIANTS_CONFIG}"
    --models-config "${MODELS_CONFIG}"
  )

  append_arg args_ref --model-label "${MODEL_LABEL}"
  append_arg args_ref --output-path "${OUTPUT_PATH}"
}

resolved_output_path() {
  env \
    RUN_PROVIDER="${RUN_PROVIDER}" \
    VARIANT_NAME="${VARIANT_NAME}" \
    MODEL_CONFIG="${MODEL_CONFIG}" \
    VARIANTS_CONFIG="${VARIANTS_CONFIG}" \
    MODELS_CONFIG="${MODELS_CONFIG}" \
    INPUT_PATH="${INPUT_PATH}" \
    SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH}" \
    PROMPT_PATH="${PROMPT_PATH}" \
    OUTPUT_PATH="${OUTPUT_PATH}" \
    ERROR_LOG_PATH="${ERROR_LOG_PATH}" \
    TEMPERATURE="${TEMPERATURE}" \
    TEMPERATURE_TAG="${TEMPERATURE_TAG}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    MAX_RECORDS="${MAX_RECORDS}" \
    MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
    RETRY_BASE_SLEEP_S="${RETRY_BASE_SLEEP_S}" \
    DROP_ARTICLE_TEXT="${DROP_ARTICLE_TEXT:-0}" \
    MODEL_LABEL="${MODEL_LABEL}" \
    LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME}" \
    ROUTER_MODEL_NAME="${ROUTER_MODEL_NAME}" \
    MODEL_DEVICE="${MODEL_DEVICE}" \
    REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "src"))

from analyzing_llm_rationale.cli import resolve_run_config

def as_int(value: str) -> int:
    return int(value) if value else 0

ns = SimpleNamespace(
    provider=os.environ["RUN_PROVIDER"],
    variant=os.environ["VARIANT_NAME"],
    model=os.environ["MODEL_CONFIG"],
    variants_config=Path(os.environ["VARIANTS_CONFIG"]),
    models_config=Path(os.environ["MODELS_CONFIG"]),
    input_path=Path(os.environ["INPUT_PATH"]),
    system_prompt_path=Path(os.environ["SYSTEM_PROMPT_PATH"]),
    user_prompt_path=Path(os.environ["PROMPT_PATH"]) if os.environ.get("PROMPT_PATH") else None,
    output_path=Path(os.environ["OUTPUT_PATH"]) if os.environ.get("OUTPUT_PATH") else None,
    error_log_path=Path(os.environ["ERROR_LOG_PATH"]) if os.environ.get("ERROR_LOG_PATH") else None,
    output_fields=None,
    temperature=float(os.environ["TEMPERATURE"]),
    temperature_tag=os.environ.get("TEMPERATURE_TAG"),
    max_tokens=as_int(os.environ["MAX_TOKENS"]),
    max_records=as_int(os.environ["MAX_RECORDS"]),
    max_attempts=as_int(os.environ["MAX_ATTEMPTS"]),
    retry_base_sleep_s=float(os.environ["RETRY_BASE_SLEEP_S"]),
    reprocess_nulls=False,
    drop_article_text=os.environ.get("DROP_ARTICLE_TEXT", "0") == "1",
    model_label=os.environ.get("MODEL_LABEL") or None,
    local_model_name=os.environ.get("LOCAL_MODEL_NAME") or None,
    router_model_name=os.environ.get("ROUTER_MODEL_NAME") or None,
    device=os.environ["MODEL_DEVICE"],
    request_timeout_s=float(os.environ["REQUEST_TIMEOUT_S"]),
)
print(resolve_run_config(ns).output_path)
PY
}

count_null_predictions() {
  local output_path="$1"
  "${PYTHON_BIN}" - "$output_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0")
    raise SystemExit(0)

payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, list):
    print("0")
    raise SystemExit(0)

print(sum(1 for row in payload if isinstance(row, dict) and row.get("predicted_answer") is None))
PY
}

run_phase() {
  local label="$1"
  shift
  echo "=== ${label} ==="
  "${PYTHON_BIN}" scripts/run_variant.py "$@"
}

RUN_ARGS=()
build_run_args RUN_ARGS

if [[ "${REPROCESS_NULLS}" == "1" ]]; then
  run_phase "null-only rerun" "${RUN_ARGS[@]}" --reprocess-nulls
else
  run_phase "full run" "${RUN_ARGS[@]}"

  if [[ "${AUTO_RERUN_NULLS}" == "1" && "${MAX_RECORDS}" == "0" ]]; then
    OUTPUT_PATH_RESOLVED=$(resolved_output_path)
    NULL_COUNT=$(count_null_predictions "${OUTPUT_PATH_RESOLVED}")
    echo "Null predictions after full run: ${NULL_COUNT}"
    if [[ "${NULL_COUNT}" != "0" ]]; then
      run_phase "null-only rerun" "${RUN_ARGS[@]}" --reprocess-nulls
    fi
  fi
fi

if [[ "${VERIFY_RESULTS}" == "1" && "${MAX_RECORDS}" == "0" ]]; then
  VERIFY_ARGS=()
  build_verify_args VERIFY_ARGS
  echo "=== verification ==="
  if "${PYTHON_BIN}" scripts/verify_results.py "${VERIFY_ARGS[@]}"; then
    echo "Verification passed."
  else
    echo "Verification failed."
    if [[ "${FAIL_ON_VERIFY}" == "1" ]]; then
      exit 1
    fi
  fi
fi
