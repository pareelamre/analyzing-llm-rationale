#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  slurm/submit_sweep.sh --model MODEL_CONFIG --prefix JOB_PREFIX [options]

Options:
  --model MODEL_CONFIG         Required model key from configs/models.yaml
  --prefix JOB_PREFIX          Required SLURM job-name prefix
  --variants "0 1 ... 8"       Variant indices to submit (default: 0 1 2 3 4 5 6 7 8)
  --temperatures "..."         Temperature values to submit (default: 0 0.25 0.75 1.25 1.75 2)
  --sbatch-extra ARG          Additional sbatch argument to append (repeatable)
  --dry-run                    Print planned submissions without calling sbatch

Notes:
  - Additional environment overrides such as REQUEST_TIMEOUT_S, MAX_ATTEMPTS,
    RETRY_BASE_SLEEP_S, DROP_ARTICLE_TEXT, and REPROCESS_NULLS are forwarded
    through sbatch --export=ALL.
EOF
}

MODEL_CONFIG=""
JOB_PREFIX=""
VARIANTS=(0 1 2 3 4 5 6 7 8)
TEMPERATURES=(0 0.25 0.75 1.25 1.75 2)
DRY_RUN=0
SBATCH_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_CONFIG="${2:-}"
      shift 2
      ;;
    --prefix)
      JOB_PREFIX="${2:-}"
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
    --sbatch-extra)
      SBATCH_EXTRA_ARGS+=("${2:-}")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ -z "${MODEL_CONFIG}" || -z "${JOB_PREFIX}" ]]; then
  usage >&2
  exit 1
fi

mkdir -p logs
timestamp=$(date +%Y%m%d_%H%M%S)
submission_log="logs/${JOB_PREFIX}_submit_${timestamp}.tsv"
printf 'job_id\tjob_name\tvariant\ttemperature\ttemperature_tag\tdependency\tscript\n' > "${submission_log}"

temperature_tag() {
  python - <<PY
value = float("${1}")
normalized = f"{value:.3f}".rstrip("0").rstrip(".")
if not normalized:
    normalized = "0"
print(f"temperature_{normalized.replace('.', '')}")
PY
}

temperature_suffix() {
  python - <<PY
value = float("${1}")
normalized = f"{value:.3f}".rstrip("0").rstrip(".")
if not normalized:
    normalized = "0"
print(normalized.replace('.', ''))
PY
}

for variant in "${VARIANTS[@]}"; do
  for temperature in "${TEMPERATURES[@]}"; do
    temp_tag=$(temperature_tag "${temperature}")
    temp_suffix=$(temperature_suffix "${temperature}")
    job_name="${JOB_PREFIX}_v${variant}_t${temp_suffix}"
    script_path="slurm/variant${variant}.sh"
    out_path="logs/${job_name}_%j.out"
    err_path="logs/${job_name}_%j.err"

    sbatch_args=(
      --job-name="${job_name}"
      --output="${out_path}"
      --error="${err_path}"
      --export=ALL,MODEL_CONFIG="${MODEL_CONFIG}",TEMPERATURE="${temperature}",TEMPERATURE_TAG="${temp_tag}"
    )
    if [[ "${#SBATCH_EXTRA_ARGS[@]}" -gt 0 ]]; then
      sbatch_args+=("${SBATCH_EXTRA_ARGS[@]}")
    fi
    sbatch_args+=("${script_path}")

    if [[ "${DRY_RUN}" == "1" ]]; then
      job_id="DRY_RUN_${variant}_${temp_suffix}"
      printf '%s\t%s\tvariant%s\t%s\t%s\t%s\t%s\n' \
        "${job_id}" "${job_name}" "${variant}" "${temperature}" "${temp_tag}" "" "${script_path}" \
        >> "${submission_log}"
    else
      job_id=$(sbatch "${sbatch_args[@]}" | awk '{print $4}')
      printf '%s\t%s\tvariant%s\t%s\t%s\t%s\t%s\n' \
        "${job_id}" "${job_name}" "${variant}" "${temperature}" "${temp_tag}" "" "${script_path}" \
        >> "${submission_log}"
    fi
  done
done

printf '%s\n' "${submission_log}"
