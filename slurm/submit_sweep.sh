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
  --max-concurrent N           Maximum concurrently running jobs in this sweep (default: 6)
  --variants "0 1 ... 8"       Variant indices to submit (default: 0 1 2 3 4 5 6 7 8)
  --temperatures "..."         Temperature values to submit (default: 0 0.25 0.75 1.25 1.75 2)
  --dry-run                    Print planned submissions without calling sbatch

Notes:
  - Jobs are submitted in dependency lanes to cap active concurrency.
  - Additional environment overrides such as REQUEST_TIMEOUT_S, MAX_ATTEMPTS,
    RETRY_BASE_SLEEP_S, DROP_ARTICLE_TEXT, and REPROCESS_NULLS are forwarded
    through sbatch --export=ALL.
EOF
}

MODEL_CONFIG=""
JOB_PREFIX=""
MAX_CONCURRENT=6
VARIANTS=(0 1 2 3 4 5 6 7 8)
TEMPERATURES=(0 0.25 0.75 1.25 1.75 2)
DRY_RUN=0

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
    --max-concurrent)
      MAX_CONCURRENT="${2:-}"
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

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-concurrent must be a positive integer" >&2
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

declare -a lane_last_job_ids
for ((i = 0; i < MAX_CONCURRENT; i++)); do
  lane_last_job_ids[i]=""
done

submission_index=0
for variant in "${VARIANTS[@]}"; do
  for temperature in "${TEMPERATURES[@]}"; do
    lane=$((submission_index % MAX_CONCURRENT))
    dependency_job_id="${lane_last_job_ids[$lane]}"
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
    if [[ -n "${dependency_job_id}" ]]; then
      sbatch_args+=(--dependency="afterany:${dependency_job_id}")
    fi
    sbatch_args+=("${script_path}")

    if [[ "${DRY_RUN}" == "1" ]]; then
      job_id="DRY_RUN_${submission_index}"
      lane_last_job_ids[$lane]="${job_id}"
      dependency_label="${dependency_job_id:-}"
      printf '%s\t%s\tvariant%s\t%s\t%s\t%s\t%s\n' \
        "${job_id}" "${job_name}" "${variant}" "${temperature}" "${temp_tag}" "${dependency_label}" "${script_path}" \
        >> "${submission_log}"
    else
      job_id=$(sbatch "${sbatch_args[@]}" | awk '{print $4}')
      lane_last_job_ids[$lane]="${job_id}"
      dependency_label="${dependency_job_id:-}"
      printf '%s\t%s\tvariant%s\t%s\t%s\t%s\t%s\n' \
        "${job_id}" "${job_name}" "${variant}" "${temperature}" "${temp_tag}" "${dependency_label}" "${script_path}" \
        >> "${submission_log}"
    fi

    submission_index=$((submission_index + 1))
  done
done

printf '%s\n' "${submission_log}"
