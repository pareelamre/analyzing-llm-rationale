#!/bin/bash
#SBATCH --job-name=variant12
#SBATCH --output=logs/variant12_%j.out
#SBATCH --error=logs/variant12_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=capella
#SBATCH --account=p_scads_lv_llm
#SBATCH --time=48:00:00

VARIANT_NAME="variant12_random_structure_control"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurm/run_variant_common.sh"
