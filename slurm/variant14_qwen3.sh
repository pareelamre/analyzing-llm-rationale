#!/bin/bash
#SBATCH --job-name=variant14_qwen3
#SBATCH --output=logs/variant14_qwen3_%j.out
#SBATCH --error=logs/variant14_qwen3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=capella
#SBATCH --account=p_scads_finetune
#SBATCH --time=48:00:00

VARIANT_NAME="variant14_temporal_credibility_combined"
MODEL_CONFIG="qwen3-32b"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurm/run_variant_common.sh"
