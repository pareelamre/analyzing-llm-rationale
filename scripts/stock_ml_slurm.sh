#!/bin/bash
#SBATCH --job-name=stock_ml
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/stock_ml/slurm_%j.out
#SBATCH --error=results/stock_ml/slurm_%j.err

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

module load Anaconda3 2>/dev/null || true

mkdir -p results/stock_ml

python3 scripts/stock_ml.py \
    --holding 21 \
    --train-years 4 \
    --top-pct 0.2 \
    --cost-bps 5 \
    --out results/stock_ml
