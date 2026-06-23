#!/bin/bash
#SBATCH --job-name=stock_sweep
#SBATCH --partition=capella
#SBATCH --array=0-767%80        # 768 param combinations; %80 = 80 concurrent
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=results/stock_sweep/slurm_%A_%a.out
#SBATCH --error=results/stock_sweep/slurm_%A_%a.err

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load Anaconda if available
module load Anaconda3 2>/dev/null || true

PYTHON="${PYTHON:-python3}"

# Install deps silently if missing
"$PYTHON" -c "import yfinance" 2>/dev/null || pip install yfinance --quiet

"$PYTHON" scripts/stock_sweep.py --job-id "$SLURM_ARRAY_TASK_ID"
