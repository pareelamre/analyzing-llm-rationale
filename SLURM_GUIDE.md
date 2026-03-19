# SLURM Job Guide - Variant 3 Processing

This guide shows how to run variant 3 processing on HPC using SLURM using a single launcher that can target either local GPU inference or Hugging Face Router.

## One Script, Two Provider Modes

- **Script**: `slurm/variant3.sh`
- **Default**: local GPU inference via `RUN_PROVIDER=local-qwen`
- **Alternative**: Hugging Face Router via `RUN_PROVIDER=hf-router`

---

## Using Hugging Face Router

### Setup

1. Set your HF token:
```bash
export HF_TOKEN='your_huggingface_token'
```

2. Create logs directory:
```bash
mkdir -p logs
```

3. Submit job:
```bash
RUN_PROVIDER=hf-router sbatch slurm/variant3.sh
```

### Key Features
- Temperature: **0.0** (deterministic)
- Automatic retry with exponential backoff
- Handles context length errors by dropping article text
- Processes in chunks of 10 records
- Loops until all records complete with no nulls

## Using Local GPU Inference

### Prerequisites

1. Download the model (one-time, ~15GB):
```bash
# On login node or interactive session
python download_qwen_model.py
```

This downloads to `~/.cache/huggingface/hub/` and takes 30-60 minutes.

2. Install dependencies (if not already installed):
```bash
pip install --user transformers torch huggingface_hub accelerate
```

### Submit Job

```bash
sbatch slurm/variant3.sh
```

### Key Features
- Temperature: **0.0** (deterministic)
- Runs on GPU (8-16GB VRAM recommended)
- No API costs or rate limits
- Processes records sequentially
- Automatic resume support

---

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### Watch output logs
```bash
# For Hugging Face Router
tail -f logs/variant3_JOBID.out

# For local GPU
tail -f logs/variant3_JOBID.out
```

### Check progress
```bash
python -c "
import json
from pathlib import Path
p = Path('results/Qwen2.5-7b-instruct/temperature_00/results_variant3_reasoning_type.json')
if p.exists():
    data = json.load(p.open())
    total = len(data)
    nulls = sum(1 for r in data if r.get('predicted_answer') is None)
    print(f'Progress: {total} records, {nulls} nulls, {total-nulls} complete')
"
```

### Cancel job
```bash
scancel JOBID
```

---

## Resource Requirements

| Option | GPU | CPUs | RAM | Time | Partition |
|--------|-----|------|-----|------|-----------|
| Hugging Face Router | No | 4 | 16GB | 48h | standard |
| Local GPU | Yes (1x) | 4 | 32GB | 48h | gpu |

---

## Performance Comparison

| Method | Speed/Record | Total Time (700 records) | Cost |
|--------|--------------|--------------------------|------|
| HF API | 5-10s | 1-2 hours | $$ (uses credits) |
| Local GPU | 10-15s | 2-3 hours | Free |
| Local CPU | 2-3 min | 20-30 hours | Free (very slow) |

**Recommendation**: Use **Local GPU** (Option 2) for best balance of speed and cost.

---

## Configuration

Both scripts use temperature **0.0** for deterministic outputs.

### Common Script Environment Variables
Set in `slurm/variant3.sh`:
- `CHUNK_SIZE=10` - Records per batch run
- `RETRY_MAX=6` - Max retries per record
- `RETRY_BASE_SLEEP_S=1.5` - Initial retry delay
- `REQUEST_TIMEOUT_S=120` - HTTP timeout
- `RUN_PROVIDER=local-qwen|hf-router` - Provider selection
- `MODEL_CONFIG=qwen2.5-7b-instruct` - Model key from `configs/models.yaml`
- `TEMPERATURE=0.0` - Generation temperature
- `MODEL_LABEL=Qwen2.5-7b-instruct` - Result directory segment

---

## Troubleshooting

### Hugging Face Router Issues

**"HF_TOKEN not set"**
```bash
export HF_TOKEN='your_token'
RUN_PROVIDER=hf-router sbatch slurm/variant3.sh
```

**"Out of credits"**
Switch to local GPU inference by omitting `RUN_PROVIDER=hf-router`

### Local GPU Issues

**"Model not found"**
```bash
python download_qwen_model.py
```

**"CUDA out of memory"**
- Request GPU with more VRAM: `#SBATCH --gres=gpu:a100:1`
- Or fall back to CPU (very slow)

**"ImportError: No module named transformers"**
```bash
pip install --user transformers torch huggingface_hub accelerate
```

---

## Which Mode Should I Use?

Use **Local GPU** if:
- ✓ You're out of HF API credits
- ✓ You want free, unlimited processing
- ✓ You have access to GPU nodes
- ✓ You've downloaded the model

Use **Hugging Face Router** if:
- ✓ You have HF API credits
- ✓ You don't want to download the model
- ✓ You only have CPU nodes available
- ✓ You need slightly faster per-record processing

---

## Next Steps

After variant 3 completes, use the matching scripts in `slurm/variant0.sh` through `slurm/variant8.sh`.
You can verify a results file with `python scripts/verify_results.py --variant variant3_reasoning_type`.
