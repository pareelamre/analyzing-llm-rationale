# HPC Setup Guide for Variant 3 Processing

This guide explains how to run variant 3 (reasoning type) processing on HPC using SLURM.

## Files Created

### Modular Runner
- `scripts/run_variant.py` - Config-driven runner for any variant/model/temperature

### SLURM Job Script
- `slurm/variant3.sh` - SLURM batch job for variant 3

## Quick Start

### 1. Set Your HuggingFace Token

```bash
export HF_TOKEN='your_huggingface_token_here'
```

To make it persistent, add to your `~/.bashrc`:
```bash
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Create Logs Directory

```bash
mkdir -p logs
```

### 3. Submit the Job

```bash
sbatch slurm/variant3.sh
```

To use Hugging Face Router instead of local GPU inference:
```bash
RUN_PROVIDER=hf-router sbatch slurm/variant3.sh
```

### 4. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch output log (replace JOBID with your job ID)
tail -f logs/variant3_JOBID.out

# Check error log
tail -f logs/variant3_JOBID.err
```

## How It Works

The SLURM script delegates to `scripts/run_variant.py`, which resolves:
- the variant prompt and output fields from `configs/variants.yaml`
- the model provider and names from `configs/models.yaml`
- the output directory from the configured model label and temperature tag

The packaged pipeline then:
- retries provider failures with backoff
- trims article text on context-limit failures
- resumes from existing results
- can rerun only null predictions when requested

## Configuration

Environment variables (set in `slurm/variant3.sh`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 10 | Records per batch run |
| `RETRY_MAX` | 6 | Max retries per record |
| `RETRY_BASE_SLEEP_S` | 1.5 | Base sleep for retries |
| `REQUEST_TIMEOUT_S` | 120 | HTTP timeout (seconds) |
| `RUN_PROVIDER` | `local-qwen` | `local-qwen` or `hf-router` |
| `MODEL_CONFIG` | `qwen2.5-7b-instruct` | Model key from `configs/models.yaml` |
| `TEMPERATURE` | `0.0` | Generation temperature |
| `TEMPERATURE_TAG` | auto | Output directory suffix |

## Resource Allocation

Default SLURM settings:
- **Time**: 48 hours
- **CPUs**: 4
- **Memory**: 16GB
- **Partition**: standard (adjust for your HPC)

To modify resources, edit the `#SBATCH` directives in `slurm/variant3.sh`.

## Testing Locally Before SLURM

Test with a small batch:

```bash
export HF_TOKEN='your_token'
export CHUNK_SIZE=5
python scripts/run_variant.py --variant variant3_reasoning_type --provider hf-router
```

This processes 5 records and exits.

## Monitoring & Debugging

### Check Progress
```bash
# Count total records processed
python -c "
import json
data = json.load(open('results/Qwen2.5-7b-instruct/temperature_00/results_variant3_reasoning_type.json'))
print(f'Total: {len(data)}')
print(f'Nulls: {sum(1 for r in data if r.get(\"predicted_answer\") is None)}')
"
```

### View Error Log
```bash
# See recent errors
tail -50 results/Qwen2.5-7b-instruct/temperature_00/errors_variant3_reasoning_type.jsonl
```

### Resume After Interruption
The scripts automatically resume from where they left off. Just resubmit:
```bash
sbatch slurm/variant3.sh
```

## Adapting for Other Variants

Use the matching launcher in `slurm/variant0.sh` through `slurm/variant8.sh`, or call:

```bash
python scripts/run_variant.py --variant variant5_key_conditions
```

## Troubleshooting

### "HF_TOKEN is not set" Error
Make sure you've exported your token:
```bash
export HF_TOKEN='your_token_here'
sbatch slurm/variant3.sh
```

### Job Keeps Failing
Check error log:
```bash
cat logs/variant3_JOBID.err
```

Common issues:
- Invalid HF_TOKEN
- Network connectivity problems
- Python dependencies missing (install `requests`)

### Slow Progress
- Check if the provider is rate-limiting you
- Increase `CHUNK_SIZE` for larger partial runs
- Check error log for frequent retries

### Out of Credits
If you run out of HuggingFace API credits, use local inference instead:
- Follow `LOCAL_INFERENCE_README.md`
- Add GPU resources: `#SBATCH --gres=gpu:1`

## Expected Runtime

- **Total records**: ~700 (varies by dataset)
- **Chunk size**: 10
- **Time per record**: ~5-10 seconds (with retries)
- **Estimated total**: 1-2 hours (can vary with API performance)

The job will automatically complete when all records are processed with no nulls.
