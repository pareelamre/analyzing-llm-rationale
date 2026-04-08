# SLURM Job Guide

This repository uses one SLURM launcher per prompt variant plus a shared wrapper that resolves the effective model, provider, output paths, verification, and null-only reruns.

## Current Layout

- Variant launchers: `slurm/variant0.sh` through `slurm/variant8.sh`
- Shared wrapper: `slurm/run_variant_common.sh`
- Sweep submitter: `slurm/submit_sweep.sh`

Each variant launcher only sets `VARIANT_NAME` and the SLURM resource request, then sources the shared wrapper.

## Current Resource Request

All current variant launchers request:

- `--nodes=1`
- `--ntasks=1`
- `--cpus-per-task=4`
- `--mem=32G`
- `--gres=gpu:1`
- `--partition=capella`
- `--time=48:00:00`

If you need different resources, update the relevant `slurm/variant*.sh` file.

## Provider Resolution

The effective provider is resolved from `configs/models.yaml` unless you explicitly override it with `RUN_PROVIDER`.

Examples from the current model config:

- `qwen2.5-7b-instruct` -> `local-qwen`
- `qwen3-32b` -> `local-qwen`
- `deepseek-v3` -> `openai-compatible`

This means the variant launchers do not hardcode `local-qwen`. The provider follows the chosen model config by default.

## Single-Variant Submission

Submit one variant with the model and temperature you want:

```bash
mkdir -p logs
MODEL_CONFIG=qwen3-32b TEMPERATURE=0.75 sbatch slurm/variant0.sh
```

The wrapper derives the temperature directory tag automatically:

- `0` -> `temperature_0`
- `0.25` -> `temperature_025`
- `0.75` -> `temperature_075`
- `1.25` -> `temperature_125`

Example result path for the command above:

```text
results/Qwen3-32B/temperature_075/results_variant0_neutral_baseline.json
```

## Full Sweep Submission

Use the sweep helper to submit multiple variants and temperatures:

```bash
slurm/submit_sweep.sh \
  --model qwen3-32b \
  --prefix qwen3 \
  --variants "0 1 2 3 4 5 6 7 8" \
  --temperatures "0.75"
```

The sweep script:

- submits one `sbatch` job per variant/temperature pair
- writes a submission log under `logs/<prefix>_submit_<timestamp>.tsv`
- sets `MODEL_CONFIG`, `TEMPERATURE`, and `TEMPERATURE_TAG` via `sbatch --export=ALL`
- does not add inter-job dependencies

If you need dependency chaining, you must add it yourself or modify the submit script.

## Useful Environment Overrides

These are consumed by `slurm/run_variant_common.sh` and are forwarded through `submit_sweep.sh` because it uses `--export=ALL`.

- `MODEL_CONFIG=...`: model key from `configs/models.yaml`
- `RUN_PROVIDER=...`: override the provider resolved from the model config
- `TEMPERATURE=...`: generation temperature
- `TEMPERATURE_TAG=...`: manual override for the output directory tag
- `MODEL_LABEL=...`: override the result directory name under `results/`
- `LOCAL_MODEL_NAME=...`: override the local inference model name
- `ROUTER_MODEL_NAME=...`: override the router model name
- `MAX_RECORDS=...`: limit a run to a subset of records
- `MAX_TOKENS=...`: generation length limit
- `MAX_ATTEMPTS=...`: retry count per record
- `RETRY_BASE_SLEEP_S=...`: exponential backoff base delay
- `REQUEST_TIMEOUT_S=...`: request timeout
- `DROP_ARTICLE_TEXT=1`: trim raw article text from prompts
- `REPROCESS_NULLS=1`: rerun only rows with null predictions
- `AUTO_RERUN_NULLS=0`: disable the automatic null-only rerun after a full run
- `VERIFY_RESULTS=0`: disable post-run verification
- `FAIL_ON_VERIFY=0`: do not fail the SLURM job when verification fails

## Runtime Behavior

For a full run, the wrapper currently does the following:

1. Resolves the effective provider from `configs/models.yaml` unless overridden.
2. Runs `scripts/run_variant.py`.
3. If `AUTO_RERUN_NULLS=1` and this is not a partial run, checks for null predictions and reruns null rows only when needed.
4. Runs `scripts/verify_results.py` unless verification is disabled.

## Monitoring

Check your jobs:

```bash
squeue -u "$USER"
```

Watch a job log:

```bash
tail -f logs/qwen3_v0_t075_JOBID.out
tail -f logs/qwen3_v0_t075_JOBID.err
```

The sweep submitter sets custom log names like:

- `logs/qwen3_v0_t075_<jobid>.out`
- `logs/qwen3_v0_t075_<jobid>.err`

The launcher scripts themselves contain default `#SBATCH --output/--error` lines, but `submit_sweep.sh` overrides those at submission time.

## Common Examples

Run variant 3 for Qwen3 at temperature 0.75:

```bash
MODEL_CONFIG=qwen3-32b TEMPERATURE=0.75 sbatch slurm/variant3.sh
```

Run only null predictions for an existing result:

```bash
MODEL_CONFIG=qwen3-32b TEMPERATURE=0.25 REPROCESS_NULLS=1 sbatch slurm/variant5.sh
```

Run a small partial test:

```bash
MODEL_CONFIG=qwen3-32b TEMPERATURE=0.75 MAX_RECORDS=20 sbatch slurm/variant0.sh
```

Override to a different provider explicitly:

```bash
RUN_PROVIDER=hf-router MODEL_CONFIG=qwen3-32b TEMPERATURE=0.75 sbatch slurm/variant0.sh
```

## Troubleshooting

`sbatch` hangs or `scontrol ping` times out:

- This is a cluster-side SLURM availability problem, not a repo problem.
- Check `scontrol ping` and `squeue -u "$USER"` from the login node.
- Retry submission once the controller is responsive again.

Local GPU model errors:

```bash
python download_qwen_model.py
```

Missing Python dependencies:

```bash
pip install --user transformers torch huggingface_hub accelerate
```

HF Router auth issues:

```bash
export HF_TOKEN='your_token'
export HUGGINGFACEHUB_API_TOKEN="$HF_TOKEN"
```

## Source Of Truth

When this guide and the code diverge, trust the code in:

- `slurm/variant*.sh`
- `slurm/run_variant_common.sh`
- `slurm/submit_sweep.sh`
- `configs/models.yaml`
