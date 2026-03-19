# SLURM Layout

This directory contains SLURM batch launchers for cluster runs.

Variant launchers:

- `variant0.sh`
- `variant1.sh`
- `variant2.sh`
- `variant3.sh`
- `variant4.sh`
- `variant5.sh`
- `variant6.sh`
- `variant7.sh`
- `variant8.sh`

Shared helper:

- `run_variant_common.sh`: resolves provider/model/temperature/output paths and calls the packaged CLI

The variant launchers default to `RUN_PROVIDER=local-qwen`, but can also run against Hugging Face Router with `RUN_PROVIDER=hf-router`.

Useful overrides:

- `MODEL_CONFIG=...`: model key from `configs/models.yaml`
- `MODEL_LABEL=...`: result directory name under `results/`
- `LOCAL_MODEL_NAME=...`: Hugging Face repo for local inference
- `ROUTER_MODEL_NAME=...`: router model identifier
- `TEMPERATURE=...`: generation temperature
- `TEMPERATURE_TAG=...`: override output directory tag, for example `temperature_070`
- `MAX_RECORDS=...`: limit records for a partial run
- `REPROCESS_NULLS=1`: rerun only rows with null predictions
- `DROP_ARTICLE_TEXT=1`: trim raw article text from prompts
