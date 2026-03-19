# Scripts Layout

The `scripts/` directory contains the supported modular entrypoint:

- `run_variant.py`: config-driven batch runner for any configured variant/model/temperature
- `verify_results.py`: result completeness and structure checker

Examples:

```bash
python scripts/run_variant.py --variant variant5_key_conditions
python scripts/run_variant.py --variant variant3_reasoning_type --temperature 0.7 --temperature-tag temperature_07
python scripts/run_variant.py --variant variant4_credibility --model llama-3.3-70b-instruct
python scripts/run_variant.py --variant variant6_step_by_step_reasoning --reprocess-nulls --drop-article-text
python scripts/verify_results.py --variant variant3_reasoning_type --temperature 0.0
```

Variant definitions live in `configs/variants.yaml`. Model definitions live in `configs/models.yaml`.
Each batch run writes `run_metadata_<variant>.json` beside the output file for provenance and prompt hashing.
