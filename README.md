# Analyzing LLM Rationale

Productionized batch inference tooling for the repo's forecasting rationale experiments.

## What changed

- A real Python package now owns the batch-processing logic.
- Local Qwen inference and Hugging Face Router inference share one retry/resume pipeline.
- Variant/model/temperature resolution is config-driven through `configs/variants.yaml` and `configs/models.yaml`.
- The `scripts/` directory has been reduced to a single supported modular runner.
- The repo has install metadata, test coverage for the core pipeline, lint configuration, and CI.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .[dev]
```

## Primary entrypoint

Run the variant 3 pipeline with the packaged CLI:

```bash
analyze-llm-rationale run-batch --variant variant3_reasoning_type
```

For a remote OpenAI-compatible provider:

```bash
export SCADS_AI_API_KEY=your_token
analyze-llm-rationale run-batch --variant variant3_reasoning_type --model llama-3.3-70b-instruct
```

If you do not want to install the package into the environment, you can invoke it directly with `PYTHONPATH=src`.

Useful options:

- `--variant variant6_step_by_step_reasoning`: choose the prompt/output contract
- `--model qwen2.5-7b-instruct`: choose a configured model definition
- `--temperature 0.7 --temperature-tag temperature_07`: control generation temperature and output directory
- `--max-records 10`: process only a bounded number of records.
- `--reprocess-nulls`: rerun existing rows with `predicted_answer = null`.
- `--drop-article-text`: remove raw article text from prompts before inference.
- `--device auto`: select `cuda` when available, otherwise `cpu`.
- `verify-results --variant ...`: verify completeness, duplicates, malformed rows, and missing IDs
- `validate-dataset`: validate the dataset schema before a run

## Scripts

Supported modular runner:

- `python scripts/run_variant.py --variant variant5_key_conditions`
- `python scripts/run_variant.py --variant variant3_reasoning_type --temperature 0.7 --temperature-tag temperature_07`
- `python scripts/run_variant.py --variant variant4_credibility --model llama-3.3-70b-instruct`
- `python scripts/verify_results.py --variant variant3_reasoning_type`
- `python download_qwen_model.py`
- `python test_local_inference.py`

Repo layout:

- `scripts/`: modular runner entrypoint
- `slurm/`: batch launchers

Auditability:

- Each run writes `run_metadata_<variant>.json` next to the results file.
- Metadata includes provider, model key, resolved model identifier, temperature, output fields, and prompt SHA-256 hashes.
- Existing malformed results JSON now fails fast instead of being silently ignored.

## Quality checks

```bash
python -m unittest discover -s tests
ruff check src tests
```

## Current scope

This refactor now covers variant/model/temperature selection through config-backed paths. Schema validation and broader integration coverage are still the next production gaps.
