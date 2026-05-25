# Analyzing LLM Rationale

Conference artifact for studying how explicit rationale instructions affect LLM
forecasting behavior on Metaculus-style binary forecasting questions. The codebase
contains the prompt variants, batch inference runner, generated result tables, and
plotting/analysis scripts used for the paper figures.

## Repository Contents

- `src/analyzing_llm_rationale/`: packaged inference, provider, validation, and CLI logic.
- `configs/`: model and rationale-variant definitions.
- `prompts/`: system prompt and the nine rationale-variant prompts.
- `scripts/`: evaluation, recovery, SHAP, plotting, and utility scripts.
- `slurm/`: HPC launchers for the variant/temperature sweeps.
- `results/`: model outputs and run metadata.
- `analysis/`: aggregate metric tables and rationale-analysis outputs.
- `paper/`: paper figures, Draw.io sources, PDFs, and qualitative case studies.
- `tests/`: unit tests for the package and metric parsing.

See `ARTIFACT_MANIFEST.md` for the submission checklist and file-level notes.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,analysis]"
```

Use `.[dev]` for the core runner and tests only. Use `.[analysis]` when
regenerating plots or SHAP analyses.

## Quick Validation

```bash
PYTHONPATH=src python -m analyzing_llm_rationale validate-dataset
python -m unittest discover -s tests
ruff check src tests scripts/*.py
```

`PYTHONPATH=src` is useful when the repository has not been installed yet or an
older user-local install shadows the working tree.

## Primary Entry Point

Run the variant 3 pipeline with the packaged CLI:

```bash
analyze-llm-rationale run-batch --variant variant3_reasoning_type
```

For a remote OpenAI-compatible provider:

```bash
export SCADS_AI_API_KEY=your_token
analyze-llm-rationale run-batch --variant variant3_reasoning_type --model llama-3.3-70b-instruct
```

If you do not want to install the package into the environment, invoke it directly:

```bash
PYTHONPATH=src python -m analyzing_llm_rationale run-batch --variant variant3_reasoning_type
```

Useful options:

- `--variant variant6_step_by_step_reasoning`: choose the prompt/output contract.
- `--model qwen2.5-7b-instruct`: choose a configured model definition.
- `--temperature 0.7`: control generation temperature and output directory.
- `--max-records 10`: process only a bounded number of records.
- `--reprocess-nulls`: rerun existing rows with `predicted_answer = null`.
- `--drop-article-text`: remove raw article text from prompts before inference.
- `--device auto`: select `cuda` when available, otherwise `cpu`.
- `verify-results --variant ...`: verify completeness, duplicates, malformed rows, and missing IDs.
- `validate-dataset`: validate the dataset schema before a run.

## Reproducing Core Outputs

Validate an existing result file:

```bash
PYTHONPATH=src python -m analyzing_llm_rationale verify-results \
  --model qwen2.5-7b-instruct \
  --variant variant3_reasoning_type \
  --temperature 0.0 \
  --temperature-tag temperature_000
```

Regenerate aggregate metrics from `results/`:

```bash
python scripts/evaluate_metrics.py
```

Regenerate paper figures after metrics are present:

```bash
python scripts/plot_model_variant_metric_heatmap.py
python scripts/plot_variant_delta_from_v0.py
python scripts/plot_temperature_frontier.py
python scripts/plot_frs_ablation_slopegraph.py
python scripts/plot_uncertainty_language_calibration_disconnect.py
python scripts/plot_shap_importance_attribute_gaps.py
```

## Scripts

Common runner and verification commands:

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
ruff check src tests scripts/*.py
```

## Data, Models, and Secrets

The included dataset is `forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json`.
Model access is configured in `configs/models.yaml`. Open-weight Qwen models run
locally through Hugging Face; hosted models use OpenAI-compatible endpoints and
require API keys through environment variables or local key files.

Never commit key files such as `SCADS_AI_API_KEY.txt`, `DEEPSEEK_API_KEY.txt`,
`HF_TOKEN.txt`, or `OPEN_AI_API_KEY.txt`. Large local caches (`.cache/`, `envs/`,
`.venv/`) are intentionally ignored and excluded from source archives.

## Citation

If this repository supports a publication, cite the artifact with the metadata in
`CITATION.cff` and cite the upstream datasets/models according to their licenses.
