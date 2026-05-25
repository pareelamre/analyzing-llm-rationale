# Artifact Manifest

This repository is prepared as a conference artifact for the LLM rationale
forecasting experiments.

## Include in Submission

- Source package: `src/analyzing_llm_rationale/`
- Configuration: `configs/models.yaml`, `configs/variants.yaml`
- Prompts: `prompts/*.txt`
- Dataset snapshot: `forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json`
- Evaluation scripts: `scripts/evaluate_metrics.py`, `scripts/evaluate_rationale_quality.py`,
  `scripts/evaluate_rationales_with_llm_judges.py`, `scripts/run_shap_analysis.py`
- Plot scripts: `scripts/plot_*.py`
- Results and summaries: `results/`, `analysis/`
- Paper assets: `paper/*.png`, `paper/*.pdf`, `paper/*.drawio`, `paper/*.md`
- HPC instructions: `HPC_SETUP.md`, `SLURM_GUIDE.md`, `slurm/`
- Tests: `tests/`

## Exclude from Submission

- Local model and package caches: `.cache/`
- Local environments: `.venv/`, `envs/`
- Runtime logs: `logs/`, `analysis/judge_logs/`
- API keys and tokens: `SCADS_AI_API_KEY.txt`, `DEEPSEEK_API_KEY.txt`,
  `HF_TOKEN.txt`, `OPEN_AI_API_KEY.txt`
- Build outputs: `build/`, `dist/`, `*.egg-info/`

## Reproducibility Checklist

- Install with `python -m pip install -e ".[dev,analysis]"`.
- Validate the dataset with `PYTHONPATH=src python -m analyzing_llm_rationale validate-dataset`.
- Run unit tests with `python -m unittest discover -s tests`.
- Validate selected results with `PYTHONPATH=src python -m analyzing_llm_rationale verify-results`.
  Pass `--temperature-tag` for historical output directories such as `temperature_000`.
- Regenerate aggregate metrics with `python scripts/evaluate_metrics.py`.
- Regenerate paper figures with the `scripts/plot_*.py` scripts listed in `README.md`.

## Notes

Generated model outputs are included for auditability. Full model checkpoints are
not included; use `PYTHONPATH=src python -m analyzing_llm_rationale download-model`
or `python download_qwen_model.py` to populate the local Hugging Face cache.
