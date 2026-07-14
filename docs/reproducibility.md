# Reproducibility and Provenance

This document records the artifacts needed to reproduce the forecasting and rationale-quality results. The executable source of truth is the repository state plus the run artifacts under `results/` and `analysis/`.

## Core Data

- Forecasting dataset: `forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json`
- Each record contains the question, description, resolution criteria, categories, answer, and the retrieved news evidence used by the prompting pipeline.
- Results are written under `results/<model_label>/<temperature_tag>/`.
- Metrics are generated from stored result files, not from live model calls.

## Model Registry

`configs/models.yaml` is the model registry. It records the provider, result label, model identifier, endpoint, and access mode.

| Role | Model key | Result label | Provider/source | Model identifier |
|---|---|---|---|---|
| Forecaster | `qwen2.5-7b-instruct` | `Qwen2.5-7b-instruct` | Local Hugging Face checkpoint | `Qwen/Qwen2.5-7B-Instruct` |
| Forecaster | `qwen3-32b` | `Qwen3-32B` | Local Hugging Face checkpoint | `Qwen/Qwen3-32B` |
| Forecaster | `gpt-oss-120b` | `GPT-OSS-120B` | SCADS OpenAI-compatible endpoint | `openai/gpt-oss-120b` |
| Judge | `gemma-4-31b-it` | `Gemma-4-31B-it` | SCADS OpenAI-compatible endpoint | `google/gemma-4-31B-it` |
| Judge | `kimi-k2.5` | `Kimi-K2.5` | SCADS OpenAI-compatible endpoint | `moonshotai/Kimi-K2.5` |

Other configured models are also listed in `configs/models.yaml`; only the models above are central to the current thesis analyses.

## Inference Provenance

Every batch run writes `run_metadata_<variant>.json` beside its result file. This metadata records:

- `model_key`, `model_label`, `provider`, and `model_identifier`
- prompt variant and output fields
- temperature and temperature directory
- maximum output tokens
- retry settings
- evidence mode, including `drop_article_text` and `omit_evidence`
- dataset path, output path, error-log path, system-prompt path, and user-prompt path
- SHA-256 hashes of the system and user prompt templates
- `written_at`, the inference completion timestamp
- processed count, total stored results, and null prediction count

Use these metadata files as the primary record of inference date and run configuration.

## Decoding Parameters

Batch inference is implemented in `src/analyzing_llm_rationale/pipeline.py` and `src/analyzing_llm_rationale/providers.py`.

- CLI default maximum output length: `2048` tokens unless overridden.
- CLI default retry count: `3` attempts unless overridden.
- OpenAI-compatible hosted models receive `model`, `messages`, `max_tokens`, and `temperature`.
- GPT-5-family models use provider-default temperature handling when required by the provider wrapper.
- Local Qwen models use Hugging Face `AutoTokenizer` and `AutoModelForCausalLM`.
- Local Qwen chat formatting uses `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- Qwen3 sets `enable_thinking=False` when the installed tokenizer supports it.
- Local Qwen uses greedy decoding when `temperature == 0.0`.
- Local Qwen uses sampling with the configured temperature when `temperature > 0.0`.
- No explicit `top_p`, `top_k`, or quantization setting is applied by the batch pipeline beyond the model's generation config.

## Chat Templates and Quantization

For local Qwen runs, the chat template is the checkpoint tokenizer's own Hugging Face chat template at inference time. For hosted SCADS models, the server-side chat template is provider-managed and is not visible to this repository.

Local Qwen model loading uses:

- CUDA when available, otherwise CPU fallback
- `torch.float16` on CUDA by default
- `torch.float32` on CPU by default
- `device_map="auto"` on CUDA
- no explicit 4-bit or 8-bit quantization

For hosted models, quantization, serving stack, and exact server-side checkpoint revision are provider-managed and should be reported as unavailable unless the provider supplies them separately.

## Prompts

Prompt definitions are versioned files:

- System prompt: `prompts/system.txt`
- User prompt variants: `prompts/variant*.txt`
- Variant registry: `configs/variants.yaml`

The metadata file stores SHA-256 hashes of the exact system and user prompt templates used by a run.

## Raw Outputs and Parsing

Successful batch outputs are stored as parsed JSON result rows in:

- `results/<model_label>/<temperature_tag>/results_<variant>.json`

Malformed or failed outputs are logged in:

- `results/<model_label>/<temperature_tag>/errors_<variant>.jsonl`

The parser is implemented in `src/analyzing_llm_rationale/pipeline.py`, mainly `parse_model_response`, `_parse_json_dict`, and normalization helpers. Current successful runs do not persist a separate verbatim provider-response field; because prompts require strict JSON, the parsed result file is the main successful-output artifact. For maximal reproducibility in future runs, also store the verbatim provider text before parsing.

## Metric and Analysis Scripts

- Forecast metrics: `scripts/evaluate_metrics.py`
- Metric implementations: `src/analyzing_llm_rationale/metrics.py`
- Simple non-LLM baselines: `scripts/evaluate_simple_baselines.py`
- SHAP/rationale-feature analysis: `scripts/run_shap_analysis.py`
- Rationale judge evaluation: `scripts/evaluate_rationales_with_llm_judges.py`
- Human-annotation sheet construction: `scripts/build_human_annotation_sheet.py`
- Human-annotation scoring: `scripts/score_human_annotation.py`

The main metrics file is `analysis/metrics_by_model_temperature_variant.csv`. Simple baseline outputs are `analysis/simple_baselines.csv` and `analysis/simple_baselines.md`.

## LLM Judge Provenance

Rationale quality judging is implemented in `scripts/evaluate_rationales_with_llm_judges.py`.

- Judge models: `gemma-4-31b-it` and `kimi-k2.5`
- Judge identifiers are resolved through `configs/models.yaml`.
- Gemma judge temperature: `0.0`
- Kimi judge temperature: `1.0`
- Default judge max tokens: `5000` for Gemma, `20000` for Kimi
- Default judge timeouts: `180` seconds for Gemma, `600` seconds for Kimi
- Batch size: `8`
- Maximum workers: `16`
- Maximum retries: `5`
- Judge system prompt: embedded as `SYSTEM_PROMPT` in `scripts/evaluate_rationales_with_llm_judges.py`

Judge outputs are stored under `analysis/llm_judge_rationale_eval*`.

## Known Reproducibility Limits

- Local Hugging Face checkpoints are identified by repository name, but no explicit commit revision is currently pinned in `configs/models.yaml`.
- Hosted SCADS models are identified by model string and endpoint, but the provider-managed exact serving checkpoint is not recorded by this repository.
- Successful provider text is parsed and stored as JSON; separate verbatim successful responses are not currently preserved.
- Metaculus community-prediction baselines depend on API access permissions and are not yet part of the completed result set.

Before final thesis submission, record any available Hugging Face snapshot revisions and any provider-supplied hosted model version metadata alongside the run artifacts.

## Regeneration Checklist

1. Confirm the repository commit and preserve `configs/models.yaml`, `configs/variants.yaml`, and all `prompts/` files.
2. Run or verify the relevant batch inference jobs.
3. Keep each `results_<variant>.json`, `errors_<variant>.jsonl`, and `run_metadata_<variant>.json` together.
4. Regenerate metrics with `PYTHONPATH=src python scripts/evaluate_metrics.py`.
5. Regenerate simple baselines with `PYTHONPATH=src python scripts/evaluate_simple_baselines.py`.
6. Regenerate judge analyses only if rationale outputs or judge settings changed.
7. Report conditional metrics and coverage-aware metrics together when malformed or missing outputs are possible.
