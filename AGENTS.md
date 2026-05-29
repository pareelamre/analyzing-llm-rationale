# AGENTS.md — Codex agent setup guide

## Repository overview

Batch inference system for evaluating LLM reasoning on binary forecasting questions (Metaculus dataset). The pipeline runs 9 prompt variants across multiple models, stores results as JSON, and exposes a FastAPI server deployed to GCP Cloud Run and Vertex AI.

## Environment setup

```bash
# Install core + serving + pipeline dependencies
pip install -e ".[dev,serve,pipeline]"

# Required environment variables
export SCADS_AI_API_KEY=<key>       # SCADS AI — used by all hosted models
export NEWSAPI_KEY=<key>            # Optional — improves news fetch quality
```

If `pip install -e .` fails (editable install issue with system Python), sync source files manually:
```bash
cp src/analyzing_llm_rationale/*.py \
   /home/paam844f/.local/lib/python3.9/site-packages/analyzing_llm_rationale/
```

Or prefix commands with `PYTHONPATH=src`.

## Running tests and lint

```bash
python -m unittest discover -s tests   # unit tests
ruff check src tests                   # lint (E501 is ignored)
```

Always run both before committing.

## Key CLI commands

```bash
# Batch inference
analyze-llm-rationale run-batch \
  --variant variant0_neutral_baseline \
  --model gpt-oss-120b \
  --temperature 0.0 --temperature-tag temperature_00

# Start API server locally
analyze-llm-rationale serve \
  --model gpt-oss-120b \
  --variant variant0_neutral_baseline

# Fetch + rank news for a question (LangChain pipeline)
PYTHONPATH=src analyze-llm-rationale fetch-and-rank \
  --question "Will X happen by date Y?"

# DuckDB analytics — ingest all results and run 10 SQL queries
python scripts/sql_analytics.py --ingest

# Prefect pipeline — fetch news, run inference, store in DuckDB
python flows/forecasting_flow.py --question-id 124
```

## Project structure

```
src/analyzing_llm_rationale/
  cli.py            # CLI entrypoints (run-batch, serve, fetch-and-rank, ...)
  pipeline.py       # Core batch inference loop
  providers.py      # LLM provider abstractions (OpenAICompatible, LocalQwen, HFRouter)
  server.py         # FastAPI — /health, /predict, /vertex-predict
  news_pipeline.py  # LangChain news fetcher + summarizer + ranker
  db.py             # DuckDB schema, ingestion, helpers
  config.py         # YAML config loaders
  metrics.py        # Accuracy, Brier score, ECE

configs/
  models.yaml       # Model definitions (provider, endpoint, API key env var)
  variants.yaml     # Prompt variant definitions

prompts/
  system.txt
  user_variant0_neutral_baseline.txt  # ... one per variant

flows/
  forecasting_flow.py   # Prefect flow (fetch → rank → infer → store)

scripts/
  sql_analytics.py      # 10 DuckDB SQL queries on forecasting results

results/<model>/<temperature>/
  results_variant*.json
  errors_variant*.jsonl
  run_metadata_variant*.json
```

## Models

All hosted models use `openai-compatible` provider pointing to `https://llm.scads.ai/v1`, authenticated via `SCADS_AI_API_KEY`. Default for serving: `gpt-oss-120b`, variant `variant0_neutral_baseline`.

## Deployment

### Cloud Run (public, scales to zero)
```
https://analyzing-llm-rationale-hy7gvnvt4a-uc.a.run.app
```
- `GET /health` → `{"status": "ok"}`
- `POST /predict` — PredictRequest → PredictResponse

### Vertex AI endpoint (requires Bearer token)
```
https://us-central1-aiplatform.googleapis.com/v1/projects/brave-drive-471109-d9/locations/us-central1/endpoints/7325853011580813312:predict
```
- `POST` with `{"instances": [PredictRequest]}` → `{"predictions": [PredictResponse]}`

### CI/CD
Push to `main` triggers GitHub Actions:
1. `ci.yml` — lint + tests
2. `docker.yml` — build CPU image → push to GHCR + GCP Artifact Registry → deploy to Cloud Run → upload to Vertex AI + deploy to endpoint

GCP project: `brave-drive-471109-d9`, region: `us-central1`.
Required GitHub secrets: `GCP_SA_KEY`.
Required GitHub variables: `VERTEX_ENDPOINT_ID=7325853011580813312`.

## Adding a new prompt variant

1. Add entry to `configs/variants.yaml` with `name`, `prompt_path`, `output_fields`
2. Create `prompts/user_<variant_name>.txt` with `[question]` placeholder
3. Run smoke test: `analyze-llm-rationale run-batch --variant <name> --max-records 3`

## Adding a new model

1. Add entry to `configs/models.yaml` with provider, endpoint, API key env var
2. Test: `analyze-llm-rationale smoke-test --model <key>`
