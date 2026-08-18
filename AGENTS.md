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

Use Python 3.10+ for local development. If `pip install -e .` fails (editable install issue with system Python), sync source files manually:
```bash
cp src/analyzing_llm_rationale/*.py \
   /home/paam844f/.local/lib/python3.10/site-packages/analyzing_llm_rationale/
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

# Start API server locally (Note: Port 8000 is reserved, run on 8080 instead)
PYTHONPATH=src python -m uvicorn analyzing_llm_rationale.server:app --port 8080

# Fetch + rank news for a question (LangChain pipeline)
PYTHONPATH=src analyze-llm-rationale fetch-and-rank \
  --question "Will X happen by date Y?"

# DuckDB analytics — ingest all results and run 10 SQL queries
python scripts/sql_analytics.py --ingest

# Prefect pipeline — fetch news, run inference, store in DuckDB
python flows/forecasting_flow.py --question-id 124
```

## SLURM submission rule

Submit SLURM jobs only from the `/data/horse/ws/...` workspace, not from `/home`.
Use `--chdir` or run `sbatch` with the working directory set to the project path
under `/data/horse/ws` so logs, outputs, and temporary files stay off the home
quota.

## Project structure

```
src/analyzing_llm_rationale/
  cli.py            # CLI entrypoints (run-batch, serve, fetch-and-rank, ...)
  pipeline.py       # Core batch inference loop
  providers.py      # LLM provider abstractions (OpenAICompatible, LocalQwen, HFRouter)
  server.py         # FastAPI — /health, /predict, /vertex-predict
  mcp_server.py     # Model Context Protocol (FastMCP) server
  market_data.py    # Polymarket & Kalshi market data, orderbooks, and stats
  agent_capabilities.py # ReAct tool loop and action parsers
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
https://foresea.ink
```
- `GET /health` → `{"status": "ok"}`
- `POST /predict` — PredictRequest → PredictResponse
- `GET /mcp/` — Model Context Protocol Streamable-HTTP endpoint

## Foresea runtime notes

- The homepage "Market desk" uses `GET /radar`, which is built from
  `static/track_record_live.json` / `edge_board` and surfaces model-vs-market gaps.
- Product analytics are separate from page visits: `POST /analytics/event`,
  `GET /analytics/events/summary`. Use these for funnel events such as
  `forecast_completed`, `watchlist_add`, `share_created`, and `digest_sent`.
- Anonymous chats are stored only in browser `localStorage`; signed-in users sync
  conversations through `/chat/conversations`. Watchlist/favorites require sign-in.
- Track buttons write `FavoriteMarket` entities. The daily digest is
  `.github/workflows/favorites-digest.yml` running `scripts/favorites_digest.py`.
- Forecast sharing is explicit only: `POST /forecasts/share` creates a public
  `GET /forecast/{share_id}` page. Do not expose full private chat history.

### Agent Tools & MCP Protocols
Foresea provides a 19-tool ReAct execution loop for autonomous agents and mounts a public Model Context Protocol server at `/mcp` (`https://foresea.ink/mcp/`):
- **Forecasting & Research**: `forecast`, `get_market`, `scan_markets`, `batch_quotes`, `search_evidence`, `web_search`, `track_record`, `edge_board`, `market_leaderboard`
- **Exchange & Venue Data**: `exchange_status`, `orderbook`, `market_tags`, `price_history`, `live_data`, `polymarket_meta`, `recent_trades`
- **Trading & Execution**: `place_trade` (IOC shadow paper execution), `manage_notes`, `fetch_api`
- **Aliases**: `TOOL_ALIASES` in `agent_capabilities.py` automatically normalizes common LLM calling conventions (e.g. `http_get`, `candlesticks`, `comments`, `sports`, `series`, `game_stats`, `trades`, `leaderboard`).

### CI/CD
Push to `main` triggers GitHub Actions:
1. `ci.yml` — lint + tests
2. `docker.yml` — build CPU image → push to GHCR + GCP Artifact Registry → deploy to Cloud Run

Required GitHub secrets: `GCP_SA_KEY`.

### Codex GitHub helper skill
Use the GitHub publish skill for commit/push/PR flows when available:
```
/home/h3/paam844f/.codex/plugins/cache/openai-curated-remote/github/0.1.5/skills/yeet/SKILL.md
```

## Adding a new prompt variant

1. Add entry to `configs/variants.yaml` with `name`, `prompt_path`, `output_fields`
2. Create `prompts/user_<variant_name>.txt` with `[question]` placeholder
3. Run smoke test: `analyze-llm-rationale run-batch --variant <name> --max-records 3`

## Adding a new model

1. Add entry to `configs/models.yaml` with provider, endpoint, API key env var
2. Test: `analyze-llm-rationale smoke-test --model <key>`

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. On Windows, use `py -m graphify ...` if `graphify` is not on PATH. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` (or `py -m graphify update .`) to keep the graph current (AST-only, no API cost).
