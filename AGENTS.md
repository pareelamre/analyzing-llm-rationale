# AGENTS.md — Codex agent setup guide

## Agent coordination log

Multiple agents work in this repo. Record decisions here so the next agent
does not redo or undo them. Newest entries first.

### 2026-10-06 — GCP cost collapse: OCI migration on hold, twin scheduler paused (Copilot)

Two changes cut the projected Cloud Run bill from ~$86/month to ~$1.50/month:

1. **PR #671** set the main service to `--min-instances 0`. Measured:
   $2.08/day (always-on) → $0.04/day (scale-to-zero). Do not re-add
   `--min-instances 1` without checking the bill impact.
2. **`twin-due-work` Cloud Scheduler job paused** (2026-10-06). The
   twin-maintenance runtime was ~93% of the bill (~$15–19/month) while its
   account-maintenance/research adapters were never wired — every cycle
   degraded (`account_maintenance_adapter_unconfigured`), and the forward
   trial gate G1 shows 0 collected days. Resume with
   `gcloud scheduler jobs resume twin-due-work --location=us-central1` only
   after wiring the adapters; otherwise tear down per `infra/twin/README.md`.

Consequences:

- **The OCI/VPS migration is prepared but on hold.** All tooling is merged
  (#669 data layer, #670 runbook, #673 bootstrap/deploy/CI automation) and
  ready to execute if the calculus changes — most plausibly a twin revival
  (its runtime costs ~$15–19/month on GCP, free on an OCI Ampere box). Do not
  re-run the migration runbook without a cost or capability reason; see
  `deploy/vps/README.md` (status note at top).
- A **€20 monthly budget alert** (`bf9e0444-a41b-42ea-89d2-62f812ffe26b`,
  billing account `014119-99A1B6-A1BA39`, scoped to this project) now watches
  spend at 50/90/100%. If it fires, something regressed — check
  `--min-instances` and the twin scheduler state first.
- The VPS cron's `twin-due-work` entry only works if the twin runtime moves
  to the box (shared-secret auth); against the GCP-deployed twin it fails
  auth by design. Leave it alone during any future cutover decision.

### 2026-10-06 — Llama retry-loop fix: capacity pre-filter + credible-edge sizing cap (Copilot)

Llama claimed 66–83pp edges on the Khamenei market and re-attempted the same
blocked trade 26 times in October (34 attempts, all `rejected_before_execution`).
Two fixes:

1. **`_drop_capacity_exhausted_candidates`** (scripts/agent_trading_tick.py)
   now also drops candidates where the *smallest possible order* (profile
   `max_order_notional_pct` of account value) would breach the market or
   cluster cap — not just when existing cost is already ≥95% of cap. Llama
   sat at ~$1,355 of a ~$1,500 cap, under the 95% threshold, so every cycle
   burned a full LLM run on a guaranteed rejection.
2. **`_sizing_plan`** (src/analyzing_llm_rationale/benchmark_tools.py) caps
   the edge used for *sizing* at `FORESEA_AGENT_MAX_CREDIBLE_EDGE` (10pp).
   The guard already rejects entries above it; sizing must not stake a
   fantasy edge. The stated edge is still reported untouched in
   `plan["edge"]` for the audit trail; only the stake derivation is
   tempered. Float boundary handled: a capped edge is snapped to exactly the
   ceiling so the guard's and sizing's edge buckets agree.

Do not revert the sizing cap thinking it hides information: the raw stated
probability and edge remain in the audit; only Kelly's stake input changes.
Tests: `CredibleEdgeSizingCapTests` (test_edge_reliability_sizing.py) and
`CapacityExhaustedCandidateFilterTests` (test_fleet_durability_v3.py).

### 2026-10-06 — Moved redundant Cloud Run work to GitHub Actions (Copilot)

Deleted the `metaculus-github-dispatch` Cloud Run job and paused its
scheduler: `.github/workflows/metaculus-futureeval.yml` already has its own
6x-hourly GitHub cron at the same cadence, so the job was a redundant
duplicate trigger (~$13/mo for nothing). If tournament dispatches ever stop,
check the GitHub cron first, not Cloud Run.

Repointed `agent_trading_tick.py`'s leaderboard fetch from
`https://foresea.ink/agent-trading/board` to the raw GitHub payload it
proxies (`raw.githubusercontent.com/.../static/agent_trading_live.json`).
The tick runs on GitHub runners, so this removes ~58% of all requests
hitting production (the board endpoint was the single biggest warm-keeper)
at zero cost. Do not point scheduled scripts at foresea.ink when the same
data is published to raw.githubusercontent.com or a bucket.

Twin dispatch stays on Cloud Scheduler: moving it to Actions saves nothing
(the scheduler is ~free; the twin compute remains either way).

### 2026-10-06 — GCP cost reduction applied; target host is OCI, not a generic VPS (Copilot)

Executed the two cheapest cost actions on project `brave-drive-471109-d9`:

1. **Deleted the three idle staging chat services** (last deployed Jul 31,
   unused since): `analyzing-llm-rationale-staging-chat`,
   `-chat-branch`, `-chat-ui`. Kept `analyzing-llm-rationale-staging`
   because `.github/workflows/staging.yml` still targets it via
   workflow_dispatch / push to the `staging` branch.
2. **Reduced Cloud Scheduler `twin-due-work` from every 5 min to every
   15 min** (`infra/twin/deploy.ps1` updated to match). Safe because twin
   work is durable in the `TwinWorkerJob` table and recovered by the next
   `dispatch_due_jobs` pass — the schedule only controls latency, not
   correctness.

**Target host decision: OCI Always Free (Ampere A1, ARM), not Hetzner.**
The compose stack in `deploy/vps/` is arch-agnostic (builds for the host
arch by default; PyTorch aarch64 CPU wheels verified). See
`434a71719 docs(deploy): add Oracle Cloud (OCI) Always Free walkthrough`
for the OCI-specific runbook. Do not provision a paid Hetzner box for this.

### 2026-10-06 — Datastore backup must NOT be a GitHub artifact (Copilot, reviewing DeepSeek's change)

DeepSeek moved the daily Datastore backup from GCS to a GitHub Actions
artifact (`datastore-backup.yml`). Two blocking problems:

1. **Privacy leak.** This repo is **public**. Artifacts on public repos are
   downloadable by anyone with a GitHub account. The export contains user
   emails, OAuth ids (`alt_subs`), and private chat conversations
   (`Conversation` kind). Never upload Datastore exports, user data, or
   anything derived from them as repo artifacts, release assets, or commits.
2. **Verify step fails most days.** `migrate_datastore_to_sql.py --verify`
   compares per-kind counts, but the twin runtime writes
   `TwinPublicEvidenceCache` / `TwinResearchCapture` every ~5 minutes, so
   counts drift between export and verify. Point-in-time drift is expected
   (see `deploy/vps/README.md`), so verify must be advisory, not fatal.

Resolution: backup is a SQLite file (portable, restorable into the SQL
backend) mirrored **only** to the private Cloudflare R2 bucket
(`datastore-backups/<date>/`, 7-day retention), with an advisory verify and a
hard failure if `R2_STATE_BUCKET` is unset (a green run with nowhere to store
the backup is worse than a red one). The GCS export bucket
`brave-drive-471109-d9-datastore-backups` keeps its 30-day lifecycle rule as a
second recovery point until the VPS cutover retires Datastore entirely.

**Status: applied.** `datastore-backup.yml` now exports to SQLite, verifies
advisory, uploads to `R2_STATE_BUCKET` (`foresea-state`) under
`datastore-backups/<date>/`, and expires objects past the retention window.
There is no `actions/upload-artifact` step and no `gs://` destination. The
workflow fails fast if `R2_STATE_BUCKET` is unset.

Note for future agents: the same rule applies to any new workflow. Before
adding an artifact, release asset, or committed file, ask whether it contains
or derives from user data. This repo is public.

### 2026-10-06 — GCS lifecycle rules applied (Copilot)

`gcp_cost_optimizer.py` policies existed but were never applied. Applied via
`gcloud storage buckets update`: 30-day delete on the datastore-backups
bucket, noncurrent-version cleanup on the track-record-store bucket, and the
cloudbuild bucket policy. Do not re-apply; do not delete the backups bucket
while Datastore is still the primary store.

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
# Use the `serve` command, NOT `uvicorn server:app`: importing the module
# directly leaves `_state` empty, so /ready stays 503 and every data endpoint
# fails. `serve` calls init_server_state() first.
PYTHONPATH=src analyze-llm-rationale serve \
  --model gpt-oss-120b --variant variant0_neutral_baseline --port 8080

# Note: `python -m analyzing_llm_rationale.cli` is a no-op -- cli.py has no
# `if __name__ == "__main__"` guard. Use the console script, or call main().

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

Verify a route is live before debugging anything else — the provider publishes
a status probe, and a retired route fails with only a generic "provider
unavailable":

```bash
curl -s https://llm.scads.ai/status/state.json | python -c \
  "import json,sys; [print(m['name'], m['state']) for m in json.load(sys.stdin)['models']['Chat']]"
analyze-llm-rationale smoke-test --model glm-5-3
```

Note that the reasoning models (`glm-5-3-flash`, `deepseek-v4-flash`) emit
`reasoning_content` before `content`. A smoke test with a very small
`max_tokens` can return an empty `content` because the budget was spent on
reasoning — that is not a broken route.

## Data storage

Durable state lives in Cloud Datastore, reached through
`analyzing_llm_rationale.datastore_backend`, which selects the implementation
from `FORESEA_DATASTORE_BACKEND`:

| Value | Backend |
|---|---|
| `gcp` (default) | `google.cloud.datastore` — unchanged production behaviour |
| `sql` | SQLite via `datastore_sql.py` (portable; used off GCP) |

`datastore_sql.py` is a drop-in for the Datastore surface the app uses
(`Client`, `Key`, `Entity`, `PropertyFilter`, ancestor queries, namespaces,
transactions). `trackrec_store.py` is the same pattern for the track record.
See `deploy/vps/README.md` for the migration and cutover runbook.

## Deployment

### Cloud Run (public, scales to zero)
```
https://foresea.ink
```
- `GET /health` → `{"status": "ok"}`
- `POST /predict` — PredictRequest → PredictResponse
- `GET /mcp/` — Model Context Protocol Streamable-HTTP endpoint

Cloud Run sizing is set in `docker.yml` via `CLOUD_RUN_MEMORY` /
`CLOUD_RUN_CPU` (default `1Gi` / `1`). Override a manual run with
`gh workflow run docker.yml -f memory=2Gi`. Do not hardcode these in the
deploy step — a hardcoded value silently reverts any manual change on the
next push to `main`.

### Self-hosted VPS
`deploy/vps/` holds a provider-agnostic Docker Compose stack (app + Caddy TLS
+ nightly SQLite backup + the metaculus-dispatch job), a cron schedule
replacing Cloud Scheduler, and the cutover runbook. See
`deploy/vps/README.md` and `deploy/vps/OCI.md`.

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
3. `oci-deploy.yml` — SSH deploy to the OCI instance (opt-in: only runs once
   the repo variable `OCI_DEPLOY_ENABLED=true` and the `OCI_HOST`,
   `OCI_SSH_USER`, `OCI_SSH_PRIVATE_KEY` secrets are set; see
   `deploy/vps/OCI.md` §8)

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
