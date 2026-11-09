# Foresea API Reference

Welcome to the **Foresea Intelligence & Trading API** specification. Foresea is an autonomous prediction market intelligence framework combining real-time probability forecasting, statistical edge discovery across Polymarket and Kalshi, multi-agent adversarial debates, portfolio optimization, and non-custodial execution.

---

## 📑 Table of Contents

1. [Architecture & Getting Started](#1-architecture--getting-started)
2. [Authentication & Security](#2-authentication--security)
3. [Inference & Probability Forecasting](#3-inference--probability-forecasting)
4. [Market Data & Microstructure](#4-market-data--microstructure)
5. [Statistical Edge, Arbitrage & Whale Flow](#5-statistical-edge-arbitrage--whale-flow)
6. [Autonomous Multi-Agent Intelligence & Debates](#6-autonomous-multi-agent-intelligence--debates)
7. [Autonomous Operator Twin Platform](#7-autonomous-operator-twin-platform)
8. [Specialized Alpha Desks (Weather & Crypto 5m)](#8-specialized-alpha-desks)
9. [Portfolio Optimization & Sizing](#9-portfolio-optimization--sizing)
10. [Guarded Trading Execution](#10-guarded-trading-execution)
11. [Direct Venue Pass-Through Operations (49 Ops)](#11-direct-venue-pass-through-operations)
12. [Real-Time Streaming & WebSockets](#12-real-time-streaming--websockets)
13. [Webhooks & Event Subscriptions](#13-webhooks--event-subscriptions)
14. [Knowledge Base & Personal RAG](#14-knowledge-base--personal-rag)
15. [Track Record & Dataset Exports](#15-track-record--dataset-exports)
16. [System Health & Telemetry](#16-system-health--telemetry)

---

## 1. Architecture & Getting Started

### Base URLs
- **Production Public API**: `https://foresea.ink`
- **Model Context Protocol (MCP)**: `https://foresea.ink/mcp/`
- **Interactive Swagger UI**: `https://foresea.ink/docs`
- **Interactive Redoc**: `https://foresea.ink/redoc`
- **OpenAPI 3.1 Spec**: `https://foresea.ink/openapi.json`

### Rate Limits
- **Global API Rate Limit**: 60 requests per minute per IP address. Exceeding returns `429 Too Many Requests` with a `Retry-After: 60` header.
- **LLM Prediction Rate Limit**: 10 LLM inference requests per minute per IP address on `/predict`, `/predict/stream`, and `/agent/analyze`.

---

## 2. Authentication & Security

Foresea uses tiered authentication depending on endpoint criticality:

1. **Public Endpoints** (Quotes, Track Record, MCP, Health): No authentication required.
2. **API-Key Guarded Endpoints**: When configured via `API_KEY` on the server, protected administrative endpoints require:
   ```http
   X-API-Key: <your-api-key>
   ```
3. **User-Authenticated Endpoints** (Trading, Connections, Guardrails, Personal RAG, Twin, Private Runs): Require a JWT Bearer session token:
   ```http
   Authorization: Bearer <foresea-session-jwt>
   ```
   Tokens are obtained via `/auth/google`, `/auth/github`, `/auth/login`, or `/auth/register`.

---

## 3. Inference & Probability Forecasting

### `POST /predict`
Run probability prediction for a binary, multiple-choice, numeric, or date market question with structured rationale and live news evidence.

**Request Payload:**
```json
{
  "question": "Will the Federal Reserve reduce interest rates at the next FOMC meeting?",
  "question_type": "binary",
  "market_platform": "Polymarket",
  "market_probability": 0.42,
  "variant": "variant0_neutral_baseline",
  "attach_evidence": true,
  "evidence_top_k": 5
}
```

**Response (`PredictResponse`):**
```json
{
  "question_type": "binary",
  "predicted_answer": "Yes",
  "confidence": 0.78,
  "options": [],
  "range_forecast": null,
  "rationale": "Recent CPI and PCE readings show inflation moving sustainably toward target...",
  "model_rationale": "Recent CPI and PCE readings show inflation moving sustainably toward target...",
  "variant": "variant0_neutral_baseline",
  "model_key": "gpt-oss-120b",
  "evidence_sources": [
    {
      "source": "Reuters",
      "title": "Fed policymakers signal rate cuts on track",
      "url": "https://reuters.com/markets/rates",
      "publish_date": "2026-11-01T12:00:00Z",
      "relevance_score": 0.91
    }
  ],
  "market_analysis": {
    "platform": "Polymarket",
    "outcome": "Yes",
    "market_probability": 0.42,
    "model_probability": 0.78,
    "edge": 0.36,
    "stance": "model_above_market",
    "summary": "Foresea is 36 percentage points above the market on Yes."
  }
}
```

### `POST /predict/stream`
Server-Sent Events (SSE) streaming endpoint returning chunked tokens for real-time conversational forecasting interfaces.

### `POST /predict/batch`
Batch forecast up to 25 prediction market questions concurrently with bounded concurrency limits.

**Request:**
```json
{
  "questions": [
    {"question": "Will GDP expand by >= 2% in Q3?", "market_probability": 0.60},
    {"question": "Will headline CPI fall below 2.5%?", "market_probability": 0.45}
  ],
  "concurrency_limit": 5
}
```

### `POST /predict/ensemble`
Calculates a Brier-weighted ensemble forecast blending predictions across multiple models (e.g. `gemma-4-26b-a4b-it`, `gpt-oss-120b`, `minimax-m3`).

### `POST /v1/batch/forecast`
Institutional-grade concurrent batch forecasting with per-item status tracking and latency telemetry.

---

## 4. Market Data & Microstructure

### `GET /markets/polymarket`
Fetch live normalized Polymarket market quotes, token IDs, liquidity, and volume.
- **Parameters**: `slug` (string) or `market_id` (string).

### `GET /markets/kalshi`
Fetch live normalized Kalshi market quotes, strike ranges, ticker metadata, and resting prices.
- **Parameters**: `ticker` (string).

### `GET /market/batch`
Fetch quotes for up to 50 markets across venues in a single round-trip.
- **Parameters**: `refs` (comma-separated list of tickers/slugs).

### `GET /market/orderbook`
Fetch live Level-2 bids and asks orderbook depth.
- **Parameters**: `ticker_or_token` (string), `platform` (`kalshi` or `polymarket`).

### `GET /market/trades`
Fetch recent executed trade tape prints including contract sizes, execution prices, and timestamps.
- **Parameters**: `platform` (`kalshi` or `polymarket`), `ticker_or_token` (string), `limit` (int, default 20).

### `GET /market/price-history`
Fetch historical candlestick OHLCV timeseries.
- **Parameters**: `ticker_or_market` (string), `series_ticker` (optional string).

### `GET /market/leaderboard`
Top trader rankings, total volume leaders, and profitability rankings.

### `GET /market/exchange-status`
Operational status, exchange halt states, and scheduled maintenance windows.

### `GET /market/history` & `POST /market/explain-shift`
Retrieve historical forecast snapshots for a market and explain the temporal shift in model odds over time.

---

## 5. Statistical Edge, Arbitrage & Whale Flow

### `GET /edge-board`
Live model-vs-market pricing discrepancies across Polymarket and Kalshi, ranked by statistical edge:
$$\text{Edge} = P_{\text{model}} - P_{\text{market}}$$

### `GET /radar` & `GET /radar/delta`
Homepage market desk feed surfacing live model-vs-market gaps, liquidity flags, and catalyst dates.

### `GET /v1/arbitrage/cross-venue`
Scans for synthetic arbitrage and cross-venue pricing discrepancies between Polymarket and Kalshi (e.g., $P_{\text{YES, Poly}} + P_{\text{NO, Kalshi}} < 1.00$).
- **Parameters**: `min_spread` (float, default 0.03).

### `GET /v1/market/whale-flow`
Detects and tracks large institutional block trades (>= $250 USD) across prediction venues.
- **Parameters**: `min_notional` (float, default 250.0), `limit` (int, default 30).
- **Response**: Net whale volume, bullish accumulation vs bearish distribution index, and individual prints.

---

## 6. Autonomous Multi-Agent Intelligence & Debates

### `POST /agent/analyze` & `POST /agent/analyze/stream`
Orchestrates autonomous end-to-end market analysis:
1. Resolves market and fetches live orderbook quotes.
2. Gathers and cleans multi-source news evidence.
3. Computes calibrated probability and statistical edge.
4. Executes custom user-specified analysis skills.
5. Produces actionable recommendation (`buy_yes`, `buy_no`, `hold`).

### `POST /agent/debate`
Executes a 3-agent adversarial cross-examination (Bull Agent, Bear Agent, and Impartial Judge) to identify hidden cognitive biases, tail risks, and blind spots.

**Request:**
```json
{
  "question": "Will SpaceX successfully land Starship on Mars by 2028?",
  "platform": "Polymarket",
  "market_probability": 0.28
}
```

**Response:**
Returns Bull argument, Bear counter-thesis, Judge synthesis, identified blind spots, and probability adjustments.

### `GET /agent/scan`
Scans a prediction venue, forecasts all qualified active markets, and returns markets clearing `min_edge`.

### `GET /agent/runs` & `GET /agent/runs/{run_id}`
Durable research run audit trails for signed-in users.

### `GET /agent-profiles`, `POST /agent-profiles/copy`, `DELETE /agent-profiles/{profile_id}`
Copy public tournament models into private, immutable version-1 research recipes.

---

## 7. Autonomous Operator Twin Platform

Foresea provides an autonomous operator twin framework (`/twin/*`) for running sovereign portfolio managers under strict governance mandates:

| Endpoint | Method | Purpose |
|---|---|---|
| `/twin/mandates` | `POST` | Create a durable risk and execution mandate. |
| `/twin/mandates/{mandate_id}` | `GET` | Read mandate details, risk constraints, and active status. |
| `/twin/mandates/{mandate_id}/revisions` | `POST` | Submit revisions to mandate sizing or trading parameters. |
| `/twin/mandates/{mandate_id}/approve` | `POST` | Explicit risk approval of a proposed mandate. |
| `/twin/mandates/{mandate_id}/revoke` | `POST` | Immediately revoke mandate permissions. |
| `/twin/status` | `GET` | Current operational state, loop cadences, and system health. |
| `/twin/portfolio` | `GET` | Mark-to-market portfolio value, open positions, and cash allocations. |
| `/twin/decisions` | `GET` | Audit trail of autonomous decisions made by the twin runtime. |
| `/twin/commands` | `GET` | Execution commands dispatched to venue connectors. |
| `/twin/readiness` | `GET` | Pre-flight validation checklist for automated trading execution. |
| `/twin/pause` | `POST` | Emergency kill switch to pause all automated twin activity. |

---

## 8. Specialized Alpha Desks

### `GET /market/weather-radar`
Domain-specific weather market intelligence evaluating temperature anomalies across major US NWS stations compared against Kalshi weather brackets.

### `GET /crypto-5m/equity` & `GET /crypto-5m/kalshi-edge`
Real-time tracking of high-frequency 5-minute crypto micro-markets (BTC, ETH, SOL), tracking paper equity curves and model-vs-market calibration.

---

## 9. Portfolio Optimization & Sizing

### `GET /portfolio/optimal-allocation`
Computes Fractional Kelly position sizing across Grade A/B market opportunities.
- **Parameters**:
  - `bankroll` (float, default 1000.0)
  - `kelly_fraction` (float, default 0.25 for Quarter-Kelly)
  - `min_edge` (float, default 0.05)
  - `min_credibility` (float, default 0.60)

### `POST /portfolio/optimize`
Applies portfolio optimization constraints across caller-supplied candidate positions.

---

## 10. Guarded Trading Execution

All real-money trading is protected by institutional-grade safety guardrails:
- Must have `FORESEA_ENABLE_TRADING=true`.
- Requires authenticated user session (`Authorization: Bearer <token>`).
- Requires two-step confirmation (`execute=true` and `confirmation="PLACE REAL ORDER"`).
- Exchange credentials are never stored in plaintext — encrypted per-user using Cloud KMS data-encryption keys.

| Endpoint | Method | Description |
|---|---|---|
| `/trading/connections` | `GET`, `PUT`, `DELETE` | Manage KMS-encrypted exchange credentials for Kalshi or Polymarket. |
| `/trading/guardrails` | `GET`, `PUT` | Read or narrow user-specific risk boundaries (max order notional, daily loss limits). |
| `/trading/preview` | `POST` | Dry-run order pre-flight validation, tick collar checks, and exposure sizing. |
| `/trading/orders` | `POST` | Submit an order to venue with explicit user confirmation. |
| `/trading/portfolio` | `GET` | Reconcile account balances, open positions, and resting orders. |
| `/trading/orders` | `GET` | View audit trail of submitted orders. |
| `/trading/orders/{audit_order_id}` | `DELETE` | Cancel resting orders on exchange. |
| `/trading/orders/{audit_order_id}/reconcile` | `POST` | Reconcile order fill status against exchange records. |
| `/trading/runs` | `GET`, `POST` | Create and review durable live-trade execution runs. |

---

## 11. Direct Venue Pass-Through Operations

Foresea standardizes 49 native exchange operations across Kalshi and Polymarket:

### Discovery Routes
- `GET /market/venue/catalog`: Public operations and schema catalog.
- `GET /trading/venue/catalog`: Authenticated account reads and order actions catalog.

### Execution Routes
- `POST /market/venue/{platform}/{operation}`: Execute public read operation.
- `POST /trading/venue/{platform}/{operation}`: Execute authenticated account read.
- `POST /trading/venue/{platform}/actions/{operation}`: Execute guarded order action.

### Operations Reference (49 Operations)
- **Kalshi Public (13)**: `historical_cutoff`, `historical_markets`, `historical_market`, `historical_candles`, `historical_trades`, `candlesticks`, `series_fee_changes`, `event_fee_changes`, `event_metadata`, `milestones`, `milestone`, `weather`, `weather_calibrations`.
- **Kalshi Account Reads (11)**: `historical_fills`, `historical_orders`, `historical_positions`, `settlements`, `balance`, `positions`, `orders`, `fills`, `orderbooks`, `order_groups`, `order_group`.
- **Kalshi Order Actions (9)**: `amend_order`, `decrease_order`, `cancel_orders`, `cancel_all`, `create_order_group`, `delete_order_group`, `reset_order_group`, `trigger_order_group`, `limit_order_group`.
- **Polymarket Public (7)**: `books`, `midpoints`, `spreads`, `fee_rate`, `holders`, `oi`, `live_volume`.
- **Polymarket Account Reads (6)**: `activity`, `closed_positions`, `value`, `positions`, `orders`, `fills`.
- **Polymarket Order Actions (3)**: `cancel_all`, `cancel_market_orders`, `heartbeat`.

---

## 12. Real-Time Streaming & WebSockets

### Server-Sent Events (SSE)
- `GET /feed/stream`: Unified Foresea Alpha & Autonomous Agent live event feed.
- `GET /radar/stream`: Live delta updates for the market radar desk.
- `GET /stream/radar`: High-frequency radar price updates.

### WebSockets
- `WS /ws/venue/{platform}`: Direct streaming of orderbooks, trades, and account fills.
- `WS /ws/radar`: Low-latency market radar websocket feed.

---

## 13. Webhooks & Event Subscriptions

Register external HTTP callback endpoints to receive instant alerts when market probabilities shift, arbitrage opportunities arise, or agent tournament trades occur:

- `POST /webhooks/subscribe`: Register webhook URL and event filters.
- `GET /webhooks/subscriptions`: List active subscriptions.
- `DELETE /webhooks/subscriptions/{sub_id}`: Remove webhook subscription.

---

## 14. Knowledge Base & Personal RAG

Personalized Retrieval-Augmented Generation (RAG) store for injecting proprietary documents, research notes, and domain reports into forecasts:

- `POST /rag/ingest`: Upload and index document text.
- `GET /rag/search`: Search knowledge base with semantic similarity scoring.
- `GET /rag/documents`: List user-ingested documents.
- `DELETE /rag/documents`: Delete documents from personal knowledge base.

---

## 15. Track Record & Dataset Exports

- `GET /track-record`: Complete public track record including Brier scores, log scores, and calibration statistics across lead-time horizons.
- `GET /track-record/digest`: Markdown summary suitable for syndication.
- `GET /analytics/export.parquet`: High-performance analytical export in Apache Parquet format:
  - `?dataset=edge_board`: Complete edge board timeseries.
  - `?dataset=models_comparison`: Side-by-side model forecasting calibration data.

---

## 16. System Health & Telemetry

- `GET /health`: Basic liveness check (`{"status": "ok"}`).
- `GET /ready`: Readiness check verifying server state and models are loaded.
- `GET /v1/system/health`: Institutional health metrics, venue connectivity status, latency percentiles, and engine calibration state.
