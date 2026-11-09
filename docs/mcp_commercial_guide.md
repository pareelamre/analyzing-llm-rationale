# Foresea MCP Server: Commercial Harness Integration Guide

Foresea provides a high-performance **Model Context Protocol (MCP)** server that equips AI agents (Claude Code, Google Antigravity, OpenAI Codex, Cursor, Windsurf, OpenHands) with **calibrated probability forecasting**, **real-time prediction market odds (Polymarket & Kalshi)**, **orderbook depth**, **3-agent bull-bear adversarial debates**, **portfolio optimization**, and **guarded execution**.

---

## 1. Quick Launch & Transports

### Using `uvx` (Fast, isolated Python startup)
```bash
uvx --from git+https://github.com/pareelamre/analyzing-llm-rationale.git foresea-mcp
```

### Streamable-HTTP / Remote SSE Endpoint
For cloud agents and IDEs, Foresea serves remote MCP at:
```text
https://foresea.ink/mcp/
```
Discovery manifest: `https://foresea.ink/.well-known/mcp/server.json`

### Local Stdio (Default)
```bash
python -m analyzing_llm_rationale.mcp_server --transport stdio
```

### Self-Hosted SSE Daemon
```bash
foresea-mcp --transport sse --port 8080
```

---

## 2. Commercial Harness Configurations

### Google Antigravity
Add to `~/.gemini/antigravity/mcp/foresea.json`:

```json
{
  "name": "foresea",
  "command": "uvx",
  "args": [
    "--from",
    "git+https://github.com/pareelamre/analyzing-llm-rationale.git",
    "foresea-mcp"
  ],
  "env": {
    "FORESEA_BASE_URL": "https://foresea.ink",
    "FORESEA_API_KEY": "YOUR_FORESEA_API_KEY"
  }
}
```

### Claude Desktop & Claude Code
Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "foresea": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/pareelamre/analyzing-llm-rationale.git",
        "foresea-mcp"
      ],
      "env": {
        "FORESEA_BASE_URL": "https://foresea.ink",
        "FORESEA_API_KEY": "YOUR_FORESEA_API_KEY"
      }
    }
  }
}
```

### Cursor IDE
Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "foresea-remote": {
      "url": "https://foresea.ink/mcp/",
      "headers": {
        "Authorization": "Bearer YOUR_FORESEA_API_KEY"
      }
    }
  }
}
```

### Windsurf (Codeium Cascade)
Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "foresea": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/pareelamre/analyzing-llm-rationale.git",
        "foresea-mcp"
      ],
      "env": {
        "FORESEA_BASE_URL": "https://foresea.ink",
        "FORESEA_API_KEY": "YOUR_FORESEA_API_KEY"
      }
    }
  }
}
```

---

## 3. MCP Tool Inventory (29 Native Tools)

Foresea exposes 29 domain-specialized tools covering the full lifecycle of prediction market intelligence:

### Core Forecasting & Research
| Tool Name | Parameters | Description |
|---|---|---|
| `foresea_forecast` | `question`, `question_type`, `market_platform`, `market_probability`, `variant`, `model`, `attach_evidence`, `evidence_top_k` | Calibrated probability forecasting with structured rationale and live news evidence grounding. |
| `foresea_analyze_market` | `question`, `platform`, `slug`, `ticker`, `skills`, `history` | Orchestrated end-to-end market analysis, statistical edge derivation, and modular skill execution. |
| `foresea_scan_markets` | `platform`, `limit`, `min_edge`, `evidence_top_k` | Scans prediction venues for mispriced contracts ranked by statistical edge. |
| `foresea_batch_quotes` | `refs` (list of tickers/slugs) | High-throughput batch quote lookup across Polymarket and Kalshi. |
| `foresea_check_run` | `client_run_key` | Polls or inspects the durable state and report of an asynchronous research run. |
| `foresea_batch_forecast` | `payload` (`questions` list) | Concurrent batch probability forecasting across multiple market questions. |

### Prediction Market Microstructure & Quotes
| Tool Name | Parameters | Description |
|---|---|---|
| `foresea_exchange_status` | *(none)* | Operational status, trading halts, and market session schedules for exchanges. |
| `foresea_orderbook` | `ticker_or_token`, `platform` | Live Level-2 bids and asks orderbook depth for Kalshi tickers or Polymarket tokens. |
| `foresea_recent_trades` | `platform`, `ticker_or_token`, `limit` | Executed trade prints, taker sides, contract sizes, and trade timestamps. |
| `foresea_market_tags` | *(none)* | Taxonomy, category tags, and classification metadata for prediction contracts. |
| `foresea_price_history` | `ticker_or_market`, `series_ticker` | Historical candlestick OHLCV timeseries and price trajectories. |
| `foresea_live_data` | `event_ticker`, `data_type`, `milestone_id` | Live event status, sports play-by-play statistics, and scheduled milestone outcomes. |
| `foresea_polymarket_meta` | `target`, `market_id` | Resolution rules, series groupings, and community discussions. |
| `foresea_market_leaderboard` | `limit` | Volume leaders, PnL rankings, and top trader performance on prediction markets. |

### Statistical Edge & Trading Desks
| Tool Name | Parameters | Description |
|---|---|---|
| `foresea_edge_board` | `limit`, `min_edge`, `fields` | Live ranked model-vs-market alpha opportunities backed by Brier skill scores. |
| `foresea_track_record` | `fields` | Public live track record, historical calibration curves, and Brier lead-time scores. |
| `foresea_debate_market` | `question`, `platform`, `market_probability`, `resolution_criteria` | 3-agent adversarial cross-examination (Bull, Bear, Judge) for stress-testing blind spots. |
| `foresea_optimize_portfolio` | `bankroll_usd`, `kelly_fraction`, `min_edge` | Computes optimal position sizes across opportunities using the Fractional Kelly criterion. |
| `foresea_feed_latest` | `limit`, `min_edge` | Real-time alpha feed items, trade signals, and autonomous agent tournament events. |
| `foresea_weather_radar` | `target_date` | City-level high-temperature prediction radar and climate forecast deviations. |
| `foresea_weather_forecast` | `station_or_query`, `target_date` | Multi-source ensemble weather model predictions calibrated against historical NWS stations. |
| `foresea_crypto_edge` | `min_edge`, `limit` | 5-minute crypto micro-market signals and real-time paper equity calibration. |

### Venue Operations & Direct Pass-Through
| Tool Name | Parameters | Description |
|---|---|---|
| `foresea_venue_data` | `platform`, `operation`, `parameters`, `body` | Direct pass-through access to 49 native exchange operations across Kalshi and Polymarket. |

### Guarded Non-Custodial Trading Execution
| Tool Name | Parameters | Description |
|---|---|---|
| `foresea_trade_account_status` | *(none)* | Inspects user exchange connection status and trading guardrail limits. |
| `foresea_preview_order` | `payload` | Dry-run order pre-flight validation, collar checks, and exposure estimation. |
| `foresea_place_order` | `payload` | Submits a live order requiring explicit confirmation (`PLACE REAL ORDER`). |
| `foresea_cancel_order` | `order_id`, `platform`, `ticker` | Cancels resting order on venue. |

### External Event Webhooks
| Tool Name | Parameters | Description |
|---|---|---|
| `foresea_webhook_subscribe` | `payload` | Subscribes an external agent webhook URL to market price and edge change events. |

---

## 4. MCP Resources (8 Live Context Streams)

Agents can directly read dynamic resource URIs for instant context without invoking tool functions:

| Resource URI | Description |
|---|---|
| `foresea://weather/radar` | Real-time weather market radar across US metropolitan stations. |
| `foresea://track-record` | Up-to-date public forecasting track record and calibration metrics. |
| `foresea://edge-board` | Ranked live model-vs-market pricing discrepancies. |
| `foresea://trending-markets` | Most active and liquid prediction markets across Polymarket and Kalshi. |
| `foresea://openapi` | Machine-readable OpenAPI 3.1 specification of the Foresea server. |
| `foresea://leaderboard` | Top prediction market traders and liquidity provider rankings. |
| `foresea://radar` | Foresea market desk radar summary snapshot. |
| `foresea://crypto/edge` | Real-time 5-minute crypto micro-market calibration and paper equity curves. |

---

## 5. MCP Prompt Templates (6 Standard Workflows)

Foresea provides pre-engineered prompt workflows that AI agent harnesses can trigger:

| Prompt Name | Arguments | Purpose |
|---|---|---|
| `foresea_forecast` | `question` | Orchestrates single-question forecasting with news grounding. |
| `foresea_batch_forecast` | `questions_csv` | Processes a batch of market questions concurrently. |
| `foresea_crypto_edge` | *(none)* | Audits short-horizon crypto opportunities on Kalshi/Polymarket. |
| `foresea_market_risk` | `market_or_topic` | Deep dive into resolution criteria ambiguity, catalyst dates, and liquidity risks. |
| `foresea_calibrate_hypothesis` | `hypothesis` | Cross-examines a forecasting hypothesis using empirical base rates. |
| `foresea_system_prompt` | *(none)* | Returns Foresea's core probabilistic forecasting and risk persona. |
