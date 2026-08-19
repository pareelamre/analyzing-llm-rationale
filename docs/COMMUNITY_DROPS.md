# Foresea MCP Server — Developer Community Drop & Integration Kit

Use these tailored, copy-paste ready drops formatted specifically for **ElizaOS**, **LangChain**, **Claude Desktop**, **Cursor**, and **Reddit/Discord** builder communities.

---

## 1. ElizaOS (ai16z) Developer Discord (`#plugins` / `#general`) & GitHub

### Discord / Forum Post:
> **Title:** 🌊 Foresea Prediction Market & Forecasting MCP Plugin (Live Polymarket & Kalshi Data)
> 
> Hey everyone! We built a remote Model Context Protocol (MCP) server for prediction market intelligence that you can plug into any ElizaOS agent with zero extra infrastructure.
> 
> **What your agent gets access to:**
> • `foresea_feed_latest` — Stream live mispricings & alpha signals from Polymarket & Kalshi
> • `foresea_forecast` — Calibrated probability forecasts with evidence decomposition & citations
> • `foresea_scan_markets` & `foresea_orderbook` — Real-time venue orderbooks, candlesticks, and recent trade tapes
> • `foresea_edge_board` & `foresea_optimize_portfolio` — Mathematical Fractional Kelly capital allocation
> 
> **Quick Setup:**
> Streamable-HTTP MCP endpoint (Public & free):
> `https://foresea.ink/mcp/`
> 
> GitHub & Starter examples: https://github.com/pareelamre/analyzing-llm-rationale  
> Demo Web Desk: https://foresea.ink  
> 
> We have 10 autonomous LLMs running shadow trading cycles every 15 mins on this exact MCP loop. Let us know if you want to build custom agents on top of it!

---

## 2. Claude Desktop (`claude_desktop_config.json`) Integration Drop

### Copy for Claude Desktop Builders:
Add Foresea directly to your Claude Desktop config to give Claude real-time prediction market tools:

```json
{
  "mcpServers": {
    "foresea": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-everything"],
      "env": {
        "FORESEA_MCP_URL": "https://foresea.ink/mcp/"
      }
    }
  }
}
```

Or using native Python Streamable-HTTP client:
```json
{
  "mcpServers": {
    "foresea": {
      "command": "python",
      "args": ["-m", "analyzing_llm_rationale.mcp_server", "--transport", "stdio"]
    }
  }
}
```

---

## 3. LangChain / Langflow Community (`#showcase` / Discussion)

### Message:
> **Subject:** Foresea Prediction Market Tool for LangChain Agents
> 
> Built a high-precision forecasting tool for LangChain and LangGraph workflows:
> • Endpoint: `https://foresea.ink/mcp/` (FastMCP Streamable-HTTP) or REST `https://foresea.ink/predict`
> • Features: Calibrated probability evaluation, evidence gathering across news/filings, and model-vs-market edge calculation against live Polymarket/Kalshi orderbooks.
> • OpenAPI Schema: `https://foresea.ink/openapi.json`
> • Python SDK Demo: `pip install mcp` -> connect to `https://foresea.ink/mcp/`
> 
> Perfect for agents needing grounded probability estimations or automated risk management.

---

## 4. Reddit (`r/LocalLLaMA`, `r/PredictionMarkets`, `r/ClaudeAI`)

### Post Draft:
> **Title:** We built an open MCP server for prediction market forecasting + 10 autonomous AI paper traders (Polymarket & Kalshi)
> 
> Most LLM forecasting fails because models are overconfident on tail events. We built **Foresea** (`https://foresea.ink`), an evidence-grounded forecasting system that calibrates model probabilities against historical resolutions and live market prices.
> 
> **Key features we released:**
> 1. **Public MCP Server (`https://foresea.ink/mcp/`)**: 19 tools for market scans, orderbooks, trade tapes, calibrated probability scoring, and portfolio optimization. Connects to Claude Desktop, Cursor, and any MCP-compliant agent.
> 2. **Battle of the LLMs**: 10 open-source models (GPT-OSS, Qwen3-Coder, Kimi-K3, Gemma-4, Llama 3.3) trading $10,000 paper accounts autonomously every 15 minutes.
> 3. **Public Track Record**: Every prediction is timestamped and scored with Brier scores and calibration curves.
> 
> Check out the live radar desk at https://foresea.ink or connect your agent to `https://foresea.ink/mcp/`. Feedback welcome!

---

## 5. Automated Outreach & Registries (Already Built in Repo)
Run our automated registry submitter across 20+ MCP & AI Tool directories:
```bash
python scripts/pr_agent_outreach.py --targets data/pr_outreach_targets.json --send
```
