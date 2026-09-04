# Foresea MCP Server: Commercial Harness Integration Guide

Foresea provides a high-performance **Model Context Protocol (MCP)** server that equips AI agents (Claude Code, Google Antigravity, OpenAI Codex, Cursor, Windsurf, OpenHands) with **calibrated probability forecasting**, **real-time prediction market odds (Polymarket & Kalshi)**, **orderbook depth**, and **edge analysis**.

---

## 1. Quick Launch (Zero-Config)

### Using `uvx` (Recommended for Python/Fast startup)
```bash
uvx --from git+https://github.com/pareelamre/analyzing-llm-rationale.git foresea-mcp
```

### Local Stdio (Default)
```bash
python -m analyzing_llm_rationale.mcp_server --transport stdio
```

---

## 2. Configuration for Commercial Agent Harnesses

### Claude Code & Claude Desktop
Add Foresea to your `claude_desktop_config.json`:

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

### Google Antigravity
Create or add to `~/.gemini/antigravity/mcp/foresea.json`:

```json
{
  "name": "foresea",
  "command": "uvx",
  "args": ["--from", "git+https://github.com/pareelamre/analyzing-llm-rationale.git", "foresea-mcp"],
  "env": {
    "FORESEA_BASE_URL": "https://foresea.ink",
    "FORESEA_API_KEY": "YOUR_FORESEA_API_KEY"
  }
}
```

---

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

---

### Windsurf (Codeium Cascade)
Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "foresea": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/pareelamre/analyzing-llm-rationale.git", "foresea-mcp"],
      "env": {
        "FORESEA_API_KEY": "YOUR_FORESEA_API_KEY"
      }
    }
  }
}
```

---

### OpenHands / Custom AI Agent Loops (HTTP / SSE Transport)
```bash
foresea-mcp --transport sse --port 8000
```
