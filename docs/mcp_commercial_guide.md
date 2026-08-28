# Foresea MCP integration guide

Foresea exposes read-only forecasting and prediction-market research through a
remote Model Context Protocol (MCP) endpoint:

```text
https://foresea.ink/mcp/
```

The public endpoint is currently anonymous. Do not ask customers to supply a
`FORESEA_API_KEY` for this endpoint: individual accounts, paid access, and
OAuth-based sign-in are not available yet.

## Recommended: remote Streamable HTTP

Remote HTTP gives a customer a single hosted endpoint, without requiring a
local Python installation or granting the harness access to a local process.

### Claude Code

```bash
claude mcp add --transport http foresea https://foresea.ink/mcp/
```

### Cursor

Add this to `.cursor/mcp.json` or `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "foresea": {
      "url": "https://foresea.ink/mcp/"
    }
  }
}
```

### Windsurf

Add this to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "foresea": {
      "serverUrl": "https://foresea.ink/mcp/"
    }
  }
}
```

### Codex

Codex integrations should be distributed as a versioned plugin that declares
the remote MCP server. Until that package is published, use the MCP connection
flow provided by the particular Codex environment and point it at the URL
above.

## Local development only

The local stdio server is useful for contributors and private deployments. It
is not the recommended customer installation route.

```bash
uvx --from git+https://github.com/pareelamre/analyzing-llm-rationale.git foresea-mcp
```

Or, from a checkout with dependencies installed:

```bash
foresea-mcp --transport stdio
```

## Available capabilities

The remote server publishes tools for forecasting, market analysis, market
scanning, batch quotes, calibration/track record, edge boards, market metadata,
order books, prices, recent trades, debate, portfolio research, and the latest
feed. It also publishes resources for the track record, edge board, trending
markets, and OpenAPI schema, plus reusable forecasting and market-risk prompts.

Clients should discover the actual tool, resource, and prompt list at
connection time rather than depend on a copied inventory in this document.

## Requirements before paid distribution

Before offering paid or enterprise access, Foresea needs a dedicated protected
MCP endpoint with per-customer authentication, scoped authorization, usage
metering, quotas, revocation, and a published privacy/security policy. Market
data rights and redistribution terms must be cleared for the intended customer
segment before marketing trading or portfolio features.
