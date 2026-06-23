# Kite MCP Setup for Claude Desktop

## 1. Get a Kite Connect API key

1. Go to https://developers.kite.trade/ and create an app.
2. Set the redirect URL to `http://localhost:8080/login/callback` (for self-hosted) or follow the Zerodha OAuth flow.
3. Note your `API_KEY` and `API_SECRET`.

---

## 2. Build the kite-mcp-server binary

```bash
git clone https://github.com/zerodha/kite-mcp-server
cd kite-mcp-server
go build -o kite-mcp-server
```

Requires Go 1.21+. Move the binary somewhere permanent, e.g. `~/bin/kite-mcp-server`.

---

## 3. Add to Claude Desktop config

Edit `~/.config/Claude/claude_desktop_config.json` (Linux/Mac) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "kite": {
      "command": "/home/YOUR_USERNAME/bin/kite-mcp-server",
      "env": {
        "APP_MODE": "stdio",
        "KITE_API_KEY": "YOUR_API_KEY",
        "KITE_API_SECRET": "YOUR_API_SECRET"
      }
    }
  }
}
```

Restart Claude Desktop. You should see "kite" in the MCP tools panel.

---

## 4. First-time login

In any Claude Desktop conversation (or the project tabs), ask:

> "Login to Kite"

Claude will call the `login` tool, which returns a Zerodha OAuth URL. Open it in your browser, complete the 2FA, and the session token will be stored for the day. You re-login each trading day (Zerodha sessions expire at end of day).

---

## 5. Security note

- The binary only runs when Claude Desktop is open — no background process.
- Your API key never leaves your machine in stdio mode.
- To create a read-only instance (no trading), add to the env:
  `"EXCLUDED_TOOLS": "place_order,modify_order,cancel_order,place_gtt_order,modify_gtt_order,delete_gtt_order"`

---

## Available tools once connected

| Tool | Use |
|------|-----|
| `login` | OAuth login (run once per day) |
| `get_holdings` | Current stock holdings |
| `get_positions` | Intraday positions |
| `get_margins` | Available cash and margins |
| `get_quotes` | Live price for any NSE symbol |
| `search_instruments` | Look up instrument token by ticker name |
| `get_historical_data` | OHLC history |
| `place_order` | Place a buy/sell order |
| `modify_order` | Modify a pending order |
| `cancel_order` | Cancel a pending order |
| `get_orders` | Today's order list |
| `get_order_history` | Status history of a specific order |
| `place_gtt_order` | Set a Good Till Triggered (stop-loss) order |
