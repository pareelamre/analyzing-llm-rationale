# @foresea/plugin-eliza 🌊

The official **ElizaOS (ai16z)** plugin for **Foresea** — the intelligence & calibration layer for prediction markets.

Give your ElizaOS agents real-time prediction market super-powers:
- 🔮 **Calibrated Probability Forecasts**: Ask any question and receive a calibrated YES/NO confidence rating with cited news evidence.
- ⚡ **Live Polymarket & Kalshi Edge Detection**: Scan orderbooks for model-vs-market pricing gaps.
- 📡 **Alpha Context Provider**: Automatically injects live mispriced market opportunities into your agent's context.

---

## 📦 Installation

```bash
pnpm add @foresea/plugin-eliza
# or
npm install @foresea/plugin-eliza
```

---

## 🚀 Quickstart

### 1. Register Plugin in Eliza Runtime
```typescript
import { foreseaPlugin } from "@foresea/plugin-eliza";

export default {
  // ...
  plugins: [foreseaPlugin],
};
```

### 2. Or use the Included Character:
```bash
pnpm start --character characters/foresea_oracle.character.json
```

---

## 🛠️ Actions & Capabilities

| Action | Description | Example Trigger |
| :--- | :--- | :--- |
| `FORECAST_MARKET` | Evaluates evidence and generates a calibrated probability score. | *"Will SpaceX complete a Starship orbital flight by December?"* |
| `SCAN_MARKET_EDGES` | Scans Polymarket and Kalshi for mispriced opportunities where AI fair value diverges from market price. | *"Show me the biggest mispricings on Polymarket right now"* |

---

## 🌐 Public Endpoints
- **Streamable MCP Server:** `https://foresea.ink/mcp/`
- **Web App / Radar Desk:** `https://foresea.ink`
- **API Documentation:** `https://foresea.ink/docs`
