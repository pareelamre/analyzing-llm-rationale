import type { Action, Plugin, Provider, IAgentRuntime, Memory, State, HandlerCallback } from "@elizaos/core";

export const FORESEA_BASE_URL = "https://foresea.ink";

/**
 * Foresea Alpha Feed Provider: Injects live mispriced opportunities from Polymarket & Kalshi
 * directly into the agent's context window.
 */
export const foreseaFeedProvider: Provider = {
  get: async (_runtime: IAgentRuntime, _message: Memory, _state?: State) => {
    try {
      const response = await fetch(`${FORESEA_BASE_URL}/feed/latest?limit=5&min_edge=0.05`);
      if (!response.ok) return "";
      const data = await response.json();
      const signals = data.market_edge_signals || [];
      if (signals.length === 0) return "";

      const lines = signals.map((s: any, idx: number) => {
        const edge = s.edge ? (s.edge * 100).toFixed(1) : "?";
        return `${idx + 1}. [${s.platform || "Market"}] "${s.question}" -> Action: ${s.recommendation || "BUY"} (${edge}% edge vs market)`;
      });

      return `\n=== FORESEA REAL-TIME PREDICTION MARKET ALPHA ===\n${lines.join("\n")}\n================================================\n`;
    } catch {
      return "";
    }
  },
};

/**
 * Forecast Market Action: Triggers Foresea's calibrated probability model.
 */
export const forecastAction: Action = {
  name: "FORECAST_MARKET",
  similes: ["PREDICT_EVENT", "CALCULATE_PROBABILITY", "CHECK_ODDS", "FORECAST_OUTCOME"],
  description: "Get a calibrated probability forecast with evidence and rationale for any binary future event.",
  validate: async (_runtime: IAgentRuntime, message: Memory) => {
    const text = message.content?.text || "";
    return text.includes("will") || text.includes("forecast") || text.includes("chance") || text.includes("probability") || text.includes("odds");
  },
  handler: async (_runtime: IAgentRuntime, message: Memory, _state?: State, _options?: Record<string, unknown>, callback?: HandlerCallback) => {
    const question = message.content?.text || "";
    try {
      const response = await fetch(`${FORESEA_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        if (callback) callback({ text: "⚠️ Foresea forecasting API is currently unreachable." });
        return false;
      }

      const data = await response.json();
      const prob = data.predicted_probability != null ? `${(data.predicted_probability * 100).toFixed(1)}%` : "N/A";
      const answer = data.predicted_answer || "Unknown";
      const rationale = data.rationale || "";

      const reply = `🔮 **Foresea Calibrated Forecast**\n**Probability:** \`${prob}\` (${answer.toUpperCase()})\n\n**Rationale:**\n_${rationale.slice(0, 400)}_\n\n🔗 Verified on [Foresea Engine](${FORESEA_BASE_URL})`;
      if (callback) callback({ text: reply });
      return true;
    } catch (err: any) {
      if (callback) callback({ text: `❌ Error forecasting: ${err.message}` });
      return false;
    }
  },
  examples: [
    [
      { user: "{{user1}}", content: { text: "What is the probability that SpaceX lands Starship in 2026?" } },
      { user: "{{agent}}", content: { text: "Let me check Foresea's calibrated probability model...", action: "FORECAST_MARKET" } },
    ],
  ],
};

/**
 * Scan Edges Action: Queries top prediction market mispricings across Polymarket & Kalshi.
 */
export const scanEdgesAction: Action = {
  name: "SCAN_MARKET_EDGES",
  similes: ["FIND_MISPRICINGS", "CHECK_ALPHA", "SCAN_PREDICTION_MARKETS", "GET_EDGES"],
  description: "Scans Polymarket and Kalshi for mispriced contracts where Foresea model disagrees with market price.",
  validate: async (_runtime: IAgentRuntime, _message: Memory) => true,
  handler: async (_runtime: IAgentRuntime, _message: Memory, _state?: State, _options?: Record<string, unknown>, callback?: HandlerCallback) => {
    try {
      const response = await fetch(`${FORESEA_BASE_URL}/edge-board?limit=5`);
      if (!response.ok) {
        if (callback) callback({ text: "⚠️ Unable to fetch edge board." });
        return false;
      }

      const data = await response.json();
      const opps = data.edge_board || data.opportunities || [];
      if (opps.length === 0) {
        if (callback) callback({ text: "⚡ No prediction market edges meeting the threshold currently." });
        return true;
      }

      const lines = opps.slice(0, 5).map((o: any, i: number) => {
        const edge = o.edge ? `${(o.edge * 100).toFixed(1)}%` : "";
        const mkt = o.market_probability != null ? `${(o.market_probability * 100).toFixed(0)}%` : "?";
        const model = o.model_probability != null ? `${(o.model_probability * 100).toFixed(0)}%` : "?";
        return `**${i + 1}. [${o.platform || "Venue"}]** ${o.question}\n   Action: \`${o.recommendation || "BUY"}\` | Market: \`${mkt}\` → Foresea: \`${model}\` (**${edge} edge**)`;
      });

      const reply = `⚡ **Foresea Top Prediction Market Edges**\n\n${lines.join("\n\n")}\n\n📊 [Explore Live Radar Desk](${FORESEA_BASE_URL}/#radar)`;
      if (callback) callback({ text: reply });
      return true;
    } catch (err: any) {
      if (callback) callback({ text: `❌ Failed to fetch edge board: ${err.message}` });
      return false;
    }
  },
  examples: [
    [
      { user: "{{user1}}", content: { text: "Show me the biggest mispricings on Polymarket right now" } },
      { user: "{{agent}}", content: { text: "Scanning live Polymarket & Kalshi orderbooks for edge...", action: "SCAN_MARKET_EDGES" } },
    ],
  ],
};

export const foreseaPlugin: Plugin = {
  name: "foresea",
  description: "Foresea prediction market intelligence, calibrated forecasting, and Polymarket/Kalshi edge detection.",
  actions: [forecastAction, scanEdgesAction],
  providers: [foreseaFeedProvider],
  evaluators: [],
  services: [],
};

export default foreseaPlugin;
