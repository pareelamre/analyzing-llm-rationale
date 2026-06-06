# Launch / distribution post drafts — Foresea for agents

The hook that almost no other MCP server has: **a forecasting tool that publishes
whether it's actually right** (live, resolved track record). Lead with that.

URL: https://foresea.ink · MCP: https://foresea.ink/mcp/ · repo: https://github.com/pareelamre/analyzing-llm-rationale

---

## Show HN

**Title:** Show HN: Foresea – an MCP server that forecasts any question, with a live track record

**Body:**

Foresea is a remote MCP server that gives AI agents calibrated probabilities for
resolvable future events — with evidence, a written rationale, and the edge vs
prediction-market prices (Polymarket/Kalshi).

The part I care about: it publishes a **resolved track record**. Every forecast on
a live market is logged before resolution and scored when the market settles, so
you can see accuracy, Brier score, calibration, and skill-vs-market by horizon —
no cherry-picking. There's an "edge board" that shows where the model disagrees
with the market *and* whether disagreements that size have historically paid.

It's anonymous and zero-install — point any MCP client at the URL:

    claude mcp add --transport http foresea https://foresea.ink/mcp/

Tools: `foresea_forecast`, `foresea_analyze_market`, `foresea_scan_markets`,
`foresea_edge_board`, `foresea_track_record`. Default model is gpt-oss-120b (SCADS);
there's also a multi-model paper-trading comparison (gpt-oss vs Gemma vs Kimi).

Honest status: it's early — the live track record is still accumulating resolutions,
so the "proven edge" numbers are mostly "not yet proven." That's shown honestly
rather than hidden. Would love feedback on the calibration methodology and the
edge-board framing.

---

## X / Twitter thread

1/ Most AI tools claim they're smart. Foresea publishes whether it's *right*.

It's an MCP server that forecasts any resolvable question — with evidence, a
rationale, and the edge vs Polymarket/Kalshi. And it scores itself on resolved
markets, in public. 🧵

2/ Add it to any agent in one line — anonymous, no key:

   claude mcp add --transport http foresea https://foresea.ink/mcp/

Tools: forecast · analyze_market · scan_markets · edge_board · track_record

3/ The edge board shows where our fair value disagrees with the market price —
and tags each gap with whether disagreements *that size* have historically paid.
A forecast you can audit, not just trust.

4/ It even compares models head-to-head (gpt-oss-120b vs Gemma vs Kimi) on the
same resolved markets — paper-traded, no cherry-picking.

5/ Early + honest: the live record is still accumulating, so most edges read
"unproven" for now. That's the point — the scoreboard is real.
Try it: https://foresea.ink/mcp/ · https://foresea.ink

---

## r/mcp (and r/LocalLLaMA, lightly adapted)

**Title:** I built an MCP server that forecasts any question and publishes its own track record

**Body:**

Sharing a remote MCP server I've been working on: **Foresea** (https://foresea.ink/mcp/).

It gives your agent calibrated probabilities for resolvable future events — with
evidence + rationale, and the model-vs-market edge against Polymarket/Kalshi. Five
tools: `foresea_forecast`, `foresea_analyze_market`, `foresea_scan_markets`,
`foresea_edge_board`, `foresea_track_record`.

What makes it different from "ask an LLM to guess": it **logs forecasts before
resolution and scores them after**, so there's a public, resolved track record
(accuracy, Brier, calibration/ECE, skill-vs-market by horizon). The edge board
surfaces current disagreements *and* whether gaps that size have actually paid.

Anonymous + zero install:

    { "mcpServers": { "foresea": { "url": "https://foresea.ink/mcp/" } } }

It's listed on the official MCP registry (`ink.foresea/forecasting`) and Smithery.
Early days — track record is still filling in — but the plumbing's solid and it's
free to hammer. Feedback on the tools / calibration welcome. Repo:
https://github.com/pareelamre/analyzing-llm-rationale

---

## Where to post (order of leverage)
1. **r/mcp**, **MCP Discord** (#showcase), **PulseMCP / Glama** "new servers".
2. **Show HN** (Tue–Thu morning ET). One strong comment explaining the track-record angle.
3. **X** thread; tag the MCP / agent-tooling accounts.
4. **r/LocalLLaMA** (lead with the gpt-oss/Gemma/Kimi comparison angle there).
5. Framework tool catalogs: LangChain tools, Composio, LlamaIndex.

Keep the framing honest ("early, track record still accumulating") — credibility
is the product; overclaiming on a forecasting tool is the one unrecoverable mistake.
