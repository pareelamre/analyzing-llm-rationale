# Edge Board agent workflow

The scheduled agent in `scripts/agent_trading_tick.py` receives its candidate
quotes and portfolio as before. Before inference, the system prepares research
for the first two distinct valid candidates (new candidates, then held markets):
up to three levels per orderbook side, resolution criteria when supplied by the
quote, and two relevant source snippets with citation URLs and publication dates
when available. Other candidates remain available through the normal tools.
Kalshi ladders are explicitly labeled as bids with their price units; derived
buy prices are dollar fractions and must be rechecked before execution.

Research uses the existing venue adapters and news sources. Query planning,
article summaries and embeddings are disabled, so this preparation makes zero
model calls. Missing evidence/books are explicitly marked; an empty search does
not establish that an event is unlikely. Snippets and criteria are untrusted
reference material. Publication dates must be checked before treating a source
as current evidence. A Polymarket book requires the quote's numeric YES token.
Preparation stops waiting after 20 seconds and marks unfinished packets
unavailable. At most two daemon workers remain active, including across retries;
already-started source requests can finish in the background without delaying
process exit. The ordinary research retry remains active when all packets lack
usable books and evidence. Malformed packets and source entries are skipped.

The agent can return its probability and complete thesis immediately, or use
targeted follow-up tools for missing information and guarded paper execution.
On this internal path, `web_search` also uses model-free snippets, and nested
`forecast`/`search_evidence` pipelines are unavailable. The existing risk guards,
Kelly sizing requirement, paper-only execution and thesis reconciliation remain
in force. Public `/agent/analyze` callers cannot supply the internal prefetch
attribute through JSON and retain their existing tool behavior.

## Calls and measurement

`MAX_TOOL_STEPS` defaults to **4** for the scheduled worker. Each turn can produce
a tool call or a final answer; the loop allows one additional finalization call
when the allowance is exhausted. A successful invocation therefore uses **1–5
logical orchestration calls**, depending on research gaps and execution. The
environment override remains available. Provider transport retries and a failed
worker invocation retried by its existing recovery policy can cause additional
billable attempts; this is not a hard provider billing limit.

`AgentReport.tool_loop_model_calls`, the durable cycle transcript's `model_calls`
field and the worker completion log record logical turns. The `agent.model.turns` metric also records them when the deployment has
configured an OpenTelemetry meter provider. Prefetched packets carry
`initiated_by: system_prefetch` in the durable tool transcript and are excluded
from the cycle's tool-step count. `agent.research.prefetches` and research spans
record preparation activity. The default bootstrap keeps local diagnostics;
remote metric/log export requires deployment-side configuration. Existing
provider telemetry records transport attempts and token usage separately.

This change reduces the turn allowance from the previous default of eight
(twelve for flash/MiniMax), without establishing measured cost or forecast-quality
improvements. Check completed cycles' actual turns, token usage, publication
quality and guarded execution outcomes before assigning a monthly API budget.
It does not restore a discontinued provider key, configure a replacement model,
alter the agent roster, or deploy a schedule.
