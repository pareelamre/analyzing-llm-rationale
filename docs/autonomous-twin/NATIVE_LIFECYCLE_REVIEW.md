# Native lifecycle review and remediation

Review: `aud_c81nwJMleH`, terminal partial, four usable completed voices and one
failed voice. This records the existing review; no second audit was submitted.
Voice numbers follow the saved audit response order. Nineteen findings are
consolidated into sixteen rows. An accepted decision does not mean an outstanding
fix is complete. The code-blocking history finding is now addressed; actual private
venue/account validation and the other production release gates remain open.

Fresh local verification on 2026-10-21: full unittest suite3169, no skips, exit0;
initial native/history/emulator subset234, exit0; Ruff src/tests and diff checks
pass. The repository release-contract verifier passes; that is source/config
evidence, not a deployed runtime, IAM, rollback or performance result. Debug
OpenTelemetry export excludes fixture credential and private SDK error text.

| finding | decision | action | verification |
|---|---|---|---|
| V1/f1 major, native_lifecycle.py: ambiguous primary-subaccount binding | accepted | FIXED: historical queries select subaccount0; matched orders and fills require an actual integer0. Missing, boolean and string fields fail closed. | Native binding regression and real signed historical-read contracts pass on SQL and the Datastore emulator. |
| V1/f2 and V4/f1 major, native_lifecycle.py: pending cancelled Polymarket trades finalize too early | accepted | FIXED: pending trades keep acknowledged/partially-filled command state until final evidence is available. | Pending cancellation followed by confirmed fill reconciles without false quarantine. |
| V2/f1 blocking major and V2/f2 major, native_lifecycle.py and fixture: taker maker-wallet assumption | accepted | FIXED: taker attribution requires the owned taker order and correct token/side; only owned maker legs require the own maker wallet. | Distinct counterparty wallet accepted for taker; wrong maker wallet rejected. |
| V2/f3 and V4/f9 minor, native timestamps: proposed clock-skew allowance | rejected | Keep truthful observation times and strict lease bounds. No measured skew bound justifies inventing a later observation time. Future evidence can degrade conservatively; this accepts that availability tradeoff. | Reader compares actual creation/observation times and FillObservation rejects future occurrences; neither path releases exposure on unavailable evidence. |
| V2/f4 minor, historical query filters | accepted | FIXED: historical queries use primary subaccount0 and an immutable min_ts just before the intent creation time. Persisted cursors continue the same query. No unsupported historical order_id filter is invented. | Official Kalshi historical references and real signed-read contract tests;25-page continuation regression on both venues. |
| V3/f1 blocking major, recovery.py: legacy False becomes mark_no_order | accepted | FIXED: negative legacy lookup returns pause_and_reconcile. No production caller was found; this hardens the retained helper rather than proving an observed cash release. | Regression covers both submitting and submission_unknown with False. |
| V3/f2 major, native_lifecycle.py: every capped history retry restarts from page1 | accepted | FIXED: private context-bound checkpoints save selected rows/cursors, resume across jobs, reject stale authority and clear with lifecycle state. Explicit finite storage/page/row and original job deadline budgets remain. |25-page captures resume without rereading pages; database-restart and two-client Datastore cursor-write fencing regressions pass. Full3169 pass. |
| V4/f2 minor, native maintenance: reserved commands acquire submission state | accepted | FIXED: only post-dispatch command states enter native recovery. | Never-dispatched command remains RESERVED and performs no provider reads. |
| V4/f3 minor, native tests: real Kalshi query contract absent | accepted | FIXED: route the full native capture through venue_api.read with signed-transport fixtures for all live/historical order/fill operations. | Both venues pass the real read-catalog regression, including primary historical parameters. |
| V4/f4 minor, native Polymarket trade lookup bypasses read catalog | accepted | FIXED: batch capture reuses the existing authenticated fills catalog operation. Its required maker_address filter is pinned to the independently verified funder. No per-trade bypass or new trade endpoint remains. | Real signed-read contract regression verifies the required scoped funder and condition/time parameters. Official CLOB OpenAPI requires maker_address. |
| V4/f5 minor, native Polymarket economics: field semantics and real-shape evidence | accepted | SOFTWARE ADDRESSED; PRODUCTION EVIDENCE OPEN: cite the official Trade schema, require intent/order quantity agreement, check maker quantity and same-token taker notional, and preserve explicit actual fees. Synthetic fixtures cover multi-maker, complementary maker and nonzero-fee cases. Owner-captured field agreement remains the existing T15 production gate. | Full3169 pass. NATIVE_LIFECYCLE.md records exact field sources and uncertainty; no private venue capture or live economics claim is made. |
| V4/f6 minor, native maintenance: nonterminal jobs have no durable next observation | accepted | FIXED LOCALLY: atomically stage one delayed follow-up, reuse it on replay, retain original deadline and cap the chain at16 follow-ups. At the runtime boundary, missing/degraded/malformed native recovery or exhausted continuation persists PAUSED with holds retained. Cadence and activation remain production gates. | SQL restart, lost enqueue, early delivery, duplicate, deadline/round exhaustion and rollback tests pass; real Datastore emulator proves lifecycle/child rollback together. Seven pause/replay cases pass on SQL and Datastore; full3183 pass. |
| V4/f7 minor, native Polymarket capture:1000 sequential per-trade reads exceed a job budget | accepted | FIXED: one condition/time-filtered authenticated fill collection replaces per-ID requests. Relevant associated/owned trades are selected locally; incomplete or unlisted owned trades keep the hold. |1000 associated trades use10 paginated requests over2 jobs without rereading pages; exercised on SQL and the Datastore emulator. |
| V4/f8 minor, native tests: full suite and misleading page/action test names | accepted | FIXED: accurately named SDK/market tests, wrong-action and missing/repeated/malformed cursor regressions, restart continuation and rejected-cache refresh. | Full3169 pass with no skips; initial native/history/emulator234 pass; final economics/contract-focused119 pass. |
| V4/f10 minor, native Polymarket packet: missing YES/NO token binding | rejected | Existing production preparation selects the outcome token and compares it against signed tokenId; dispatch repeats the comparison. Recovery uses the unchanged private packet and validates its command fingerprint. venue_instrument_id identifies the market, not the outcome token. | polymarket_execution.py:201,253,284,290 retain those guards; native reader verifies prepared packet hash plus maker, market, token, action, size and price. |
| V4/f11 minor, documentation: all fill IDs claimed to include trade/order identity | accepted | FIXED: docs distinguish command-scoped native Kalshi fill IDs from composite Polymarket IDs. | NATIVE_LIFECYCLE.md matches both constructors in native_lifecycle.py. |

Additional local finding: native account binding previously accepted a Kalshi demo
host under a live scope. Native scoped account plans now reject demo hosts before
provider reads. Generic non-native demo account plans are unchanged.

The new optional WorkerJob.not_before field is part of the immutable job identity
only when present, preserving legacy hashes. Delayed jobs require compatible
dispatcher/worker versions. Local deferred work requires the existing due scan;
its cadence must fit the original deadline. None of this enables a native reader,
provisions a queue, changes a deployed schedule or authorizes an order.
