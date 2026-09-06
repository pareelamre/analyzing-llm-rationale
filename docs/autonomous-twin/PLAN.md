# Foresea autonomous trading twin: implementation plan

Status: proposed design; implementation has not started. Prepared 2026-09-06.
Inspected checkout: `d6b7a162ffca05649ba8246932a7c7f6ccc9d9a7` on `main`.
Revalidate symbols against the execution branch; line numbers and upstream APIs can change.

The user chose **an autonomous Foresea trading agent, without modeling an individual user**. This plan builds a real trading product. Shadow trading is a verification environment on the path to live execution.

Read this document once for architecture, then execute the bounded cards in [TASKS.md](TASKS.md) using [EXECUTOR.md](EXECUTOR.md). [progress.json](progress.json) records progress across inexpensive model sessions. A planning request authorizes writing this pack, not enabling real trading or spending capital.

## 1. Product definition and completion criteria

Foresea maintains a digital representation of an authorized trading account and its market environment. It researches opportunities, estimates outcome probabilities, simulates candidate actions against current inventory and liquidity, submits permitted orders, reconciles what actually happened, and measures whether its forecasts and execution deserve continued capital.

The twin contains six connected views:

| View | What it represents | Authoritative inputs |
| --- | --- | --- |
| Market | Contract identity, resolution rules, tradability, executable depth and fees | Venue metadata, books and status |
| Belief | Probability estimates, uncertainty, evidence, freshness and strategy version | Timestamped research and forecast ledger |
| Account | Cash, collateral, orders, fills, inventory, settlement and reservations | Authenticated venue reconciliation plus local pending commands |
| Policy | Permitted instruments, actions, spending, exposure and expiry | Owner-approved, versioned mandate |
| Execution | Intent, risk checks, submission identity, acknowledgements and recovery | Durable local commands and venue receipts |
| Counterfactual | What a candidate action or alternative strategy would do | The same captured state, simulated fills and explicit assumptions |

A twin is useful when it can answer: “What do I own, what can I lose, what might change, what am I about to do, and why?” A narrative transcript is supporting evidence; it is never the account balance or execution record.

**End-to-end acceptance:** after an authorized owner activates an expiring mandate, a worker can independently research, buy, monitor, reduce/close and reconcile positions on each enabled venue. It survives restarts and duplicate task delivery without duplicate exposure, pauses on uncertain account state, and displays independently reconciled results. A replay can reproduce every deterministic risk decision using captured inputs. Live capability and profitable strategy evidence are separate deliverables.

### Initial scope

- One Foresea strategy, one owner, dedicated venue account/subaccount or wallet. Separate real cash balances per venue; no pooled customer funds.
- Binary, fully collateralized positions, initially bought and later sold/settled. No naked positions, leverage, collateral transfers, deposits, withdrawals or automated wallet approvals.
- Both Kalshi and Polymarket adapters. Enable venues independently after their own tests and eligibility checks. Build Kalshi execution first, then Polymarket; available credentials do not determine technical correctness.
- Strategy `foresea_edge_v1`: conservatively calibrated probability versus executable price, subject to fees, depth, uncertainty and portfolio limits. A single configured research model; no default multi-model council.
- Eligible markets have an unambiguous supported settlement specification and category. Initial discovery targets liquid markets closing in 24 hours to 30 days, with no new exposure during the last hour. These are initial configuration choices, not claims of optimal strategy.
- Research refresh every 15 minutes at most; account maintenance every minute while active; submission always takes a fresh snapshot. This is a bounded, low-frequency agent.
- An operator desk showing real state and controls. Existing public benchmark boards retain explicit shadow/live provenance.

### Defer until after the release gates

User personality imitation, per-user fine-tuning, a new frontend framework, reinforcement learning, autonomous code/prompt deployment, market making, continuous WebSocket infrastructure, cross-venue arbitrage, short-horizon latency strategies, weather-specific new modeling, copy trading, multi-tenant capital management and a paid billing product. Later strategy work must compete against a frozen baseline; it must not become a prerequisite for a functioning account twin.

## 2. What exists and what must change

The following findings were read in the current source. They establish reuse opportunities, not production readiness.

| Existing component | Reuse | Required boundary/change |
| --- | --- | --- |
| `market_data.py`, `venue_api.py`, `venue_routes.py` | Venue metadata, books, history, account API coverage | Build a narrow execution capability matrix; pagination and freshness are mandatory for financial state |
| `trading.py` | Signing, SDK wiring, preview, submission, cancellation, reconciliation | Correct and fixture-test venue semantics; introduce a validated-command submission seam |
| `server.py`: trading connections | Per-connection KMS envelope encryption | Reuse for dedicated owner accounts; keep secrets inside execution/reconciliation privileges |
| `server.py`: trade runs and guardrails | Durable runs, kill switch, quote/balance/exposure checks, atomic run claim, `submission_unknown` | Extract small services; add account-wide reservations and automatic authorization; preserve manual confirmation semantics |
| `benchmark_tools.py:place_trade` | Shadow fills, risk checks, paper accounting | Keep this tool shadow-only. Never enable autonomous live trading by changing its mode flag |
| `agent_capabilities.py`, `scripts/agent_trading_tick.py` | Research loop, tools, evidence and structured forecast capture | New research mode returns proposals only; remove execution and arbitrary URL tools from the twin allowlist |
| `forecast_ledger.py`, evaluation modules | Prospective forecasts, timestamp integrity, calibration and promotion artifacts | Extend for strategy/version cohorts; existing `eligible_for_shadow_promotion` does not authorize money |
| `accounting.py`, `agent_trading_stats.py` | Accounting semantics and public metrics concepts | Real account state must come from venue facts; simulate separately and account for deposits/fees/settlement |
| `portfolio_optimizer.py` | Fractional Kelly starting point | Existing midpoint-based, independent sizing is not sufficient for execution or correlated exposure |
| `venue_streams.py` | Subscription, sequence gaps, reconnect/reset behaviors | Optional later latency improvement; it currently does not constitute a persistent account twin |
| `.github/workflows/_agent-trading-tick-reusable.yml` | Repeatable shadow experiments | Hard-pinned to shadow; retain this property |
| `.github/workflows/trading-reconcile.yml` | Existing internal reconciliation hook | Currently manual-only; create a runtime maintenance loop independent of GitHub Actions |
| `scripts/live_trader_bridge.py` | Possibly its CLI interface | Legacy direct HTTP submissions, in-process allocation, and HTTP-success-as-fill are unsuitable. Replace with a guarded service client or retire its live path |
| `frontend/trade.html`, Vite config | Existing trading interface and build pipeline | Add the twin desk without a new application framework |

### Immediate contract gaps to resolve before live work

The inspected Kalshi preview emits V2 bid/ask but passes the selected outcome price unchanged and does not include a self-trade-prevention field. Verify and correct NO-side conversion and required fields against current official contracts. The current V2 API expresses prices on the YES leg. [Kalshi Create Order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2)

The legacy bridge separately uses bearer-style requests and assumes a successful HTTP response is a fill. Its tests cover dry-run allocation, not a full exchange lifecycle. Treat it as a migration target, never as proof that autonomous execution is implemented.

Current trade-run claiming is per run. Account-wide simultaneous intents and manual orders require a shared reservation transaction. Financial snapshots currently have bounded list requests; prove completeness before using them to approve exposure. Classify these as implementation risks to test, rather than assuming every possible race is already a confirmed production incident.

## 3. Architecture decisions the executor should follow

### Keep the current stack

- Python/FastAPI for the existing public API and two **private Cloud Run worker deployments**, research and maintenance/execution, built from the same repository/image with distinct service identities.
- Existing Google Cloud Datastore for durable commands, account projections, event records, leases and budgets. Do not introduce PostgreSQL, Kafka, Redis locks or a vector database for this release.
- Cloud Scheduler creates due work once a minute. Cloud Tasks delivers bounded research and account-maintenance jobs to the private worker. Use separate queues so slow model requests cannot block reconciliation or stopping trading.
- GCS stores immutable bounded evidence/replay artifacts and archived events. DuckDB is for offline analysis and existing benchmark pipelines, not the live command authority.
- Structured Cloud Logging and existing native OpenTelemetry hooks. No paid Superlog dependency. Telemetry export failure must not change risk decisions or block reconciliation.
- GitHub Actions runs CI, fixture checks and deployment. It is not the live order scheduler.

Cloud Tasks can deliver a task more than once, so the worker must deduplicate using durable commands even when task names are deterministic. [Cloud Tasks execution model](https://docs.cloud.google.com/tasks/docs/dual-overview)

### Modules to add

Under `src/analyzing_llm_rationale/`:

| Module/package | Responsibility |
| --- | --- |
| `trading_control/service.py`, `store.py`, `policy.py` | Extract only the existing shared execution, persistence and guardrail seams needed by both manual routes and the twin |
| `twin/models.py` | Versioned typed records and canonical IDs |
| `twin/store.py` | Atomic event append, projections, reservations, command claims and leases |
| `twin/market.py` | Tradable instrument registry, rules, depth snapshots, capability metadata |
| `twin/account.py` | Venue-normalized account snapshots and completeness checks |
| `twin/reconcile.py` | Incremental/complete reconciliation, ambiguous submissions and divergence |
| `twin/research.py` | Budgeted evidence/forecast proposal generation |
| `twin/budget.py`, `mandates.py` | Atomic research allowances and owner-approved authority |
| `twin/strategy.py`, `risk.py` | Deterministic candidate scoring, sizing, exposure and exit decisions |
| `twin/simulator.py`, `replay.py` | Simulated venue adapter and causal replay |
| `twin/execution.py` | Mandate authorization, durable order lifecycle and venue submission |
| `twin/scheduler.py`, `worker.py` | Due work, task delivery, maintenance, startup recovery and OIDC auth |
| `twin/routes.py` | Owner-facing control/read APIs |
| `twin/evaluation.py` | Strategy evidence and readiness artifacts |

Do not create empty abstractions for all these modules in one commit. Add each when its task needs it. Inject clocks, stores, venue adapters and model clients so tests need no credentials. Never import the entire web server to run a worker.

### Two independent loops

1. **Research:** select eligible changed markets, capture evidence, request a probability proposal, validate it, record the forecast, emit a candidate. A candidate grants no execution authority.
2. **Account maintenance/execution:** reconcile, evaluate exits, inspect valid candidates, compute fees and risk, reserve exposure, persist a command, authorize, submit, reconcile and publish state. This loop continues when the LLM is unavailable or its budget is exhausted.

Holding an account lock during a model call or HTTP request is prohibited. Short database transactions surround state changes; fencing tokens prevent a stale worker from progressing a command. Network calls occur outside transactions.

## 4. Data contracts

All durable records contain `schema_version`, UTC timestamps, an owning account namespace and stable IDs. Use `Decimal` internally and decimal strings at storage/API boundaries for money, price and quantity. Reject NaN, infinity, negative quantities and unsupported precision. Never use a title or slug alone as an execution identifier.

| Record | Required fields beyond common metadata |
| --- | --- |
| `AccountScope` | `owner_id`, `venue`, immutable `venue_account_ref`, `environment`, currency/collateral asset, `connection_ref`, `account_epoch` |
| `Instrument` | venue ID/ticker, condition ID where applicable, YES/NO token IDs, settlement-spec hash, category, event/cluster IDs, tick/min size, fee/capability version, status and close/resolution timestamps |
| `MarketSnapshot` | instrument ID, venue timestamp, receive time, depth by actual outcome, sequence/generation, source, fee snapshot, `complete`, `stale_after` |
| `AccountSnapshot` | cash available/total, collateral, inventory, open orders, fill cursor, settlement cursor, local reservations, completeness and divergence status, generation |
| `Forecast` | instrument, `p_yes_raw`, nullable `p_yes_calibrated`, `calibration_status`, uncertainty method/range or `unknown`, evidence IDs, `as_of`, `expires_at`, model/prompt/strategy hashes, calibration version, prospective provenance |
| `Proposal` | forecast/snapshot refs, `BUY_YES`/`BUY_NO`/`HOLD`/`PASS`, reason codes and citations; optional preferred limit. Final size comes from deterministic code |
| `TradeIntent` | immutable instrument/action/quantity/limit/TIF, forecast ref or deterministic exit reason, policy/account versions, fee/slippage allowance, expiry, order fingerprint |
| `Mandate` | owner approval, immutable version/hash, account/venue/strategy scope, allowed categories/actions, numeric budgets, permitted modes, start/expiry, revocation, model/config hash, exit permissions |
| `Reservation` | intent/command ID, cash and worst-case loss reserved, account revision, status and reconciliation linkage |
| `ExecutionCommand` | intent hash, authority kind/ref, client ID, venue identity/hash if available, state, attempt info, lease/fence, request fingerprint, sanitized receipt |
| `TwinEvent` | event ID, account sequence, type, `occurred_at`, `observed_at`, command/venue IDs, payload hash, previous projection version |
| `DecisionRecord` | candidate set, explicit reason for trade/hold/pass, inputs and versions, risk result, cost, resulting command ID if any |
| `BudgetUsage` | strategy/account/day key, reserved and actual model tokens/cost, external request counts, timestamps |

Store large evidence in GCS by content hash, with short sanitized excerpts only where retention rights allow. Database records hold references. Never persist private keys, decrypted credentials or authenticated response headers in evidence, task payloads, traces or public feeds.

### Persistence semantics

- Account aggregate is the serialization point for authorization, cumulative limits and revision. Atomically insert a unique event/command and update its projection/reservation in the same account-scoped transaction. External order IDs and fill IDs are unique within venue/account scope.
- Use key reads/ancestor-scoped access with verified Datastore consistency. Do not approve risk using an eventually consistent global query. Keep transactions small; prove emulator concurrency behavior in T04.
- Local `submitting`/`submission_unknown` reservations survive expiry and worker crashes. Only confirmed no-order/terminal reconciliation releases unfilled exposure. A lease expiring does not mean the exchange rejected an order.
- Inbox deduplication handles repeated venue events; an outbox persists work before queue delivery. Mark outbox delivery after successful enqueue, and tolerate duplicate enqueue.
- Snapshot replacement requires a complete fetch generation; a partially fetched portfolio never replaces a complete one. If a venue exposes no consistent snapshot token, fence reads with timestamps and order/fill cursors and reject a generation that changes materially during collection.
- Account binding changes require pause, reconciliation and a new epoch. Never mix a new wallet's inventory with the previous wallet's ledger.

## 5. Market, portfolio and strategy behavior

### Market twin

Capture resolution source, precise condition, timezone, close time and dispute/finality state. Reject ambiguous, already resolved, suspended, unknown-type or expired contracts. Market similarity is not outcome equivalence: group related markets for conservative risk, but never net them as guaranteed offsets unless identical settlement is established.

Derive execution from actual ask/bid depth and verified contract semantics. Missing prices, fees or balances block new exposure. `holders: null` is empty optional analytics; it cannot justify treating a missing account inventory response as an empty portfolio. Exhaust pagination or mark the result incomplete.

Use REST snapshots for the initial runtime. If streams are added later, sequence gaps invalidate the cached book until a replacement snapshot is applied. A display midpoint is never an executable order price.

### Forecasting and evidence

Freeze `as_of` before research. Record publication time and retrieval time independently. Save supporting and contrary evidence; external text is untrusted data and cannot change tools, policies or instruction hierarchy. The LLM receives no credentials and cannot choose arbitrary network destinations or task identities.

Validate a strict proposal schema. One bounded schema-repair attempt is allowed; further invalid output becomes `PASS_INVALID_PROPOSAL`. Do not infer a trade from a prose thesis. Unknown uncertainty is explicit; a model-written confidence score alone does not establish calibration.

Calibrate using prospective historical observations available before the evaluated decision, with a frozen, versioned method. Research initially supplies only raw probability; deterministic calibration code fills the calibrated value and status. An LLM-supplied calibrated value is never authoritative. If calibration is insufficient, remain in research/shadow. Reuse existing ledger/cohort validation, including rejection of post-resolution forecasts.

To keep the first implementation concrete, `calibration_v1` uses ten fixed raw-probability bins, stratified by the frozen model/prompt and approved category family. Select the earliest eligible prospective forecast per instrument, then at most one instrument per canonical resolution-event cluster using a predeclared stable ID ordering independent of outcomes. Never include observations whose resolution was not known at calibration time. Within a bin, report the Laplace-smoothed frequency `(wins + 1) / (n + 2)` and a 95% Wilson interval on `wins/n`. Require at least 30 independent eligible observations in a bin for live sizing; otherwise mark calibration insufficient and allow only diagnostic shadow candidates. For BUY YES use the lower interval bound, and for BUY NO use one minus the upper bound as conservative `p`. The model's self-reported confidence cannot replace this interval. Use fixed bins and this simple auditable baseline before considering a fitted calibrator. Save exact training IDs, cutoff, sample rule and hash; the sample requirement can exceed G2's overall minimum.

### Deterministic entry and sizing

For a purchased outcome with conservative probability `p`, executable price `a`, estimated per-share fee `c` and adverse-fill allowance `s`, the screening edge is `p - a - c - s`. Estimate costs for the proposed quantity; re-evaluate after depth/rounding changes. NO uses the complement of the calibrated YES probability, not a second contradictory forecast.

An initial fractional-Kelly candidate can use `max(0, (p-a)/(1-a))`; clip it by configured fraction and all cash/loss limits. This is only a starting size. The final order must pass a worst-case cash-flow/scenario calculation including fees, current inventory, open orders and pending reservations. Do not copy the existing optimizer's illustrative annualized growth estimate into live performance claims.

Use conservative additive cluster exposure for related events. Unknown correlation defaults to the broader configured cluster, not zero correlation. Currency/collateral on the other venue cannot fund this order. Unsupported margin/negative-risk combinations are rejected.

### Versioned initial policy defaults

These values are engineering starting points for shadow experiments, not financial recommendations or automatic live authorization. The owner must enter absolute live limits and approve the resulting mandate; absent/zero budgets disable new exposure.

| Setting | Initial proposal |
| --- | --- |
| Mode | `shadow`; live false by default |
| Strategy | `foresea_edge_v1`, one model selected from existing provider config |
| New positions | At most 1 per research cycle; at most 5 open instruments |
| Kelly fraction | 0.10, subordinate to every hard limit |
| Per order / instrument / correlated cluster | At most 0.5% / 1% / 3% of allocated bankroll, also limited by owner-entered absolute USD caps |
| Total open worst-case loss | At most 5% of allocated bankroll, plus explicit absolute ceiling |
| Trailing 24h risk additions | At most 5% of allocated bankroll and explicit USD ceiling; separate from realized losses |
| Loss circuit breakers | Owner-entered daily USD loss and peak-equity drawdown limits; required for live |
| Net edge floor | 0.05 probability units after modeled costs; tune only by out-of-sample evidence |
| Freshness | Execution book <=10 seconds, account reconciliation <=60 seconds, candidate <=15 minutes; reject implausible future timestamps |
| Price movement | Max 1 percentage point adverse move since approved intent; final fees/depth still required |
| Mandate lifetime | Up to 24 hours for initial live pilot; never self-renew |
| Immediate execution | Prefer bounded-price immediate orders where verified. Resting orders remain disabled in MVP |

Small capital may make minimum venue order sizes incompatible with caps. Return a documented `PASS_BELOW_MINIMUM`, rather than rounding up or silently enlarging the budget. HOLD/PASS is a successful decision.

### Position lifecycle and valuation

- Reconcile positions before research on startup and on every maintenance tick.
- Evaluate deterministic exits for policy expiry, maximum holding time, market-rule changes, drawdown and invalidated evidence; allow discretionary exits only from valid, versioned proposals.
- A close uses confirmed available inventory, accounting for existing sell orders. No shorting or position flipping through an oversized close. Reduce-only is proven by the resulting position, not trusted from a request flag.
- Partial fills alter inventory and fees incrementally. Cancellation is a request until acknowledged; a fill racing a cancel must still be booked exactly once.
- Separate available cash, total cash, collateral, reserved cash, cost basis, executable liquidation value, realized/unrealized P&L and net strategy P&L after inference/data costs. Do not subtract venue reservations twice when available cash already excludes them.
- Track deposits, withdrawals performed externally, transfers, manual trades and settlement payouts as separate facts. Unexpected external activity pauses new exposure until reconciled. Returns exclude external cash flows.
- Resolution is not always immediate cash availability. Track pending settlement/finality and redemption separately; no automatic approval or redemption transaction in this release. Revised outcomes/failed settlement confirmations produce corrective events, never silent history edits.

Core replay invariants are explicit: cash equals opening cash plus external inflows, confirmed sale proceeds and credited payouts, minus purchase costs, fees and external outflows; inventory equals acquired minus sold/settled quantity; cumulative fills never exceed the accepted order size absent an explicit venue correction; and available inventory excludes pending sells. Reservations restrict spending but are not expenses or realized losses. Compare these identities to venue totals after every complete reconciliation. Do not write a balancing adjustment merely to erase unexplained drift.

## 6. Authorization and order lifecycle

The owner approves a mandate once. Within that mandate the agent acts autonomously; per-order human confirmation is not required. The human-facing manual trading API retains its existing confirmation requirements.

Add an internal `submit_authorized_command` service. It accepts only a persisted command whose authorization is verified server-side: either a manual approval record or a current mandate. Never fabricate `PLACE REAL ORDER` on behalf of the agent or expose a public `skip_confirmation` parameter. The benchmark `place_trade` remains shadow-only.

Command states:

`proposed -> validated -> reserved -> submitting -> acknowledged -> partially_filled -> filled`

Other transitions: validation to `blocked`/`expired`; submitting to `submission_unknown` or confirmed rejection; open orders to `cancel_requested` then confirmed `cancelled`/`filled`; completed fills to position/settlement events. A filled order is not necessarily a settled position. Terminal states cannot regress from delayed events.

Submission sequence:

1. Validate command identity, current mandate and account binding.
2. Capture fresh complete account/book snapshots and final cost estimates.
3. In one transaction verify current account/policy revision, duplicate fingerprint and all budget limits; reserve cash/risk and persist the command/outbox.
4. Claim the command with a fenced lease. Recheck expiry, revocation, kill switch and maximum snapshot age immediately before dispatch. Persist the exact venue submission identity before network transmission where supported.
5. Submit once. Persist receipt, normalize acknowledgement independently of fills, and enqueue reconciliation.
6. Timeout or crash after potential submission => `submission_unknown`. Search the venue using stable IDs/hash and reconcile. Never generate a new order just because the response was lost.

There is no atomic transaction spanning Datastore and an exchange. The target guarantee is durable at-most-once intent with conservative recovery, not magical exactly-once HTTP. If identity lookup cannot prove an order absent, keep the reservation and pause that account for operator resolution. A lease token cannot cancel a request already in flight.

Kill switches: global, account, strategy and venue. Pause stops new exposure and requests cancellation of resting orders if any exist. It continues reconciliation. Verified reductions are permitted only if the mandate explicitly grants them and fresh state proves they cannot increase exposure. An emergency hard-disable blocks all submissions but still permits read-only reconciliation. Expiry permits only the explicit post-expiry maintenance/cancel policy, not discretionary new trades.

All channels capable of trading the same account, including manual routes, must share account serialization/reservations. Prefer dedicated pilot accounts to simplify external activity. Per-run locks or a single Cloud Run instance are not substitutes for this guarantee.

## 7. Venue readiness requirements

Create a capability matrix with source/version, fixtures and live-verification status per operation. Only capabilities necessary to the current strategy need integration; full endpoint coverage is not a release goal.

- Both venues: identity, balance, complete positions/orders/fills, metadata/rules, fees, books, bounded-price submission, stable identity lookup, single-order cancel and settlement status. Test pagination, rate limits, token expiry, malformed shapes, precision and timeouts.
- Kalshi: verify the four BUY/SELL x YES/NO conversions, fixed-point rounding, current authentication, immediate-order behavior and cancellation/reconciliation contracts. A successful response is an acknowledgement, not proof of a full fill.
- Polymarket: verify the installed SDK's signing flow, wallet/funder/signature type, token/outcome mapping, precision, fee/collateral configuration, allowance readiness and order identity persistence. Do not assume a generic IOC label maps to its supported order types. [Polymarket order documentation](https://docs.polymarket.com/trading/place-orders)
- Eligibility is checked for the actual execution environment and authorized account. A server IP check alone does not prove account eligibility. Current Foresea deployment is in `us-central1`; verify venue eligibility before enabling it there, and do not silently route around a restriction. [Polymarket geographic restrictions](https://docs.polymarket.com/api-reference/geoblock)
- Venue heartbeat-based cancellation must be separately verified before any future resting-order mode. It may affect all orders on an account and is not the same as WebSocket PING/PONG. [Polymarket heartbeat](https://docs.polymarket.com/api-reference/trade/send-heartbeat)

No real orders are sent by unit tests, CI, ordinary deploy smoke tests or this planning task. A demo environment or synthetic venue validates mechanics; it does not prove liquidity or production profitability.

## 8. Runtime, cost and operator experience

### Runtime

Both workers are IAM-private. Verify Google-issued OIDC issuer, exact audience and allowed service identity; queue headers are not authorization. Research task payloads contain IDs only. The research role cannot read execution credentials, write the trading Datastore, or invoke live submission. Maintenance owns durable account, budget and mandate mutations. Do not assume Datastore IAM can isolate entity kinds within the same database.

Expose role-scoped handlers: maintenance owns `/internal/twin/dispatch` and `/internal/twin/maintain`; research owns `/internal/twin/research`. The research identity may call only maintenance's narrowly validated `/internal/twin/research-jobs/{id}` read/claim/result interfaces. Maintenance verifies the registered assignment, reserves provider budget before granting a claim, and accepts only a typed research result for that assignment. It never accepts authorization, capital or account mutation fields from research. Research has access only to its model credentials and public evidence artifacts, using a separate key/bucket policy where necessary. Handler-level service-identity checks supplement Cloud Run invoker IAM. The infrastructure task must demonstrate denied credential access, denied direct trading-store access and denied execution-handler access.

Start with one dispatch at a time per account and a small global research concurrency cap. Configure request/task deadlines below the platform limit, bounded exponential backoff for safe reads, a retry budget, stalled-command alerts and a recovery scan. Retry research on a new cycle only when no finished proposal exists for its idempotency key. Never retry an entire side-effecting loop.

On deploy/restart: acquire a new worker generation, recover outstanding commands, reconcile all active accounts, mark readiness, then allow new candidates. Runtime health and trading readiness are distinct. An API can be healthy while trading is paused.

### Keep model and engineering costs bounded

The implementation model and the deployed research model are separate choices. Use the user's cheaper coding model for the task cards; this plan requires no expensive-model subscription, price estimate or automatic escalation.

Runtime default shadow budget: at most 3 researched candidates per cycle, 8 evidence/tool calls per candidate, one schema repair, and a configurable daily call/token ceiling. Share cached public evidence across candidate evaluation, deduplicate by source hash, and research again only after meaningful price/evidence change or forecast expiry. Use the existing provider abstraction; avoid a second LLM SDK stack.

Required live configuration includes an approved daily model/data spending ceiling and a current per-provider cost table. Atomically reserve the worst permitted request cost before calling the model; reconcile usage afterward. Unknown pricing blocks paid live research. If configured zero-cost hosted capacity is used, still enforce tokens, time and rate limits. Price changes invalidate the estimate until refreshed.

Daily cost estimate: `cycles x researched_candidates x (input_tokens x input_rate + output_tokens x output_rate) + data_calls + infrastructure`. Record actual cost per valid forecast, per executed trade and per independent resolved market. No-data-change cycles should need no LLM call. Exceeding a budget suspends research, never maintenance/reconciliation.

### Operator desk

Extend `frontend/trade.html`, using modular JS under `frontend/` where appropriate. Show:

- Mode and authority: shadow/live, mandate expiry, allocated budget and kill-switch state.
- Account truth: last complete reconciliation, discrepancies, available/reserved funds, inventory and worst-case loss by market/cluster.
- Decisions: evidence, calibrated probability, executable price, costs, action or reason for PASS, and strategy version.
- Execution: separate proposed, sent, acknowledged, partial, filled, cancelled and unknown states.
- Performance: realized/unrealized/net P&L, costs, drawdown, calibration, sample sizes, execution slippage, and shadow/live provenance.
- Controls: approve/revoke mandate, pause new exposure, request cancel, and explicitly approved reduce-only operation. No generic “resume” that silently expands limits or renews authorization.

Use existing session authorization and ownership checks. Public endpoints and anonymous MCP retain read-only/public-safe views. No private account inventory, mandate, evidence or signer access through public agent tools. Build with `npm run frontend:build`; inspect rendered controls and generated output in `static/`.

## 9. Evaluation and release gates

### Separate four questions

1. Is the account representation correct? Reconciliation and ledger invariants.
2. Does the execution service obey authority and recover safely? Fault injection and concurrency tests.
3. Do forecasts improve on a market baseline? Prospective, event-clustered evaluation.
4. Does the strategy earn after execution and research costs? Out-of-sample replay and forward shadow/live measurement.

Replay uses both occurrence and observation time; evidence/labels arriving later are invisible to earlier decisions. Split by event cluster and time so related contracts and repeated snapshots cannot leak across train/test. Fit calibration only on prior data. Count independent resolved markets, not repeated predictions. Freeze strategy/fees/prompts before forward evaluation; use champions/challengers without automatic promotion.

Simulator assumptions are explicit: buy against ask depth, sell against bid depth, fees per venue/size, latency/adverse selection, partial/no fill, cancelled remainder and settlement delay. No fills at midpoints. With no historical depth, report unsupported execution assumptions and sensitivity bands; do not manufacture tick-level backtests from daily prices.

| Gate | Evidence required | Unlocks |
| --- | --- | --- |
| G0: correctness | All task-specific invariants, fixture contracts, full tests/lint, two-process reservation race, duplicate delivery, crash/timeout/cancel-fill races, auth isolation | Staging shadow runtime |
| G1: mechanics | 7 consecutive days of forward shadow on captured live inputs; zero unexplained ledger divergence or duplicate simulated command; all fault drills pass; stale data halts exposure | Technical live-readiness report only |
| G2: strategy | Existing prospective cohort checks plus strategy-specific out-of-sample execution/cost report; initial target >=100 independent resolved markets and >=30 completed shadow trades; positive lower bound for baseline skill and conservative net result under the declared evaluation method | Eligible for owner review, never auto-live |
| G3: bounded live pilot | G0-G2 current for exact code/config; operator enters capital/budgets, verifies account/venue eligibility and credentials, reviews IAM/recovery evidence, activates expiring mandate | Low-capital autonomous trading on individually cleared venues |
| G4: measured expansion | Pilot has no unresolved accounting/control incident, reviewed slippage/costs and drawdown within approved bounds, sufficient independent evidence; owner approves revised limits | Larger mandate or additional strategy scope |

The G2 sample targets are evidence collection thresholds, not proof of returns or a reason to weaken checks. Existing historical data counts only if provenance/version compatibility is demonstrated. If evidence is insufficient or unprofitable, implementation may be finished while live activation remains ineligible. Never relax gates just to complete a task.

Publish a versioned `twin_readiness` artifact containing exact source/config hashes, verification timestamps, per-gate evidence, missing capabilities and expiry. It fails closed when stale, malformed or inconsistent, following the existing evaluation artifact pattern. Never rewrite `eligible_for_shadow_promotion` to mean live-approved.

## 10. Delivery and cost-aware sequencing

| Milestone | Tasks | Concrete demonstration |
| --- | --- | --- |
| M0: reliable foundations | T00-T06 | Correct venue fixtures, complete account twin, shared manual execution boundary |
| M1: autonomous shadow twin | T07-T12 | Research -> deterministic decision -> realistic shadow fill -> reconciled account -> replay |
| M2: autonomous execution capability | T13-T16 | Authorized fake/demo orders survive faults and respect immutable mandates |
| M3: operated service | T17-T20 | Private runtime, operator desk, telemetry, release evidence and deployment runbook |
| M4: evaluated release | T21-T22 | Forward trial, strategy evidence and owner-approved bounded live pilot when eligible |

Execute one card per focused model session; split the marked larger cards into their listed substeps. Expect roughly 25-40 implementation/review sessions depending on discovered baseline failures, plus wall-clock time for forward evaluation. This is a scope estimate, not a completion date or token-price promise. The first useful deliverable is M1; live-capability work must not be substituted by polishing benchmark UI.

Critical review checkpoints are venue semantics (T01), account concurrency (T04), authority/submission recovery (T14-T15), and release evidence (T20/T22). A second focused review pass with the cheaper model and executable tests is required. Do not automatically invoke an expensive model; record a narrow unresolved question if tests or official specifications cannot settle it.

The following choices require operator input only when their dependent gate is reached: dedicated account identifiers, actual eligible venue/environment, capital and loss limits, model/data budget, approved research model, and live activation. Build all reversible code, tests, UI and staging evidence first. Ordinary implementation tasks do not need repeated product clarification.


