# Foresea twin: executable task cards

Read [PLAN.md](PLAN.md) for binding architecture and [EXECUTOR.md](EXECUTOR.md) for the work protocol. All tasks are initially **pending**. Current status lives in [progress.json](progress.json), not in claims made by a previous model.

Paths below are repository-relative. `src/` module paths refer to `src/analyzing_llm_rationale/`. Listed new files are proposed implementation targets, not existing capabilities. Test modules listed under each task must be created where absent. Do not run a nonexistent test and report success; add meaningful cases first.

Every card requires its focused verification, `git diff --check`, and the shared full-test/lint gate before committing. Assertions must verify economic/state behavior rather than merely mirror the function's implementation. No test sends real orders or uses private credentials.

## T00 — Establish the execution baseline

**Depends on:** none. **Size:** S. **Outcome:** known checkout, source map and reproducible test baseline.

Read `AGENTS.md`, this plan, `pyproject.toml`, `package.json`, `vite.config.mjs` and the named trading workflow files. Check branch, remote, worktree status and upstream commits. The planning checkout had unrelated untracked `tmp_glm.sqlite`; leave it untouched. Start a `codex/` worktree from the latest verified main when implementation is requested.

Run the required Graphify query, locate all live submission callers with `rg`, and record module symbols and existing tests in `docs/autonomous-twin/BASELINE.md`. Inspect the current venue contract docs and record retrieval dates without downloading secrets. Capture `py -m unittest discover -s tests` and `py -m ruff check src tests` output and exit codes. Record pre-existing failures individually rather than silently changing them.

**Done when:** baseline report identifies the real execution paths, test commands and the effective source commit; no application behavior changed. Update only the report and progress entry.

## T01 — Correct and lock venue execution contracts

**Depends on:** T00. **Size:** L, split into Kalshi then Polymarket. **Files:** `trading.py`, `tests/test_trading.py`, `tests/test_trading_byo.py`, new `tests/fixtures/twin/venues/`, `docs/autonomous-twin/VENUE_CAPABILITIES.md`.

For Kalshi, fixture-test selected-outcome price semantics across all four actions. At a selected-outcome price of 0.30, expected V2 book direction/price is BUY YES -> bid/0.30, SELL YES -> ask/0.30, BUY NO -> ask/0.70, SELL NO -> bid/0.70. Keep cash-risk accounting in the selected outcome's units. Confirm required fields, precision, authentication and the current response/cancel shape using the official contract linked in PLAN. Add the missing transformations/fields if still missing on the implementation branch.

For Polymarket, verify the installed SDK version and public signatures, token identity, BUY amount versus SELL share units, bounded-price immediate behavior, ticks/minimums, neg-risk restrictions and fees. Add an adapter operation that prepares a stable signed order identity before posting if the SDK supports it. If not, record the ambiguity and keep live eligibility blocked until recoverable identity is established. Do not invent client-order-id support.

Test acknowledged-but-unfilled, partial fill, terminal rejection, invalid/missing fields, duplicate responses, precision boundaries and oversized responses. Unknown fields may be retained as sanitized metadata; unknown financial shape must fail closed.

**Verify:** `py -m unittest tests.test_trading tests.test_trading_byo`.
**Done when:** the capability matrix distinguishes fixture-verified, demo-verified and production-verified operations; all unsupported live capabilities are explicit blockers. A second focused review checks the four action mappings and rounding independently.

## T02 — Extract one shared execution boundary and retire bypasses

**Depends on:** T01. **Size:** L, split extraction from bridge migration. **Files:** `server.py`, `trading_control/{service,store,policy}.py`, `trading.py`, `scripts/live_trader_bridge.py`, related existing tests.

Extract only connection access interfaces, saved-run lifecycle, live guardrail policy and submission orchestration needed by later tasks. Preserve route names, response contracts, status codes and exact manual confirmation rules. Use injected store/clock/credential resolver/venue adapter dependencies; avoid importing FastAPI globals from worker code.

Replace the legacy bridge's direct exchange POSTs with a disabled-by-default client of the shared intent/run flow, or remove its live option with a clear migration message. It may still preview/simulate. It must not claim `filled` from HTTP status, use in-memory allocation as capital authority, or accept a question as token ID.

Document all remaining order-write call sites. Low-level signing functions must be reachable through the shared service, not through public tools. Keep benchmark `place_trade` shadow-only even if an environment variable is set to live.

**Verify:** `py -m unittest tests.test_trading tests.test_trading_byo tests.test_live_trader_bridge tests.test_server` plus existing benchmark guard tests located in T00.
**Done when:** manual clients behave as before, the worker can import the extracted services without booting the server, and tests prove the old bridge cannot bypass authority. No mandate execution exists yet.

## T03 — Define typed twin records and state transitions

**Depends on:** T02. **Size:** M. **Files:** new `twin/__init__.py`, `twin/models.py`, `tests/test_twin_models.py`, `tests/fixtures/twin/records/`.

Implement PLAN section 4 records with strict finite numbers, decimal-string serialization, timezone-aware timestamps, explicit environment/account namespace and version fields. Encode command transition rules separately from display status. Add immutable intent hashing with sorted canonical serialization; exclude presentation text from identity only when it cannot affect execution. Include policy, account binding, market and strategy versions in authorization checks.

Build canonical instrument IDs from actual venue identifiers and environment. Preserve token outcome identity. A slug/title can be a display alias only. Define rejection reason enums, cursor/completeness records and structured `PASS` decisions.

**Verify:** `py -m unittest tests.test_twin_models` covering round trips, NaN/infinity, naive/future timestamps, price/quantity precision, identifier collisions, changed-intent hash, terminal-state regression and schema migration rejection.
**Done when:** the model cannot smuggle an execution flag or credential field into a proposal; every later task uses these records rather than free-form financial dictionaries.

## T04 — Add durable events, projections, reservations and claims

**Depends on:** T03. **Size:** L, split store operations from concurrent integration tests. **Files:** `twin/store.py`, extracted trading store/service, `tests/test_twin_store.py`, `tests/test_twin_store_integration.py`, test emulator setup documentation.

Implement account-scoped Datastore transactions for event deduplication, projection versions, command identity, budget reservations and fenced claims. A reservation covers cash plus incremental maximum loss; maintain explicit linkage to venue-held cash to avoid double counting. Add outbox/inbox entities and idempotent projection rebuild. Use an in-memory implementation for fast tests only; live mode refuses a missing durable store.

Route manual and twin execution on the same account through the same reservation/claim boundary. Existing manual order behavior remains externally compatible. Do not perform venue/model calls inside a transaction. Unknown/submitting reservations cannot be released by a TTL cleanup job.

**Verify:** `py -m unittest tests.test_twin_store`; then `py -m unittest tests.test_twin_store_integration` against a documented local Datastore emulator. Run two independent processes competing for the last permitted cash/risk allocation. Repeat with manual versus autonomous commands, transaction conflict, stale lease and repeated event IDs.

**Done when:** only one competing allocation succeeds, a killed worker cannot overwrite the new generation, projection replay is stable, and concurrent integration evidence is recorded. Do not replace this gate with mock-only lock tests.

## T05 — Build the executable market twin

**Depends on:** T03. **Size:** M. **Files:** `twin/market.py`, small adapter additions in `market_data.py`/`venue_api.py`, `tests/test_twin_market.py`, venue fixtures.

Normalize rules, status, close/resolution times, outcome IDs, price increments, fee schedules, available book depth and supported order capabilities. Fetch actual outcome prices and available quantity. Mark source/received times, age and completeness independently. Resolve a market to an exact immutable instrument before creating a candidate.

Implement deterministic eligibility filtering: allowed category, settlement-spec present, supported binary economics, horizon window, no suspended/resolved/disputed contract, and known fee/precision/capability metadata. Group related events conservatively; unknown mapping is not proof of independent risk.

**Verify:** `py -m unittest tests.test_twin_market` covering missing NO book, nullable optional holders, malformed account-like data, incomplete pagination, stale/future timestamps, changing tick size, zero liquidity, suspended market and same-title/different-settlement contracts.

**Done when:** every accepted market can produce a precise priced order intent; unsupported data becomes a typed reason for PASS. No endpoint inventory expansion beyond needed capabilities.

## T06 — Synchronize a complete account snapshot

**Depends on:** T04, T05. **Size:** L, split normalization from generation consistency. **Files:** `twin/account.py`, `twin/reconcile.py`, adapter portfolio methods, `tests/test_twin_account.py`.

Fetch all pages of balances, positions, orders, fills and settlement updates for one dedicated account. Deduplicate by immutable venue IDs. Associate orders/fills with local commands where known; external activity receives explicit attribution. Keep an old complete generation if any fetch fails. Empty successful responses and unknown/malformed financial responses are different states.

Compute available/total/reserved cash, actual holdings, basis, settled cash, fees and conservative liquidation value. Distinguish provisional matched fills from settled/final venue states where necessary. Compare venue facts with local projection using explicit currency/quantity tolerances from venue precision, not arbitrary broad epsilon. Drift pauses new exposure and records the discrepancy.

**Verify:** `py -m unittest tests.test_twin_account` with multi-page fixtures, repeated fills, concurrent order activity during snapshot capture, external deposits/manual trades, failed second page, settlement delay/correction and unavailable balance.

**Done when:** restarting and re-importing the same account produces the same economics; incomplete reads cannot erase inventory or increase permitted spending. This is initial synchronization, not yet full ambiguous-submit recovery.

## T07 — Enforce research budgets before provider calls

**Depends on:** T04. **Size:** M. **Files:** `twin/budget.py`, provider configuration integration, `tests/test_twin_budget.py`, proposed `configs/twin.yaml`.

Create explicit model/tool/token/time and USD limits. Implement atomic worst-case request-cost reservation, actual-usage reconciliation, cancellation and stale in-flight handling. Keep unknown charges reserved until safely reconciled; a process timeout is not proof that a provider did not bill. Support explicitly configured zero-cost capacity with token/rate limits. Missing price data blocks paid research when a currency ceiling is required.

Configure one research model, three candidates per cycle, eight tool calls per candidate and one schema repair. Cache public evidence by content/market/as-of version. Maintenance does not consume the research allowance.

**Verify:** `py -m unittest tests.test_twin_budget` including two workers spending the last allowance, failed/timeout calls, missing price, usage above the estimate and day-boundary handling.
**Done when:** the system stops before exceeding the authorized request budget; it reports actual and uncertain spend separately and continues account maintenance.

## T08 — Produce evidence-backed, non-executing forecasts

**Depends on:** T05, T07. **Size:** M. **Files:** `twin/research.py`, proposed `prompts/twin_research_v1.txt`, narrow reuse of `agent_capabilities.py`, `forecast_ledger.py`, `tests/test_twin_research.py`.

Create an allowlisted research loop using existing provider/evidence tools. Inputs include market rules, frozen as-of time, captured prices, earlier calibrated history and evidence refs. Tools can retrieve public research but cannot trade, manage credentials, approve mandates or fetch arbitrary authenticated URLs. No unbounded chat transcript in the prompt.

Request strict `Proposal`/`Forecast` fields with raw probability, uncertainty provenance, supporting/contrary evidence and expiry. Validate, allow one schema repair, then persist a PASS on failure. Persist prospective forecasts before later resolution is known, using existing ledger invariants. Capture prompt/model/config hashes.

**Verify:** `py -m unittest tests.test_twin_research tests.test_forecast_ledger` covering malformed JSON, dict-valued narrative, fabricated IDs, no citations, stale evidence, prompt injection in a page, future data and exhausted budget.

**Done when:** an offline fixture creates a traceable forecast or a clear PASS, cannot invoke an exchange write, and never parses a prose trade recommendation as an execution command.

## T09 — Implement deterministic sizing and portfolio risk

**Depends on:** T06, T08. **Size:** L, split sizing from portfolio/race integration. **Files:** `twin/risk.py`, selected reusable helpers from `portfolio_optimizer.py`/`accounting.py`, `tests/test_twin_risk.py`.

Implement PLAN's fixed-bin `calibration_v1`, conservative probability, costs, clipped fractional sizing, venue rounding and final cash-flow checks. Load only pre-cutoff resolved observations, enforce the bin sample minimum and persist calibration hashes. Evaluate every candidate after known inventory, venue open orders and local unmatched reservations. Compute additive cluster worst-case loss and cash needed per venue. Separate trailing-day additions, realized losses and peak-equity drawdown. Missing calibration/fee/depth/cash blocks new risk.

Add verified reduce-only handling with available inventory and pending sells; do not allow an oversized close to flip exposure. Candidate order sorts deterministically. If minimum size exceeds any cap, PASS. Recheck inside the reservation transaction using captured version/freshness preconditions; stale versions restart validation, never bypass it.

**Verify:** `py -m unittest tests.test_twin_risk` with explicit cash-flow examples for YES/NO buy/close, partial fills, external balances, correlated markets, simultaneous pending intents, fee rounding, zero bankroll, missing uncertainty and max drawdown.

**Done when:** a fixed snapshot produces a reproducible risk result, all limits bind cumulatively and every rejection has a stable reason. Review monetary units independently.

## T10 — Build a realistic shadow venue adapter

**Depends on:** T06, T09. **Size:** M. **Files:** `twin/simulator.py`, `tests/test_twin_simulator.py`, replay fixtures.

Implement the same preview/submit/status/cancel/account interface as real adapters with no network writes. Consume captured bid/ask depth, tick/size rules, fees and a seeded latency/adverse-price scenario. Support no fill, partial fill, canceled remainder, cancel-fill race and delayed settlement. Shadow account IDs and events cannot collide with live accounts.

Do not manufacture fills from a midpoint or a model statement. Do not replay against the current book when testing a past snapshot. Store simulator version, assumptions and random seed in the run.

**Verify:** `py -m unittest tests.test_twin_simulator` with hand-calculated cash/position outcomes, depth exhaustion and settlement/fee events. The same event stream and seed must reproduce the same result.

**Done when:** complete shadow cycles use the same intent/risk/account abstractions as live execution; only the venue adapter differs. Existing benchmark accounts are not silently migrated or credited with live performance.

## T11 — Assemble the autonomous strategy and exit loop

**Depends on:** T08, T09, T10. **Size:** M. **Files:** `twin/strategy.py`, `tests/test_twin_strategy.py`, `configs/twin.yaml`.

Build `foresea_edge_v1`: reconcile held positions, evaluate deterministic exits, select changed eligible candidates, request bounded research, score net edge, size through risk, and emit an intent or HOLD/PASS. Limit new entries to one per cycle initially. Exits and reconciliation must not depend on the LLM being online.

Review holdings even when discovery returns no new markets. Implement thesis expiry, maximum holding time, settlement-rule changes and policy stop behavior. Prevent churn using cooldown and versioned material-change criteria. Stable cycle keys combine strategy/account/time bucket/config version; a repeated key resumes/reuses finished results.

**Verify:** `py -m unittest tests.test_twin_strategy` with buy -> partial fill -> revised forecast -> close -> settlement, plus no-op cycle, provider outage, budget exhaustion, invalidated thesis, cooldown and changed policy.

**Done when:** a fixture account completes a full autonomous shadow lifecycle with an attributable decision record at every step; HOLD/PASS is not reported as a system failure.

## T12 — Add causal replay and strategy evaluation

**Depends on:** T11. **Size:** L, split replay from evaluation artifacts. **Files:** `twin/replay.py`, `twin/evaluation.py`, `scripts/twin_replay.py`, `tests/test_twin_replay.py`, `tests/test_twin_evaluation.py`.

Add a replay CLI consuming versioned captured datasets and frozen configuration. Filter inputs by observed-at as well as occurred-at. Deduplicate event clusters and split calibration/test data by time and event. Include market baseline and simple fixed-policy baseline, an out-of-sample strategy report, net P&L after all modeled costs, drawdown, calibration, turnover and abstention.

Reuse existing forecast evaluation functions where correct; do not change their shadow promotion semantics. Produce a separate twin readiness artifact with code/config/data hashes, evidence timestamps, unsupported assumptions and per-gate status. Reject stale/corrupted reports and prevent training on test outcomes. Stress fees, slippage, missing data and correlated losses.

**Verify:** `py -m unittest tests.test_twin_replay tests.test_twin_evaluation tests.test_forecast_evaluation_report`. Replay the same fixture twice and compare canonical results. Insert late-arriving evidence and confirm earlier decisions do not change.

**Done when:** a second model can reproduce the report from the command and hashes; insufficient historical depth or sample size is explicit, not disguised as successful backtesting.

## T13 — Add immutable, revocable autonomous mandates

**Depends on:** T04, T09, T12. **Size:** M. **Files:** `twin/mandates.py`, `twin/routes.py`, `server.py` router registration, `tests/test_twin_mandates.py`.

Implement owner-scoped mandate draft/approve/revoke. Approval captures the exact hash, current identity, absolute capital/loss/model budgets, venue/account bindings, strategy/model/config version, actions and expiry. API-supplied authority fields are ignored/rejected; the server derives ownership. Editing an active mandate creates a new version requiring approval. Never self-renew or auto-expand scope.

Separate manual order approval from autonomous mandate authority. Live activation validates the current T12 readiness artifact and exact release/config hashes; a draft or shadow mandate can exist before live eligibility. Mandates cannot grant withdrawal, credential management, arbitrary HTTP or public MCP execution. Record global/account/strategy/venue pauses and explicit expiry behavior. Revoke idempotently and block unsent commands under that authority; maintain read-only reconciliation.

**Verify:** `py -m unittest tests.test_twin_mandates` for cross-owner access, stale approval hash, expired/revoked mandate, widened budget, unknown account epoch, research-role request and replayed activation request.

**Done when:** no live command can be authorized solely by an LLM/request payload. Document approval API semantics without enabling live flags.

## T14 — Implement authorized durable submission

**Depends on:** T01, T02, T04, T09, T13. **Size:** L, split authorization/reservation from adapters. **Files:** `twin/execution.py`, shared trading service, `trading.py`, `tests/test_twin_execution.py`.

Implement `submit_authorized_command` following PLAN section 6. Both manual approvals and autonomous mandates produce server-validated authority records; the low-level adapter does not accept a public bypass boolean. Recheck policy/mandate/account epochs and current strategy readiness after risk evaluation and immediately before sending. Existing manual orders do not acquire a new strategy-performance requirement; this eligibility gate applies to autonomous new exposure.

Persist stable client ID or prepared signed-order identity and exact request fingerprint before network dispatch. Classify confirmed rejection separately from unknown acceptance. A 200/201 response without validated fill data is acknowledged, not filled. Do not repeat an entire trading loop on any network exception.

**Verify:** `py -m unittest tests.test_twin_execution` with fake venue submissions: duplicate tasks, mutated intent, concurrent manual order, revoke-before-send, disabled live flags, precise fee reservation and loss of response after venue acceptance.

**Done when:** one authorized intent yields at most one logical order identity and unauthorized paths yield zero venue writes. The saved manual confirmation behavior and benchmark shadow-only tests remain green. Independently review every raw venue-write call site.

## T15 — Recover ambiguous submissions and all order lifecycle races

**Depends on:** T06, T14. **Size:** L, split recovery from cancellation/settlement drills. **Files:** `twin/reconcile.py`, `twin/execution.py`, adapter lookup/cancel methods, `tests/test_twin_recovery.py`.

Resolve `submission_unknown` using the persisted venue identity, exact account/instrument and complete order/fill queries. Confirmed absence must meet a documented venue-specific consistency/retry condition. If it cannot be established, hold the reservation and mark operator attention; never issue a new identity. Reconcile before retrying cancellation. Preserve fills received after cancel acknowledgement.

Apply fills once by identity/version; do not count cumulative quantities repeatedly. Respect provisional/final settlement states and corrective events. Recover startup commands in `reserved`, `submitting` and `cancel_requested` without assuming stale leases imply no order. A stale worker must not progress after losing its fence.

**Verify:** `py -m unittest tests.test_twin_recovery` including crashes immediately before send, after acceptance, before receipt persistence and after partial fill; duplicate/out-of-order events; cancel-fill race; account reconnect; manual trade drift; settlement revision.

**Done when:** every fault has a bounded automatic recovery or explicit paused state with an operator runbook, never silent exposure loss or duplicate resubmission. Preserve evidence for independent review.

## T16 — Implement the private bounded worker

**Depends on:** T07, T11, T15. **Size:** M. **Files:** `twin/scheduler.py`, `twin/worker.py`, `tests/test_twin_worker.py`.

Build the exact role-scoped dispatch, research, maintenance and research-job claim/result handlers from PLAN section 8. Research uses the narrow maintenance interfaces to acquire its budgeted assignment and return validated results; it has no direct access to trading Datastore. Task payloads contain stable IDs, not credentials or arbitrary URLs. All handlers claim durable jobs with bounded leases and respect task deadlines. Duplicate delivery returns the stored completed result; in-progress delivery cannot create a second order.

Maintenance priority is recovery/reconciliation, then exits, then new candidates. Startup marks execution unready until complete reconciliation. Add stale-job detection, capped safe-read retries and explicit provider/data degradation. Hard order ambiguity stays paused. No in-process infinite loop or FastAPI background task as the sole scheduler.

**Verify:** `py -m unittest tests.test_twin_worker` with fake clock/queue, duplicate deliveries, lost task acknowledgement, expired lease, shutdown, stale state, exhausted research budget and complete model outage.

**Done when:** account maintenance completes without a model call and restarting the worker never resets capital, state, authorization or budgets.

## T17 — Provision private runtime and queue isolation

**Depends on:** T16. **Size:** L, split infrastructure definition from staging verification. **Files:** new `infra/twin/` deployment definitions/scripts and README, `pyproject.toml` if Tasks client needed, container/deployment files, `tests/test_twin_worker_auth.py`.

Add repeatable deployment commands using the current GCP project conventions, explicit environment and artifact tag. Provision Cloud Tasks queues for research and maintenance plus Cloud Scheduler's due-work call. Define retry/deadline/max-dispatch settings. Use IAM and Google OIDC audience verification. Do not rely on request headers asserting a queue name.

Deploy the same image as `twin-research` and `twin-maintenance` with distinct identities. Research has evidence/model access but no trading connection KMS decrypt, trading Datastore access or execution-handler permission. Maintenance owns durable mutations and enforces exact service identity on each narrow research-job interface. Do not rely on nonexistent per-entity-kind Datastore IAM restrictions. Public API remains separately accessible. Record infrastructure cost drivers and cleanup commands; do not promise a free tier covers usage.

Default all new services to shadow, zero live capital and no live mandate. Configure secret references, never secret values in source. Check actual execution-region eligibility before any venue live readiness claim. Do not repurpose the existing shadow workflows as live order runners.

**Verify:** `py -m unittest tests.test_twin_worker_auth`; staging requests with valid/wrong audience, expired token, unauthorized identity, spoofed headers and anonymous caller. Demonstrate denied research-role trading decrypt, direct Datastore mutation and execution-handler access while budgeted claim/result calls work. Execute a duplicate queue task in staging.

**Done when:** staging shadow runs resume after restart, maintenance remains responsive during saturated research, and IAM evidence plus rollback commands are recorded.

## T18 — Deliver the operator desk and control API

**Depends on:** T12, T13, T16. **Size:** L, split APIs from UI. **Files:** `twin/routes.py`, `frontend/trade.html`, proposed `frontend/twin.js`/styles, generated `static/` outputs, `tests/test_twin_routes.py`, `tests/test_twin_ui.py`.

Implement owner-scoped `GET /twin/status`, `/twin/portfolio`, `/twin/decisions`, `/twin/commands`, `/twin/readiness`; `POST /twin/mandates`, `/twin/mandates/{id}/approve`, `/twin/mandates/{id}/revoke`, `/twin/pause`, and `/twin/commands/{id}/cancel`. Lists use bounded pagination. Approval includes expected mandate hash and idempotency key. Cancel requests are deduplicated and enqueue maintenance; no raw order parameters on a cancel request.

Return timestamps, reasons and real command states, not optimistic “executed” text. UI requires explicit input of absolute live limits and shows exact scope/expiry before activation. Hide credential values and keep private results out of public boards/MCP. Include keyboard-accessible controls, loading, stale-data, disconnected, partial-fill and unknown-submission states.

**Verify:** `py -m unittest tests.test_twin_routes tests.test_twin_ui`; `npm run frontend:build`; inspect the built page at local port 8080 using fixture data. Verify pause/approve/cancel interactions and cross-owner 404/403 cases, not only HTML string presence.

**Done when:** an operator can explain current exposure, find an uncertain order and revoke authority from the desk; rendered shadow/live provenance and generated asset references are correct.

## T19 — Add operational signals and incident runbooks

**Depends on:** T15, T16. **Size:** M. **Files:** relevant twin modules, observability integration, `docs/autonomous-twin/OPERATIONS.md`, `tests/test_twin_observability.py`.

Instrument bounded spans for research, risk, reserve, submit, reconcile and maintenance using existing native OpenTelemetry conventions. Counters/histograms cover decision reasons, duplicate suppression, drift, ambiguous submissions, stale data, retry exhaustion, queue lag, budget and actual cost. Metric labels are bounded enums/venue/strategy, never order IDs, wallet addresses or free text. Use trace/event references for high-cardinality audit linkage.

Write concrete runbooks: kill/pause, lost provider, lost store, stalled queue, uncertain submission, broken credentials, missing portfolio page, venue halt, deployment rollback and settlement correction. Exporter errors must not fail trading logic; standard logs remain usable. No Superlog installation or paid subscription prerequisite.

**Verify:** `py -m unittest tests.test_twin_observability`; assert secrets and authorization material are redacted from exceptions, task payloads and logs. Break the telemetry exporter and confirm risk/reconciliation results are unchanged.

**Done when:** every live-blocking reason has an operator action and reference to the affected account/command through access-controlled logs.

## T20 — Establish CI, readiness and release evidence

**Depends on:** T12, T17, T18, T19. **Size:** M. **Files:** `.github/workflows/ci.yml`, new bounded twin integration workflow if needed, readiness evaluation, release documentation/tests.

Add deterministic unit/contract tests to CI, plus a dedicated emulator integration job for account reservation/claims. Default network-deny fakes prevent live orders even if a developer machine has credentials. Validate config/schema migrations and generate/read a sample readiness artifact. Keep scheduled benchmark workflows shadow-only.

Build/deploy smoke tests verify health and readiness separately and inspect captured worker state without submitting real orders. Test additive migrations and rollback with existing manual trades present. Preserve prior event formats and projections; retain command IDs and reservations across releases. A rollback first pauses new exposure and reconciles unknown orders; never reset or delete the ledger.

**Verify:** full repo tests/lint, focused twin tests, emulator tests, frontend build and staging fault drill. A second review pass checks release artifact assertions against actual evidence and current code/config hashes.

**Done when:** G0 is evidenced, stale or forged readiness fails closed, and a clean protected-branch PR can deploy the shadow runtime with a reproducible rollback path.

## T21 — Run the forward shadow and strategy evidence gates

**Depends on:** T20. **Size:** elapsed evaluation period, not a large coding session. **Files/artifacts:** trial configuration, versioned GCS evidence, `docs/autonomous-twin/TRIAL_REPORT.md`, progress entry.

Freeze the strategy/config/model and run G1 for seven consecutive days using the private runtime. Capture actual market snapshots, decision provenance, order simulation, per-cycle cost and reconciliation drift. Reuse older prospective data for G2 only with demonstrable compatible provenance and versions. Do not repeatedly wake an expensive model to watch unchanged trial status; produce a compact automated report when evidence changes.

Evaluate independent event clusters, market baseline skill, costs and net results. Run the prescribed outage, duplicate task, cancel-fill and kill/restart drills. Mark insufficient data as `collecting`, poor strategy evidence as `ineligible`, and missing venue mechanics as `blocked`; none is a failed attempt to write code.

**Verify:** reproducible report command from T12, successful causal-data checks, immutable hashes, zero unexplained accounting divergence, and G1/G2 status from exact evidence.

**Done when:** trial report says which gates pass and which do not. If forward observation is still needed, record the next measurement/time; do not claim the entire roadmap is live-ready or loosen thresholds.

## T22 — Execute the bounded live release when eligible

**Depends on:** T21, and G0-G2 passing for the exact release. **Size:** operator-led activation and verification. **Artifacts:** approved mandate, staging/live release records, `docs/autonomous-twin/LIVE_RELEASE.md`.

Prepare a concrete review package: deployed image/config hashes, eligible venue/account, exact capital/order/loss/model limits, expiry, cancellation/exit behavior, contract/identity recovery evidence, private worker IAM and fresh account snapshot. All code, tests, staging deployment and desk behavior must already be complete. If capital/account parameters are missing, obtain only those values at this point.

Actual live activation requires the owner to authorize this specific mandate. The earlier request for a development plan is not approval to spend money. Enable one independently cleared venue first; then the other with its own evidence. Do not bypass jurisdiction/account restrictions, configure unlimited wallet allowance, or deposit/transfer funds as part of an implementation test.

Observe an authorized bounded order lifecycle, matching real fills/fees/cash to the twin. Exercise pause/revocation and verify no new exposure; reconcile to completion. Record actual execution costs and any variance from simulation. Do not force a trade when the strategy returns PASS merely to produce a successful demo.

**Done when:** authorized live operation is reconciled and the report records venue-specific status. If G2 does not pass or authorization is absent, the correct deliverable is an implemented, deployed shadow system with live activation explicitly pending. Broader limits require a new mandate; the agent cannot self-promote.

## Shared verification and status rules

From the task worktree in Windows PowerShell:

```powershell
$env:PYTHONPATH = 'src'
py -m unittest discover -s tests
py -m ruff check src tests
git diff --check
```

After source changes, run `py -m graphify update .`. Inspect changes; generated graphs or unrelated data are not automatically part of the commit. For frontend changes, run `npm run frontend:build`, inspect rendered output and stage only the relevant generated files required by repository conventions. The server preview uses port 8080.

If new `scripts/` Python files are touched, include them in a focused Ruff check as well. Tests marked “emulator” must report whether they ran or skipped; a skip cannot satisfy G0. Run the full suite and `ruff check src tests` before every commit as AGENTS requires. Do not repeat already-passing checks without a new change or unresolved concern.

Record task status, commit/PR, exact commands and exit codes, evidence paths, remaining risks and next task in `progress.json`. A later task can implement dependent code against fakes while external staging credentials are unavailable, but it cannot mark an unmet deployment/evaluation gate complete.


