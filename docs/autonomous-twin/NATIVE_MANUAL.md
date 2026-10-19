# Opt-in native manual Polymarket execution

The direct `POST /trading/orders` and saved-run
`POST /trading/runs/{run_id}/execute` boundaries now prepare Polymarket requests
through the existing native backend. This is implementation wiring, not live
activation. `FORESEA_ENABLE_NATIVE_MANUAL_TRADING` defaults to **false** and must
stay disabled until the recovery and release checks below are complete. The
existing BYO/market-order switches, authenticated session, inline-credential ban,
fresh guardrails and exact `PLACE REAL ORDER` confirmation remain required.
Kalshi retains its existing confirmed execution path. Polymarket has no fallback
to the old sign-and-post helper if native preparation is unavailable.
Deploying this draft without the new opt-in therefore leaves Polymarket manual
execution unavailable; this is a deliberate migration boundary, not a transparent
replacement for an already-enabled legacy service. No deployment occurred here.

## One request, one durable native identity

1. Resolve the signed-in user's stored venue credentials and confirm the request.
   No API key or signing key enters the command, request-authority row or response.
2. Read one authoritative Gamma market and exactly two CLOB outcome books using
   existing readers. Reuse `normalize_market` and the captured fee model. Unknown
   fees, stale/incomplete books, wrong outcomes or unsupported protocols block.
3. Reuse an owner-bound live account root for the actual maker wallet. Concurrent
   manual requests share its cash/loss serialization with other intents. A legacy
   synthetic manual root with unresolved commands/holds blocks automatic cutover;
   it is not silently edited or discarded. Strong key/ancestor reads enforce
   this check; global scope inventory is advisory. Terminal history beyond the
   old 200-command page limit does not prevent cutover.
4. Reserve exact shares times limit price plus a conservatively rounded fee bound
   for BUYs. Respect `max_cost`; record the fee allowance in the immutable intent.
   Existing server guardrails remain responsible for portfolio/market/day limits.
5. Sign once, then atomically store encrypted packet, native hash, reservation,
   outbox and `TwinRequestAuthority` children for both the external approval/run
   reference and client/intent alias, bound to the immutable intent. A retry
   with the same manual request ID cannot allocate another intent merely because
   its clock/capture changed. The authority is retained after rejection/settlement;
   a genuinely new confirmed request needs a new client request ID. Direct
   Polymarket submissions require that ID before dispatch. The shipped quick
   trade client retains it across lost/invalid responses and suppresses duplicate
   clicks; saved runs receive a stable ID before approval.
6. Re-resolve the connection and recheck current policy, live switches, account
   epoch, market mapping, tick/minimum/fees and original capture freshness before
   posting the same stored object through the shared dispatch fence. Unknown
   acknowledgement retains the hold; a request is never re-signed/re-posted.
   Current collateral is read in six-decimal base units and rechecked against all
   held cash before transport. A rejected current-fenced, explicitly unstarted
   command releases its hold atomically. This cleanup cannot release a started,
   legacy-unknown or re-fenced sender. The collateral precision follows the
   [official pUSD specification](https://docs.polymarket.com/concepts/pusd).
7. Return the native hash in the existing audit envelope. A matched/delayed order
   response establishes acknowledgement, not a confirmed fill or realized P&L.
   Confirmed venue rejection follows the normal persistent audit path. Disabled
   native gates and failed public capture checks do not consume saved-run approval.

New native manual orders are limited to **FAK/FOK**, including price-limited IOC
(normalized to FAK), under `FORESEA_ALLOW_MARKET_ORDERS`. GTC/GTD/post-only and
position/domain-3 preparation remain blocked until lifecycle integration is
complete. Read/cancel support never authorizes a new resting order. Autonomous
requests remain immediate-only and still require a valid mandate/readiness context.
No autonomous mandate is fabricated for a manual confirmation.

## Corrected capture contract

The normalizer uses `polymarket:live:<condition ID>:<Gamma market ID>`, with
capability label `polymarket-clob-limit`. The initial native fixture duplicated
the condition ID and used a synthetic capability label. The integration now
tests the **actual normalizer output** with distinct IDs through the real SDK.
Capability vocabulary is separate from protocol selection: CTF books omit
`version`; any book carrying that field is rejected by this token-only backend.

Prepared packet schema **2** binds venue market ID, condition, fee version, tick,
minimum quantity, authority mode, explicit approval-reference digest and post-only
alongside the native signed body.
Older prepared packets are not silently rewritten/re-signed. Drain or reconcile
existing work before changing the packet schema or exact supported SDK version.
Encryption and key access follow [the native backend contract](NATIVE_POLYMARKET.md).

## Evidence and outstanding release work

The test suite covers real normalization/signing, manual confirmation/enablement,
native immediate preparation without autonomous authority, the shared server boundary,
fee-inclusive cash, wrong outcome/max-cost rejection, delayed retries, and lost
acknowledgements. The Datastore emulator contention case uses four different
intents and one request-authority key and verifies exactly one reservation.
No real venue request, model request, cloud provisioning, deployment or key/flag
activation is performed for these checks. Exact final check/review results are
recorded in `progress.json` after adjudication.

2026-10-19 checks: full emulator-enabled unittest passed **2,892 tests with no
skips** in 211.561 seconds; 15 emulator integration tests also passed separately.
The final isolated SDK 1.2 matrix passed 51 tests; 15 final wallet/alias, route
and browser retry regressions passed. A real local OTel debug export exercised
successful/ambiguous manual sends with mocked transport, emitted both manual
spans and seven bounded operation measurements, and excluded signed bodies.
Ruff, adjudication validation, Graphify and the staged AQG gate passed. One deep
review (`aud_1WCqLzwCI4`) returned 18 findings across five completed voices (one
provider fallback); [14 dispositions](NATIVE_MANUAL_REVIEW.md) fix 13 accepted
issues and reject one disproved fixture-leak claim. Fixes were verified locally,
not represented as a second panel review. Semgrep was unavailable locally, so no
SAST pass is claimed.

Before native manual activation, finish T15's concrete account/order/fill/
settlement maintenance and operator recovery for these same account roots and
native hashes. Verify the approved host, exact SDK/image, private key/ciphertext
backup access, required index, current venue eligibility, inventory and approvals,
and account reconciliation across a rollout/rollback. Quiesce legacy writers and
drain old revisions before enabling the new native account root; new code cannot
make an old binary honor a migration guard. Existing local tests do not
prove production IAM, live fills, settlement finality or forward strategy skill.

The private worker factory remains shadow-only. Autonomous live deployment and
an expiring, bounded capital/loss mandate remain separate gates. Historical
screening still qualifies zero cohorts; this manual wiring does not promote an
unvalidated forecasting cohort into an autonomous trading signal.
