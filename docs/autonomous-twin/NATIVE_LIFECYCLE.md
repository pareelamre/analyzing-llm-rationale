# Native order and fill recovery

Native recovery now restores the original immutable TradeIntent from the same
reservation transaction as its command. TwinIntent payloads are private, excluded
from indexes, bounded to64KB and verified against a whole-payload digest plus the
command scope/intent hash. Legacy commands with no payload stay unresolved; there
is no reconstruction from display text or automatic legacy backfill.

The existing account-scoped venue catalog adds single-order reads:
`kalshi.order` (`GET /portfolio/orders/{order_id}`) and `polymarket.order`
(`GET /data/order/{order_id}`). Existing authentication, parameter validation and
signed transports apply; neither is public or a venue-write operation.
Polymarket trade capture reuses the existing authenticated `fills` catalog
operation, filtered by condition and immutable intent creation time. Its required
maker-address parameter comes from the independently verified account funder.

NativeOrderReader verifies independently derived connection/account identity using
the existing native account plan, then checks the exact client/native order ID,
account, instrument, original size, side/action, price and creation time. Kalshi
discovery traverses live and historical orders by ticker and matches the persisted
client ID. Both live and historical fills are read; primary subaccount only is
supported. Primary-subaccount queries explicitly select zero, and
each matched order/fill must affirm an integer zero subaccount; missing fields or
boolean values cannot establish account identity. Native account bindings reject
Kalshi demo hosts before provider reads. An expired archived direct order lookup
remains unresolved and never proves absence. Native Polymarket CTF packets must match their stored
request fingerprint and economic/condition/token bindings. The original signed
packet is not re-signed, resubmitted or used to authorize a new order.

Reads bracket fill capture with two consistent positive order observations.
Private jobs save cursor continuation in `TwinNativeHistory` and
`TwinNativeHistoryChunk`, bound to account/epoch, command/request and the observed
order frame. Each job reads at most8 collection pages. Only relevant rows are
saved; incomplete captures never become inventory. Restarted jobs reuse completed
pages and resume the saved cursor, rather than starting at page1. State changes
discard the old cohort; rejected complete captures refresh on the next attempt.
Cursor repetition/generation drift/corruption fails closed across restarts.

Each collection has explicit limits of4096 pages,4096 relevant rows and900KB per
private entity. Associated Polymarket trade IDs are bounded to1000 and are matched
against one market/time-filtered paginated collection. Even an empty associated
list is checked for an owned but unlisted trade before accepting cancellation.
Checkpoints are excluded from indexes and cleared with the financial/lifecycle
commit. Malformed or over-budget captures retain exposure and require further
bounded recovery or operator reconciliation. Every provider call checks the
current job lease and capture budget
(at most90 seconds); existing transport timeouts bound an already started call.
Missing pages, unknown shapes, inconsistent quantities, unsupported current API
fields and missing orders keep the reservation. A404 is never absence proof.

Kalshi fills retain actual fixed-point quantity, price, fee and timestamp. For
Polymarket, only CONFIRMED trades become fills. Taker fills use the owned taker
order; maker fills use exactly one matching maker leg's amount, asset, wallet,
side and price, never the parent trade's full size. Pending MATCHED/MINED/RETRYING
trades retain their hold and cannot finalize a cancelled command. The same command
can accept the later confirmed fill. Failed/unknown finality is explicit degraded work. An
actual fee is required when the rate is nonzero; no approximate fee is invented.
Kalshi retains its native fill ID within the command-scoped projection; Polymarket
fill IDs include trade/order identity. Both use immutable version1;
unversioned economic corrections are quarantined for operator handling. Decimal
format and receive-time differences alone do not create revisions.

The field-level contract is the official
[Polymarket CLOB OpenAPI](https://docs.polymarket.com/api-spec/clob-openapi.yaml):
`Trade.price`, `size`, `fee_rate_bps`, `maker_orders`, and each maker leg's
`matched_amount`, `asset_id`, `price` and `side`. Quantities must agree with the
signed intent's original size and positive order matched size. For taker trades
with maker legs, the unique positive leg quantities must sum to the native trade
quantity. When all legs use the owned token, their notional must agree exactly
with the trade price; disagreement is not repaired with an invented price.
Complementary maker trades use the owned leg's token/price, never the parent
trade's different asset or full size. A zero explicitly reported venue fee rate
means zero venue fee; nonzero fees require an explicit actual fee amount. Rebates,
additional charges and unexplained unit/rounding differences are not invented.
The fixtures are contract-shaped synthetic examples, not private owner captures;
agreement with actual venue/account records remains a production release gate.

The existing reducer commits unique fills, command state and cash together. New
native price/fee fields are additive: old observations still deserialize, while
native economic conflicts are visible. Actual buy cost above its reserved cash or
loss hold, or mixed incomplete economics, creates private attention and pauses
new exposure. It never silently increases limits. A coherent explicit zero-fill
cancellation releases its hold once; fills arriving after sealed terminal evidence
go to the existing private quarantine instead of disappearing.

NativeLifecycleMaintenance handles a claimed private RECOVERY job containing
command_id and account_epoch_id. Follow-up jobs also carry stable recovery_root_id
and recovery_round_id fields. The exact job, scope epoch and command claim are
checked inside claim, recovery, lifecycle and attention transactions using the same
durable client. Provider reads run outside those transactions. Positive presence
may first persist ACKNOWLEDGED while holding all cash; a later lifecycle failure
does not erase that valid presence acknowledgement. Full fills still retain the
market-risk hold until verified market settlement; trade finality is not a market
redemption. No aggregate settlement is assigned to an arbitrary order.

Nonterminal observations stage one deterministic follow-up in the same transaction
as the lifecycle projection. The original deadline is preserved; the next read is
at least 30 seconds later and after the current command claim expires. Duplicate
delivery reuses the saved job instead of adding or postponing work. There are at
most 16 follow-ups per recovery root. Exhausted time/poll budgets return
`reconciliation_required` with `followup_budget_exhausted=true` and retain exposure.
Neither a successful read nor an exhausted budget means an open order is complete.

The private maintenance boundary maps unconfigured, degraded, malformed or
budget-exhausted native recovery to the existing `WorkerPaused` protocol. The
durable job is `PAUSED`, and duplicate delivery returns that saved outcome without
rerunning the adapter. Fixed reason identifiers keep private adapter errors out
of responses. Completed work and pending work with an automatic follow-up retain
their normal completion semantics; unrelated account/research degradation is not
remapped. This changes job outcomes, not capital, commands or held exposure.

After correcting the connection/data issue, a trusted maintenance operator can
enqueue a new bounded recovery job for the same command and authorized epoch.
Redelivering the paused job does not resume it. Never mint a new order identity,
change an epoch to force eligibility, clear a financial hold, or repeat order
submission merely to resume read-only recovery. Reconnected accounts with old
unresolved authority need the existing owner-reviewed reconciliation procedure.

The existing WorkerJob queue now persists an optional, identity-bound `not_before`
timestamp. Claim and due scans refuse early delivery. Cloud Tasks receives that
timestamp as schedule_time; local dispatch leaves future work queued for the next
due scan. Lost enqueue acknowledgements do not remove durable work. Operator setup
must provide a dispatch cadence shorter than the recovery deadline; the existing
5/15-minute schedules cannot guarantee delivery for a three-minute job. Both
worker and dispatcher versions must understand delayed jobs before native recovery
is enabled. Legacy jobs without a delay keep their original identity hash.

`create_environment_app(native_lifecycle_reader_resolver=...)` can install the
reader dependency for the maintenance role. Default startup remains unconfigured;
execution/live flags and owner mandates are not changed. Research-role requests
gain no trading-store or credential capability. A command recovery job without
the configured adapter degrades rather than falsely completing research recovery.

## Operator handling and remaining release gates

Keep unresolved exposure held. A missing legacy intent requires verified original
authority from durable source records and an owner-reviewed recovery procedure;
do not mint a replacement identity. Missing actual fees require an authoritative
fee/receipt source. Inspect private attention, full venue history and actual
balances before applying corrections. Never clear the recovery pause or directly
edit a filled quantity to make eligibility pass. Private snapshots, signed packets
and attention bodies must never become public GitHub artifacts.

Account-wide incremental history, native market-settlement allocation, complete account
ledger reconciliation, production owner/connection registration and IAM, default
maintenance scheduling and forward/live release evidence remain required work.
No provisioning, deployment, collector activation, model inference or real order
is included. Existing private R2 object storage remains unchanged.

## Sources

The wire formats and positive lookup boundaries were checked against the official
[Kalshi order](https://docs.kalshi.com/api-reference/orders/get-order) and
[fills](https://docs.kalshi.com/api-reference/portfolio/get-fills) references and
Polymarket's [order/trade reads](https://docs.polymarket.com/trading/manage-orders)
and [trade finality states](https://docs.polymarket.com/concepts/order-lifecycle).
These local contracts do not prove production consistency, IAM or live readiness.
