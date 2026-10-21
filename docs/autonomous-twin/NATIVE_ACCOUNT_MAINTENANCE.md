# Native account capture in maintenance

`NativeAccountMaintenance` connects the existing complete Kalshi/Polymarket read
plans to a claimed private `RECONCILE` job. It uses the existing account snapshot
repository and never updates capital limits, submits an order, releases a hold,
invokes a model or interprets an open-order lookup as proof of absence.

The server supplies an `AccountReadBinding` with the exact registered scope and
read plan. The resolver must bind its credentials to that scope's connection and
actual venue account; it must not borrow deployment credentials or accept an
address/URL from task data. Native plans require complete scoped credentials and
official hosts. Their independent account binding contains the connection ref
and credential-derived identity: the actual SDK maker for Polymarket, or
`kalshi-key-<sha256(API key ID)>` for Kalshi. It must match the registered scope
both before and after creating the guarded plan. Existing opaque Kalshi scope
identities require explicit owner-reviewed mapping; they are not migrated or
guessed here. A task contains only `account_epoch_id`; its scope comes from the
durable job. Trusted resolver/connection-store integrity and IAM remain mandatory.

`create_environment_app(native_account_plan_resolver=...)` installs the operation
only for the maintenance role. The ordinary startup remains unconfigured for
native account reads, and existing shadow/live execution restrictions remain.
The resolver is a trusted application dependency, not a public request parameter
or environment string that imports arbitrary code. Registering a production
connection resolver and validating its IAM remain release work.

All pages are read outside storage transactions. The starting account projection,
command inventory and exact claimed job must remain current. Each page/accounting
read checks the processing lease, job deadline and a capture budget of at most
90 seconds, including individual position-liquidation requests. A request already
in flight retains its existing transport timeout (public books: 12 seconds);
no new request starts after the budget expires. The same client must own command,
snapshot and job stores. Snapshot persistence rechecks authority before and after
the write within its transaction; conflicting job/root updates abort its commit.
A failed/incomplete read retains the prior generation and reports an allowlisted
degradation reason. Page checks use only clock and job point reads; full account
and command checks run at admission and snapshot commit, avoiding per-page command
query amplification. Financial projections and recovery pauses are never changed.
Timestamps start at capture initiation, so slow reads cannot appear freshly read.

Native capture compares before, middle and after complete collection passes.
Polymarket balances/positions share the first accounting ZIP during the middle
pass, so that accounting component has two independently fetched observations.
Preserved fill quantity, price, status, timing, maker legs and source fingerprints
make economic revisions visible to that comparison. Collections are sorted for
comparison, and market-derived liquidation/portfolio marks and valuation timestamps
are excluded from its account-mutation hash. The captured snapshot retains those
fields. Liquidation prices use explicit dollar/cents units and reject values
outside the normalized 0..1 range. Matching repeated reads are
observational consistency checks, not a venue-wide atomicity guarantee or an
absence policy. In particular Polymarket's open orders are not historical orders.
Market-level settlements/redemptions are not assigned to individual commands.
Maker legs and fees still require exact order-level attribution before lifecycle
application. These limitations must not be bypassed to release reservations.
Unmodified Polymarket orders lacking client IDs report explicit account drift
until their native identities are attributed; synthetic ID-bearing tests do not
prove a production no-drift capture.

The current command inventory API caps reads at 200. A capture at that cap is
explicitly degraded rather than pretending the history is complete. Cursor-based
command history is a remaining scalability dependency. Full fill/settlement
history also has a release-blocking scalability limit: each snapshot is capped
at 900,000 bytes. At that cap the previous generation is retained; new captures
remain degraded until incremental history storage is implemented. This patch
does not claim accounts with arbitrary history sizes are supported. Unrecognized external
orders/fills remain account drift; account capture never clears recovery pauses.

Private worker responses contain only status, generation, epoch, drift flag and
order/fill counts. Credential-bearing provider text is excluded from the new
operation's spans/logs/errors and snapshot persistence emits a fixed error label.
Native OTel is reused without a new exporter, connector or paid service.

## Verification and remaining work

### Economic journal comparison

Native capture now compares against `DatastoreAccountJournal` under the same
account root/client. An immutable, explicitly supplied `JournalOpening` pins the
registered scope/epoch, opening cash, inventory/basis, cumulative fees and a private
source digest. Captured balances never bootstrap or repair the journal. Until a
trusted source procedure initializes it, captures persist `journal_uninitialized`
drift and block new exposure.

Canonical `JournalFact` records carry exact cash and per-instrument quantity/basis
deltas, actual fees and a source digest. Imports append every source revision,
require contiguous versions and reject conflicting duplicate versions. Replay uses
only the latest version, so a correction replaces a payout rather than crediting
it twice. Provisional settlements retain cash/inventory and block exposure. External
cash flows are explicit facts and are excluded from realized strategy P&L. Facts
cannot predate the approved opening or be later than the capture being compared.
Negative cash, invalid basis, malformed decimals and invalid settlement consumption
fail closed. No balancing event is synthesized from an observed discrepancy.

Complete captures compare total cash (including venue-held cash), inventory,
per-instrument basis and cumulative fees. The snapshot records the exact journal
revision and fingerprint even when it reports drift. Missing journals, pending
settlements or mismatches block exposure. Journal reads participate in the snapshot
transaction's authority checks, including initialization/import during a capture.
Reserved cash is a spending restriction rather than an expense. Journal imports
advance the account revision to invalidate already evaluated risk preconditions;
cash amounts, reservation totals, orders and loss limits remain unchanged.
Snapshot reads recheck the current native journal revision/fingerprint and economics;
changed, missing or corrupt journal authority produces a blocking view even for a
previously reconciled snapshot. Legacy snapshots remain readable but cannot authorize
exposure without journal provenance. The network-free shadow simulator supplies its
own economic-event digest, accepted only within the existing shadow account scope;
it is not native receipt or live-account authority.

Facts use a stable source sequence to disambiguate equal timestamps; ambiguous
ordering is rejected. Corrections preserve their source time and sequence. A higher
version may explicitly void a fact with zero economics, preserving its history.
Rebate/reward income is a separate verified cash fact included in strategy P&L,
not an external capital contribution. Standalone fee facts are restricted to actual
fees represented by the venue's captured fill/settlement fee totals; other fee
collections require a source adapter before reconciliation can succeed. Imports
reject future source timestamps against an injected trusted server clock, with no
implicit clock-skew allowance. If capture time regresses behind a journal fact,
the captured snapshot persists blocking drift rather than advertising readiness.

This is an independent economic comparison, **not** a native receipt verifier:
a SHA256 source reference alone proves neither an upstream receipt nor owner approval.
Only trusted server source adapters/operator procedures may initialize or import
verified facts. No public endpoint, tool or worker task accepts raw facts. Native source
adapters still need to verify full receipts, cash flows, maker legs, actual fees,
settlement finality/corrections and venue-specific netting before posting facts.
No aggregate settlement is assigned to a command. Automatic reservation release
and capacity refresh remain separate financial integration work.
Source adapters must explicitly verify that opening and captured cumulative fee
totals cover the same full history, that basis deltas use the venue's accounting
method, and that source times match the declared server clock bounds. The journal
reconciles cash flow and inventory, not the allocation between available/reserved
or pending/settled cash. Complete native order/balance reads preserve that split;
the existing risk gate rejects a projection whose available cash differs from the
snapshot, and this operation never refreshes capacity. Full order/collateral and
execution integration remain release requirements.

The journal retains at most4096 source versions and900,000 encoded bytes per epoch;
reaching either bound pauses import rather than pruning unaccounted history. Exact
Decimal comparisons are used until venue precision is explicitly verified. These
bounds and source-adapter requirements are release gates, not production scalability
or live-readiness claims. Private journal bodies must never be public artifacts.

The SQLite-backed worker/capture regressions cover both venues, epoch changes,
expired/refenced jobs, scope/connection mismatch, local capacity changes, retained
prior snapshots, middle-read economic changes, capture deadlines, truncated local
command history and rollback of a post-write validation failure. Real local
Datastore cases repeat those tests and prove replacement jobs commit before old
snapshot transactions abort for both venue types **in the local emulator**.
Production was inspected read-only: its mode is `PESSIMISTIC`. Its locks can
serialize the replacement behind the original transaction instead; emulator
ordering is not production evidence. The required invariant is serializable
authority at commit, with unchanged or aborted stale writes. No production race
or IAM drill was performed. Modern Datastore transactions span these roots without
an obsolete `xg` option; see [Google's transaction and concurrency documentation](https://docs.cloud.google.com/datastore/docs/concepts/transactions).
Fresh full-suite, lint, review
and CI results are recorded in the PR and progress tracker before merge.

This delivers the account observation path. Native per-command lifecycle reads,
fill/settlement attribution, automatic maintenance scheduling/connection selection,
production reconciliation and forward-performance release evidence remain open.
No deployment, storage migration, live activation or new cloud service is included.
Private object storage remains the existing R2 destination.

## Owner-reviewed first opening

The authenticated owner can use `GET /twin/accounts/{scope_id}/opening` to review
the first baseline of an existing native live scope. The response binds the full
scope identity, epoch, snapshot generation/time/content, account revision, cash,
holdings/basis, cumulative fees and observed external activity. The owner-only
response includes the actual `scope` and complete normalized `snapshot` as well
as their hashes. Both references are reproducible as SHA256 over UTF-8 JSON with
sorted keys and separators `(',', ':')`; numeric amounts remain decimal strings.
These private source bodies must not be posted as public issue/PR artifacts.
This GET never
initializes anything. Refresh the complete native capture if it is more than
120 seconds old. The operation uses the existing session owner resolver and the
existing transactional store; there is no new service or storage destination.

After independently checking the displayed totals and retaining private source
evidence, the owner sends the same route a POST with exactly:

```json
{
  "review_hash": "<64 lowercase hex characters from GET>",
  "evidence_ref": "<SHA256 of the privately retained baseline evidence>",
  "confirmation": "REGISTER ACCOUNT OPENING"
}
```

`evidence_ref` is an owner attestation/reference, not a cryptographic verification
of venue history. The opening adopts a starting point; earlier activity remains
outside prospective strategy P&L. Neither a hash nor a model instruction is owner
authentication. The server derives ownership from the signed-in session and does
not accept replacement cash, position, owner, connection or epoch fields in POST.
This API does not create account scopes or choose a trading connection. The scope
must already be registered through the existing trusted runtime setup. Operator
desk UI integration remains separate; this is an authenticated API flow.

Registration refuses any existing local command, held cash/loss reservation,
unconfirmed/active venue order, pending Polymarket fill, other account drift or
existing journal without a matching approval. This first-baseline flow is for an
epoch that has not yet submitted local twin commands. Historical venue activity
is visible in the review and explicitly adopted by the owner; it is not assigned
to an old strategy order. The opening retains identity-to-source hashes for the
adopted order/fill groups. Subsequent reconciliation explains only exactly matching
groups: a changed row, a new identity or even another fill under the same old client
identity remains external drift. Raw source rows remain in the snapshot; adopted
identities are removed only from its unexplained-activity projection. Existing
journals are never reset to observed balances. Terminal Polymarket `FAILED` rows
can be recorded as historical evidence only with zero fee; pending and unknown
statuses still block. Zero-quantity holdings must have zero basis and are excluded
consistently from the reviewed and journal inventory.

Approval and the immutable journal opening commit in one account transaction.
Source/root changes conflict with the commit, and expiry during initialization
rolls back all writes. A retry of the same approval returns the original result,
including after later journal imports; a different evidence/review pair conflicts.
The approval JSON is private and excluded from indexes. It is not a public repo
artifact. Actual balances, capital/loss limits, reservations and mandates remain
unchanged. Previously captured snapshots remain blocking until a subsequent
complete native reconciliation records the new journal proof. Baseline approval
does not authorize live trading or attest future netting/cash-flow receipts.

The current restriction checks the whole account scope's local command history,
including earlier epochs. A previously traded/reconnected scope needs the existing
trusted scope-transition/reconciliation procedure; deleting commands or inventing
an empty history is not a remedy. No new epoch or scope is created by this API.

Refresh is an operator maintenance action, independent of models. On an already
configured runtime with its verified native reader resolver, add one existing
`WorkerJobKind.RECONCILE` job with the scope ID, payload
`{"account_epoch_id": "<current epoch>"}`, a unique job ID and a three-minute
deadline, then pass it to the existing dispatcher's `enqueue(job)`. Wait for that
job to complete before GET/review/POST. This uses the existing maintenance queue;
it does not provision a queue, start a collector or approve a trade. Native reader
configuration and production IAM remain release requirements; the default runtime
still refuses unconfigured native reads. Prepare the independent evidence before
requesting refresh. If scheduled capture replaces the snapshot during review, the
POST deliberately conflicts and the owner must review the new result. The API
does not extend its freshness window or automatically confirm a replacement hash.

Opening race tests exercise the local emulator's optimistic concurrency, including
snapshot replacement, distinct competing approvals and duplicate retries. The
production database was read-only checked as `PESSIMISTIC` on 2026-10-21: its locks
may serialize writers in a different order. No production race/IAM drill is claimed.
Synchronous route handlers use FastAPI's worker threadpool for database operations.
