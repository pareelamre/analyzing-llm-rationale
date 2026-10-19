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
