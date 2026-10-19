# Native Polymarket prepared execution — 2026-10-19

The private backend now prepares a real SDK CTF order, persists its native hash
and exact encrypted body with the existing reservation, and posts that same body
through the shared fenced execution engine. Manual/public routes and the private
worker factory are **not activated** on this backend. The subsequent
[opt-in manual wiring](NATIVE_MANUAL.md) connects the two manual HTTP paths, with
new native execution disabled by default. The private factory stays shadow-only.
T14/T15 remain open.

## Behavior

- Reuse `TwinStore`, immutable intents, account epochs, mandate/readiness/pause
  checks, account-root capacity transactions, command leases and lifecycle recovery.
- Prepare/sign once outside transactions; atomically persist the signed hash,
  fingerprint, encrypted request, reservation and outbox before a possible send.
- Check exact account maker, signer/type, selected outcome, condition, current
  instrument mapping, protocol, tick, size and six-decimal amounts. Reject silent
  rounding and a root reservation limit exceeding the current mandate. Loss holds
  cover cash, so concurrent cash commitments also remain within the root limit.
- An atomic `dispatch_started`/`dispatch_fence` marker permits one callback.
  Same-claim duplicates cannot change an active sender's state. A replacement
  worker cannot replay a started or legacy-unknown dispatch after lease expiry.
- A trusted pre-transport rejection releases the hold; an exception or conflicting
  acknowledgement after invoking the SDK retains funds as `submission_unknown`.
  Acknowledgement identity must equal the persisted native hash; no inferred fills.
- Positive order lookup checks the exact maker, token, condition, side, price and
  quantity. Missing/404/error is incomplete, never proof of absence. Cancellation
  requires durable state, current lease/fence and fresh positive lookup. Cancel
  acknowledgements retain funds for late-fill reconciliation.
- Generic recovery now refuses absence release for started/unknown dispatches.
  Lifecycle settlement requires explicit complete terminal evidence. A complete
  zero-fill cancellation may release capacity; an acknowledgement alone cannot.
  A settled hold cannot be released twice. Store release APIs take an explicit
  account scope, including scope IDs containing colons.

## Private storage and operating contract

Nonempty durable prepared requests use authenticated Fernet encryption, bound to
account scope and intent hash. The existing `cryptography` serving dependency is
reused. Set `FORESEA_TWIN_ORDER_ENCRYPTION_KEY` to an owner-generated random Fernet
key in the private runtime. There is **no default, plaintext fallback or paid KMS
call**. No key or venue credential is stored in the request, repository or report.
Keep the key outside SQLite backups/R2 objects, restrict its readers, and retain
it for recovery of existing ciphertext. Missing/changed keys block recovery;
drain existing commands before rotation. Legacy plaintext prepared packets are
rejected, not silently migrated. No production key has been provisioned here.

`ExecutionCommand` is an internal private object; its decoded body must never be
serialized generically with `asdict`/`vars`. Operator routes use explicit safe
projections. Logs, metrics and debug spans contain bounded outcomes/stages, no
signed body or SDK exception text. Encrypted requests remain available for exact
recovery; this does not claim cryptographic expiration of the native signature.
FAK/FOK packet expiration remains the tested SDK value `0`; do not invent a GTD
expiration rule for immediate orders.

Only `py-clob-client-v2` 1.1.0 and 1.2.0 are accepted. Submission requires the same
version that prepared the order. Supported CTF token paths use signing domain 2,
chain 137, EOA/proxy/Safe types 0/1/2 and the canonical CLOB host. Books containing
`version`, position IDs/domain 3, deposit-wallet signing and other SDK versions
remain blocked. Binary SELL support requires an explicitly permitting mandate;
this component does not establish available inventory, approvals or venue fees.

## Local verification

2026-10-19 evidence, without venue requests, model requests or cloud writes:

| Check | Result |
|---|---|
| Full `py -m unittest discover -s tests` with local in-memory Datastore emulator | 2,866 tests, no skips, exit 0, 255.696 seconds |
| Final native/dispatch/authority regressions after refinements | 42 tests, exit 0 |
| Native tests under isolated official SDK 1.2.0 | 28 tests, exit 0; no global dependency change |
| Direct SDK 1.2.0 offline contract probe | 24 signed cases; zero network requests; SQL native-identity compatibility |
| Local Datastore integration suite | 14 tests, exit 0; real emulator transactions, encrypted field exclusion and one atomic dispatch winner |
| Ruff over `src tests` | Pass |
| Real native OTel debug export | Prepared/submit spans and counters emitted; signed material absent; no external exporter |
| Deep review | `aud_GCsxMcJAZA`, five completed voices, one provider fallback; 26 findings normalized to 21 dispositions, 18 accepted/fixed and 3 rejected with evidence |
| Adjudication validation | `adj-6b2ylb8zcvkY5bRS`; local four-column validator passes |

The real installed SDK builds/signs/serializes orders; tests intercept its HTTP
methods and deny socket/DNS access. The isolated 1.2.0 wheel SHA256 is
`c549a801ae39af47e57c8c04546d3d365b3dc2fd5638e22b563e875a44814043`.
The local emulator is not production IAM, eventual-consistency or forward trading
evidence. Semgrep was unavailable locally (no CLI or running Docker engine);
this report does not claim a SAST pass. See [review dispositions](NATIVE_POLYMARKET_REVIEW.md).

## Remaining integration and release gates

1. Wire manual confirmation and private worker selection to this same prepared
   backend, with server-derived authority, authoritative market capture, inventory,
   fees, balances and approval readiness. Retain current shadow defaults.
2. Provision the declared `TwinAccount` owner/time index and private encryption-key
   access on an approved existing host; verify the exact deployable SDK/image,
   backup privacy and operator projections. No provisioning/deployment was done.
3. Collect independent compatible forward evidence using an available free/local
   BYOM endpoint. The historical signal screens found **zero** qualifying cohorts;
   Brier skill alone is not an executable trade edge after costs.
4. Require bounded, expiring owner mandate/capital/loss limits and all live gates.
   User support for live trading is not a configured finite mandate. Verify venue
   eligibility, actual responses, cancellation/fill/settlement and rollback before
   claiming production recovery or enabling orders.

Official references: [place orders](https://docs.polymarket.com/trading/place-orders),
[manage orders](https://docs.polymarket.com/trading/manage-orders),
[SDK contract configuration](https://github.com/Polymarket/py-clob-client-v2/blob/v1.2.0/py_clob_client_v2/config.py).
The configured addresses match the SDK's `exchange_v2`/`neg_risk_exchange_v2`,
not its older `exchange`/`neg_risk_exchange` fields. The official response contract
distinguishes accepted orders from false-flag/message rejections; contradictory
identity, trade, fill or acceptance evidence is deliberately treated as unknown.
