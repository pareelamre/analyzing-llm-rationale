# Polymarket SDK contract evidence — 2026-10-19

**Offline CTF signing and the private native backend are verified; route/worker
activation is not wired.** See [native execution follow-up](NATIVE_POLYMARKET.md)
for exact prepared-order persistence, encryption, one-send dispatch and recovery
tests. The original helper gap described below remains in manual/public routes.
The local optional dependency is `py-clob-client-v2==1.1.0`, imported as
`py_clob_client_v2`. The similarly named `py-clob-client` is a different
distribution: its absence does not establish that Foresea's SDK is absent.
No dependency was installed or upgraded for this check.

## Verification

Run with serving/development dependencies and the optional trading SDK installed:

```powershell
$env:PYTHONPATH = 'src'
py -m unittest tests.test_polymarket_sdk_contract
```

The test launches a sanitized child process with no inherited venue credentials.
It constructs Foresea's real SDK client using a public Anvil test key and dummy
API credentials, then calls the existing `_place_polymarket_order` helper.
Only SDK HTTP methods are intercepted. Every endpoint uses the reserved
`offline.invalid` host, and a Python audit hook denies connect/DNS and requires
zero attempted unmocked connections. This guard is not an OS sandbox.

The 24 cases cover limit/market BUY/SELL, EOA/proxy/Safe signature types 0/1/2,
and normal/negative-risk CTF contracts. Assertions inspect the actual serialized
SDK request: selected token, maker, signer, side, immediate FAK policy and exact
six-decimal amounts. At price 0.42 and size 10, BUY signs 4,200,000 cash units for
10,000,000 share units; SELL reverses those amounts. EIP-712 signatures recover
the test signer under the documented CTF version-2 domain and exchange address.
Synthetic delayed acknowledgements stay acknowledged with no inferred fill.

A separate SDK `create_order` call prepares a signed hash before any post. The
hash is saved as `client_order_id` through the existing `DatastoreTwinStore`
using SQLite, recovered after closing/reopening the database, and passed to the
real SDK's order-lookup and cancellation methods behind synthetic transport.
This proves SDK/local-store compatibility. It does not prove that an exchange
accepted an order, that a wallet is eligible, or that production recovery works.
The test skips explicitly if the optional SDK is absent; a skip supplies no
SDK verification evidence for a release.

## Production gap requiring T14/T15 follow-up

`reserve_confirmed_manual_order` derives a local Foresea identity before calling
the venue. `_place_polymarket_order` subsequently calls the SDK's
`create_and_post_order` or `create_and_post_market_order`. The native signed hash
is not returned to the reservation boundary before dispatch. An opaque local
Foresea ID must not be assumed to work as a Polymarket order hash.

The generic durable command store and recovery engine accept prepared identities,
but their callback tests do not establish this concrete venue binding. The
private worker factory is still shadow-only. T14 and the dependent Polymarket
integration in T15 are reopened; their earlier library and shadow evidence is
retained rather than discarded.

The next implementation must reuse these existing boundaries:

1. Build/sign once from the frozen intent, selected outcome, exact wallet/account,
   market protocol, tick, fees, size, price and immediate-order policy.
2. Persist the native lookup identity and exact request binding before dispatch;
   bind them to the existing account epoch, intent, reservation and command fence.
3. Recheck authority, freshness and limits, then post that same prepared object
   once. The SDK high-level helper can rebuild on a version update; it must not
   replace the persisted identity with a newly signed order during dispatch.
4. Validate acknowledgement identity and reconcile lost responses through that
   native hash. Retain reservations and pause when identity/acceptance is unknown.
   Test crashes before/after send and before receipt persistence with actual SDK
   objects; synthetic lookups alone cannot close the venue release gate.

No preparation capability or signed payload may reach a public board, research
worker, log, or unapproved execution path. Keep manual confirmation behavior and
the shadow-only worker defaults intact while completing the backend integration.

## Protocol/version boundary

Current official order documentation distinguishes CTF token IDs from Polymarket
Protocol V2 position IDs; their signing-domain versions are respectively 2 and 3.
The SDK's own package name ending in `v2` does not identify the market protocol.
[Official order documentation](https://docs.polymarket.com/trading/place-orders)
describes the asset selection and domains.

The official SDK 1.2.0 release adds `position_id` routing and version-3 signing
while preserving token-ID routes. This local 1.1.0 check does not verify those
new paths, deposit-wallet signature type 3, onchain wallet validation, allowance
or fee readiness, or region/account eligibility. Do not silently substitute a
position ID into the tested token-ID path. A live release needs its exact tested
dependency version and explicit protocol support.
[Official 1.2.0 release](https://github.com/Polymarket/py-clob-client-v2/releases/tag/v1.2.0)

The existing `pyproject.toml` lower bound is not an exact release pin. No change
to that dependency contract or migration to the newer unified SDK is made here.
