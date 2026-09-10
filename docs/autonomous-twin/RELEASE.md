# Autonomous twin shadow release

G0 is the engineering release gate for the private shadow runtime. It does not
authorize live trading. The checked-in sample
`tests/fixtures/twin/release_readiness_v1.json` proves the artifact schema and
hashing rules; a real G0 claim additionally requires green protected-branch
checks for the exact PR commit.

## Reproduce G0

Run these commands from a clean checkout of the candidate commit:

```powershell
py -m unittest discover -s tests
py -m ruff check src tests
py -m ruff check scripts --select F821,F811,F632,F502,F522,F701,B002
npm ci
npm run frontend:build
py scripts/verify_twin_release.py
py -m graphify update .
git diff --check
```

The `datastore-integration` CI job separately starts the Datastore emulator and
runs `tests.test_twin_store_integration`. G0 requires both CI jobs to pass. The
release verifier confirms the CI steps, emulator job, network-deny test,
shadow-only worker deployment, separate health/readiness checks, and rollback
contract remain present. Its hashes cover the twin source and operational
configuration, so the output identifies the reviewed candidate without using a
mutable branch name.

## Stage and smoke

Deploy the exact immutable image with `infra/twin/deploy.ps1 -Apply`. The deploy
script then calls `infra/twin/smoke.ps1`, which performs read-only authenticated
checks against both private services. It verifies `/health` and `/ready`
separately, confirms one ready revision receives 100% of traffic, and inspects
the deployed environment for `shadow`, zero live capital, and no mandate. The
smoke script cannot enqueue work or call a trading route.

The smoke script uses the active gcloud identity by default. Grant that identity
`roles/run.invoker` on both private services for the duration of the probe, then
remove the binding. Alternatively, pass `-InvokerServiceAccount` to the smoke or
deploy script. Impersonation requires temporary `roles/iam.serviceAccountTokenCreator`
and `roles/iam.serviceAccountUser` bindings on that service account; remove both
immediately afterward.

Record the image digest, ready revisions, service roles, smoke output, PR commit,
CI run URLs, artifact hash, source hash, and config hash in release evidence.
Treat a healthy but unready service as a failed release.

## Migration compatibility

Twin ledger schema version 1 is additive and the release artifact records both
the writer version and minimum rollback reader version. Existing manual and
autonomous commands share stable intent, command, client-order, reservation,
event, fill, and settlement identities. A release must rebuild projections from
the existing ledger in tests; it must never rename, reset, or delete those
records during deploy or rollback.

Before introducing schema version 2, add dual-read tests using version 1
fixtures, write the migration as an additive projection change, and prove the
previous reader can still pause and reconcile every outstanding command.

## Rollback

1. Pause new exposure globally and stop scheduler dispatch.
2. Reconcile `submitting`, `submission_unknown`, `cancel_requested`, and partial
   fills using the persisted venue identity. Do not resend an order.
3. Route traffic to the last known-good immutable image revision.
4. Keep Datastore, jobs, reservations, events, fills, and settlements intact.
5. Run `infra/twin/smoke.ps1` and rebuild account/command projections.
6. Resume shadow scheduling only after fresh G0 evidence matches the restored
   code and config hashes.

G1 remains the seven-day forward shadow trial. G2 is strategy evidence. G3
requires explicit owner capital authorization. Passing G0 cannot satisfy or
bypass any later gate.

