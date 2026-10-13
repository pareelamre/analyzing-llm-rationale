# Local progress under the cloud cost constraint

Owner instruction, 2026-10-13: a GCP bucket has been removed; make progress
without increasing billing. The owner reports existing Cloudflare buckets.
The deleted bucket's exact identity and live Cloudflare contents have not
been independently read back in this task. Do not infer those facts from a
historical deployment or recreate the bucket to make an old workflow succeed.

## Allowed work in this increment

Use local source, fake providers, local emulators and bounded evidence files.
Keep live trading disabled. Preserve immutable inputs and record exact source,
config and test provenance. Local fault tests validate mechanics; they do not
establish seven real forward days, account eligibility, live IAM or G2 strategy
performance.

Do not provision cloud services or buckets, redeploy, dispatch cloud jobs,
start a forward collector, call paid models, or merge to main while the Docker
workflow would trigger a cloud build/deployment. Do not perform storage reads or
writes merely to test whether an existing bucket works. Existing schedules and
resources may continue to incur charges; this document does not claim to pause
them or certify zero total billing.
The constraints recorded in `progress.json` guide agent execution; they are
not runtime kill switches and do not alter existing scheduled cloud jobs.

Reuse the existing R2 adapter when external storage is later necessary. First
verify the intended existing bucket, access isolation, bounded object retention
and request volume, and a cost envelope acceptable to the owner. Do not presume
that Cloudflare free-tier capacity is available or add a new storage stack.

## Fresh source findings

Inspected current main at `0b10028878b9ae48b5c789a4385c76c74547b752`.
The old task checkout at `e3d54fcb9a33745a514ffb6de1578abd12bbbc1d`
is 5,591 commits behind that main and contains unfinished September edits.
Those edits remain preserved in their original worktree. Work in this increment
uses a new checkout; no old financial code or stale tracker was transplanted.

GitHub Actions metadata was read without retrieving secret values or running
workflows. Variables `FORESEA_STORAGE_BACKEND=r2` and
`R2_STATE_BUCKET=foresea-state` are present; so are `R2_ENDPOINT_URL`,
`R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`. This verifies configured names
and the intended existing R2 destination, not object availability or charges.

- `src/analyzing_llm_rationale/r2_store.py` already provides an explicit R2
  state boundary. Backend defaults still require review at each consumer;
  the adapter's presence does not prove that every consumer migrated.
- `.github/workflows/metaculus-futureeval.yml` still restores write-ahead
  audit logs from `gs://brave-drive-471109-d9-track-record-store/metaculus/...`.
  `metaculus_audit_storage.py` uses GCS directly, with generation-conditional
  writes. If that bucket was deleted, keep forecasts fail closed until the audit
  history is restored and a tested conditional-write migration is ready. Missing
  history must not be treated as an empty audit, and an unconditional R2 copy is
  not a substitute for the generation concurrency check.
  The existing Actions run `38369027250`, created at `2026-10-13T08:24:19Z`,
  reports successful audit restore for all four bot profiles. That historic run
  does not identify the deleted bucket or prove access after the owner's deletion.
- `.github/workflows/docker.yml` still describes a separate GCS model volume.
  The owner's bucket removal does not prove that volume was removed or broken.
  No deployment or volume mutation was performed here.
- `infra/twin/README.md` documents requests, Datastore, queues, secrets, logs,
  network and provider costs. Shadow mode and scale-to-zero do not by themselves
  establish that a forward collector is free.

## Next implementation slices

1. Continue local replay and the outage, duplicate delivery, cancel/fill and
   restart regression tests; retain outputs as fixture verification.
2. Prepare the remaining Metaculus audit migration with conditional-write,
   missing-history, partial-download and archive-integrity tests. Preserve the
   existing audit and forecast submission safety boundaries.
3. Reassess T21 collection using existing evidence before proposing any running
   collector. Keep unresolved gates and live authority explicitly pending.

This constraint supersedes the earlier automatic deploy instruction. It does
not claim that the existing cloud services have stopped billing, that R2 is
free, or that T21/T22 are complete.
