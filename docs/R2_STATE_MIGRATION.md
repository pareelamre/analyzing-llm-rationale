# Move frequent state transfers to private R2 storage

This first slice moves `track_record_store.duckdb` and per-model trading databases
and notes. It does not migrate models, crypto payloads, dashboard JSON, Datastore
exports, or the Metaculus prefix. Those still have GCS consumers and publishers.
There are no live infrastructure changes in this PR.

## Behavior and configuration

`FORESEA_STORAGE_BACKEND` defaults to `gcs`. Setting it to `r2` requires
`R2_STATE_BUCKET`, `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, and
`R2_SECRET_ACCESS_KEY`. The endpoint is the account's HTTPS S3 endpoint, not a
public bucket URL. Keep the bucket private. Use bucket-scoped read/write
credentials for workflows and separate read-only credentials for Cloud Run.
Store credentials in GitHub secrets and Secret Manager, never repository variables.

The five state workflows accept the backend and bucket/endpoint as repository
variables. The runtime accepts the same settings as environment variables.
Changing the writer and runtime independently risks reading stale state.
Dashboard GCS readers remain independent of this selector.

R2 downloads pin the ETag observed in metadata and replace files only after a
complete download. The default GCS CLI still uses gcloud. Authentication and
transfer errors stop state workflows; optional downloads tolerate only confirmed
absent objects. A genuinely new trading model can initialize an absent database
and empty notes. A failed authorization or transfer must never reset history.
`TRACK_STORE_BUCKET_NAME` overrides the source-bucket allowlist; otherwise workflow
bucket settings determine it. Runtime configuration failures preserve the cached
local file, without falling back to GCS when R2 is selected.

R2 uploads validate JSON, SQLite integrity, or DuckDB readability as applicable
and copy the prior object into `<key>.previous` before overwriting it. If that
recovery copy fails, upload stops. Each key retains only one previous revision;
this protects against a single bad write, not multiple successive corrupt writes.
Download and restore-test that revision before enabling schedules. Schedule a
separate bounded daily backup for longer recovery coverage before production cutover.
The existing shared `track-record-store-gcs-write` concurrency group serializes
all three forecast database writers even when R2 is selected. Out-of-band writers
must remain paused or use that same dispatch path; uploads do not implement CAS.

## Cutover checklist

1. Create a Cloudflare account, enable R2, and create a private state bucket.
2. Configure credentials and bucket/endpoint settings without selecting R2 yet.
3. Merge the code and deploy the reader while the backend remains `gcs`.
4. Pause every state writer, including scheduled/manual GitHub workflows and
   Cloud Scheduler dispatches. Wait for active writers to finish. Record the
   original scheduler/workflow state so it can be restored exactly.
5. Download the final GCS objects for the forecast database and all trading
   databases/notes. Preserve object names. Record sizes and SHA-256 hashes.
6. Upload to R2, download to a separate verification directory, and compare
   SHA-256 hashes and sizes. Open DuckDB read-only and run SQLite integrity checks.
   Never publish these private files as GitHub release assets.
7. Set the runtime and repository backend to `r2` while writers remain paused.
   Verify history/explain-shift and run one controlled state workflow. Read back
   the resulting object and verify preserved history plus the new records.
8. Restore the recorded scheduling configuration. Watch errors, freshness,
   R2 usage and GCS billing. R2 storage and operation allowances are finite;
   bound backup retention rather than retaining every five-minute snapshot.

Rollback: pause writers again. Download the newest verified R2 state into an
intermediate local directory, then upload those same bytes to GCS. One CLI call
cannot transfer between backends. For each key, the POSIX-shell sequence is:

```sh
FORESEA_STORAGE_BACKEND=r2 PYTHONPATH=src python -m analyzing_llm_rationale.r2_store cp \
  gs://brave-drive-471109-d9-track-record-store/track_record_store.duckdb /secure/restore.duckdb
sha256sum /secure/restore.duckdb
gcloud storage cp /secure/restore.duckdb gs://brave-drive-471109-d9-track-record-store/track_record_store.duckdb
gcloud storage cp gs://brave-drive-471109-d9-track-record-store/track_record_store.duckdb /secure/readback.duckdb
sha256sum /secure/readback.duckdb
```

Compare both hashes and database integrity before selecting `gcs` in runtime and
repository configuration. Repeat for every trading database and notes key.
If current R2 state is corrupt, use its `.previous` object, or the frozen pre-cutover
GCS snapshot if necessary; explicitly record any unrecoverable newer records.
Restore schedules only after validation. Keep intermediate private files outside Git.

Do not delete the GCS bucket: this slice leaves other objects and jobs there.
Do not promise a zero GCP bill: existing soft-deleted data, other GCS objects,
Cloud Run, backups, and other services remain billable. This also retains the
whole-database transfer architecture; persistent local scheduling is a later slice.
