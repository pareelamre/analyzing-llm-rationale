# Serving the published payloads from GCS

The eight artifacts the tick workflows publish are committed to `main` and
read back over `raw.githubusercontent.com`. That works, and it is why the
repository is ~2.6GB: `static/mark_to_market_live.json` alone carries about
5,400 revisions of a 2.4MB file, republished every five minutes.

`data/track_record_store.duckdb` already made this move, for the same reason
(see the docstring at the top of `src/analyzing_llm_rationale/gcs_store.py`).
This is the same move for the payloads.

## What is already in place

Every reader accepts a GCS origin and is **inert until configured**. Nothing
below has been switched on.

| env prefix | object | approx revisions |
| --- | --- | ---: |
| `MARK_TO_MARKET` | `mark_to_market_live.json` | 5,360 |
| `TRACK_RECORD_LIVE` | `track_record_live.json` | 2,790 |
| `AGENT_TRADING_BOARD` | `agent_trading_live.json` | 667 |
| `FORECAST_EVALUATION` | `forecast_evaluation.json` | 566 |
| `AGENT_TRADING_AUDIT_ARCHIVE` | `agent_trading_audit_archive_manifest.json` | 135 |
| `AGENT_TRADING_AUDIT` | `agent_trading_audit_live.json` | 133 |
| `CRYPTO_KALSHI_EDGE` | `crypto_kalshi_edge_payload.json` | 35 |
| `CRYPTO_5M_EQUITY` | `crypto_5m_equity_payload.json` | 808 |

Each reads `<PREFIX>_GCS_BUCKET` and optionally `<PREFIX>_GCS_OBJECT`. An
unset or blank bucket leaves that reader on its existing
raw-GitHub -> bundled path, unchanged.

The uploads are already in the publishing workflows, each guarded by an unset
repository variable and **additive** -- the payload is still committed:

| workflow | repository variable |
| --- | --- |
| `track-record-tick.yml` | `MTM_PAYLOAD_BUCKET` |
| `track-record-resolved.yml` | `TRACK_RECORD_PAYLOAD_BUCKET` |
| `agent-trading-board-publish.yml` | `AGENT_TRADING_PAYLOAD_BUCKET` |

Two payloads have a reader but no upload step yet:
`crypto_5m_equity_payload.json` and `crypto_kalshi_edge_payload.json`. Their
commits ("crypto 5m: tick [skip ci]") are not produced by any workflow in
`.github/workflows`, so whatever publishes them -- as with the even-hour
agent-trading dispatch, likely a Cloud Scheduler job -- is where their
`gcloud storage cp` belongs. Their readers work the moment an object exists
at the configured bucket.

## Why the reader checks the generation

The obvious migration is to point the existing URL at a public GCS object --
no code at all. Do not: the read TTL is 30s against a 5-minute write cadence,
so roughly nine in ten reads re-download bytes the process already has. Free
from raw GitHub; egress from GCS, on the order of 7GB/day per Cloud Run
instance for the MTM payload alone.

`gcs_store.read_json_object` calls `blob.reload()` and downloads only when the
generation moves. A metadata failure keeps serving the last good payload
rather than dropping to a staler committed copy over one bad call.

## Activation, in an order that is reversible at every step

Do one payload first. `MARK_TO_MARKET` is the largest and its workflow already
authenticates to GCP for the duckdb sync.

1. **Bucket.** Either a new one or a prefix in the existing
   `brave-drive-471109-d9-track-record-store`.

2. **Read access** for the Cloud Run service account,
   `664177666636-compute@developer.gserviceaccount.com`
   (`roles/storage.objectViewer` on the bucket). The workflow already has its
   own credentials via `google-github-actions/auth@v2`.

3. **Start uploading, still committing.** Set the repository variable
   (`MTM_PAYLOAD_BUCKET`). The upload step stops being skipped. Nothing reads
   the object yet.

4. **Verify the object is fresh** before anything depends on it:

   ```bash
   gcloud storage ls -L "gs://$BUCKET/mark_to_market_live.json" | grep -E "Update time|Content-Length"
   ```

   It should track the 5-minute tick. If it does not, stop here -- nothing has
   changed for users.

5. **Point the reader at it.** Set `MARK_TO_MARKET_GCS_BUCKET` on the Cloud Run
   service. Confirm `/track` and `/edge/mtm` still render and that
   `freshness.age_seconds` moves.

6. **Only then remove the commit step** from `track-record-tick.yml`, in its
   own PR. This is the step that actually stops the repository growing.

Steps 1-5 are individually reversible: unset the variable and the reader falls
back to raw GitHub on its next cache miss. Step 6 is the one to take
deliberately.

## Coordination note

Setting a Cloud Run env var creates a new revision. If the service is also
deployed from a config that does not know about these variables, the next
deploy from that path will drop them and the reader will silently revert to
raw GitHub -- which fails safe, but quietly. Add the variable wherever that
config lives, not only via `gcloud run services update`.

## What this does not fix

Removing the commit steps stops the growth; it does not shrink the existing
~2.6GB. That would need history rewriting, which is a separate decision --
and note the two 66MB files that GitHub warns about on every push
(`data/kg_dataset.json`, the metaculus dataset) have 4 and 1 revisions
between them. They are not the problem; the republished payloads are.
