# Operator-owned model configuration for the shadow twin

This bundle replaces the twin's SCADS research model with an operator-supplied
OpenAI-compatible endpoint. It changes no public chat or forecasting model and
contains no credential. The twin remains in shadow mode with zero live capital.

Start with the [delivery acceptance and limitations](../../docs/autonomous-twin/BYOM_DELIVERY.md).
Its network-disabled Linux image check runs a synthetic model-to-shadow-order
cycle, verifies accounting and accepted-result recovery, and needs no SCADS or
other model subscription. Passing it establishes the shadow package's software
contract; real-provider/venue and live-trading evidence are separate milestones.

1. Copy `twin.yaml` and `models.yaml` to a private directory outside the repo.
   Set `models.yaml`'s `local_model_name`, `router_model_name`, and
   `api_base_url` to the exact model identity and endpoint you operate. Both
   private worker roles must see the same operator-owned, read-only files at
   the same absolute path. The strategy, shadow-trial, and replay sections
   must match Foresea's bundled `configs/twin.yaml`; startup rejects changes
   to them. Refresh the copied bundle after a Foresea upgrade. Do not edit
   the shared `configs/models.yaml` to distribute a twin.
2. Set `FORESEA_TWIN_CONFIG_DIR` to that absolute directory on both workers.
   Set `FORESEA_TWIN_BYOM_API_KEY` only on the research worker. This exact name
   is required so a copied config cannot select another worker secret. A local server
   that ignores bearer tokens can use a non-secret placeholder. Do not commit
   a real key, put it in either YAML file, or expose it through the operator UI.
3. The example has a zero USD budget because it assumes a locally operated
   model with no metered API charge. Keep the token, request, timeout, and
   fanout ceilings. For a metered provider, enter its current input/output
   prices and an owner-approved positive daily USD limit before use. Unknown
   prices fail closed. A remote endpoint with zero listed prices also fails
   closed unless you explicitly set `zero_cost_remote_attested: true` after
   verifying that no API usage is billed. `price_valid_until` must be renewed
   with price review.
4. Keep the model endpoint private. `127.0.0.1` in the example means the
   worker's own network namespace; change it if the model runs elsewhere.
   The twin accepts plaintext HTTP only for loopback URLs; use HTTPS for any
   other host. Test model compatibility and usage accounting with synthetic requests in
   an isolated environment before any shadow collection.

`compose.yaml` packages both private roles for an existing Linux host. It uses
Foresea's existing CPU image and app factory, with a persistent maintenance SQL
volume and separate role credentials. The native SQL subprocess regression
checks startup, loopback delivery, authorization and restart; a container-host
smoke and bounded forward shadow trial remain release gates. Durable accepted-result
recovery from PR #683 is integrated: receipts use the existing maintenance SQL
volume, without a research database mount or another cloud service. Live order
submission remains gated by T21/T22 and an explicit owner mandate; installing
this config does not grant trading authority.

## Run the private workers on an existing Linux host

This is an opt-in shadow package. Starting the workers installs no schedule and
makes no model or venue request with a new, empty state volume. Resuming an
existing volume may recover due work. Stop dispatch and review pending jobs
before a planned restart; automatic crash/reboot restarts intentionally recover
previously authorized pending work. Reserve the fixed ports 8081/8082 and use
an unused project name for an isolated trial. These ports are repeated in the
listener, peer URL and healthcheck configuration; changing them requires
coordinated edits to all three. The default volume is
`foresea-twin_state`. Never mount the public app's SQL database into a trial.

The package uses [Linux host networking](https://docs.docker.com/engine/network/drivers/host/)
so worker-to-worker HTTP and a host-local model endpoint remain on loopback.
Both Uvicorn listeners bind to `127.0.0.1`; do not change them to `0.0.0.0`,
add a public proxy, or add port mappings. Host networking shares access to host
services. Use a dedicated trusted host with no untrusted local users/services;
a separate OS account alone does not prevent another process taking a loopback
port and intercepting its bearer secret. Loopback supplies no server identity.
The host must expose no instance-principal or other cloud credentials through
metadata endpoints to the containers; use a host without that identity, or
verify host-level metadata isolation before activation. This is a target-host
release prerequisite, including on OCI; this package does not configure that
firewall or prove host isolation. Trust the image and model config.
The package is intended for Linux, including OCI A1; Docker Desktop networking
is not a verified runtime for this bundle.

Create a private directory outside the checkout with mode 700, copy the two
config files there as described above, and create mode-600 `maintenance.env`
and `research.env`. Generate three distinct 64-character hex values with
`openssl rand -hex 32`. Do not paste keys into shell history or a PR.

| File | Required values |
| --- | --- |
| `maintenance.env` | `FORESEA_TWIN_MAINTENANCE_DISPATCH_SECRET`, `FORESEA_TWIN_RESEARCH_DISPATCH_SECRET`, `FORESEA_TWIN_RESEARCH_GATEWAY_SECRET`; optional `FORESEA_TWIN_MAINTENANCE_URL=http://127.0.0.1:8081` for cron |
| `research.env` | The same research-dispatch and research-gateway secrets, plus `FORESEA_TWIN_BYOM_API_KEY` |

The research file must never contain the maintenance-dispatch secret. The
maintenance file needs no model, venue, GCP or R2 key. Keep only these variables
in the role files: `env_file` imports every line, so adding a key shares it with
that role. Compose overrides the SQL, shadow and loopback settings explicitly.

Set these non-secret paths in the shell (example installation paths):

```bash
export FORESEA_BYOM_CONFIG_DIR=/opt/foresea-twin/config
export FORESEA_BYOM_MAINTENANCE_ENV_FILE=/opt/foresea-twin/maintenance.env
export FORESEA_BYOM_RESEARCH_ENV_FILE=/opt/foresea-twin/research.env
```

Use an existing image built from this revision, or build it locally with the
existing Dockerfile. Building downloads dependencies and consumes host resources;
it provisions no service. From the repository root:

```bash
docker compose -f deploy/byom/compose.yaml --profile twin build
docker compose -f deploy/byom/compose.yaml --profile twin up -d --no-build
curl --fail http://127.0.0.1:8081/ready
curl --fail http://127.0.0.1:8082/ready
```

Set `FORESEA_BYOM_IMAGE` to a local image tag if reusing a build. Each worker
is limited to one CPU and 1 GiB; the local model runs separately and needs its
own capacity. The workers download no embedding model. This bundle starts no
public app, remote telemetry exporter, cloud resource, backup uploader or paid model.
Research requests begin only after dispatching work, using the configured BYOM
endpoint and bounded research policy.

Before any collection, run the offline regression with serving dependencies
installed: `PYTHONPATH=src python -m unittest tests.test_twin_byom_package`.
It launches real loopback workers, blocks Python-level non-loopback connect/DNS,
sends synthetic recovery work and replays a previously completed synthetic
research result through the research gateway, kills both processes and verifies
completed SQL work after restart. Its guard is not a network sandbox for native
extensions or subprocesses. Concurrent dispatch is bounded: excess callers
receive 503 with `Retry-After`, leaving work durable for the next dispatch.
This proves the native runtime, not a built container or live venue behavior.

The separate offline test `PYTHONPATH=src python -m unittest
tests.test_twin_sql_recovery` injects finalizer failure after a synthetic result
is saved, then retries in two fresh processes using the same SQLite file. It
verifies direct gateway and BYOM app-factory persistence: one research operation,
one budget authorization, successful replay and no repeated finalization on a
terminal duplicate. Business operations and finalization are synthetic; this
is not an end-to-end provider or forward-trading test. A result lost before
maintenance durably accepts it is outside this recovery boundary. Such unknown
usage remains conservatively accounted for; this change does not authorize
repeating a paid model call.

Do not start collection until the target-host smoke, host isolation and model
pricing checks pass. After those release checks,
to collect shadow evidence after reviewing endpoint pricing and pending state,
use the existing VPS `foresea-cron.sh twin-due-work` command with
`FORESEA_TWIN_MAINTENANCE_ENV_FILE` pointing at the private maintenance file
and `FORESEA_ENV_FILE` pointing at an empty private file. Add
`FORESEA_TWIN_MAINTENANCE_URL=http://127.0.0.1:8081` to that maintenance file
for cron. A manual dispatch creates work and can call venues and the model.
Install a 15-minute cron entry only when intentionally starting a bounded trial;
do not run a second Cloud Scheduler or other dispatcher for the same SQL state.

Stop dispatch first, then run
`docker compose -f deploy/byom/compose.yaml --profile twin down` to stop workers.
The SQL volume survives `down`, image rebuilds and container replacement. Never
use `down -v` on state you need. This package has a dedicated local SQL store;
it does not migrate production Datastore or replace the private R2 backup
destination. No R2 or GCS operation is automatic here.

Before upgrades, stop dispatch and the workers, inspect pending job counts, and
take a consistent SQLite backup using Python's standard backup API. These
one-off commands override the service entrypoint and do not start a worker:

```bash
docker compose -f deploy/byom/compose.yaml --profile twin stop
docker compose -f deploy/byom/compose.yaml --profile twin run --rm --no-deps maintenance python -c 'from collections import Counter; from analyzing_llm_rationale.datastore_sql import Client; c=Client(path="/data/twin.sqlite3"); print(Counter((e.get("kind"), e.get("status")) for e in c.query(kind="TwinWorkerJob").fetch() if e.get("result_json") is None)); c.close()'
docker compose -f deploy/byom/compose.yaml --profile twin run --rm --no-deps maintenance python -c 'import sqlite3; from datetime import datetime, timezone; p="/data/backup-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".sqlite3"; src=sqlite3.connect("file:/data/twin.sqlite3?mode=ro", uri=True); dst=sqlite3.connect(p); src.backup(dst); dst.close(); src.close(); print(p)'
```

Copy the printed backup filename out of the volume with `docker compose -f
deploy/byom/compose.yaml --profile twin cp maintenance:/data/<printed-filename>
/private/backup-directory/`. Keep that destination private and verify it with
`sqlite3.connect(path).execute("PRAGMA integrity_check").fetchone()` before
upgrading. Restore only with workers and dispatch stopped, retaining the old
file for recovery. A snapshot kept only inside the same volume is not an
independent backup. The existing private R2 backup process can mirror the
exported snapshot after its destination/credentials are configured separately.

The operator-configured shadow worker reads Foresea's bundled, dated
`static/track_record_live.json`
to prioritize already discovered markets from domains whose corrected published
skill interval has a positive lower bound over at least 30 distinct markets.
The method must be `market_means_normal_bonferroni_v1`, with a comparison count
covering the domain family. Legacy snapshot-only reports do not prioritize
research. It uses the same question-domain tagger as
the published track record. The report's declared generation time must predate
the decision and be at most seven days old; otherwise discovery order is unchanged. A changed order
stores the artifact hash with the market capture. The hash identifies the
bytes used; it is not a producer signature or an archive. Updating this
bundled report requires a new build. This is a research-allocation prior,
not proof that the BYOM model is calibrated: live sizing still requires the
twin's own prospective, audit-grade resolved forecasts. The selected domain
mix can affect later evaluation, so compare calibration within domain and
selection cohort before promoting any live strategy. The publisher's new normal
intervals average within each venue/market before computing uncertainty and
apply a Bonferroni multiplier within each displayed bucket family. Snapshot
averages remain descriptive; `skill_ci_mean` is the separate market-mean
estimate. Markets may still share an event, and normal intervals are approximate.
Malformed identity/error data makes a bucket's inference unavailable
(`n_markets=0`, null interval) even when descriptive snapshot `n` is positive.
No executable-return/cost or BYOM-version claim follows from Brier skill.
At small market counts and with skewed data, normal intervals can be too narrow;
the within-family adjustment also does not cover repeated report publications
or selecting among different models/families. These approximate research
intervals are not a substitute for frozen, event-disjoint forward validation.
This remains a cheap research heuristic. The bundled file
is trusted as part of the operator-owned code image; a signed or otherwise
attested report is still needed before using the prior in a live release.

The shadow benchmark's horizon reader keeps its historical 14–30 day fallback
when the report is unreadable, as required by the existing compatibility policy.
That fallback is not statistical evidence and can allow more shadow sizing than
a readable legacy report, which now validates no horizon. It must not satisfy a
live-release evidence gate. The twin's own live risk/calibration gates remain
separate. Recompute reports from raw snapshots; never add method labels to an
old interval to make it pass. Changing shared evaluation source also invalidates
replay code hashes: the checked-in synthetic `REPLAY_BASELINE.json` is regenerated
by its documented fixture command, without granting live eligibility.

If the two worker roles load different model or budget settings, the stored
research configuration ID differs and the research operation degrades before
calling the model. An existing queued assignment from the old ID likewise
fails closed after a configuration change; clear or reconcile stale work
before collecting new shadow evidence.
