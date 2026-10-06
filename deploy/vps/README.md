# Migrating Foresea off GCP to a single VPS

## Why

The GCP bill is **~$86/month**, and ~$87 of that is Cloud Run compute. The
Datastore data layer — the hard part to move — costs roughly **$0** (free
tier). So the migration is not about saving money on data; it is about getting
the container onto a box that costs a flat **~€5–15/month**.

| | GCP (now) | VPS (target) |
|---|---|---|
| Compute | Cloud Run, 8 services, 859 instance-hours | 1 VPS, always on |
| Database | Datastore (~40 kinds) | SQLite on a volume |
| Object storage | GCS (2.3 GB) | Cloudflare R2 (already built) |
| Queue | Cloud Tasks | in-process / cron |
| Scheduler | Cloud Scheduler (5 jobs) | cron |
| Secrets | Secret Manager (14) | `.env` (chmod 600) |
| TLS | Cloud Run managed domain | Caddy + Let's Encrypt |
| **Cost** | **~$86/mo** | **~€5–15/mo** |

## What was built

| Piece | File | Status |
|---|---|---|
| Datastore→SQL shim | `src/analyzing_llm_rationale/datastore_sql.py` | ✅ tested |
| Backend switch | `src/analyzing_llm_rationale/datastore_backend.py` | ✅ tested |
| Call-site rewiring | 43 imports across 14 modules | ✅ done |
| Data copier | `scripts/migrate_datastore_to_sql.py` | ✅ tested |
| Cloud Tasks replacement | `LocalTaskDispatcher` in `twin/scheduler.py` | ✅ tested |
| Worker auth off GCP | shared-secret branch in `twin/runtime.py` | ✅ tested |
| Compose stack | `deploy/vps/docker-compose.yml` | ✅ written |
| TLS | `deploy/vps/Caddyfile` | ✅ written |
| Scheduler replacement | `deploy/vps/cron/` | ✅ written |
| Cloud Run job replacement | `metaculus-dispatch` compose service | ✅ tested |
| Instance bootstrap | `deploy/vps/bootstrap-oci.sh` | ✅ written |
| On-box deploy + rollback | `deploy/vps/deploy.sh` | ✅ written |
| CI deploy to the box | `.github/workflows/oci-deploy.yml` | ✅ written (opt-in) |

The shim is a **drop-in for the `google.cloud.datastore` surface the app
actually calls** — `Client`, `Key`, `Entity`, `PropertyFilter`, ancestor
queries, namespaces, transactions, `get_multi`/`put_multi`/`delete_multi`, and
the `=`, `<`, `<=`, `>`, `>=`, `IN` operators. It follows the pattern already
proven by `trackrec_store.py`, which has shipped a Datastore-shaped SQL store
for the track record for some time.

## Cutover runbook

### 0. Prerequisites

- A VPS with **2 vCPU / 4 GB RAM / 40 GB disk** or larger (Hetzner CX22 or
  equivalent). The RAG embedding model
  and torch need the RAM; 4 GB is the practical floor.
- Docker + Compose installed on the box.
- DNS access for `foresea.ink`.
- `gcloud` authenticated locally (for the data copy).

### 1. Copy the data (safe, read-only)

Run from a machine with GCP credentials. This does **not** touch Datastore —
it only reads — so it is safe to run while the old deployment is live.

```bash
FORESEA_DATASTORE_BACKEND=gcp python scripts/migrate_datastore_to_sql.py \
    --project brave-drive-471109-d9 \
    --output foresea.sqlite3

python scripts/migrate_datastore_to_sql.py --verify \
    --project brave-drive-471109-d9 --output foresea.sqlite3
```

If you have not run `gcloud auth application-default login`, the script falls
back to your active `gcloud auth login` session, so no extra setup is needed.

**Quiesce the writers before the final copy.** The twin runtime writes
continuously (`TwinPublicEvidenceCache` and `TwinResearchCapture` grow every
few minutes), so a copy taken while it runs is a point-in-time snapshot and
`--verify` will report those two kinds as off by a few. That is expected, not
corruption — but it means the copy is not a clean cutover point. Pause the
writer, let in-flight work drain, then copy:

```bash
gcloud scheduler jobs pause twin-due-work --location=us-central1
sleep 120   # let in-flight twin work finish
python scripts/migrate_datastore_to_sql.py --project brave-drive-471109-d9 \
    --output foresea.sqlite3
python scripts/migrate_datastore_to_sql.py --verify \
    --project brave-drive-471109-d9 --output foresea.sqlite3
gcloud scheduler jobs resume twin-due-work --location=us-central1
```

**Do not proceed unless `--verify` prints "All kinds match."**

### 2. Stand up the box

```bash
scp foresea.sqlite3 root@<vps>:/opt/foresea/
scp -r deploy/vps root@<vps>:/opt/foresea/deploy/
```

On the VPS, create `/opt/foresea/deploy/vps/.env` from the values currently
in Secret Manager and the Cloud Run env vars:

```bash
# Pull the secrets out of GCP once, then delete them from GCP after cutover.
for s in SCADS_AI_API_KEY GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET \
         GITHUB_CLIENT_ID GITHUB_CLIENT_SECRET SESSION_SECRET \
         GOOGLE_WEATHER_API_KEY SERPER_API_KEY TAVILY_API_KEY \
         FRED_API_KEY PREDICT_API_KEY; do
  echo "$s=$(gcloud secrets versions access latest --secret=$s \
        --project=brave-drive-471109-d9)"
done > .env
chmod 600 .env
```

Add the non-secret settings:

```bash
cat >> .env <<'EOF'
CUSTOM_DOMAIN=foresea.ink
INTERACTIVE_DEFAULT_MODEL=gemma-4-26b-a4b-it
INTERACTIVE_MAX_TOKENS=384
CHAT_PROVIDER_TIMEOUT_S=15
CHAT_PROVIDER_MAX_RETRIES=0
EVIDENCE_TIMEOUT_S=0
EVIDENCE_MAX_CONCURRENCY=4
FORESEA_ENABLE_BYO_TRADING=true
EOF
```

If you are also moving the twin runtime, add its dispatch settings. The
secret is shared between the dispatcher and the worker, so generate it once:

```bash
cat >> .env <<EOF
FORESEA_TWIN_DISPATCH_SECRET=$(openssl rand -hex 32)
FORESEA_TWIN_MAINTENANCE_URL=http://app:8000
FORESEA_TWIN_RESEARCH_URL=http://app:8000
GH_DISPATCH_TOKEN=<a GitHub token with actions:write>
EOF
```

Then place the database where the compose file expects it:

```bash
docker volume create foresea_state
docker run --rm -v foresea_state:/data -v /opt/foresea:/src alpine \
  cp /src/foresea.sqlite3 /data/foresea.sqlite3
```

### 2b. Install the cron schedule

Replaces the five Cloud Scheduler jobs:

```bash
install -d /opt/foresea/bin
install -m 755 /opt/foresea/deploy/vps/cron/foresea-cron.sh /opt/foresea/bin/
crontab /opt/foresea/deploy/vps/crontab
crontab -l   # confirm
```

The crontab appends to `/var/log/foresea-cron.log`; create it first and make
sure it is owned by the cron user, or every entry fails silently:

```bash
sudo touch /var/log/foresea-cron.log
sudo chown "$(id -u):$(id -g)" /var/log/foresea-cron.log
```

The `metaculus-dispatch` entry runs the app image with a different command
(`python -m analyzing_llm_rationale.metaculus_dispatch`), which is exactly what
the Cloud Run job did. It needs `METACULUS_GITHUB_DISPATCH_TOKEN` in `.env`.
Test it once by hand before trusting cron:

```bash
cd /opt/foresea/deploy/vps
docker compose --profile jobs run --rm metaculus-dispatch
```

### 3. Start it

```bash
cd /opt/foresea/deploy/vps
docker compose up -d --build
docker compose logs -f app
```

Wait for `/health` to return `{"status": "ok"}`. The first boot downloads the
embedding model into the `models` volume, so it takes a few minutes.

### 3b. Ongoing deploys and rollback

`deploy/vps/deploy.sh` (on the box) automates the update path:

```bash
sh /opt/foresea/deploy/vps/deploy.sh             # pull main, rebuild, health-gate
sh /opt/foresea/deploy/vps/deploy.sh --rollback  # restore the previous image
```

It records the running image ID before each deploy, polls `/ready` (probed
inside the app container, since port 8000 is not published to the host) for up
to 5 minutes after the restart, and auto-rolls back if the gate fails. On OCI
the same script is invoked by `.github/workflows/oci-deploy.yml` on every push
to `main` once the `OCI_*` secrets and the `OCI_DEPLOY_ENABLED=true` variable
are set (see [OCI.md](./OCI.md) §8).

### 4. Verify before moving DNS

The compose stack does not publish port 8000 — only Caddy's 80/443 are
exposed. Verify from inside the container, then through Caddy once DNS is
pointed:

```bash
docker compose exec app python -c \
  "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').read())"
docker compose exec app python -c \
  "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/ready').read())"
```

Then point `foresea.ink` at the VPS. Caddy issues the certificate on first
request. Confirm:

```bash
curl -fsS https://foresea.ink/health
```

### 5. Rollback

The old Cloud Run service is untouched until you delete it. To roll back,
point DNS back at Cloud Run — no redeploy needed. Keep the GCP project alive
for at least a week after cutover.

## Why not Vercel (or other serverless)

Vercel is the wrong shape for this app, and the blocker is architectural rather
than a limit you can pay to raise:

| Requirement | Vercel |
|---|---|
| Persistent SQLite file | ❌ project dir is read-only; `/tmp` is per-invocation and capped at **512 MB, not upgradable** |
| 2.45 GB image (torch + transformers) | ❌ 500 MB Python bundle (5 GB only in a beta) |
| Local RAG embedding model | ❌ no persistent cache; re-downloaded on cold start |
| Twin runtime + cron jobs | ❌ no long-running process model |

The database alone is 159.6 MB, so it would not even fit in `/tmp` alongside
anything else — and `/tmp` does not survive the invocation regardless.

Using Vercel would mean rewriting the storage layer to Postgres, which is the
one piece deliberately left unshipped (see "Known gaps"). That trades a
verified SQLite migration for an unverified Postgres one, and costs more:
Vercel Pro is $20/user/mo plus a database (Neon ~$19/mo), versus €5–15 for a
VPS.

## Host portability

The stack is provider-agnostic. The Docker image and SQLite file run
identically anywhere; only the hostname changes.

| Provider | Price | Specs | Location |
|---|---|---|---|
| **Hetzner CX22** | **~€4.5/mo** | 2 vCPU / 4 GB | Germany / Finland |
| Hostinger KVM1 | ₹599 (~€6.3) | 1 vCPU / 4 GB | Mumbai |
| E2E Networks | ₹1,100–1,200 (~€11–12) | 2 vCPU / 4 GB | Delhi / Mumbai |
| DigitalOcean | ~$24 (~₹2,000) | 2 vCPU / 4 GB | Bangalore |

Hetzner is the cheapest; India is comparable, not cheaper. The gain from an
Indian region is latency (~10–30 ms from Mumbai vs ~120–150 ms from Germany),
but this app is **LLM-latency-bound, not web-serving-bound**: every forecast
waits seconds on `llm.scads.ai` (a German institution) and the market APIs.
Hosting in India would also *add* latency to the SCADS calls.

Choose an Indian provider only for INR/GST billing or data residency — E2E
Networks is the cleanest for that (Indian company, no forex/RCM overhead).

To deploy on any of them, follow the runbook below unchanged; only the
`Caddyfile` domain and DNS differ.

### Measured memory profile

The app was run under a hard 1 GB cgroup limit to find the real floor:

| State | RSS |
|---|---|
| Idle, serving requests | 146 MB |
| Peak, RAG embedder loaded | **611 MB** |
| OOMKilled under 1 GB | no |

So **1 GB is the practical minimum** (611 MB peak + ~150–200 MB OS), and
**2 GB is comfortable**. Cloud Run was configured with 2 GiB in production,
which matches.

### Free tiers

| Option | RAM | Egress | Region | Arch |
|---|---|---|---|---|
| GCP `e2-micro` | 1 GB | **1 GB/mo, then $0.12/GB** | US only | x86 |
| **OCI Ampere** | **12 GB** | **10 TB/mo** | incl. Mumbai | **ARM** |

**The e2-micro's binding constraint is egress, not RAM.** `/radar` is 3.6 MB
and `/track-record` is 3.4 MB, so the 1 GB/month free allowance is roughly
**280 page loads** — against 1,885 visits in a recent month. A "free" tier
becomes a metered bill almost immediately, and it is US-only.

**OCI Ampere is the better free option** and the ARM concern is resolved:
PyTorch publishes aarch64 CPU wheels and the dependency set resolves cleanly
(verified with `pip install --dry-run --platform linux/arm64`). The compose
file builds for the host architecture by default, so no change is needed on an
Ampere instance. Note the free allowance was **halved to 2 OCPU / 12 GB in June
2026** and is tenancy-wide; the real cost is capacity scarcity — ARM instances
often need retries to provision.

**See [OCI.md](./OCI.md) for the full Oracle Cloud walkthrough**, including the
two-layer firewall that is the usual cause of "server up but unreachable".

## Known gaps (be honest about these)

1. **Postgres is not implemented.** The shim is SQLite-only. SQLite is fine
   for this workload (single writer, low write volume, WAL mode), but if you
   later need concurrent writers, the dialect layer is the follow-up.
2. **KMS is not replaced.** `model_providers.py` already falls back to a
   Fernet key derived from `SESSION_SECRET_KEY` when KMS is unavailable, so
   user API keys keep working — but the fallback is weaker than KMS. Rotate
   `SESSION_SECRET_KEY` deliberately and back it up.
3. **Google OAuth stays.** Sign-in uses Google/GitHub OAuth, which is not a
   GCP-hosted dependency and works unchanged off GCP.
4. **The shared secret is weaker than OIDC.** `LocalTaskDispatcher` and the
   worker share `FORESEA_TWIN_DISPATCH_SECRET`; the comparison is constant-time
   and fails closed, but it authenticates the caller rather than a Google
   identity. Keep the worker port off the public internet (the compose file
   only exposes Caddy).

## Verifying the shim

```bash
python -m unittest tests.test_datastore_sql tests.test_migrate_datastore_to_sql \
                   tests.test_local_dispatch
```

38 shim tests + 11 migration tests + 17 dispatch/auth tests cover key paths,
ancestors, namespaces, transactions, rich-type round-trips, the copy/verify
logic, queue routing, and both authentication paths.

## Verifying the app on the SQL backend (before deploying)

The strongest pre-deploy check is to boot the real server against the migrated
database and read live data back through the app's own endpoints:

```bash
export FORESEA_DATASTORE_BACKEND=sql
export FORESEA_DATASTORE_PATH=$PWD/foresea.sqlite3
export SCADS_AI_API_KEY=$SCADS_API_KEY      # note: server.py reads the _AI_ name
export SESSION_SECRET=$(openssl rand -hex 32)
export PYTHONPATH=src

analyze-llm-rationale serve --model gpt-oss-120b \
    --variant variant0_neutral_baseline --host 127.0.0.1 --port 8099
```

Then confirm the data-backed endpoints answer from SQLite:

```bash
curl -fsS localhost:8099/health          # {"status":"ok"}
curl -fsS localhost:8099/ready           # {"ready":true,...}
curl -fsS localhost:8099/analytics/summary
curl -fsS localhost:8099/track-record
curl -fsS localhost:8099/analytics/users
```

`/ready` returns 503 with `provider_configured: false` when
`SCADS_AI_API_KEY` is unset — that is a missing LLM key, not a storage
problem. `/health` and the data endpoints still work.

Two gotchas when running locally:

* `python -m analyzing_llm_rationale.cli` is a **no-op** — `cli.py` has no
  `if __name__ == "__main__"` guard. Use the `analyze-llm-rationale` console
  script (as above), which is what the container uses.
* Importing `server:app` directly under uvicorn leaves `_state` empty, so
  `/ready` stays 503. The `serve` command calls `init_server_state()` first.

## Verifying the container stack

The whole stack was built and run before this runbook was written:

```bash
cd deploy/vps
cp .env.example .env && chmod 600 .env   # fill in the secrets
docker compose config --quiet            # validates the compose file
docker compose build app                 # ~4 min, torch makes the image large
docker compose up -d app
docker compose ps                        # expect: Up (healthy)
```

Then load the migrated database and confirm the app serves it:

```bash
docker volume create foresea_state
docker run --rm -v foresea_state:/data -v "$PWD/../..:/src:ro" alpine \
  cp /src/foresea.sqlite3 /data/foresea.sqlite3
docker compose exec -T app python -c \
  "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/ready').read().decode())"
```

The backup service's core operation (a consistent SQLite snapshot) can be
checked independently:

```bash
docker run --rm -v foresea_state:/data:ro -v foresea_backups:/backups \
  -v "$PWD/cron:/scripts:ro" alpine sh /scripts/verify-backup.sh
# expect: PRAGMA integrity_check -> ok, and the entity count
```

### Shell scripts must keep LF line endings

`cron/foresea-cron.sh`, `cron/verify-backup.sh`, and `crontab` are run by
`sh` on Linux. With CRLF endings `sh` fails with `set: illegal option -` and
cron silently does nothing. `.gitattributes` pins these to `eol=lf`; if you
edit them on Windows, confirm with:

```bash
file deploy/vps/cron/foresea-cron.sh   # expect: ASCII text, not "with CRLF"
```
