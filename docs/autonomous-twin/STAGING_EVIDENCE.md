# Private runtime staging evidence

## T20 release qualification follow-up

Verified on 2026-09-10 after PR #597 merged as
`b16f9b344b671d3537f79713ba42e4864e2eb7eb`. The replacement production image
`us-central1-docker.pkg.dev/brave-drive-471109-d9/docker/analyzing-llm-rationale:sha-e2ce24c`
contains that commit and is deployed to both private services.

The protected PR test and Datastore jobs passed. The replacement Docker workflow
also passed its build, deploy, public health, readiness, and streamed forecast
smokes. A read-only authenticated private smoke then confirmed:

| Service | Ready revision | Traffic | Health | Readiness | Runtime authority |
| --- | --- | --- | --- | --- | --- |
| `twin-maintenance` | `twin-maintenance-00007-k8j` | 100% | `ok` | `ready` | shadow, zero live capital, no mandate |
| `twin-research` | `twin-research-00007-h4t` | 100% | `ok` | `ready` | shadow, zero live capital, no mandate |

The probe called only `GET /health` and `GET /ready`. A temporary
`roles/run.invoker` grant for the active operator identity was removed from both
services immediately afterward. The repository release verifier passed all ten
G0 checks with code hash
`a93473e027c1138a317daf0c1caf6c33235bd08ae4e2bff92acacc9b107f7261` and config
hash `18b6d5511deb296cecd686dd46dd56b8649bc84cfc574bb8bd7888d343125751`.

T20 therefore satisfies G0. T06 account synchronization and T07 research-worker
wiring remain prerequisites for the forward shadow trial.

---

## T17 initial staging

Verified on 2026-09-08 in GCP project `brave-drive-471109-d9`, region
`us-central1`. The runtime is shadow-only and carries zero live capital and no
live mandate.

## Release under test

- Main commit: `a21e6bf9529d7e682224bd32445176c2b9d37ff9`
- Immutable image tag: `us-central1-docker.pkg.dev/brave-drive-471109-d9/docker/analyzing-llm-rationale:sha-a21e6bf`
- Maintenance revision after restart: `twin-maintenance-00005-gvv`
- Research revision after restart: `twin-research-00005-l4r`

## Private request and identity checks

Requests were sent to the Cloud Run `status.url` audiences with Google-issued
service-account ID tokens. Only HTTP status codes were retained; tokens were not
written to this repository.

| Probe | Result |
| --- | --- |
| Anonymous maintenance health | `403` |
| Anonymous request with spoofed `X-CloudTasks-QueueName` | `403` |
| Dispatcher identity, maintenance health | `200` |
| Dispatcher identity, maintenance readiness | `200` |
| Scheduler identity, due-work dispatch | `200` |
| Research identity, maintenance execution handler | `401` |
| Research identity, typed research status handler | authenticated; missing fixture returned `409` |
| Dispatcher token with wrong audience | `401` |
| Expired-shaped unsigned token | `401` at Cloud Run; exact expiry behavior is fixture-tested in `tests/test_twin_worker_auth.py` |
| Dispatcher identity, research health/readiness | `200` / `200` |
| Scheduler identity, research service | `403` |

The Cloud Run IAM policies contain no anonymous member. `twin-research` grants
invocation only to `twin-task-dispatcher`. `twin-maintenance` grants invocation
to the dispatcher, scheduler and research identities; route-level identity
checks restrict each of those callers to its narrow handler.

## Data and key isolation

An access token impersonating `twin-research` received:

- `403` from the Datastore `runQuery` API for `TwinWorkerJob`.
- `403` from the KMS decrypt API for the exchange-connections key.

The project policy grants `roles/datastore.user` only to `twin-maintenance` for
the twin runtime. The KMS key policy grants
`roles/cloudkms.cryptoKeyDecrypter` only to `twin-maintenance`. Research has
secret accessor access only to `SCADS_AI_API_KEY`.

Temporary user-to-service-account token-creator bindings used to perform these
probes were removed after verification. The dispatcher account retains token
creator only for the Google-managed Cloud Tasks service agent and service
account user only for maintenance.

## Durable delivery, budgets and restart

- A durable recovery job named `t17-duplicate-probe-a21e6bf` was submitted twice
  with the same Cloud Tasks name. The first create succeeded and the second
  returned `AlreadyExists`.
- Datastore recorded the recovery job as `completed` with exactly one attempt.
- A research job named `t17-research-probe-a21e6bf` had a pre-reserved USD/token
  budget. The private research path claimed the reservation exactly once and
  durably completed the job as `degraded` with reason
  `research_pipeline_unconfigured`. This is the intended fail-closed staging
  adapter state and did not call a model.
- Both Cloud Run services were explicitly revised and restarted. Their
  readiness endpoints returned `200` afterward, and the completed recovery job
  remained present with one attempt.
- During 20 concurrent authenticated research requests, maintenance due-work
  dispatch returned `200` in 391 ms. All research fixture requests returned the
  expected missing-job `409`, demonstrating separate service capacity.

## Queues and scheduler

- `twin-research`: 2 dispatches/s, 2 concurrent, 5 attempts, 10-300 second
  backoff, one-hour retry window.
- `twin-maintenance`: 5 dispatches/s, 1 concurrent, 10 attempts, 5-300 second
  backoff, one-day retry window.
- `twin-due-work`: enabled every five minutes with the scheduler identity, exact
  maintenance audience and 120-second deadline. A forced run completed with an
  empty status object and a recorded `lastAttemptTime`.

## Current limitation

The deployed account-maintenance and research-pipeline operations intentionally
degrade until their real venue-account and research-capture adapters are wired.
This deployment proves the private runtime, IAM, queue, budget, duplicate and
restart boundaries. It is not evidence for a forward shadow trial or live
trading eligibility.

## Rollback

Pause `twin-due-work`, then delete the scheduler, both queues, both Cloud Run
services and the four twin service accounts using the commands in
`infra/twin/README.md`. No exchange credential or live authority is present in
this staging runtime.
