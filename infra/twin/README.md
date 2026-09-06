# Foresea private twin runtime

This directory defines the **shadow-only** private runtime used by T17. It is separate from the public `analyzing-llm-rationale` Cloud Run service. The deployer supplies an immutable Artifact Registry image tag and *secret names*, never secret values.

The default project/region mirror the repository deployment workflow: `brave-drive-471109-d9` and `us-central1`. Use a separate staging project where available.

## Architecture and authority

| Component | Identity | Can do | Cannot do |
| --- | --- | --- | --- |
| `twin-research` | `twin-research` service account | Read its model secret and public evidence; process ID-only research jobs | Read trading Datastore, enqueue tasks, decrypt exchange connections, invoke a live submission handler |
| `twin-maintenance` | `twin-maintenance` service account | Own durable account/budget/mandate mutations; enqueue both queues; decrypt the declared exchange key | Receive anonymous traffic or any live authority from deployment configuration |
| Cloud Tasks delivery | `twin-task-dispatcher` service account | Invoke the two private worker services with Google OIDC | Read secrets, mutate Datastore, decrypt exchange connections |
| Cloud Scheduler | `twin-scheduler` service account | Invoke maintenance `/internal/twin/dispatch` with Google OIDC | Invoke research or access trading state directly |

Cloud Run invoker IAM is service-wide. Therefore it is only the first check: worker handlers must verify Google OIDC issuer, exact service URL audience, and allowed service-account email for each route. Research may invoke only the typed maintenance research-job interface. A queue-name header is delivery metadata and is never authorization.

The repository does not assume Datastore can isolate entity kinds. Research has no Datastore role; its claim/result exchange must stay on narrowly typed maintenance endpoints. The script grants KMS decrypt only to maintenance and does not grant it to research.

## Dry run and apply

From this directory, first inspect the exact commands. `ResearchModelSecret` and `TradingKmsKey` are resource names, not values. The image must be a digest or a `sha-...` Artifact Registry tag built by CI.

```powershell
./deploy.ps1 `
  -Image 'us-central1-docker.pkg.dev/brave-drive-471109-d9/docker/analyzing-llm-rationale:sha-<commit>' `
  -ResearchModelSecret 'SCADS_AI_API_KEY' `
  -TradingKmsKey 'projects/brave-drive-471109-d9/locations/us-central1/keyRings/foresea-trading/cryptoKeys/exchange-connections'
```

Apply the reviewed plan only to a staging project:

```powershell
./deploy.ps1 -Apply `
  -ProjectId '<staging-project>' `
  -Region 'us-central1' `
  -Image 'us-central1-docker.pkg.dev/<staging-project>/docker/analyzing-llm-rationale:sha-<commit>' `
  -ResearchModelSecret 'SCADS_AI_API_KEY' `
  -TradingKmsKey 'projects/<staging-project>/locations/us-central1/keyRings/foresea-trading/cryptoKeys/exchange-connections'
```

Both Cloud Run services are private, scale from zero, use the same image, and receive `FORESEA_TWIN_MODE=shadow`, `FORESEA_TWIN_LIVE_CAPITAL=0`, and an empty live mandate. The script cannot accept a live-capital or mandate parameter. It creates independent queues:

| Queue | Dispatch/concurrency | Retry envelope | Purpose |
| --- | --- | --- | --- |
| `twin-research` | 2/s, 2 concurrent | 5 attempts, 10–300 seconds, one hour | Bounded read-only research |
| `twin-maintenance` | 5/s, 1 concurrent | 10 attempts, 5–300 seconds, one day | Account recovery, reconciliation and exits |

`twin-due-work` invokes maintenance every five minutes using a scheduler OIDC token. Side-effecting maintenance code must use stable job IDs, durable claims, and its own deadline; scheduler or queue retries are never permission to replay a venue submission.

## Staging evidence checklist

T17 remains incomplete until the private handlers and durable queue backend are mounted. Once they are present, record command output for all of the following:

```powershell
gcloud run services get-iam-policy twin-research --region us-central1 --project <staging-project>
gcloud run services get-iam-policy twin-maintenance --region us-central1 --project <staging-project>
gcloud projects get-iam-policy <staging-project> --format=json
gcloud tasks queues describe twin-research --location us-central1 --project <staging-project>
gcloud tasks queues describe twin-maintenance --location us-central1 --project <staging-project>
gcloud scheduler jobs describe twin-due-work --location us-central1 --project <staging-project>
```

Use Google-issued ID tokens to exercise a valid request, wrong audience, expired token, unauthorized identity, anonymous request, and a request carrying only a spoofed `X-CloudTasks-QueueName` header. Confirm that the research identity has no project `roles/datastore.*` grant, no KMS decrypt binding on the trading key, and cannot reach a maintenance execution route. Confirm it can only complete its typed claim/result interaction. Submit a single stable queue task ID twice and preserve the duplicate-delivery result, restart each worker, then verify maintenance remains responsive while research is saturated.

The repository unit tests cover the claim checks in `tests/test_twin_worker_auth.py`; staging is the authoritative proof of Google IAM and token issuance.

## Costs and cleanup

Costs come from Cloud Run requests/CPU/memory, Cloud Tasks operations, Scheduler jobs, Datastore operations/storage, KMS and Secret Manager operations, logs, network egress, and the configured research provider. Scale-to-zero reduces idle Cloud Run cost but does not make any component free. Keep shadow concurrency and the research budget bounded.

To remove a staging runtime, first pause work, then delete scheduled dispatch and queues before services and service accounts:

```powershell
gcloud scheduler jobs pause twin-due-work --location us-central1 --project <staging-project>
gcloud scheduler jobs delete twin-due-work --location us-central1 --project <staging-project> --quiet
gcloud tasks queues delete twin-research --location us-central1 --project <staging-project> --quiet
gcloud tasks queues delete twin-maintenance --location us-central1 --project <staging-project> --quiet
gcloud run services delete twin-research --region us-central1 --project <staging-project> --quiet
gcloud run services delete twin-maintenance --region us-central1 --project <staging-project> --quiet
gcloud iam service-accounts delete twin-research@<staging-project>.iam.gserviceaccount.com --project <staging-project> --quiet
gcloud iam service-accounts delete twin-maintenance@<staging-project>.iam.gserviceaccount.com --project <staging-project> --quiet
gcloud iam service-accounts delete twin-task-dispatcher@<staging-project>.iam.gserviceaccount.com --project <staging-project> --quiet
gcloud iam service-accounts delete twin-scheduler@<staging-project>.iam.gserviceaccount.com --project <staging-project> --quiet
```
