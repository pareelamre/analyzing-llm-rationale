# Independent FutureEval trigger

GitHub Actions remains the only forecast runner. Its cron is a best-effort backup;
Cloud Scheduler triggers a small authenticated Cloud Run **job**, which dispatches
`metaculus-futureeval.yml` on `main` with `submit=true`. The dispatcher does not
hold Metaculus, SCADS or GCS credentials. Forecast identity, enable switch,
default-branch restriction, concurrency and write-ahead audit remain in Actions.
The fixed dispatcher assumes the repository default branch remains `main`.
Disabling bots skips automated live jobs entirely; explicit submit=false human
previews remain available. Existing workflow concurrency is
`group: metaculus-futureeval-bots`, `cancel-in-progress: false`.

The job checks active workflow runs before dispatch. This is coalescing, not an
atomic distributed lock: simultaneous triggers can race. Existing Actions
concurrency serializes forecasts, and authoritative prior-forecast checks skip
already forecasted questions. Uncertain dispatch POSTs are not retried by the
process. Job task retries MUST be zero. Scheduler retries MUST be zero to avoid
automatic replay of uncertain job-start responses. Later normal ticks recover.

## Provisioning boundary

Activate only after reviewed changes merge and a CPU Foresea image containing
`analyzing_llm_rationale.metaculus_dispatch` is built. Pin its immutable digest;
do not use a moving tag. These commands are an operator runbook, not an automatic
CI deployment. Replace `REVIEWED_IMAGE_AT_SHA256_DIGEST` before deployment.

Create a fine-grained GitHub token restricted to this repository with **Actions:
read and write**. Save it locally as `METACULUS_GITHUB_DISPATCH_TOKEN`; do not put
its value in shell arguments, scheduler headers, documentation or chat. Upload
through Secret Manager's stdin API. Runtime environment injection MUST use a
Secret Manager binding, not `--set-env-vars` with the credential value. Choose an
expiry covering the tournament and record its date separately; expiry produces
a failed dispatch job, not a successful no-op.

```bash
PROJECT=brave-drive-471109-d9
REGION=us-central1
RUNTIME=metaculus-dispatch@$PROJECT.iam.gserviceaccount.com
INVOKER=metaculus-scheduler@$PROJECT.iam.gserviceaccount.com
gcloud iam service-accounts create metaculus-dispatch --project "$PROJECT"
gcloud iam service-accounts create metaculus-scheduler --project "$PROJECT"
gcloud secrets create metaculus-github-dispatch --replication-policy automatic --project "$PROJECT"
# Read from the environment and supply stdin; never put the value in argv.
python -c 'import os, subprocess; subprocess.run(["gcloud", "secrets", "versions", "add", "metaculus-github-dispatch", "--data-file=-", "--project", "brave-drive-471109-d9"], input=os.environ["METACULUS_GITHUB_DISPATCH_TOKEN"].encode(), check=True)'
gcloud secrets versions describe 1 --secret metaculus-github-dispatch --project "$PROJECT" --format='value(state)'
# Proceed only if version 1 exists and is ENABLED.
gcloud secrets add-iam-policy-binding metaculus-github-dispatch --project "$PROJECT" \
  --member "serviceAccount:$RUNTIME" --role roles/secretmanager.secretAccessor
gcloud run jobs deploy metaculus-github-dispatch --project "$PROJECT" --region "$REGION" \
  --image REVIEWED_IMAGE_AT_SHA256_DIGEST --service-account "$RUNTIME" \
  --command python --args=-m,analyzing_llm_rationale.metaculus_dispatch \
  --set-secrets METACULUS_GITHUB_DISPATCH_TOKEN=metaculus-github-dispatch:1 \
  --tasks 1 --parallelism 1 --max-retries 0 --task-timeout 120s \
  --cpu 1 --memory 512Mi
gcloud run jobs add-iam-policy-binding metaculus-github-dispatch --project "$PROJECT" --region "$REGION" \
  --member "serviceAccount:$INVOKER" --role roles/run.invoker
gcloud scheduler jobs create http metaculus-github-dispatch --project "$PROJECT" --location "$REGION" \
  --schedule '3,13,23,33,43,53 * * * *' --time-zone Etc/UTC \
  --uri "https://run.googleapis.com/v2/projects/$PROJECT/locations/$REGION/jobs/metaculus-github-dispatch:run" \
  --http-method POST --headers Content-Type=application/json --message-body '{}' \
  --oauth-service-account-email "$INVOKER" --max-retry-attempts 0 --max-retry-duration 0s
```

Runtime SA gets access to this one secret only. Scheduler SA gets invocation of
this one job only. Neither gets project-wide editor or forecast credentials.
Cloud Scheduler's service agent retains its platform-required role; the operator
needs service-account `actAs` and provisioning permissions. Enable Cloud Run,
Secret Manager and Cloud Scheduler APIs if not already enabled. Check existing
resources before creating them; do not overwrite a conflicting job/secret.

## Acceptance and recovery

1. Execute the job once and verify a safe `dispatched` or `skipped` outcome. A
   dispatched event is not proof of a forecast; inspect the matching Actions run.
2. Verify Scheduler ticks through a real question window and authoritative
   forecast/private-comment readback for every profile. Do not infer readiness
   from an empty-question successful run.
3. Monitor failed job executions, token expiry and gaps longer than 20 minutes
   in Actions runs during open windows. Alerts are an operator responsibility;
   this patch does not provision alert policies or guarantee start latency.
4. Workflow handles at most five new questions/profile/run with a 65-minute
   cap: five times the previous 12-minute allowance plus five minutes of setup.
   This is a conservative provisioning bound, not measured p95 or a guarantee
   that network failures cannot exhaust it. Existing per-question model budgets
   remain unchanged. Audit quarantine stops
   unsafe repeat submissions after uncertain forecast/comment outcomes.
5. Pause this scheduler job to disable the independent trigger. Set repository
   `METACULUS_BOTS_ENABLED=false` to stop both triggers from submitting forecasts.
   Rotate GitHub credentials through a new Secret Manager version and explicitly
   update the job's version binding; never log the credential.

Verify scheduler state with `gcloud scheduler jobs describe metaculus-github-dispatch --location us-central1 --project brave-drive-471109-d9 --format='yaml(state,lastAttemptTime,status)'`
and executions with `gcloud run jobs executions list --job metaculus-github-dispatch --region us-central1 --project brave-drive-471109-d9`.
No guarantee is made for windows shorter than a trigger interval plus GitHub
queue/model latency, service outages or unbounded question backlog. Cloud
Scheduler reduces dependence on GitHub cron; it cannot guarantee deadlines.
Observability uses the existing stdout decision-span mirror collected by Cloud
Logging. OTel metrics have no remote exporter in this deployment; no dashboard
or Cloud Trace delivery is claimed, and no telemetry writer roles are needed.

References: [GitHub dispatch API](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event),
[Cloud Scheduler OAuth](https://docs.cloud.google.com/scheduler/docs/http-target-auth).
