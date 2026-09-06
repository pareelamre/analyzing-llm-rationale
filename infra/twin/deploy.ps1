[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Image,
    [string]$ProjectId = "brave-drive-471109-d9",
    [string]$Region = "us-central1",
    [Parameter(Mandatory)]
    [string]$ResearchModelSecret,
    [Parameter(Mandatory)]
    [string]$TradingKmsKey,
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# This script is intentionally dry-run by default.  It never accepts secret
# values, a live-capital value, or a mandate ID.  `-Apply` is required for
# every Google Cloud mutation.
function Invoke-Gcloud {
    param([string[]]$GcloudArgs)
    Write-Host ("+ gcloud " + ($GcloudArgs -join " "))
    if ($Apply) {
        & gcloud @GcloudArgs
        if ($LASTEXITCODE -ne 0) {
            throw "gcloud command failed: $($GcloudArgs -join ' ')"
        }
    }
}

function Ensure-ServiceAccount {
    param([string]$Name, [string]$DisplayName)
    $email = "$Name@$ProjectId.iam.gserviceaccount.com"
    if ($Apply) {
        & gcloud iam service-accounts describe $email --project $ProjectId 2>$null
        if ($LASTEXITCODE -ne 0) {
            Invoke-Gcloud @("iam", "service-accounts", "create", $Name, "--display-name", $DisplayName, "--project", $ProjectId)
        }
    }
    else {
        Write-Host "+ ensure service account $email"
    }
    return $email
}

function Ensure-Queue {
    param([string]$Name, [string]$MaxAttempts, [string]$MaxRetryDuration, [string]$MinBackoff, [string]$MaxBackoff, [string]$MaxDispatches, [string]$MaxConcurrent)
    if ($Apply) {
        & gcloud tasks queues describe $Name --location $Region --project $ProjectId 2>$null
        if ($LASTEXITCODE -eq 0) {
            Invoke-Gcloud @("tasks", "queues", "update", $Name, "--location", $Region, "--project", $ProjectId,
                "--max-attempts", $MaxAttempts, "--max-retry-duration", $MaxRetryDuration,
                "--min-backoff", $MinBackoff, "--max-backoff", $MaxBackoff,
                "--max-dispatches-per-second", $MaxDispatches, "--max-concurrent-dispatches", $MaxConcurrent)
            return
        }
    }
    Invoke-Gcloud @("tasks", "queues", "create", $Name, "--location", $Region, "--project", $ProjectId,
        "--max-attempts", $MaxAttempts, "--max-retry-duration", $MaxRetryDuration,
        "--min-backoff", $MinBackoff, "--max-backoff", $MaxBackoff,
        "--max-dispatches-per-second", $MaxDispatches, "--max-concurrent-dispatches", $MaxConcurrent)
}

Invoke-Gcloud @("services", "enable", "run.googleapis.com", "cloudtasks.googleapis.com", "cloudscheduler.googleapis.com", "iamcredentials.googleapis.com", "secretmanager.googleapis.com", "cloudkms.googleapis.com", "datastore.googleapis.com", "--project", $ProjectId)

$researchServiceAccount = Ensure-ServiceAccount "twin-research" "Foresea twin research worker"
$maintenanceServiceAccount = Ensure-ServiceAccount "twin-maintenance" "Foresea twin maintenance worker"
$taskDispatcherServiceAccount = Ensure-ServiceAccount "twin-task-dispatcher" "Foresea Cloud Tasks OIDC dispatcher"
$schedulerServiceAccount = Ensure-ServiceAccount "twin-scheduler" "Foresea twin scheduler"

Ensure-Queue "twin-research" "5" "3600s" "10s" "300s" "2" "2"
Ensure-Queue "twin-maintenance" "10" "86400s" "5s" "300s" "5" "1"

# Both services use the exact same immutable image.  There is no public
# invoker, no live capital, and no mandate supplied by this deployment path.
Invoke-Gcloud @("run", "deploy", "twin-research", "--image", $Image, "--region", $Region, "--project", $ProjectId,
    "--no-allow-unauthenticated", "--service-account", $researchServiceAccount,
    "--port", "8000", "--timeout", "120", "--cpu", "1", "--memory", "512Mi",
    "--min-instances", "0", "--max-instances", "2", "--concurrency", "2",
    "--set-env-vars", "FORESEA_TWIN_WORKER_ROLE=research,FORESEA_TWIN_MODE=shadow,FORESEA_TWIN_LIVE_CAPITAL=0,FORESEA_TWIN_LIVE_MANDATE=")
Invoke-Gcloud @("run", "deploy", "twin-maintenance", "--image", $Image, "--region", $Region, "--project", $ProjectId,
    "--no-allow-unauthenticated", "--service-account", $maintenanceServiceAccount,
    "--port", "8000", "--timeout", "120", "--cpu", "1", "--memory", "512Mi",
    "--min-instances", "0", "--max-instances", "1", "--concurrency", "1",
    "--set-env-vars", "FORESEA_TWIN_WORKER_ROLE=maintenance,FORESEA_TWIN_MODE=shadow,FORESEA_TWIN_LIVE_CAPITAL=0,FORESEA_TWIN_LIVE_MANDATE=")

# Research receives only its model secret.  It is intentionally never granted
# Datastore, Cloud Tasks enqueue, or KMS decrypt permissions below.
Invoke-Gcloud @("secrets", "add-iam-policy-binding", $ResearchModelSecret, "--project", $ProjectId,
    "--member", "serviceAccount:$researchServiceAccount", "--role", "roles/secretmanager.secretAccessor")
Invoke-Gcloud @("run", "services", "update", "twin-research", "--region", $Region, "--project", $ProjectId,
    "--update-secrets", "SCADS_AI_API_KEY=$ResearchModelSecret`:latest")

# Maintenance alone owns durable state and the exchange-connection decrypt key.
Invoke-Gcloud @("projects", "add-iam-policy-binding", $ProjectId, "--member", "serviceAccount:$maintenanceServiceAccount", "--role", "roles/datastore.user")
Invoke-Gcloud @("kms", "keys", "add-iam-policy-binding", $TradingKmsKey, "--project", $ProjectId,
    "--member", "serviceAccount:$maintenanceServiceAccount", "--role", "roles/cloudkms.cryptoKeyDecrypter")
Invoke-Gcloud @("tasks", "queues", "add-iam-policy-binding", "twin-research", "--location", $Region, "--project", $ProjectId,
    "--member", "serviceAccount:$maintenanceServiceAccount", "--role", "roles/cloudtasks.enqueuer")
Invoke-Gcloud @("tasks", "queues", "add-iam-policy-binding", "twin-maintenance", "--location", $Region, "--project", $ProjectId,
    "--member", "serviceAccount:$maintenanceServiceAccount", "--role", "roles/cloudtasks.enqueuer")

# Queue and scheduler delivery receive a service-account OIDC token.  Cloud Run
# IAM rejects anonymous callers; the application additionally checks issuer,
# audience, and the caller identity on each narrow internal handler.
$projectNumber = "<resolved-project-number>"
if ($Apply) {
    $projectNumber = (& gcloud projects describe $ProjectId --format="value(projectNumber)").Trim()
}
else {
    Write-Host "+ resolve project number for Cloud Tasks service agent"
}
$cloudTasksServiceAgent = "service-$projectNumber@gcp-sa-cloudtasks.iam.gserviceaccount.com"
Invoke-Gcloud @("iam", "service-accounts", "add-iam-policy-binding", $taskDispatcherServiceAccount, "--project", $ProjectId,
    "--member", "serviceAccount:$cloudTasksServiceAgent", "--role", "roles/iam.serviceAccountTokenCreator")

foreach ($service in @("twin-research", "twin-maintenance")) {
    Invoke-Gcloud @("run", "services", "add-iam-policy-binding", $service, "--region", $Region, "--project", $ProjectId,
        "--member", "serviceAccount:$taskDispatcherServiceAccount", "--role", "roles/run.invoker")
}
Invoke-Gcloud @("run", "services", "add-iam-policy-binding", "twin-maintenance", "--region", $Region, "--project", $ProjectId,
    "--member", "serviceAccount:$schedulerServiceAccount", "--role", "roles/run.invoker")
Invoke-Gcloud @("run", "services", "add-iam-policy-binding", "twin-maintenance", "--region", $Region, "--project", $ProjectId,
    "--member", "serviceAccount:$researchServiceAccount", "--role", "roles/run.invoker")

$maintenanceUrl = "<resolved-maintenance-url>"
$researchUrl = "<resolved-research-url>"
if ($Apply) {
    $maintenanceUrl = (& gcloud run services describe twin-maintenance --region $Region --project $ProjectId --format="value(status.url)").Trim()
    $researchUrl = (& gcloud run services describe twin-research --region $Region --project $ProjectId --format="value(status.url)").Trim()
}
else {
    Write-Host "+ resolve private service URLs and set exact OIDC audiences"
}
Invoke-Gcloud @("run", "services", "update", "twin-maintenance", "--region", $Region, "--project", $ProjectId,
    "--update-env-vars", "FORESEA_TWIN_MAINTENANCE_AUDIENCE=$maintenanceUrl,FORESEA_TWIN_RESEARCH_AUDIENCE=$researchUrl")
Invoke-Gcloud @("run", "services", "update", "twin-research", "--region", $Region, "--project", $ProjectId,
    "--update-env-vars", "FORESEA_TWIN_RESEARCH_AUDIENCE=$researchUrl")

$schedulerFlags = @("--schedule", "*/5 * * * *", "--time-zone", "UTC",
    "--uri", "$maintenanceUrl/internal/twin/dispatch", "--http-method", "POST",
    "--oidc-service-account-email", $schedulerServiceAccount, "--oidc-token-audience", $maintenanceUrl,
    "--attempt-deadline", "120s", "--location", $Region, "--project", $ProjectId)
$schedulerVerb = "update"
if ($Apply) {
    & gcloud scheduler jobs describe twin-due-work --location $Region --project $ProjectId 2>$null
    if ($LASTEXITCODE -ne 0) {
        $schedulerVerb = "create"
    }
}
else {
    Write-Host "+ ensure scheduler job twin-due-work"
}
if ($schedulerVerb -eq "create") {
    Invoke-Gcloud (@("scheduler", "jobs", "create", "http", "twin-due-work") + $schedulerFlags)
}
else {
    Invoke-Gcloud (@("scheduler", "jobs", "update", "http", "twin-due-work") + $schedulerFlags)
}

Write-Host "Private twin runtime definition applied in shadow mode. Record IAM evidence before staging traffic."
