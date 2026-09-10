[CmdletBinding()]
param(
    [string]$ProjectId = "brave-drive-471109-d9",
    [string]$Region = "us-central1",
    [string]$InvokerServiceAccount = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ServiceEvidence {
    param([Parameter(Mandatory)][string]$Service)
    $raw = & gcloud run services describe $Service --project $ProjectId --region $Region --format=json
    if ($LASTEXITCODE -ne 0) { throw "Could not describe $Service" }
    $description = $raw | ConvertFrom-Json
    $url = [string]$description.status.url
    if (-not $url.StartsWith("https://")) { throw "$Service has no HTTPS URL" }
    $tokenArgs = @("auth", "print-identity-token")
    if (-not [string]::IsNullOrWhiteSpace($InvokerServiceAccount)) {
        $tokenArgs += @("--audiences=$url", "--impersonate-service-account=$InvokerServiceAccount")
    }
    $rawToken = & gcloud @tokenArgs
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace([string]$rawToken)) {
        throw "Could not obtain an identity token for $Service"
    }
    $token = ([string]$rawToken).Trim()
    $headers = @{ Authorization = "Bearer $token" }
    $health = Invoke-RestMethod -Method Get -Uri "$url/health" -Headers $headers
    if ($health.status -ne "ok" -or $health.role -notin @("maintenance", "research")) {
        throw "$Service health response is invalid"
    }
    try {
        $ready = Invoke-RestMethod -Method Get -Uri "$url/ready" -Headers $headers
    }
    catch {
        throw "$Service is healthy but not ready"
    }
    if ($ready.status -ne "ready" -or $ready.role -ne $health.role) {
        throw "$Service readiness response is invalid"
    }
    $revision = [string]$description.status.latestReadyRevisionName
    $traffic = @($description.status.traffic | Where-Object { $_.percent -gt 0 })
    if ([string]::IsNullOrWhiteSpace($revision) -or $traffic.Count -ne 1 -or $traffic[0].percent -ne 100) {
        throw "$Service does not have one fully promoted ready revision"
    }
    $containers = @($description.spec.template.spec.containers)
    $environment = @{}
    foreach ($entry in $containers[0].env) {
        $valueProperty = $entry.PSObject.Properties["value"]
        if ($null -ne $valueProperty) {
            $environment[[string]$entry.name] = [string]$valueProperty.Value
        }
    }
    if (
        $environment["FORESEA_TWIN_MODE"] -ne "shadow" -or
        $environment["FORESEA_TWIN_LIVE_CAPITAL"] -ne "0" -or
        -not [string]::IsNullOrEmpty($environment["FORESEA_TWIN_LIVE_MANDATE"])
    ) {
        throw "$Service is not pinned to zero-authority shadow mode"
    }
    return [ordered]@{
        service = $Service
        revision = $revision
        role = [string]$ready.role
        health = [string]$health.status
        readiness = [string]$ready.status
        mode = [string]$environment["FORESEA_TWIN_MODE"]
        live_capital = [string]$environment["FORESEA_TWIN_LIVE_CAPITAL"]
        live_mandate_present = -not [string]::IsNullOrEmpty($environment["FORESEA_TWIN_LIVE_MANDATE"])
    }
}

$evidence = @(
    Get-ServiceEvidence -Service "twin-maintenance"
    Get-ServiceEvidence -Service "twin-research"
)
$evidence | ConvertTo-Json -Depth 4

