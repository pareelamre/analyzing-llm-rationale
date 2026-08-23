from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'brave-drive-471109-d9')
REGION = os.environ.get('GCP_REGION', 'us-central1')
ARTIFACT_REPO = os.environ.get('ARTIFACT_REPO', 'docker')
DATASTORE_BACKUP_BUCKET = os.environ.get('DATASTORE_BACKUP_BUCKET', f'gs://{PROJECT_ID}-datastore-backups')
TRACK_STORE_BUCKET = os.environ.get('TRACK_STORE_BUCKET', f'gs://{PROJECT_ID}-track-record-store')

ROOT_DIR = Path(__file__).resolve().parent.parent
INFRA_DIR = ROOT_DIR / 'infra'


def get_optimization_actions() -> List[Dict[str, Any]]:
    return [
        {
            "service": "Artifact Registry",
            "name": "artifact_registry_cleanup",
            "description": "Enforce automated retention/cleanup on Docker repository (keep 5 recent, delete old tagged and untagged).",
            "policy_file": INFRA_DIR / "artifact-registry-cleanup-policy.json",
            "command": [
                "gcloud", "artifacts", "repositories", "set-cleanup-policies", ARTIFACT_REPO,
                "--location", REGION,
                "--project", PROJECT_ID,
                "--policy", str(INFRA_DIR / "artifact-registry-cleanup-policy.json"),
                "--no-dry-run",
            ],
            "estimated_savings": "$2 - $10+ / month in image storage accumulation",
        },
        {
            "service": "Cloud Storage",
            "name": "datastore_backup_lifecycle",
            "description": "Enforce 30-day deletion lifecycle rule on Datastore backups bucket.",
            "policy_file": INFRA_DIR / "datastore-backup-lifecycle.json",
            "command": [
                "gcloud", "storage", "buckets", "update", DATASTORE_BACKUP_BUCKET,
                f"--lifecycle-file={str(INFRA_DIR / 'datastore-backup-lifecycle.json')}",
                "--project", PROJECT_ID,
            ],
            "estimated_savings": "Prevents unbounded backup storage growth",
        },
        {
            "service": "Cloud Storage",
            "name": "track_record_lifecycle",
            "description": "Enforce 1-day deletion lifecycle rule on noncurrent track record versions.",
            "policy_file": INFRA_DIR / "track-record-store-lifecycle.json",
            "command": [
                "gcloud", "storage", "buckets", "update", TRACK_STORE_BUCKET,
                f"--lifecycle-file={str(INFRA_DIR / 'track-record-store-lifecycle.json')}",
                "--project", PROJECT_ID,
            ],
            "estimated_savings": "Prevents versioned overwrite storage inflation",
        },
        {
            "service": "Cloud Logging",
            "name": "exclude_health_checks",
            "description": "Exclude 200 OK /health and /ready requests from Cloud Logging storage.",
            "policy_file": None,
            "command": [
                "gcloud", "logging", "sinks", "update", "_Default",
                "--log-filter=NOT (resource.type=\"cloud_run_revision\" AND httpRequest.status=200 AND (httpRequest.requestUrl=~\"/health\" OR httpRequest.requestUrl=~\"/ready\"))",
                "--project", PROJECT_ID,
            ],
            "estimated_savings": "Keeps log ingestion safely within 50 GiB/month Free Tier",
        },
    ]


def find_gcloud() -> str:
    """Find the path to the gcloud executable on PATH or standard install directories."""
    import shutil
    resolved = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if resolved:
        return resolved

    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
        os.path.expandvars(r"%ProgramFiles%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
        os.path.expanduser("~/google-cloud-sdk/bin/gcloud"),
        "/usr/bin/gcloud",
        "/usr/local/bin/gcloud",
        "/snap/bin/gcloud",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return "gcloud"


def run_command(cmd: List[str], dry_run: bool = False) -> Dict[str, Any]:
    """Execute a shell command or simulate it in dry-run mode."""
    exec_cmd = list(cmd)
    if exec_cmd and exec_cmd[0] == "gcloud":
        exec_cmd[0] = find_gcloud()

    cmd_str = " ".join(cmd)
    if dry_run:
        return {"cmd": cmd_str, "status": "skipped (dry-run)", "returncode": 0, "output": ""}
    try:
        res = subprocess.run(exec_cmd, capture_output=True, text=True, check=True)
        return {"cmd": cmd_str, "status": "success", "returncode": res.returncode, "output": res.stdout.strip()}
    except subprocess.CalledProcessError as err:
        return {"cmd": cmd_str, "status": "error", "returncode": err.returncode, "output": err.stderr.strip()}
    except FileNotFoundError:
        return {"cmd": cmd_str, "status": "error", "returncode": -1, "output": "gcloud command not found in PATH"}


def audit_status() -> Dict[str, Any]:
    actions = get_optimization_actions()
    status_report = {
        "project_id": PROJECT_ID,
        "region": REGION,
        "policies": [],
        "recommendations": [
            "Keep Cloud Run cpu-throttling enabled so idle instance time is billed at discount rate.",
            "Ensure Cloud Run concurrency is >= 20 to handle burst requests on warm min-instances=1 without scaling out.",
            "Keep all buckets (models, track-records, backups) in us-central1 to ensure $0 intra-region egress.",
            "Destroy unneeded/deprecated KMS key versions ($0.06/key version/month).",
        ]
    }
    for act in actions:
        exists = act["policy_file"].exists() if act.get("policy_file") else True
        status_report["policies"].append({
            "service": act["service"],
            "name": act["name"],
            "description": act["description"],
            "policy_file_exists": exists,
            "estimated_savings": act["estimated_savings"],
            "command": " ".join(act["command"]),
        })
    return status_report


def print_report(report: Dict[str, Any]) -> None:
    print(f"\n=== GCP Cost Optimization Audit: {report['project_id']} ({report['region']}) ===\n")
    print("Target Policies:")
    for pol in report["policies"]:
        status_icon = "[OK]" if pol["policy_file_exists"] else "[MISSING FILE]"
        print(f"  * {pol['service']} ({pol['name']}): {status_icon}")
        print(f"    Description: {pol['description']}")
        print(f"    Savings:     {pol['estimated_savings']}")
        print(f"    Command:     {pol['command']}\n")

    print("Architecture Best Practices:")
    for rec in report["recommendations"]:
        print(f"  - {rec}")
    print("")


def main() -> int:
    parser = argparse.ArgumentParser(description="GCP Cost Optimization Tool")
    parser.add_argument("--audit", action="store_true", help="Audit cost posture and display recommendations.")
    parser.add_argument("--apply", action="store_true", help="Apply all cost optimization policies via gcloud.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands that would be executed without executing.")
    parser.add_argument("--json", action="store_true", help="Output status as JSON.")

    args = parser.parse_args()

    if args.json:
        print(json.dumps(audit_status(), indent=2))
        return 0

    if args.apply or args.dry_run:
        actions = get_optimization_actions()
        print(f"Applying GCP Cost Optimizations (dry_run={args.dry_run})...")
        failures = 0
        for act in actions:
            print(f"\nApplying: {act['name']} ({act['service']})...")
            res = run_command(act["command"], dry_run=args.dry_run)
            print(f"Status: {res['status']}")
            if res["output"]:
                print(f"Output: {res['output']}")
            if res["status"] == "error":
                failures += 1
        return 1 if failures > 0 else 0

    print_report(audit_status())
    return 0


if __name__ == "__main__":
    sys.exit(main())
