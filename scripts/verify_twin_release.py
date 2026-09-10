"""Verify the repository-side contracts required before a shadow twin release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _hash_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_repository_contract(root: Path = ROOT) -> dict[str, object]:
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    deploy = (root / "infra" / "twin" / "deploy.ps1").read_text(encoding="utf-8")
    reusable = (root / ".github" / "workflows" / "_agent-trading-tick-reusable.yml").read_text(encoding="utf-8")
    release_tests = (root / "tests" / "test_twin_release.py").read_text(encoding="utf-8")
    runtime_tests = (root / "tests" / "test_twin_runtime.py").read_text(encoding="utf-8")
    smoke = (root / "infra" / "twin" / "smoke.ps1").read_text(encoding="utf-8")
    operations = (root / "docs" / "autonomous-twin" / "OPERATIONS.md").read_text(encoding="utf-8")
    requirements = {
        "unit_contract_tests": "python -m unittest discover -s tests" in ci,
        "datastore_emulator": all(value in ci for value in (
            "datastore-integration:", "cloud-datastore-emulator", "tests.test_twin_store_integration",
        )),
        "frontend_build": "npm run frontend:build" in ci,
        "lint": "ruff check src tests" in ci,
        "network_deny": all(value in release_tests for value in (
            "credentials_cannot_bypass_network_deny_gate", "runtime_live_enabled=False", "network-write",
        )),
        "shadow_runtime": all(value in deploy for value in (
            "FORESEA_TWIN_MODE=shadow", "FORESEA_TWIN_LIVE_CAPITAL=0", "FORESEA_TWIN_LIVE_MANDATE=",
            "--no-allow-unauthenticated",
        )),
        "health_smoke": all(value in smoke + runtime_tests for value in (
            'client.get("/health")', '"$url/health"', "Invoke-RestMethod -Method Get",
        )),
        "readiness_smoke": all(value in smoke + runtime_tests for value in (
            'client.get("/ready")', '"$url/ready"', "status_code, 503",
        )),
        "rollback_drill": all(value in release_tests + operations for value in (
            "rollback_rebuild_preserves_manual_command_and_reservation", "Deployment rollback",
            "Do not delete jobs, commands, reservations",
        )),
        "scheduled_shadow_only": "FORESEA_AGENT_PLACE_TRADE_MODE: shadow" in reusable,
    }
    requirements["health_smoke"] = requirements["health_smoke"] and all(
        value in smoke + deploy for value in (
            "--impersonate-service-account=", "-InvokerServiceAccount",
            'FORESEA_TWIN_MODE"] -ne "shadow"',
        )
    ) and "-Method Post" not in smoke
    requirements["readiness_smoke"] = requirements["readiness_smoke"] and "-Method Post" not in smoke
    failed = sorted(name for name, passed in requirements.items() if not passed)
    source_paths = list((root / "src" / "analyzing_llm_rationale" / "twin").glob("*.py"))
    source_paths.extend((root / "infra" / "twin").glob("*.ps1"))
    result: dict[str, object] = {
        "status": "pass" if not failed else "fail",
        "failed": failed,
        "checks": requirements,
        "code_hash": _hash_paths(source_paths),
        "config_hash": _hash_paths([root / "configs" / "twin.yaml"]),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_repository_contract()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
