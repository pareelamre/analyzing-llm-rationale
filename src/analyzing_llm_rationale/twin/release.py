"""Fail-closed shadow release evidence for the autonomous twin."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from .replay import canonical_hash

RELEASE_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
REQUIRED_G0_CHECKS = (
    "unit_contract_tests",
    "datastore_emulator",
    "frontend_build",
    "lint",
    "network_deny",
    "shadow_runtime",
    "health_smoke",
    "readiness_smoke",
    "rollback_drill",
    "scheduled_shadow_only",
)


class ReleaseReadinessError(ValueError):
    """Release evidence is incomplete, stale, forged, or incompatible."""


def _digest(name: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64:
        raise ReleaseReadinessError(f"{name} must be a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ReleaseReadinessError(f"{name} must be a SHA-256 digest") from exc
    return text.lower()


def build_shadow_release_artifact(
    *, code_hash: str, config_hash: str, generated_at: datetime,
    expires_at: datetime, checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Create immutable G0 evidence; this format can never authorize live trading."""
    if generated_at.tzinfo is None or expires_at.tzinfo is None or expires_at <= generated_at:
        raise ReleaseReadinessError("release timestamps must be aware and increasing")
    if set(checks) != set(REQUIRED_G0_CHECKS) or any(type(value) is not bool for value in checks.values()):
        raise ReleaseReadinessError("release checks are incomplete or invalid")
    passed = all(checks.values())
    payload: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "minimum_reader_schema_version": LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
        "expires_at": expires_at.astimezone(timezone.utc).isoformat(),
        "code_hash": _digest("code_hash", code_hash),
        "config_hash": _digest("config_hash", config_hash),
        "mode": "shadow",
        "live_eligible": False,
        "rollback_protocol": "pause-reconcile-preserve-v1",
        "checks": {name: checks[name] for name in REQUIRED_G0_CHECKS},
        "status": "g0_pass" if passed else "blocked",
    }
    payload["artifact_hash"] = canonical_hash(payload)
    return payload


def validate_shadow_release_artifact(
    artifact: Mapping[str, Any], *, now: datetime,
    expected_code_hash: str, expected_config_hash: str,
) -> None:
    """Reject stale, forged, mismatched, live-shaped, or incomplete release evidence."""
    required = {
        "schema_version", "ledger_schema_version", "minimum_reader_schema_version",
        "generated_at", "expires_at", "code_hash", "config_hash", "mode",
        "live_eligible", "rollback_protocol", "checks", "status", "artifact_hash",
    }
    issues: list[str] = []
    if set(artifact) != required:
        issues.append("release schema fields are invalid")
    if artifact.get("schema_version") != RELEASE_SCHEMA_VERSION:
        issues.append("release schema version is unsupported")
    if artifact.get("ledger_schema_version") != LEDGER_SCHEMA_VERSION:
        issues.append("ledger schema version is unsupported")
    if artifact.get("minimum_reader_schema_version") != LEDGER_SCHEMA_VERSION:
        issues.append("rollback reader compatibility is unsupported")
    if artifact.get("mode") != "shadow" or artifact.get("live_eligible") is not False:
        issues.append("release evidence must remain shadow-only")
    if artifact.get("rollback_protocol") != "pause-reconcile-preserve-v1":
        issues.append("rollback protocol is unsupported")
    try:
        expected_code = _digest("expected_code_hash", expected_code_hash)
        expected_config = _digest("expected_config_hash", expected_config_hash)
        if artifact.get("code_hash") != expected_code:
            issues.append("release code hash does not match")
        if artifact.get("config_hash") != expected_config:
            issues.append("release config hash does not match")
    except ReleaseReadinessError as exc:
        issues.append(str(exc))
    checks = artifact.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(REQUIRED_G0_CHECKS):
        issues.append("release checks are incomplete")
    elif any(checks.get(name) is not True for name in REQUIRED_G0_CHECKS):
        issues.append("one or more G0 checks did not pass")
    if artifact.get("status") != "g0_pass":
        issues.append("release status is not g0_pass")
    if now.tzinfo is None:
        issues.append("validation time must be timezone-aware")
    else:
        try:
            generated_at = datetime.fromisoformat(str(artifact.get("generated_at")).replace("Z", "+00:00"))
            expires_at = datetime.fromisoformat(str(artifact.get("expires_at")).replace("Z", "+00:00"))
            current = now.astimezone(timezone.utc)
            if generated_at.tzinfo is None or expires_at.tzinfo is None:
                raise ValueError
            if current < generated_at.astimezone(timezone.utc) or current >= expires_at.astimezone(timezone.utc):
                issues.append("release evidence is stale or future-dated")
        except (TypeError, ValueError):
            issues.append("release timestamps are invalid")
    payload = dict(artifact)
    saved_hash = payload.pop("artifact_hash", None)
    if saved_hash != canonical_hash(payload):
        issues.append("release artifact hash does not match its contents")
    if issues:
        raise ReleaseReadinessError("; ".join(dict.fromkeys(issues)))
