"""Deterministic forward-shadow evidence gates for the autonomous twin."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping, Sequence

from opentelemetry import metrics, trace

from .replay import canonical_hash

tracer = trace.get_tracer(__name__)
trial_reports = metrics.get_meter(__name__).create_counter("twin.trial.reports", unit="1")

REQUIRED_DRILLS = (
    "provider_outage",
    "duplicate_task",
    "cancel_fill_race",
    "kill_restart",
)


class TrialEvidenceError(ValueError):
    """Forward evidence is malformed, mutable, or belongs to another release."""


def _aware(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise TrialEvidenceError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TrialEvidenceError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _digest(value: Any, name: str) -> str:
    result = str(value)
    if len(result) != 64:
        raise TrialEvidenceError(f"{name} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise TrialEvidenceError(f"{name} must be a SHA-256 digest") from exc
    return result.lower()


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TrialEvidenceError(f"{name} must be a non-negative integer")
    return value


def _amount(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise TrialEvidenceError(f"{name} must be a decimal amount") from exc
    if not result.is_finite() or result < 0:
        raise TrialEvidenceError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class TrialObservation:
    id: str
    observed_at: datetime
    code_hash: str
    config_hash: str
    complete_market_snapshots: int
    decisions: int
    simulated_commands: int
    duplicate_commands: int
    unexplained_divergences: int
    stale_exposure_attempts: int
    actual_cost_usd: Decimal
    uncertain_cost_usd: Decimal

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrialObservation":
        required = {
            "id", "observed_at", "code_hash", "config_hash",
            "complete_market_snapshots", "decisions", "simulated_commands",
            "duplicate_commands", "unexplained_divergences",
            "stale_exposure_attempts", "actual_cost_usd", "uncertain_cost_usd",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise TrialEvidenceError("trial observation schema is invalid")
        identifier = str(value["id"])
        if not identifier or len(identifier) > 255:
            raise TrialEvidenceError("trial observation ID is invalid")
        return cls(
            identifier,
            _aware(value["observed_at"], "observed_at"),
            _digest(value["code_hash"], "observation code_hash"),
            _digest(value["config_hash"], "observation config_hash"),
            _count(value["complete_market_snapshots"], "complete_market_snapshots"),
            _count(value["decisions"], "decisions"),
            _count(value["simulated_commands"], "simulated_commands"),
            _count(value["duplicate_commands"], "duplicate_commands"),
            _count(value["unexplained_divergences"], "unexplained_divergences"),
            _count(value["stale_exposure_attempts"], "stale_exposure_attempts"),
            _amount(value["actual_cost_usd"], "actual_cost_usd"),
            _amount(value["uncertain_cost_usd"], "uncertain_cost_usd"),
        )


def _consecutive_days(days: Sequence[date], start: date) -> int:
    unique = set(days)
    count = 0
    cursor = start
    while cursor in unique:
        count += 1
        cursor += timedelta(days=1)
    return count


def _strategy_gate(value: Any) -> dict[str, Any]:
    required = {
        "independent_resolved_markets", "completed_shadow_trades",
        "baseline_skill_lower_bound", "conservative_net_result",
    }
    if value is None:
        return {"status": "collecting", "reason": "strategy_evidence_missing"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise TrialEvidenceError("strategy evidence schema is invalid")
    resolved = _count(value["independent_resolved_markets"], "independent_resolved_markets")
    trades = _count(value["completed_shadow_trades"], "completed_shadow_trades")
    try:
        skill = Decimal(str(value["baseline_skill_lower_bound"]))
        net = Decimal(str(value["conservative_net_result"]))
    except Exception as exc:
        raise TrialEvidenceError("strategy evidence results must be decimals") from exc
    if not skill.is_finite() or not net.is_finite():
        raise TrialEvidenceError("strategy evidence results must be finite")
    if resolved < 100 or trades < 30:
        status, reason = "collecting", "strategy_sample_insufficient"
    elif skill <= 0 or net <= 0:
        status, reason = "ineligible", "strategy_evidence_not_positive"
    else:
        status, reason = "pass", "strategy_evidence_passed"
    return {
        "status": status, "reason": reason,
        "independent_resolved_markets": resolved,
        "completed_shadow_trades": trades,
        "baseline_skill_lower_bound": str(skill),
        "conservative_net_result": str(net),
        "required": {
            "independent_resolved_markets": 100,
            "completed_shadow_trades": 30,
            "baseline_skill_lower_bound": "> 0",
            "conservative_net_result": "> 0",
        },
    }


@tracer.start_as_current_span("twin.trial.evaluate")
def build_trial_report(evidence: Mapping[str, Any], *, as_of: datetime) -> dict[str, Any]:
    """Evaluate exact-release forward mechanics and strategy evidence."""
    required = {
        "schema_version", "release", "collection_operational", "observations",
        "drills", "strategy_evidence", "blockers",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required:
        raise TrialEvidenceError("trial evidence schema is invalid")
    if evidence["schema_version"] != 1:
        raise TrialEvidenceError("trial evidence schema version is unsupported")
    as_of = _aware(as_of, "as_of")
    release = evidence["release"]
    if not isinstance(release, Mapping) or set(release) != {
        "code_hash", "config_hash", "image_digest", "started_at",
    }:
        raise TrialEvidenceError("trial release schema is invalid")
    code_hash = _digest(release["code_hash"], "release code_hash")
    config_hash = _digest(release["config_hash"], "release config_hash")
    image_digest = _digest(release["image_digest"], "release image_digest")
    started_at = _aware(release["started_at"], "release started_at")
    if started_at > as_of:
        raise TrialEvidenceError("trial cannot start after its report time")
    if type(evidence["collection_operational"]) is not bool:
        raise TrialEvidenceError("collection_operational must be boolean")
    blockers = evidence["blockers"]
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item.strip() or len(item) > 500 for item in blockers
    ):
        raise TrialEvidenceError("trial blockers must be bounded strings")
    if not isinstance(evidence["observations"], list):
        raise TrialEvidenceError("trial observations must be a list")
    observations = tuple(
        TrialObservation.from_mapping(item) for item in evidence["observations"]
    )
    if len({item.id for item in observations}) != len(observations):
        raise TrialEvidenceError("trial observation IDs must be unique")
    if any(item.code_hash != code_hash or item.config_hash != config_hash for item in observations):
        raise TrialEvidenceError("trial observation belongs to another release or config")
    if any(item.observed_at < started_at or item.observed_at > as_of for item in observations):
        raise TrialEvidenceError("trial observation is outside the report window")

    drills = evidence["drills"]
    if not isinstance(drills, Mapping) or set(drills) != set(REQUIRED_DRILLS):
        raise TrialEvidenceError("trial fault-drill evidence is incomplete")
    drill_status = {}
    for name in REQUIRED_DRILLS:
        item = drills[name]
        if not isinstance(item, Mapping) or set(item) != {"status", "evidence_id"}:
            raise TrialEvidenceError("trial fault-drill schema is invalid")
        status = str(item["status"])
        evidence_id = str(item["evidence_id"])
        if status not in {"pending", "pass", "fail"} or (status != "pending" and not evidence_id):
            raise TrialEvidenceError("trial fault-drill result is invalid")
        drill_status[name] = {"status": status, "evidence_id": evidence_id}

    consecutive = _consecutive_days(
        [item.observed_at.date() for item in observations], started_at.date(),
    )
    totals = {
        "observations": len(observations),
        "complete_market_snapshots": sum(item.complete_market_snapshots for item in observations),
        "decisions": sum(item.decisions for item in observations),
        "simulated_commands": sum(item.simulated_commands for item in observations),
        "duplicate_commands": sum(item.duplicate_commands for item in observations),
        "unexplained_divergences": sum(item.unexplained_divergences for item in observations),
        "stale_exposure_attempts": sum(item.stale_exposure_attempts for item in observations),
        "actual_cost_usd": str(sum((item.actual_cost_usd for item in observations), Decimal("0"))),
        "uncertain_cost_usd": str(sum((item.uncertain_cost_usd for item in observations), Decimal("0"))),
    }
    mechanics_clean = (
        totals["complete_market_snapshots"] > 0
        and totals["decisions"] > 0
        and totals["duplicate_commands"] == 0
        and totals["unexplained_divergences"] == 0
        and totals["stale_exposure_attempts"] == 0
        and all(item["status"] == "pass" for item in drill_status.values())
    )
    if not evidence["collection_operational"] or blockers:
        g1_status, g1_reason = "blocked", "forward_collection_not_operational"
    elif consecutive < 7:
        g1_status, g1_reason = "collecting", "seven_consecutive_days_incomplete"
    elif not mechanics_clean:
        g1_status, g1_reason = "ineligible", "mechanics_gate_failed"
    else:
        g1_status, g1_reason = "pass", "mechanics_gate_passed"
    next_measurement = None
    if g1_status == "collecting":
        next_day = started_at.date() + timedelta(days=consecutive)
        next_measurement = datetime.combine(next_day, datetime.min.time(), timezone.utc).isoformat()
    g1 = {
        "status": g1_status, "reason": g1_reason,
        "consecutive_days": consecutive, "required_consecutive_days": 7,
        "drills": drill_status, "totals": totals,
        "next_measurement_at": next_measurement,
    }
    g2 = _strategy_gate(evidence["strategy_evidence"])
    payload = {
        "schema_version": 1,
        "generated_at": as_of.isoformat(),
        "release": {
            "code_hash": code_hash, "config_hash": config_hash,
            "image_digest": image_digest, "started_at": started_at.isoformat(),
        },
        "evidence_hash": canonical_hash(evidence),
        "g1": g1, "g2": g2,
        "blockers": list(blockers),
        "live_eligible": False,
    }
    payload["artifact_hash"] = canonical_hash(payload)
    span = trace.get_current_span()
    span.set_attributes({"twin.trial.g1": g1_status, "twin.trial.g2": g2["status"]})
    trial_reports.add(1, {"g1": g1_status, "g2": g2["status"]})
    return payload
