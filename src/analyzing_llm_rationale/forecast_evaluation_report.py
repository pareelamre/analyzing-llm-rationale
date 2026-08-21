"""Audit-cohort reports and explicit promotion gates for forecast strategies."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from analyzing_llm_rationale.forecast_evaluation import (
    ResolvedForecast,
    build_trades,
    evaluation_report,
    simulate_compounded_portfolio,
)

SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("foresea.forecast_evaluation")
meter = metrics.get_meter("foresea.forecast_evaluation")
evaluation_reports = meter.create_counter(
    "forecast_evaluation.reports",
    unit="1",
    description="Deterministic forecast evaluation reports generated",
)
evaluation_report_duration = meter.create_histogram(
    "forecast_evaluation.report.duration",
    unit="s",
    description="Forecast evaluation report generation latency",
)


class EvaluationArtifactValidationError(ValueError):
    """Raised when a serialized evaluation artifact breaks harness invariants."""

    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True)
class EvaluationPolicy:
    min_resolved_markets: int = 100
    min_paper_trades: int = 30
    min_skill_lower_bound: float = 0.0
    max_drawdown: float = 0.20
    min_edge: float = 0.05
    requested_fraction: float = 0.02
    fee_fraction: float = 0.01
    max_total_exposure: float = 0.25

    def __post_init__(self) -> None:
        if self.min_resolved_markets < 1:
            raise ValueError("min_resolved_markets must be positive")
        if self.min_paper_trades < 1:
            raise ValueError("min_paper_trades must be positive")
        if not isinstance(self.min_skill_lower_bound, (int, float)) or not math.isfinite(
            self.min_skill_lower_bound
        ):
            raise ValueError("min_skill_lower_bound must be finite")
        if not 0.0 <= self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown must be between 0 and 1")
        if self.min_edge < 0.0:
            raise ValueError("min_edge cannot be negative")
        if not 0.0 < self.requested_fraction <= 1.0:
            raise ValueError("requested_fraction must be in (0, 1]")
        if self.fee_fraction < 0.0:
            raise ValueError("fee_fraction cannot be negative")
        if not 0.0 < self.max_total_exposure <= 1.0:
            raise ValueError("max_total_exposure must be in (0, 1]")


def _cohort_report(
    forecasts: Sequence[ResolvedForecast],
    *,
    provenance: str,
    policy: EvaluationPolicy,
) -> dict[str, Any]:
    scores = evaluation_report(forecasts)
    trades = build_trades(
        forecasts,
        min_edge=policy.min_edge,
        requested_fraction=policy.requested_fraction,
        fee_fraction=policy.fee_fraction,
    )
    portfolio = simulate_compounded_portfolio(
        trades,
        max_total_exposure=policy.max_total_exposure,
    )
    interval = scores["market_clustered_skill_interval"]
    return {
        "provenance": provenance,
        "promotion_eligible_source": provenance == "prospective_audit",
        "resolved_forecasts": scores["n"],
        "resolved_markets": interval["n_markets"],
        "model_brier": scores["model_brier"],
        "market_brier": scores["market_brier"],
        "skill_vs_market": scores["skill_vs_market"],
        "market_clustered_skill_interval": interval,
        "log_loss": scores["log_loss"],
        "calibration": scores["calibration"],
        "domain_probability_buckets": scores["domain_probability_buckets"],
        "portfolio": portfolio,
    }


def _gate(
    *,
    actual: Any,
    required: Any,
    passed: bool,
    comparison: str,
) -> dict[str, Any]:
    return {
        "passed": passed,
        "actual": actual,
        "required": required,
        "comparison": comparison,
    }


def _promotion_result(
    prospective: dict[str, Any],
    *,
    policy: EvaluationPolicy,
) -> dict[str, Any]:
    interval = prospective["market_clustered_skill_interval"]
    portfolio = prospective["portfolio"]
    lower = interval["lower"]
    checks = {
        "minimum_resolved_markets": _gate(
            actual=prospective["resolved_markets"],
            required=policy.min_resolved_markets,
            passed=prospective["resolved_markets"] >= policy.min_resolved_markets,
            comparison="greater_than_or_equal",
        ),
        "positive_skill_lower_bound": _gate(
            actual=lower,
            required=policy.min_skill_lower_bound,
            passed=lower is not None and lower > policy.min_skill_lower_bound,
            comparison="greater_than",
        ),
        "minimum_paper_trades": _gate(
            actual=portfolio["n_opened"],
            required=policy.min_paper_trades,
            passed=portfolio["n_opened"] >= policy.min_paper_trades,
            comparison="greater_than_or_equal",
        ),
        "positive_compound_return_after_fees": _gate(
            actual=portfolio["compound_return"],
            required=0.0,
            passed=portfolio["compound_return"] > 0.0,
            comparison="greater_than",
        ),
        "maximum_drawdown": _gate(
            actual=portfolio["max_drawdown"],
            required=policy.max_drawdown,
            passed=portfolio["max_drawdown"] <= policy.max_drawdown,
            comparison="less_than_or_equal",
        ),
    }
    eligible = all(check["passed"] for check in checks.values())
    enough_observations = (
        checks["minimum_resolved_markets"]["passed"]
        and checks["minimum_paper_trades"]["passed"]
    )
    if eligible:
        status = "eligible_for_shadow_promotion"
    elif enough_observations:
        status = "not_qualified"
    else:
        status = "collecting"
    return {
        "eligible": eligible,
        "status": status,
        "checks": checks,
        "warning": (
            "Only prospective_audit observations can satisfy promotion gates. "
            "Eligibility permits a controlled shadow experiment, not live capital."
        ),
    }


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_evaluation_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject an artifact whose published claims disagree with its source cohort.

    The scheduled harness calls this before publishing. It prevents a partial
    write, stale manual edit, or future report change from disconnecting the
    promotion decision from prospective observations.
    """
    issues: list[str] = []
    if artifact.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            f"schema_version must be {SCHEMA_VERSION}, got {artifact.get('schema_version')!r}"
        )
    if not isinstance(artifact.get("model"), str) or not artifact["model"].strip():
        issues.append("model must be a non-empty string")

    policy_data = artifact.get("policy")
    policy: EvaluationPolicy | None = None
    if not isinstance(policy_data, Mapping):
        issues.append("policy must be an object")
    else:
        try:
            policy = EvaluationPolicy(**dict(policy_data))
        except (TypeError, ValueError) as exc:
            issues.append(f"policy is invalid: {exc}")

    cohorts = artifact.get("cohorts")
    validated_cohorts: dict[str, Mapping[str, Any]] = {}
    if not isinstance(cohorts, Mapping):
        issues.append("cohorts must be an object")
    else:
        for name, promotion_eligible in (
            ("snapshot_mirror", False),
            ("prospective_audit", True),
        ):
            cohort = cohorts.get(name)
            if not isinstance(cohort, Mapping):
                issues.append(f"cohorts.{name} must be an object")
                continue
            validated_cohorts[name] = cohort
            if cohort.get("provenance") != name:
                issues.append(f"cohorts.{name}.provenance must be {name!r}")
            if cohort.get("promotion_eligible_source") is not promotion_eligible:
                issues.append(
                    f"cohorts.{name}.promotion_eligible_source must be "
                    f"{promotion_eligible!r}"
                )
            for field in ("resolved_forecasts", "resolved_markets"):
                if not _non_negative_int(cohort.get(field)):
                    issues.append(f"cohorts.{name}.{field} must be a non-negative integer")
            interval = cohort.get("market_clustered_skill_interval")
            if not isinstance(interval, Mapping):
                issues.append(f"cohorts.{name}.market_clustered_skill_interval must be an object")
            elif (
                interval.get("n_forecasts") != cohort.get("resolved_forecasts")
                or interval.get("n_markets") != cohort.get("resolved_markets")
            ):
                issues.append(f"cohorts.{name} counts must match its skill interval")
            portfolio = cohort.get("portfolio")
            if not isinstance(portfolio, Mapping) or not _non_negative_int(
                portfolio.get("n_opened") if isinstance(portfolio, Mapping) else None
            ):
                issues.append(f"cohorts.{name}.portfolio.n_opened must be a non-negative integer")
            if (
                _non_negative_int(cohort.get("resolved_markets"))
                and _non_negative_int(cohort.get("resolved_forecasts"))
                and cohort["resolved_markets"] > cohort["resolved_forecasts"]
            ):
                issues.append(f"cohorts.{name}.resolved_markets cannot exceed resolved_forecasts")

    prospective = validated_cohorts.get("prospective_audit")

    promotion = artifact.get("promotion")
    if not isinstance(promotion, Mapping):
        issues.append("promotion must be an object")
    elif prospective is not None and policy is not None:
        try:
            expected_promotion = _promotion_result(dict(prospective), policy=policy)
        except (KeyError, TypeError):
            issues.append("prospective_audit cannot be used to recompute promotion")
        else:
            if dict(promotion) != expected_promotion:
                issues.append("promotion must exactly match prospective_audit and policy")

    if issues:
        raise EvaluationArtifactValidationError(issues)


def build_evaluation_artifact(
    *,
    model: str,
    snapshot_mirror: Sequence[ResolvedForecast],
    prospective_audit: Sequence[ResolvedForecast],
    generated_at: datetime | None = None,
    policy: EvaluationPolicy | None = None,
) -> dict[str, Any]:
    """Build the versioned operator report from explicit provenance cohorts."""
    started = time.perf_counter()
    policy = policy or EvaluationPolicy()
    generated_at = generated_at or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)

    with tracer.start_as_current_span("forecast_evaluation.generate") as span:
        try:
            historical = _cohort_report(
                snapshot_mirror,
                provenance="snapshot_mirror",
                policy=policy,
            )
            prospective = _cohort_report(
                prospective_audit,
                provenance="prospective_audit",
                policy=policy,
            )
            promotion = _promotion_result(prospective, policy=policy)
            artifact = {
                "schema_version": SCHEMA_VERSION,
                "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
                "model": model,
                "methodology": {
                    "primary_cohort": "prospective_audit",
                    "diagnostic_cohort": "snapshot_mirror",
                    "scoring": "Brier skill relative to contemporaneous market probability",
                    "uncertainty": "95% normal interval over market-level mean Brier differences",
                    "execution": "executable bid/ask side with fees and capped total exposure",
                },
                "policy": asdict(policy),
                "cohorts": {
                    "snapshot_mirror": historical,
                    "prospective_audit": prospective,
                },
                "promotion": promotion,
            }
            validate_evaluation_artifact(artifact)
            span.set_attributes(
                {
                    "outcome": "success",
                    "model": model,
                    "snapshot_mirror.forecasts": len(snapshot_mirror),
                    "prospective_audit.forecasts": len(prospective_audit),
                    "promotion.status": promotion["status"],
                }
            )
            evaluation_reports.add(
                1,
                {"outcome": "success", "promotion_status": promotion["status"]},
            )
            logger.info(
                "forecast evaluation report generated: model=%s status=%s prospective=%d",
                model,
                promotion["status"],
                len(prospective_audit),
            )
            return artifact
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            evaluation_reports.add(
                1,
                {"outcome": "failure", "promotion_status": "unknown"},
            )
            logger.exception("forecast evaluation report generation failed")
            raise
        finally:
            evaluation_report_duration.record(
                time.perf_counter() - started,
                {"operation": "artifact_generation"},
            )


def compact_evaluation_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded scheduled-job log payload without curves or buckets."""
    cohorts: dict[str, Any] = {}
    for name, cohort in artifact["cohorts"].items():
        portfolio = cohort["portfolio"]
        cohorts[name] = {
            "resolved_forecasts": cohort["resolved_forecasts"],
            "resolved_markets": cohort["resolved_markets"],
            "model_brier": cohort["model_brier"],
            "market_brier": cohort["market_brier"],
            "skill_vs_market": cohort["skill_vs_market"],
            "skill_lower_95": cohort["market_clustered_skill_interval"]["lower"],
            "domain_probability_buckets": len(cohort["domain_probability_buckets"]),
            "paper_trades": portfolio["n_opened"],
            "paper_compound_return": portfolio["compound_return"],
            "paper_max_drawdown": portfolio["max_drawdown"],
        }
    return {
        "model": artifact["model"],
        **cohorts,
        "promotion": artifact["promotion"],
    }
