"""Out-of-sample strategy evaluation and reproducible twin readiness artifacts."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from opentelemetry import metrics, trace

from ..forecast_evaluation import (
    ResolvedForecast,
    build_trades,
    evaluation_report,
    simulate_compounded_portfolio,
)
from .replay import ReplayValidationError, canonical_hash, split_replay_dataset

tracer = trace.get_tracer(__name__)
replay_evaluations = metrics.get_meter(__name__).create_counter("twin.replay.evaluations", unit="1")


class ReadinessArtifactError(ValueError):
    """A readiness artifact is malformed, stale, or has failed integrity checks."""


@dataclass(frozen=True)
class ReplayPolicy:
    split_at: datetime
    evaluation_as_of: datetime
    min_calibration_records: int = 30
    min_test_records: int = 30
    min_edge: float = 0.05
    requested_fraction: float = 0.02
    fee_fraction: float = 0.01
    slippage: float = 0.01
    max_total_exposure: float = 0.25
    max_drawdown: float = 0.20
    fixed_probability: float = 0.50
    max_artifact_age_seconds: int = 7 * 24 * 60 * 60

    def __post_init__(self) -> None:
        for name in ("split_at", "evaluation_as_of"):
            value = getattr(self, name)
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError(f"{name} must be ISO-8601") from exc
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(timezone.utc))
        if self.evaluation_as_of <= self.split_at:
            raise ValueError("evaluation_as_of must follow split_at")
        for name in ("min_calibration_records", "min_test_records", "max_artifact_age_seconds"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_edge", "fee_fraction", "slippage"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or value >= 1:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, value)
        for name in ("requested_fraction", "max_total_exposure", "max_drawdown"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
        probability = float(self.fixed_probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("fixed_probability must be in [0, 1]")
        object.__setattr__(self, "fixed_probability", probability)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReplayPolicy":
        allowed = {field.name for field in fields(cls)}
        if set(value) - allowed:
            raise ValueError("replay policy contains unknown fields")
        return cls(**dict(value))

    def to_storage(self) -> dict[str, Any]:
        return {
            "split_at": self.split_at.isoformat(), "evaluation_as_of": self.evaluation_as_of.isoformat(),
            "min_calibration_records": self.min_calibration_records, "min_test_records": self.min_test_records,
            "min_edge": self.min_edge, "requested_fraction": self.requested_fraction,
            "fee_fraction": self.fee_fraction, "slippage": self.slippage,
            "max_total_exposure": self.max_total_exposure, "max_drawdown": self.max_drawdown,
            "fixed_probability": self.fixed_probability,
            "max_artifact_age_seconds": self.max_artifact_age_seconds,
        }


def readiness_artifact(outcomes: Sequence[Mapping], *, code_hash: str, config_hash: str) -> Mapping[str, object]:
    """Compatibility helper for the original minimal artifact contract."""
    payload = {"code_hash": code_hash, "config_hash": config_hash, "outcomes": list(outcomes)}
    return {**payload, "artifact_hash": canonical_hash(payload), "complete": bool(outcomes)}


def _adjusted(rows: Sequence[ResolvedForecast], slippage: float) -> list[ResolvedForecast]:
    return [replace(
        row,
        market_bid=max(0.0, (row.market_bid if row.market_bid is not None else row.market_probability) - slippage),
        market_ask=min(1.0, (row.market_ask if row.market_ask is not None else row.market_probability) + slippage),
    ) for row in rows]


def _strategy_report(
    forecasts: Sequence[ResolvedForecast], *, policy: ReplayPolicy,
    fee_fraction: float | None = None, slippage: float | None = None,
) -> dict[str, Any]:
    fee = policy.fee_fraction if fee_fraction is None else fee_fraction
    slip = policy.slippage if slippage is None else slippage
    adjusted = _adjusted(forecasts, slip)
    trades = build_trades(
        adjusted, min_edge=policy.min_edge, requested_fraction=policy.requested_fraction,
        fee_fraction=fee,
    )
    portfolio = simulate_compounded_portfolio(
        trades, initial_bankroll=100.0, max_total_exposure=policy.max_total_exposure,
    )
    return {
        "forecast_metrics": evaluation_report(forecasts),
        "portfolio": portfolio,
        "net_pnl_after_costs": portfolio["final_bankroll"] - portfolio["initial_bankroll"],
        "turnover": len(trades) / len(forecasts) if forecasts else 0.0,
        "abstention": 1.0 - len(trades) / len(forecasts) if forecasts else 1.0,
        "modeled_fee_fraction": fee,
        "modeled_slippage": slip,
    }


def _fixed_baseline(rows: Sequence[ResolvedForecast], probability: float) -> list[ResolvedForecast]:
    return [replace(row, model="fixed-policy", model_probability=probability) for row in rows]


def _correlated_loss(report_rows: Sequence[ResolvedForecast], policy: ReplayPolicy) -> dict[str, Any]:
    trades = build_trades(
        _adjusted(report_rows, policy.slippage), min_edge=policy.min_edge,
        requested_fraction=policy.requested_fraction, fee_fraction=policy.fee_fraction,
    )
    losing = [replace(trade, outcome=0 if trade.side == "YES" else 1) for trade in trades]
    return simulate_compounded_portfolio(losing, max_total_exposure=policy.max_total_exposure)


@tracer.start_as_current_span("twin.replay.evaluate")
def evaluate_replay(
    dataset: Mapping[str, Any], *, policy: ReplayPolicy, code_hash: str,
    config_hash: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic report; no network, current clock, or hidden state is read."""
    if len(code_hash) != 64:
        raise ValueError("code_hash must be a SHA-256 digest")
    frozen = split_replay_dataset(
        dataset, split_at=policy.split_at, evaluation_as_of=policy.evaluation_as_of,
    )
    config_hash = config_hash or canonical_hash(policy.to_storage())
    calibration = [item.to_forecast() for item in frozen.calibration]
    test = [item.to_forecast() for item in frozen.test]
    strategy = _strategy_report(test, policy=policy)
    fixed = _strategy_report(_fixed_baseline(test, policy.fixed_probability), policy=policy)
    quote_complete = [row for row in test if row.market_bid is not None and row.market_ask is not None]
    stresses = {
        "double_fees": _strategy_report(test, policy=policy, fee_fraction=min(0.99, policy.fee_fraction * 2)),
        "double_slippage": _strategy_report(test, policy=policy, slippage=min(0.99, policy.slippage * 2)),
        "missing_quotes": _strategy_report(quote_complete, policy=policy),
        "correlated_losses": _correlated_loss(test, policy),
    }
    gates = {
        "calibration_depth": {"status": "pass" if len(calibration) >= policy.min_calibration_records else "insufficient", "actual": len(calibration), "required": policy.min_calibration_records},
        "test_depth": {"status": "pass" if len(test) >= policy.min_test_records else "insufficient", "actual": len(test), "required": policy.min_test_records},
        "event_disjoint": {"status": "pass", "actual": True, "required": True},
        "positive_net_pnl": {"status": "pass" if strategy["net_pnl_after_costs"] > 0 else "fail", "actual": strategy["net_pnl_after_costs"], "required": "> 0"},
        "drawdown": {"status": "pass" if strategy["portfolio"]["max_drawdown"] <= policy.max_drawdown else "fail", "actual": strategy["portfolio"]["max_drawdown"], "required": policy.max_drawdown},
    }
    status = "ready_for_shadow_trial" if all(item["status"] == "pass" for item in gates.values()) else (
        "insufficient_evidence" if any(item["status"] == "insufficient" for item in gates.values()) else "not_qualified"
    )
    payload = {
        "schema_version": 1,
        "generated_at": policy.evaluation_as_of.isoformat(),
        "dataset_hash": frozen.dataset_hash,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "evidence": {
            "captured_at": frozen.captured_at.isoformat(),
            "split_at": policy.split_at.isoformat(),
            "evaluation_as_of": policy.evaluation_as_of.isoformat(),
            "calibration_records": len(calibration), "test_records": len(test),
            "excluded": dict(frozen.excluded),
        },
        "calibration_report": evaluation_report(calibration),
        "out_of_sample": {"strategy": strategy, "market_baseline_brier": strategy["forecast_metrics"]["market_brier"], "fixed_policy": fixed},
        "stress": stresses,
        "unsupported_assumptions": [
            "captured REST quotes do not prove queue position",
            "modeled fees and slippage may differ from venue execution",
            "historical availability does not prove future provider availability",
        ],
        "gates": gates,
        "status": status,
        "live_eligible": False,
    }
    payload["artifact_hash"] = canonical_hash(payload)
    trace.get_current_span().set_attributes({"replay.status": status, "replay.test_records": len(test)})
    replay_evaluations.add(1, {"status": status})
    return payload


def validate_readiness_artifact(
    artifact: Mapping[str, Any], *, now: datetime, max_age_seconds: int,
) -> None:
    required = {
        "schema_version", "generated_at", "dataset_hash", "config_hash", "code_hash", "evidence",
        "calibration_report", "out_of_sample", "stress", "unsupported_assumptions", "gates",
        "status", "live_eligible", "artifact_hash",
    }
    issues: list[str] = []
    if set(artifact) != required or artifact.get("schema_version") != 1:
        issues.append("artifact schema is invalid")
    payload = dict(artifact)
    saved_hash = payload.pop("artifact_hash", None)
    if saved_hash != canonical_hash(payload):
        issues.append("artifact hash does not match its contents")
    for name in ("dataset_hash", "config_hash", "code_hash"):
        value = artifact.get(name)
        if not isinstance(value, str) or len(value) != 64:
            issues.append(f"{name} must be a SHA-256 digest")
        else:
            try:
                int(value, 16)
            except ValueError:
                issues.append(f"{name} must be a SHA-256 digest")
    try:
        generated_at = datetime.fromisoformat(str(artifact.get("generated_at")).replace("Z", "+00:00"))
        if generated_at.tzinfo is None or now.tzinfo is None:
            raise ValueError
        age = (now.astimezone(timezone.utc) - generated_at.astimezone(timezone.utc)).total_seconds()
        if age < 0 or age > max_age_seconds:
            issues.append("artifact is stale or future-dated")
    except ValueError:
        issues.append("generated_at must be timezone-aware")
    if artifact.get("live_eligible") is not False:
        issues.append("T12 readiness cannot authorize live trading")
    if issues:
        raise ReadinessArtifactError("; ".join(issues))


def evaluate_dataset(
    dataset: Mapping[str, Any], config: Mapping[str, Any], *, code_hash: str,
) -> dict[str, Any]:
    """Convenience boundary shared by the CLI and tests."""
    try:
        policy = ReplayPolicy.from_mapping(config)
        return evaluate_replay(dataset, policy=policy, code_hash=code_hash, config_hash=canonical_hash(config))
    except (ReplayValidationError, ValueError):
        replay_evaluations.add(1, {"status": "invalid"})
        raise
