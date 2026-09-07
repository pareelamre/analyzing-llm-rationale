"""Versioned causal replay inputs and event-disjoint time splits."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

from ..forecast_evaluation import ResolvedForecast


class ReplayValidationError(ValueError):
    """Captured replay input is ambiguous, future-dated, or malformed."""


def _timestamp(value: Any, *, field: str = "timestamp") -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReplayValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ReplayValidationError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def causal_events(events: Sequence[Mapping], *, as_of: datetime) -> list[Mapping]:
    """Return one immutable fact per ID available at the frozen decision time."""
    as_of = _timestamp(as_of, field="as_of")
    selected: dict[str, Mapping] = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise ReplayValidationError("replay event must be an object")
        observed = _timestamp(event.get("observed_at"), field="observed_at")
        occurred = _timestamp(event.get("occurred_at", event.get("observed_at")), field="occurred_at")
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            raise ReplayValidationError("replay events require an ID")
        if observed > as_of or occurred > as_of:
            continue
        prior = selected.get(event_id)
        if prior is not None and _canonical(prior) != _canonical(event):
            raise ReplayValidationError(f"conflicting replay event ID: {event_id}")
        selected[event_id] = dict(event)
    return [selected[key] for key in sorted(selected)]


@dataclass(frozen=True)
class ReplayRecord:
    forecast_id: str
    event_cluster_id: str
    instrument_id: str
    platform: str
    model: str
    forecast_observed_at: datetime
    forecast_occurred_at: datetime
    outcome_observed_at: datetime
    resolved_at: datetime
    model_probability: float
    market_probability: float
    outcome: int
    domain: str = "other"
    market_bid: float | None = None
    market_ask: float | None = None

    def __post_init__(self) -> None:
        for name in ("forecast_id", "event_cluster_id", "instrument_id", "platform", "model"):
            if not str(getattr(self, name)).strip():
                raise ReplayValidationError(f"{name} is required")
        for name in ("forecast_observed_at", "forecast_occurred_at", "outcome_observed_at", "resolved_at"):
            object.__setattr__(self, name, _timestamp(getattr(self, name), field=name))
        if self.forecast_observed_at < self.forecast_occurred_at:
            raise ReplayValidationError("forecast cannot be observed before it occurs")
        if self.resolved_at < self.forecast_occurred_at:
            raise ReplayValidationError("resolution cannot precede forecast occurrence")
        if self.outcome_observed_at < self.resolved_at:
            raise ReplayValidationError("outcome cannot be observed before resolution")
        for name in ("model_probability", "market_probability", "market_bid", "market_ask"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ReplayValidationError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        if self.outcome not in {0, 1}:
            raise ReplayValidationError("outcome must be binary")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ReplayRecord":
        allowed = {
            "forecast_id", "event_cluster_id", "instrument_id", "platform", "model",
            "forecast_observed_at", "forecast_occurred_at", "outcome_observed_at", "resolved_at",
            "model_probability", "market_probability", "outcome", "domain", "market_bid", "market_ask",
        }
        if set(row) - allowed:
            raise ReplayValidationError("replay record contains unknown fields")
        try:
            return cls(**dict(row))
        except TypeError as exc:
            raise ReplayValidationError("replay record schema is incomplete") from exc

    def to_forecast(self) -> ResolvedForecast:
        return ResolvedForecast(
            self.forecast_id, self.platform, self.instrument_id, self.model,
            self.forecast_occurred_at, self.resolved_at, self.model_probability,
            self.market_probability, self.outcome, self.domain,
            market_bid=self.market_bid, market_ask=self.market_ask,
        )


@dataclass(frozen=True)
class FrozenReplay:
    dataset_hash: str
    captured_at: datetime
    calibration: tuple[ReplayRecord, ...]
    test: tuple[ReplayRecord, ...]
    excluded: Mapping[str, int]


def split_replay_dataset(
    dataset: Mapping[str, Any], *, split_at: datetime, evaluation_as_of: datetime,
) -> FrozenReplay:
    """Validate, causally freeze, deduplicate, and split by time and event cluster."""
    split_at = _timestamp(split_at, field="split_at")
    evaluation_as_of = _timestamp(evaluation_as_of, field="evaluation_as_of")
    if evaluation_as_of <= split_at:
        raise ReplayValidationError("evaluation_as_of must follow split_at")
    if set(dataset) != {"schema_version", "captured_at", "records"} or dataset.get("schema_version") != 1:
        raise ReplayValidationError("replay dataset schema version is invalid")
    captured_at = _timestamp(dataset.get("captured_at"), field="captured_at")
    if captured_at > evaluation_as_of:
        raise ReplayValidationError("dataset was captured after the frozen evaluation time")
    rows = dataset.get("records")
    if not isinstance(rows, list):
        raise ReplayValidationError("replay records must be a list")
    records_by_id: dict[str, ReplayRecord] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ReplayValidationError("replay records must be objects")
        record = ReplayRecord.from_mapping(raw)
        prior = records_by_id.get(record.forecast_id)
        if prior is not None and prior != record:
            raise ReplayValidationError(f"conflicting replay forecast ID: {record.forecast_id}")
        records_by_id[record.forecast_id] = record

    excluded = {
        "future_forecast": 0, "future_outcome": 0, "leaked_outcome": 0,
        "cross_split_unresolved": 0, "duplicate_cluster": 0, "training_cluster": 0,
    }
    available_at = min(captured_at, evaluation_as_of)
    calibration_candidates: list[ReplayRecord] = []
    test_candidates: list[ReplayRecord] = []
    for record in records_by_id.values():
        if record.forecast_observed_at > available_at or record.forecast_occurred_at > available_at:
            excluded["future_forecast"] += 1
            continue
        if record.outcome_observed_at > available_at:
            excluded["future_outcome"] += 1
            continue
        if record.outcome_observed_at <= record.forecast_observed_at:
            excluded["leaked_outcome"] += 1
            continue
        if record.forecast_occurred_at < split_at and record.outcome_observed_at < split_at:
            calibration_candidates.append(record)
        elif record.forecast_occurred_at >= split_at:
            test_candidates.append(record)
        else:
            excluded["cross_split_unresolved"] += 1

    calibration = _one_per_cluster(calibration_candidates, excluded)
    training_clusters = {item.event_cluster_id for item in calibration}
    before = len(test_candidates)
    test_candidates = [item for item in test_candidates if item.event_cluster_id not in training_clusters]
    excluded["training_cluster"] += before - len(test_candidates)
    test = _one_per_cluster(test_candidates, excluded)
    return FrozenReplay(canonical_hash(dataset), captured_at, tuple(calibration), tuple(test), dict(excluded))


def _one_per_cluster(records: Sequence[ReplayRecord], excluded: dict[str, int]) -> list[ReplayRecord]:
    chosen: dict[str, ReplayRecord] = {}
    for record in sorted(records, key=lambda item: (item.forecast_occurred_at, item.forecast_id)):
        if record.event_cluster_id in chosen:
            excluded["duplicate_cluster"] += 1
            continue
        chosen[record.event_cluster_id] = record
    return sorted(chosen.values(), key=lambda item: (item.forecast_occurred_at, item.forecast_id))
