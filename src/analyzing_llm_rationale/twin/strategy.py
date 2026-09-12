"""Deterministic ``foresea_edge_v1`` lifecycle and durable cycle decisions."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from opentelemetry import metrics, trace

from ..config import load_yaml
from .account import AccountSnapshot
from .budget import BudgetExceeded
from .models import (
    AccountScope,
    Completeness,
    Forecast,
    Instrument,
    MarketSnapshot,
    ProposalAction,
    TradeIntent,
)
from .research_gateway import ResearchResult
from .risk import (
    CalibrationResult,
    RiskExposure,
    RiskLimits,
    RiskResult,
    calibrate_probability,
    evaluate_binary_candidate,
)
from .store import AccountProjection

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
strategy_cycles = metrics.get_meter(__name__).create_counter("twin.strategy.cycles", unit="1")
decision_reasons = metrics.get_meter(__name__).create_counter("twin.decisions", unit="1")

_ZERO = Decimal("0")
_ONE = Decimal("1")
STRATEGY_VERSION = "foresea_edge_v1"


@dataclass(frozen=True)
class StrategyStep:
    stage: str
    outcome: str
    reason: str
    reference_id: Optional[str] = None

    def to_storage(self) -> dict[str, Any]:
        return {
            "stage": self.stage, "outcome": self.outcome,
            "reason": self.reason, "reference_id": self.reference_id,
        }


@dataclass(frozen=True)
class StrategyCycle:
    key: str
    decision: str
    reason: str
    exits_evaluated: bool = False
    intent: Optional[TradeIntent] = None
    risk_result: Optional[RiskResult] = None
    steps: tuple[StrategyStep, ...] = ()
    created_at: Optional[datetime] = None
    strategy_version: str = STRATEGY_VERSION
    account_scope_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.key).strip() or self.decision not in {"INTENT", "HOLD", "PASS", "SHADOW_SUBMITTED"}:
            raise ValueError("invalid strategy cycle identity or decision")
        if (self.decision == "INTENT") != (self.intent is not None and self.risk_result is not None):
            raise ValueError("INTENT cycles require a trade intent and accepted risk result")
        if self.decision != "INTENT" and (self.intent is not None or self.risk_result is not None):
            raise ValueError("non-INTENT cycles cannot carry execution material")
        if self.created_at is not None and self.created_at.tzinfo is None:
            raise ValueError("strategy cycle time must be timezone-aware")
        if self.account_scope_id is not None and not str(self.account_scope_id).strip():
            raise ValueError("strategy cycle account scope must be nonempty")

    def to_storage(self) -> dict[str, Any]:
        return {
            "key": self.key, "decision": self.decision, "reason": self.reason,
            "exits_evaluated": self.exits_evaluated,
            "intent": self.intent.to_storage() if self.intent is not None else None,
            "risk_result": self.risk_result.to_storage() if self.risk_result is not None else None,
            "steps": [step.to_storage() for step in self.steps],
            "created_at": self.created_at.isoformat() if self.created_at is not None else None,
            "strategy_version": self.strategy_version,
            "account_scope_id": self.account_scope_id,
        }

    @classmethod
    def from_storage(cls, payload: Mapping[str, Any]) -> "StrategyCycle":
        required = {
            "key", "decision", "reason", "exits_evaluated", "intent", "risk_result", "steps",
            "created_at", "strategy_version",
        }
        allowed = required | {"account_scope_id"}
        if not required.issubset(payload) or not set(payload).issubset(allowed) or not isinstance(payload.get("steps"), list):
            raise ValueError("stored strategy cycle schema is invalid")
        intent_payload = payload.get("intent")
        intent = TradeIntent.from_storage(intent_payload) if isinstance(intent_payload, Mapping) else None
        risk_payload = payload.get("risk_result")
        risk_result = RiskResult.from_storage(risk_payload) if isinstance(risk_payload, Mapping) else None
        created = datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00")) if payload.get("created_at") else None
        return cls(
            str(payload["key"]), str(payload["decision"]), str(payload["reason"]),
            bool(payload["exits_evaluated"]), intent, risk_result,
            tuple(StrategyStep(**dict(step)) for step in payload["steps"]),
            created, str(payload["strategy_version"]),
            str(payload["account_scope_id"]) if payload.get("account_scope_id") else None,
        )


@dataclass(frozen=True)
class StrategyPolicy:
    config_version: str
    risk_limits: RiskLimits
    net_edge_floor: Decimal = Decimal("0.05")
    maximum_holding_seconds: int = 7 * 24 * 60 * 60
    candidate_cooldown_seconds: int = 60 * 60
    material_price_change: Decimal = Decimal("0.01")
    cycle_bucket_seconds: int = 5 * 60
    intent_ttl_seconds: int = 60
    max_research_candidates: int = 3
    account_max_age_seconds: int = 60
    max_new_positions_per_cycle: int = 1
    max_open_instruments: int = 5

    def __post_init__(self) -> None:
        if not str(self.config_version).strip() or not isinstance(self.risk_limits, RiskLimits):
            raise ValueError("strategy policy requires versioned risk limits")
        for name in ("net_edge_floor", "material_price_change"):
            value = _decimal(name, getattr(self, name))
            if not _ZERO <= value < _ONE:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, value)
        for name in (
            "maximum_holding_seconds", "candidate_cooldown_seconds", "cycle_bucket_seconds",
            "intent_ttl_seconds", "max_research_candidates", "account_max_age_seconds",
            "max_new_positions_per_cycle", "max_open_instruments",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_new_positions_per_cycle != 1:
            raise ValueError("foresea_edge_v1 permits exactly one new position per cycle")


@dataclass(frozen=True)
class HeldPosition:
    instrument: Instrument
    outcome: str
    quantity: Decimal
    opened_at: datetime
    thesis_expires_at: datetime
    settlement_spec_hash: str
    policy_version: str
    pending_sell_quantity: Decimal = _ZERO
    thesis_valid: bool = True
    latest_forecast: Optional[Forecast] = None

    def __post_init__(self) -> None:
        outcome = str(self.outcome).lower().strip()
        if outcome not in {"yes", "no"}:
            raise ValueError("held outcome must be yes or no")
        quantity = _decimal("quantity", self.quantity)
        pending = _decimal("pending_sell_quantity", self.pending_sell_quantity)
        if quantity <= _ZERO or pending < _ZERO or pending > quantity:
            raise ValueError("held and pending quantities are inconsistent")
        if self.opened_at.tzinfo is None or self.thesis_expires_at.tzinfo is None:
            raise ValueError("position lifecycle timestamps must be timezone-aware")
        if self.latest_forecast is not None:
            if not isinstance(self.latest_forecast, Forecast):
                raise ValueError("latest position forecast must be a typed Forecast")
            if self.latest_forecast.instrument_id != self.instrument.id:
                raise ValueError("latest position forecast must match the held instrument")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "pending_sell_quantity", pending)


@dataclass(frozen=True)
class StrategyCandidate:
    instrument: Instrument
    snapshot: MarketSnapshot
    yes_depth: Decimal
    no_depth: Decimal
    fee_per_share: Decimal
    slippage_per_share: Decimal
    calibration_observations: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.snapshot.instrument_id != self.instrument.id:
            raise ValueError("candidate market and instrument identities do not match")
        if self.snapshot.complete is not Completeness.COMPLETE:
            raise ValueError("strategy candidates require a complete market snapshot")
        for name in ("yes_depth", "no_depth", "fee_per_share", "slippage_per_share"):
            value = _decimal(name, getattr(self, name))
            if value < _ZERO:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        observations = tuple(self.calibration_observations)
        if any(not isinstance(item, Mapping) for item in observations):
            raise ValueError("calibration observations must be objects")
        object.__setattr__(self, "calibration_observations", observations)

    @property
    def midpoint(self) -> Optional[Decimal]:
        if self.snapshot.yes_bid is None or self.snapshot.yes_ask is None:
            return None
        return (self.snapshot.yes_bid + self.snapshot.yes_ask) / Decimal("2")

    @property
    def rules_hash(self) -> str:
        return _hash({
            "settlement": self.instrument.settlement_spec_hash,
            "fee": self.snapshot.fee_version, "capability": self.instrument.capability_version,
            "tick": str(self.instrument.tick_size), "minimum": str(self.instrument.min_quantity),
        })


@dataclass(frozen=True)
class StrategyAccountState:
    account_snapshot: AccountSnapshot
    account_projection: AccountProjection
    positions: tuple[HeldPosition, ...]
    exposures: tuple[RiskExposure, ...]
    trailing_additions: Decimal
    realized_losses: Decimal
    peak_equity: Decimal
    current_equity: Decimal
    portfolio_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.account_snapshot, AccountSnapshot):
            raise ValueError("strategy state requires a canonical account snapshot")
        if not isinstance(self.account_projection, AccountProjection):
            raise ValueError("strategy state requires an account projection")
        if self.account_snapshot.scope_id != self.account_projection.scope_id:
            raise ValueError("strategy account snapshot and projection scopes do not match")
        positions = tuple(self.positions)
        exposures = tuple(self.exposures)
        if any(not isinstance(item, HeldPosition) for item in positions):
            raise ValueError("strategy positions must be typed held positions")
        if any(not isinstance(item, RiskExposure) for item in exposures):
            raise ValueError("strategy exposures must be typed risk exposures")
        for name in ("trailing_additions", "realized_losses", "peak_equity", "current_equity"):
            value = _decimal(name, getattr(self, name))
            if value < _ZERO:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.portfolio_complete, bool):
            raise ValueError("portfolio completeness must be boolean")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "exposures", exposures)


@dataclass(frozen=True)
class CandidateMemory:
    instrument_id: str
    market_snapshot_id: str
    midpoint: Optional[Decimal]
    rules_hash: str
    config_version: str
    researched_at: datetime

    def __post_init__(self) -> None:
        if not str(self.instrument_id).strip() or not str(self.market_snapshot_id).strip():
            raise ValueError("candidate memory requires market identity")
        if len(str(self.rules_hash)) != 64 or not str(self.config_version).strip():
            raise ValueError("candidate memory requires rules and config versions")
        try:
            int(self.rules_hash, 16)
        except ValueError as exc:
            raise ValueError("candidate rules hash must be a SHA-256 digest") from exc
        if self.midpoint is not None:
            midpoint = _decimal("midpoint", self.midpoint)
            if not _ZERO <= midpoint <= _ONE:
                raise ValueError("candidate midpoint must be in [0, 1]")
            object.__setattr__(self, "midpoint", midpoint)
        if self.researched_at.tzinfo is None:
            raise ValueError("candidate research time must be timezone-aware")

    def to_storage(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id, "market_snapshot_id": self.market_snapshot_id,
            "midpoint": str(self.midpoint) if self.midpoint is not None else None,
            "rules_hash": self.rules_hash, "config_version": self.config_version,
            "researched_at": self.researched_at.isoformat(),
        }

    @classmethod
    def from_storage(cls, payload: Mapping[str, Any]) -> "CandidateMemory":
        return cls(
            str(payload["instrument_id"]), str(payload["market_snapshot_id"]),
            _decimal("midpoint", payload["midpoint"]) if payload.get("midpoint") is not None else None,
            str(payload["rules_hash"]), str(payload["config_version"]),
            datetime.fromisoformat(str(payload["researched_at"]).replace("Z", "+00:00")),
        )


class StrategyStore(Protocol):
    durable: bool

    def get_cycle(self, key: str) -> Optional[StrategyCycle]: ...

    def record_cycle(self, cycle: StrategyCycle) -> bool: ...

    def cycles(self, scope_ids: frozenset[str], *, limit: int = 100) -> tuple[StrategyCycle, ...]: ...

    def get_candidate(self, scope_id: str, instrument_id: str) -> Optional[CandidateMemory]: ...

    def record_candidate(self, scope_id: str, memory: CandidateMemory) -> None: ...


class InMemoryStrategyStore:
    durable = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cycles: dict[str, StrategyCycle] = {}
        self._candidates: dict[tuple[str, str], CandidateMemory] = {}

    def get_cycle(self, key: str) -> Optional[StrategyCycle]:
        with self._lock:
            return self._cycles.get(key)

    def record_cycle(self, cycle: StrategyCycle) -> bool:
        with self._lock:
            existing = self._cycles.get(cycle.key)
            if existing is not None:
                return False
            self._cycles[cycle.key] = cycle
            return True

    def cycles(self, scope_ids: frozenset[str], *, limit: int = 100) -> tuple[StrategyCycle, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("strategy cycle page limit must be within 1..200")
        with self._lock:
            return tuple(sorted(
                (
                    cycle for cycle in self._cycles.values()
                    if cycle.account_scope_id in scope_ids
                ),
                key=lambda cycle: (cycle.created_at or datetime.min.replace(tzinfo=timezone.utc), cycle.key),
                reverse=True,
            )[:limit])

    def get_candidate(self, scope_id: str, instrument_id: str) -> Optional[CandidateMemory]:
        with self._lock:
            return self._candidates.get((scope_id, instrument_id))

    def record_candidate(self, scope_id: str, memory: CandidateMemory) -> None:
        with self._lock:
            key = (scope_id, memory.instrument_id)
            existing = self._candidates.get(key)
            if existing is not None:
                if existing.researched_at > memory.researched_at:
                    return
                if existing.researched_at == memory.researched_at and existing != memory:
                    raise ValueError("candidate memory timestamp has conflicting observations")
            self._candidates[key] = memory


class DatastoreStrategyStore:
    """Create-once decisions and durable cooldown memory for worker restarts."""

    durable = True

    def __init__(self, client: Any, *, namespace: str = "foresea-twin") -> None:
        self._client = client
        self._namespace = namespace

    def _cycle_key(self, key: str):
        return self._client.key("TwinStrategyCycle", sha256(key.encode("utf-8")).hexdigest(), namespace=self._namespace)

    def _candidate_key(self, scope_id: str, instrument_id: str):
        identity = sha256(f"{scope_id}:{instrument_id}".encode("utf-8")).hexdigest()
        return self._client.key("TwinStrategyCandidate", identity, namespace=self._namespace)

    def get_cycle(self, key: str) -> Optional[StrategyCycle]:
        entity = self._client.get(self._cycle_key(key))
        if entity is None:
            return None
        payload = json.loads(str(entity["payload_json"]))
        if entity.get("cycle_key") != key or entity.get("fingerprint") != _hash(payload):
            raise ValueError("stored strategy cycle failed integrity validation")
        return StrategyCycle.from_storage(payload)

    def record_cycle(self, cycle: StrategyCycle) -> bool:
        from google.cloud import datastore

        payload = cycle.to_storage()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        if len(encoded.encode("utf-8")) > 200_000:
            raise ValueError("strategy cycle exceeds the persistence boundary")
        key = self._cycle_key(cycle.key)
        with self._client.transaction():
            existing = self._client.get(key)
            if existing is not None:
                stored = json.loads(str(existing["payload_json"]))
                if existing.get("cycle_key") != cycle.key or existing.get("fingerprint") != _hash(stored):
                    raise ValueError("stored strategy cycle failed integrity validation")
                return False
            entity = datastore.Entity(key=key, exclude_from_indexes=("payload_json",))
            entity.update({
                "cycle_key": cycle.key, "fingerprint": _hash(payload), "payload_json": encoded,
                "account_scope_id": cycle.account_scope_id,
                "created_at": cycle.created_at,
                "decision": cycle.decision,
                "reason": cycle.reason,
            })
            self._client.put(entity)
            return True

    def cycles(self, scope_ids: frozenset[str], *, limit: int = 100) -> tuple[StrategyCycle, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("strategy cycle page limit must be within 1..200")
        from google.cloud.datastore.query import PropertyFilter

        cycles: list[StrategyCycle] = []
        for scope_id in sorted(scope_ids):
            query = self._client.query(kind="TwinStrategyCycle", namespace=self._namespace)
            query.add_filter(filter=PropertyFilter("account_scope_id", "=", scope_id))
            for entity in query.fetch(limit=limit):
                payload = json.loads(str(entity["payload_json"]))
                if entity.get("cycle_key") != payload.get("key") or entity.get("fingerprint") != _hash(payload):
                    raise ValueError("stored strategy cycle failed integrity validation")
                cycles.append(StrategyCycle.from_storage(payload))
        return tuple(sorted(
            cycles,
            key=lambda cycle: (cycle.created_at or datetime.min.replace(tzinfo=timezone.utc), cycle.key),
            reverse=True,
        )[:limit])

    def get_candidate(self, scope_id: str, instrument_id: str) -> Optional[CandidateMemory]:
        entity = self._client.get(self._candidate_key(scope_id, instrument_id))
        if entity is None:
            return None
        return CandidateMemory.from_storage(entity)

    def record_candidate(self, scope_id: str, memory: CandidateMemory) -> None:
        from google.cloud import datastore

        key = self._candidate_key(scope_id, memory.instrument_id)
        with self._client.transaction():
            existing_entity = self._client.get(key)
            if existing_entity is not None:
                existing = CandidateMemory.from_storage(existing_entity)
                if existing.researched_at > memory.researched_at:
                    return
                if existing.researched_at == memory.researched_at and existing != memory:
                    raise ValueError("candidate memory timestamp has conflicting observations")
            entity = datastore.Entity(key=key)
            entity.update(memory.to_storage())
            self._client.put(entity)


def strategy_cycle_key(
    *, scope: AccountScope, now: datetime, config_version: str,
    bucket_seconds: int, strategy_version: str = STRATEGY_VERSION,
) -> str:
    return strategy_cycle_key_for_identity(
        scope_id=scope.id, account_epoch=scope.account_epoch, now=now,
        config_version=config_version, bucket_seconds=bucket_seconds,
        strategy_version=strategy_version,
    )


def strategy_cycle_key_for_identity(
    *, scope_id: str, account_epoch: int, now: datetime, config_version: str,
    bucket_seconds: int, strategy_version: str = STRATEGY_VERSION,
) -> str:
    """Build the cycle key before account material is loaded by the worker."""
    if now.tzinfo is None or bucket_seconds <= 0:
        raise ValueError("cycle key requires an aware time and positive bucket")
    if not str(scope_id).strip() or type(account_epoch) is not int or account_epoch < 1:
        raise ValueError("cycle key requires a stable scope and positive account epoch")
    bucket = int(now.astimezone(timezone.utc).timestamp()) // bucket_seconds
    return "strategy-cycle:" + _hash({
        "strategy": strategy_version, "scope": scope_id, "account_epoch": account_epoch,
        "bucket": bucket, "config_version": config_version,
    })[:32]


class ForeseaEdgeStrategy:
    def __init__(
        self, *, store: Optional[StrategyStore] = None,
        policy: Optional[StrategyPolicy] = None,
    ) -> None:
        self.store = store or InMemoryStrategyStore()
        self.policy = policy
        self._finished: dict[str, StrategyCycle] = {}

    def run(
        self, key: str, *, reconcile: Callable[[], bool], research: Callable[[], bool],
        risk: Callable[[], bool], submit_shadow: Callable[[], None],
        evaluate_exits: Optional[Callable[[], bool]] = None,
    ) -> StrategyCycle:
        """Compatibility surface for the original shadow-only smoke tests."""
        if not key.strip():
            raise ValueError("strategy cycle key is required")
        if key in self._finished:
            return self._finished[key]
        if not reconcile():
            cycle = StrategyCycle(key, "HOLD", "account_incomplete")
        elif evaluate_exits is not None and not evaluate_exits():
            cycle = StrategyCycle(key, "HOLD", "exit_evaluation_failed", True)
        elif not research():
            cycle = StrategyCycle(key, "PASS", "research_unavailable", evaluate_exits is not None)
        elif not risk():
            cycle = StrategyCycle(key, "PASS", "risk_rejected", evaluate_exits is not None)
        else:
            submit_shadow()
            cycle = StrategyCycle(key, "SHADOW_SUBMITTED", "eligible", evaluate_exits is not None)
        self._finished[key] = cycle
        return cycle

    @tracer.start_as_current_span("twin.strategy.run_cycle")
    def run_cycle(
        self, *, scope: AccountScope, now: datetime,
        reconcile: Callable[[], Optional[StrategyAccountState]],
        load_position_market: Callable[[HeldPosition], Optional[StrategyCandidate]],
        discover: Callable[[], Sequence[StrategyCandidate]],
        research: Callable[[StrategyCandidate], ResearchResult],
        policy_stop: bool = False,
    ) -> StrategyCycle:
        """Reconcile, maintain positions, then consider at most one new intent."""
        if self.policy is None:
            raise ValueError("typed strategy cycles require a StrategyPolicy")
        if now.tzinfo is None:
            raise ValueError("strategy cycle time must be timezone-aware")
        key = strategy_cycle_key(
            scope=scope, now=now, config_version=self.policy.config_version,
            bucket_seconds=self.policy.cycle_bucket_seconds,
        )
        span = trace.get_current_span()
        span.set_attributes({"strategy.version": STRATEGY_VERSION, "account.scope_id": scope.id, "strategy.cycle_key": key})
        existing = self.store.get_cycle(key)
        if existing is not None:
            span.set_attribute("outcome", "reused")
            strategy_cycles.add(1, {"decision": existing.decision, "outcome": "reused"})
            decision_reasons.add(1, {"decision": existing.decision, "reason": existing.reason})
            return existing
        steps: list[StrategyStep] = []
        try:
            state = reconcile()
        except Exception as exc:
            logger.warning("Twin strategy reconciliation failed (%s)", type(exc).__name__)
            span.record_exception(ValueError(type(exc).__name__))
            return self._finish_with_scope(scope.id, key, "HOLD", "account_reconciliation_failed", steps, now, span)
        if (
            state is None or state.account_snapshot.completeness is not Completeness.COMPLETE
            or state.account_snapshot.blocks_new_exposure or not state.portfolio_complete
        ):
            steps.append(StrategyStep("reconcile", "hold", "account_incomplete"))
            return self._finish_with_scope(scope.id, key, "HOLD", "account_incomplete", steps, now, span)
        if (
            state.account_snapshot.scope_id != scope.id
            or state.account_projection.scope_id != scope.id
            or state.account_projection.account_epoch != scope.account_epoch
        ):
            steps.append(StrategyStep("reconcile", "hold", "account_scope_mismatch"))
            return self._finish_with_scope(scope.id, key, "HOLD", "account_scope_mismatch", steps, now, span)
        steps.append(StrategyStep("reconcile", "ok", "complete", str(state.account_snapshot.generation)))

        for position in sorted(state.positions, key=lambda item: (item.instrument.id, item.outcome)):
            candidate = load_position_market(position)
            if candidate is None or candidate.instrument.id != position.instrument.id:
                steps.append(StrategyStep("exit", "hold", "exit_market_unavailable", position.instrument.id))
                return self._finish_with_scope(scope.id, key, "HOLD", "exit_market_unavailable", steps, now, span, exits=True)
            exit_reason = self._exit_reason(position, candidate=candidate, now=now, policy_stop=policy_stop)
            if exit_reason is None:
                steps.append(StrategyStep("exit", "hold", "position_retained", position.instrument.id))
                continue
            action = ProposalAction.SELL_YES if position.outcome == "yes" else ProposalAction.SELL_NO
            result = self._risk(
                action, candidate, state, now=now, calibration=None,
                requested_quantity=position.quantity, held_quantity=position.quantity,
                pending_sell_quantity=position.pending_sell_quantity,
            )
            if result.reason is not None:
                steps.append(StrategyStep("exit", "hold", result.reason, position.instrument.id))
                return self._finish_with_scope(scope.id, key, "HOLD", "exit_risk_blocked", steps, now, span, exits=True)
            trade = self._intent(
                scope, candidate, result, action=action, now=now,
                forecast=None, exit_reason=exit_reason,
            )
            steps.append(StrategyStep("exit", "intent", exit_reason, trade.id))
            return self._finish_with_scope(
                scope.id,
                key, "INTENT", exit_reason, steps, now, span,
                intent=trade, risk_result=result, exits=True,
            )
        steps.append(StrategyStep("exit", "ok", "positions_reviewed"))
        if len({position.instrument.id for position in state.positions}) >= self.policy.max_open_instruments:
            steps.append(StrategyStep("selection", "hold", "open_position_limit"))
            return self._finish_with_scope(scope.id, key, "HOLD", "open_position_limit", steps, now, span, exits=True)

        try:
            discovered = tuple(discover())
            if any(not isinstance(item, StrategyCandidate) for item in discovered):
                raise ValueError("discovery returned an invalid strategy candidate")
            candidates_by_instrument: dict[str, StrategyCandidate] = {}
            for item in sorted(
                discovered,
                key=lambda candidate: (
                    candidate.instrument.id, -candidate.snapshot.sequence, candidate.snapshot.id,
                ),
            ):
                candidates_by_instrument.setdefault(item.instrument.id, item)
            candidates = list(candidates_by_instrument.values())
        except Exception as exc:
            logger.warning("Twin strategy discovery failed (%s)", type(exc).__name__)
            span.record_exception(ValueError(type(exc).__name__))
            steps.append(StrategyStep("discovery", "hold", "discovery_unavailable"))
            return self._finish_with_scope(scope.id, key, "HOLD", "discovery_unavailable", steps, now, span, exits=True)
        eligible = [candidate for candidate in candidates if self._candidate_changed(scope.id, candidate, now)]
        if not eligible:
            steps.append(StrategyStep("discovery", "hold", "no_changed_candidates"))
            return self._finish_with_scope(scope.id, key, "HOLD", "no_changed_candidates", steps, now, span, exits=True)

        attempted = 0
        for candidate in eligible:
            if attempted >= self.policy.max_research_candidates:
                break
            attempted += 1
            self.store.record_candidate(scope.id, CandidateMemory(
                candidate.instrument.id, candidate.snapshot.id, candidate.midpoint,
                candidate.rules_hash, self.policy.config_version, now,
            ))
            try:
                result = research(candidate)
            except BudgetExceeded:
                steps.append(StrategyStep("research", "pass", "budget_exhausted", candidate.instrument.id))
                return self._finish_with_scope(scope.id, key, "PASS", "budget_exhausted", steps, now, span, exits=True)
            except Exception as exc:
                logger.warning("Twin strategy research unavailable (%s)", type(exc).__name__)
                span.record_exception(ValueError(type(exc).__name__))
                steps.append(StrategyStep("research", "pass", "research_unavailable", candidate.instrument.id))
                return self._finish_with_scope(scope.id, key, "PASS", "research_unavailable", steps, now, span, exits=True)
            if result.forecast is None:
                reason = result.proposal.pass_decision.reason.value if result.proposal.pass_decision else "research_pass"
                steps.append(StrategyStep("research", "pass", reason, candidate.instrument.id))
                continue
            forecast = result.forecast
            if (
                forecast.instrument_id != candidate.instrument.id
                or result.proposal.market_snapshot_id != candidate.snapshot.id
                or not forecast.as_of <= now < forecast.expires_at
            ):
                steps.append(StrategyStep("research", "pass", "forecast_stale_or_mismatched", forecast.id))
                continue
            calibration = calibrate_probability(
                forecast.p_yes_raw, candidate.calibration_observations, as_of=now,
                model_hash=forecast.model_hash, prompt_hash=forecast.prompt_hash,
                category_family=candidate.instrument.category,
            )
            if calibration.probability is None:
                steps.append(StrategyStep("calibration", "pass", calibration.reason or "calibration_unavailable", forecast.id))
                continue
            action = self._entry_action(candidate, calibration)
            if action is None:
                steps.append(StrategyStep("score", "pass", "no_net_edge", forecast.id))
                continue
            risk_result = self._risk(action, candidate, state, now=now, calibration=calibration)
            if risk_result.reason is not None:
                steps.append(StrategyStep("risk", "pass", risk_result.reason, forecast.id))
                continue
            trade = self._intent(
                scope, candidate, risk_result, action=action, now=now,
                forecast=forecast, exit_reason=None,
            )
            steps.extend((
                StrategyStep("research", "ok", "forecast_ready", forecast.id),
                StrategyStep("calibration", "ok", "calibration_v1", calibration.calibration_hash),
                StrategyStep("risk", "intent", "accepted", trade.id),
            ))
            return self._finish_with_scope(
                scope.id,
                key, "INTENT", "entry_eligible", steps, now, span,
                intent=trade, risk_result=risk_result, exits=True,
            )
        steps.append(StrategyStep("selection", "pass", "no_candidate_qualified"))
        return self._finish_with_scope(scope.id, key, "PASS", "no_candidate_qualified", steps, now, span, exits=True)

    def _candidate_changed(self, scope_id: str, candidate: StrategyCandidate, now: datetime) -> bool:
        memory = self.store.get_candidate(scope_id, candidate.instrument.id)
        if memory is None:
            return True
        if memory.market_snapshot_id == candidate.snapshot.id:
            return False
        if memory.config_version != self.policy.config_version or memory.rules_hash != candidate.rules_hash:
            return True
        if memory.midpoint is not None and candidate.midpoint is not None:
            if abs(memory.midpoint - candidate.midpoint) >= self.policy.material_price_change:
                return True
        return (now - memory.researched_at).total_seconds() >= self.policy.candidate_cooldown_seconds

    def _exit_reason(
        self, position: HeldPosition, *, candidate: StrategyCandidate,
        now: datetime, policy_stop: bool,
    ) -> Optional[str]:
        if policy_stop:
            return "policy_stop"
        if position.policy_version != self.policy.config_version:
            return "policy_changed"
        if now >= position.thesis_expires_at:
            return "thesis_expired"
        if (now - position.opened_at).total_seconds() >= self.policy.maximum_holding_seconds:
            return "maximum_holding_time"
        if position.settlement_spec_hash != position.instrument.settlement_spec_hash:
            return "settlement_rule_changed"
        if not position.thesis_valid:
            return "thesis_invalidated"
        forecast = position.latest_forecast
        if forecast is not None and forecast.as_of <= now < forecast.expires_at:
            costs = candidate.fee_per_share + candidate.slippage_per_share
            if (
                position.outcome == "yes"
                and forecast.uncertainty_high is not None
                and candidate.snapshot.yes_bid is not None
                and forecast.uncertainty_high <= candidate.snapshot.yes_bid + costs
            ):
                return "revised_forecast_invalidated"
            if (
                position.outcome == "no"
                and forecast.uncertainty_low is not None
                and candidate.snapshot.no_bid is not None
                and _ONE - forecast.uncertainty_low <= candidate.snapshot.no_bid + costs
            ):
                return "revised_forecast_invalidated"
        return None

    def _entry_action(self, candidate: StrategyCandidate, calibration: CalibrationResult) -> Optional[ProposalAction]:
        if calibration.lower_bound is None or calibration.upper_bound is None:
            return None
        yes_edge = (
            calibration.lower_bound - candidate.snapshot.yes_ask
            - candidate.fee_per_share - candidate.slippage_per_share
            if candidate.snapshot.yes_ask is not None else Decimal("-1")
        )
        no_edge = (
            (_ONE - calibration.upper_bound) - candidate.snapshot.no_ask
            - candidate.fee_per_share - candidate.slippage_per_share
            if candidate.snapshot.no_ask is not None else Decimal("-1")
        )
        best = max(yes_edge, no_edge)
        if best < self.policy.net_edge_floor:
            return None
        return ProposalAction.BUY_YES if yes_edge >= no_edge else ProposalAction.BUY_NO

    def _risk(
        self, action: ProposalAction, candidate: StrategyCandidate, state: StrategyAccountState,
        *, now: datetime, calibration: Optional[CalibrationResult],
        requested_quantity: Optional[Decimal] = None, held_quantity: Optional[Decimal] = None,
        pending_sell_quantity: Decimal = _ZERO,
    ) -> RiskResult:
        depth = candidate.yes_depth if action in {ProposalAction.BUY_YES, ProposalAction.SELL_YES} else candidate.no_depth
        return evaluate_binary_candidate(
            action=action, instrument_id=candidate.instrument.id, cluster_id=candidate.instrument.cluster_id,
            venue=candidate.instrument.venue, market_snapshot=candidate.snapshot,
            account_snapshot=state.account_snapshot, account_projection=state.account_projection,
            exposures=state.exposures, limits=self.policy.risk_limits, now=now,
            calibration=calibration, fee_per_share=candidate.fee_per_share,
            slippage_per_share=candidate.slippage_per_share, fee_version=candidate.snapshot.fee_version,
            available_depth=depth, tick_size=candidate.instrument.tick_size,
            min_quantity=candidate.instrument.min_quantity, trailing_additions=state.trailing_additions,
            realized_losses=state.realized_losses, peak_equity=state.peak_equity,
            current_equity=state.current_equity, requested_quantity=requested_quantity,
            held_quantity=held_quantity, pending_sell_quantity=pending_sell_quantity,
            account_max_age_seconds=self.policy.account_max_age_seconds,
            portfolio_complete=state.portfolio_complete,
        )

    def _intent(
        self, scope: AccountScope, candidate: StrategyCandidate, risk: RiskResult,
        *, action: ProposalAction, now: datetime, forecast: Optional[Forecast],
        exit_reason: Optional[str],
    ) -> TradeIntent:
        price = (
            candidate.snapshot.yes_ask if action is ProposalAction.BUY_YES else
            candidate.snapshot.no_ask if action is ProposalAction.BUY_NO else
            candidate.snapshot.yes_bid if action is ProposalAction.SELL_YES else candidate.snapshot.no_bid
        )
        if price is None:
            raise ValueError("accepted risk result has no executable price")
        expiry = min(
            candidate.instrument.close_at,
            now + timedelta(seconds=self.policy.intent_ttl_seconds),
            forecast.expires_at if forecast is not None else candidate.instrument.close_at,
        )
        identity = {
            "scope": scope.id, "epoch": scope.account_epoch, "instrument": candidate.instrument.id,
            "action": action.value, "quantity": str(risk.quantity), "price": str(price),
            "forecast": forecast.id if forecast is not None else None, "exit": exit_reason,
            "policy": self.policy.config_version, "market": candidate.snapshot.id, "expires": expiry.isoformat(),
        }
        return TradeIntent(
            "intent-" + _hash(identity)[:24], scope.id, scope.account_epoch,
            candidate.instrument.id, action, risk.quantity, price, "IOC",
            forecast.id if forecast is not None else None, exit_reason,
            self.policy.config_version, STRATEGY_VERSION, candidate.snapshot.id,
            candidate.fee_per_share, candidate.slippage_per_share, expiry, now,
        )

    def _finish_with_scope(self, scope_id: str, *args: Any, **kwargs: Any) -> StrategyCycle:
        kwargs["account_scope_id"] = scope_id
        return self._finish(*args, **kwargs)

    def _finish(
        self, key: str, decision: str, reason: str, steps: Sequence[StrategyStep],
        now: datetime, span: Any, *, intent: Optional[TradeIntent] = None,
        risk_result: Optional[RiskResult] = None,
        exits: bool = False, account_scope_id: Optional[str] = None,
    ) -> StrategyCycle:
        cycle = StrategyCycle(
            key, decision, reason, exits, intent, risk_result, tuple(steps), now,
            account_scope_id=account_scope_id,
        )
        created = self.store.record_cycle(cycle)
        if not created:
            existing = self.store.get_cycle(key)
            if existing is None:
                raise RuntimeError("strategy decision lost during idempotent persistence")
            cycle = existing
        span.set_attributes({"outcome": decision.lower(), "strategy.reason": reason, "strategy.steps": len(steps)})
        strategy_cycles.add(1, {"decision": decision, "outcome": "created" if created else "reused"})
        decision_reasons.add(1, {"decision": decision, "reason": reason})
        return cycle


def _hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_strategy_policy(path: Path) -> StrategyPolicy:
    """Load the shared YAML with fail-closed absolute monetary limits."""
    data = load_yaml(path)
    raw = data.get("strategy")
    if not isinstance(raw, Mapping) or raw.get("mode") != "shadow" or raw.get("live_enabled") is not False:
        raise ValueError("twin strategy configuration must remain explicitly shadow-only")
    risk = raw.get("risk")
    if not isinstance(risk, Mapping):
        raise ValueError("twin strategy risk configuration is required")
    return StrategyPolicy(
        config_version=str(raw["config_version"]),
        risk_limits=RiskLimits(
            _decimal("kelly_fraction", risk["kelly_fraction"]),
            _decimal("max_order_cash", risk["max_order_cash"]),
            _decimal("max_market_loss", risk["max_market_loss"]),
            _decimal("max_cluster_loss", risk["max_cluster_loss"]),
            _decimal("max_drawdown", risk["max_drawdown"]),
            _decimal("max_total_loss", risk["max_total_loss"]),
            _decimal("max_trailing_additions", risk["max_trailing_additions"]),
            _decimal("max_daily_realized_loss", risk["max_daily_realized_loss"]),
        ),
        net_edge_floor=_decimal("net_edge_floor", raw["net_edge_floor"]),
        maximum_holding_seconds=int(raw["maximum_holding_seconds"]),
        candidate_cooldown_seconds=int(raw["candidate_cooldown_seconds"]),
        material_price_change=_decimal("material_price_change", raw["material_price_change"]),
        cycle_bucket_seconds=int(raw["cycle_bucket_seconds"]),
        intent_ttl_seconds=int(raw["intent_ttl_seconds"]),
        max_research_candidates=int(raw["max_research_candidates"]),
        account_max_age_seconds=int(raw["account_max_age_seconds"]),
        max_new_positions_per_cycle=int(raw["max_new_positions_per_cycle"]),
        max_open_instruments=int(raw["max_open_instruments"]),
    )


def _decimal(name: str, value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite")
    return parsed
