"""Strict durable contracts for the autonomous trading twin.

These records deliberately contain no credential fields and no boolean capable
of bypassing execution authority.  They are plain Python models so workers,
stores, simulation and HTTP routes can share one schema without importing the
FastAPI application.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256
from typing import Any, Iterable, Mapping, Optional

SCHEMA_VERSION = 1
MAX_DECIMAL_PLACES = 8
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class SchemaValidationError(ValueError):
    """A persisted twin record is incomplete, ambiguous, or unsafe to use."""


class Completeness(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    MALFORMED = "malformed"
    STALE = "stale"


class ProposalAction(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"
    PASS = "PASS"


class RejectionReason(str, Enum):
    PASS_INVALID_PROPOSAL = "PASS_INVALID_PROPOSAL"
    PASS_STALE_DATA = "PASS_STALE_DATA"
    PASS_INCOMPLETE_DATA = "PASS_INCOMPLETE_DATA"
    PASS_UNSUPPORTED_INSTRUMENT = "PASS_UNSUPPORTED_INSTRUMENT"
    PASS_BELOW_MINIMUM = "PASS_BELOW_MINIMUM"
    PASS_NO_EDGE = "PASS_NO_EDGE"
    PASS_RISK_LIMIT = "PASS_RISK_LIMIT"
    PASS_AUTHORITY_MISSING = "PASS_AUTHORITY_MISSING"
    PASS_MODEL_UNAVAILABLE = "PASS_MODEL_UNAVAILABLE"


class CommandState(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    RESERVED = "reserved"
    SUBMITTING = "submitting"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    BLOCKED = "blocked"
    EXPIRED = "expired"
    REJECTED = "rejected"
    SUBMISSION_UNKNOWN = "submission_unknown"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


_COMMAND_TRANSITIONS: Mapping[CommandState, frozenset[CommandState]] = {
    CommandState.PROPOSED: frozenset({CommandState.VALIDATED, CommandState.BLOCKED, CommandState.EXPIRED}),
    CommandState.VALIDATED: frozenset({CommandState.RESERVED, CommandState.BLOCKED, CommandState.EXPIRED}),
    CommandState.RESERVED: frozenset({CommandState.SUBMITTING, CommandState.BLOCKED, CommandState.EXPIRED}),
    CommandState.SUBMITTING: frozenset(
        {CommandState.ACKNOWLEDGED, CommandState.REJECTED, CommandState.SUBMISSION_UNKNOWN}
    ),
    CommandState.ACKNOWLEDGED: frozenset(
        {CommandState.PARTIALLY_FILLED, CommandState.FILLED, CommandState.CANCEL_REQUESTED}
    ),
    CommandState.PARTIALLY_FILLED: frozenset(
        {CommandState.PARTIALLY_FILLED, CommandState.FILLED, CommandState.CANCEL_REQUESTED}
    ),
    CommandState.CANCEL_REQUESTED: frozenset(
        {CommandState.PARTIALLY_FILLED, CommandState.FILLED, CommandState.CANCELLED}
    ),
    CommandState.SUBMISSION_UNKNOWN: frozenset(
        {CommandState.ACKNOWLEDGED, CommandState.PARTIALLY_FILLED, CommandState.FILLED, CommandState.REJECTED}
    ),
    CommandState.FILLED: frozenset(),
    CommandState.BLOCKED: frozenset(),
    CommandState.EXPIRED: frozenset(),
    CommandState.REJECTED: frozenset(),
    CommandState.CANCELLED: frozenset(),
}


def can_transition_command(current: CommandState, target: CommandState) -> bool:
    """Return whether a durable command can advance without terminal regression."""
    return target in _COMMAND_TRANSITIONS[current]


def _required_identifier(name: str, value: str) -> str:
    cleaned = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(cleaned):
        raise SchemaValidationError(f"{name} must be a stable non-display identifier")
    return cleaned


def _required_text(name: str, value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise SchemaValidationError(f"{name} is required")
    return cleaned


def _decimal(name: str, value: Any, *, minimum: Decimal = Decimal("0"), maximum: Optional[Decimal] = None) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or parsed < minimum or (maximum is not None and parsed > maximum):
        raise SchemaValidationError(f"{name} is outside its supported range")
    if -parsed.as_tuple().exponent > MAX_DECIMAL_PLACES:
        raise SchemaValidationError(f"{name} exceeds {MAX_DECIMAL_PLACES} decimal places")
    return parsed


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")


def _timestamp(
    name: str, value: datetime | str, *, now: Optional[datetime] = None, allow_future: bool = False
) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError(f"{name} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SchemaValidationError(f"{name} must be timezone-aware")
    normalised = value.astimezone(timezone.utc)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not allow_future and normalised > current:
        raise SchemaValidationError(f"{name} cannot be in the future")
    return normalised


def _timestamp_string(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _enum(enum_type: type[Enum], name: str, value: Any) -> Enum:
    try:
        return enum_type(value)
    except ValueError as exc:
        raise SchemaValidationError(f"{name} is unsupported") from exc


def _require_keys(payload: Mapping[str, Any], allowed: Iterable[str], *, record_type: str) -> None:
    unexpected = set(payload).difference(allowed)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise SchemaValidationError(f"{record_type} contains unsupported fields: {names}")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _storage_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return _decimal_string(value)
    if isinstance(value, datetime):
        return _timestamp_string(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_storage_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _storage_value(item) for key, item in value.items()}
    return value


def _dataclass_storage(record: Any) -> dict[str, Any]:
    return {name: _storage_value(value) for name, value in record.__dict__.items()}


def canonical_instrument_id(
    *, venue: str, environment: str, venue_instrument_id: str, condition_id: Optional[str] = None
) -> str:
    """Build an execution identity from venue IDs, never a title or slug."""
    venue_id = _required_identifier("venue", venue).lower()
    environment_id = _required_identifier("environment", environment).lower()
    instrument_id = _required_identifier("venue_instrument_id", venue_instrument_id)
    parts = [venue_id, environment_id]
    if condition_id is not None:
        parts.append(_required_identifier("condition_id", condition_id))
    parts.append(instrument_id)
    return ":".join(parts)


@dataclass(frozen=True)
class AccountScope:
    id: str
    owner_id: str
    venue: str
    venue_account_ref: str
    environment: str
    collateral_asset: str
    connection_ref: str
    account_epoch: int
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError("unsupported AccountScope schema version")
        for name in ("id", "owner_id", "venue", "venue_account_ref", "environment", "collateral_asset", "connection_ref"):
            object.__setattr__(self, name, _required_identifier(name, getattr(self, name)))
        if not isinstance(self.account_epoch, int) or self.account_epoch < 1:
            raise SchemaValidationError("account_epoch must be a positive integer")
        object.__setattr__(self, "created_at", _timestamp("created_at", self.created_at))

    def to_storage(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "owner_id": self.owner_id,
            "venue": self.venue,
            "venue_account_ref": self.venue_account_ref,
            "environment": self.environment,
            "collateral_asset": self.collateral_asset,
            "connection_ref": self.connection_ref,
            "account_epoch": self.account_epoch,
            "created_at": _timestamp_string(self.created_at),
        }


@dataclass(frozen=True)
class Instrument:
    id: str
    venue: str
    environment: str
    venue_instrument_id: str
    condition_id: Optional[str]
    yes_token_id: Optional[str]
    no_token_id: Optional[str]
    settlement_spec_hash: str
    category: str
    event_id: str
    cluster_id: str
    tick_size: Decimal
    min_quantity: Decimal
    fee_version: str
    capability_version: str
    status: str
    close_at: datetime
    resolution_at: datetime
    created_at: datetime
    schema_version: int = SCHEMA_VERSION
    display_title: Optional[str] = field(default=None, compare=False)
    display_slug: Optional[str] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError("unsupported Instrument schema version")
        for name in (
            "venue", "environment", "venue_instrument_id", "settlement_spec_hash", "category", "event_id",
            "cluster_id", "fee_version", "capability_version", "status",
        ):
            object.__setattr__(self, name, _required_identifier(name, getattr(self, name)))
        if self.condition_id is not None:
            object.__setattr__(self, "condition_id", _required_identifier("condition_id", self.condition_id))
        for name in ("yes_token_id", "no_token_id"):
            token = getattr(self, name)
            if token is not None:
                object.__setattr__(self, name, _required_identifier(name, token))
        canonical = canonical_instrument_id(
            venue=self.venue,
            environment=self.environment,
            venue_instrument_id=self.venue_instrument_id,
            condition_id=self.condition_id,
        )
        if self.id != canonical:
            raise SchemaValidationError("Instrument id must be derived from venue identifiers and environment")
        object.__setattr__(self, "tick_size", _decimal("tick_size", self.tick_size, minimum=Decimal("0.00000001")))
        object.__setattr__(self, "min_quantity", _decimal("min_quantity", self.min_quantity, minimum=Decimal("0.00000001")))
        object.__setattr__(self, "created_at", _timestamp("created_at", self.created_at))
        object.__setattr__(self, "close_at", _timestamp("close_at", self.close_at, allow_future=True))
        object.__setattr__(self, "resolution_at", _timestamp("resolution_at", self.resolution_at, allow_future=True))
        if self.resolution_at < self.close_at:
            raise SchemaValidationError("resolution_at cannot precede close_at")

    def to_storage(self) -> dict[str, Any]:
        return _dataclass_storage(self)


@dataclass(frozen=True)
class MarketCursor:
    source: str
    cursor: Optional[str]
    completeness: Completeness
    observed_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError("unsupported MarketCursor schema version")
        object.__setattr__(self, "source", _required_identifier("source", self.source))
        object.__setattr__(self, "completeness", _enum(Completeness, "completeness", self.completeness))
        object.__setattr__(self, "observed_at", _timestamp("observed_at", self.observed_at))

    def to_storage(self) -> dict[str, Any]:
        return _dataclass_storage(self)


@dataclass(frozen=True)
class MarketSnapshot:
    id: str
    instrument_id: str
    venue_at: datetime
    received_at: datetime
    sequence: int
    source: str
    complete: Completeness
    stale_after_seconds: int
    yes_bid: Optional[Decimal]
    yes_ask: Optional[Decimal]
    no_bid: Optional[Decimal]
    no_ask: Optional[Decimal]
    fee_version: str
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError("unsupported MarketSnapshot schema version")
        for name in ("id", "instrument_id", "source", "fee_version"):
            object.__setattr__(self, name, _required_identifier(name, getattr(self, name)))
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise SchemaValidationError("sequence must be non-negative")
        if not isinstance(self.stale_after_seconds, int) or self.stale_after_seconds <= 0:
            raise SchemaValidationError("stale_after_seconds must be positive")
        object.__setattr__(self, "complete", _enum(Completeness, "complete", self.complete))
        object.__setattr__(self, "venue_at", _timestamp("venue_at", self.venue_at))
        object.__setattr__(self, "received_at", _timestamp("received_at", self.received_at))
        object.__setattr__(self, "created_at", _timestamp("created_at", self.created_at))
        if self.received_at < self.venue_at:
            raise SchemaValidationError("received_at cannot precede venue_at")
        for name in ("yes_bid", "yes_ask", "no_bid", "no_ask"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _decimal(name, value, maximum=Decimal("1")))
        for bid, ask, outcome in ((self.yes_bid, self.yes_ask, "yes"), (self.no_bid, self.no_ask, "no")):
            if bid is not None and ask is not None and bid > ask:
                raise SchemaValidationError(f"{outcome} bid cannot exceed ask")

    def to_storage(self) -> dict[str, Any]:
        return _dataclass_storage(self)


@dataclass(frozen=True)
class Forecast:
    id: str
    instrument_id: str
    p_yes_raw: Decimal
    p_yes_calibrated: Optional[Decimal]
    calibration_status: str
    uncertainty_low: Optional[Decimal]
    uncertainty_high: Optional[Decimal]
    evidence_ids: tuple[str, ...]
    as_of: datetime
    expires_at: datetime
    model_hash: str
    prompt_hash: str
    strategy_hash: str
    calibration_version: str
    prospective_provenance: str
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError("unsupported Forecast schema version")
        for name in ("id", "instrument_id", "calibration_status", "model_hash", "prompt_hash", "strategy_hash", "calibration_version", "prospective_provenance"):
            object.__setattr__(self, name, _required_identifier(name, getattr(self, name)))
        object.__setattr__(self, "p_yes_raw", _decimal("p_yes_raw", self.p_yes_raw, maximum=Decimal("1")))
        if self.p_yes_calibrated is not None:
            object.__setattr__(self, "p_yes_calibrated", _decimal("p_yes_calibrated", self.p_yes_calibrated, maximum=Decimal("1")))
        if (self.uncertainty_low is None) != (self.uncertainty_high is None):
            raise SchemaValidationError("uncertainty bounds must be present together or both unknown")
        if self.uncertainty_low is not None:
            low = _decimal("uncertainty_low", self.uncertainty_low, maximum=Decimal("1"))
            high = _decimal("uncertainty_high", self.uncertainty_high, maximum=Decimal("1"))
            if low > high:
                raise SchemaValidationError("uncertainty_low cannot exceed uncertainty_high")
            object.__setattr__(self, "uncertainty_low", low)
            object.__setattr__(self, "uncertainty_high", high)
        evidence = tuple(_required_identifier("evidence_id", value) for value in self.evidence_ids)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "as_of", _timestamp("as_of", self.as_of))
        object.__setattr__(self, "expires_at", _timestamp("expires_at", self.expires_at, allow_future=True))
        object.__setattr__(self, "created_at", _timestamp("created_at", self.created_at))
        if self.expires_at <= self.as_of:
            raise SchemaValidationError("expires_at must follow as_of")

    def to_storage(self) -> dict[str, Any]:
        return _dataclass_storage(self)


@dataclass(frozen=True)
class PassDecision:
    reason: RejectionReason
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _enum(RejectionReason, "reason", self.reason))
        object.__setattr__(self, "detail", _required_text("detail", self.detail))

    def to_storage(self) -> dict[str, str]:
        return {"reason": self.reason.value, "detail": self.detail}


@dataclass(frozen=True)
class Proposal:
    id: str
    forecast_id: str
    market_snapshot_id: str
    action: ProposalAction
    reason_codes: tuple[RejectionReason, ...]
    citation_ids: tuple[str, ...]
    preferred_limit: Optional[Decimal]
    pass_decision: Optional[PassDecision]
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError("unsupported Proposal schema version")
        for name in ("id", "forecast_id", "market_snapshot_id"):
            object.__setattr__(self, name, _required_identifier(name, getattr(self, name)))
        action = _enum(ProposalAction, "action", self.action)
        object.__setattr__(self, "action", action)
        reasons = tuple(_enum(RejectionReason, "reason_code", item) for item in self.reason_codes)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "citation_ids", tuple(_required_identifier("citation_id", item) for item in self.citation_ids))
        if self.preferred_limit is not None:
            object.__setattr__(self, "preferred_limit", _decimal("preferred_limit", self.preferred_limit, maximum=Decimal("1")))
        if action is ProposalAction.PASS and self.pass_decision is None:
            raise SchemaValidationError("PASS proposals require a structured pass_decision")
        if action is not ProposalAction.PASS and self.pass_decision is not None:
            raise SchemaValidationError("only PASS proposals may carry pass_decision")
        object.__setattr__(self, "created_at", _timestamp("created_at", self.created_at))

    def to_storage(self) -> dict[str, Any]:
        data = _dataclass_storage(self)
        data["pass_decision"] = self.pass_decision.to_storage() if self.pass_decision is not None else None
        return data

    @classmethod
    def from_storage(cls, payload: Mapping[str, Any]) -> "Proposal":
        allowed = {
            "schema_version", "id", "forecast_id", "market_snapshot_id", "action", "reason_codes", "citation_ids",
            "preferred_limit", "pass_decision", "created_at",
        }
        _require_keys(payload, allowed, record_type="Proposal")
        values = dict(payload)
        pass_data = values.get("pass_decision")
        if pass_data is not None:
            if not isinstance(pass_data, Mapping):
                raise SchemaValidationError("Proposal pass_decision must be an object")
            _require_keys(pass_data, {"reason", "detail"}, record_type="PassDecision")
            values["pass_decision"] = PassDecision(**dict(pass_data))
        return cls(**values)


@dataclass(frozen=True)
class TradeIntent:
    """A deterministic execution candidate with no authority or secret payload."""

    id: str
    account_scope_id: str
    account_epoch: int
    instrument_id: str
    action: ProposalAction
    quantity: Decimal
    limit_price: Decimal
    time_in_force: str
    forecast_id: Optional[str]
    exit_reason: Optional[str]
    policy_version: str
    strategy_version: str
    market_version: str
    fee_allowance: Decimal
    slippage_allowance: Decimal
    expires_at: datetime
    created_at: datetime
    schema_version: int = SCHEMA_VERSION
    presentation_text: Optional[str] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError("unsupported TradeIntent schema version")
        for name in ("id", "account_scope_id", "instrument_id", "time_in_force", "policy_version", "strategy_version", "market_version"):
            object.__setattr__(self, name, _required_identifier(name, getattr(self, name)))
        if not isinstance(self.account_epoch, int) or self.account_epoch < 1:
            raise SchemaValidationError("account_epoch must be a positive integer")
        action = _enum(ProposalAction, "action", self.action)
        if action not in {ProposalAction.BUY_YES, ProposalAction.BUY_NO}:
            raise SchemaValidationError("TradeIntent action must be BUY_YES or BUY_NO")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "quantity", _decimal("quantity", self.quantity, minimum=Decimal("0.00000001")))
        object.__setattr__(self, "limit_price", _decimal("limit_price", self.limit_price, maximum=Decimal("1")))
        object.__setattr__(self, "fee_allowance", _decimal("fee_allowance", self.fee_allowance))
        object.__setattr__(self, "slippage_allowance", _decimal("slippage_allowance", self.slippage_allowance))
        if (self.forecast_id is None) == (self.exit_reason is None):
            raise SchemaValidationError("TradeIntent requires exactly one forecast_id or deterministic exit_reason")
        if self.forecast_id is not None:
            object.__setattr__(self, "forecast_id", _required_identifier("forecast_id", self.forecast_id))
        if self.exit_reason is not None:
            object.__setattr__(self, "exit_reason", _required_identifier("exit_reason", self.exit_reason))
        object.__setattr__(self, "created_at", _timestamp("created_at", self.created_at))
        object.__setattr__(self, "expires_at", _timestamp("expires_at", self.expires_at, allow_future=True))
        if self.expires_at <= self.created_at:
            raise SchemaValidationError("expires_at must follow created_at")

    def identity_payload(self) -> dict[str, Any]:
        """Fields that can affect execution or authorization, canonically sorted."""
        return {
            "schema_version": self.schema_version,
            "account_scope_id": self.account_scope_id,
            "account_epoch": self.account_epoch,
            "instrument_id": self.instrument_id,
            "action": self.action.value,
            "quantity": _decimal_string(self.quantity),
            "limit_price": _decimal_string(self.limit_price),
            "time_in_force": self.time_in_force,
            "forecast_id": self.forecast_id,
            "exit_reason": self.exit_reason,
            "policy_version": self.policy_version,
            "strategy_version": self.strategy_version,
            "market_version": self.market_version,
            "fee_allowance": _decimal_string(self.fee_allowance),
            "slippage_allowance": _decimal_string(self.slippage_allowance),
            "expires_at": _timestamp_string(self.expires_at),
        }

    @property
    def intent_hash(self) -> str:
        return sha256(_canonical_json(self.identity_payload()).encode("utf-8")).hexdigest()

    def assert_authorization_context(
        self,
        scope: AccountScope,
        *,
        policy_version: str,
        strategy_version: str,
        market_version: str,
    ) -> None:
        """Verify account and configuration bindings before later authorization."""
        if self.account_scope_id != scope.id or self.account_epoch != scope.account_epoch:
            raise SchemaValidationError("TradeIntent account binding does not match the active account scope")
        for name, actual, expected in (
            ("policy_version", self.policy_version, policy_version),
            ("strategy_version", self.strategy_version, strategy_version),
            ("market_version", self.market_version, market_version),
        ):
            if actual != _required_identifier(name, expected):
                raise SchemaValidationError(f"TradeIntent {name} does not match current authorization context")

    def to_storage(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "account_scope_id": self.account_scope_id,
            "account_epoch": self.account_epoch,
            "instrument_id": self.instrument_id,
            "action": self.action.value,
            "quantity": _decimal_string(self.quantity),
            "limit_price": _decimal_string(self.limit_price),
            "time_in_force": self.time_in_force,
            "forecast_id": self.forecast_id,
            "exit_reason": self.exit_reason,
            "policy_version": self.policy_version,
            "strategy_version": self.strategy_version,
            "market_version": self.market_version,
            "fee_allowance": _decimal_string(self.fee_allowance),
            "slippage_allowance": _decimal_string(self.slippage_allowance),
            "expires_at": _timestamp_string(self.expires_at),
            "created_at": _timestamp_string(self.created_at),
            "presentation_text": self.presentation_text,
            "intent_hash": self.intent_hash,
        }

    @classmethod
    def from_storage(cls, payload: Mapping[str, Any]) -> "TradeIntent":
        allowed = {
            "schema_version", "id", "account_scope_id", "account_epoch", "instrument_id", "action", "quantity",
            "limit_price", "time_in_force", "forecast_id", "exit_reason", "policy_version", "strategy_version",
            "market_version", "fee_allowance", "slippage_allowance", "expires_at", "created_at", "presentation_text",
            "intent_hash",
        }
        _require_keys(payload, allowed, record_type="TradeIntent")
        expected_hash = payload.get("intent_hash")
        values = {key: value for key, value in payload.items() if key != "intent_hash"}
        intent = cls(**values)
        if expected_hash is not None and expected_hash != intent.intent_hash:
            raise SchemaValidationError("TradeIntent intent_hash does not match immutable fields")
        return intent


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    account_scope_id: str
    candidate_ids: tuple[str, ...]
    action: ProposalAction
    pass_decision: Optional[PassDecision]
    policy_version: str
    strategy_version: str
    resulting_intent_id: Optional[str]
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError("unsupported DecisionRecord schema version")
        for name in ("id", "account_scope_id", "policy_version", "strategy_version"):
            object.__setattr__(self, name, _required_identifier(name, getattr(self, name)))
        object.__setattr__(self, "candidate_ids", tuple(_required_identifier("candidate_id", item) for item in self.candidate_ids))
        object.__setattr__(self, "action", _enum(ProposalAction, "action", self.action))
        if self.action is ProposalAction.PASS and self.pass_decision is None:
            raise SchemaValidationError("PASS DecisionRecord requires pass_decision")
        if self.action is not ProposalAction.PASS and self.pass_decision is not None:
            raise SchemaValidationError("only PASS DecisionRecord may include pass_decision")
        if self.resulting_intent_id is not None:
            object.__setattr__(self, "resulting_intent_id", _required_identifier("resulting_intent_id", self.resulting_intent_id))
        object.__setattr__(self, "created_at", _timestamp("created_at", self.created_at))

    def to_storage(self) -> dict[str, Any]:
        data = _dataclass_storage(self)
        data["pass_decision"] = self.pass_decision.to_storage() if self.pass_decision is not None else None
        return data
