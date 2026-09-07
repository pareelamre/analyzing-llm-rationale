"""Deterministic sizing and calibration with no provider or venue calls."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal, InvalidOperation
from hashlib import sha256
from math import sqrt
from typing import Any, Mapping, Optional, Sequence

from .account import AccountSnapshot
from .models import Completeness, MarketSnapshot, ProposalAction
from .store import AccountProjection, ReservationPreconditions

_ZERO = Decimal("0")
_ONE = Decimal("1")


@dataclass(frozen=True)
class RiskLimits:
    """Caps expressed in the account collateral currency except drawdown."""

    kelly_fraction: Decimal
    max_order_cash: Decimal
    max_market_loss: Decimal
    max_cluster_loss: Decimal
    max_drawdown: Decimal
    max_total_loss: Optional[Decimal] = None
    max_trailing_additions: Optional[Decimal] = None
    max_daily_realized_loss: Optional[Decimal] = None

    def __post_init__(self) -> None:
        kelly = _finite("kelly_fraction", self.kelly_fraction)
        drawdown = _finite("max_drawdown", self.max_drawdown)
        if not _ZERO < kelly <= _ONE or not _ZERO < drawdown <= _ONE:
            raise ValueError("kelly_fraction and max_drawdown must be in (0, 1]")
        object.__setattr__(self, "kelly_fraction", kelly)
        object.__setattr__(self, "max_drawdown", drawdown)
        for name in ("max_order_cash", "max_market_loss", "max_cluster_loss"):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        for name in ("max_total_loss", "max_trailing_additions", "max_daily_realized_loss"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative(name, value))


@dataclass(frozen=True)
class RiskResult:
    quantity: Decimal
    cash: Decimal
    max_loss: Decimal
    reason: Optional[str] = None
    cash_delta: Decimal = _ZERO
    expected_account_revision: Optional[int] = None
    account_generation: Optional[int] = None
    market_snapshot_id: Optional[str] = None
    calibration_hash: Optional[str] = None
    reservation_preconditions: Optional[ReservationPreconditions] = None

    def to_storage(self) -> dict[str, Any]:
        preconditions = self.reservation_preconditions
        return {
            "quantity": str(self.quantity), "cash": str(self.cash), "max_loss": str(self.max_loss),
            "reason": self.reason, "cash_delta": str(self.cash_delta),
            "expected_account_revision": self.expected_account_revision,
            "account_generation": self.account_generation,
            "market_snapshot_id": self.market_snapshot_id,
            "calibration_hash": self.calibration_hash,
            "reservation_preconditions": ({
                "account_revision": preconditions.account_revision,
                "market_version": preconditions.market_version,
                "market_received_at": preconditions.market_received_at.isoformat(),
                "market_stale_after_seconds": preconditions.market_stale_after_seconds,
            } if preconditions is not None else None),
        }

    @classmethod
    def from_storage(cls, payload: Mapping[str, Any]) -> "RiskResult":
        allowed = {
            "quantity", "cash", "max_loss", "reason", "cash_delta",
            "expected_account_revision", "account_generation", "market_snapshot_id",
            "calibration_hash", "reservation_preconditions",
        }
        if set(payload) != allowed:
            raise ValueError("stored risk result schema is invalid")
        precondition_payload = payload.get("reservation_preconditions")
        preconditions = None
        if precondition_payload is not None:
            if not isinstance(precondition_payload, Mapping):
                raise ValueError("stored risk preconditions are invalid")
            preconditions = ReservationPreconditions(
                int(precondition_payload["account_revision"]),
                str(precondition_payload["market_version"]),
                datetime.fromisoformat(str(precondition_payload["market_received_at"]).replace("Z", "+00:00")),
                int(precondition_payload["market_stale_after_seconds"]),
            )
        return cls(
            _finite("quantity", payload["quantity"]), _finite("cash", payload["cash"]),
            _finite("max_loss", payload["max_loss"]), payload.get("reason"),
            _finite("cash_delta", payload["cash_delta"]), payload.get("expected_account_revision"),
            payload.get("account_generation"), payload.get("market_snapshot_id"),
            payload.get("calibration_hash"), preconditions,
        )


@dataclass(frozen=True)
class CalibrationResult:
    probability: Optional[Decimal]
    sample_size: int
    calibration_hash: str
    reason: Optional[str] = None
    mean_probability: Optional[Decimal] = None
    lower_bound: Optional[Decimal] = None
    upper_bound: Optional[Decimal] = None
    training_ids: tuple[str, ...] = ()
    cutoff: Optional[datetime] = None
    calibration_version: str = "calibration_v1"


@dataclass(frozen=True)
class RiskExposure:
    """Worst-case loss already present in inventory, venue orders, or reservations."""

    instrument_id: str
    cluster_id: str
    venue: str
    max_loss: Decimal
    source: str

    def __post_init__(self) -> None:
        if not all(str(getattr(self, name)).strip() for name in ("instrument_id", "cluster_id", "venue")):
            raise ValueError("risk exposure identifiers are required")
        if self.source not in {"inventory", "open_order", "local_reservation"}:
            raise ValueError("risk exposure source is invalid")
        object.__setattr__(self, "max_loss", _nonnegative("max_loss", self.max_loss))


def _finite(name: str, value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _nonnegative(name: str, value: Any) -> Decimal:
    parsed = _finite(name, value)
    if parsed < _ZERO:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _lot_floor(quantity: Decimal, lot: Decimal) -> Decimal:
    """Floor to an arbitrary venue lot, including lots such as 5 or 0.25."""
    return (quantity / lot).to_integral_value(rounding=ROUND_DOWN) * lot


def _cash_ceil(amount: Decimal, increment: Decimal) -> Decimal:
    return (amount / increment).to_integral_value(rounding=ROUND_CEILING) * increment


def _cash_floor(amount: Decimal, increment: Decimal) -> Decimal:
    return (amount / increment).to_integral_value(rounding=ROUND_DOWN) * increment


def _is_tick_aligned(price: Decimal, tick: Decimal) -> bool:
    return price % tick == _ZERO


def _pass(reason: str) -> RiskResult:
    return RiskResult(_ZERO, _ZERO, _ZERO, reason)


def size_binary_entry(
    *, probability: Optional[Decimal], ask: Optional[Decimal], fee_per_share: Optional[Decimal],
    slippage_per_share: Optional[Decimal], available_cash: Optional[Decimal], current_market_loss: Decimal,
    current_cluster_loss: Decimal, drawdown: Decimal, tick_size: Decimal, min_quantity: Decimal,
    limits: RiskLimits, cash_increment: Decimal = Decimal("0.01"), available_depth: Optional[Decimal] = None,
) -> RiskResult:
    """Size one selected-outcome binary buy from a complete, executable quote.

    Cash and worst-case loss are both the fully-costed purchase amount. Existing
    market and cluster exposure therefore bind cumulatively without mixing
    selected-outcome payouts with collateral units.
    """
    if any(value is None for value in (probability, ask, fee_per_share, slippage_per_share, available_cash)):
        return _pass("missing_risk_input")
    try:
        p = _finite("probability", probability)
        price = _finite("ask", ask)
        fee = _nonnegative("fee_per_share", fee_per_share)
        slip = _nonnegative("slippage_per_share", slippage_per_share)
        cash = _nonnegative("available_cash", available_cash)
        market_loss = _nonnegative("current_market_loss", current_market_loss)
        cluster_loss = _nonnegative("current_cluster_loss", current_cluster_loss)
        observed_drawdown = _nonnegative("drawdown", drawdown)
        tick = _finite("tick_size", tick_size)
        lot = _finite("min_quantity", min_quantity)
        currency_increment = _finite("cash_increment", cash_increment)
        depth = None if available_depth is None else _nonnegative("available_depth", available_depth)
    except ValueError:
        return _pass("invalid_risk_input")
    if (
        not _ZERO < p < _ONE or not _ZERO < price < _ONE or tick <= _ZERO or lot <= _ZERO
        or currency_increment <= _ZERO or not _is_tick_aligned(price, tick)
    ):
        return _pass("invalid_market_input")
    if observed_drawdown >= limits.max_drawdown:
        return _pass("drawdown_limit")
    unit_cost = price + fee + slip
    if unit_cost <= _ZERO or unit_cost >= _ONE:
        return _pass("invalid_market_input")
    edge = p - unit_cost
    if edge <= _ZERO:
        return _pass("no_net_edge")
    remaining_market = limits.max_market_loss - market_loss
    remaining_cluster = limits.max_cluster_loss - cluster_loss
    if remaining_market <= _ZERO or remaining_cluster <= _ZERO:
        return _pass("cumulative_loss_limit")
    # Full Kelly for a claim bought for `unit_cost` and paying one at resolution.
    full_kelly = edge / (_ONE - unit_cost)
    target_cash = min(
        cash,
        limits.max_order_cash,
        cash * full_kelly * limits.kelly_fraction,
        remaining_market,
        remaining_cluster,
    )
    final_cap = min(cash, limits.max_order_cash, remaining_market, remaining_cluster)
    rounded_cap = _cash_floor(final_cap, currency_increment)
    quantity = _lot_floor(max(_ZERO, min(target_cash, rounded_cap) / unit_cost), lot)
    if depth is not None:
        quantity = min(quantity, _lot_floor(depth, lot))
    if quantity < lot:
        return _pass("minimum_size_exceeds_cap")
    spent = _cash_ceil(quantity * unit_cost, currency_increment)
    if spent > cash or spent > limits.max_order_cash or spent > remaining_market or spent > remaining_cluster:
        return _pass("final_cash_flow_check_failed")
    return RiskResult(quantity, spent, spent, cash_delta=-spent)


def size_reduce_only(
    *, held_quantity: Decimal, requested_quantity: Decimal, min_quantity: Decimal,
    pending_sell_quantity: Decimal = _ZERO, available_depth: Optional[Decimal] = None,
) -> RiskResult:
    """Clamp a verified close to existing inventory so it cannot flip exposure."""
    try:
        held = _nonnegative("held_quantity", held_quantity)
        requested = _nonnegative("requested_quantity", requested_quantity)
        pending = _nonnegative("pending_sell_quantity", pending_sell_quantity)
        lot = _finite("min_quantity", min_quantity)
        depth = None if available_depth is None else _nonnegative("available_depth", available_depth)
    except ValueError:
        return _pass("invalid_reduce_only_input")
    if lot <= _ZERO:
        return _pass("invalid_reduce_only_input")
    available = max(_ZERO, held - pending)
    quantity = _lot_floor(min(available, requested, depth if depth is not None else requested), lot)
    return RiskResult(quantity, _ZERO, _ZERO, None if quantity >= lot else "insufficient_inventory")


def calibrate_probability(
    raw_probability: Decimal,
    observations: Sequence[Mapping[str, Any]],
    *, as_of: datetime, minimum_bin_samples: int = 30, bin_width: Decimal = Decimal("0.1"),
    model_hash: Optional[str] = None, prompt_hash: Optional[str] = None,
    category_family: Optional[str] = None, outcome_side: str = "YES",
) -> CalibrationResult:
    """Apply auditable ``calibration_v1`` without outcome leakage.

    The method uses ten fixed bins, the earliest prospective forecast per
    instrument, and one stable instrument per resolution cluster.  It returns
    the Laplace-smoothed frequency plus a 95% Wilson interval.  Entry sizing
    uses the lower bound for YES and one minus the upper bound for NO.
    """
    raw = _finite("raw_probability", raw_probability)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    side = str(outcome_side).upper()
    if (
        not _ZERO <= raw <= _ONE or bin_width != Decimal("0.1")
        or minimum_bin_samples < 1 or side not in {"YES", "NO"}
    ):
        raise ValueError("invalid calibration configuration")
    cutoff = as_of.astimezone(timezone.utc)
    bin_index = min(int(raw / bin_width), 9)
    eligible: list[dict[str, Any]] = []
    for row in observations:
        if not isinstance(row, Mapping):
            continue
        try:
            forecast_probability = _finite("forecast_probability", row.get("probability"))
            forecast_at = _parse_timestamp(row.get("forecast_at") or row.get("as_of"))
            resolved_at = _parse_timestamp(row.get("resolved_at"))
            outcome = int(row.get("outcome"))
        except (TypeError, ValueError):
            continue
        instrument_id = str(row.get("instrument_id") or "").strip()
        cluster_id = str(row.get("cluster_id") or "").strip()
        row_id = str(row.get("id") or "").strip()
        if (
            forecast_at is None or resolved_at is None or forecast_at >= resolved_at or resolved_at >= cutoff
            or outcome not in (0, 1) or not instrument_id or not cluster_id or not row_id
            or not _ZERO <= forecast_probability <= _ONE
        ):
            continue
        if model_hash is not None and row.get("model_hash") != model_hash:
            continue
        if prompt_hash is not None and row.get("prompt_hash") != prompt_hash:
            continue
        if category_family is not None and row.get("category_family") != category_family:
            continue
        row_bin = min(int(forecast_probability / bin_width), 9)
        if row_bin != bin_index:
            continue
        eligible.append({
            "id": row_id, "instrument_id": instrument_id, "cluster_id": cluster_id,
            "p": str(forecast_probability), "outcome": outcome,
            "forecast_at": forecast_at.isoformat(), "resolved_at": resolved_at.isoformat(),
        })

    earliest_by_instrument: dict[str, dict[str, Any]] = {}
    for row in sorted(eligible, key=lambda item: (item["forecast_at"], item["id"])):
        earliest_by_instrument.setdefault(row["instrument_id"], row)
    selected_by_cluster: dict[str, dict[str, Any]] = {}
    for row in sorted(earliest_by_instrument.values(), key=lambda item: (item["instrument_id"], item["id"])):
        selected_by_cluster.setdefault(row["cluster_id"], row)
    canonical = sorted(selected_by_cluster.values(), key=lambda item: item["id"])
    training_ids = tuple(row["id"] for row in canonical)
    hash_payload = {
        "version": "calibration_v1", "cutoff": cutoff.isoformat(), "bin_width": str(bin_width),
        "bin_index": bin_index, "minimum_bin_samples": minimum_bin_samples,
        "model_hash": model_hash, "prompt_hash": prompt_hash, "category_family": category_family,
        "training_rows": canonical,
    }
    digest = sha256(json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    count = len(canonical)
    if count < minimum_bin_samples:
        return CalibrationResult(
            None, count, digest, "insufficient_calibration_sample",
            training_ids=training_ids, cutoff=cutoff,
        )
    successes = sum(row["outcome"] for row in canonical)
    mean = successes / count
    laplace = Decimal(successes + 1) / Decimal(count + 2)
    z = 1.959963984540054  # two-sided 95% interval
    denominator = 1 + z * z / count
    center = (mean + z * z / (2 * count)) / denominator
    radius = z * sqrt((mean * (1 - mean) + z * z / (4 * count)) / count) / denominator
    lower = Decimal(str(max(0.0, min(1.0, center - radius))))
    upper = Decimal(str(max(0.0, min(1.0, center + radius))))
    conservative = lower if side == "YES" else _ONE - upper
    return CalibrationResult(
        conservative, count, digest, mean_probability=laplace, lower_bound=lower,
        upper_bound=upper, training_ids=training_ids, cutoff=cutoff,
    )


def evaluate_binary_candidate(
    *, action: ProposalAction | str, instrument_id: str, cluster_id: str, venue: str,
    market_snapshot: MarketSnapshot, account_snapshot: AccountSnapshot,
    account_projection: AccountProjection, exposures: Optional[Sequence[RiskExposure]],
    limits: RiskLimits, now: datetime, calibration: Optional[CalibrationResult] = None,
    fee_per_share: Optional[Decimal] = None, slippage_per_share: Optional[Decimal] = None,
    fee_version: Optional[str] = None,
    available_depth: Optional[Decimal] = None, cash_increment: Decimal = Decimal("0.01"),
    tick_size: Optional[Decimal] = None, min_quantity: Optional[Decimal] = None,
    trailing_additions: Optional[Decimal] = None, realized_losses: Optional[Decimal] = None,
    peak_equity: Optional[Decimal] = None, current_equity: Optional[Decimal] = None,
    requested_quantity: Optional[Decimal] = None, held_quantity: Optional[Decimal] = None,
    pending_sell_quantity: Decimal = _ZERO, account_max_age_seconds: int = 60,
    portfolio_complete: bool = True,
) -> RiskResult:
    """Evaluate an entry or verified close from one immutable state version.

    ``exposures`` must include inventory, venue open orders, and unmatched local
    reservations.  The returned account revision must be supplied to
    ``reserve_intent`` so concurrent changes cause a validation restart.
    """
    try:
        selected_action = ProposalAction(action)
    except ValueError:
        return _pass("invalid_action")
    if now.tzinfo is None:
        return _pass("invalid_evaluation_time")
    current_time = now.astimezone(timezone.utc)
    if account_snapshot.scope_id != account_projection.scope_id:
        return _pass("account_scope_mismatch")
    if account_snapshot.completeness is not Completeness.COMPLETE or account_snapshot.blocks_new_exposure:
        return _pass("account_reconciliation_blocked")
    if not portfolio_complete or exposures is None:
        return _pass("incomplete_portfolio_state")
    if (
        account_snapshot.received_at.tzinfo is None
        or account_snapshot.received_at > current_time
        or (current_time - account_snapshot.received_at).total_seconds() > account_max_age_seconds
        or account_projection.venue_available_cash != account_snapshot.available_cash
    ):
        return _pass("stale_account_snapshot")
    if (
        market_snapshot.instrument_id != instrument_id
        or market_snapshot.complete is not Completeness.COMPLETE
        or market_snapshot.received_at > current_time
        or (current_time - market_snapshot.received_at).total_seconds() > market_snapshot.stale_after_seconds
    ):
        return _pass("stale_market_snapshot")
    if not fee_version or fee_version != market_snapshot.fee_version:
        return _pass("stale_fee_schedule")
    try:
        normalized_exposures = tuple(exposures)
        if any(not isinstance(item, RiskExposure) for item in normalized_exposures):
            return _pass("incomplete_portfolio_state")
        local_reserved = sum(
            (item.max_loss for item in normalized_exposures if item.venue == venue and item.source == "local_reservation"),
            _ZERO,
        )
        if local_reserved != account_projection.reserved_max_loss:
            return _pass("incomplete_portfolio_state")
        additions = _nonnegative("trailing_additions", trailing_additions)
        losses = _nonnegative("realized_losses", realized_losses)
        peak = _nonnegative("peak_equity", peak_equity)
        equity = _nonnegative("current_equity", current_equity)
        currency_increment = _finite("cash_increment", cash_increment)
    except (TypeError, ValueError):
        return _pass("missing_or_invalid_portfolio_metric")
    if currency_increment <= _ZERO:
        return _pass("missing_or_invalid_portfolio_metric")
    if peak <= _ZERO:
        return _pass("zero_bankroll")
    drawdown = max(_ZERO, (peak - equity) / peak)
    if drawdown >= limits.max_drawdown:
        return _pass("drawdown_limit")
    if limits.max_daily_realized_loss is not None and losses >= limits.max_daily_realized_loss:
        return _pass("realized_loss_limit")

    is_buy = selected_action in {ProposalAction.BUY_YES, ProposalAction.BUY_NO}
    if not is_buy:
        if selected_action not in {ProposalAction.SELL_YES, ProposalAction.SELL_NO}:
            return _pass("invalid_action")
        if any(value is None for value in (
            requested_quantity, held_quantity, available_depth, fee_per_share,
            slippage_per_share, min_quantity,
        )):
            return _pass("missing_reduce_only_input")
        result = size_reduce_only(
            held_quantity=held_quantity, requested_quantity=requested_quantity,
            pending_sell_quantity=pending_sell_quantity, available_depth=available_depth,
            min_quantity=min_quantity,
        )
        if result.reason is not None:
            return result
        bid = market_snapshot.yes_bid if selected_action is ProposalAction.SELL_YES else market_snapshot.no_bid
        if bid is None:
            return _pass("missing_executable_price")
        try:
            tick = _finite("tick_size", tick_size)
            unit_proceeds = _finite("bid", bid) - _nonnegative("fee_per_share", fee_per_share) - _nonnegative("slippage_per_share", slippage_per_share)
        except ValueError:
            return _pass("invalid_risk_input")
        if tick <= _ZERO or not _is_tick_aligned(bid, tick):
            return _pass("invalid_market_input")
        if unit_proceeds <= _ZERO:
            return _pass("nonpositive_close_proceeds")
        proceeds = _cash_floor(result.quantity * unit_proceeds, currency_increment)
        return replace(
            result, cash_delta=proceeds, expected_account_revision=account_projection.revision,
            account_generation=account_snapshot.generation,
            market_snapshot_id=market_snapshot.id,
            reservation_preconditions=ReservationPreconditions(
                account_projection.revision, market_snapshot.id,
                market_snapshot.received_at, market_snapshot.stale_after_seconds,
            ),
        )

    if calibration is None or calibration.probability is None or calibration.lower_bound is None or calibration.upper_bound is None:
        return _pass("missing_calibration")
    if any(value is None for value in (fee_per_share, slippage_per_share, available_depth, tick_size, min_quantity)):
        return _pass("missing_cost_or_depth")
    probability = calibration.lower_bound if selected_action is ProposalAction.BUY_YES else _ONE - calibration.upper_bound
    ask = market_snapshot.yes_ask if selected_action is ProposalAction.BUY_YES else market_snapshot.no_ask
    if ask is None:
        return _pass("missing_executable_price")
    market_loss = sum((item.max_loss for item in normalized_exposures if item.instrument_id == instrument_id), _ZERO)
    cluster_loss = sum((item.max_loss for item in normalized_exposures if item.cluster_id == cluster_id), _ZERO)
    total_loss = sum((item.max_loss for item in normalized_exposures), _ZERO)
    remaining_order = limits.max_order_cash
    if limits.max_total_loss is not None:
        remaining_total = limits.max_total_loss - total_loss
        if remaining_total <= _ZERO:
            return _pass("total_loss_limit")
        remaining_order = min(remaining_order, remaining_total)
    if limits.max_trailing_additions is not None:
        remaining_additions = limits.max_trailing_additions - additions
        if remaining_additions <= _ZERO:
            return _pass("trailing_additions_limit")
        remaining_order = min(remaining_order, remaining_additions)
    bounded_limits = replace(limits, max_order_cash=remaining_order)
    result = size_binary_entry(
        probability=probability, ask=ask, fee_per_share=fee_per_share,
        slippage_per_share=slippage_per_share,
        available_cash=account_projection.available_cash_for_reservation,
        current_market_loss=market_loss, current_cluster_loss=cluster_loss,
        drawdown=drawdown, tick_size=tick_size, min_quantity=min_quantity,
        limits=bounded_limits, cash_increment=cash_increment, available_depth=available_depth,
    )
    return replace(
        result, expected_account_revision=account_projection.revision,
        account_generation=account_snapshot.generation,
        market_snapshot_id=market_snapshot.id, calibration_hash=calibration.calibration_hash,
        reservation_preconditions=ReservationPreconditions(
            account_projection.revision, market_snapshot.id,
            market_snapshot.received_at, market_snapshot.stale_after_seconds,
        ),
    )


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def sorted_candidate_ids(candidates: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Stable prioritization avoids accidental ordering changes between workers."""
    valid = [row for row in candidates if isinstance(row, Mapping) and str(row.get("id") or "").strip()]
    return tuple(str(row["id"]) for row in sorted(valid, key=lambda row: (-_safe_decimal(row.get("net_edge")), str(row["id"]))))


def _safe_decimal(value: Any) -> Decimal:
    try:
        return _finite("net_edge", value)
    except ValueError:
        return _ZERO
