"""Deterministic sizing and calibration with no provider or venue calls."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from hashlib import sha256
from math import sqrt
from typing import Any, Mapping, Optional, Sequence

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

    def __post_init__(self) -> None:
        kelly = _finite("kelly_fraction", self.kelly_fraction)
        drawdown = _finite("max_drawdown", self.max_drawdown)
        if not _ZERO < kelly <= _ONE or not _ZERO < drawdown <= _ONE:
            raise ValueError("kelly_fraction and max_drawdown must be in (0, 1]")
        object.__setattr__(self, "kelly_fraction", kelly)
        object.__setattr__(self, "max_drawdown", drawdown)
        for name in ("max_order_cash", "max_market_loss", "max_cluster_loss"):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))


@dataclass(frozen=True)
class RiskResult:
    quantity: Decimal
    cash: Decimal
    max_loss: Decimal
    reason: Optional[str] = None


@dataclass(frozen=True)
class CalibrationResult:
    probability: Optional[Decimal]
    sample_size: int
    calibration_hash: str
    reason: Optional[str] = None


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


def _is_tick_aligned(price: Decimal, tick: Decimal) -> bool:
    return price % tick == _ZERO


def _pass(reason: str) -> RiskResult:
    return RiskResult(_ZERO, _ZERO, _ZERO, reason)


def size_binary_entry(
    *, probability: Optional[Decimal], ask: Optional[Decimal], fee_per_share: Optional[Decimal],
    slippage_per_share: Optional[Decimal], available_cash: Optional[Decimal], current_market_loss: Decimal,
    current_cluster_loss: Decimal, drawdown: Decimal, tick_size: Decimal, min_quantity: Decimal,
    limits: RiskLimits,
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
    except ValueError:
        return _pass("invalid_risk_input")
    if not _ZERO < p < _ONE or not _ZERO < price < _ONE or tick <= _ZERO or lot <= _ZERO or not _is_tick_aligned(price, tick):
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
    quantity = _lot_floor(max(_ZERO, target_cash / unit_cost), lot)
    if quantity < lot:
        return _pass("minimum_size_exceeds_cap")
    spent = quantity * unit_cost
    if spent > cash or spent > limits.max_order_cash or spent > remaining_market or spent > remaining_cluster:
        return _pass("final_cash_flow_check_failed")
    return RiskResult(quantity, spent, spent)


def size_reduce_only(*, held_quantity: Decimal, requested_quantity: Decimal, min_quantity: Decimal) -> RiskResult:
    """Clamp a verified close to existing inventory so it cannot flip exposure."""
    try:
        held = _nonnegative("held_quantity", held_quantity)
        requested = _nonnegative("requested_quantity", requested_quantity)
        lot = _finite("min_quantity", min_quantity)
    except ValueError:
        return _pass("invalid_reduce_only_input")
    if lot <= _ZERO:
        return _pass("invalid_reduce_only_input")
    quantity = _lot_floor(min(held, requested), lot)
    return RiskResult(quantity, _ZERO, _ZERO, None if quantity >= lot else "insufficient_inventory")


def calibrate_probability(
    raw_probability: Decimal,
    observations: Sequence[Mapping[str, Any]],
    *, as_of: datetime, minimum_bin_samples: int = 20, bin_width: Decimal = Decimal("0.05"),
) -> CalibrationResult:
    """Use only resolved, pre-cutoff observations in a fixed probability bin.

    The returned probability is a one-sided 90% Wilson lower bound. It is
    intentionally conservative for entry sizing and remains deterministic for a
    frozen observation set.
    """
    raw = _finite("raw_probability", raw_probability)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if not _ZERO <= raw <= _ONE or not _ZERO < bin_width <= _ONE or minimum_bin_samples < 1:
        raise ValueError("invalid calibration configuration")
    cutoff = as_of.astimezone(timezone.utc)
    bin_index = min(int(raw / bin_width), int(_ONE / bin_width))
    selected: list[int] = []
    canonical: list[dict[str, Any]] = []
    for row in observations:
        if not isinstance(row, Mapping):
            continue
        try:
            forecast_probability = _finite("forecast_probability", row.get("probability"))
            resolved_at = _parse_timestamp(row.get("resolved_at"))
            outcome = int(row.get("outcome"))
        except (TypeError, ValueError):
            continue
        if resolved_at is None or resolved_at >= cutoff or outcome not in (0, 1):
            continue
        row_bin = min(int(forecast_probability / bin_width), int(_ONE / bin_width))
        if row_bin != bin_index:
            continue
        selected.append(outcome)
        canonical.append({"id": str(row.get("id") or ""), "p": str(forecast_probability), "outcome": outcome, "resolved_at": resolved_at.isoformat()})
    digest = sha256(json.dumps(sorted(canonical, key=lambda item: (item["resolved_at"], item["id"])), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if len(selected) < minimum_bin_samples:
        return CalibrationResult(None, len(selected), digest, "insufficient_calibration_sample")
    successes, count = sum(selected), len(selected)
    mean = successes / count
    z = 1.6448536269514722  # one-sided 90% interval
    denominator = 1 + z * z / count
    center = (mean + z * z / (2 * count)) / denominator
    radius = z * sqrt((mean * (1 - mean) + z * z / (4 * count)) / count) / denominator
    conservative = max(0.0, min(1.0, center - radius))
    return CalibrationResult(Decimal(str(conservative)), count, digest)


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