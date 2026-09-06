"""Deterministic entry sizing; no provider or venue calls occur here."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional


@dataclass(frozen=True)
class RiskLimits:
    kelly_fraction: Decimal
    max_order_cash: Decimal
    max_market_loss: Decimal
    max_cluster_loss: Decimal
    max_drawdown: Decimal


@dataclass(frozen=True)
class RiskResult:
    quantity: Decimal
    cash: Decimal
    max_loss: Decimal
    reason: Optional[str] = None


def size_binary_entry(
    *, probability: Optional[Decimal], ask: Optional[Decimal], fee_per_share: Optional[Decimal],
    slippage_per_share: Optional[Decimal], available_cash: Optional[Decimal], current_market_loss: Decimal,
    current_cluster_loss: Decimal, drawdown: Decimal, tick_size: Decimal, min_quantity: Decimal,
    limits: RiskLimits,
) -> RiskResult:
    """Conservatively size a YES/NO purchase using the executable ask."""
    if None in {probability, ask, fee_per_share, slippage_per_share, available_cash}:
        return RiskResult(Decimal("0"), Decimal("0"), Decimal("0"), "missing_risk_input")
    p, price, fee, slip, cash = map(Decimal, (probability, ask, fee_per_share, slippage_per_share, available_cash))
    if not (Decimal("0") < p < Decimal("1") and Decimal("0") < price < Decimal("1")) or min(tick_size, min_quantity) <= 0:
        return RiskResult(Decimal("0"), Decimal("0"), Decimal("0"), "invalid_market_input")
    if drawdown >= limits.max_drawdown:
        return RiskResult(Decimal("0"), Decimal("0"), Decimal("0"), "drawdown_limit")
    edge = p - price - fee - slip
    if edge <= 0:
        return RiskResult(Decimal("0"), Decimal("0"), Decimal("0"), "no_net_edge")
    full_kelly = edge / (Decimal("1") - price)
    target_cash = min(cash, limits.max_order_cash, cash * full_kelly * limits.kelly_fraction)
    unit_cost, unit_loss = price + fee + slip, Decimal("1") + fee + slip
    maximum = min(target_cash / unit_cost, (limits.max_market_loss - current_market_loss) / unit_loss, (limits.max_cluster_loss - current_cluster_loss) / unit_loss)
    quantity = max(Decimal("0"), maximum).quantize(min_quantity, rounding=ROUND_DOWN)
    if quantity < min_quantity:
        return RiskResult(Decimal("0"), Decimal("0"), Decimal("0"), "minimum_size_exceeds_cap")
    return RiskResult(quantity, quantity * unit_cost, quantity * unit_loss)
