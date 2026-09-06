"""Seeded, network-free binary venue adapter for shadow execution."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from random import Random


@dataclass(frozen=True)
class ShadowReceipt:
    order_id: str
    filled_quantity: Decimal
    remaining_quantity: Decimal
    fee: Decimal
    status: str


class ShadowVenue:
    def __init__(self, *, account_id: str, seed: int, fee_rate: Decimal = Decimal("0")) -> None:
        if account_id.startswith("live"):
            raise ValueError("shadow account identifiers must not overlap live accounts")
        self.account_id, self._random, self.fee_rate = account_id, Random(seed), fee_rate
        self._orders: dict[str, ShadowReceipt] = {}

    def submit(self, order_id: str, *, quantity: Decimal, ask: Decimal, depth: Decimal) -> ShadowReceipt:
        if order_id in self._orders:
            return self._orders[order_id]
        executable = min(quantity, depth)
        # A deterministic latency scenario can leave a small remainder.
        filled = executable if Decimal(str(self._random.random())) >= Decimal("0.1") else Decimal("0")
        fee = filled * ask * self.fee_rate
        receipt = ShadowReceipt(order_id, filled, quantity - filled, fee, "filled" if filled == quantity else "partial" if filled else "open")
        self._orders[order_id] = receipt
        return receipt

    def cancel(self, order_id: str) -> ShadowReceipt:
        current = self._orders[order_id]
        receipt = ShadowReceipt(order_id, current.filled_quantity, current.remaining_quantity, current.fee, "cancelled")
        self._orders[order_id] = receipt
        return receipt
