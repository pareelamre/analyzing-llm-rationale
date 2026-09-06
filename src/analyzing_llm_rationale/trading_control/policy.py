"""Pure policy checks used before any exchange or credential operation."""
from __future__ import annotations

from typing import Any, Mapping


class GuardrailViolation(ValueError):
    """A deterministic policy failure that a transport layer can render."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_pre_submission_policy(
    policy: Mapping[str, Any], *, estimated_notional: float, duplicate: bool
) -> None:
    """Check policy facts that require no external call or server framework."""
    if bool(policy.get("paused")):
        raise GuardrailViolation("user_paused", "Trading is paused in your risk controls. Resume it before submitting a new order.")
    if estimated_notional <= 0:
        raise GuardrailViolation("invalid_notional", "The order did not produce a positive risk notional.")
    maximum = float(policy["max_order_notional"])
    if estimated_notional > maximum:
        raise GuardrailViolation(
            "max_order_notional",
            f"Order risk ${estimated_notional:.2f} exceeds your ${maximum:.2f} per-order limit.",
        )
    if duplicate:
        raise GuardrailViolation(
            "duplicate_cooldown",
            f"An equivalent order is already active or was created within the {int(policy['cooldown_seconds'])}-second cooldown.",
        )
