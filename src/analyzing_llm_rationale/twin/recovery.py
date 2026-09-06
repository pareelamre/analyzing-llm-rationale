"""Conservative recovery state machine for ambiguous venue submissions."""
from __future__ import annotations


def recovery_action(command_state: str, venue_order_found: bool | None) -> str:
    if command_state in {"filled", "cancelled", "rejected"}:
        return "terminal"
    if command_state in {"submitting", "submission_unknown"}:
        if venue_order_found is True:
            return "reconcile"
        if venue_order_found is False:
            return "mark_no_order"
        return "pause_and_reconcile"
    return "resume"
