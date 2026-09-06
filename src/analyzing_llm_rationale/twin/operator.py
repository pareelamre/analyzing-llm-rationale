"""Safe operator-facing state derived from twin records, without credentials."""
from __future__ import annotations

from typing import Mapping


def operator_status(*, mandate_active: bool, paused: bool, unknown_commands: int, shadow_only: bool = True) -> Mapping[str, object]:
    return {
        "mode": "shadow" if shadow_only else "live",
        "mandate_active": mandate_active,
        "paused": paused,
        "unknown_commands": unknown_commands,
        "new_exposure_allowed": mandate_active and not paused and unknown_commands == 0 and shadow_only,
    }
