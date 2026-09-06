"""Pure saved-run state transitions shared by HTTP and future workers."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def claim_for_submission(
    record: Mapping[str, Any], preview: Mapping[str, Any], *, approved_at: str
) -> tuple[dict[str, Any], bool]:
    """Return one immutable transition into ``submitting``.

    Persistence and locking belong to the caller.  This function deliberately
    knows nothing about FastAPI, credentials, exchanges, or user identity so a
    worker can use the exact same transition under its durable-store claim.
    """
    claimed = deepcopy(dict(record))
    if claimed.get("status") != "awaiting_approval":
        return claimed, False
    claimed["preview"] = deepcopy(dict(preview))
    claimed["estimated_notional"] = preview.get("estimated_notional")
    claimed["status"] = "submitting"
    claimed["approved_at"] = approved_at
    claimed["updated_at"] = approved_at
    claimed["error_code"] = None
    return claimed, True
