"""Is the published resolved track record due for a refresh?

track-record-resolved.yml is woken after every MTM tick (the box's cron
dispatches one every 15 minutes) because its own hourly cron fired only every
4-8 hours, which left /track-record marked stale. Its pass holds the store lock
for about 12 minutes, so it runs only when the published record is at least
TRACK_RECORD_REFRESH_AFTER_S old: about once an hour.

Usage: python scripts/track_record_due.py static/track_record_live.json
Prints ``run=true`` or ``run=false`` for $GITHUB_OUTPUT. Stdlib only.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REFRESH_AFTER_S = 3000.0


def is_due(generated_at: Optional[str], now: datetime, refresh_after_s: float = REFRESH_AFTER_S) -> bool:
    """A record with no readable timestamp is always due."""
    if not generated_at:
        return True
    try:
        made = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
    except ValueError:
        return True
    if made.tzinfo is None:
        made = made.replace(tzinfo=timezone.utc)
    return (now - made).total_seconds() >= refresh_after_s


def _generated_at(path: Path) -> Optional[str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("generated_at")
    except (OSError, ValueError, AttributeError):
        return None


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "static/track_record_live.json")
    refresh_after_s = float(os.environ.get("TRACK_RECORD_REFRESH_AFTER_S") or REFRESH_AFTER_S)
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    generated_at = _generated_at(path)
    due = manual or is_due(generated_at, datetime.now(timezone.utc), refresh_after_s)
    print(f"published record generated_at={generated_at}; due={due}", file=sys.stderr)
    print(f"run={'true' if due else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
