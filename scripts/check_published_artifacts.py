#!/usr/bin/env python3
"""Fail a pull request that rewinds a bot-published artifact.

Several JSON files under ``static/`` are written by scheduled workflows and
committed straight to main with ``[skip ci]``. They are data, not source:
the board, the audit index, the mark-to-market tick, the track record.

A branch cut before one of those publishes carries the older copy. If the
branch also touches the file -- regenerating it locally is enough -- merging
replaces the newer published data with the stale one, and nothing complains,
because the file is valid JSON either way and no test reads it.

That happened on 2026-09-07. static/agent_trading_live.json was published at
20:21; PR #546, opened at 21:11 and merged three minutes later, carried the
18:23 copy and put it back. The live board then served a three-hour-old
payload while its own audit index -- written by the same run, in the same
commit -- still read 20:21. The two disagreed by two hours.

The check is one comparison: for each artifact the branch changed, the
``generated_at`` it carries must not be older than the one on the base
branch. Publishing forward is fine. Standing still is fine, so a branch that
does not touch these files passes trivially. Only going backwards fails.

    python scripts/check_published_artifacts.py --base origin/main
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

#: Artifacts a scheduled workflow publishes. Each carries ``generated_at``.
PUBLISHED_ARTIFACTS: Tuple[str, ...] = (
    "static/agent_trading_live.json",
    "static/agent_trading_audit_live.json",
    "static/agent_trading_audit_archive_manifest.json",
    "static/mark_to_market_live.json",
    "static/track_record_live.json",
    "static/forecast_evaluation.json",
)

OK, REWOUND, UNCHECKABLE = 0, 1, 2


def _git(*args: str) -> Optional[str]:
    proc = subprocess.run(("git",) + args, capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def _generated_at(blob: Optional[str]) -> Optional[datetime]:
    """The artifact's own timestamp, or None if it has none we can read."""
    if not blob:
        return None
    try:
        value = json.loads(blob).get("generated_at")
    except (ValueError, AttributeError):
        return None
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def check(base: str, paths: Sequence[str]) -> Tuple[int, List[str]]:
    messages: List[str] = []
    status = OK
    for path in paths:
        head_blob = _git("show", f":{path}") or _git("show", f"HEAD:{path}")
        base_blob = _git("show", f"{base}:{path}")
        if base_blob is None:
            # New file, or the base ref is unavailable. Not a rewind.
            continue
        head_at, base_at = _generated_at(head_blob), _generated_at(base_blob)
        if head_at is None or base_at is None:
            # Say so rather than passing silently: a check that cannot read
            # the field is not the same as one that read it and was happy.
            messages.append(f"  ? {path}: no readable generated_at to compare")
            status = max(status, UNCHECKABLE)
            continue
        if head_at < base_at:
            messages.append(
                f"  x {path}\n"
                f"      this branch: {head_at.isoformat()}\n"
                f"      {base}: {base_at.isoformat()}\n"
                f"      merging would discard {base_at - head_at} of published data"
            )
            status = REWOUND
        elif head_at > base_at:
            messages.append(f"  + {path}: advances to {head_at.isoformat()}")
    return status, messages


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main",
                        help="ref to compare against (default: origin/main)")
    parser.add_argument("paths", nargs="*", default=list(PUBLISHED_ARTIFACTS),
                        help="artifacts to check (default: all published ones)")
    args = parser.parse_args(argv)

    status, messages = check(args.base, args.paths or list(PUBLISHED_ARTIFACTS))
    if status == REWOUND:
        print("A published artifact would be rewound by this branch:", file=sys.stderr)
        print("\n".join(messages), file=sys.stderr)
        print(
            "\nThese files are published by scheduled workflows, not written by\n"
            "hand. Restore them from the base branch:\n"
            f"    git checkout {args.base} -- " + " ".join(PUBLISHED_ARTIFACTS),
            file=sys.stderr,
        )
        return REWOUND
    if messages:
        print("\n".join(messages))
    print("No published artifact is rewound.")
    return OK if status == OK else OK  # unreadable is reported, not fatal


if __name__ == "__main__":
    raise SystemExit(main())
