#!/usr/bin/env python3
"""Cold outreach Foresea to explicit agent endpoints.

Example targets file:

{
  "targets": [
    {
      "name": "Example Agent Directory",
      "endpoint": "https://agent-directory.example/inbox",
      "audience": "catalog",
      "headers": {"Authorization": "Bearer ..."}
    }
  ]
}

Dry-run is the default. Add ``--send`` only after reviewing the target list and
payloads. This script never discovers recipients on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import pr_outreach  # noqa: E402


def _run_once(args) -> int:
    targets = pr_outreach.load_targets(args.targets)
    state = pr_outreach.load_state(args.state)
    selected = pr_outreach.filter_unsent(targets, state, resend=args.resend)
    results = pr_outreach.send_outreach(
        selected,
        canonical=args.canonical,
        send=args.send,
        timeout_s=args.timeout_s,
        pause_s=args.pause_s,
    )
    skipped = len(targets) - len(selected)
    if args.send:
        pr_outreach.mark_sent(state, results)
        pr_outreach.save_state(args.state, state)
    text = json.dumps({
        "sent": args.send,
        "count": len(results),
        "skipped_already_sent": skipped,
        "state": str(args.state),
        "results": results,
    }, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if all(r["status"] in {"dry_run", "sent"} for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Send Foresea PR-agent outreach to explicit agent endpoints.")
    parser.add_argument("--targets", type=Path, required=True, help="JSON target list.")
    parser.add_argument("--canonical", default="https://foresea.ink", help="Canonical Foresea base URL.")
    parser.add_argument("--send", action="store_true", help="Actually POST outreach. Default is dry-run.")
    parser.add_argument("--resend", action="store_true", help="Send even if the target is already in state.")
    parser.add_argument("--state", type=Path, default=ROOT / "data" / "pr_outreach_state.json",
                        help="Send-once state file. Only updated with --send.")
    parser.add_argument("--timeout-s", type=float, default=20.0, help="HTTP timeout per target.")
    parser.add_argument("--pause-s", type=float, default=2.0, help="Delay between sends.")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write JSON results.")
    parser.add_argument("--watch", action="store_true", help="Run continuously, polling the target list.")
    parser.add_argument("--interval-s", type=float, default=300.0, help="Polling interval for --watch.")
    args = parser.parse_args()

    if not args.watch:
        return _run_once(args)

    interval = max(10.0, args.interval_s)
    while True:
        code = _run_once(args)
        if code != 0:
            return code
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
