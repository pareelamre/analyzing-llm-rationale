#!/usr/bin/env python3
"""Cold outreach Foresea to explicit agent endpoints.

With ``--discover``, new targets are found each run (from a static seed list
and optional GitHub search) and merged into the targets file before sending, so
any newly discovered directory is contacted on the same run it's found.

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
payloads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import pr_outreach  # noqa: E402
from analyzing_llm_rationale.pr_outreach_discover import discover_all  # noqa: E402


def _merge_new_targets(targets_path: Path, github_token: str | None) -> int:
    """Discover new targets and append them to the targets file. Returns count added."""
    raw = json.loads(targets_path.read_text())
    existing = raw.get("targets", []) if isinstance(raw, dict) else raw
    existing_endpoints = {t.get("endpoint", "") for t in existing}

    new = discover_all(
        existing_endpoints=existing_endpoints,
        github_token=github_token,
        include_github=bool(github_token),
    )
    if not new:
        return 0

    merged = existing + new
    targets_path.write_text(json.dumps({"targets": merged}, indent=2) + "\n")
    print(json.dumps({"discovered": len(new), "targets": [t["name"] for t in new]}))
    return len(new)


def _run_once(args) -> int:
    if args.discover:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        added = _merge_new_targets(args.targets, github_token=token)
        if added:
            print(f"Added {added} new target(s) to {args.targets}")

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
    # API rejections (409 already-registered, 403 Cloudflare, 404 gone, 405 method
    # not allowed, 422 validation error) are expected when contacting directories and
    # should not fail CI. Only exit 1 if _every_ attempted send failed with a
    # network-level error (no HTTP response at all), which indicates a genuine
    # infrastructure problem rather than normal endpoint churn.
    attempted = [r for r in results if r.get("status") not in {"dry_run", "skipped"}]
    all_network_errors = bool(attempted) and all(
        r.get("status") == "failed" and not r.get("status_code") for r in attempted
    )
    return 1 if all_network_errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Send Foresea PR-agent outreach to explicit agent endpoints.")
    parser.add_argument("--targets", type=Path, required=True, help="JSON target list.")
    parser.add_argument("--canonical", default="https://foresea.ink", help="Canonical Foresea base URL.")
    parser.add_argument("--send", action="store_true", help="Actually POST outreach. Default is dry-run.")
    parser.add_argument("--resend", action="store_true", help="Send even if the target is already in state.")
    parser.add_argument("--discover", action="store_true",
                        help="Discover new targets from seeds + GitHub and merge into the targets file before sending.")
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
