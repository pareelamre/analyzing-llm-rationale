"""Generate deterministic JSON and Markdown reports for T21 forward evidence."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.twin.trial import build_trial_report  # noqa: E402


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _markdown(report: dict) -> str:
    g1, g2 = report["g1"], report["g2"]
    blockers = report["blockers"] or ["None recorded."]
    drills = "\n".join(
        f"| `{name}` | {item['status']} | {item['evidence_id'] or 'pending'} |"
        for name, item in g1["drills"].items()
    )
    blocker_lines = "\n".join(f"- {item}" for item in blockers)
    return f"""# Autonomous twin forward trial

Generated at `{report['generated_at']}` from immutable evidence
`{report['evidence_hash']}` for image `{report['release']['image_digest']}`.

| Gate | Status | Reason |
| --- | --- | --- |
| G1 mechanics | **{g1['status']}** | `{g1['reason']}` |
| G2 strategy | **{g2['status']}** | `{g2['reason']}` |

G1 has `{g1['consecutive_days']}` of 7 required consecutive UTC days,
`{g1['totals']['complete_market_snapshots']}` complete market snapshots,
`{g1['totals']['decisions']}` decisions, and
`{g1['totals']['simulated_commands']}` simulated commands. Unexplained ledger
divergences: `{g1['totals']['unexplained_divergences']}`. Duplicate simulated
commands: `{g1['totals']['duplicate_commands']}`. Attempts to add exposure from
stale data: `{g1['totals']['stale_exposure_attempts']}`.

| Required fault drill | Status | Evidence |
| --- | --- | --- |
{drills}

## Blockers

{blocker_lines}

This artifact never authorizes live trading. `live_eligible` is fixed to
`false`; G3 still requires the owner's explicit, expiring capital mandate.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args(argv)
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    report = build_trial_report(evidence, as_of=datetime.fromisoformat(args.as_of))
    _write_atomic(args.json_output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_atomic(args.markdown_output, _markdown(report))
    print(json.dumps({
        "artifact_hash": report["artifact_hash"],
        "g1": report["g1"]["status"], "g2": report["g2"]["status"],
        "json_output": str(args.json_output),
        "markdown_output": str(args.markdown_output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
