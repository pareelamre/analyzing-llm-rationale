"""Read publishers' current forecast tables into the knowledge base (no model involved).

For each reader in src/analyzing_llm_rationale/knowledge_tables.py (EIA STEO, Fed projections,
ECB projections, IMF WEO), finds the current edition from the watch list, skips it when the base
already has it, and otherwise writes a new source folder after the plausibility checks pass.

    python scripts/knowledge_tables.py [--kb DIR] [--only eia-steo ...] [--dry-run]

In GitHub Actions a failed reader becomes a run warning and the table goes to the job summary.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402
from analyzing_llm_rationale import knowledge_tables as tables  # noqa: E402
from analyzing_llm_rationale import knowledge_watch as watch  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kb", help="knowledge repo clone (default: KNOWLEDGE_DIR or ../foresea-knowledge)")
    parser.add_argument("--only", nargs="*", default=[], help=f"readers to run: {', '.join(tables.READERS)}")
    parser.add_argument("--dry-run", action="store_true", help="read and check, but write nothing")
    args = parser.parse_args(argv)

    root = Path(args.kb).expanduser() if args.kb else kb.default_dir(allow_sibling=True)
    if root is None or not root.is_dir():
        sys.exit("No knowledge base: pass --kb or set KNOWLEDGE_DIR.")
    today = datetime.now(timezone.utc).date()
    try:
        results = tables.update(root, watch.Crawler(), today=today, only=args.only, dry_run=args.dry_run)
    except tables.TableError as exc:
        print(f"knowledge tables: {exc}", file=sys.stderr)
        return 1
    table = tables.summary(results)
    print(table)
    for r in results:
        for line in r.get("preview") or []:
            print(f"  {r['reader']}: {line}")
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for r in results:
        if r["status"] == "failed" and in_actions:
            print(f"::warning title=Table reader failed::{r['reader']}: {r.get('detail', '')}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(f"### Knowledge tables, {today.isoformat()}\n\n{table}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
