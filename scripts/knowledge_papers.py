"""Collect forecasting, quantitative-finance and prediction-market papers into the knowledge base.

Daily: new papers from arXiv's RSS feeds and from OpenAlex become cards in
``papers/cards/<YYYY-MM>.jsonl``. Mondays (or ``--weekly``): the best few unread papers with an
open PDF go to ``inbox.jsonl`` for the Mac's night reader, and ``papers/digest/latest.md`` lists the
week's best papers by the part of Foresea they could improve. See
src/analyzing_llm_rationale/knowledge_papers.py.

    python scripts/knowledge_papers.py [--kb DIR] [--weekly | --no-weekly] [--per-week 3]

In GitHub Actions a feed that fails becomes a run warning and the table goes to the job summary.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import knowledge_papers as papers  # noqa: E402
from analyzing_llm_rationale import knowledge_watch as watch  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb", default=os.environ.get("KNOWLEDGE_DIR", "."), help="the knowledge repo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--weekly", action="store_true", help="queue papers and write the digest today")
    group.add_argument("--no-weekly", action="store_true", help="never the weekly step")
    parser.add_argument("--per-week", type=int, default=papers.PER_WEEK)
    args = parser.parse_args(argv)
    root = Path(args.kb).expanduser()
    today = datetime.now(timezone.utc).date()

    result = papers.collect(root, watch.Crawler(), today)
    rows = [f"| {name} | {r['status']} | {r.get('papers', '')} | {r.get('new', '')} | {r.get('why', '')} |"
            for name, r in result["feeds"].items()]
    table = "| Feed | Status | Papers | New | Why |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    print(table)
    print(f"{len(result['found'])} new papers")
    out = {"found": len(result["found"])}
    if args.weekly or (not args.no_weekly and today.weekday() == 0):
        out.update(papers.weekly(root, today, per_week=args.per_week))
        print("queued for reading:", ", ".join(out["queued"]) or "none")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for name, r in result["feeds"].items():
            if r["status"] != "ok":
                print(f"::warning title=Paper feed {name}::{r.get('why')}")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write("## Papers\n\n" + table + f"\n{len(result['found'])} new papers.\n")
    failed = sum(r["status"] != "ok" for r in result["feeds"].values())
    return 1 if failed == len(result["feeds"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
