"""List Foresea's open long-horizon questions that the knowledge base has nothing on.

Writes ``gaps/latest.md`` (to read), ``gaps/latest.json`` and one line per day in
``gaps/history.jsonl`` inside the knowledge repo. See
src/analyzing_llm_rationale/knowledge_gaps.py for what counts as a question and a gap.

    python scripts/knowledge_gaps.py [--kb DIR] [--foresea ROOT] [--out DIR] [--min-days 30]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402
from analyzing_llm_rationale import knowledge_gaps as gaps  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kb", help="knowledge repo clone (default: KNOWLEDGE_DIR or ../foresea-knowledge)")
    parser.add_argument("--foresea", default=str(Path(__file__).resolve().parents[1]),
                        help="Foresea checkout holding data/ and static/ (default: this repo)")
    parser.add_argument("--out", help="report folder (default: <kb>/gaps)")
    parser.add_argument("--min-days", type=int, default=gaps.MIN_DAYS,
                        help="only questions closing at least this many days from today")
    parser.add_argument("--today", help="YYYY-MM-DD (default: today, UTC)")
    args = parser.parse_args(argv)

    root = Path(args.kb).expanduser() if args.kb else kb.default_dir(allow_sibling=True)
    if root is None or not root.is_dir():
        sys.exit("No knowledge base: pass --kb or set KNOWLEDGE_DIR.")
    corpus = kb.load(root)
    if corpus.errors:
        for error in corpus.errors:
            print(error, file=sys.stderr)
        return 1
    today = kb.parse_when(args.today) if args.today else gaps.today_utc()
    if today is None:
        sys.exit(f"--today must be YYYY-MM-DD, got {args.today!r}")
    foresea = Path(args.foresea).expanduser()
    report = gaps.scan(kb.Index(corpus), gaps.open_questions(foresea, today, min_days=args.min_days),
                       gaps.fund_pairs(foresea), today)
    out = Path(args.out).expanduser() if args.out else root / "gaps"
    gaps.write(report, out)
    line = (f"{report['covered']} of {report['questions']} open long-horizon questions covered; "
            f"{report['pairs_covered']} of {report['pairs']} fund pairs have a view")
    print(line)
    print(f"wrote {out / 'latest.md'}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(f"### Knowledge gaps, {report['date']}\n\n{line}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
