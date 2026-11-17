"""Check the knowledge base's watch list for new report editions.

Reads ``watchlist.json`` and ``watch_state.json`` in the knowledge repo, appends new editions to
``inbox.jsonl`` and downloads their files under ``.kb/inbox/`` (not committed). See
src/analyzing_llm_rationale/knowledge_watch.py for the rules the crawler follows.

    python scripts/knowledge_watch.py [--kb DIR] [--only eia-steo ...] [--no-download]
    python scripts/knowledge_watch.py --validate

In GitHub Actions, sources that are blocked, broken or suspect become run warnings and the
table goes to the job summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402
from analyzing_llm_rationale import knowledge_watch as watch  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kb", help="knowledge repo clone (default: KNOWLEDGE_DIR or ../foresea-knowledge)")
    parser.add_argument("--only", nargs="*", default=[], help="check only these source ids")
    parser.add_argument("--download", help="where new files go (default: <kb>/.kb/inbox)")
    parser.add_argument("--no-download", action="store_true", help="record new editions without downloading")
    parser.add_argument("--validate", action="store_true", help="only check watchlist.json")
    args = parser.parse_args(argv)

    root = Path(args.kb).expanduser() if args.kb else kb.default_dir(allow_sibling=True)
    if root is None or not root.is_dir():
        sys.exit("No knowledge base: pass --kb or set KNOWLEDGE_DIR.")
    watchlist = watch.load_json(root / "watchlist.json", None)
    errors = watch.validate(watchlist)
    for error in errors:
        print(f"watchlist.json: {error}", file=sys.stderr)
    if errors or args.validate:
        if not errors:
            print(f"watchlist.json: {len(watchlist['sources'])} sources, fine")
        return 1 if errors else 0

    state_path = root / "watch_state.json"
    state = watch.load_json(state_path, {})
    download_dir = None if args.no_download else Path(args.download or root / ".kb" / "inbox").expanduser()
    today = datetime.now(timezone.utc).date()
    results, new_state, inbox = watch.run(watchlist, state, watch.Crawler(), today=today,
                                          download_dir=download_dir, only=args.only)

    state_path.write_text(json.dumps(new_state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if inbox:
        with (root / "inbox.jsonl").open("a", encoding="utf-8") as fh:
            for row in inbox:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    table = watch.summary(results)
    print(table)
    for row in inbox:
        print(f"new: {row['source']} {row['edition']} -> {row['url']}")
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for result in results:
        if result["status"] not in ("ok",) and in_actions:
            print(f"::warning title=Watch {result['status']}::{result['id']}: {result.get('detail', '')}")
        for edition in result.get("new") or []:
            for f in edition["files"]:
                if f.get("error") and in_actions:
                    print(f"::warning title=Download failed::{result['id']} {f['url']}: {f['error']}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(f"### Knowledge watch, {today.isoformat()}\n\n{table}\n")
            fh.write("".join(f"- new: {r['source']} {r['edition']}\n" for r in inbox) or "No new editions.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
