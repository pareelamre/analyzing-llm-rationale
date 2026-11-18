"""Read reports into the knowledge base with the self-hosted model, on the Mac (phase 4).

    python scripts/knowledge_reports.py run [--minutes 50] [--hours 23-7] [--any-time]
    python scripts/knowledge_reports.py queue report.pdf|URL --id iea-2026-10-world-energy-outlook \
        --title "..." --publisher IEA --published 2026-10-15 [--licence "CC BY 4.0"]
    python scripts/knowledge_reports.py status
    python scripts/knowledge_reports.py sheet <id>        # candidates beside their passages
    python scripts/knowledge_reports.py decide <id> --drop c003,c007 --rest keep [--edits fixes.json]
    python scripts/knowledge_reports.py entities <id>     # names the base knows, and new ones to map
    python scripts/knowledge_reports.py publish <id> --reviewer Claude --entities map.json [--title "..."]
    python scripts/knowledge_reports.py drop <id> --why "duplicate of ..."

``run`` (launchd, every 30 minutes) queues new report editions from the watch list's inbox and
reads passages with the model at 127.0.0.1:8080 while the live site is idle, only on AC power and
inside the night window. Nothing reaches ``sources/`` until a reviewer decides every candidate and
runs ``publish``. See src/analyzing_llm_rationale/knowledge_reports.py.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402
from analyzing_llm_rationale import knowledge_reports as reports  # noqa: E402
from analyzing_llm_rationale import knowledge_watch as watch  # noqa: E402


def _root(args) -> Path:
    root = Path(args.kb).expanduser() if args.kb else kb.default_dir(allow_sibling=True)
    if root is None or not root.is_dir():
        sys.exit("No knowledge base: pass --kb or clone pareelamre/foresea-knowledge beside this repo.")
    return root


def _log(line: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}", flush=True)


def on_ac_power() -> bool:
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return "Battery Power" not in out


def cmd_run(args) -> int:
    root = _root(args)
    if not args.any_time and not reports.in_window(args.hours):
        return 0
    if not args.any_time and not on_ac_power():
        _log("on battery: not reading tonight")
        return 0
    lock_path = root / reports.WORK / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0  # the previous run is still reading
    busy = reports.live_gate(idle_s=args.idle)
    url = os.environ.get("KNOWLEDGE_LLM_URL", reports.LOCAL_URL)
    key = os.environ.get("SELF_HOSTED_LLM_API_KEY", "") if url != reports.LOCAL_URL else ""

    def call(prompt: str) -> str:
        return reports.stream_call(prompt, busy, url=url, key=key)

    deadline = time.time() + args.minutes * 60
    results = reports.run(root, call=call, busy=busy, deadline=deadline, crawler=watch.Crawler(), log=_log)
    for r in results:
        _log(json.dumps(r))
    return 0


def cmd_queue(args) -> int:
    root = _root(args)
    target = args.file
    is_url = target.startswith(("http://", "https://"))
    extra = {"licence": args.licence, "topics": [t.strip() for t in (args.topics or "").split(",") if t.strip()]}
    job = reports.queue(root, jid=args.id, title=args.title, publisher=args.publisher, published=args.published,
                        url=args.url, file_url=target if is_url else None,
                        path=None if is_url else Path(target), extra=extra)
    print(json.dumps(job, indent=2))
    return 0


def cmd_status(args) -> int:
    print(reports.status_table(_root(args)), end="")
    return 0


def cmd_sheet(args) -> int:
    print(reports.review_sheet(_root(args), args.id), end="")
    return 0


def _names(text: str | None) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def cmd_decide(args) -> int:
    edits = json.loads(Path(args.edits).read_text(encoding="utf-8")) if args.edits else None
    out = reports.decide(_root(args), args.id, keep=_names(args.keep), drop=_names(args.drop), rest=args.rest,
                         edits=edits)
    print(json.dumps(out))
    return 0


def cmd_publish(args) -> int:
    root = _root(args)
    job = reports.load_job(root, args.id)
    extra = {k: job.get(k) for k in ("licence", "topics", "summary")}
    extra.update({k: getattr(args, k) for k in ("title", "summary", "licence") if getattr(args, k)})
    entity_map = json.loads(Path(args.entities).read_text(encoding="utf-8")) if args.entities else None
    target = reports.publish(root, args.id, reviewer=args.reviewer, source_extra=extra, entity_map=entity_map)
    print(f"published {target.relative_to(root)}")
    left = reports.load_job(root, args.id).get("unlinked_entities") or []
    if left:
        print(f"left off the graph (not in the base or the entity map): {', '.join(left)}")
    print(json.dumps(kb.stats(kb.load(root))))
    return 0


def cmd_entities(args) -> int:
    print(json.dumps(reports.entity_names(_root(args), args.id), indent=2, ensure_ascii=False))
    return 0


def cmd_drop(args) -> int:
    print(json.dumps(reports.drop_job(_root(args), args.id, args.why)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb", help="knowledge clone (default: KNOWLEDGE_DIR or ../foresea-knowledge)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--minutes", type=float, default=50, help="stop starting passages after this long")
    p.add_argument("--hours", default=os.environ.get("KNOWLEDGE_REPORT_HOURS", "23-7"),
                   help="local night window, e.g. 23-7")
    p.add_argument("--any-time", action="store_true", help="ignore the window and the power check")
    p.add_argument("--idle", type=float, default=reports.IDLE_S, help="seconds of live quiet before a passage")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("queue")
    p.add_argument("file", help="a local PDF or its URL")
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--publisher", required=True)
    p.add_argument("--published", required=True)
    p.add_argument("--url", help="the report's page")
    p.add_argument("--licence")
    p.add_argument("--topics")
    p.set_defaults(fn=cmd_queue)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("sheet")
    p.add_argument("id")
    p.set_defaults(fn=cmd_sheet)
    p = sub.add_parser("decide")
    p.add_argument("id")
    p.add_argument("--keep")
    p.add_argument("--drop")
    p.add_argument("--rest", choices=("keep", "drop"))
    p.add_argument("--edits", help='JSON file: {"c004": {"text": "...", "horizon": "2030"}}')
    p.set_defaults(fn=cmd_decide)
    p = sub.add_parser("entities", help="entity names in the kept candidates: known ones and new ones")
    p.add_argument("id")
    p.set_defaults(fn=cmd_entities)
    p = sub.add_parser("publish")
    p.add_argument("id")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--entities", help='entity map JSON: {"map": {"name": "id" or null}, "new": {"id": {"name", "type", "aliases"}}}')
    p.add_argument("--title", help="the report's real title, when the watch list only knew its edition key")
    p.add_argument("--summary")
    p.add_argument("--licence")
    p.set_defaults(fn=cmd_publish)
    p = sub.add_parser("drop")
    p.add_argument("id")
    p.add_argument("--why", required=True)
    p.set_defaults(fn=cmd_drop)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except reports.ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
