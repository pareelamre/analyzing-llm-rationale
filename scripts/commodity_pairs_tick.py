#!/usr/bin/env python3
"""Publish the daily commodity pair tracker -- runs in a GitHub Action.

Each run:
  1. skips without spending Tapetide quota when the newest complete weekday is published,
     and waits six hours between fetches while Tapetide catches up,
  2. fetches every daily commodity row since prices began (3-5 requests of the free tier's 50),
  3. advances open signals through new sessions and opens signals on the newest session,
  4. writes the public payload served at ``GET /markets/commodity-pairs``.

Outputs (committed back to the repo by the workflow):
  - ``data/commodity_pairs_ledger.jsonl``: every live signal, appended and then closed.
  - ``static/commodity_pairs_live.json``: today's pair states and the track record.

Env:
  TAPETIDE_API_TOKEN  Tapetide personal token (tpt_rt_...)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import commodity_pairs as cp  # noqa: E402
from analyzing_llm_rationale.index_forecast import IST  # noqa: E402

logger = logging.getLogger("commodity_pairs_tick")
LEDGER = ROOT / "data" / "commodity_pairs_ledger.jsonl"
PAYLOAD = ROOT / "static" / "commodity_pairs_live.json"
REFETCH_AFTER_S = 6 * 3600


def _last_complete_weekday(now: datetime) -> date:
    day = now.astimezone(IST).date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def already_published(previous: dict | None, now: datetime) -> bool:
    if not previous or not previous.get("data_through"):
        return False
    if previous["data_through"] >= _last_complete_weekday(now).isoformat():
        return True
    fetched = datetime.fromisoformat(previous["generated_at"])
    return (now - fetched).total_seconds() < REFETCH_AFTER_S


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--payload", type=Path, default=PAYLOAD)
    parser.add_argument("--token-file", type=Path, help="Read the Tapetide token from this file instead of the env")
    parser.add_argument("--force", action="store_true", help="Fetch even when the newest weekday is published")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    now = datetime.now(timezone.utc)
    previous = json.loads(args.payload.read_text(encoding="utf-8")) if args.payload.exists() else None
    if not args.force and already_published(previous, now):
        logger.info("pairs through %s are already published; nothing to fetch", previous["data_through"])
        return 0
    token = (args.token_file.read_text(encoding="utf-8") if args.token_file else os.environ.get("TAPETIDE_API_TOKEN", "")).strip()
    if not token:
        print("::warning::TAPETIDE_API_TOKEN is not set; no commodity pairs published")
        return 0

    today = now.astimezone(IST).date()
    # Today's row is still moving; the tracker uses complete sessions only.
    rows = [row for row in cp.fetch_prices(token, today) if str(row.get("date")) < today.isoformat()]
    ledger = [json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines() if line.strip()] if args.ledger.exists() else []
    before = sum(e["status"] == "open" for e in ledger), len(ledger)
    payload, ledger = cp.run(rows, ledger, now)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in ledger), encoding="utf-8")
    args.payload.parent.mkdir(parents=True, exist_ok=True)
    args.payload.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    logger.info(
        "data_through=%s rows=%d stretched=%s signals: open %d->%d, total %d->%d",
        payload["data_through"], len(rows), [p["key"] for p in payload["pairs"] if p.get("stretched")],
        before[0], sum(e["status"] == "open" for e in ledger), before[1], len(ledger),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
