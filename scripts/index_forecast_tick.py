#!/usr/bin/env python3
"""Publish the daily NSE index range forecasts -- runs in a GitHub Action after the close.

Each run:
  1. skips when today's forecasts are already published,
  2. fetches every close since 2008 of ten indices and India VIX, and NSE holidays, from
     Upstox's open data routes (about 35 requests; no account or token),
  3. scores ledger rows whose target session has closed,
  4. appends today's forecasts to the ledger, only while the issue window is open
     (after the close they are made from, before the next session opens),
  5. writes the public payload served at ``GET /markets/index-forecast``.

Outputs (committed back to the repo by the workflow):
  - ``data/index_forecast_ledger.jsonl``: every recorded forecast and, once its target
    session closes, its score. Rows are only ever appended or scored, never rewritten.
  - ``static/index_forecast_live.json``: today's forecasts and the track record.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import index_forecast as ixf  # noqa: E402

logger = logging.getLogger("index_forecast_tick")
LEDGER = ROOT / "data" / "index_forecast_ledger.jsonl"
PAYLOAD = ROOT / "static" / "index_forecast_live.json"


def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _latest_close(now: datetime, holidays: list[date]) -> date:
    """The most recent session whose close has happened by ``now``."""
    local = now.astimezone(ixf.IST)
    day = local.date()
    closed = set(holidays)
    if not (day.weekday() < 5 and day not in closed and local.time() >= time(15, 30)):
        while True:
            day = date.fromordinal(day.toordinal() - 1)
            if day.weekday() < 5 and day not in closed:
                break
    return day


def already_published(previous: dict | None, now: datetime) -> bool:
    if not previous or not previous.get("as_of"):
        return False
    # An index Upstox is missing does not force a refetch; the next close brings it back.
    holidays = [date.fromisoformat(d) for d in previous.get("holidays", [])]
    return previous["as_of"] >= _latest_close(now, holidays).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--payload", type=Path, default=PAYLOAD)
    parser.add_argument("--force", action="store_true", help="Fetch even when today's forecasts are published")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    now = datetime.now(timezone.utc)
    previous = json.loads(args.payload.read_text(encoding="utf-8")) if args.payload.exists() else None
    if not args.force and already_published(previous, now):
        logger.info("forecasts as of %s are already published; nothing to fetch", previous["as_of"])
        return 0
    data = ixf.fetch_market_data(now)
    missing = [name for name, series in data["levels"].items() if not series]
    if len(missing) == len(ixf.INDICES):
        logger.error("Upstox returned no index levels")
        return 1
    ledger = _read_ledger(args.ledger)
    before = len(ledger)
    payload, ledger = ixf.run(data, ledger, now)
    payload["holidays"] = [d.isoformat() for d in data["holidays"]]
    _write_ledger(args.ledger, ledger)
    args.payload.parent.mkdir(parents=True, exist_ok=True)
    args.payload.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    logger.info(
        "as_of=%s indices=%d new_rows=%d pending=%d missing=%s",
        payload["as_of"], len(payload["indices"]), len(ledger) - before, payload["track_record"]["pending"], missing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
