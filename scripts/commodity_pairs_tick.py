#!/usr/bin/env python3
"""Publish the daily commodity pair tracker -- runs in a GitHub Action.

Each run:
  1. skips when the newest complete weekday is published,
  2. fetches Brent, WTI and Henry Hub spot prices from FRED, and appends the MCX front-month
     closes (Upstox) and gold and platinum snapshots (fawazahmed0 currency API) after the last
     stored day (no account or token for any of them),
  3. advances open signals through new sessions and opens signals on the newest session,
  4. writes the public payload served at ``GET /markets/commodity-pairs``.

Outputs (committed back to the repo by the workflow):
  - ``data/commodity_prices.jsonl``: MCX front-month closes and gold and platinum snapshots,
    one row per commodity, source and session, appended only. Expired contracts leave Upstox's
    route, so this is the history.
  - ``data/commodity_pairs_ledger.jsonl``: every live signal, appended and then closed.
  - ``static/commodity_pairs_live.json``: today's pair states and the track record.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import commodity_pairs as cp  # noqa: E402

logger = logging.getLogger("commodity_pairs_tick")
LEDGER = ROOT / "data" / "commodity_pairs_ledger.jsonl"
PAYLOAD = ROOT / "static" / "commodity_pairs_live.json"
PRICES = ROOT / "data" / "commodity_prices.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _last_complete_weekday(now: datetime) -> date:
    day = now.astimezone(cp.IST).date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def already_published(previous: dict | None, now: datetime) -> bool:
    if not previous or not previous.get("data_through"):
        return False
    # FRED lags a few days, so the newest weekday is the MCX one; FRED pairs catch up on later runs.
    return previous["data_through"] >= _last_complete_weekday(now).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--payload", type=Path, default=PAYLOAD)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--force", action="store_true", help="Fetch even when the newest weekday is published")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    now = datetime.now(timezone.utc)
    previous = json.loads(args.payload.read_text(encoding="utf-8")) if args.payload.exists() else None
    if not args.force and already_published(previous, now):
        logger.info("pairs through %s are already published; nothing to fetch", previous["data_through"])
        return 0
    today = now.astimezone(cp.IST).date()
    fred = cp.fetch_fred(today)
    stored = _read_jsonl(args.prices)
    new_prices = cp.fetch_mcx(stored, today) + cp.fetch_metals(stored, today)
    stored += new_prices
    ledger = _read_jsonl(args.ledger)
    before = sum(e["status"] == "open" for e in ledger), len(ledger)
    payload, ledger = cp.run(cp.build_series(stored, fred), ledger, now)
    _write_jsonl(args.prices, stored)
    _write_jsonl(args.ledger, ledger)
    args.payload.parent.mkdir(parents=True, exist_ok=True)
    args.payload.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    logger.info(
        "data_through=%s new_price_rows=%d stretched=%s signals: open %d->%d, total %d->%d",
        payload["data_through"], len(new_prices), [p["key"] for p in payload["pairs"] if p.get("stretched")],
        before[0], sum(e["status"] == "open" for e in ledger), before[1], len(ledger),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
