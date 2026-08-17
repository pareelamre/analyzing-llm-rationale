#!/usr/bin/env python3
"""Aggregate every agentic-trading model's shadow account into one board JSON.

Each of the 10 SCADS models trades through its own isolated GCS-synced SQLite
store (see scripts/agent_trading_tick.py); this script reads all 10 (already
downloaded locally, one directory per model, by the publish workflow),
fetches a live quote (Kalshi or Polymarket, matching each position's own
venue) for every ticker anyone currently holds, and writes
static/agent_trading_live.json -- the artifact GET /agent-trading/board serves,
following the same committed-static-JSON pattern as track_record_live.json /
mark_to_market_live.json (raw.githubusercontent.com, no dedicated read API
storage of its own).

Env:
  AGENT_TRADING_BOARD_STORE_DIR      dir containing <model>/store.sqlite and
                                      <model>/notes.json per model (default
                                      tmp/agent-trading-board)
  AGENT_TRADING_BOARD_OUTPUT         output path (default
                                      static/agent_trading_live.json)
  AGENT_TRADING_BOARD_ACTIVITY_LIMIT max transparency-feed entries (default 50)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import agent_trading_stats, benchmark_tools, market_data  # noqa: E402
from analyzing_llm_rationale.config import load_model_configs  # noqa: E402

STORE_DIR = Path(os.environ.get("AGENT_TRADING_BOARD_STORE_DIR", "tmp/agent-trading-board"))
OUTPUT_PATH = Path(os.environ.get("AGENT_TRADING_BOARD_OUTPUT", "static/agent_trading_live.json"))
RECENT_ACTIVITY_LIMIT = int(os.environ.get("AGENT_TRADING_BOARD_ACTIVITY_LIMIT", "50"))


def _chat_capable_models() -> List[str]:
    models = load_model_configs(ROOT / "configs" / "models.yaml")
    return sorted(name for name, cfg in models.items() if cfg.chat_interface_enabled)


def _open_store(model: str) -> sqlite3.Connection:
    # A model's store may not exist yet (never traded, or this cycle's
    # download failed) -- an empty in-memory store yields a correctly-empty
    # row for that model rather than crashing the whole board build.
    path = STORE_DIR / model / "store.sqlite"
    conn = sqlite3.connect(str(path) if path.exists() else ":memory:")
    conn.row_factory = sqlite3.Row
    benchmark_tools._ensure_account_schema(conn)
    return conn


def _load_model_notes(model: str) -> Dict[str, List[Dict[str, Any]]]:
    path = STORE_DIR / model / "notes.json"
    return benchmark_tools._load_notes(path if path.exists() else None)


def _held_tickers(conn: sqlite3.Connection) -> Set[tuple]:
    return {
        (str(row["platform"] or "kalshi").lower(), str(row["ticker"]))
        for row in conn.execute("SELECT DISTINCT platform, ticker FROM agent_positions WHERE quantity > 0")
    }


def _fetch_quotes(positions: Set[tuple]) -> Dict[Any, Dict[str, Any]]:
    quotes: Dict[Any, Dict[str, Any]] = {}
    for platform, ticker in sorted(positions):
        try:
            quotes[(platform, ticker)] = (
                market_data.fetch_polymarket(slug=ticker)
                if platform == "polymarket"
                else market_data.fetch_kalshi(ticker)
            )
        except market_data.MarketDataError as exc:
            print(f"  could not quote [{platform}] {ticker} for the board: {exc}", file=sys.stderr)
    return quotes


def build_board() -> Dict[str, Any]:
    models = _chat_capable_models()
    conns = {model: _open_store(model) for model in models}
    try:
        held: Set[tuple] = set()
        for conn in conns.values():
            held |= _held_tickers(conn)
        quotes = _fetch_quotes(held)

        leaderboard: List[Dict[str, Any]] = []
        equity_curves: Dict[str, Any] = {}
        eligibility: Dict[str, Any] = {}
        activity: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for model, conn in conns.items():
            rows = agent_trading_stats.compute_agent_leaderboard(conn, quotes)
            leaderboard.extend(rows)
            acct_val = rows[0]["account_value"] if rows else None
            equity = agent_trading_stats.agent_equity_curve(
                conn, model, current_account_value=acct_val, current_ts=now_iso
            )
            equity_curves[model] = equity
            if rows:
                eligibility[model] = agent_trading_stats.compute_promotion_eligibility(rows[0], equity)
            activity.extend(agent_trading_stats.recent_activity(
                conn, _load_model_notes(model), limit=RECENT_ACTIVITY_LIMIT,
            ))
    finally:
        for conn in conns.values():
            conn.close()

    leaderboard.sort(key=lambda r: r["account_value"], reverse=True)
    activity.sort(key=lambda item: str(item.get("ts") or ""), reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow",
        "note": "Paper trading only -- no real money is ever at risk.",
        "models": models,
        "leaderboard": leaderboard,
        "equity_curves": equity_curves,
        "eligibility": eligibility,
        "recent_activity": activity[:RECENT_ACTIVITY_LIMIT],
    }


def main() -> int:
    board = build_board()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(board, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"agent-trading-board built models={len(board['models'])} "
        f"leaderboard_rows={len(board['leaderboard'])} activity={len(board['recent_activity'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
