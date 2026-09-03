#!/usr/bin/env python3
"""Aggregate every agentic-trading model's shadow account into one board JSON.

Each of the eight distinct SCADS models trades through its own isolated
GCS-synced SQLite store (see scripts/agent_trading_tick.py); this script reads
all eight (already
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import agent_trading_stats, benchmark_tools, market_data  # noqa: E402
from analyzing_llm_rationale.config import scads_agent_trading_model_labels  # noqa: E402

STORE_DIR = Path(os.environ.get("AGENT_TRADING_BOARD_STORE_DIR", "tmp/agent-trading-board"))
OUTPUT_PATH = Path(os.environ.get("AGENT_TRADING_BOARD_OUTPUT", "static/agent_trading_live.json"))
RECENT_ACTIVITY_LIMIT = int(os.environ.get("AGENT_TRADING_BOARD_ACTIVITY_LIMIT", "50"))
WEATHER_OPERATIONS_WINDOW_HOURS = max(
    1, min(168, int(os.environ.get("AGENT_TRADING_WEATHER_OPERATIONS_WINDOW_HOURS", "24")))
)
# Agent ticks are scheduled every 15 minutes.  Let a queued GitHub Actions lane
# run for three expected intervals before calling it delayed, and retain a wider
# window before calling the model stalled.  The board's own artifact freshness
# cannot establish either of these facts because another model can publish it.
MODEL_HEALTH_EXPECTED_CYCLE_SECONDS = int(
    os.environ.get("AGENT_TRADING_MODEL_HEALTH_EXPECTED_CYCLE_SECONDS", "900")
)
MODEL_HEALTH_DELAYED_AFTER_SECONDS = int(
    os.environ.get("AGENT_TRADING_MODEL_HEALTH_DELAYED_AFTER_SECONDS", "2700")
)
MODEL_HEALTH_STALE_AFTER_SECONDS = int(
    os.environ.get("AGENT_TRADING_MODEL_HEALTH_STALE_AFTER_SECONDS", "14400")
)


def _agent_trading_models() -> List[str]:
    return list(scads_agent_trading_model_labels(ROOT / "configs" / "models.yaml"))


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


def _latest_thesis(conn: sqlite3.Connection, model: str) -> Dict[str, Any] | None:
    """Return one model's most recent published thesis, independent of feed cap."""
    row = conn.execute(
        "SELECT agent_id, cycle_id, ts, thesis FROM agent_cycles "
        "WHERE agent_id = ? AND thesis IS NOT NULL AND thesis != '' "
        "ORDER BY ts DESC LIMIT 1",
        (model,),
    ).fetchone()
    if row is None:
        return None
    thesis = agent_trading_stats.clean_thesis_display(row["thesis"])
    if not thesis:
        return None
    return {
        "ts": row["ts"],
        "agent_id": row["agent_id"],
        "type": "thesis",
        "cycle_id": row["cycle_id"],
        "thesis": thesis,
    }


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse persisted ISO timestamps without letting a bad row break the board."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _model_health(
    conn: sqlite3.Connection,
    model: str,
    *,
    store_present: bool,
    now: datetime,
) -> Dict[str, Any]:
    """Describe one model's latest confirmed trading cycle and worker state.

    A fresh board artifact only proves that the publisher ran.  ``agent_cycles``
    is instead a per-model success heartbeat written after each completed agent
    turn.  Matching its cycle id against ``agent_actions`` lets the UI state
    clearly distinguish a deliberate no-trade turn from a stalled model.
    """
    cycle = conn.execute(
        "SELECT cycle_id, ts FROM agent_cycles WHERE agent_id = ? ORDER BY ts DESC LIMIT 1",
        (model,),
    ).fetchone()
    account = conn.execute(
        "SELECT updated_at FROM agent_accounts WHERE agent_id = ? LIMIT 1", (model,)
    ).fetchone()
    action = conn.execute(
        "SELECT cycle_id, ts FROM agent_actions WHERE agent_id = ? ORDER BY ts DESC LIMIT 1",
        (model,),
    ).fetchone()
    telemetry = conn.execute(
        """
        SELECT cycle_id, started_at, finished_at, outcome, failure_kind, failure_detail,
               candidate_count, tool_steps, thesis_published, forecast_records,
               weather_candidates_offered, weather_candidates_researched,
               paper_execution_outcome, provider_model, duration_ms
        FROM agent_cycle_telemetry
        WHERE agent_id = ?
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (model,),
    ).fetchone()

    cycle_ts = cycle["ts"] if cycle else None
    cycle_at = _parse_timestamp(cycle_ts)
    age_seconds = max(0, int((now - cycle_at).total_seconds())) if cycle_at else None
    latest_action_cycle_id = action["cycle_id"] if action else None
    action_this_cycle = bool(cycle and latest_action_cycle_id == cycle["cycle_id"])
    telemetry_at = _parse_timestamp(telemetry["started_at"]) if telemetry else None
    latest_attempt_failed = bool(
        telemetry
        and telemetry["outcome"] == "failure"
        and (cycle_at is None or (telemetry_at is not None and telemetry_at >= cycle_at))
    )
    latest_attempt_deferred = bool(
        telemetry
        and telemetry["outcome"] == "deferred"
        and (cycle_at is None or (telemetry_at is not None and telemetry_at >= cycle_at))
    )

    if latest_attempt_deferred:
        status = "provider_paused"
        detail = "Provider status check has temporarily paused this cycle"
        if telemetry["failure_detail"]:
            detail += f": {telemetry['failure_detail']}"
    elif latest_attempt_failed:
        # A failed worker turn is historical execution information.  It does
        # not prove that the model provider is unavailable *now*; a status
        # probe or a later turn may already have recovered.  Keep the cause in
        # ``last_failure_kind`` for maintenance, but use a neutral UI state.
        status = "last_attempt_failed"
        detail = "Latest worker attempt failed"
        if telemetry["failure_detail"]:
            detail += f": {telemetry['failure_detail']}"
    elif cycle_at is None:
        status = "unverified"
        detail = (
            "No confirmed agent cycle has been persisted yet."
            if store_present
            else "No downloaded model ledger is available to verify a cycle."
        )
    elif age_seconds <= MODEL_HEALTH_DELAYED_AFTER_SECONDS:
        status = "active" if action_this_cycle else "no_trade"
        detail = (
            "A confirmed cycle recorded an execution."
            if action_this_cycle
            else "A confirmed cycle completed without an execution."
        )
    elif age_seconds <= MODEL_HEALTH_STALE_AFTER_SECONDS:
        status = "delayed"
        detail = "The most recent confirmed cycle is later than the expected cadence."
    else:
        status = "stale"
        detail = "The most recent confirmed cycle is beyond the stale window."

    return {
        "status": status,
        "detail": detail,
        "last_cycle_at": cycle_ts,
        "last_cycle_id": cycle["cycle_id"] if cycle else None,
        "last_action_at": action["ts"] if action else None,
        "last_account_updated_at": account["updated_at"] if account else None,
        "last_attempt_at": telemetry["started_at"] if telemetry else None,
        "last_attempt_outcome": telemetry["outcome"] if telemetry else None,
        "last_failure_kind": telemetry["failure_kind"] if telemetry else None,
        "last_failure_detail": telemetry["failure_detail"] if telemetry else None,
        "last_duration_ms": telemetry["duration_ms"] if telemetry else None,
        "last_candidate_count": telemetry["candidate_count"] if telemetry else None,
        "last_tool_steps": telemetry["tool_steps"] if telemetry else None,
        "last_thesis_published": bool(telemetry["thesis_published"]) if telemetry else None,
        "last_forecast_records": telemetry["forecast_records"] if telemetry else None,
        "last_weather_candidates_offered": telemetry["weather_candidates_offered"] if telemetry else 0,
        "last_weather_candidates_researched": telemetry["weather_candidates_researched"] if telemetry else 0,
        "last_paper_execution_outcome": telemetry["paper_execution_outcome"] if telemetry else None,
        "last_provider_model": telemetry["provider_model"] if telemetry else None,
        "cycle_age_seconds": age_seconds,
        "expected_cycle_seconds": MODEL_HEALTH_EXPECTED_CYCLE_SECONDS,
        "delayed_after_seconds": MODEL_HEALTH_DELAYED_AFTER_SECONDS,
        "stale_after_seconds": MODEL_HEALTH_STALE_AFTER_SECONDS,
        "store_present": store_present,
    }


def _recent_cycle_telemetry(conn: sqlite3.Connection, model: str, *, limit: int = 5) -> List[Dict[str, Any]]:
    """Return a small structured operational trail for maintenance tooling."""
    return [
        {
            "agent_id": model,
            "run_id": row["run_id"],
            "cycle_id": row["cycle_id"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "outcome": row["outcome"],
            "failure_kind": row["failure_kind"],
            "failure_detail": row["failure_detail"],
            "candidate_count": row["candidate_count"],
            "tool_steps": row["tool_steps"],
            "settled_count": row["settled_count"],
            "thesis_published": bool(row["thesis_published"]),
            "forecast_records": row["forecast_records"],
            "weather_candidates_offered": row["weather_candidates_offered"],
            "weather_candidates_researched": row["weather_candidates_researched"],
            "paper_execution_outcome": row["paper_execution_outcome"],
            "provider_model": row["provider_model"],
            "duration_ms": row["duration_ms"],
        }
        for row in conn.execute(
            """
            SELECT run_id, cycle_id, started_at, finished_at, outcome, failure_kind,
                   failure_detail, candidate_count, tool_steps, settled_count,
                   thesis_published, forecast_records, weather_candidates_offered,
                   weather_candidates_researched, paper_execution_outcome, provider_model, duration_ms
            FROM agent_cycle_telemetry
            WHERE agent_id = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (model, max(1, min(limit, 20))),
        )
    ]


def _operational_health(model_health: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise recent execution outcomes without claiming live availability."""
    last_attempt_failures = [
        {"agent_id": model, "status": health["status"]}
        for model, health in sorted(model_health.items())
        if health.get("status") == "last_attempt_failed"
    ]
    provider_paused = [
        {"agent_id": model, "status": health["status"]}
        for model, health in sorted(model_health.items())
        if health.get("status") == "provider_paused"
    ]
    cycle_failed = [
        {"agent_id": model, "status": health["status"]}
        for model, health in sorted(model_health.items())
        if health.get("status") in {"cycle_error", "unverified", "stale"}
    ]
    confirmed_cycles = sum(
        health.get("status") in {"active", "no_trade"}
        for health in model_health.values()
    )
    if cycle_failed:
        status = "attention"
        detail = f"{len(cycle_failed)} model ledger(s) need maintenance attention."
    elif provider_paused:
        status = "provider_paused"
        detail = (
            f"{len(provider_paused)} model cycle(s) are paused because the public SCADS status check "
            "currently reports them unavailable. They will retry automatically when that status recovers."
        )
    elif last_attempt_failures:
        status = "attempts_failed"
        detail = (
            f"{len(last_attempt_failures)} model agent attempt(s) most recently failed. "
            "This is historical execution status, not a live provider availability check."
        )
    else:
        status = "healthy"
        detail = "All latest model attempts completed without failure."
    return {
        "status": status,
        "detail": detail,
        "models_total": len(model_health),
        # ``models_verified`` is retained for old API clients.  The UI uses
        # the precise name below: a completed worker cycle is not a claim that
        # a model, forecast, or provider has been independently verified.
        "models_verified": confirmed_cycles,
        "models_with_confirmed_cycle": confirmed_cycles,
        "last_attempt_failures": last_attempt_failures,
        "provider_paused": provider_paused,
        "attention_required": cycle_failed,
    }


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


def _weather_operations(conns: Dict[str, sqlite3.Connection], now: datetime) -> Dict[str, int]:
    """Aggregate a compact, audit-friendly weather funnel across model ledgers."""
    window_start = (now - timedelta(hours=WEATHER_OPERATIONS_WINDOW_HOURS)).isoformat()
    counts = {key: 0 for key in ("candidates_offered", "researched", "forecasted", "traded", "resolved")}
    for conn in conns.values():
        telemetry = conn.execute(
            """
            SELECT COALESCE(SUM(weather_candidates_offered), 0) AS candidates_offered,
                   COALESCE(SUM(weather_candidates_researched), 0) AS researched
            FROM agent_cycle_telemetry
            WHERE started_at >= ?
            """,
            (window_start,),
        ).fetchone()
        counts["candidates_offered"] += int(telemetry["candidates_offered"] or 0)
        counts["researched"] += int(telemetry["researched"] or 0)
        forecasts = conn.execute(
            """
            SELECT COUNT(*) AS forecasted,
                   SUM(CASE WHEN resolved_outcome IS NOT NULL AND resolved_at >= ? THEN 1 ELSE 0 END) AS resolved
            FROM agent_thesis_forecasts
            WHERE weather_market_type IS NOT NULL AND forecast_ts >= ?
            """,
            (window_start, window_start),
        ).fetchone()
        counts["forecasted"] += int(forecasts["forecasted"] or 0)
        counts["resolved"] += int(forecasts["resolved"] or 0)
        trades = conn.execute(
            """
            SELECT COUNT(DISTINCT action.id) AS traded
            FROM agent_actions AS action
            JOIN agent_thesis_forecasts AS forecast
              ON forecast.agent_id = action.agent_id
             AND forecast.platform = action.platform
             AND forecast.ticker = action.ticker
            WHERE action.action_type = 'trade'
              AND (action.quantity IS NULL OR action.quantity > 0)
              AND action.ts >= ?
              AND forecast.weather_market_type IS NOT NULL
            """,
            (window_start,),
        ).fetchone()
        counts["traded"] += int(trades["traded"] or 0)
    return {"window_hours": WEATHER_OPERATIONS_WINDOW_HOURS, **counts}


def build_board() -> Dict[str, Any]:
    models = _agent_trading_models()
    store_presence = {model: (STORE_DIR / model / "store.sqlite").exists() for model in models}
    conns = {model: _open_store(model) for model in models}
    try:
        held: Set[tuple] = set()
        for conn in conns.values():
            held |= _held_tickers(conn)
        quotes = _fetch_quotes(held)

        leaderboard: List[Dict[str, Any]] = []
        equity_curves: Dict[str, Any] = {}
        eligibility: Dict[str, Any] = {}
        forecast_learning: Dict[str, Any] = {}
        latest_theses: Dict[str, Dict[str, Any] | None] = {}
        model_health: Dict[str, Dict[str, Any]] = {}
        cycle_telemetry: List[Dict[str, Any]] = []
        activity: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        weather_operations = _weather_operations(conns, now)
        for model, conn in conns.items():
            rows = agent_trading_stats.compute_agent_leaderboard(conn, quotes)
            leaderboard.extend(rows)
            acct_val = rows[0]["account_value"] if rows else None
            equity = agent_trading_stats.agent_equity_curve(
                conn, model, current_account_value=acct_val, current_ts=now_iso
            )
            equity_curves[model] = equity
            forecast_learning[model] = agent_trading_stats.compute_forecast_learning(conn, model)
            latest_theses[model] = _latest_thesis(conn, model)
            model_health[model] = _model_health(
                conn, model, store_present=store_presence[model], now=now,
            )
            cycle_telemetry.extend(_recent_cycle_telemetry(conn, model))
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
    cycle_telemetry.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow",
        "note": "Paper trading only -- no real money is ever at risk.",
        "models": models,
        "leaderboard": leaderboard,
        "equity_curves": equity_curves,
        "eligibility": eligibility,
        "forecast_learning": forecast_learning,
        "weather_operations": weather_operations,
        "model_health": model_health,
        "operational_health": _operational_health(model_health),
        "recent_cycle_telemetry": cycle_telemetry[:RECENT_ACTIVITY_LIMIT],
        "latest_theses": latest_theses,
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
