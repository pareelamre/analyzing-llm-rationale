"""Pure read-side functions for the agentic shadow-trading board.

Unlike the deterministic Kelly-sizing ledgers in ``track_record_live.py``,
these summarise ``benchmark_tools.py``'s real cash/position accounting for
models that were given genuine tool-use trading agency (see
``scripts/agent_trading_tick.py``). Every function here takes an already-open
sqlite3 connection and does no I/O of its own, so it's unit-testable against a
hand-built fixture without a server or network access.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Mapping, Tuple

from analyzing_llm_rationale.accounting import Position, PredictionMarketAccount
from analyzing_llm_rationale.track_record_live import _sharpe_and_max_drawdown

QuoteMap = Mapping[Tuple[str, str], Any]


def _account_from_rows(acct_row: sqlite3.Row, position_rows: List[sqlite3.Row]) -> PredictionMarketAccount:
    account = PredictionMarketAccount(starting_cash=float(acct_row["starting_cash"]))
    account.cash = float(acct_row["cash"])
    account.realized_pnl = float(acct_row["realized_pnl"])
    account.fees_paid = float(acct_row["fees_paid"]) + float(acct_row["settlement_fees_paid"])
    for pos in position_rows:
        side = str(pos["side"]).upper()
        key = (str(pos["platform"]), str(pos["ticker"]), side)
        account.positions[key] = Position(
            platform=str(pos["platform"]),
            ident=str(pos["ticker"]),
            side=side,
            quantity=float(pos["quantity"]),
            cost_basis=float(pos["cost_basis"]),
        )
    return account


def compute_agent_leaderboard(conn: sqlite3.Connection, quotes: QuoteMap) -> List[Dict[str, Any]]:
    """One row per agent account in ``conn``, mark-to-market valued against
    ``quotes`` (keyed ``(platform, ticker)`` -- platform lowercase, e.g.
    ``("kalshi", "KXFOO-26")``, matching how ``agent_positions.platform`` is
    stored, regardless of what casing a raw quote dict's own field carries).

    Win rate is computed over *settled* markets only (``action_type =
    'settlement'`` rows) -- rejected trades never reach the exchange and
    open positions haven't resolved yet, so neither belongs in a win/loss
    count.
    """
    rows: List[Dict[str, Any]] = []
    for acct_row in conn.execute("SELECT * FROM agent_accounts ORDER BY agent_id"):
        agent_id = str(acct_row["agent_id"])
        position_rows = list(conn.execute(
            "SELECT platform, ticker, side, quantity, cost_basis "
            "FROM agent_positions WHERE agent_id = ? AND quantity > 0",
            (agent_id,),
        ))
        account = _account_from_rows(acct_row, position_rows)
        snap = account.snapshot(quotes)

        trade_count = conn.execute(
            "SELECT COUNT(*) FROM agent_actions WHERE agent_id = ? AND action_type = 'trade'",
            (agent_id,),
        ).fetchone()[0]
        settlement_pnls = [
            float(r[0]) for r in conn.execute(
                "SELECT realized_pnl FROM agent_actions "
                "WHERE agent_id = ? AND action_type = 'settlement'",
                (agent_id,),
            )
        ]
        settled_count = len(settlement_pnls)
        won_count = sum(1 for pnl in settlement_pnls if pnl > 0)
        win_rate = (won_count / settled_count) if settled_count else None

        starting_cash = float(acct_row["starting_cash"])
        return_pct = (
            ((snap["account_value"] - starting_cash) / starting_cash) * 100.0
            if starting_cash > 0 else 0.0
        )
        rows.append({
            "agent_id": agent_id,
            "starting_cash": round(starting_cash, 2),
            "cash": snap["cash"],
            "account_value": snap["account_value"],
            "unrealized_pnl": snap["unrealized_pnl"],
            "realized_pnl": snap["realized_pnl"],
            "fees_paid": round(account.fees_paid, 6),
            "return_pct": round(return_pct, 4),
            "open_positions": snap["open_positions"],
            "illiquid_positions": snap["illiquid_positions"],
            "trade_count": trade_count,
            "settled_count": settled_count,
            "won_count": won_count,
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
            "updated_at": acct_row["updated_at"],
        })
    rows.sort(key=lambda r: r["account_value"], reverse=True)
    return rows


def agent_equity_curve(conn: sqlite3.Connection, agent_id: str) -> Dict[str, Any]:
    """Cash-only book-value curve for one agent, plus Sharpe/max-drawdown
    computed over it.

    This is an explicit approximation, not a true mark-to-market curve: each
    point is running cash after one ``agent_actions`` row (starting from
    ``starting_cash``), so it does *not* credit the current worth of open
    positions -- opening a trade dips the curve by its full cost, and it only
    recovers once that position settles and cash comes back in. True
    historical bid-marks for every past cycle aren't stored, so this proxy
    trades precision for being exactly reconstructable from data that is.

    Only ``trade``/``settlement`` rows actually move cash. A rejected order
    never reaches the exchange, so its row is included as a zero-height
    marker rather than trusted for ``cash_delta`` -- defends against old rows
    written before a bug fix stored the rejected order's hypothetical delta
    there instead of 0, which cratered this curve for every future point.
    """
    CASH_MOVING_ACTION_TYPES = {"trade", "settlement"}

    acct_row = conn.execute(
        "SELECT starting_cash FROM agent_accounts WHERE agent_id = ?", (agent_id,),
    ).fetchone()
    starting_cash = float(acct_row["starting_cash"]) if acct_row else 0.0

    running_cash = starting_cash
    points: List[Dict[str, Any]] = [{
        "ts": None,
        "account_value": round(running_cash, 6),
        "event_type": "starting_cash",
    }]
    for row in conn.execute(
        "SELECT ts, action_type, cash_delta FROM agent_actions "
        "WHERE agent_id = ? ORDER BY ts ASC",
        (agent_id,),
    ):
        if row["action_type"] in CASH_MOVING_ACTION_TYPES:
            running_cash += float(row["cash_delta"])
        points.append({
            "ts": row["ts"],
            "account_value": round(running_cash, 6),
            "event_type": row["action_type"],
        })

    risk = _sharpe_and_max_drawdown(points)
    return {
        "agent_id": agent_id,
        "value_curve": points,
        "sharpe": risk["sharpe"],
        "max_drawdown": risk["max_drawdown"],
    }


def recent_activity(
    conn: sqlite3.Connection,
    notes_by_agent: Mapping[str, List[Mapping[str, Any]]],
    *,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Merge trades/settlements, per-cycle theses, and notes into one
    newest-first transparency feed, capped at ``limit``."""
    items: List[Dict[str, Any]] = []

    for row in conn.execute(
        "SELECT ts, agent_id, action_type, mode, platform, ticker, side, "
        "quantity, price, realized_pnl, outcome FROM agent_actions "
        "WHERE action_type IN ('trade', 'settlement', 'rejected_trade') "
        "ORDER BY ts DESC LIMIT ?",
        (limit,),
    ):
        items.append({
            "ts": row["ts"],
            "agent_id": row["agent_id"],
            "type": row["action_type"],
            "mode": row["mode"],
            "platform": row["platform"],
            "ticker": row["ticker"],
            "side": row["side"],
            "quantity": row["quantity"],
            "price": row["price"],
            "realized_pnl": row["realized_pnl"],
            "outcome": row["outcome"],
        })

    for row in conn.execute(
        "SELECT agent_id, cycle_id, ts, thesis FROM agent_cycles "
        "WHERE thesis IS NOT NULL AND thesis != '' ORDER BY ts DESC LIMIT ?",
        (limit,),
    ):
        items.append({
            "ts": row["ts"],
            "agent_id": row["agent_id"],
            "type": "thesis",
            "cycle_id": row["cycle_id"],
            "thesis": row["thesis"],
        })

    for agent_id, notes in notes_by_agent.items():
        for note in notes:
            items.append({
                "ts": note.get("updated_at") or note.get("created_at"),
                "agent_id": agent_id,
                "type": "note",
                "text": note.get("text"),
                "tags": note.get("tags") or [],
            })

    items.sort(key=lambda item: str(item.get("ts") or ""), reverse=True)
    return items[:limit]
