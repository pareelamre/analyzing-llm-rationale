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


def _latest_reset_ts(conn: sqlite3.Connection, agent_id: str) -> Any:
    row = conn.execute(
        "SELECT ts FROM agent_actions WHERE agent_id = ? AND action_type = 'admin_reset' "
        "ORDER BY ts DESC LIMIT 1",
        (agent_id,),
    ).fetchone()
    return row["ts"] if row else None


def compute_agent_leaderboard(conn: sqlite3.Connection, quotes: QuoteMap) -> List[Dict[str, Any]]:
    """One row per agent account in ``conn``, mark-to-market valued against
    ``quotes`` (keyed ``(platform, ticker)`` -- platform lowercase, e.g.
    ``("kalshi", "KXFOO-26")``, matching how ``agent_positions.platform`` is
    stored, regardless of what casing a raw quote dict's own field carries).

    Win rate is computed over *settled* markets only (``action_type =
    'settlement'`` rows) -- rejected trades never reach the exchange and
    open positions haven't resolved yet, so neither belongs in a win/loss
    count.

    Like ``agent_equity_curve``, ``trade_count``/``settled_count``/
    ``won_count``/``win_rate`` only cover activity since the agent's latest
    ``admin_reset`` (all of it, if there hasn't been one) -- otherwise these
    would keep counting pre-reset trades the equity chart no longer shows,
    silently disagreeing with it.
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

        since_ts = _latest_reset_ts(conn, agent_id)
        trade_sql = "SELECT COUNT(*) FROM agent_actions WHERE agent_id = ? AND action_type = 'trade'"
        settlement_sql = (
            "SELECT realized_pnl FROM agent_actions WHERE agent_id = ? AND action_type = 'settlement'"
        )
        params: List[Any] = [agent_id]
        if since_ts is not None:
            trade_sql += " AND ts >= ?"
            settlement_sql += " AND ts >= ?"
            params.append(since_ts)

        trade_count = conn.execute(trade_sql, params).fetchone()[0]
        settlement_pnls = [float(r[0]) for r in conn.execute(settlement_sql, params)]
        settled_count = len(settlement_pnls)
        won_count = sum(1 for pnl in settlement_pnls if pnl > 0)
        win_rate = (won_count / settled_count) if settled_count else None

        starting_cash = float(acct_row["starting_cash"])
        total_pnl = snap["account_value"] - starting_cash
        return_pct = (
            (total_pnl / starting_cash) * 100.0
            if starting_cash > 0 else 0.0
        )
        rows.append({
            "agent_id": agent_id,
            "starting_cash": round(starting_cash, 2),
            "cash": snap["cash"],
            "account_value": snap["account_value"],
            "total_pnl": round(total_pnl, 6),
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


def agent_equity_curve(
    conn: sqlite3.Connection,
    agent_id: str,
    *,
    current_account_value: float | None = None,
    current_ts: str | None = None,
) -> Dict[str, Any]:
    """Portfolio equity curve for one agent, plus Sharpe/max-drawdown
    computed over it.

    Tracks total portfolio book value (cash + open positions basis) across
    `agent_actions` (trades convert cash to position basis with net impact
    being only the transaction fee, settlements realize P&L and return cash,
    and admin resets/corrections adjust the account).

    If `current_account_value` is provided (e.g. from the live mark-to-market
    snapshot in `compute_agent_leaderboard`), the latest point reflects the
    mark-to-market portfolio value.
    """
    acct_row = conn.execute(
        "SELECT starting_cash, cash, updated_at FROM agent_accounts WHERE agent_id = ?", (agent_id,),
    ).fetchone()
    starting_cash = float(acct_row["starting_cash"]) if acct_row else 0.0

    running_cash = starting_cash
    open_positions_basis = 0.0
    points: List[Dict[str, Any]] = [{
        "ts": None,
        "account_value": round(starting_cash, 6),
        "event_type": "starting_cash",
    }]
    for row in conn.execute(
        "SELECT ts, action_type, cash_delta, fee, notional, payout, realized_pnl, outcome FROM agent_actions "
        "WHERE agent_id = ? ORDER BY ts ASC",
        (agent_id,),
    ):
        act = row["action_type"]
        if act == "admin_reset":
            running_cash = starting_cash
            open_positions_basis = 0.0
            account_val = running_cash
        elif act == "trade":
            notional = float(row["notional"] or 0)
            cash_delta = float(row["cash_delta"] or 0)
            running_cash += cash_delta
            if row["outcome"] == "realized":
                # A closing/netting trade (buying the opposite side of a
                # held position -- the only way to exit here) is ALSO
                # recorded as action_type='trade' with a positive notional
                # (the closing order's own dollar size), not as a
                # 'settlement'. Regression: that notional was being added to
                # open_positions_basis exactly like an opening trade,
                # double-counting every single close -- once from the
                # original open, again from the close -- and inflating the
                # reported account value by the closing order's full
                # notional on every exit. Mirrors the settlement branch's
                # cost-basis-closed formula below, using cash_delta in place
                # of payout since a netting close has no payout column.
                realized_pnl = float(row["realized_pnl"] or 0)
                cost_basis_closed = max(0.0, cash_delta - realized_pnl)
                open_positions_basis = max(0.0, open_positions_basis - cost_basis_closed)
            elif notional > 0:
                open_positions_basis += notional
            account_val = running_cash + open_positions_basis
        elif act == "settlement":
            payout = float(row["payout"] or 0)
            realized_pnl = float(row["realized_pnl"] or 0)
            cost_basis_closed = max(0.0, payout - realized_pnl) if payout or realized_pnl else 0.0
            cash_delta = float(row["cash_delta"] or payout)
            running_cash += cash_delta
            open_positions_basis = max(0.0, open_positions_basis - cost_basis_closed)
            account_val = running_cash + open_positions_basis
        elif act == "admin_correction":
            running_cash += float(row["cash_delta"] or 0)
            account_val = running_cash + open_positions_basis
        else:
            account_val = running_cash + open_positions_basis

        points.append({
            "ts": row["ts"],
            "account_value": round(account_val, 6),
            "event_type": row["action_type"],
        })

    reset_indices = [i for i, p in enumerate(points) if p["event_type"] == "admin_reset"]
    if reset_indices:
        points = points[reset_indices[-1]:]

    if current_account_value is not None:
        c_val = round(float(current_account_value), 6)
        ts_to_use = current_ts or (acct_row["updated_at"] if acct_row else None)
        if points:
            last_p = points[-1]
            if last_p.get("ts") == ts_to_use:
                last_p["account_value"] = c_val
            elif ts_to_use and (not last_p.get("ts") or str(last_p["ts"]) < str(ts_to_use)):
                points.append({
                    "ts": ts_to_use,
                    "account_value": c_val,
                    "event_type": "mark_to_market",
                })
            else:
                last_p["account_value"] = c_val

    risk = _sharpe_and_max_drawdown(points)
    return {
        "agent_id": agent_id,
        "value_curve": points,
        "sharpe": risk["sharpe"],
        "max_drawdown": risk["max_drawdown"],
    }


# Reasoned, adjustable starting thresholds for shadow-trading "promotion
# eligibility" -- there is no real-money consequence to calibrate against
# yet, so these are deliberately conservative rather than tuned:
#   - 30 settled trades is a common rough floor for a Sharpe/win-rate
#     estimate to start meaning more than noise.
#   - Sharpe >= 0.5 is comfortably above "indistinguishable from luck"
#     (~0) while well short of "strong" (1.0+) -- a low bar on purpose.
#   - 25% max drawdown is a common early-stage risk ceiling for vetting a
#     new strategy, well above the 15% single-market concentration cap
#     already enforced per-trade (a different thing: that's a position
#     limit, this is an observed historical portfolio drawdown).
PROMOTION_MIN_SETTLED_TRADES = 30
PROMOTION_MIN_SHARPE = 0.5
PROMOTION_MAX_DRAWDOWN = 0.25


def compute_promotion_eligibility(
    leaderboard_row: Mapping[str, Any], equity: Mapping[str, Any]
) -> Dict[str, Any]:
    """Whether one agent's shadow-trading track record currently clears a
    conservative, adjustable bar for "worth a human's attention as a
    promotion candidate."

    Purely observational -- this reports a verdict, it does not grant or
    restrict anything. There is no "more autonomy" lever wired to this yet;
    any future graduation to real trading must still go through Trade Runs,
    guardrails, and explicit human confirmation regardless of this verdict.
    """
    settled_count = int(leaderboard_row.get("settled_count") or 0)
    return_pct = leaderboard_row.get("return_pct")
    sharpe = equity.get("sharpe")
    max_drawdown = equity.get("max_drawdown")

    checks = {
        "sufficient_sample": settled_count >= PROMOTION_MIN_SETTLED_TRADES,
        "positive_return": (return_pct or 0) > 0,
        "sharpe_above_floor": sharpe is not None and sharpe >= PROMOTION_MIN_SHARPE,
        "drawdown_within_cap": max_drawdown is not None and max_drawdown <= PROMOTION_MAX_DRAWDOWN,
    }
    return {
        "agent_id": leaderboard_row.get("agent_id"),
        "eligible": all(checks.values()),
        "checks": checks,
        "settled_count": settled_count,
        "min_settled_trades_required": PROMOTION_MIN_SETTLED_TRADES,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "min_sharpe_required": PROMOTION_MIN_SHARPE,
        "max_drawdown": max_drawdown,
        "max_drawdown_allowed": PROMOTION_MAX_DRAWDOWN,
    }


def clean_thesis_display(raw_thesis: Optional[str]) -> str:
    """Normalize and format an agent's thesis for display on the trading board.

    Extracts substantive thought/rationale if the model produced a JSON envelope
    (e.g. `{"thought": "...", "action": "..."}` or `{"final": "..."}`).
    """
    if not raw_thesis:
        return ""
    text = str(raw_thesis).strip()
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("```json") and text.endswith("```")):
        clean_json = text
        if text.startswith("```json"):
            clean_json = text.strip("`").removeprefix("json").strip()
        try:
            parsed = json.loads(clean_json)
            if isinstance(parsed, dict):
                thought = parsed.get("thought") or parsed.get("reasoning") or parsed.get("analysis")
                final = parsed.get("final") or parsed.get("answer") or parsed.get("thesis")
                if final and thought:
                    return f"{final}\n\n**Detailed Analysis:**\n{thought}".strip()
                if final:
                    return str(final).strip()
                if thought:
                    return str(thought).strip()
        except Exception:
            pass
    return text


def recent_activity(
    conn: sqlite3.Connection,
    notes_by_agent: Mapping[str, List[Mapping[str, Any]]],
    *,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Merge trades/settlements, per-cycle theses, and notes into one
    newest-first transparency feed, capped at ``limit``.

    Manual account adjustments (``admin_correction``/``admin_reset``, see
    scripts/reset_agent_trading_accounts.py) are included too -- an
    adjustment that changed an agent's balance shouldn't be invisible in the
    one feed meant to show what happened to that balance and why."""
    items: List[Dict[str, Any]] = []

    for row in conn.execute(
        "SELECT ts, agent_id, action_type, mode, platform, ticker, side, "
        "quantity, price, realized_pnl, outcome FROM agent_actions "
        "WHERE action_type IN "
        "('trade', 'settlement', 'rejected_trade', 'admin_correction', 'admin_reset') "
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
            "thesis": clean_thesis_display(row["thesis"]),
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
