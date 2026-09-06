"""Pure read-side functions for the agentic shadow-trading board.

Unlike the deterministic Kelly-sizing ledgers in ``track_record_live.py``,
these summarise ``benchmark_tools.py``'s real cash/position accounting for
models that were given genuine tool-use trading agency (see
``scripts/agent_trading_tick.py``). Every function here takes an already-open
sqlite3 connection and does no I/O of its own, so it's unit-testable against a
hand-built fixture without a server or network access.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Tuple

from analyzing_llm_rationale.accounting import (
    MIN_POSITION_QUANTITY,
    Position,
    PredictionMarketAccount,
)
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

    Win rate is computed over every *realized* outcome: final market
    settlements (``action_type = 'settlement'``) and voluntary netting exits
    (``action_type = 'trade'`` with ``outcome = 'realized'``). The latter
    are genuine closed P&L decisions; excluding them made the public board
    show an empty or one-outcome win rate even after an agent had closed
    several positions. Rejected trades and open positions remain excluded.

    ``settled_count`` intentionally remains the count of final market
    settlements only, so the promotion-sample guard still requires outcomes
    that ran to the contract's stated resolution rather than allowing rapid
    exit churn to satisfy that separate control.

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
            "FROM agent_positions WHERE agent_id = ? AND quantity > ?",
            (agent_id, MIN_POSITION_QUANTITY),
        ))
        account = _account_from_rows(acct_row, position_rows)
        snap = account.snapshot(quotes)

        since_ts = _latest_reset_ts(conn, agent_id)
        trade_sql = (
            "SELECT COUNT(*) FROM agent_actions WHERE agent_id = ? AND action_type = 'trade' "
            "AND (quantity IS NULL OR quantity > 0 OR outcome = 'realized')"
        )
        settlement_sql = (
            "SELECT realized_pnl FROM agent_actions WHERE agent_id = ? AND action_type = 'settlement'"
        )
        realized_sql = (
            "SELECT realized_pnl FROM agent_actions WHERE agent_id = ? AND "
            "(action_type = 'settlement' OR (action_type = 'trade' AND outcome = 'realized'))"
        )
        params: List[Any] = [agent_id]
        if since_ts is not None:
            trade_sql += " AND ts >= ?"
            settlement_sql += " AND ts >= ?"
            realized_sql += " AND ts >= ?"
            params.append(since_ts)

        trade_count = conn.execute(trade_sql, params).fetchone()[0]
        settlement_pnls = [float(r[0]) for r in conn.execute(settlement_sql, params)]
        realized_pnls = [float(r[0]) for r in conn.execute(realized_sql, params)]
        settled_count = len(settlement_pnls)
        realized_count = len(realized_pnls)
        won_count = sum(1 for pnl in realized_pnls if pnl > 0)
        win_rate = (won_count / realized_count) if realized_count else None

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
            "mark_coverage": snap["mark_coverage"],
            "trade_count": trade_count,
            "settled_count": settled_count,
            "realized_count": realized_count,
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


def compute_forecast_learning(
    conn: sqlite3.Connection, agent_id: str, *, review_limit: int = 3
) -> Dict[str, Any]:
    """Summarize only final-outcome-scored thesis forecasts for one agent.

    This is deliberately independent of P&L: a forecast is informative when
    the contract's final outcome is known, not when a position happens to be
    marked up or voluntarily closed.  ``market_brier_score`` can be absent on
    older/fallback records, so the comparison is optional rather than guessed.
    """
    total_row = conn.execute(
        """
        SELECT COUNT(*) AS recorded_forecasts,
               SUM(CASE WHEN resolved_outcome IS NOT NULL THEN 1 ELSE 0 END) AS resolved_forecasts,
               AVG(CASE WHEN resolved_outcome IS NOT NULL THEN brier_score END) AS brier_score,
               AVG(CASE WHEN resolved_outcome IS NOT NULL THEN market_brier_score END) AS market_brier_score,
               AVG(CASE WHEN resolved_outcome IS NOT NULL
                        THEN model_probability - resolved_outcome END) AS probability_bias
        FROM agent_thesis_forecasts
        WHERE agent_id = ?
        """,
        (agent_id,),
    ).fetchone()
    recorded = int(total_row["recorded_forecasts"] or 0)
    resolved = int(total_row["resolved_forecasts"] or 0)
    brier = total_row["brier_score"]
    market_brier = total_row["market_brier_score"]
    bias = total_row["probability_bias"]
    reviews = [
        {
            "ticker": row["ticker"],
            "platform": row["platform"],
            "forecast_ts": row["forecast_ts"],
            "resolved_at": row["resolved_at"],
            "model_probability": round(float(row["model_probability"]), 4),
            "market_probability": (
                round(float(row["market_probability"]), 4)
                if row["market_probability"] is not None else None
            ),
            "resolved_outcome": int(row["resolved_outcome"]),
            "brier_score": round(float(row["brier_score"]), 6),
            "market_brier_score": (
                round(float(row["market_brier_score"]), 6)
                if row["market_brier_score"] is not None else None
            ),
        }
        for row in conn.execute(
            """
            SELECT platform, ticker, forecast_ts, resolved_at, model_probability,
                   market_probability, resolved_outcome, brier_score, market_brier_score
            FROM agent_thesis_forecasts
            WHERE agent_id = ? AND resolved_outcome IS NOT NULL
            ORDER BY resolved_at DESC
            LIMIT ?
            """,
            (agent_id, max(1, min(int(review_limit), 10))),
        )
    ]
    # Weather contracts settle against an explicit source, so do not merge them
    # into a single generic score.  A small, source-labelled cohort gives the
    # agent feedback that is useful for calibration without pretending a few
    # outcomes establish a strategy edge.
    weather_calibration = [
        {
            "market_type": str(row["weather_market_type"]),
            "settlement_source": str(row["weather_settlement_source"] or "unspecified"),
            "resolved_forecasts": int(row["resolved_forecasts"] or 0),
            "brier_score": round(float(row["brier_score"]), 6),
            "market_brier_score": (
                round(float(row["market_brier_score"]), 6)
                if row["market_brier_score"] is not None else None
            ),
            "probability_bias": (
                round(float(row["probability_bias"]), 6)
                if row["probability_bias"] is not None else None
            ),
        }
        for row in conn.execute(
            """
            SELECT weather_market_type, weather_settlement_source,
                   COUNT(*) AS resolved_forecasts,
                   AVG(brier_score) AS brier_score,
                   AVG(market_brier_score) AS market_brier_score,
                   AVG(model_probability - resolved_outcome) AS probability_bias
            FROM agent_thesis_forecasts
            WHERE agent_id = ?
              AND resolved_outcome IS NOT NULL
              AND weather_market_type IS NOT NULL
            GROUP BY weather_market_type, weather_settlement_source
            ORDER BY resolved_forecasts DESC, weather_market_type, weather_settlement_source
            LIMIT 4
            """,
            (agent_id,),
        )
    ]
    if not recorded:
        status = "not_recording"
    elif not resolved:
        status = "collecting_outcomes"
    elif resolved < 5:
        status = "small_sample"
    else:
        status = "learning"
    return {
        "agent_id": agent_id,
        "status": status,
        "recorded_forecasts": recorded,
        "resolved_forecasts": resolved,
        "brier_score": round(float(brier), 6) if brier is not None else None,
        "market_brier_score": round(float(market_brier), 6) if market_brier is not None else None,
        "probability_bias": round(float(bias), 6) if bias is not None else None,
        "recent_reviews": reviews,
        "weather_calibration": weather_calibration,
    }


def _salvage_final_text(text: str) -> str:
    """Pull the final/answer copy out of a malformed or truncated envelope.

    Returns "" unless a final field is present with real prose, so a bare tool
    envelope (thought + action only) still yields nothing reader-facing.
    """
    match = re.search(r'"(?:final|answer|thesis)"\s*:\s*"', text)
    if not match:
        return ""
    body = text[match.end():]
    out: list[str] = []
    escape = False
    for ch in body:
        if escape:
            # Only the escapes a model actually emits in prose; anything else
            # passes through as written rather than corrupting the text.
            out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(ch, ch))
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            break  # closing quote of a complete final field
        out.append(ch)
    return "".join(out).strip()


def clean_thesis_display(raw_thesis: Optional[str]) -> str:
    """Normalize and format an agent's thesis for display on the trading board.

    A reader-facing card only accepts a model's explicit final/thesis field.
    Tool calls and private reasoning remain in the durable transcript for audit,
    but are never promoted into public board copy.
    """
    if not raw_thesis:
        return ""
    text = str(raw_thesis).strip()
    # Old cycles may predate the server-side publication guard. Their ReAct
    # traces are useful in the durable run record, but a card must never render
    # raw {thought, action, args} tool envelopes as a thesis.
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("```json") and text.endswith("```")):
        clean_json = text
        if text.startswith("```json"):
            clean_json = text.strip("`").removeprefix("json").strip()
        try:
            parsed = json.loads(clean_json)
            if isinstance(parsed, dict):
                final = parsed.get("final") or parsed.get("answer") or parsed.get("thesis")
                if final:
                    text = str(final).strip()
                else:
                    return ""
        except Exception:
            # Malformed/truncated envelope. A provider is often cut off *after*
            # writing most of its final copy, so discarding the whole payload
            # loses a thesis the model actually wrote and leaves a blank card.
            # Recover the final text when it is unambiguously present; only
            # fall back to dropping the payload (it stays in the transcript)
            # when there is no readable final section at all.
            salvaged = _salvage_final_text(text)
            if salvaged:
                text = salvaged
            elif text.startswith("{"):
                return ""
    # A provider may be interrupted while serialising its final/tool payload,
    # leaving an unterminated JSON object such as ``{"thought": ...``. Such a
    # payload never reaches the parse branch above (it has no closing brace),
    # so try the same recovery here before dropping it to the transcript.
    if text.startswith("{"):
        salvaged = _salvage_final_text(text)
        if salvaged:
            text = salvaged
    if text.startswith("{") or re.search(
        r'\{\s*"(?:thought|reasoning|analysis)"\s*:\s*.*?"(?:action|args|query)"\s*:',
        text,
        re.DOTALL,
    ):
        return ""
    if text.upper() in {"PASS", "HOLD", "NO ACTION", "N/A"}:
        # Say what happened rather than rendering an empty card. A bare verdict
        # carries no reasoning, but blanking it loses the fact that the model
        # answered at all -- llama-3.3-70b-instruct produced exactly this five
        # times in two days and the board showed nothing whatsoever.
        return (
            f"**{text.upper()}** — no trade this cycle. The model returned only "
            "this verdict, without stating the reasoning behind it."
        )
    # Strip any preamble or conversational thoughts appearing before the template.
    start_match = re.search(
        r"(?im)^(?:#{1,6}\s*)?(?:0\.\s*Research\s+Delta|1\.\s*Decision\s*&\s*Execution)\b",
        text,
    )
    if start_match and start_match.start() > 0:
        text = text[start_match.start():].strip()

    # A provider can append a second entire final template after the first
    # one. It is not a second audited decision, so preserve the first complete
    # thesis rather than rendering conflicting actions in one card.
    duplicate_template = list(re.finditer(r"(?im)^###\s*0\.\s*research\s+delta\b", text))
    if len(duplicate_template) > 1:
        text = text[:duplicate_template[1].start()].rstrip()

    # If text is unstructured raw deliberation without standard template sections
    # (e.g. "Let me reconsider my analysis..."), avoid rendering thousands of characters
    # of raw scratchpad to the user.
    if not any(k in text.lower() for k in ("research delta", "decision & execution", "model edge")):
        if re.search(r"(?i)\b(?:let me (?:reconsider|evaluate|think)|scratchpad|candidate markets)\b", text) or len(text) > 500:
            return (
                "**Research cycle completed without standard thesis template** — "
                "The model returned internal deliberation instead of the concise thesis card. "
                "The private tool record is retained for audit."
            )

    return text


def _normalized_thesis_text(value: str) -> str:
    """Compare theses independent of markdown and whitespace presentation."""
    return re.sub(r"\s+", " ", re.sub(r"[*_`#]", "", value).lower()).strip()


def _same_thesis_content(left: str, right: str) -> bool:
    left_normalized = _normalized_thesis_text(left)
    right_normalized = _normalized_thesis_text(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    shorter, longer = sorted((left_normalized, right_normalized), key=len)
    return len(shorter) >= 80 and shorter in longer


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
        (limit * 2,),
    ):
        action_type = str(row["action_type"] or "")
        quantity = row["quantity"]
        # IOC simulation attempts with no executable fill are retained in the
        # ledger, but they did not open or close a position. Calling them
        # trades on the public feed was misleading.
        try:
            zero_fill = quantity is not None and float(quantity) <= 0
        except (TypeError, ValueError):
            zero_fill = False
        if action_type == "trade" and zero_fill:
            action_type = "unfilled_order"
        items.append({
            "ts": row["ts"],
            "agent_id": row["agent_id"],
            "type": action_type,
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
        (limit * 2,),
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
    visible: List[Dict[str, Any]] = []
    last_thesis_by_agent: Dict[str, str] = {}
    for item in items:
        if item.get("type") == "thesis":
            agent_id = str(item.get("agent_id") or "")
            thesis = str(item.get("thesis") or "")
            previous = last_thesis_by_agent.get(agent_id)
            if previous and _same_thesis_content(previous, thesis):
                continue
            last_thesis_by_agent[agent_id] = thesis
        visible.append(item)
        if len(visible) >= limit:
            break
    return visible
