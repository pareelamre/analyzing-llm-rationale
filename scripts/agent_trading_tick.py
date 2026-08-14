#!/usr/bin/env python3
"""Run one autonomous shadow-trading cycle for one SCADS model.

Unlike the plain forecasting tick (``track_record_tick.py``), this doesn't just
score a probability -- it gives the model real tool access (``place_trade``,
``web_search``, ``manage_notes``) via ``/agent/analyze``'s ReAct tool loop
(``tool_loop=True, benchmark_tools=True``) and lets it decide for itself
whether to trade, informed by its own current cash/positions.

One process drives exactly one model, matching one call to this script per
15-minute schedule trigger (see the per-model GitHub Actions workflows) --
there is no matrix/merge step here, unlike the forecast pipeline.

Shadow (paper) trading only. This script hard-asserts
``FORESEA_AGENT_PLACE_TRADE_MODE=shadow`` at startup and refuses to run
otherwise, independent of benchmark_tools.py's own default.

Env:
  AGENT_TRADING_MODEL              required; a configs/models.yaml key
  FORESEA_AGENT_ACCOUNT_DB_PATH    local SQLite path (GCS-synced by the workflow)
  FORESEA_AGENT_NOTES_PATH         local notes JSON path (GCS-synced by the workflow)
  FORESEA_AGENT_PLACE_TRADE_MODE   must be "shadow" (the default) -- hard-checked
  CANDIDATE_COUNT                  new markets to consider per cycle (default 3)
  MAX_TOOL_STEPS                   tool-loop step cap per cycle (default 4)
  AGENT_TRADING_MIN_CLOSE_DAYS     candidate discovery window, days (default 1)
  AGENT_TRADING_MAX_CLOSE_DAYS     candidate discovery window, days (default 30)
  FORESEA_AGENT_MAX_ORDER_NOTIONAL_PCT   per-order cap, fraction of current
                                          account value (default 0.08)
  FORESEA_AGENT_CONCENTRATION_LIMIT      per-ticker cap, fraction of current
                                          account value (default 0.15, read by
                                          benchmark_tools.py)
  FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT   per-cycle cap, fraction of current
                                             account value (default 0.20, read
                                             by benchmark_tools.py)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402
from analyzing_llm_rationale.accounting import MarketQuote  # noqa: E402

MODEL = os.environ.get("AGENT_TRADING_MODEL", "").strip()
VARIANT = os.environ.get("TRACK_VARIANT", "variant0_neutral_baseline")
CANDIDATE_COUNT = max(1, int(os.environ.get("CANDIDATE_COUNT", "3")))
MAX_TOOL_STEPS = max(1, min(8, int(os.environ.get("MAX_TOOL_STEPS", "4"))))
MIN_CLOSE_DAYS = float(os.environ.get("AGENT_TRADING_MIN_CLOSE_DAYS", "1"))
MAX_CLOSE_DAYS = float(os.environ.get("AGENT_TRADING_MAX_CLOSE_DAYS", "30"))
AGENT_ANALYZE_RETRIES = max(1, int(os.environ.get("AGENT_TRADING_RETRIES", "2")))
AGENT_ANALYZE_RETRY_BACKOFF_S = float(os.environ.get("AGENT_TRADING_RETRY_BACKOFF_S", "10"))
# AgentAnalyzeRequest.question has a hard 2000-char server-side limit; a
# single verbose thesis echoed back verbatim can exceed that on its own
# (observed live: 2334 chars), so it's excerpted, not replayed in full.
MAX_LAST_THESIS_CHARS = max(1, int(os.environ.get("AGENT_TRADING_MAX_LAST_THESIS_CHARS", "500")))
# Same hard 2000-char limit -- kept below it with margin, since the candidate
# block also scales with how many tickers an agent currently holds (each now
# carries an extra resolution-window date pair) and isn't otherwise bounded.
MAX_QUESTION_CHARS = 1900
# trading.py's FORESEA_MAX_ORDER_NOTIONAL is a flat-dollar cap shared by every
# order path (human BYO trading included), so it can't be changed here without
# affecting those too. Instead this driver overrides it per-cycle, scoped to
# this one process, as a percentage of the agent's own account value --
# consistent with how benchmark_tools' own concentration/per-cycle-spend
# guards already define "account value" (FORESEA_AGENT_ACCOUNT_VALUE, the
# static starting baseline, not a fluctuating mark-to-market figure).
MAX_ORDER_NOTIONAL_PCT = float(os.environ.get("FORESEA_AGENT_MAX_ORDER_NOTIONAL_PCT", "0.08"))


def _assert_shadow_mode() -> None:
    """Independent safety check on top of benchmark_tools.place_trade's own
    default -- fail loudly at startup rather than silently trading live if a
    workflow's env is ever misconfigured."""
    mode = os.environ.get("FORESEA_AGENT_PLACE_TRADE_MODE", "shadow").strip().lower()
    if mode != "shadow":
        raise RuntimeError(
            f"agent_trading_tick.py refuses to run with FORESEA_AGENT_PLACE_TRADE_MODE={mode!r} "
            "-- this driver is shadow-only by design."
        )


_agent_ready = False


def _init_local_agent(model: str) -> None:
    """Initialise server state for one model, in-process -- mirrors
    track_record_tick.py's _init_local_predict, but disables the evidence
    pipeline entirely since benchmark tools never call forecast/search_evidence."""
    global _agent_ready
    if _agent_ready:
        return
    from analyzing_llm_rationale.cli import init_server_state

    init_server_state(SimpleNamespace(
        model=model,
        variant=VARIANT,
        variants_config=ROOT / "configs" / "variants.yaml",
        models_config=ROOT / "configs" / "models.yaml",
        temperature=0.0,
        max_tokens=int(os.environ.get("MAX_TOKENS", "2048")),
        provider=None,
        local_model_name=None,
        router_model_name=None,
        api_base_url=None,
        api_key_env_var=None,
        api_key_file=None,
        device=os.environ.get("MODEL_DEVICE", "cpu"),
        request_timeout_s=float(os.environ.get("PROVIDER_TIMEOUT_S", "120")),
        model_label=None,
        disable_evidence=True,
        newsapi_key_env_var="NEWSAPI_KEY",
        evidence_source=None,
        disable_query_planner=True,
    ))
    _agent_ready = True
    print(f"agent-trading-tick initialized model={model}")


def _fmt_money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _build_portfolio_block(conn, agent_id: str, last_thesis: Optional[str]) -> str:
    summary = benchmark_tools._account_summary(conn, agent_id, benchmark_tools.DEFAULT_AGENT_ACCOUNT_VALUE)
    lines = [
        "=== Your portfolio (shadow account -- paper trading, no real money) ===",
        f"Cash: {_fmt_money(summary['cash'])} | Starting cash: {_fmt_money(summary['starting_cash'])}",
        f"Realized P&L: {_fmt_money(summary['realized_pnl'])} | Fees paid so far: {_fmt_money(summary['fees_paid'])}",
    ]
    if summary["open_positions"]:
        lines.append("Open positions:")
        for p in summary["open_positions"]:
            lines.append(
                f"  - {p['ticker']} {p['side']}: {p['quantity']:.1f} contracts, "
                f"avg entry {p['avg_entry_price']:.2f}, cost basis {_fmt_money(p['cost_basis'])}"
            )
    else:
        lines.append("Open positions: none.")
    if last_thesis:
        excerpt = (
            last_thesis if len(last_thesis) <= MAX_LAST_THESIS_CHARS
            else last_thesis[:MAX_LAST_THESIS_CHARS].rstrip() + "…"
        )
        lines.append(f"Your own reasoning from the previous cycle: {excerpt}")
    return "\n".join(lines)


def _fmt_candidate_line(quote: Dict[str, Any]) -> str:
    opens = quote.get("created_time") or "unknown"
    close = quote.get("close_time") or "unknown"
    bid = quote.get("yes_bid")
    ask = quote.get("yes_ask")
    return (
        f"  - {quote.get('ident')}: \"{quote.get('question')}\" "
        f"(yes bid/ask {bid}/{ask}, resolution window {opens} -> {close})"
    )


def _build_candidates_block(held_quotes: List[Dict[str, Any]], new_quotes: List[Dict[str, Any]]) -> str:
    lines = ["=== Markets you can act on this cycle (Kalshi only) ==="]
    if held_quotes:
        lines.append("Markets you currently hold (buying the opposite side closes the position):")
        lines.extend(_fmt_candidate_line(q) for q in held_quotes)
    if new_quotes:
        lines.append("New candidate markets:")
        lines.extend(_fmt_candidate_line(q) for q in new_quotes)
    if not held_quotes and not new_quotes:
        lines.append("(No priced candidates this cycle.)")
    return "\n".join(lines)


def _discover_candidates(known_tickers: set) -> List[Dict[str, Any]]:
    new_quotes: List[Dict[str, Any]] = []
    try:
        listed = market_data.list_kalshi(
            limit=CANDIDATE_COUNT * 3,
            min_close_days=MIN_CLOSE_DAYS,
            max_close_days=MAX_CLOSE_DAYS,
            paginate=True,
        )
    except market_data.MarketDataError as exc:
        print(f"  candidate discovery failed: {exc}", file=sys.stderr)
        return new_quotes
    for quote in listed:
        ident = quote.get("ident")
        if not ident or ident in known_tickers or quote.get("probability") is None:
            continue
        new_quotes.append(quote)
        known_tickers.add(ident)
        if len(new_quotes) >= CANDIDATE_COUNT:
            break
    return new_quotes


def _requote_held(tickers: List[str]) -> List[Dict[str, Any]]:
    quotes = []
    for ticker in tickers:
        try:
            quotes.append(market_data.fetch_kalshi(ticker))
        except market_data.MarketDataError as exc:
            print(f"  could not re-quote held position {ticker}: {exc}", file=sys.stderr)
    return quotes


def _current_account_value(conn, agent_id: str, held_quotes: List[Dict[str, Any]]) -> float:
    """Cash + mark-to-market value of open positions, using the freshest bid
    quote for each held ticker -- so every agent-trading risk guard (order
    size, concentration, per-cycle spend) scales with the account's REAL
    performance, not a frozen starting baseline."""
    summary = benchmark_tools._account_summary(conn, agent_id, benchmark_tools.DEFAULT_AGENT_ACCOUNT_VALUE)
    quotes_by_ticker = {q["ident"]: MarketQuote.from_mapping(q) for q in held_quotes if q.get("ident")}
    liquidation_value = 0.0
    for pos in summary["open_positions"]:
        quote = quotes_by_ticker.get(pos["ticker"])
        bid = quote.bid(pos["side"]) if quote else None
        # No live bid (re-quote failed, or the position isn't in held_quotes)
        # -- fall back to cost basis rather than dropping it from the total,
        # so a stale quote never silently shrinks what the guards see.
        liquidation_value += pos["quantity"] * bid if bid is not None else pos["cost_basis"]
    return summary["cash"] + liquidation_value


async def _call_agent_analyze(question: str):
    from analyzing_llm_rationale.server import AgentAnalyzeRequest, agent_analyze

    req = AgentAnalyzeRequest(
        question=question,
        tool_loop=True,
        benchmark_tools=True,
        max_tool_steps=MAX_TOOL_STEPS,
    )
    last_exc: Optional[Exception] = None
    for attempt in range(AGENT_ANALYZE_RETRIES):
        try:
            return await agent_analyze(req, request=None)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"  agent_analyze attempt {attempt + 1}/{AGENT_ANALYZE_RETRIES} failed: {exc}", file=sys.stderr)
            if attempt + 1 < AGENT_ANALYZE_RETRIES:
                time.sleep(AGENT_ANALYZE_RETRY_BACKOFF_S)
    assert last_exc is not None
    raise last_exc


_TRADING_INSTRUCTION = (
    "Decide what, if anything, to do this cycle. You may place_trade on any "
    "ticker listed above (buying the opposite side of a held position "
    "closes it), use web_search for research, and manage_notes to record "
    "anything worth remembering next cycle. Before betting on news you find, "
    "check that the event's date actually falls within THIS market's "
    "resolution window (shown above for each ticker) -- many Kalshi tickers "
    "are one of several in a recurring dated series (e.g. one ending "
    "'-26APR' and another '-26MAY22-26SEP' for the same underlying "
    "question), so real evidence of something that already happened before "
    "this window opened does not resolve this specific contract. If you "
    "don't want to act, just explain why in your final answer."
)


def _build_question(portfolio_block: str, candidates_block: str) -> str:
    question = "\n\n".join([portfolio_block, candidates_block, _TRADING_INSTRUCTION])
    if len(question) <= MAX_QUESTION_CHARS:
        return question
    # Over the server's hard 2000-char limit -- trim the candidates block
    # first, since it's the part that scales with how many tickers an agent
    # currently holds; the portfolio summary and the instruction (which the
    # model needs on every cycle) stay intact.
    overage = len(question) - MAX_QUESTION_CHARS
    keep = max(0, len(candidates_block) - overage - 1)
    trimmed_candidates = candidates_block[:keep].rstrip() + "…"
    return "\n\n".join([portfolio_block, trimmed_candidates, _TRADING_INSTRUCTION])


def _configure_max_order_notional() -> float:
    """Override trading.py's flat-dollar FORESEA_MAX_ORDER_NOTIONAL for this
    process only, as MAX_ORDER_NOTIONAL_PCT of the agent's account value
    (FORESEA_AGENT_ACCOUNT_VALUE -- by the time this runs in run_cycle(),
    already set to the current mark-to-market value, not the static
    starting baseline, so this scales with real performance). Scoped to
    this one cycle's process (each cycle is a fresh process), so every
    other trading path -- human BYO trading, other scripts -- stays at
    trading.py's own unmodified default."""
    account_value = benchmark_tools._env_float(
        "FORESEA_AGENT_ACCOUNT_VALUE", benchmark_tools.DEFAULT_AGENT_ACCOUNT_VALUE
    )
    max_notional = round(account_value * MAX_ORDER_NOTIONAL_PCT, 2)
    os.environ["FORESEA_MAX_ORDER_NOTIONAL"] = str(max_notional)
    return max_notional


def run_cycle(model: str) -> None:
    _assert_shadow_mode()
    _init_local_agent(model)

    agent_id = model
    cycle_id = benchmark_tools._current_cycle_id()

    with benchmark_tools._account_transaction() as conn:
        held_tickers = [
            row["ticker"]
            for row in conn.execute(
                "SELECT DISTINCT ticker FROM agent_positions WHERE agent_id = ? AND quantity > 0",
                (agent_id,),
            )
        ]
        last_cycle = conn.execute(
            "SELECT thesis FROM agent_cycles WHERE agent_id = ? ORDER BY ts DESC LIMIT 1",
            (agent_id,),
        ).fetchone()
        last_thesis = last_cycle["thesis"] if last_cycle else None
        portfolio_block = _build_portfolio_block(conn, agent_id, last_thesis)

    held_quotes = _requote_held(held_tickers)
    known = {q.get("ident") for q in held_quotes if q.get("ident")}
    new_quotes = _discover_candidates(known)

    # Every agent-trading risk guard scales off FORESEA_AGENT_ACCOUNT_VALUE,
    # so it must reflect the account's real, current mark-to-market value
    # here -- computed after held_quotes exist, before any place_trade call
    # in this cycle's tool loop could read a guard.
    with benchmark_tools._account_transaction() as conn:
        os.environ["FORESEA_AGENT_ACCOUNT_VALUE"] = str(_current_account_value(conn, agent_id, held_quotes))
    _configure_max_order_notional()

    question = _build_question(portfolio_block, _build_candidates_block(held_quotes, new_quotes))

    report = asyncio.run(_call_agent_analyze(question))

    candidates_offered = [q.get("ident") for q in (*held_quotes, *new_quotes)]
    with benchmark_tools._account_transaction() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO agent_cycles
            (agent_id, cycle_id, ts, thesis, transcript_json, steps, truncated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                cycle_id,
                datetime.now(timezone.utc).isoformat(),
                report.thesis,
                json.dumps({
                    "candidates_offered": candidates_offered,
                    "tool_transcript": report.tool_transcript,
                }),
                len(report.tool_transcript),
                1 if len(report.tool_transcript) >= MAX_TOOL_STEPS else 0,
            ),
        )

    print(
        f"agent-trading-tick done model={model} cycle={cycle_id} "
        f"steps={len(report.tool_transcript)} candidates={len(candidates_offered)}"
    )


def main() -> int:
    if not MODEL:
        print("AGENT_TRADING_MODEL must be set", file=sys.stderr)
        return 1
    try:
        run_cycle(MODEL)
    except Exception as exc:  # noqa: BLE001
        print(f"agent-trading-tick FAILED model={MODEL}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
