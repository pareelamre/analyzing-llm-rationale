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
from analyzing_llm_rationale.pipeline import parse_model_response  # noqa: E402

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
# Specialist pipeline: research (evidence only, no trade opinion) -> sizing
# (a number-only decision, no tools, computed BEFORE any narrative exists) ->
# execution (place_trade against that number, or an explicit justified
# override). This isn't just a relabeling of one call into three: an LLM
# asked in a single breath to both write a persuasive thesis and size the
# trade tends to anchor the size to whatever "sounds right" for the case it
# just built. Forcing sizing into its own tool-less, JSON-only turn -- fed
# the same live prices and portfolio state, but not yet the thesis -- makes
# it an independent check, and the execution stage is then held to that
# number (must place exactly it, or explain in its final answer why it's
# overriding it).
RESEARCH_MAX_STEPS = max(1, int(os.environ.get("AGENT_TRADING_RESEARCH_MAX_STEPS", "2")))
SIZING_MAX_STEPS = max(1, int(os.environ.get("AGENT_TRADING_SIZING_MAX_STEPS", "1")))
THESIS_MAX_STEPS = max(1, int(os.environ.get("AGENT_TRADING_THESIS_MAX_STEPS", "2")))


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


def _excerpt(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


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
        lines.append(f"Your own reasoning from the previous cycle: {_excerpt(last_thesis, MAX_LAST_THESIS_CHARS)}")
    return "\n".join(lines)


def _fmt_px(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _fmt_candidate_line(quote: Dict[str, Any]) -> str:
    opens = quote.get("created_time") or "unknown"
    close = quote.get("close_time") or "unknown"
    # NO isn't a separate field Kalshi returns -- it's derived from the YES
    # book (no_ask = 1 - yes_bid, etc., see accounting.MarketQuote.ask/bid),
    # the same derivation place_trade's own guards use to price a closing
    # order. Showing both sides here (not just YES) means an agent can read
    # the real price to close a position off this line instead of having to
    # rederive it -- or guess, which is how a real live position ended up
    # rejected for pricing its NO-side close near its YES entry price.
    q = MarketQuote.from_mapping(quote)
    yes_bid, yes_ask = q.bid("YES"), q.ask("YES")
    no_bid, no_ask = q.bid("NO"), q.ask("NO")
    return (
        f"  - {quote.get('ident')}: \"{quote.get('question')}\" "
        f"(yes bid/ask {_fmt_px(yes_bid)}/{_fmt_px(yes_ask)}, "
        f"no bid/ask {_fmt_px(no_bid)}/{_fmt_px(no_ask)}, "
        f"resolution window {opens} -> {close})"
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


async def _call_agent_analyze(
    question: str,
    *,
    tool_names: Optional[List[str]] = None,
    max_steps: Optional[int] = None,
):
    from analyzing_llm_rationale.server import AgentAnalyzeRequest, agent_analyze

    req = AgentAnalyzeRequest(
        question=question,
        tool_loop=True,
        benchmark_tools=True,
        benchmark_tool_names=tool_names,
        max_tool_steps=max_steps or MAX_TOOL_STEPS,
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
    "anything worth remembering next cycle. Price every order off the live "
    "yes/no bid/ask shown above for that ticker -- both sides are the real "
    "current market, not an estimate. Never guess a price, and never reuse "
    "the price you entered a position at when pricing the opposite side to "
    "close it: yes and no move independently and are not the same number. "
    "If you spot a real mispricing against your own view, that is exactly "
    "when to trade it. Before betting on news you find, check that the "
    "event's date actually falls within THIS market's resolution window "
    "(shown above for each ticker) -- many Kalshi tickers are one of "
    "several in a recurring dated series (e.g. one ending '-26APR' and "
    "another '-26MAY22-26SEP' for the same underlying question), so real "
    "evidence of something that already happened before this window opened "
    "does not resolve this specific contract. If you don't want to act, "
    "just explain why in your final answer."
)


def _build_question(portfolio_block: str, candidates_block: str, instruction: str = _TRADING_INSTRUCTION) -> str:
    question = "\n\n".join([portfolio_block, candidates_block, instruction])
    if len(question) <= MAX_QUESTION_CHARS:
        return question
    # Over the server's hard 2000-char limit -- trim the candidates block
    # first, since it's the part that scales with how many tickers an agent
    # currently holds; the portfolio summary and the instruction (which the
    # model needs on every cycle) stay intact.
    overage = len(question) - MAX_QUESTION_CHARS
    keep = max(0, len(candidates_block) - overage - 1)
    trimmed_candidates = candidates_block[:keep].rstrip() + "…"
    return "\n\n".join([portfolio_block, trimmed_candidates, instruction])


_RESEARCH_INSTRUCTION = (
    "You are the RESEARCH specialist on a 3-agent trading team (research -> "
    "sizing -> execution). Your only job this cycle is to gather evidence on "
    "the markets listed above using web_search -- do NOT recommend a trade, "
    "a price, or a size; other specialists own that. In your final answer, "
    "summarize the concrete, dated facts you found for each ticker you "
    "looked into (or state plainly that you found nothing new). Be specific: "
    "what you found, when it happened, and whether it actually falls inside "
    "that ticker's resolution window (shown above)."
)

_SIZING_INSTRUCTION_TEMPLATE = (
    "You are the SIZING specialist on a 3-agent trading team. You do not "
    "write the trade thesis -- you output ONE number-only decision that the "
    "execution specialist will be held to, decided before any narrative is "
    "written. Research findings from this cycle:\n{research}\n\n"
    "Using ONLY the live prices already shown above (never invent a price) "
    "and the portfolio state above (never size past available cash, and "
    "size smaller for markets the research above leaves uncertain), decide "
    "AT MOST ONE trade for this cycle. Respond with EXACTLY ONE JSON object "
    "and nothing else -- no markdown fences, no prose before or after it:\n"
    '{{"action": "trade" or "no_trade", "ticker": "<ticker or null>", '
    '"side": "YES" or "NO" (or null), "price": <the exact live bid/ask you '
    'are pricing off, or null>, "quantity": <integer contracts, or null>, '
    '"rationale": "<one sentence>"}}\n'
    'Set action to "no_trade" and leave the other fields null if nothing on '
    "the board justifies a trade."
)

_THESIS_INSTRUCTION_TEMPLATE = (
    "You are the EXECUTION specialist on a 3-agent trading team. The sizing "
    "specialist, working from the same live prices and portfolio shown "
    "above, recommends:\n{sizing}\n\n"
    "Research findings from this cycle:\n{research}\n\n"
    "If you agree, call place_trade with exactly that ticker, side, price, "
    "and quantity, then give your final answer explaining the trade. If you "
    "disagree, do NOT place a trade -- your final answer must explain "
    "exactly why you are overriding the sizing recommendation, citing the "
    "research or portfolio state above; a disagreement without a stated "
    "reason is not acceptable. Either way you may use manage_notes to "
    "record anything worth remembering next cycle."
)

_SIZING_FIELDS = ("action", "ticker", "side", "price", "quantity", "rationale")


def _coerce_sizing(raw: Dict[str, Any]) -> Dict[str, Any]:
    def _num(value: Any, cast):
        if value in (None, ""):
            return None
        try:
            return cast(value)
        except (TypeError, ValueError):
            return None

    action = (raw.get("action") or "").strip().lower()
    return {
        "action": "trade" if action == "trade" else "no_trade",
        "ticker": raw.get("ticker") or None,
        "side": raw.get("side") or None,
        "price": _num(raw.get("price"), float),
        "quantity": _num(raw.get("quantity"), int),
        "rationale": raw.get("rationale") or "",
    }


def _fmt_sizing(sizing: Dict[str, Any]) -> str:
    if sizing["action"] != "trade":
        return f"no_trade -- {sizing['rationale'] or '(no rationale given)'}"
    return (
        f"trade {sizing['ticker']} {sizing['side']} qty={sizing['quantity']} "
        f"price={sizing['price']} -- {sizing['rationale'] or '(no rationale given)'}"
    )


async def _run_specialist_pipeline(portfolio_block: str, candidates_block: str) -> SimpleNamespace:
    """Research -> sizing -> execution, each its own agent_analyze call with
    its own tool budget (see the constants above for why sizing gets no
    tools and runs before the narrative). Every stage's transcript is kept
    so the record shows what each specialist actually saw and decided, not
    just the final trade."""
    research_question = _build_question(portfolio_block, candidates_block, _RESEARCH_INSTRUCTION)
    research_report = await _call_agent_analyze(
        research_question, tool_names=["web_search"], max_steps=RESEARCH_MAX_STEPS
    )
    research_digest = _excerpt(research_report.thesis, MAX_LAST_THESIS_CHARS) or "(none)"

    sizing_question = _build_question(
        portfolio_block, candidates_block,
        _SIZING_INSTRUCTION_TEMPLATE.format(research=research_digest),
    )
    sizing_report = await _call_agent_analyze(sizing_question, tool_names=[], max_steps=SIZING_MAX_STEPS)
    sizing = _coerce_sizing(parse_model_response(sizing_report.thesis or "", _SIZING_FIELDS))

    thesis_question = _build_question(
        portfolio_block, candidates_block,
        _THESIS_INSTRUCTION_TEMPLATE.format(sizing=_fmt_sizing(sizing), research=research_digest),
    )
    thesis_report = await _call_agent_analyze(
        thesis_question, tool_names=["place_trade", "manage_notes"], max_steps=THESIS_MAX_STEPS
    )

    tool_transcript = [
        *research_report.tool_transcript,
        *sizing_report.tool_transcript,
        *thesis_report.tool_transcript,
    ]
    truncated = bool(
        research_report.tool_loop_truncated
        or sizing_report.tool_loop_truncated
        or thesis_report.tool_loop_truncated
    )
    return SimpleNamespace(
        thesis=thesis_report.thesis,
        tool_transcript=tool_transcript,
        truncated=truncated,
        stages={
            "research": {"thesis": research_report.thesis, "tool_transcript": research_report.tool_transcript},
            "sizing": {"thesis": sizing_report.thesis, "parsed": sizing, "tool_transcript": sizing_report.tool_transcript},
            "thesis": {"thesis": thesis_report.thesis, "tool_transcript": thesis_report.tool_transcript},
        },
    )


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

    candidates_block = _build_candidates_block(held_quotes, new_quotes)

    report = asyncio.run(_run_specialist_pipeline(portfolio_block, candidates_block))

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
                    "stages": report.stages,
                }),
                len(report.tool_transcript),
                1 if report.truncated else 0,
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
