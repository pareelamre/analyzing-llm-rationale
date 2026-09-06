"""Agent capability upgrades, kept pure + dependency-injected for testing:

1. BUILTIN_SKILLS — a curated, always-on forecasting toolkit (base-rate /
   reference-class, scenario decomposition, red-team, key drivers) that sharpens
   every forecast using the existing skill machinery.
2. build_grounding_note() — turns the live track-record aggregate into a short
   self-calibration note the agent can condition its forecast on (e.g. "you tend
   to overprice longshots — shade low-probability YES down").
3. run_tool_loop() — a bounded ReAct-style loop where the model plans and calls
   tools (fetch market, search evidence, scan venue, forecast, track record)
   over several steps, then answers. Tools and the chat function are injected.

The server wires the concrete tools (which call market_data / predict / rag /
track_record_live); this module owns only the reusable logic.
"""
from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

# ── 1. Built-in forecasting skills ────────────────────────────────────────────
# (name, instruction) — run through the same path as user-defined skills.
# A calibration bin has to hold enough resolved forecasts to say anything. The
# thinnest real bin observed carried 94, so this keeps genuine signal while
# refusing to turn a handful of resolutions into a stated bias.
_MIN_CALIBRATION_BIN_N = 30

BUILTIN_SKILLS: List[Tuple[str, str]] = [
    ("Base rate",
     "Name the reference class for this kind of event and state the outside-view "
     "base rate BEFORE any case-specific adjustment. Then say whether the current "
     "forecast is above or below that base rate and whether that's justified."),
    ("Scenario decomposition",
     "Break the question into the 2-4 key conditions or paths that must hold for "
     "YES. Give a rough probability for each and how they combine."),
    ("Red team",
     "Argue the strongest case that the leading forecast is WRONG. What's the best "
     "evidence or mechanism for the opposite outcome, and what would you need to see?"),
    ("Key drivers",
     "List the 2-4 variables that most determine this outcome and exactly what to "
     "watch (events, dates, data releases) that would move the forecast."),
]


def builtin_skills() -> List[Dict[str, str]]:
    return [{"name": n, "instruction": i} for n, i in BUILTIN_SKILLS]


# ── 2. Track-record self-grounding ────────────────────────────────────────────
def _disagreement_skill_notes(by_edge: Any) -> List[str]:
    """What this fleet's disagreements with the market have actually been worth.

    This is the single most decision-relevant fact an agent has, and it is not
    encouraging: realized skill degrades monotonically with the size of the
    disagreement. Measured over 2,710 resolutions --

        0-5pp    n=2515  accuracy 84.0%  skill -0.0001  (level with the market)
        5-10pp   n=  63  accuracy 71.4%  skill -0.021
        10-20pp  n=  38  accuracy 44.7%  skill -0.148
        20pp+    n=  94  accuracy 26.6%  skill -0.415

    -- every disagreement bucket is worse than simply taking the price, and
    the 20pp+ bucket is worse than a coin flip. The fee, spread and
    minimum-net-edge stack means a tradeable position needs a ~6pp divergence
    to clear, which lands squarely in the region where this record says we
    lose. An agent that knows this can weigh a big disagreement as the warning
    it is, rather than as the opportunity it looks like.
    """
    rows = [
        b for b in (by_edge or [])
        if isinstance(b, dict)
        and isinstance(b.get("accuracy"), (int, float))
        and (b.get("n") or 0) >= _MIN_CALIBRATION_BIN_N
        and str(b.get("edge_bucket") or "").strip()
    ]
    if not rows:
        return []
    # Widest disagreement first: that is the bucket an agent is most likely to
    # think it has found something in.
    def _floor_pp(bucket: str) -> float:
        digits = re.findall(r"\d+", bucket)
        return float(digits[0]) if digits else 0.0

    rows.sort(key=lambda b: _floor_pp(str(b["edge_bucket"])), reverse=True)
    worst = rows[0]
    parts = ", ".join(
        "%s: %.0f%% right (n=%d)" % (b["edge_bucket"], 100 * b["accuracy"], b["n"])
        for b in rows
    )
    note = (
        f"- Your own disagreements, by size, over {sum(b['n'] for b in rows)} resolved "
        f"forecasts -- {parts}. Accuracy falls as the gap widens"
    )
    skill = worst.get("skill_vs_market")
    if isinstance(skill, (int, float)) and skill < 0:
        note += (
            f", and in the {worst['edge_bucket']} bucket you have trailed the market "
            f"price by {abs(skill):.3f} Brier"
        )
    note += (
        ". A wide gap is the strongest available signal that you are wrong, not that "
        "the crowd is. Treat it as a reason to re-examine your read, and require "
        "verified primary-source proof before acting on one."
    )
    return [note]


def _calibration_bias_notes(bins: Any) -> List[str]:
    """Self-calibration lines derived from resolved bins, not asserted.

    These two lines used to be hardcoded, and both aged badly in different
    ways. The high-confidence figure ("80-90% resolve YES only ~68% of the
    time") happened to be right when written, but a fixed number silently goes
    stale as the record grows. The tail line was worse: it told every model it
    had "overpriced low-probability outcomes" and to shade tail YES estimates
    DOWN, while the resolved bins showed the opposite -- forecasts near 3.5%
    were resolving YES at 8.2%, so the advice pushed every tail estimate the
    wrong way. Derive the direction from the data instead of stating it.
    """
    rows = [
        b for b in (bins or [])
        if isinstance(b, dict)
        and isinstance(b.get("avg_predicted"), (int, float))
        and isinstance(b.get("observed_yes_rate"), (int, float))
        and (b.get("n") or 0) >= _MIN_CALIBRATION_BIN_N
    ]
    if not rows:
        return []
    notes: List[str] = []

    high = [b for b in rows if b["avg_predicted"] >= 0.8]
    if high:
        n = sum(b["n"] for b in high)
        predicted = sum(b["avg_predicted"] * b["n"] for b in high) / n
        observed = sum(b["observed_yes_rate"] * b["n"] for b in high) / n
        if observed < predicted:
            notes.append(
                f"- High-confidence bias: forecasts averaging {predicted:.0%} resolved YES "
                f"{observed:.0%} of the time ({n} resolved). Damp extreme high-confidence "
                "calls: allow for 11th-hour cancellations, appeals, and procedural delay.")

    tail = [b for b in rows if b["avg_predicted"] <= 0.2]
    if tail:
        n = sum(b["n"] for b in tail)
        predicted = sum(b["avg_predicted"] * b["n"] for b in tail) / n
        observed = sum(b["observed_yes_rate"] * b["n"] for b in tail) / n
        # Direction comes from the record, in whichever way it actually points.
        if observed > predicted:
            notes.append(
                f"- Tail bias: forecasts averaging {predicted:.1%} resolved YES "
                f"{observed:.1%} of the time ({n} resolved) -- longshots have been "
                "UNDER-priced, so do not reflexively shade tail YES estimates down. "
                "A prior, not a hard rule.")
        elif observed < predicted:
            notes.append(
                f"- Tail bias: forecasts averaging {predicted:.1%} resolved YES "
                f"{observed:.1%} of the time ({n} resolved) -- longshots have been "
                "OVER-priced, so shade tail YES estimates down. A prior, not a hard rule.")
    return notes


def build_grounding_note(aggregate: Optional[Dict[str, Any]]) -> str:
    """Short self-calibration note from the live track-record aggregate, or '' ."""
    if not aggregate:
        return ""
    n = aggregate.get("n_snapshots_resolved") or 0
    if not n:
        return ("Forecaster self-calibration: the live track record has no resolved "
                "forecasts yet, so there's no calibration signal — forecast normally.")
    parts = [f"Forecaster self-calibration (live track record, {n} resolved forecasts):"]
    overall = aggregate.get("overall") or {}
    skill = overall.get("skill_vs_market")
    if skill is not None:
        direction = "beating" if skill > 0 else "trailing"
        parts.append(f"- Overall skill vs market: {skill:+.3f} Brier ({direction} the market price).")
    cal = aggregate.get("calibration_model") or {}
    if cal.get("applied"):
        parts.append(f"- Calibration: raw ECE {cal.get('raw_ece')}, model is miscalibrated; "
                     "adjust extreme probabilities toward the calibrated mapping.")
    # Correction: the figure removed here in #411 -- "disagreements >20pp have
    # a 73.4% error rate" -- was NOT unsourced. It is this fleet's own 20pp+
    # by_edge bucket, whose accuracy is 26.6% on n=94, i.e. exactly a 73.4%
    # error rate. It was dropped after searching for the literal string rather
    # than checking whether the number reproduced from our own record. The
    # live version is restored below, derived per-cycle so it cannot go stale.
    parts.extend(_disagreement_skill_notes(aggregate.get("by_edge")))
    parts.append("- Discrepancy Discipline: Kalshi's resolved history (2,243,741 markets) shows prices "
                 "behaving like genuine probabilities -- Brier about 0.02 at close, and calibration improving "
                 "monotonically with volume at every horizon. A large disagreement with a liquid, "
                 "near-resolution price is more likely your error than the market's, so challenge your thesis "
                 "and anchor toward market odds unless you hold verified primary-source proof. Room for a real "
                 "edge widens with distance from resolution: long-dated markets never reach a 0.05 Brier at "
                 "any level of participation.")
    parts.extend(_calibration_bias_notes(aggregate.get("calibration")))
    return "\n".join(parts)


# ── 3. ReAct-style tool loop ──────────────────────────────────────────────────
# A tool is an async callable: (args: dict) -> observation string.
Tool = Callable[[Dict[str, Any]], Awaitable[str]]
# chat_fn: async (messages: list) -> assistant text.
ChatFn = Callable[[List[Dict[str, str]]], Awaitable[str]]


def build_system_prompt(tool_specs: List[Dict[str, str]], max_steps: int, extra_rules: str = "") -> str:
    """System prompt framing the ReAct loop and listing available tools."""
    lines = [
        "You are an autonomous research and forecasting agent. Plan, call "
        "tools, then give a final answer.",
        "",
        "REASONING & EVIDENCE STANDARDS:",
        "- Causal Grounding: Anchor reasoning in concrete, verified facts (dates, official statements, filings) rather than speculation.",
        "- Rule Verification: Actively verify facts against the contract's resolution criteria and explicit exclusions before drawing conclusions.",
        "- Probabilistic Rigor & Discrepancy Discipline: Distinguish theoretical possibility from calibrated probability. If diverging >15pp from market odds, explain why the crowd is mispriced and verify that you are not missing unindexed news.",
        "- Tail Risk & Variance: Avoid assigning >80% certainty to pending human/political decisions with execution risk, and model variance for binned numeric ranges.",
        "",
        "Respond with EXACTLY ONE JSON object per turn — nothing else — in one of two forms:",
        '  {"thought": "...", "action": "TOOL_NAME", "args": { ... }}   to call a tool',
        '  {"thought": "...", "final": "your answer"}                    when you are done',
        "",
        f"Call at most {max_steps} tools; be efficient and stop as soon as you can answer.",
        "",
        "Available tools:",
    ]
    for t in tool_specs:
        lines.append(f"- {t['name']}({t.get('args', '')}): {t['description']}")
    if extra_rules:
        lines.extend(["", extra_rules])
    return "\n".join(lines)


TOOL_ALIASES: Dict[str, str] = {
    "place_order": "place_trade",
    "buy": "place_trade",
    "buy_order": "place_trade",
    "place_trade_order": "place_trade",
    "execute_trade": "place_trade",
    "search": "web_search",
    "google_search": "web_search",
    "web_search_google": "web_search",
    "news_search": "web_search",
    "fetch_market": "get_market",
    "scan": "scan_markets",
    "probability_forecast": "forecast",
    "get_track_record": "track_record",
    "grounding_note": "track_record",
    "calibration": "track_record",
    "news": "search_evidence",
    "evidence": "search_evidence",
    "fetch_evidence": "search_evidence",
    "http_request": "fetch_api",
    "http_get": "fetch_api",
    "call_api": "fetch_api",
    "api_request": "fetch_api",
    "curl": "fetch_api",
    "api": "fetch_api",
    "web_get": "fetch_api",
    "foresea_edge_board": "edge_board",
    "get_edge_board": "edge_board",
    "foresea_batch_quotes": "batch_quotes",
    "get_batch_quotes": "batch_quotes",
    "get_exchange_status": "exchange_status",
    "kalshi_status": "exchange_status",
    "exchange_schedule": "exchange_status",
    "kalshi_schedule": "exchange_status",
    "get_orderbook": "orderbook",
    "clob_orderbook": "orderbook",
    "depth": "orderbook",
    "order_book_arbitrage": "orderbook_arbitrage",
    "arb": "orderbook_arbitrage",
    "arbitrage": "orderbook_arbitrage",
    "tags": "market_tags",
    "get_tags": "market_tags",
    "categories": "market_tags",
    "polymarket_tags": "market_tags",
    "candlesticks": "price_history",
    "history": "price_history",
    "ohlc": "price_history",
    "get_price_history": "price_history",
    "kalshi_live_data": "live_data",
    "game_stats": "live_data",
    "event_live_data": "live_data",
    "get_live_data": "live_data",
    "series": "polymarket_meta",
    "polymarket_series": "polymarket_meta",
    "comments": "polymarket_meta",
    "polymarket_comments": "polymarket_meta",
    "sports": "polymarket_meta",
    "polymarket_sports": "polymarket_meta",
    "trades": "recent_trades",
    "get_trades": "recent_trades",
    "trade_history": "recent_trades",
    "tape": "recent_trades",
    "prints": "recent_trades",
    "leaderboard": "market_leaderboard",
    "top_traders": "market_leaderboard",
    "trader_rankings": "market_leaderboard",
}


def _normalize_action(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Recognize a model's native function-calling JSON shape and normalize it
    to this loop's own {"action": ..., "args": ...} schema, while mapping known
    tool name aliases (e.g. place_order -> place_trade, search -> web_search).

    Not every model reliably follows the schema given in the system prompt --
    observed live: minimax-m3 defaulting to OpenAI-style
    {"name": "web_search", "parameters": {...}} instead, which parsed as valid
    JSON but was silently treated as a "final answer" (that JSON, verbatim)
    because it had no "action" key -- the tool never actually ran even though
    the model was clearly trying to call it.
    """
    if "action" in obj:
        act = obj.get("action")
        if isinstance(act, str) and act in TOOL_ALIASES:
            obj["action"] = TOOL_ALIASES[act]
        if "args" not in obj or not isinstance(obj.get("args"), dict):
            params = obj.get("parameters", obj.get("arguments", {}))
            if isinstance(params, str):
                try:
                    parsed = json.loads(params)
                    obj["args"] = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    obj["args"] = {}
            elif isinstance(params, dict):
                obj["args"] = params
        return obj
    if "final" in obj:
        return obj
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return obj
    name = TOOL_ALIASES.get(name, name)
    params = obj.get("parameters", obj.get("arguments"))
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    if not isinstance(params, dict):
        params = {}
    return {"thought": obj.get("thought", ""), "action": name, "args": params}


def _salvage_action(text: str) -> Optional[Dict[str, Any]]:
    """Recover a tool call from a turn whose JSON is malformed but whose intent
    is unambiguous.

    Live incident: glm-5.2-fp8 emitted the same near-miss every cycle --
    ``{"thought": "...on each topic."," "action": "web_search", "args": {...}}``
    -- a stray ``,"`` after the thought string. Strict parsing rejects it, so
    24 of its last 25 cycles ended at step 0 having called nothing, with the
    raw envelope stored as the model's thesis. The action and args are plainly
    readable, so recover them rather than discarding the whole cycle.

    Only an explicit tool name plus a balanced args object counts; anything
    less returns None so a genuine prose answer is never mistaken for a call.
    """
    name_match = re.search(r'"(?:action|name|tool)"\s*:\s*"([A-Za-z0-9_.\-]+)"', text)
    if not name_match:
        return None
    args: Dict[str, Any] = {}
    args_match = re.search(r'"(?:args|arguments|parameters)"\s*:\s*\{', text)
    if args_match:
        # Walk from the opening brace to its match so nested objects survive.
        depth = 0
        in_string = False
        escape = False
        start = args_match.end() - 1
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        args = parsed
                    break
    thought_match = re.search(r'"(?:thought|reasoning)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    thought = thought_match.group(1) if thought_match else ""
    name = TOOL_ALIASES.get(name_match.group(1), name_match.group(1))
    return {"thought": thought, "action": name, "args": args}


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model turn. Returns the dict, or None
    if no JSON is found (caller treats that as a plain final answer)."""
    if not text:
        return None
    # Strip code fences.
    cleaned = re.sub(r"```(?:json)?", "", text)
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(cleaned):
        if ch == '"' and not escape:
            in_string = not in_string
        elif ch == '\\' and in_string:
            escape = not escape
            continue
        if not in_string:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    blob = cleaned[start:i + 1]
                    try:
                        obj = json.loads(blob)
                        if isinstance(obj, dict):
                            return _normalize_action(obj)
                    except ValueError:
                        start = -1  # keep scanning for a valid object
        escape = False
    # No strictly-valid object anywhere. Before giving up (which costs the model
    # its whole cycle), try to recover an unambiguous tool call from malformed
    # JSON -- see _salvage_action.
    return _salvage_action(cleaned)


def _estimate_message_tokens(messages: List[Dict[str, str]]) -> int:
    """Rough token count for a turn's payload, at the usual ~4 chars/token.

    Deliberately an estimate: the point is to keep a cycle inside a provider
    quota, and an approximate ceiling does that without a tokenizer
    dependency or a per-provider special case.
    """
    return sum(len(str(m.get("content") or "")) for m in messages) // 4


def _trade_call_from_transcript(transcript: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The last place_trade attempt in a cycle, if there was one."""
    for entry in reversed(list(transcript or [])):
        if isinstance(entry, dict) and str(entry.get("action") or "") == "place_trade":
            return entry
    return None


def _synthesise_thesis(answer: str, transcript: Sequence[Dict[str, Any]]) -> str:
    """Build a parseable thesis from what the cycle actually did.

    A model that will not produce the template after being asked leaves its
    decision unreadable: the action, market and probabilities are pulled out
    of those headings by regex, so a wall of prose is worth nothing
    downstream and the cycle records no forecast. Live, five of eight agents
    published unparseable output in one tick -- glm-5-3 echoed the parser's
    own "could not be parsed" hint back as its public thesis, and glm-5-3 and
    glm-5-3-flash have never traded once because they never emit a decision
    anything can act on.

    Rather than publish that, report the execution record. This describes
    only what happened -- whether place_trade was called, with what, and what
    came back -- and never invents a view the model did not state. An
    unstated probability is reported as unstated. The model's own prose is
    preserved verbatim underneath, so nothing is lost, and the header says
    plainly that the harness wrote the structure.
    """
    call = _trade_call_from_transcript(transcript)
    tools = [str(e.get("action")) for e in (transcript or []) if isinstance(e, dict)]
    tool_summary = ", ".join(sorted(set(tools))) if tools else "none"

    if call is not None:
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        side = str(args.get("side") or "").strip().upper()
        ticker = str(args.get("ticker") or "unknown")
        platform = str(args.get("platform") or "kalshi")
        price, quantity = args.get("price"), args.get("quantity")
        model_p = args.get("model_probability")
        action = f"BUY {side}" if side in ("YES", "NO") else "BUY"
        market = f"{ticker} on {platform}"
        sizing = f"{quantity if quantity is not None else 'tool-sized'} contracts @ {price}"
        observation = str(call.get("observation") or "")[:200].replace("\n", " ")
        edge = (f"**Model Probability**: {model_p} vs **Market Price**: {price}"
                if model_p is not None else
                "**Model Probability**: not stated in a parseable form")
    else:
        action, market, sizing = "PASS", "No new position", "No new order"
        observation = "No place_trade call was made this cycle."
        edge = "**Model Probability**: not stated in a parseable form"

    return "\n".join([
        "### 0. Research Delta",
        "- **Strategy**: reported by the harness -- this model's own answer did "
        "not follow the required template, so the sections below describe what "
        "the cycle actually did rather than what it said.",
        f"- **New evidence**: tools called this cycle: {tool_summary}.",
        "- **Belief update**: not stated in a parseable form.",
        "",
        "### 1. Decision & Execution",
        f"- **Action**: {action}",
        f"- **Market & Venue**: {market}",
        f"- **Order Sizing**: {sizing}",
        f"- **Paper execution**: {observation}",
        "",
        "### 3. Model Edge & Valuation",
        f"- {edge}",
        "",
        "---",
        "The model's own answer, unmodified:",
        "",
        (answer or "").strip() or "(empty)",
    ])


def _missing_sections(answer: str, required: Sequence[str]) -> List[str]:
    """Which required section headings a final thesis failed to include.

    Matched case-insensitively on the heading text alone, so a model that
    writes "### 1. Decision and Execution" or drops the number still counts as
    having the section. The point is to catch an answer that ignored the
    template outright, not to police punctuation.
    """
    text = (answer or "").lower()
    return [name for name in required if name.lower() not in text]


def _coerce_answer_text(value: Any) -> str:
    """A model's final answer as text, whatever shape it arrived in.

    Providers do not reliably put a string in ``final``: a dict or list there
    used to raise AttributeError on .strip(), which reached the board as an
    unexplained 502 and discarded an otherwise complete cycle.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False).strip()
        except (TypeError, ValueError):
            return str(value).strip()
    return str(value).strip()


_CONTENT_FREE_ANSWERS = {
    "pass", "hold", "no action", "n/a", "none", "no trade", "no trades",
    "nothing", "skip", "wait",
}


def _is_content_free_answer(answer: str) -> bool:
    """True when a final answer carries a verdict but no reasoning at all.

    Deliberately narrow: only a bare one-liner counts. Any answer that
    actually explains itself -- even briefly -- is left alone.
    """
    stripped = (answer or "").strip().strip(".!*_# ").lower()
    return not stripped or stripped in _CONTENT_FREE_ANSWERS


OnStep = Callable[[Dict[str, Any]], Awaitable[None]]


def _bounded_trade_observation(observation: str, limit: int) -> str:
    """Keep large paper-trade observations valid and useful to later readers.

    ``place_trade`` returns a deliberately rich account snapshot.  Cutting its
    serialized JSON at a character boundary corrupts the transcript, which in
    turn makes a recorded fill look like an unknown attempt on the Agentic
    board.  Retain the bounded execution/audit fields rather than a malformed
    prefix; the full durable account action remains the source of record.
    """
    if len(observation) <= limit:
        return observation
    try:
        result = json.loads(observation)
    except (TypeError, ValueError):
        return observation[:limit]
    if not isinstance(result, dict):
        return observation[:limit]

    compact: Dict[str, Any] = {"observation_truncated": True}
    for key in (
        "ok", "tool", "message", "error", "error_type", "reason", "rejected",
        "skipped", "submitted", "mode", "action_id",
    ):
        value = result.get(key)
        if isinstance(value, str):
            compact[key] = value[:400]
        elif value is not None:
            compact[key] = value
    for key, fields in {
        "execution": (
            "fill_status", "fill_outcome", "filled_quantity", "requested_quantity",
            "unfilled_quantity_cancelled", "immediate_only", "time_in_force",
        ),
        "normalized_order": ("platform", "ticker", "outcome", "price", "quantity"),
        "risk_guard": ("allowed", "reasons", "cycle_id", "notional", "filled_notional"),
        "sizing": ("mode", "label", "eligible", "reason", "target_notional"),
    }.items():
        value = result.get(key)
        if isinstance(value, dict):
            compact[key] = {field: value[field] for field in fields if field in value}
    encoded = json.dumps(compact, sort_keys=True, separators=(",", ":"), default=str)
    if len(encoded) <= limit:
        return encoded
    # Keep context machine-readable even for a pathological result with a
    # huge error/reasons payload. The durable transcript above remains whole.
    minimal = {
        "observation_truncated": True,
        "ok": result.get("ok"),
        "tool": result.get("tool"),
        "reason": str(result.get("reason") or "execution_summary_omitted")[:120],
    }
    return json.dumps(minimal, sort_keys=True, separators=(",", ":"), default=str)


def _bounded_observation(action: str, observation: Any, limit: int) -> str:
    """Bound model context without changing the durable tool transcript."""
    text = str(observation or "")
    if str(action).strip().lower() in {"place_trade", "place_order", "buy", "buy_order"}:
        return _bounded_trade_observation(text, limit)
    return text[:limit]


def _compact_observation_context(messages: List[Dict[str, str]], max_chars: int) -> None:
    """Keep the prompt bounded while retaining the full durable transcript.

    Long tool-use work is more likely to fail late in a run because every
    observation is replayed to every subsequent model turn. Preserve the most
    recent observations verbatim, then compact the oldest context only when
    the combined observation budget would otherwise be exceeded.
    """
    if max_chars <= 0:
        return
    observation_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and message.get("content", "").startswith("Observation: ")
    ]
    total = sum(len(messages[index]["content"]) for index in observation_indexes)
    for index in observation_indexes:
        if total <= max_chars:
            break
        original = messages[index]["content"]
        compact = "Observation: [Earlier observation compacted; full result is in the audit transcript.]"
        if original == compact:
            continue
        messages[index]["content"] = compact
        total -= len(original) - len(compact)


async def run_tool_loop(
    question: str,
    tools: Dict[str, Tool],
    tool_specs: List[Dict[str, str]],
    chat_fn: ChatFn,
    *,
    max_steps: int = 5,
    obs_limit: int = 4000,
    observation_context_limit: int = 12000,
    extra_rules: str = "",
    on_step: Optional[OnStep] = None,
    on_step_start: Optional[OnStep] = None,
    retry_unusable_final: bool = False,
    required_final_sections: Optional[Sequence[str]] = None,
    max_structure_retries: int = 2,
    token_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Drive the ReAct loop. Returns {answer, transcript, steps, truncated}.

    `on_step_start`, if given, is awaited once per parsed tool call *before*
    the tool runs, with {index, thought, action, args} -- e.g. to durably
    record that a step was attempted before its outcome is known, so a crash
    mid-tool-call still leaves a trace rather than none at all. `on_step`, if
    given, is awaited once per completed tool call (not on the final-answer
    turn) with {index, thought, action, args, observation, error}. Both are
    intended to update the same durable record by index (start, then fill
    in the outcome), not append two unrelated entries. Any exception either
    raises is swallowed here: a caller-supplied hook must never break the loop.
    """
    system = build_system_prompt(tool_specs, max_steps, extra_rules)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}"},
    ]
    # A ReAct turn resends the whole conversation, so cost grows with every
    # step. Measured on the live fleet, one cycle runs 10-21k tokens against
    # a 10,000-token-per-minute provider quota -- a single deep cycle can
    # spend the entire minute and 429 whichever model runs next in the serial
    # lane. A step count cannot express that; a token budget can, and it lets
    # a cheap cycle keep every step it wants while stopping a runaway one.
    tokens_used = 0
    transcript: List[Dict[str, Any]] = []
    reformat_hint = (
        "That reply could not be parsed. Respond with exactly one JSON object: either "
        '{"thought": "...", "action": "<tool name>", "args": {...}} to call a tool, or '
        '{"final": "<answer>"} to give your final answer. Do not use any other format.'
    )
    reformat_retries_used = 0
    max_reformat_retries = 1
    substantive_hint = (
        "That is a verdict with no analysis behind it, and you have not called "
        "any tool this cycle. Do the research first -- check the market and the "
        "evidence -- then give your decision with the reasoning that supports "
        "it. If the answer really is no action, say why."
    )
    substantive_retry_used = False
    unusable_retry_used = False
    structure_retries_used = 0
    # Why a cycle stopped is not recoverable from `steps`/`truncated` alone:
    # a budget stop and an out-of-steps stop both report steps=max_steps and
    # truncated=True, which made a shallow cycle indistinguishable from a
    # capped one when diagnosing a slow tick. Report the cause explicitly.
    stop_reason = "max_steps"
    steps_completed = 0
    for step in range(max_steps):
        turn_tokens = _estimate_message_tokens(messages)
        # Stop before spending a turn we cannot afford, not after. The step
        # already completed is worth finalising; the one that would blow the
        # budget just earns a 429 for this model and the next one in line.
        if token_budget and step and tokens_used + turn_tokens > token_budget:
            stop_reason = "token_budget"
            break
        tokens_used += turn_tokens
        steps_completed = step + 1
        out = await chat_fn(messages)
        action = parse_action(out)
        if action is not None and "final" in action:
            # `final` is not guaranteed to be a string. gemma-4-26b-a4b-it
            # returned a nested object there, and calling .strip() on it
            # raised AttributeError, which surfaced to the board as a bare
            # 502 and cost the whole cycle. Serialise a structured final
            # rather than crashing on it -- the content is still the model's
            # answer, just not in prose form.
            answer = _coerce_answer_text(action.get("final")) or _coerce_answer_text(out)
            # A verdict with no research behind it is not a decision, however
            # well it is written. This first caught llama-3.3-70b-instruct
            # returning a bare "PASS" on turn 0 with zero tool calls, so it
            # tested whether the TEXT looked empty -- which let fluency stand
            # in for work. minimax-m3 then published a full house-format
            # thesis on a zero-tool cycle: "### 0. Research Delta", a strategy
            # line, and specific probabilities ("Stafford P(YES) ~2%, Evans
            # P(YES) ~12%"), having fetched no market and searched no
            # evidence. On a public board that reads as researched analysis
            # when it is recall and priors. What matters is whether any tool
            # ran this cycle, not how the answer is phrased -- so ask once for
            # the work regardless of how substantial the prose looks.
            # `tools` guards the turns that deliberately offer none (a pure
            # sizing or reasoning stage): with nothing to call, answering
            # straight away is the correct behaviour, not a skipped step.
            if (
                not transcript
                and tools
                and not substantive_retry_used
                and step < max_steps - 1
            ):
                substantive_retry_used = True
                messages.append({"role": "assistant", "content": out})
                messages.append({"role": "user", "content": substantive_hint})
                continue
            # The required sections are how everything downstream reads a
            # thesis: the action, the market, and the model-vs-market
            # probabilities are all pulled out of them by regex. A model that
            # answers with its deliberation instead -- "Let me reconsider...
            # Hmm, but actually... Wait" -- publishes a wall of text whose
            # decision cannot be scored, which is why five of six agents in a
            # recent tick recorded forecasts=0 while all of them believed they
            # had reported one. Ask once for the structure it was given.
            if (
                required_final_sections
                and _missing_sections(answer, required_final_sections)
            ):
                missing = _missing_sections(answer, required_final_sections)
                if (
                    structure_retries_used < max_structure_retries
                    and step < max_steps - 1
                ):
                    structure_retries_used += 1
                    messages.append({"role": "assistant", "content": out})
                    messages.append({"role": "user", "content": (
                        "That answer is missing the required section"
                        f"{'s' if len(missing) > 1 else ''}: {', '.join(missing)}. "
                        "Reply with the final thesis in the mandated template "
                        "only -- the section headings and their fields, nothing "
                        "before or after. Strictly keep it concise for human review: "
                        "1-2 sentences per field, under 200 words total. Keep your "
                        "deliberation and scratchpad out of it."
                    )})
                    continue
                # Asked and still unformatted. Publishing the prose would
                # record no decision at all, so report the execution record
                # instead -- what the cycle did, with the model's own answer
                # preserved underneath.
                answer = _synthesise_thesis(answer, transcript)
            return {"answer": answer, "transcript": transcript,
                    "steps": step, "truncated": False}
        if action is None:
            # No JSON found at all -- could be an incomplete/foreign-format
            # tool-call attempt (prose, or a different tool-call dialect)
            # rather than a genuine final answer, so give the model one
            # chance to correct its format before accepting the raw text.
            if reformat_retries_used < max_reformat_retries and step < max_steps - 1:
                reformat_retries_used += 1
                messages.append({"role": "assistant", "content": out})
                messages.append({"role": "user", "content": reformat_hint})
                continue
            raw_answer = (out or "").strip()
            if required_final_sections and _missing_sections(raw_answer, required_final_sections):
                raw_answer = _synthesise_thesis(raw_answer, transcript)
            return {"answer": raw_answer, "transcript": transcript,
                    "steps": step, "truncated": False}
        if "action" not in action:
            # Valid JSON, but not a tool-call envelope (e.g. a model that
            # answers directly with structured fields instead of {"final":
            # ...}) -- treat as a final answer as before; a caller-side
            # deterministic backstop may still extract structure from it.
            #
            # Callers with no such backstop opt into a retry instead. On the
            # agent-trading path this shape is simply a wasted cycle: the
            # board renders "completed a research pass but did not return a
            # publishable final thesis", which is what gemma-4-26b-a4b-it and
            # gpt-oss-120b both produced at step 0 with no tool calls.
            if (
                retry_unusable_final
                and not unusable_retry_used
                and step < max_steps - 1
            ):
                unusable_retry_used = True
                messages.append({"role": "assistant", "content": out})
                messages.append({"role": "user", "content": reformat_hint})
                continue
            raw_answer = (out or "").strip()
            if required_final_sections and _missing_sections(raw_answer, required_final_sections):
                raw_answer = _synthesise_thesis(raw_answer, transcript)
            return {"answer": raw_answer, "transcript": transcript,
                    "steps": step, "truncated": False}
        name = str(action.get("action", ""))
        args = action.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        thought = action.get("thought") if isinstance(action.get("thought"), str) else ""
        if on_step_start is not None:
            try:
                await on_step_start({"index": step, "thought": thought, "action": name, "args": args})
            except Exception:
                pass
        tool = tools.get(name)
        if tool is None and name in TOOL_ALIASES:
            name = TOOL_ALIASES[name]
            tool = tools.get(name)
        errored = tool is None
        try:
            if tool:
                obs = await tool(args)
            else:
                avail = ", ".join(f"'{k}'" for k in sorted(tools.keys()))
                obs = f"(unknown tool '{name}'; available tools: {avail})" if avail else f"(unknown tool '{name}')"
        except Exception:
            obs = f"(tool '{name}' failed)"
            errored = True
        # Keep every byte of the tool result in the transcript/audit record.
        # Only the follow-up prompt gets a compact representation, because
        # model context length is a provider constraint rather than a reason
        # to corrupt or discard execution evidence.
        raw_observation = str(obs or "")
        context_observation = _bounded_observation(name, raw_observation, obs_limit)
        transcript.append({"action": name, "args": args, "observation": raw_observation})
        if on_step is not None:
            try:
                await on_step({
                    "index": step,
                    "thought": thought,
                    "action": name,
                    "args": args,
                    "observation": raw_observation,
                    "error": errored,
                })
            except Exception:
                pass
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": f"Observation: {context_observation}"})
        _compact_observation_context(messages, observation_context_limit)
    # Out of steps (or out of token budget) — force one terminal JSON answer.
    # Asking for plain text contradicted the system contract and let some
    # providers emit a pasted sequence of prior tool-call envelopes instead of
    # a final thesis.
    stopped = {"stop_reason": stop_reason, "tokens_used": tokens_used,
               "steps_completed": steps_completed}
    final = await chat_fn(messages + [{"role": "user", "content": (
        "Stop calling tools. Return exactly one JSON object with a `final` field containing "
        "your concise publishable thesis using the mandated 4-section template (under 200 words): "
        '{"final":"### 0. Research Delta\\n- **Strategy**: ...\\n\\n### 1. Decision & Execution\\n..."}. '
        "Do not include an action, args, tool call, scratch work, or any other JSON object."
    )}])
    final_text = (final or "").strip()
    parsed_final = parse_action(final_text)
    if isinstance(parsed_final, dict) and "final" in parsed_final:
        final_text = str(parsed_final["final"]).strip()
        if required_final_sections and _missing_sections(final_text, required_final_sections):
            final_text = _synthesise_thesis(final_text, transcript)
        return {"answer": final_text, "transcript": transcript,
                "steps": max_steps, "truncated": True, "finalization_failed": False,
                **stopped}
    # Never promote a tool-call envelope (or a repetition of several of them)
    # to the public thesis. The durable transcript preserves that detail for
    # operators; callers can publish a truthful fallback instead.
    if isinstance(parsed_final, dict) and "action" in parsed_final:
        return {"answer": "", "transcript": transcript,
                "steps": max_steps, "truncated": True, "finalization_failed": True,
                **stopped}
    if required_final_sections and _missing_sections(final_text, required_final_sections):
        final_text = _synthesise_thesis(final_text, transcript)
    return {"answer": final_text, "transcript": transcript,
            "steps": max_steps, "truncated": True, "finalization_failed": False,
            **stopped}
