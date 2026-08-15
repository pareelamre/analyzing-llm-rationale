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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# ── 1. Built-in forecasting skills ────────────────────────────────────────────
# (name, instruction) — run through the same path as user-defined skills.
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
    # Surface the longshot-bias pattern when by-horizon/overall accuracy suggests it.
    parts.append("- Known tendency: this model has overpriced low-probability/longshot "
                 "outcomes. Treat that as a prior and shade tail YES estimates down, not a hard rule.")
    return "\n".join(parts)


# ── 3. ReAct-style tool loop ──────────────────────────────────────────────────
# A tool is an async callable: (args: dict) -> observation string.
Tool = Callable[[Dict[str, Any]], Awaitable[str]]
# chat_fn: async (messages: list) -> assistant text.
ChatFn = Callable[[List[Dict[str, str]]], Awaitable[str]]


def build_system_prompt(tool_specs: List[Dict[str, str]], max_steps: int, extra_rules: str = "") -> str:
    lines = [
        "You are a forecasting research agent. You gather information by calling "
        "tools, then give a final answer.",
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


def _normalize_action(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Recognize a model's native function-calling JSON shape and normalize it
    to this loop's own {"action": ..., "args": ...} schema.

    Not every model reliably follows the schema given in the system prompt --
    observed live: minimax-m3 defaulting to OpenAI-style
    {"name": "web_search", "parameters": {...}} instead, which parsed as valid
    JSON but was silently treated as a "final answer" (that JSON, verbatim)
    because it had no "action" key -- the tool never actually ran even though
    the model was clearly trying to call it. Left untouched if the model
    already used the prompted schema (has "action" or "final").
    """
    if "action" in obj or "final" in obj:
        return obj
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return obj
    params = obj.get("parameters", obj.get("arguments"))
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    if not isinstance(params, dict):
        params = {}
    return {"thought": obj.get("thought", ""), "action": name, "args": params}


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model turn. Returns the dict, or None
    if no JSON is found (caller treats that as a plain final answer)."""
    if not text:
        return None
    # Strip code fences.
    cleaned = re.sub(r"```(?:json)?", "", text)
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
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
    return None


OnStep = Callable[[Dict[str, Any]], Awaitable[None]]


async def run_tool_loop(
    question: str,
    tools: Dict[str, Tool],
    tool_specs: List[Dict[str, str]],
    chat_fn: ChatFn,
    *,
    max_steps: int = 5,
    obs_limit: int = 4000,
    extra_rules: str = "",
    on_step: Optional[OnStep] = None,
    on_step_start: Optional[OnStep] = None,
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
    transcript: List[Dict[str, Any]] = []
    for step in range(max_steps):
        out = await chat_fn(messages)
        action = parse_action(out)
        if action is None or "final" in action or "action" not in action:
            answer = (action or {}).get("final") if action else out
            return {"answer": (answer or out or "").strip(), "transcript": transcript,
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
        errored = tool is None
        try:
            obs = await tool(args) if tool else f"(unknown tool '{name}')"
        except Exception:
            obs = f"(tool '{name}' failed)"
            errored = True
        obs = (obs or "")[:obs_limit]
        transcript.append({"action": name, "args": args, "observation": obs})
        if on_step is not None:
            try:
                await on_step({
                    "index": step,
                    "thought": thought,
                    "action": name,
                    "args": args,
                    "observation": obs,
                    "error": errored,
                })
            except Exception:
                pass
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    # Out of steps — force a final answer.
    final = await chat_fn(messages + [{"role": "user",
                                       "content": "Stop calling tools. Give your final answer now as plain text."}])
    return {"answer": (final or "").strip(), "transcript": transcript,
            "steps": max_steps, "truncated": True}
