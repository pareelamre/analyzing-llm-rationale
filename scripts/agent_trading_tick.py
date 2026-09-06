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
  AGENT_TRADING_WEATHER_CANDIDATE_QUOTA  source-verified NWS weather candidates
                                          reserved per cycle (default 1)
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
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import requests
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import (  # noqa: E402
    agent_trading_stats,
    benchmark_tools,
    market_data,
    weather_markets,
)
from analyzing_llm_rationale.accounting import MarketQuote  # noqa: E402
from analyzing_llm_rationale.config import load_model_configs  # noqa: E402
from analyzing_llm_rationale.observability import init_observability  # noqa: E402

# Without a root handler, every logger.error/exception in the server modules
# this script drives is discarded. That is how a glm-5-3 cycle could fail with
# only "502: unexpected response" in the workflow log and no trace of the
# actual exception anywhere -- the one line that would have named it was
# written to a logger with nowhere to go.
logging.basicConfig(
    level=os.environ.get("AGENT_TRADING_LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("foresea.agent_trading_tick")
meter = metrics.get_meter("foresea.agent_trading_tick")
learning_refreshes = meter.create_counter(
    "agent_trading.learning.refreshes", unit="1", description="Resolved-trade learning refreshes"
)
learning_lessons = meter.create_counter(
    "agent_trading.learning.lessons", unit="1", description="New per-agent lessons recorded"
)
learning_refresh_duration = meter.create_histogram(
    "agent_trading.learning.refresh.duration", unit="s", description="Resolved-trade learning refresh duration"
)
thesis_forecasts_recorded = meter.create_counter(
    "agent_trading.thesis_forecasts.recorded",
    unit="1",
    description="Structured thesis forecasts persisted for later scoring",
)
thesis_forecast_outcomes = meter.create_counter(
    "agent_trading.thesis_forecasts.outcomes",
    unit="1",
    description="Structured thesis forecasts scored after a market resolves",
)
thesis_forecast_resolution_checks = meter.create_counter(
    "agent_trading.thesis_forecasts.resolution_checks",
    unit="1",
    description="Bounded venue-resolution checks for unresolved thesis forecasts",
)
thesis_forecast_refresh_duration = meter.create_histogram(
    "agent_trading.thesis_forecasts.refresh.duration",
    unit="s",
    description="Duration of resolving and scoring structured thesis forecasts",
)
agent_analyze_attempts = meter.create_counter(
    "agent_trading.analyze.attempts", unit="1", description="Agent analysis provider attempts"
)
agent_analyze_retry_delay = meter.create_histogram(
    "agent_trading.analyze.retry_delay", unit="s", description="Agent analysis retry backoff"
)
calibration_contexts = meter.create_counter(
    "agent_trading.calibration_contexts", unit="1", description="Research calibration priors added to candidates"
)
calibration_context_duration = meter.create_histogram(
    "agent_trading.calibration_context.duration", unit="s", description="Research calibration context duration"
)
research_contexts = meter.create_counter(
    "agent_trading.research.contexts", unit="1", description="Prior-cycle research contexts prepared"
)
research_sources_carried = meter.create_counter(
    "agent_trading.research.sources_carried", unit="1", description="Prior-cycle research sources carried forward"
)
research_context_duration = meter.create_histogram(
    "agent_trading.research.context.duration", unit="s", description="Prior-cycle research context duration"
)
strategy_selections = meter.create_counter(
    "agent_trading.strategy.selections",
    unit="1",
    description="Agent-declared decision strategy by completed shadow-trading cycle",
)
thesis_execution_reconciliations = meter.create_counter(
    "agent_trading.thesis_execution.reconciliations",
    unit="1",
    description="Declared thesis actions reconciled into guarded shadow orders",
)
provider_degradations = meter.create_counter(
    "agent_trading.provider.degradations",
    unit="1",
    description="Completed agent cycles unavailable because the upstream provider degraded",
)
weather_candidate_discoveries = meter.create_counter(
    "agent_trading.weather_candidates.discovery",
    unit="1",
    description="Source-verified weather candidates offered to a shadow-trading cycle",
)
weather_candidate_discovery_duration = meter.create_histogram(
    "agent_trading.weather_candidates.discovery.duration",
    unit="s",
    description="Duration of bounded weather candidate discovery",
)

MODEL = os.environ.get("AGENT_TRADING_MODEL", "").strip()
VARIANT = os.environ.get("TRACK_VARIANT", "variant0_neutral_baseline")
# How many markets an agent is offered per cycle. At 3 the menu was the
# binding constraint on trade rate, not the gate: a trade needs its edge to
# beat half-spread + fee + floor, only ~24% of live sides sit at or below
# 4pp, and three markets in listing order usually contained nothing
# reachable however good the read was. The candidates are now ranked by
# that hurdle, so a wider menu means more genuinely actionable markets
# rather than more noise -- and the merit gate still decides what is worth
# trading, so nothing here forces a position.
CANDIDATE_COUNT = max(1, int(os.environ.get("CANDIDATE_COUNT", "8")))
# No upper clamp. At 4 steps, research-heavy cycles ran out before executing:
# 489 live cycles hit the ceiling and 38 described a BUY the model never got
# to place. The loop still ends as soon as the model gives a final answer, so
# a higher ceiling costs nothing on cycles that finish early.
MAX_TOOL_STEPS = max(1, int(os.environ.get("MAX_TOOL_STEPS", "8")))
MIN_CLOSE_DAYS = float(os.environ.get("AGENT_TRADING_MIN_CLOSE_DAYS", "1"))
# Kalshi Research measures Brier at ~0.02 by close but 0.08-0.09 at a 3-month
# horizon, and finds long-dated markets never cross 0.05 no matter how many
# traders arrive -- "no amount of additional trading fully substitutes for the
# passage of time". A 30-day ceiling therefore excluded exactly the region
# where mispricing survives, leaving agents to hunt edge in the part of the
# curve that is closest to efficient. Widen to a quarter so the long horizon
# is at least in the candidate set; the merit gate still decides what is worth
# trading, and nothing here forces a position.
MAX_CLOSE_DAYS = float(os.environ.get("AGENT_TRADING_MAX_CLOSE_DAYS", "90"))
WEATHER_CANDIDATE_QUOTA = max(
    0, min(CANDIDATE_COUNT, int(os.environ.get("AGENT_TRADING_WEATHER_CANDIDATE_QUOTA", "1")))
)
# ``agent_analyze`` already retries individual provider calls. Retrying the
# complete ReAct cycle here can replay research/tool work after a transient
# provider failure and, more importantly, multiply requests during an upstream
# rate-limit incident. Leave whole-cycle retry opt-in for manual recovery; the
# scheduled workflow uses the safe default of one attempt.
AGENT_ANALYZE_RETRIES = max(1, int(os.environ.get("AGENT_TRADING_RETRIES", "1")))
# One exception to that default, for the failure it was never about: a bare
# upstream 503 is transient -- the same model answers minutes later, and a
# cycle lost to it is a whole model missing from the board. The rate-limit
# concern above still holds and still wins, because ``_failure_kind`` tests
# for 429 before 503: a quota rejection wrapped as "503 ... (upstream returned
# HTTP 429)" classifies as provider_rate_limited and is never retried here.
AGENT_ANALYZE_UNAVAILABLE_RETRIES = max(
    1, int(os.environ.get("AGENT_TRADING_UNAVAILABLE_RETRIES", "2"))
)
AGENT_ANALYZE_RETRY_BACKOFF_S = float(os.environ.get("AGENT_TRADING_RETRY_BACKOFF_S", "10"))
# A provider outage is a completed, safely-persisted agent attempt—not a
# broken paper ledger.  Workflows use this distinct code to report a degraded
# provider lane while still failing on local/data-integrity errors.
PROVIDER_DEGRADATION_EXIT_CODE = 75
PROVIDER_DEGRADATION_FAILURE_KINDS = {
    "provider_unavailable",
    "provider_rate_limited",
    "provider_timeout",
}
# SCADS publishes a five-minute model-health probe.  Treat it as an advisory
# circuit breaker: when it explicitly says a configured model is down, do not
# spend a full ReAct/tool-loop retry budget merely to obtain a predictable 503.
# The check fails open if the status page itself is unavailable, so Foresea
# never turns a status-site outage into an agent outage.
SCADS_STATUS_PRECHECK = os.environ.get("AGENT_TRADING_SCADS_STATUS_PRECHECK", "true").strip().lower() not in {
    "0", "false", "no", "off",
}
SCADS_STATUS_URL = os.environ.get("AGENT_TRADING_SCADS_STATUS_URL", "https://llm.scads.ai/status/state.json")
SCADS_STATUS_TIMEOUT_S = max(0.1, float(os.environ.get("AGENT_TRADING_SCADS_STATUS_TIMEOUT_S", "5")))
SCADS_UNAVAILABLE_STATES = {"down", "timeout", "not_listed"}
# AgentAnalyzeRequest.question used to have a tight 2000-char server-side
# limit (see server.py) that was found silently destroying almost an entire
# candidates block, including every Polymarket candidate, down to a mid-word
# fragment any time an agent held a position: candidates_offered were
# routinely never actually visible to the model, even though
# _discover_candidates was correctly surfacing them. That server-side cap is
# gone -- MAX_QUESTION_CHARS below is now a pure sanity backstop against a
# genuine runaway bug (e.g. unbounded position-list growth), not an
# operating constraint; it should never bind in practice.
MAX_LAST_THESIS_CHARS = max(1, int(os.environ.get("AGENT_TRADING_MAX_LAST_THESIS_CHARS", "2000")))
# Each venue's own resolution rules -- fetched fresh every cycle by
# market_data.py's fetch_kalshi/fetch_polymarket (never cached from
# position-open time), so a rule change the venue makes after entry is
# always visible on the very next cycle. All characters are supplied in full
# without truncation so the agent has the complete legal criteria and carveouts.
MAX_RULES_CHARS = max(1, int(os.environ.get("AGENT_TRADING_MAX_RULES_CHARS", "50000")))
LEARNING_CONTEXT_LIMIT = max(1, min(10, int(os.environ.get("AGENT_TRADING_LEARNING_CONTEXT_LIMIT", "5"))))
LEARNING_STATS_LIMIT = max(LEARNING_CONTEXT_LIMIT, min(50, int(os.environ.get("AGENT_TRADING_LEARNING_STATS_LIMIT", "20"))))
THESIS_FORECAST_RESOLUTION_CHECK_LIMIT = max(
    1, min(20, int(os.environ.get("AGENT_TRADING_THESIS_FORECAST_RESOLUTION_CHECK_LIMIT", "5")))
)
cycle_runs = meter.create_counter(
    "agent_trading.cycles",
    unit="1",
    description="Completed autonomous agent cycles by terminal outcome",
)
cycle_duration = meter.create_histogram(
    "agent_trading.cycle.duration",
    unit="s",
    description="End-to-end autonomous agent cycle duration",
)


def _model_backstop_chars(model: str) -> int:
    """The backstop scales with THIS model's actual context window rather
    than a single arbitrary constant. SCADS AI's own /v1/models listing
    (queried 2026-08-18) only publishes context length for one of the ten
    agent-trading models (glm-5-3, 524288 tokens) -- everything else,
    including all three scads-alias-* models, returns nothing, so those
    fall back to a conservative shared default (128K tokens -- safe across
    the other hosted models here: Llama 3.3, Qwen3-Coder, Kimi-K3,
    GPT-OSS-120B, Gemma-4, MiniMax-M3 all support at least that much)
    rather than a guessed per-model figure that could be wrong in the
    unsafe direction.

    Reserves half the window for everything else in the real prompt this
    field doesn't account for -- system prompt, the ~17 tool specs, and the
    ReAct loop's own accumulated conversation history across MAX_TOOL_STEPS
    -- and estimates 4 characters per token (a standard, slightly
    conservative ratio for English text: undercounting usable chars is the
    safe direction, since it can only make this backstop tighter, never
    looser than the model can actually handle).
    """
    default_context_window_tokens = 128000
    chars_per_token = 4
    reserved_fraction = 0.5
    try:
        models = load_model_configs(ROOT / "configs" / "models.yaml")
        context_window_tokens = models[model].context_window_tokens
    except (KeyError, FileNotFoundError, ValueError):
        context_window_tokens = None
    context_window_tokens = context_window_tokens or default_context_window_tokens
    return int(context_window_tokens * chars_per_token * (1 - reserved_fraction))


# No character cap by default. The trimming machinery below is retained as an
# opt-in escape hatch (set AGENT_TRADING_MAX_QUESTION_CHARS to a positive
# number to re-enable it), but nothing is withheld from the model otherwise:
# measured against SCADS, DeepSeek-V4-Flash accepted an 83k-token prompt and
# answered in ~6s, and every agent-trading route publishes a context window of
# at least 65k tokens, so the previous derived cap only ever risked dropping
# decision-relevant markets and resolution rules for no gain.
_configured_question_chars = int(os.environ.get("AGENT_TRADING_MAX_QUESTION_CHARS", "0"))
MAX_QUESTION_CHARS = (
    _configured_question_chars if _configured_question_chars > 0 else sys.maxsize
)
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
        disable_evidence=False,
        newsapi_key_env_var="NEWSAPI_KEY",
        evidence_source=None,
        disable_query_planner=False,
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


def _prior_thesis_state(last_thesis: Optional[str]) -> str:
    """Retain a compact decision state without replaying a whole old thesis.

    A prior thesis is useful as an audit anchor, but pasting it verbatim makes
    the model paraphrase itself. Preserve the decision-relevant lines only;
    fresh research and any outcome lessons remain separate contexts.
    """
    if not last_thesis:
        return ""
    text = str(last_thesis).strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = str(parsed.get("final") or parsed.get("thesis") or parsed.get("thought") or text).strip()
    except (TypeError, ValueError):
        pass

    selected = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith("###") or re.match(
            r"[-*]\s+\*\*(Action|Market & Venue|Model Probability|Key Catalysts|Invalidation Trigger)",
            cleaned,
            flags=re.IGNORECASE,
        ):
            selected.append(cleaned)
    state = "\n".join(selected) or _excerpt(text, min(MAX_LAST_THESIS_CHARS, 900))
    return _excerpt(state, MAX_LAST_THESIS_CHARS)


def _research_context(last_transcript: Optional[str]) -> str:
    """Carry forward bounded, deduplicated web sources from the prior turn.

    The tool transcript is the auditable source of research used in a cycle.
    We expose only source URLs and short search summaries to the next turn,
    avoiding unbounded raw observations or copied thesis prose.
    """
    started = time.perf_counter()
    with tracer.start_as_current_span("agent_trading.build_research_context") as span:
        try:
            if not last_transcript:
                span.set_attribute("research.source_count", 0)
                return ""
            parsed = json.loads(last_transcript)
            if isinstance(parsed, dict):
                steps = parsed.get("tool_transcript") or []
            elif isinstance(parsed, list):
                steps = parsed
            else:
                steps = []

            sources: List[tuple[str, str]] = []
            summaries: List[str] = []
            seen_urls = set()
            for step in steps:
                if not isinstance(step, dict) or str(step.get("action") or step.get("tool") or "") != "web_search":
                    continue
                observation = step.get("observation") or step.get("result")
                if isinstance(observation, str):
                    try:
                        observation = json.loads(observation)
                    except ValueError:
                        observation = {}
                if not isinstance(observation, dict):
                    continue
                summary = str(observation.get("summary") or "").strip()
                if summary:
                    summaries.append(_excerpt(summary, 280))
                for source in observation.get("sources") or []:
                    if not isinstance(source, dict):
                        continue
                    url = str(source.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    sources.append((_excerpt(str(source.get("title") or url), 100), url))
                    if len(sources) >= 3:
                        break
                if len(sources) >= 3:
                    break

            span.set_attribute("research.source_count", len(sources))
            if not sources and not summaries:
                return ""
            lines = [
                "=== Research carried forward from your prior cycle ===",
                "Treat these as leads to verify, not as a reason to repeat the old thesis. "
                "A non-risk-reducing trade needs a dated evidence delta, not a recycled URL or summary.",
            ]
            for index, (title, url) in enumerate(sources, start=1):
                lines.append(f"- Source {index}: {title} — {url}")
            if summaries:
                lines.append(f"- Prior search signal: {summaries[0]}")
            research_contexts.add(1)
            if sources:
                research_sources_carried.add(len(sources))
            return "\n".join(lines)
        except (TypeError, ValueError, json.JSONDecodeError):
            span.set_attribute("research.parse_error", True)
            return ""
        finally:
            research_context_duration.record(time.perf_counter() - started)


def _build_portfolio_block(
    conn,
    agent_id: str,
    last_thesis: Optional[str],
    learning_block: Optional[str] = None,
    last_transcript: Optional[str] = None,
) -> str:
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
    if learning_block:
        lines.append(learning_block)
    research_context = _research_context(last_transcript)
    if research_context:
        lines.append(research_context)
    prior_state = _prior_thesis_state(last_thesis)
    if prior_state:
        lines.append(f"=== Prior thesis state (do not paraphrase it) ===\n{prior_state}")
    return "\n".join(lines)


def _learning_lesson(action_type: str, realized_pnl: float) -> str:
    """Return a bounded, deterministic postmortem for one realized trade.

    This deliberately does not ask another model to self-critique, and never
    changes a risk limit. It gives the next cycle a small calibration cue based
    on an auditable realized outcome, not an instruction copied from a prior
    thesis or an overfit strategy adjustment.
    """
    event = "settlement" if action_type == "settlement" else "position close"
    if realized_pnl > 0.005:
        return (
            f"The {event} was profitable. Keep the evidence standard unchanged; "
            "a win is not evidence that a similar market is mispriced."
        )
    if realized_pnl < -0.005:
        return (
            f"The {event} lost money. Recheck base rates, price discipline, and "
            "resolution rules before taking a comparable exposure."
        )
    return (
        f"The {event} was approximately flat. Do not treat it as validation; "
        "require fresh independent evidence before re-entering a similar market."
    )


def _refresh_learning(conn, agent_id: str) -> int:
    """Persist one lesson for each newly realized settlement or position close.

    ``agent_actions`` remains the source of truth. ``agent_learning`` is a
    deduplicated, presentation-ready audit trail keyed to the immutable source
    action, so retries and later cycles cannot teach the same outcome twice.
    """
    started = time.perf_counter()
    with tracer.start_as_current_span("agent_trading.refresh_learning") as span:
        span.set_attribute("agent.id", agent_id)
        try:
            rows = conn.execute(
                """
                SELECT id, ts, action_type, platform, ticker, outcome, realized_pnl
                FROM agent_actions
                WHERE agent_id = ?
                  AND (
                    action_type = 'settlement'
                    OR (action_type = 'trade' AND realized_pairs > 0)
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM agent_learning learning
                    WHERE learning.agent_id = agent_actions.agent_id
                      AND learning.source_action_id = agent_actions.id
                  )
                ORDER BY ts ASC
                """,
                (agent_id,),
            ).fetchall()
            now = datetime.now(timezone.utc).isoformat()
            for row in rows:
                pnl = float(row["realized_pnl"] or 0.0)
                conn.execute(
                    """
                    INSERT INTO agent_learning
                    (agent_id, source_action_id, source_ts, action_type, platform,
                     ticker, outcome, realized_pnl, lesson, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agent_id,
                        str(row["id"]),
                        str(row["ts"]),
                        str(row["action_type"]),
                        row["platform"],
                        row["ticker"],
                        row["outcome"],
                        pnl,
                        _learning_lesson(str(row["action_type"]), pnl),
                        now,
                    ),
                )
            count = len(rows)
            learning_refreshes.add(1, {"outcome": "success"})
            if count:
                learning_lessons.add(count, {"outcome": "recorded"})
            span.set_attributes({"learning.new_count": count, "outcome": "success"})
            return count
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            learning_refreshes.add(1, {"outcome": "failure"})
            logger.warning("agent learning refresh failed agent=%s", agent_id, exc_info=True)
            raise
        finally:
            learning_refresh_duration.record(time.perf_counter() - started)


def _build_learning_block(conn, agent_id: str) -> str:
    """Build bounded, outcome-only learning context for the next decision."""
    rows = conn.execute(
        """
        SELECT source_ts, action_type, platform, ticker, outcome, realized_pnl, lesson
        FROM agent_learning
        WHERE agent_id = ?
        ORDER BY source_ts DESC
        LIMIT ?
        """,
        (agent_id, LEARNING_STATS_LIMIT),
    ).fetchall()
    forecast_summary = conn.execute(
        """
        SELECT COUNT(*) AS resolved_count,
               AVG(brier_score) AS brier_score,
               AVG(market_brier_score) AS market_brier_score,
               AVG(model_probability - resolved_outcome) AS probability_bias
        FROM agent_thesis_forecasts
        WHERE agent_id = ? AND resolved_outcome IS NOT NULL
        """,
        (agent_id,),
    ).fetchone()
    resolved_forecasts = int(forecast_summary["resolved_count"] or 0)
    if not rows and not resolved_forecasts:
        return ""

    recent = rows[:LEARNING_CONTEXT_LIMIT]
    wins = sum(1 for row in rows if float(row["realized_pnl"]) > 0.005)
    losses = sum(1 for row in rows if float(row["realized_pnl"]) < -0.005)
    total_pnl = sum(float(row["realized_pnl"]) for row in rows)
    lines = [
        "=== Learning from your resolved shadow trades ===",
        "Use this only as a calibration check, never as market evidence. Risk caps and eligibility rules are unchanged.",
    ]
    if rows:
        lines.extend([
            (
                f"Recent realized outcomes ({len(rows)}): {wins} profitable, {losses} loss-making, "
                f"aggregate realized P&L {_fmt_money(total_pnl)}."
            ),
            "Newest lessons:",
        ])
        for row in recent:
            venue = str(row["platform"] or "unknown venue").title()
            ticker = str(row["ticker"] or "unknown market")
            outcome = str(row["outcome"] or "unresolved")
            lines.append(
                f"  - [{venue}] {ticker}: {row['action_type']} ({outcome}), "
                f"realized P&L {_fmt_money(row['realized_pnl'])}. {row['lesson']}"
            )
    if resolved_forecasts:
        brier = float(forecast_summary["brier_score"])
        market_brier = forecast_summary["market_brier_score"]
        bias = float(forecast_summary["probability_bias"])
        bias_note = (
            "your P(YES) has averaged above the outcome rate; damp YES confidence"
            if bias > 0.05 else
            "your P(YES) has averaged below the outcome rate; do not automatically chase favourites"
            if bias < -0.05 else
            "your average P(YES) is close to the observed YES rate"
        )
        market_note = (
            f" versus market Brier {float(market_brier):.3f}"
            if market_brier is not None else ""
        )
        lines.append(
            f"Forecast calibration ({resolved_forecasts} final outcomes): Brier {brier:.3f}{market_note}; "
            f"{bias_note}. This is descriptive, not a sizing override."
        )
    weather_rows = conn.execute(
        """
        SELECT weather_market_type, weather_settlement_source, COUNT(*) AS resolved_count,
               AVG(brier_score) AS brier_score, AVG(market_brier_score) AS market_brier_score
        FROM agent_thesis_forecasts
        WHERE agent_id = ? AND resolved_outcome IS NOT NULL AND weather_market_type IS NOT NULL
        GROUP BY weather_market_type, weather_settlement_source
        ORDER BY resolved_count DESC, weather_market_type, weather_settlement_source
        LIMIT 4
        """,
        (agent_id,),
    ).fetchall()
    if weather_rows:
        lines.append("Weather calibration by contract type/source (descriptive only):")
        for row in weather_rows:
            market_brier = row["market_brier_score"]
            market_note = f", market Brier {float(market_brier):.3f}" if market_brier is not None else ""
            lines.append(
                f"  - {row['weather_market_type']} / {row['weather_settlement_source']}: "
                f"{int(row['resolved_count'])} resolved, Brier {float(row['brier_score']):.3f}{market_note}."
            )
    return "\n".join(lines)


def _fmt_px(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _paper_market_domain(quote: Dict[str, Any]) -> str:
    """Map a venue category/question into the paper's comparable domains."""
    text = " ".join(
        str(quote.get(field) or "") for field in ("category", "question", "ident")
    ).lower()
    for domain, markers in {
        "politics": ("politic", "election", "president", "congress", "senate", "governor", "government"),
        "sports": ("sport", "nfl", "nba", "mlb", "nhl", "soccer", "football", "tennis", "game"),
        "crypto": ("crypto", "bitcoin", "btc", "ethereum", "eth", "solana"),
        "finance": ("finance", "fed", "inflation", "interest rate", "s&p", "nasdaq", "gdp"),
        "weather": ("weather", "temperature", "rain", "snow", "hurricane", "precipitation"),
        "entertainment": ("entertainment", "oscar", "grammy", "movie", "film", "album"),
    }.items():
        if any(marker in text for marker in markers):
            return domain
    return "other"


def _paper_horizon_bucket(close_time: Any, *, now: Optional[datetime] = None) -> str:
    if not close_time:
        return "unknown"
    try:
        close = datetime.fromisoformat(str(close_time).strip().replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    hours = (close - reference).total_seconds() / 3600
    if hours <= 48:
        return "short"
    if hours <= 24 * 7:
        return "medium"
    if hours <= 24 * 30:
        return "long"
    return "very_long"


def _calibration_resolution_time(quote: Dict[str, Any]) -> Any:
    """Use Kalshi's expected expiry as the event-time calibration anchor.

    Kalshi Research measures calibration against the time the underlying event
    resolves, rather than a potentially earlier administrative market close.
    The public market payload exposes this as ``expected_expiration_time``.
    """
    platform = str(quote.get("platform") or "").strip().lower()
    if platform == "kalshi" and quote.get("expected_expiration_time"):
        return quote["expected_expiration_time"]
    return quote.get("close_time")


def _has_observed_volume(quote: Dict[str, Any]) -> bool:
    try:
        return float(quote.get("volume")) > 0
    except (TypeError, ValueError):
        return False


def _paper_calibration_context(quote: Dict[str, Any], *, now: Optional[datetime] = None) -> str:
    """Return bounded, non-mechanical calibration priors for a candidate.

    Every cited study is population-level and advisory. Agents must still
    derive their own probability from evidence; no prior may mechanically
    transform a price or relax a trading control.
    """
    started = time.perf_counter()
    with tracer.start_as_current_span("agent_trading.build_calibration_context") as span:
        try:
            domain = _paper_market_domain(quote)
            resolution_time = _calibration_resolution_time(quote)
            horizon = _paper_horizon_bucket(resolution_time, now=now)
            platform = str(quote.get("platform") or "kalshi").strip().lower()
            span.set_attributes({
                "market.domain": domain,
                "market.horizon_bucket": horizon,
                "market.venue": platform,
            })

            principles: List[str] = []
            sources: List[str] = []
            if platform == "kalshi":
                kalshi_principle = (
                    "Kalshi prices were broadly well calibrated and became more reliable nearer the underlying event's "
                    "true resolution time. Treat the price as a probability baseline, but derive P(YES) independently "
                    "and trade only when the remaining divergence clears the live spread, fees, and selected Kelly threshold."
                )
                if quote.get("expected_expiration_time"):
                    kalshi_principle += " Calibration timing uses this contract's expected expiration, not its administrative close."
                if not _has_observed_volume(quote):
                    kalshi_principle += " No positive reported volume is available, so do not assume the study's participation benefit applies."
                principles.append(kalshi_principle)
                sources.append("Kalshi Research, Calibration in Prediction Markets")

            if domain == "politics":
                principle = (
                    "Political prices were persistently compressed toward 50% across both venues. "
                    "An evidence-backed favourite may be underpriced, but do not mechanically extremise it."
                )
                if platform == "kalshi":
                    principle += " Treat large political prints as possible venue microstructure, not proof of informed flow."
                principles.append(principle)
                sources.append("Le, 2026, arXiv:2602.19520")
            elif domain == "weather" and horizon == "short":
                principle = (
                    "Short-horizon weather prices were historically too extreme. Demand unusually strong, "
                    "independent evidence before following a market move."
                )
                principles.append(principle)
                sources.append("Le, 2026, arXiv:2602.19520")
            elif domain == "sports" and horizon in {"short", "medium"}:
                principle = (
                    "Sports prices were closest to calibrated at short-to-medium horizons. "
                    "Require a conventional evidence-based edge after spread and fees."
                )
                principles.append(principle)
                sources.append("Le, 2026, arXiv:2602.19520")
            elif domain == "sports" and horizon == "very_long":
                principle = (
                    "Long-horizon sports prices showed favourite-longshot compression. "
                    "Check whether independent evidence supports a more decisive probability."
                )
                principles.append(principle)
                sources.append("Le, 2026, arXiv:2602.19520")
            elif horizon == "very_long":
                principle = (
                    "Across domains, long-horizon prices tended to be compressed toward 50%. "
                    "Use this as a hypothesis to investigate, not a substitute for evidence."
                )
                principles.append(principle)
                sources.append("Le, 2026, arXiv:2602.19520")

            outcome = "applied" if principles else "not_applicable"
            calibration_contexts.add(1, {"domain": domain, "horizon": horizon, "outcome": outcome})
            span.set_attribute("outcome", outcome)
            if not principles:
                return ""
            return f"Research calibration prior ({'; '.join(sources)}): {' '.join(principles)}"
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            calibration_contexts.add(1, {"domain": "unknown", "horizon": "unknown", "outcome": "failure"})
            logger.warning("paper calibration context failed", exc_info=True)
            raise
        finally:
            calibration_context_duration.record(time.perf_counter() - started)


def _edge_hurdle_by_side(quote: Dict[str, Any]) -> Dict[str, float]:
    """Edge needed on each side before a position can clear fees and the floor.

    Half the spread (a buy pays the ask, the fair estimate sits at the mid),
    plus the per-contract fee, plus FORESEA_AGENT_MIN_NET_EDGE. Sides without
    a two-sided quote, or quoted at or above 1.00, are omitted -- there is no
    reachable bar on a contract that cannot return more than it costs.
    """
    from analyzing_llm_rationale.benchmark_tools import _kalshi_fee, _min_net_edge

    q = MarketQuote.from_mapping(quote)
    out: Dict[str, float] = {}
    for side in ("YES", "NO"):
        ask, bid = q.ask(side), q.bid(side)
        if ask is None or bid is None or not 0.0 < ask < 1.0:
            continue
        try:
            fee = _kalshi_fee(ask, 1.0)
        except Exception:
            fee = 0.0
        out[side.lower()] = (ask - (ask + bid) / 2.0) + fee + _min_net_edge()
    return out


def _edge_hurdle_pp(quote: Dict[str, Any]) -> float:
    """Cheapest side's hurdle, for ranking. Untradeable sorts last."""
    hurdles = _edge_hurdle_by_side(quote)
    return min(hurdles.values()) if hurdles else float("inf")


def _fmt_edge_hurdle(quote: Dict[str, Any]) -> str:
    """State the hurdle per side, so "find an edge" becomes a number.

    Measured across live candidates this runs 3pp to 49pp: the same 2% floor
    demanding sixteen times more conviction depending on the book, none of it
    previously visible. An agent saw two quotes and could not tell that one
    needed a near-certainty to be tradeable at all.
    """
    parts = [
        f"{side} +{hurdle * 100:.1f}pp"
        for side, hurdle in sorted(_edge_hurdle_by_side(quote).items())
    ]
    if not parts:
        return "edge hurdle n/a (no two-sided quote)"
    return "edge needed vs mid to clear fees+floor: " + ", ".join(parts)


def _fmt_participation(quote: Dict[str, Any]) -> str:
    """Volume and resting depth for a candidate, or an explicit unknown.

    Kalshi Research finds Brier score falls monotonically with event volume
    within every time horizon, which makes participation depth the best
    available read on how much room is left between the price and the truth.
    The agent was previously shown only whether volume existed at all, so it
    could not tell a market with a handful of contracts from one with tens of
    thousands -- the difference between a price worth disputing and one that
    is already close to unbeatable. Unique-trader count is the study's other
    axis but neither venue exposes it, so volume stands in for it.
    """
    def _num(key: str) -> Optional[float]:
        try:
            value = float(quote.get(key))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    volume, depth = _num("volume"), _num("liquidity")
    if volume is None and depth is None:
        return "participation unreported"
    parts = []
    if volume is not None:
        parts.append(f"volume {volume:,.0f}")
    if depth is not None:
        parts.append(f"depth/open interest {depth:,.0f}")
    return ", ".join(parts)


def _fmt_candidate_line(quote: Dict[str, Any]) -> str:
    opens = quote.get("created_time") or "unknown"
    close = quote.get("close_time") or "unknown"
    expected_expiration = quote.get("expected_expiration_time")
    # NO isn't always a separate field the venue returns -- it's derived from
    # the YES book when absent (no_ask = 1 - yes_bid, etc., see accounting.MarketQuote.ask/bid),
    # the same derivation place_trade's own guards use to price a closing
    # order. Showing both sides here (not just YES) means an agent can read
    # the real price to close a position off this line instead of having to
    # rederive it -- or guess, which is how a real live position ended up
    # rejected for pricing its NO-side close near its YES entry price.
    q = MarketQuote.from_mapping(quote)
    yes_bid, yes_ask = q.bid("YES"), q.ask("YES")
    no_bid, no_ask = q.bid("NO"), q.ask("NO")
    platform = str(quote.get("platform") or "Kalshi").strip().lower()
    line = (
        f"  - [{platform}] {quote.get('ident')}: \"{quote.get('question')}\" "
        f"(yes bid/ask {_fmt_px(yes_bid)}/{_fmt_px(yes_ask)}, "
        f"no bid/ask {_fmt_px(no_bid)}/{_fmt_px(no_ask)}, "
        f"{_fmt_participation(quote)}, "
        f"{_fmt_edge_hurdle(quote)}, "
        f"resolution window {opens} -> {close})"
    )
    if expected_expiration:
        line += f"\n    Expected underlying resolution: {expected_expiration}"
    rules = str(quote.get("resolution_criteria") or "").strip()
    if rules:
        line += f"\n    Resolution rules: {rules}"
    calibration_context = _paper_calibration_context(quote)
    if calibration_context:
        line += f"\n    {calibration_context}"
    weather_context = weather_markets.format_weather_market_brief(quote)
    if weather_context:
        line += f"\n    {weather_context}"
    return line


def _build_candidates_block(held_quotes: List[Dict[str, Any]], new_quotes: List[Dict[str, Any]]) -> str:
    lines = ["=== Markets you can act on this cycle (Kalshi and Polymarket) ==="]
    if held_quotes:
        lines.append("Markets you currently hold (buying the opposite side closes the position):")
        lines.extend(_fmt_candidate_line(q) for q in held_quotes)
    if new_quotes:
        lines.append("New candidate markets:")
        lines.extend(_fmt_candidate_line(q) for q in new_quotes)
    if not held_quotes and not new_quotes:
        lines.append("(No priced candidates this cycle.)")
    return "\n".join(lines)


def _list_venue(platform: str, limit: int, *, category: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        if platform == "polymarket":
            return market_data.list_polymarket(
                limit=limit, min_close_days=MIN_CLOSE_DAYS, max_close_days=MAX_CLOSE_DAYS,
                category=category,
            )
        return market_data.list_kalshi(
            limit=limit, min_close_days=MIN_CLOSE_DAYS, max_close_days=MAX_CLOSE_DAYS, paginate=True,
            category=category,
        )
    except market_data.MarketDataError as exc:
        print(f"  candidate discovery failed ({platform}): {exc}", file=sys.stderr)
        return []


def _is_researchable_weather_candidate(quote: Dict[str, Any]) -> bool:
    """Allow only NWS daily contracts that can receive source-matched research.

    Other weather contracts can still be held/reduced under the normal paper
    guards. This discovery lane is intentionally narrower: it reserves room
    only for markets where the agent can inspect a named official source
    before deciding whether to open exposure.
    """
    brief = weather_markets.classify_weather_market(quote)
    return bool(
        brief.is_weather
        and brief.trade_permitted
        and brief.market_type == "daily_temperature"
        and brief.settlement_source == "nws_daily_climate_report"
        and brief.station
    )


def _discover_weather_candidates(known_tickers: set, *, limit: int) -> List[Dict[str, Any]]:
    """Reserve a small, source-verified NWS lane without blocking ordinary discovery."""
    if limit <= 0:
        return []
    started = time.perf_counter()
    with tracer.start_as_current_span("agent_trading.weather_candidates.discover") as span:
        selected: List[Dict[str, Any]] = []
        scanned = 0
        per_venue_limit = max(6, limit * 6)
        try:
            iterators = [
                iter(_list_venue(platform, per_venue_limit, category="Weather"))
                for platform in ("kalshi", "polymarket")
            ]
            active = list(iterators)
            while active and len(selected) < limit:
                for iterator in list(active):
                    try:
                        quote = next(iterator)
                    except StopIteration:
                        active.remove(iterator)
                        continue
                    scanned += 1
                    ident = quote.get("ident")
                    if (
                        not ident
                        or ident in known_tickers
                        or quote.get("probability") is None
                        or not _is_researchable_weather_candidate(quote)
                    ):
                        continue
                    selected.append(quote)
                    known_tickers.add(ident)
                    if len(selected) >= limit:
                        break
            outcome = "offered" if selected else "none_eligible"
            span.set_attributes({
                "weather.candidates.scanned": scanned,
                "weather.candidates.offered": len(selected),
                "outcome": outcome,
            })
            weather_candidate_discoveries.add(len(selected) or 1, {"outcome": outcome})
            return selected
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            weather_candidate_discoveries.add(1, {"outcome": "failure"})
            logger.warning("weather candidate discovery failed", exc_info=True)
            return []
        finally:
            weather_candidate_discovery_duration.record(time.perf_counter() - started)


def _discover_candidates(known_tickers: set) -> List[Dict[str, Any]]:
    new_quotes = _discover_weather_candidates(known_tickers, limit=WEATHER_CANDIDATE_QUOTA)
    # Round-robin one candidate at a time across venues (rather than filling
    # Kalshi's share first) so a shortfall in one venue's listing doesn't
    # starve the other's, and consecutive candidates aren't all one venue.
    per_venue_limit = CANDIDATE_COUNT * 3
    iterators = [iter(_list_venue(p, per_venue_limit)) for p in ("kalshi", "polymarket")]
    active = list(iterators)
    # Gather a wider pool than we will offer, so there is something to choose
    # between: taking the first N in listing order is a random draw with
    # respect to the only property that decides whether a trade is possible.
    pool: List[Dict[str, Any]] = []
    pool_target = max(CANDIDATE_COUNT * 4, CANDIDATE_COUNT + 8)
    while active and len(pool) < pool_target:
        for it in list(active):
            try:
                quote = next(it)
            except StopIteration:
                active.remove(it)
                continue
            ident = quote.get("ident")
            if not ident or ident in known_tickers or quote.get("probability") is None:
                continue
            pool.append(quote)
            known_tickers.add(ident)
            if len(pool) >= pool_target:
                break
    # Offer the markets where a position can actually clear. A trade needs its
    # net edge to beat fees plus the min-net-edge floor measured against the
    # executable ask, so the real hurdle is half-spread + fee + floor --
    # measured live, 3pp to 49pp across candidates. Only 4% of sides sit at or
    # below 3pp and only 24% at or below 4pp, so three markets taken in
    # listing order usually held nothing an agent could act on however good
    # its read was. Ranking by hurdle puts the ~3pp markets in front of it
    # every cycle, roughly halving the conviction a trade needs -- without
    # touching the gate, the fees, or the floor.
    #
    # The tension is worth stating rather than hiding: a tight spread usually
    # means a liquid, well-priced market, which the Kalshi calibration study
    # finds hardest to beat. But 20pp of genuine edge essentially never
    # exists -- this fleet's own resolved record puts 20pp+ disagreements at
    # 26.6% accuracy -- so a reachable bar on a well-priced market is a better
    # proposition than an unreachable one on a stale quote.
    pool.sort(key=_edge_hurdle_pp)
    room = max(0, CANDIDATE_COUNT - len(new_quotes))
    new_quotes.extend(pool[:room])
    return new_quotes


def _weather_candidate_count(quotes: List[Dict[str, Any]]) -> int:
    return sum(1 for quote in quotes if _is_researchable_weather_candidate(quote))


def _weather_research_call_count(transcript: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for step in transcript
        if str(step.get("tool") or step.get("action") or "") == "weather_market_research"
    )


def _requote_held(positions: List[tuple]) -> List[Dict[str, Any]]:
    quotes = []
    for platform, ticker in positions:
        try:
            if platform == "polymarket":
                quotes.append(market_data.fetch_polymarket(slug=ticker))
            else:
                quotes.append(market_data.fetch_kalshi(ticker))
        except market_data.MarketDataError as exc:
            print(f"  could not re-quote held position [{platform}] {ticker}: {exc}", file=sys.stderr)
    return quotes


def _current_account_value(conn, agent_id: str, held_quotes: List[Dict[str, Any]]) -> float:
    """Cash + mark-to-market value of open positions, using the freshest bid
    quote for each held ticker -- so every agent-trading risk guard (order
    size, concentration, per-cycle spend) scales with the account's REAL
    performance, not a frozen starting baseline."""
    summary = benchmark_tools._account_summary(conn, agent_id, benchmark_tools.DEFAULT_AGENT_ACCOUNT_VALUE)
    quotes_by_key = {
        (str(q.get("platform") or "").lower(), q["ident"]): MarketQuote.from_mapping(q)
        for q in held_quotes if q.get("ident")
    }
    liquidation_value = 0.0
    for pos in summary["open_positions"]:
        quote = quotes_by_key.get((str(pos["platform"]).lower(), pos["ticker"]))
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
        # A worker ledger is one model's paper account.  Leaving this unset
        # lets /agent/analyze auto-route the request, which means the board can
        # attribute another provider's failure (or answer) to this ledger.
        model=MODEL,
        tool_loop=True,
        benchmark_tools=True,
        max_tool_steps=MAX_TOOL_STEPS,
    )
    last_exc: Optional[Exception] = None
    max_attempts = max(AGENT_ANALYZE_RETRIES, AGENT_ANALYZE_UNAVAILABLE_RETRIES)
    for attempt in range(max_attempts):
        try:
            report = await agent_analyze(req, request=None)
            agent_analyze_attempts.add(1, {"model": MODEL or "unknown", "outcome": "success"})
            return report
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            # How many attempts this failure is worth depends on what it is,
            # not on how many are configured overall: a 503 or transient timeout
            # earns a second try, a 429 (or anything else) stops at the baseline.
            # The 503/timeout allowance is a floor, never a ceiling -- an operator
            # who raises AGENT_TRADING_RETRIES for manual recovery must not
            # thereby get *fewer* attempts on failures worth retrying.
            allowed = (
                max(AGENT_ANALYZE_RETRIES, AGENT_ANALYZE_UNAVAILABLE_RETRIES)
                if _failure_kind(exc) in {"provider_unavailable", "provider_timeout"}
                else AGENT_ANALYZE_RETRIES
            )
            agent_analyze_attempts.add(
                1,
                {
                    "model": MODEL or "unknown",
                    "outcome": "retry" if attempt + 1 < allowed else "failure",
                },
            )
            print(f"  agent_analyze attempt {attempt + 1}/{allowed} failed: {exc}", file=sys.stderr)
            if attempt + 1 >= allowed:
                break
            delay_s = AGENT_ANALYZE_RETRY_BACKOFF_S * (2 ** attempt)
            agent_analyze_retry_delay.record(delay_s, {"model": MODEL or "unknown"})
            logger.warning(
                "agent analysis retry model=%s attempt=%s/%s delay_s=%s kind=%s",
                MODEL or "unknown", attempt + 1, allowed, delay_s, _failure_kind(exc),
            )
            await asyncio.sleep(delay_s)
    assert last_exc is not None
    raise last_exc


_TRADING_INSTRUCTION = (
    "Decide what, if anything, to do this cycle. place_trade on any ticker "
    "above using its [kalshi]/[polymarket] tag as the platform arg (buying "
    "the opposite side of a held position closes it); use web_search for "
    "research, optimize_portfolio for a Kelly-criterion sizing reference "
    "(it scores today's whole platform-wide opportunity set, not just "
    "your own candidates, so treat it as one input, not a verdict), and "
    "manage_notes to BOTH read and write your own memory: call it with "
    "action='list' (or 'search') early in a cycle to recall what you "
    "concluded before, and action='add' to record what is worth carrying "
    "forward. Your notes are never shown to you automatically, so notes you "
    "do not read back are wasted work; a note recording only 'no new "
    "evidence' is not worth storing. "
    "A decision only counts if you place it: describing a BUY in your final "
    "answer without calling place_trade leaves your account unchanged, and 38 "
    "live cycles were lost exactly that way. Budget your tool calls so at "
    "least one remains for execution. "
    "Price orders off the live yes/no bid/ask shown above -- not an "
    "estimate. Never guess a price or reuse your entry price for the "
    "opposite side when closing: yes and no move independently. A real "
    "mispricing against your own view is exactly when to trade it.\n\n"
    "RESEARCH QUALITY GATE: A new position requires at least one dated, material "
    "evidence update from this cycle and an explicit comparison with the prior view. "
    "Do not re-open or add risk solely because a previous thesis, search result, or "
    "market candidate is still visible. If there is no material evidence delta, HOLD "
    "or PASS; a fresh quote alone is not research.\n\n"
    "YOU ARE COMPETING. Eight agents trade this board against the same "
    "candidates, the same starting bankroll and the same guards, and the "
    "standings above are live. The objective is to finish top by return, and "
    "you are ranked against the others every cycle whether or not you act. "
    "Standing still is a position: an agent that never trades cannot climb, "
    "and one that trades badly falls. The board is public and permanent -- "
    "every cycle you run is recorded under your name.\n\n"
    "What actually moves you up is return, not activity. Look at the "
    "standings before drawing the obvious conclusion: the most active agent "
    "on this board is last, and the leader got there on fewer trades with "
    "better ones. A losing trade does not just cost money, it hands places to "
    "everyone who declined it. Beat the others by picking better, not by "
    "picking more.\n\n"
    "EVENT MERIT GATE: Trade the event, not the number. A gap between your "
    "probability and the market's is not by itself a reason to trade -- most large "
    "gaps are your own error, not the crowd's, and the crowd has already priced the "
    "obvious. How much that applies depends on the market in front of you. Kalshi "
    "prices measure well calibrated overall (Brier about 0.02 at close), and "
    "calibration improves monotonically with volume within every horizon, so in a "
    "heavily traded market close to resolution a large gap is almost certainly your "
    "error -- treat the price as the better estimate and PASS. The same study finds "
    "long-dated markets never reach a 0.05 Brier no matter how many traders arrive, "
    "and that time to resolution, not participation, is the binding constraint. So a "
    "thinly traded market far from resolution is where a defensible edge can still "
    "exist. Read the volume and depth on each candidate line and weigh your "
    "disagreement accordingly. This tells you where to look, not what to conclude: "
    "thin markets are often thin because the event is genuinely ambiguous or hard to "
    "resolve, and they fill worse, so the burden of proof below is unchanged. "
    "Before opening a position, state three things: (1) the concrete "
    "mechanism that decides this event and the specific dates or releases that drive "
    "it; (2) why your read is better informed than the market's on THIS event -- a "
    "source the price has not absorbed yet, a rule the crowd is misapplying, a "
    "structural reason the price is stale; (3) what would prove you wrong, and at "
    "what level you would exit. If you cannot name a reason your view beats the "
    "market's, you do not have an edge, you have a disagreement: PASS. Prefer a "
    "smaller number of well-understood events to broad coverage of thin gaps -- your "
    "profit comes from being right where you genuinely know more, and every trade "
    "pays a spread and a fee whether or not it was worth taking.\n\n"
    "SIZING: For every NEW position, call place_trade with both your calibrated "
    "P(YES) as model_probability and exactly one sizing_mode: quarter_kelly "
    "(25% Kelly, 50% market shrinkage, 5% account cap) or edge_kelly "
    "(50% Kelly, 10 percentage-point edge minimum, 25% market shrinkage, "
    "8% account cap). The tool calculates the final quantity from the live "
    "ask and current account value; never invent a quantity larger than its "
    "result. For a pure exit, use sizing_mode='close' and do not increase the "
    "position. For an exact close, use the currently held quantity, including "
    "any fractional contracts. CLOSE ACCOUNTING: buying the opposite binary "
    "contract pays $1.00 per matched pair, so close P&L is quantity × (1 - "
    "existing average entry - live opposite ask) minus fees. Do not call the "
    "close order's gross cash outlay an additional loss or compare it directly "
    "with the original cost basis.\n\n"
    "EXECUTION CONTRACT: A final BUY YES, BUY NO, SELL YES, SELL NO, or CLOSE is "
    "a commitment to act in this shadow account. Call place_trade BEFORE writing "
    "that final action. If the tool rejects or cannot fill the order, say so plainly "
    "in the final thesis and do not present the recommendation as an executed trade. "
    "A final HOLD or PASS means no paper order. This is a paper-trading simulation: "
    "never say that live trading is disabled. The recorded place_trade result, not "
    "the live-trading setting, is the authoritative outcome of an attempted simulation.\n\n"
    "CRITICAL RULES COMPLIANCE: Read the 'Resolution rules' provided for every "
    "market carefully before taking any action. Market resolution is governed "
    "strictly by the venue's legal resolution rules, definitions, and specific "
    "exclusions (e.g. specific persons, titles, or events may be explicitly "
    "excluded from resolving YES/NO). Never trade on headline text alone without "
    "verifying that the underlying event satisfies all stated resolution rules. "
    "If your past reasoning relied on an excluded event or individual, re-evaluate "
    "and close or adjust the position accordingly. Before trading on news, confirm "
    "the event date falls within THIS ticker's resolution window (shown above) -- "
    "Kalshi often has several tickers for the same question on different date ranges "
    "(e.g. '-26APR' vs '-26MAY22-26SEP'), so evidence from before this window opened "
    "doesn't resolve this contract. A new position without visible resolution rules, "
    "or without a dated fact inside the observation window, is a PASS -- not a guess.\n\n"
    "STRATEGY MENU: Select exactly one strategy for this cycle and report it in "
    "the thesis. (1) EVIDENCE_EDGE: a fresh, independently sourced probability "
    "versus executable price; use Kelly sizing only after the research-quality and "
    "rules gates pass. (2) CATALYST_EDGE: a dated upcoming event inside the exact "
    "resolution window with a concrete causal path; do not extrapolate stale news. "
    "(3) ORDERBOOK_ARBITRAGE_RESEARCH: call orderbook_arbitrage with realistic fees, "
    "a latency_bps_per_leg allowance, and requested_quantity to simulate a live "
    "fill-or-kill pair. Verify identical resolution rules and real-world non-atomic "
    "leg risk; never submit a single leg as 'arbitrage'. (4) POSITION_RISK_REDUCTION: reassess an existing "
    "holding and use sizing_mode='close' only to reduce it. If no strategy clears its "
    "gate, choose PASS. A broader menu is not permission to manufacture an edge.\n\n"
    "MANDATORY UNIFIED THESIS TEMPLATE:\n"
    "DISCREPANCY DISCIPLINE: Kalshi Research measured every resolved market on the venue "
    "(2,243,741 markets, 2021 to mid-2026) and found prices behave like genuine probabilities: "
    "Brier score falls from roughly 0.08-0.09 at a 3-month horizon to about 0.02 at close, naive "
    "accuracy rises from 88.3% to 97.2%, and reliability curves track the diagonal in nearly every "
    "category. Calibration also improves monotonically with volume within every horizon. A large "
    "disagreement with a liquid, near-resolution price is therefore far more likely to be your "
    "error than the market's. When assessing an edge >15pp, actively challenge your thesis, verify "
    "you are not missing unindexed breaking events, and damp overconfident probabilities (80-95% "
    "range). The same study finds long-dated markets never reach a 0.05 Brier at any level of "
    "participation, so distance from resolution -- not your conviction -- is what leaves room for "
    "a genuine edge.\n\n"
    "In your final answer, ALL models MUST begin with this research delta, then use the exact 4-section markdown structure:\n\n"
    "### 0. Research Delta\n"
    "- **Strategy**: [EVIDENCE_EDGE / CATALYST_EDGE / ORDERBOOK_ARBITRAGE_RESEARCH / POSITION_RISK_REDUCTION] (never use N/A)\n"
    "- **New evidence**: [new, dated sources checked this cycle, including URL/domain and why they change or do not change the view; write 'No material new evidence' when none]\n"
    "- **Belief update**: [previous probability -> current probability, or 'No material change']\n"
    "- Do not paraphrase unchanged prior sections. If evidence, probability, action, catalysts, and invalidation are unchanged, state that once and keep the remaining sections concise.\n\n"
    "### 1. Decision & Execution\n"
    "- **Action**: [BUY YES / BUY NO / CLOSE / HOLD / PASS]\n"
    "- **Market & Venue**: [<ticker>] on [<Kalshi / Polymarket>] -- always name the\n"
    "  market you analysed most closely, even when you PASS or HOLD. The Action line\n"
    "  carries the decision; this line carries the market. A named PASS is scored for\n"
    "  calibration and is how you prove judgement without risking a cent, so it counts\n"
    "  in your favour. Never write 'No new position' or 'N/A' here.\n"
    "- **Order Sizing**: [<Quarter Kelly 5% cap / Edge Kelly 8% cap / Close>, <quantity> contracts @ $<price>, notional: $<total>] (write 'No new order' for HOLD/PASS; never use N/A)\n\n"
    "### 2. Resolution Rules & Compliance Audit\n"
    "- **Rules Verification**: [Explicit confirmation that the event/entity qualifies under venue criteria with zero exclusions] (or 'No new contract assessed'; never use N/A)\n"
    "- **Observation Window**: [Window start -> close check] (or 'No new contract assessed'; never use N/A)\n\n"
    "### 3. Model Edge & Valuation\n"
    "- **Model Probability**: [XX%] vs **Market Price**: [XX%] (Edge: [+/-XX%]) -- state\n"
    "  both numbers on every cycle, including a PASS. If you name the side, say which\n"
    "  ('5% YES', '95% NO'); both are read exactly as written. This line is what gets\n"
    "  scored against the outcome, so an omitted probability is a cycle of your skill\n"
    "  left unmeasured. Never use N/A.\n"
    "- **Information Asymmetry / Rationale**: [Why the crowd is mispriced / what verified evidence drives this stance]\n\n"
    "### 4. Catalysts & Invalidation\n"
    "- **Key Catalysts / Dates**: [Upcoming milestones / deadlines]\n"
    "- **Invalidation Trigger**: [Exact condition or evidence that invalidates this thesis]"
)


def _recalled_notes_block(agent_id: str, limit: int = 20) -> str:
    """The agent's own notes, injected straight into the cycle prompt.

    Recall used to depend on the model choosing to spend one of its scarce
    tool steps calling manage_notes. Most never did: gemma, gpt-oss and qwen
    each wrote notes across dozens of cycles and read them back exactly zero
    times, so memory accumulated and was never used -- writing without
    reading is not self-improvement. Surfacing notes here makes recall
    automatic and costs no tool step; manage_notes stays available for
    search, edit, and recording new observations.
    """
    try:
        from analyzing_llm_rationale import benchmark_tools

        notes = benchmark_tools._load_notes().get(agent_id, [])
    except Exception:
        return ""
    if not notes:
        return ""
    # Newest last: the model reads top-to-bottom and recency matters most.
    ordered = sorted(notes, key=lambda n: str(n.get("created_at") or ""))[-limit:]
    lines = []
    for note in ordered:
        text = " ".join(str(note.get("text") or "").split())
        if not text:
            continue
        stamp = str(note.get("created_at") or "")[:10]
        tags = ", ".join(str(t) for t in (note.get("tags") or []))
        suffix = f"  [{tags}]" if tags else ""
        lines.append(f"  - ({stamp}) {text}{suffix}")
    if not lines:
        return ""
    return (
        "=== Your notes from previous cycles ===\n"
        + "\n".join(lines)
        + "\nThese are your own prior conclusions, not external evidence. Re-verify "
        "anything time-sensitive before acting on it, and use manage_notes to add or "
        "correct entries as your view changes."
    )


LEADERBOARD_URL = os.environ.get(
    "AGENT_TRADING_LEADERBOARD_URL", "https://foresea.ink/agent-trading/board")
LEADERBOARD_TIMEOUT_S = float(os.environ.get("AGENT_TRADING_LEADERBOARD_TIMEOUT_S", "10"))


def _leaderboard_block(agent_id: str) -> str:
    """Standings for every agent, with this one's own position marked.

    Each model runs its cycle with only its own shadow account downloaded, so
    it has never had any way to know how it is doing relative to anyone else.
    The published board is the one place the whole field is visible.

    Returns "" on any failure. A cycle must not be lost because the
    scoreboard was unreachable.
    """
    try:
        response = requests.get(LEADERBOARD_URL, timeout=LEADERBOARD_TIMEOUT_S)
        if response.status_code != 200:
            return ""
        rows = (response.json() or {}).get("leaderboard") or []
    except Exception:
        return ""
    ranked = sorted(
        (r for r in rows if isinstance(r, dict) and r.get("agent_id")),
        key=lambda r: -(r.get("return_pct") if isinstance(r.get("return_pct"), (int, float)) else -999),
    )
    if not ranked:
        return ""
    lines = ["=== Standings: every agent trading this board ==="]
    mine = None
    for position, row in enumerate(ranked, start=1):
        name = str(row.get("agent_id"))
        is_me = name == agent_id
        if is_me:
            mine = position
        ret = row.get("return_pct")
        ret_s = f"{ret:+.2f}%" if isinstance(ret, (int, float)) else "n/a"
        trades = row.get("trade_count") or 0
        win = row.get("win_rate")
        win_s = f"{win:.0%} of {row.get('realized_count') or 0} closed" if isinstance(win, (int, float)) else "no closed trades"
        lines.append(
            "  %2d. %-24s %9s  %3s trades, won %s%s"
            % (position, name[:24], ret_s, trades, win_s, "   <-- YOU" if is_me else "")
        )
    if mine is not None:
        leader = ranked[0]
        gap = None
        if isinstance(leader.get("return_pct"), (int, float)) and isinstance(
            ranked[mine - 1].get("return_pct"), (int, float)
        ):
            gap = leader["return_pct"] - ranked[mine - 1]["return_pct"]
        lines.append("")
        if mine == 1:
            chaser = ranked[1] if len(ranked) > 1 else None
            behind = ""
            if chaser and isinstance(chaser.get("return_pct"), (int, float)):
                margin = ranked[0]["return_pct"] - chaser["return_pct"]
                behind = (" %s is %.2fpp behind you on %s trades and is reading "
                          "this same board." % (str(chaser.get("agent_id"))[:24],
                                                margin, chaser.get("trade_count") or 0))
            lines.append(
                "You are first. One bad cycle hands it over --" + (behind or
                " the agent below you is closer than the gap looks."))
        else:
            # Name the agent immediately above. "Eighth of eight" is abstract;
            # "0.47pp behind qwen, who got there on three trades" is a target.
            rival = ranked[mine - 2]
            rival_gap = None
            if isinstance(rival.get("return_pct"), (int, float)) and isinstance(
                ranked[mine - 1].get("return_pct"), (int, float)
            ):
                rival_gap = rival["return_pct"] - ranked[mine - 1]["return_pct"]
            gap_s = f", {gap:.2f}pp off the lead" if gap is not None else ""
            lines.append(
                "You are %d of %d%s. Directly above you is %s%s -- pass them "
                "first. Every place is taken from somebody: they do not fall "
                "because you traded, they fall because you were right and they "
                "were not."
                % (mine, len(ranked), gap_s, str(rival.get("agent_id"))[:24],
                   f" by {rival_gap:.2f}pp" if rival_gap is not None else "")
            )
        # The standings are also evidence, and the evidence is blunt: on this
        # board the most active agent is last. Say so, because "trade more to
        # climb" is the obvious inference and it is the wrong one -- llama is
        # bottom on 25 trades while the leader got there on 14.
        traded = [r for r in ranked if (r.get("trade_count") or 0) > 0]
        if len(traded) >= 3:
            busiest = max(traded, key=lambda r: r.get("trade_count") or 0)
            lines.append(
                "Read the column before you act on it: %s has traded the most "
                "(%s) and sits %d of %d. Rank comes from being right, not from "
                "being busy -- a losing trade costs you places."
                % (str(busiest.get("agent_id"))[:24], busiest.get("trade_count"),
                   ranked.index(busiest) + 1, len(ranked))
            )
        hand = _own_capability_line(agent_id)
        if hand:
            lines.append(hand)
    return "\n".join(lines)


def _own_capability_line(agent_id: str) -> str:
    """What this model has more of than the field, so it can play to it.

    The eight agents are not interchangeable: context windows run from 65k to
    1,048,576, a sixteenfold spread. A model with a million tokens can hold
    every source it finds and reason across all of them at once; one with 65k
    has to choose what to read and is wasting its cycle imitating the others.
    Telling each what it actually has is how the best of a particular model
    shows up, rather than eight models converging on the same shallow pass.
    """
    try:
        from analyzing_llm_rationale.config import load_model_configs

        configs = load_model_configs(Path("configs/models.yaml"))
    except Exception:
        return ""
    windows: Dict[str, int] = {}
    for name, cfg in (configs or {}).items():
        value = getattr(cfg, "context_window_tokens", None)
        if value is None and isinstance(cfg, dict):
            value = cfg.get("context_window_tokens")
        if isinstance(value, int) and value > 0:
            windows[name] = value
    mine = windows.get(agent_id)
    if not mine or len(windows) < 3:
        return ""
    ordered = sorted(windows.values(), reverse=True)
    rank = ordered.index(mine) + 1
    largest, smallest = ordered[0], ordered[-1]
    if mine >= largest:
        return (
            "Your hand: %s tokens of context, the largest of any agent here. "
            "You can hold every source you pull and weigh them together in one "
            "pass -- read more than the others can and let that be the edge."
            % format(mine, ",")
        )
    if mine <= smallest:
        return (
            "Your hand: %s tokens of context, the smallest here against %s at "
            "the top. You cannot out-read this field, so do not try -- pick the "
            "one market you can genuinely settle and be right about it."
            % (format(mine, ","), format(largest, ","))
        )
    return (
        "Your hand: %s tokens of context, %d of %d in this field. Enough to "
        "read deeply on a couple of markets, not enough to cover them all -- "
        "spend it where the edge hurdle is lowest."
        % (format(mine, ","), rank, len(windows))
    )


def _assemble_question(portfolio_block: str, candidates_block: str,
                       agent_id: Optional[str] = None) -> str:
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    time_anchor = (
        f"=== Current simulation time ===\n{as_of}\n"
        "Treat dates before this timestamp as historical, never as upcoming catalysts. "
        "Use a catalyst only when it is still ahead and inside the selected contract's resolution window."
    )
    notes_block = _recalled_notes_block(agent_id or MODEL)
    standings_block = _leaderboard_block(agent_id or MODEL)
    return "\n\n".join(filter(None, [
        time_anchor, portfolio_block, notes_block, standings_block,
        candidates_block, _TRADING_INSTRUCTION,
    ]))


def _trim_block_to_lines(block: str, budget: int) -> str:
    """Drop whole trailing lines from `block` until it fits `budget` chars.

    Never cuts a line mid-way. A garbled, half-visible candidate or position
    line is worse than one omitted cleanly -- and cutting mid-line is exactly
    what silently destroyed almost an entire candidates block (every
    Polymarket candidate included) down to a mid-word fragment of the first
    held position, any time a portfolio had held positions and the combined
    text ran over budget (observed live, 2026-08-18: agents effectively
    never saw the new candidates they were supposedly being offered).
    """
    if budget <= 0:
        return ""
    if len(block) <= budget:
        return block
    lines = block.split("\n")
    # Reserve room for the marker before filling. It used to be appended after
    # the budget was already spent, so a trimmed block could come back longer
    # than the caller asked for. _build_question treats that as "still too
    # long" and moves on to its next step, which drops the candidates block
    # entirely -- so a block landing within ~30 chars of the limit cost the
    # agent every market it was being offered, silently, rather than costing
    # it one line. Lengthening a candidate line by 29 characters was enough to
    # trigger it.
    marker = "  … (more omitted for space)"
    reserved = max(0, budget - (len(marker) + 1))
    kept: List[str] = []
    used = 0
    for line in lines:
        add = len(line) + (1 if kept else 0)
        if used + add > reserved:
            break
        kept.append(line)
        used += add
    if len(kept) < len(lines) and kept:
        kept.append(marker)
    return "\n".join(kept)


def _build_question(portfolio_block: str, candidates_block: str,
                    agent_id: Optional[str] = None) -> str:
    question = _assemble_question(portfolio_block, candidates_block, agent_id)
    if len(question) <= MAX_QUESTION_CHARS:
        return question
    # Over MAX_QUESTION_CHARS -- server.py no longer caps AgentAnalyzeRequest
    # .question at all, so this is a pure sanity backstop, not a routine path.
    # If it ever triggers, trim in priority order, cheapest / least
    # decision-relevant content first, and whenever a
    # block of markets has to shrink, drop whole lines (see
    # _trim_block_to_lines) rather than cutting mid-line.
    # 1) Drop the optional previous-cycle research and decision state first --
    # the model doesn't strictly need it to assess current candidates, and the
    # portfolio state already reflects whatever it decided from it. This is
    # tried BEFORE touching candidates_block, since the candidates being
    # visible at all is the entire point of offering them.
    trimmed_portfolio = portfolio_block.split("\n=== Research carried forward from your prior cycle ===")[0]
    question = _assemble_question(trimmed_portfolio, candidates_block, agent_id)
    if len(question) <= MAX_QUESTION_CHARS:
        return question
    # 2) Still too long -- trim the candidates block by whole lines. Held
    # positions are listed before new candidates in candidates_block, so
    # trimming from the end drops new candidates first (an agent can still
    # act on what it already holds without seeing anything new this cycle).
    overage = len(question) - MAX_QUESTION_CHARS
    trimmed_candidates = _trim_block_to_lines(candidates_block, len(candidates_block) - overage)
    question = _assemble_question(trimmed_portfolio, trimmed_candidates, agent_id)
    if len(question) <= MAX_QUESTION_CHARS:
        return question
    # 3) Candidates already emptied and it's still too long -- the
    # portfolio's own position list (unbounded -- an agent can hold any
    # number of tickers) is the overage. Trim it the same whole-line way.
    overage = len(question) - MAX_QUESTION_CHARS
    trimmed_portfolio = _trim_block_to_lines(trimmed_portfolio, len(trimmed_portfolio) - overage)
    question = _assemble_question(trimmed_portfolio, "", agent_id)
    if len(question) <= MAX_QUESTION_CHARS:
        return question
    # 4) Absolute last resort -- guarantee the limit is never exceeded no
    # matter how large a single remaining line is. Keep the instruction
    # intact (the model needs it every cycle) and hard-clamp the rest.
    # The clock anchor is fixed prompt context too; reserve space for it so
    # the absolute fallback remains a real hard cap.
    budget = max(0, MAX_QUESTION_CHARS - len(_assemble_question("", "", agent_id)) - 2)
    if len(trimmed_portfolio) > budget:
        trimmed_portfolio = trimmed_portfolio[: max(0, budget - 1)].rstrip() + "…"
    return _assemble_question(trimmed_portfolio, "", agent_id)


_STRATEGY_ALIASES = {
    "EVIDENCE_EDGE": "evidence_edge",
    "CATALYST_EDGE": "catalyst_edge",
    "ORDERBOOK_ARBITRAGE_RESEARCH": "orderbook_arbitrage_research",
    "POSITION_RISK_REDUCTION": "position_risk_reduction",
}

_THESIS_MARKET_RE = re.compile(
    r"\*\*Market\s*&\s*Venue\*\*\s*:\s*\[?\s*([^\]\n]+?)\s*\]?\s+on\s+\[?\s*"
    r"(Kalshi|Polymarket)\b",
    re.IGNORECASE,
)
# Tolerant of how models actually write this line. deepseek-v4-flash wrote
# "**Model Probability**: 40% (no-change) vs **Market Price**: 48.5% mid / 52%
# NO ask" -- a correct, fully specified figure that the old pattern rejected
# because a parenthetical sat between the percentage and "vs". Reconciliation
# then refused the trade for having no calibrated P(YES), so a researched,
# decided position was never opened and the agent showed zero positions.
# The side qualifier is captured, not merely tolerated. Models name the side
# they priced -- gpt-oss-120b wrote "5% YES (95% NO)", glm-5-3-flash wrote
# "10% YES" -- and the old pattern rejected a bare qualifier outright, so two
# fully specified forecasts were discarded every cycle. Skipping the qualifier
# instead would be worse than rejecting it: "95% NO" stored as P(YES) inverts
# the forecast and poisons the calibration record this feeds.
_THESIS_PROBABILITY_RE = re.compile(
    r"\*{0,2}Model\s+Probability\*{0,2}\s*:?\s*\*{0,2}\s*\[?\s*~?\s*"
    r"(?P<model>\d+(?:\.\d+)?)\s*%\s*\]?"
    r"(?P<model_side>\s+(?:YES|NO)\b)?"
    r"(?:\s*\([^)]{0,60}\))?"          # optional qualifier, e.g. "(no-change)"
    r"(?P<model_side_alt>\s+(?:YES|NO)\b)?"
    r"\s*(?:vs\.?|versus)\s+"
    r"\*{0,2}Market\s+Price\*{0,2}\s*:?\s*\*{0,2}\s*\[?\s*~?\s*"
    r"(?P<market>\d+(?:\.\d+)?)\s*%"
    r"(?P<market_side>\s+(?:YES|NO)\b)?",
    re.IGNORECASE,
)


def _yes_probability(number: Any, *sides: Any) -> Optional[float]:
    """Read one side-aware percentage from a thesis line as P(YES)."""
    probability = _as_probability(number)
    if probability is None:
        return None
    side = next((str(s).strip().upper() for s in sides if s and str(s).strip()), "")
    return round(1.0 - probability, 6) if side == "NO" else probability


def _thesis_probability_pair(match: Any) -> tuple:
    """Return (model P(YES), market P(YES)) from a probability-line match."""
    return (
        _yes_probability(
            match.group("model"), match.group("model_side"), match.group("model_side_alt")
        ),
        _yes_probability(match.group("market"), match.group("market_side")),
    )
_THESIS_ACTION_RE = re.compile(
    r"\*\*Action\*\*\s*:\s*\[?\s*([^\]\n]+?)\s*(?=\]|\n|$)", re.IGNORECASE
)
_THESIS_EVIDENCE_RE = re.compile(r"\*\*New\s+evidence\*\*\s*:\s*([^\n]+)", re.IGNORECASE)
_THESIS_SIZING_RE = re.compile(
    r"\*\*Order\s+Sizing\*\*\s*:\s*([^\n]+)", re.IGNORECASE
)
_LOOSE_ACTION_MARKET_RE = re.compile(
    r"\b(BUY\s+(?:YES|NO)|SELL\s+(?:YES|NO)|CLOSE)\b\s+(?:on\s+)?[\"'`]?"
    r"([A-Za-z0-9][A-Za-z0-9_.-]{2,119})[\"'`]?\s+on\s+(Kalshi|Polymarket)\b",
    re.IGNORECASE,
)
# Same tolerance for the fallback: markdown emphasis, a colon, and a "~"
# approximation marker all appear in real theses between the label and the
# number, and each one used to defeat this pattern.
_LOOSE_MODEL_PROBABILITY_RE = re.compile(
    r"(?:\*{0,2}model\s+probability\*{0,2}|\*{0,2}p\(yes\)\*{0,2}|calibrated\s+p\(yes\))"
    r"\s*(?:of|is|=|:)?\s*\*{0,2}\s*~?\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def _as_probability(value: Any) -> Optional[float]:
    """Coerce a thesis/tool probability to [0, 1] without guessing units."""
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if probability > 1.0:
        probability /= 100.0
    return round(probability, 6) if 0.0 <= probability <= 1.0 else None


def _normalise_forecast_ticker(value: Any) -> str:
    return str(value or "").strip().strip("`[] ").replace(" ", "")


def _declared_thesis_execution(thesis: str) -> Optional[Dict[str, Any]]:
    """Return a bounded executable intent from a final thesis, if it has one.

    A paper agent's final recommendation is an operational commitment, not
    decorative prose.  This deliberately accepts only BUY/SELL/CLOSE actions
    tied to a named venue and ticker.  It never guesses a market, direction,
    price, or Kelly mode from free text.
    """
    text = str(thesis or "")
    action_match = _THESIS_ACTION_RE.search(text)
    market_match = _THESIS_MARKET_RE.search(text)
    action = action_match.group(1).strip().upper() if action_match else ""
    ticker = _normalise_forecast_ticker(market_match.group(1)) if market_match else ""
    platform = market_match.group(2).lower() if market_match else ""

    if not (action and ticker and platform):
        loose = _LOOSE_ACTION_MARKET_RE.search(text)
        if not loose:
            return None
        action = loose.group(1).strip().upper()
        ticker = _normalise_forecast_ticker(loose.group(2))
        platform = loose.group(3).lower()

    if action not in {"BUY YES", "BUY NO", "SELL YES", "SELL NO", "CLOSE"}:
        return None

    probabilities = _THESIS_PROBABILITY_RE.search(text)
    probability = _thesis_probability_pair(probabilities)[0] if probabilities else None
    if probability is None:
        loose_probability = _LOOSE_MODEL_PROBABILITY_RE.search(text)
        probability = _as_probability(loose_probability.group(1)) if loose_probability else None

    sizing_text = _THESIS_SIZING_RE.search(text)
    sizing_value = sizing_text.group(1).lower() if sizing_text else text.lower()
    sizing_mode = (
        "edge_kelly" if "edge kelly" in sizing_value
        else "quarter_kelly" if "quarter kelly" in sizing_value
        else None
    )
    return {
        "action": action,
        "ticker": ticker,
        "platform": platform,
        "model_probability": probability,
        "sizing_mode": sizing_mode,
    }


def _candidate_for_execution(
    candidates: List[Dict[str, Any]], platform: str, ticker: str
) -> Optional[Dict[str, Any]]:
    """Find the exact live market the agent saw during this cycle."""
    target = _normalise_forecast_ticker(ticker)
    for candidate in candidates:
        candidate_platform = str(candidate.get("platform") or "").strip().lower()
        candidate_ticker = _normalise_forecast_ticker(candidate.get("ident"))
        if candidate_platform == platform and candidate_ticker == target:
            return candidate
    return None


def _has_trade_attempt(transcript: List[Dict[str, Any]]) -> bool:
    return any(
        str(step.get("action") or step.get("tool") or "").strip().lower()
        in {"place_trade", "place_order", "buy", "buy_order"}
        for step in transcript or []
        if isinstance(step, dict)
    )


def _recorded_trade_attempt_result(transcript: List[Dict[str, Any]]) -> tuple[str, str]:
    """Derive a bounded paper outcome and public-safe detail from tool output.

    A failed tool invocation is not necessarily a risk-guard rejection. Only
    the explicit ``rejected`` shape is a guardrail decision; input, quote, and
    unexpected tool failures must remain visible as execution errors.
    """
    for step in reversed(transcript or []):
        if not isinstance(step, dict):
            continue
        if str(step.get("action") or step.get("tool") or "").strip().lower() not in {
            "place_trade", "place_order", "buy", "buy_order"
        }:
            continue
        observation = step.get("observation")
        if isinstance(observation, str):
            try:
                observation = json.loads(observation)
            except (TypeError, ValueError):
                # Versions before compact tool observations cut a rich JSON
                # trade response at a character boundary. Recover a confirmed
                # fill only when the retained prefix contains both `ok: true`
                # and a positive filled quantity; do not guess any other
                # outcome from malformed legacy data.
                ok = bool(re.search(r'"ok"\s*:\s*true\b', observation, re.IGNORECASE))
                quantity = re.search(
                    r'"filled_quantity"\s*:\s*([0-9]+(?:\.[0-9]+)?)', observation
                )
                if ok and quantity and float(quantity.group(1)) > 0:
                    return "filled", "Paper fill confirmed from the retained execution summary."
                return "attempted_unknown", "The tool result was not readable."
        if not isinstance(observation, dict):
            return "attempted_unknown", "The tool result was not structured."
        execution = observation.get("execution")
        if not isinstance(execution, dict):
            execution = {}
        try:
            filled_quantity = float(execution.get("filled_quantity") or 0)
        except (TypeError, ValueError):
            filled_quantity = 0.0
        if bool(observation.get("ok")) and filled_quantity > 0:
            return "filled", "The agent called the guarded paper-trade tool and its recorded order filled."
        if bool(observation.get("ok")):
            return "unfilled", "The agent called the guarded paper-trade tool, but no executable paper fill was recorded."
        if bool(observation.get("rejected")):
            return "rejected", str(observation.get("reason") or "risk_guard")
        detail = str(
            observation.get("message")
            or observation.get("reason")
            or observation.get("error")
            or "The paper-trade tool failed before a guardrail decision."
        )
        return "error", detail
    return "attempted_unknown", "The agent called the guarded paper-trade tool; inspect its tool transcript for the exact result."


def _recorded_trade_attempt_outcome(transcript: List[Dict[str, Any]]) -> str:
    """Backward-compatible outcome-only view used by existing callers."""
    return _recorded_trade_attempt_result(transcript)[0]


_PAPER_EXECUTION_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?\*\*Paper execution\*\*:\s*[^\r\n]*(?:\r?\n)?"
)
_LIVE_DISABLED_CLOSE_CLAIM_RE = re.compile(
    r"(?im)^\s*[-*]\s*\*\*Action\*\*\s*:\s*\**\s*HOLD\s*\("
    r"[^\r\n]*?live\s+trading\s+(?:is|was)\s+disabled[^\r\n]*?\)\s*\**\s*$"
)


def _append_paper_execution(thesis: str, status: str, detail: str) -> str:
    """Make the public thesis honest about whether paper cash actually moved.

    Model prose is not an execution source of truth.  Replace any model-written
    paper-execution line with the durable tool result, and correct the known
    "live trading is disabled" close hallucination before publishing it.
    """
    normalized = _PAPER_EXECUTION_LINE_RE.sub("", str(thesis or "")).rstrip()
    normalized = _LIVE_DISABLED_CLOSE_CLAIM_RE.sub(
        "- **Action**: **CLOSE ATTEMPT RECORDED — see paper execution below.**",
        normalized,
    ).rstrip()
    return f"{normalized}\n\n**Paper execution**: {status} — {_excerpt(detail, 500)}"


def _close_order_args(
    conn: Any, agent_id: str, platform: str, ticker: str, expected_held_side: Optional[str] = None
) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Derive a reduce-only binary close; ambiguity is a safe non-execution."""
    rows = conn.execute(
        """
        SELECT side, quantity
        FROM agent_positions
        WHERE agent_id = ? AND platform = ? AND ticker = ? AND quantity > ?
        """,
        (agent_id, platform, ticker, benchmark_tools.MIN_POSITION_QUANTITY),
    ).fetchall()
    if len(rows) != 1:
        return None, None, "close_requires_exactly_one_open_position"
    held_side = str(rows[0]["side"] or "").lower()
    if held_side not in {"yes", "no"}:
        return None, None, "close_position_side_invalid"
    if expected_held_side is not None and held_side != expected_held_side:
        return None, None, "close_side_does_not_match_declared_sell"
    quantity = float(rows[0]["quantity"] or 0)
    if quantity <= benchmark_tools.MIN_POSITION_QUANTITY:
        return None, None, "close_position_quantity_invalid"
    return ("no" if held_side == "yes" else "yes"), quantity, None


def _reconcile_thesis_execution(
    *,
    agent_id: str,
    thesis: str,
    transcript: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Turn an unambiguous published trade decision into one guarded paper order.

    The ReAct loop remains the preferred execution path.  This deterministic
    backstop exists for providers that publish a final BUY/SELL/CLOSE thesis
    before calling the execution tool.  It never invents an opportunity: the
    declared venue/ticker must exactly match a current offered quote, and the
    normal ``place_trade`` guardrail chain remains authoritative.
    """
    decision = _declared_thesis_execution(thesis)
    if decision is None:
        # A model sometimes writes HOLD while still calling place_trade, then
        # incorrectly blames the unrelated live-trading gate.  Its transcript
        # is the durable simulation record, so surface that outcome even when
        # the final prose did not contain a reconcilable BUY/SELL/CLOSE action.
        if _has_trade_attempt(transcript) and _LIVE_DISABLED_CLOSE_CLAIM_RE.search(str(thesis or "")):
            outcome, detail = _recorded_trade_attempt_result(transcript)
            status = {
                "filled": "PAPER ORDER FILLED",
                "rejected": "PAPER ORDER REJECTED",
                "unfilled": "PAPER ORDER UNFILLED",
                "error": "PAPER ORDER ERROR",
                "attempted_unknown": "PAPER ORDER RESULT UNAVAILABLE",
            }[outcome]
            thesis_execution_reconciliations.add(1, {"outcome": f"attempted_without_action_{outcome}"})
            return _append_paper_execution(thesis, status, detail), {"outcome": outcome, "action": None}
        return thesis, None

    action = str(decision["action"])
    if _has_trade_attempt(transcript):
        outcome, detail = _recorded_trade_attempt_result(transcript)
        status = {
            "filled": "PAPER ORDER FILLED",
            "rejected": "PAPER ORDER REJECTED",
            "unfilled": "PAPER ORDER UNFILLED",
            "error": "PAPER ORDER ERROR",
            "attempted_unknown": "PAPER ORDER RESULT UNAVAILABLE",
        }[outcome]
        thesis_execution_reconciliations.add(1, {"outcome": f"already_attempted_{outcome}"})
        return _append_paper_execution(
            thesis, status, detail
        ), {"outcome": outcome, "action": action}

    with tracer.start_as_current_span("agent_trading.reconcile_thesis_execution") as span:
        span.set_attributes({
            "agent.model": agent_id,
            "thesis.action": action,
            "market.venue": str(decision["platform"]),
        })
        candidate = _candidate_for_execution(candidates, str(decision["platform"]), str(decision["ticker"]))
        if candidate is None:
            outcome = "not_offered"
            detail = "The declared ticker was not an exact live candidate in this cycle, so no paper order was guessed."
            span.set_attributes({"outcome": outcome, "trade.executed": False})
            thesis_execution_reconciliations.add(1, {"outcome": outcome})
            logger.info("agent thesis execution not reconciled model=%s reason=%s", agent_id, outcome)
            return _append_paper_execution(thesis, "NOT EXECUTED", detail), {"outcome": outcome, "action": action}

        if action.startswith("BUY "):
            side = action.rsplit(" ", 1)[1].lower()
            quantity = None
            sizing_mode = decision.get("sizing_mode")
            probability = decision.get("model_probability")
            if sizing_mode is None or probability is None:
                outcome = "missing_sizing_or_probability"
                detail = "A new paper position requires an explicit Kelly mode and calibrated P(YES); neither was inferred."
                span.set_attributes({"outcome": outcome, "trade.executed": False})
                thesis_execution_reconciliations.add(1, {"outcome": outcome})
                return _append_paper_execution(thesis, "NOT EXECUTED", detail), {"outcome": outcome, "action": action}
        else:
            expected_held_side = action.rsplit(" ", 1)[1].lower() if action.startswith("SELL ") else None
            with benchmark_tools._account_transaction() as conn:
                side, quantity, close_error = _close_order_args(
                    conn,
                    agent_id,
                    str(decision["platform"]),
                    str(decision["ticker"]),
                    expected_held_side,
                )
            if close_error:
                outcome = "close_not_executable"
                span.set_attributes({"outcome": outcome, "trade.executed": False})
                thesis_execution_reconciliations.add(1, {"outcome": outcome})
                return _append_paper_execution(thesis, "NOT EXECUTED", close_error), {"outcome": outcome, "action": action}
            sizing_mode = "close"
            probability = None

        try:
            price = MarketQuote.from_mapping(candidate).ask(str(side).upper())
        except (TypeError, ValueError) as exc:
            price = None
            quote_error = str(exc)
        else:
            quote_error = ""
        if price is None or price <= 0:
            outcome = "missing_live_ask"
            detail = quote_error or "No executable live ask was available for the declared paper-trade side."
            span.set_attributes({"outcome": outcome, "trade.executed": False})
            thesis_execution_reconciliations.add(1, {"outcome": outcome})
            return _append_paper_execution(thesis, "NOT EXECUTED", detail), {"outcome": outcome, "action": action}

        args: Dict[str, Any] = {
            "platform": decision["platform"],
            "ticker": decision["ticker"],
            "side": side,
            "price": price,
            "sizing_mode": sizing_mode,
        }
        if quantity is not None:
            args["quantity"] = quantity
        if probability is not None:
            args["model_probability"] = probability

        result = benchmark_tools.place_trade(
            args,
            benchmark_tools.ToolContext(agent_id=agent_id, model=agent_id, require_kelly_sizing=True),
        )
        transcript.append({
            "action": "place_trade",
            "args": args,
            "observation": benchmark_tools.observation(result),
            "source": "thesis_execution_reconciliation",
        })
        execution = result.get("execution") if isinstance(result, dict) else {}
        filled_quantity = float((execution or {}).get("filled_quantity") or 0)
        if bool(result.get("ok")) and filled_quantity > 0:
            outcome = "filled"
            detail = (
                f"Filled {filled_quantity:g} {str(side).upper()} contracts on {decision['ticker']} "
                f"at ${float(result.get('normalized_order', {}).get('price') or price):.3f}."
            )
            status = "PAPER ORDER FILLED"
        elif bool(result.get("rejected")):
            outcome = "rejected"
            detail = str(result.get("reason") or "risk_guard")
            status = "PAPER ORDER REJECTED"
        else:
            outcome = "unfilled"
            detail = str(result.get("reason") or (execution or {}).get("fill_status") or "no_fill")
            status = "PAPER ORDER UNFILLED"
        span.set_attributes({"outcome": outcome, "trade.executed": outcome == "filled"})
        thesis_execution_reconciliations.add(1, {"outcome": outcome})
        logger.info("agent thesis execution reconciled model=%s outcome=%s", agent_id, outcome)
        return _append_paper_execution(thesis, status, detail), {"outcome": outcome, "action": action}


def _candidate_probability(
    candidates: List[Dict[str, Any]], platform: str, ticker: str
) -> Optional[float]:
    for candidate in candidates:
        same_platform = str(candidate.get("platform") or "").strip().lower() == platform
        same_ticker = str(candidate.get("ident") or "").strip() == ticker
        if same_platform and same_ticker:
            return _as_probability(candidate.get("probability"))
    return None


def _candidate_matching_price(
    candidates: List[Dict[str, Any]], market_probability: Optional[float],
    tolerance: float = 0.005,
) -> Optional[Dict[str, Any]]:
    """Identify the analysed market from the market price the thesis quoted.

    Agents that PASS often name the decision instead of the market -- "No new
    position" -- while still quoting the market's own price ("vs Market Price:
    84.5%"). That price is an identifier: if exactly one candidate this cycle
    was trading there, the thesis is unambiguously about that market and the
    forecast is scoreable. If two candidates sit within the tolerance the
    reference is genuinely ambiguous, so nothing is recorded rather than
    attributing a forecast to the wrong contract.
    """
    if market_probability is None:
        return None
    matches = [
        candidate for candidate in candidates
        if (price := _as_probability(candidate.get("probability"))) is not None
        and abs(price - market_probability) <= tolerance
    ]
    return matches[0] if len(matches) == 1 else None


def _weather_forecast_metadata(
    candidates: List[Dict[str, Any]], platform: str, ticker: str
) -> tuple[Optional[str], Optional[str]]:
    """Attach weather provenance only when the thesis names an offered quote.

    This keeps retrospective calibration keyed to the contract metadata visible
    to the agent at forecast time rather than an inferred category or later
    market payload.
    """
    candidate = _candidate_for_execution(candidates, platform, ticker)
    if candidate is None:
        return None, None
    brief = weather_markets.classify_weather_market(candidate)
    if not brief.is_weather:
        return None, None
    return brief.market_type, brief.settlement_source


def _thesis_forecast_records(
    thesis: str,
    transcript: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    strategy: str,
) -> List[Dict[str, Any]]:
    """Extract only explicit, scoreable probabilities from a final thesis.

    A thesis is still publishable when it says PASS or has no probability, but
    it is not silently converted into a synthetic forecast.  The fallback for
    an actual ``place_trade`` call preserves the probability supplied to the
    guarded execution tool when a provider omitted the markdown field.
    """
    records: List[Dict[str, Any]] = []
    market = _THESIS_MARKET_RE.search(thesis or "")
    probabilities = _THESIS_PROBABILITY_RE.search(thesis or "")
    action = _THESIS_ACTION_RE.search(thesis or "")
    evidence = _THESIS_EVIDENCE_RE.search(thesis or "")
    if probabilities:
        model_probability, market_probability = _thesis_probability_pair(probabilities)
        if market:
            ticker = _normalise_forecast_ticker(market.group(1))
            platform = market.group(2).lower()
        else:
            # A PASS that quotes the market's price but names no market is
            # still a forecast, and an unscored PASS is the cheapest judgement
            # signal there is: it costs nothing and resolves anyway. Recover
            # the market from the quoted price when that is unambiguous.
            fallback = _candidate_matching_price(candidates, market_probability)
            ticker = _normalise_forecast_ticker(fallback.get("ident")) if fallback else ""
            platform = str(fallback.get("platform") or "").strip().lower() if fallback else ""
        if ticker and platform and model_probability is not None:
            records.append({
                "platform": platform,
                "ticker": ticker,
                "model_probability": model_probability,
                "market_probability": market_probability,
                "action": action.group(1).strip() if action else None,
                "strategy": strategy,
                "evidence_delta": _excerpt(evidence.group(1), 600) if evidence else None,
            })

    # A trade cannot reach the ledger without model_probability in benchmark
    # mode. Preserve that durable input if a provider failed to echo it in the
    # final markdown, but never overwrite a complete thesis record.
    existing = {(r["platform"], r["ticker"]) for r in records}
    for step in transcript or []:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("action") or step.get("tool") or "").strip().lower()
        if tool not in {"place_trade", "place_order", "buy", "buy_order"}:
            continue
        args = step.get("args") or {}
        if not isinstance(args, dict):
            continue
        platform = str(args.get("platform") or "kalshi").strip().lower()
        ticker = _normalise_forecast_ticker(args.get("ticker") or args.get("market_id"))
        model_probability = _as_probability(args.get("model_probability"))
        if not ticker or model_probability is None or (platform, ticker) in existing:
            continue
        market_probability = _candidate_probability(candidates, platform, ticker)
        records.append({
            "platform": platform,
            "ticker": ticker,
            "model_probability": model_probability,
            "market_probability": market_probability,
            "action": str(args.get("action") or args.get("side") or "TRADE").upper(),
            "strategy": strategy,
            "evidence_delta": None,
        })
        existing.add((platform, ticker))
    return records


def _persist_thesis_forecasts(
    conn: Any,
    agent_id: str,
    cycle_id: str,
    thesis: str,
    transcript: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    strategy: str,
    *,
    forecast_ts: Optional[str] = None,
) -> int:
    """Persist a thesis's explicit forecast exactly once for later scoring."""
    records = _thesis_forecast_records(thesis, transcript, candidates, strategy)
    if not records:
        return 0
    created_at = forecast_ts or datetime.now(timezone.utc).isoformat()
    for record in records:
        market_probability = record.get("market_probability")
        edge = (
            round(float(record["model_probability"]) - float(market_probability), 6)
            if market_probability is not None else None
        )
        weather_market_type, weather_settlement_source = _weather_forecast_metadata(
            candidates, str(record["platform"]), str(record["ticker"])
        )
        conn.execute(
            """
            INSERT INTO agent_thesis_forecasts
            (agent_id, cycle_id, forecast_ts, platform, ticker, model_probability,
             market_probability, edge, action, strategy, evidence_delta,
             weather_market_type, weather_settlement_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, cycle_id, platform, ticker) DO NOTHING
            """,
            (
                agent_id, cycle_id, created_at, record["platform"], record["ticker"],
                record["model_probability"], market_probability, edge, record.get("action"),
                record.get("strategy"), record.get("evidence_delta"),
                weather_market_type, weather_settlement_source,
            ),
        )
    thesis_forecasts_recorded.add(len(records), {"outcome": "recorded"})
    return len(records)


def _pending_thesis_forecast_markets(conn: Any, agent_id: str) -> List[tuple[str, str]]:
    """Return a bounded, deduplicated oldest-first resolution queue."""
    rows = conn.execute(
        """
        SELECT platform, ticker
        FROM agent_thesis_forecasts
        WHERE agent_id = ? AND resolved_outcome IS NULL
        GROUP BY platform, ticker
        ORDER BY MIN(forecast_ts) ASC
        LIMIT ?
        """,
        (agent_id, THESIS_FORECAST_RESOLUTION_CHECK_LIMIT),
    ).fetchall()
    return [(str(row["platform"]).lower(), str(row["ticker"])) for row in rows]


def _resolve_thesis_forecast_markets(
    markets: List[tuple[str, str]]
) -> Dict[tuple[str, str], tuple[int, str]]:
    """Resolve a small queue of thesis markets without holding the ledger lock."""
    resolved: Dict[tuple[str, str], tuple[int, str]] = {}
    for platform, ticker in markets:
        try:
            outcome = (
                market_data.resolve_polymarket(ticker)
                if platform == "polymarket"
                else market_data.resolve_kalshi(ticker)
            )
        except Exception:
            logger.debug("thesis forecast resolution lookup failed platform=%s ticker=%s", platform, ticker)
            thesis_forecast_resolution_checks.add(1, {"platform": platform or "unknown", "outcome": "failure"})
            continue
        if outcome is None:
            thesis_forecast_resolution_checks.add(1, {"platform": platform or "unknown", "outcome": "open"})
            continue
        try:
            actual = int(outcome)
        except (TypeError, ValueError):
            thesis_forecast_resolution_checks.add(1, {"platform": platform or "unknown", "outcome": "invalid"})
            continue
        if actual not in {0, 1}:
            thesis_forecast_resolution_checks.add(1, {"platform": platform or "unknown", "outcome": "invalid"})
            continue
        resolved[(platform, ticker)] = (actual, datetime.now(timezone.utc).isoformat())
        thesis_forecast_resolution_checks.add(1, {"platform": platform or "unknown", "outcome": "resolved"})
    return resolved


def _refresh_thesis_forecast_outcomes(
    conn: Any,
    agent_id: str,
    resolved_markets: Optional[Dict[tuple[str, str], tuple[int, str]]] = None,
) -> int:
    """Score prior thesis forecasts only after the held contract resolves.

    The settlement action is the immutable outcome source.  This intentionally
    avoids learning from mark-to-market P&L, voluntary exits, or a later model
    thesis; a probability becomes a training observation only on final market
    resolution.
    """
    started = time.perf_counter()
    with tracer.start_as_current_span("agent_trading.refresh_thesis_forecast_outcomes") as span:
        span.set_attribute("agent.id", agent_id)
        try:
            rows = conn.execute(
                """
                SELECT rowid, platform, ticker, forecast_ts, model_probability, market_probability
                FROM agent_thesis_forecasts AS forecast
                WHERE forecast.agent_id = ?
                  AND forecast.resolved_outcome IS NULL
                """,
                (agent_id,),
            ).fetchall()
            settlements = {}
            for settlement in conn.execute(
                """
                SELECT platform, ticker, outcome, ts
                FROM agent_actions
                WHERE agent_id = ? AND action_type = 'settlement'
                ORDER BY ts DESC
                """,
                (agent_id,),
            ):
                key = (str(settlement["platform"] or "").lower(), str(settlement["ticker"] or ""))
                settlements.setdefault(key, (settlement["outcome"], settlement["ts"]))
            scored = 0
            for row in rows:
                key = (str(row["platform"] or "").lower(), str(row["ticker"] or ""))
                known = settlements.get(key)
                actual: Optional[int] = None
                resolved_at: Optional[str] = None
                if known and str(known[1]) >= str(row["forecast_ts"]):
                    outcome = str(known[0] or "").strip().lower()
                    actual = 1 if outcome == "yes" else 0 if outcome == "no" else None
                    resolved_at = str(known[1])
                elif resolved_markets and key in resolved_markets:
                    actual, resolved_at = resolved_markets[key]
                if actual not in {0, 1} or not resolved_at:
                    continue
                model_probability = float(row["model_probability"])
                market_probability = row["market_probability"]
                market_brier = (
                    round((float(market_probability) - actual) ** 2, 8)
                    if market_probability is not None else None
                )
                conn.execute(
                    """
                    UPDATE agent_thesis_forecasts
                    SET resolved_outcome = ?, resolved_at = ?, brier_score = ?, market_brier_score = ?
                    WHERE rowid = ?
                    """,
                    (actual, resolved_at, round((model_probability - actual) ** 2, 8), market_brier, row["rowid"]),
                )
                scored += 1
            thesis_forecast_outcomes.add(scored, {"outcome": "scored"})
            span.set_attributes({"forecast.scored_count": scored, "outcome": "success"})
            return scored
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            thesis_forecast_outcomes.add(1, {"outcome": "failure"})
            logger.warning("agent thesis forecast outcome refresh failed agent=%s", agent_id, exc_info=True)
            raise
        finally:
            thesis_forecast_refresh_duration.record(time.perf_counter() - started)


def _selected_strategy(thesis: str) -> str:
    """Return a bounded, auditable strategy label from the final thesis.

    The choice is recorded only after a final answer is published, not from a
    speculative tool call. This keeps later strategy attribution tied to an
    actual decision rather than a research path that never traded.
    """
    match = re.search(r"\*\*Strategy\*\*\s*:\s*\[?\s*([A-Za-z_ -]+)", thesis or "", re.IGNORECASE)
    if not match:
        return "unreported"
    normalized = re.sub(r"[ -]+", "_", match.group(1).strip().upper())
    return _STRATEGY_ALIASES.get(normalized, "unreported")


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


def _safe_failure_detail(exc: BaseException) -> str:
    """Keep a bounded diagnostic without persisting credentials from an error."""
    detail = re.sub(r"\s+", " ", str(exc)).strip()
    detail = re.sub(
        r"(?i)(api[_ -]?key|authorization|token|secret|private[_ -]?key)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        detail,
    )
    return detail[:320] or exc.__class__.__name__


def _exception_messages(exc: BaseException) -> List[str]:
    """Return a short exception chain without ever persisting raw responses."""
    messages: List[str] = []
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(messages) < 6:
        seen.add(id(current))
        messages.append(str(current))
        cause = current.__cause__
        current = cause if isinstance(cause, BaseException) else None
    return messages


def _failure_detail(exc: BaseException) -> str:
    """Persist a bounded, actionable reason rather than a generic HTTP wrapper."""
    messages = _exception_messages(exc)
    for message in messages:
        match = re.search(r"(?<!\d)(4[0-9]{2}|5[0-9]{2})(?!\d)", message)
        if match:
            return f"Upstream provider returned HTTP {match.group(1)}."
    detail = " ".join(messages).lower()
    if "timeout" in detail or "timed out" in detail:
        return "Provider request timed out before completing."
    if "context" in detail and "limit" in detail:
        return "Provider rejected the request for exceeding its context limit."
    return _safe_failure_detail(exc)


def _is_timeout_exception(exc: BaseException) -> bool:
    """True when a timeout appears anywhere in the exception's cause chain.

    Text alone cannot answer this. A read timeout on the agent path is
    re-raised as "503: The '<model>' forecasting model is temporarily
    unavailable", so the 503 string test below matched first and filed every
    timeout as an outage. glm-5-3 was reported as an unavailable provider for
    days on that basis, while SCADS listed the route up with tools enabled --
    it was simply slower than the read timeout. Exception types do not lie the
    way a wrapper's message does, so ask them before matching text.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, requests.Timeout)):
            return True
        if type(current).__name__ in {"ProviderTimeoutError", "ReadTimeout", "ConnectTimeout"}:
            return True
        nxt = current.__cause__ or current.__context__
        current = nxt if isinstance(nxt, BaseException) else None
    return False


def _failure_kind(exc: BaseException) -> str:
    """Map volatile provider text into a stable maintenance-facing category."""
    detail = " ".join(_exception_messages(exc)).lower()
    # Ask the exception types before the wrapper's words: a wrapped timeout is
    # a timeout, not an outage, and mislabelling it hides a slow model behind
    # a provider-outage story that no amount of retrying can fix.
    if _is_timeout_exception(exc):
        return "provider_timeout"
    if "429" in detail or "rate limit" in detail or "max_parallel_requests" in detail:
        return "provider_rate_limited"
    if "503" in detail or "temporarily unavailable" in detail or "service unavailable" in detail:
        return "provider_unavailable"
    if "timeout" in detail or "timed out" in detail:
        return "provider_timeout"
    if "context" in detail and "limit" in detail:
        return "provider_context_limit"
    return "cycle_error"


def _scads_model_readiness(model: str) -> Tuple[Optional[str], Optional[str]]:
    """Return SCADS' current state for ``model`` without making it a dependency.

    ``None`` means the public status probe could not be trusted or reached, so
    the caller must continue normally.  A concrete unavailable state is useful
    operational evidence: the worker records a *deferred* cycle rather than a
    misleading failed tool run, then automatically tries again after SCADS'
    next status refresh reports the model up.
    """
    if not SCADS_STATUS_PRECHECK:
        return None, None
    try:
        config = load_model_configs(ROOT / "configs" / "models.yaml").get(model)
        target = (config.router_model_name if config else "").strip()
        if not target:
            return None, None
        response = requests.get(SCADS_STATUS_URL, timeout=SCADS_STATUS_TIMEOUT_S)
        response.raise_for_status()
        payload = response.json()
        groups = payload.get("models") if isinstance(payload, dict) else None
        records = (
            item
            for items in (groups or {}).values()
            if isinstance(items, list)
            for item in items
            if isinstance(item, dict)
        )
        for item in records:
            names = {str(item.get("name") or "").strip(), str(item.get("real_name") or "").strip()}
            if target in names:
                state = str(item.get("state") or "unknown").strip().lower()
                return state, f"SCADS status check reports {target} as {state}."
        return "not_listed", f"SCADS status check does not list configured model {target}."
    except Exception as exc:  # noqa: BLE001 - availability checks must fail open
        logger.info("SCADS status precheck unavailable for model=%s: %s", model, type(exc).__name__)
        return None, None


def _start_cycle_telemetry(run_id: str, agent_id: str, cycle_id: str, started_at: str) -> None:
    with benchmark_tools._account_transaction() as conn:
        conn.execute(
            """
            INSERT INTO agent_cycle_telemetry
            (run_id, agent_id, cycle_id, started_at, outcome, source)
            VALUES (?, ?, ?, ?, 'running', 'worker')
            """,
            (run_id, agent_id, cycle_id, started_at),
        )


def _finish_cycle_telemetry(
    run_id: str,
    *,
    outcome: str,
    duration_ms: int,
    summary: Optional[Dict[str, Any]] = None,
    failure_kind: Optional[str] = None,
    failure_detail: Optional[str] = None,
) -> None:
    summary = summary or {}
    with benchmark_tools._account_transaction() as conn:
        conn.execute(
            """
            UPDATE agent_cycle_telemetry
            SET finished_at = ?, outcome = ?, failure_kind = ?, failure_detail = ?,
                candidate_count = ?, tool_steps = ?, settled_count = ?,
                thesis_published = ?, forecast_records = ?, paper_execution_outcome = ?, duration_ms = ?,
                weather_candidates_offered = ?, weather_candidates_researched = ?, provider_model = ?
            WHERE run_id = ?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                outcome,
                failure_kind,
                failure_detail,
                summary.get("candidate_count"),
                summary.get("tool_steps"),
                summary.get("settled_count"),
                1 if summary.get("thesis_published") else 0,
                int(summary.get("forecast_records") or 0),
                summary.get("paper_execution_outcome"),
                duration_ms,
                int(summary.get("weather_candidates_offered") or 0),
                int(summary.get("weather_candidates_researched") or 0),
                summary.get("provider_model"),
                run_id,
            ),
        )


def run_cycle(model: str, *, cycle_id: Optional[str] = None) -> Dict[str, Any]:
    _assert_shadow_mode()
    _init_local_agent(model)

    agent_id = model
    cycle_id = cycle_id or benchmark_tools._current_cycle_id()

    # Settlement used to run only if a model tried to place another trade.
    # Run it before every decision instead, so a resolved result is available
    # to this model's next thesis even if it chooses to hold or pass.
    settled = benchmark_tools._settle_agent_open_positions(
        agent_id, benchmark_tools._risk_guard_policy()
    )
    # A thesis remains a forecast even if the agent chose HOLD/PASS and never
    # held the contract. Check only a bounded oldest-first queue, outside the
    # SQLite transaction, so outcome learning does not create API storms or
    # hold an account write lock across network I/O.
    with benchmark_tools._account_transaction() as conn:
        unresolved_markets = _pending_thesis_forecast_markets(conn, agent_id)
    resolved_forecast_markets = _resolve_thesis_forecast_markets(unresolved_markets)

    with benchmark_tools._account_transaction() as conn:
        _refresh_learning(conn, agent_id)
        _refresh_thesis_forecast_outcomes(conn, agent_id, resolved_forecast_markets)
        learning_block = _build_learning_block(conn, agent_id)
        held_positions = [
            (str(row["platform"] or "kalshi").lower(), row["ticker"])
            for row in conn.execute(
                "SELECT DISTINCT platform, ticker FROM agent_positions WHERE agent_id = ? AND quantity > ?",
                (agent_id, benchmark_tools.MIN_POSITION_QUANTITY),
            )
        ]
        last_cycle = conn.execute(
            "SELECT thesis, transcript_json FROM agent_cycles WHERE agent_id = ? ORDER BY ts DESC LIMIT 1",
            (agent_id,),
        ).fetchone()
        last_thesis = last_cycle["thesis"] if last_cycle else None
        last_transcript = last_cycle["transcript_json"] if last_cycle else None
        portfolio_block = _build_portfolio_block(
            conn, agent_id, last_thesis, learning_block, last_transcript
        )

    held_quotes = _requote_held(held_positions)
    known = {q.get("ident") for q in held_quotes if q.get("ident")}
    new_quotes = _discover_candidates(known)
    weather_candidates_offered = _weather_candidate_count(new_quotes)

    # Every agent-trading risk guard scales off FORESEA_AGENT_ACCOUNT_VALUE,
    # so it must reflect the account's real, current mark-to-market value
    # here -- computed after held_quotes exist, before any place_trade call
    # in this cycle's tool loop could read a guard.
    with benchmark_tools._account_transaction() as conn:
        os.environ["FORESEA_AGENT_ACCOUNT_VALUE"] = str(_current_account_value(conn, agent_id, held_quotes))
    _configure_max_order_notional()

    question = _build_question(
        portfolio_block, _build_candidates_block(held_quotes, new_quotes), agent_id,
    )

    report = asyncio.run(_call_agent_analyze(question))
    # The durable transcript retains raw tool output. Store only reader-ready
    # final copy in the cycle field so tool envelopes cannot masquerade as a
    # model thesis on the public board.
    report.thesis = agent_trading_stats.clean_thesis_display(report.thesis)
    all_candidates = [*held_quotes, *new_quotes]
    report.thesis, thesis_execution = _reconcile_thesis_execution(
        agent_id=agent_id,
        thesis=report.thesis,
        transcript=report.tool_transcript,
        candidates=all_candidates,
    )
    selected_strategy = _selected_strategy(report.thesis)
    weather_candidates_researched = _weather_research_call_count(report.tool_transcript)
    with tracer.start_as_current_span("agent_trading.record_strategy_selection") as span:
        outcome = "reported" if selected_strategy != "unreported" else "unreported"
        span.set_attributes({
            "agent.model": model,
            "strategy.name": selected_strategy,
            "outcome": outcome,
        })
        strategy_selections.add(1, {"strategy": selected_strategy, "outcome": outcome})

    candidates_offered = [q.get("ident") for q in (*held_quotes, *new_quotes)]
    with benchmark_tools._account_transaction() as conn:
        cycle_ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO agent_cycles
            (agent_id, cycle_id, ts, thesis, transcript_json, steps, truncated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                cycle_id,
                cycle_ts,
                report.thesis,
                json.dumps({
                    "candidates_offered": candidates_offered,
                    "selected_strategy": selected_strategy,
                    "tool_transcript": report.tool_transcript,
                }),
                len(report.tool_transcript),
                1 if len(report.tool_transcript) >= MAX_TOOL_STEPS else 0,
            ),
        )
        forecast_records = _persist_thesis_forecasts(
            conn,
            agent_id,
            cycle_id,
            report.thesis,
            report.tool_transcript,
            all_candidates,
            selected_strategy,
            forecast_ts=cycle_ts,
        )
        # A close placed during this cycle now becomes a durable learning
        # record for the next cycle. New entries are keyed to source action
        # IDs, so this is idempotent if a workflow retries.
        _refresh_learning(conn, agent_id)

    _broadcast_cycle_trades(model, report)

    print(
        f"agent-trading-tick done model={model} cycle={cycle_id} "
        f"steps={len(report.tool_transcript)} candidates={len(candidates_offered)} "
        f"weather_candidates={weather_candidates_offered} weather_researched={weather_candidates_researched} "
        f"settled={len(settled)}"
    )
    return {
        "candidate_count": len(candidates_offered),
        "tool_steps": len(report.tool_transcript),
        "settled_count": len(settled),
        "thesis_published": bool(str(report.thesis or "").strip()),
        "forecast_records": forecast_records,
        "paper_execution_outcome": (thesis_execution or {}).get("outcome"),
        "weather_candidates_offered": weather_candidates_offered,
        "weather_candidates_researched": weather_candidates_researched,
        # The provider may return a concrete revision/alias.  Preserve it for
        # audit and maintenance; do not infer it when the upstream omits it.
        "provider_model": getattr(report, "served_model_name", None),
    }


def _broadcast_cycle_trades(model: str, report: Any) -> None:
    """Broadcast any executed trades in this cycle to Discord and Telegram feeds."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from channel_broadcaster import ChannelBroadcaster
        broadcaster = ChannelBroadcaster(dry_run=False)
        for step in getattr(report, "tool_transcript", []):
            tool_name = step.get("tool") or step.get("action")
            if tool_name in ("place_trade", "place_order", "buy", "buy_order"):
                args = step.get("args") or {}
                trade_payload = {
                    "model": model,
                    "action": args.get("action", "TRADE"),
                    "side": args.get("side", args.get("action", "TRADE")),
                    "ticker": args.get("ticker") or args.get("market_id", ""),
                    "platform": args.get("platform", "Polymarket / Kalshi"),
                    "shares": args.get("count") or args.get("shares"),
                    "price": args.get("price"),
                    "thesis": getattr(report, "thesis", ""),
                }
                broadcaster.broadcast_agent_trade(trade_payload)
    except Exception as exc:  # noqa: BLE001
        print(f"agent-trading-tick broadcast hook warning: {exc}", file=sys.stderr)



def main() -> int:
    if not MODEL:
        print("AGENT_TRADING_MODEL must be set", file=sys.stderr)
        return 1
    cycle_id = benchmark_tools._current_cycle_id()
    run_id = uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        init_observability()
        _start_cycle_telemetry(run_id, MODEL, cycle_id, started_at)
        with tracer.start_as_current_span("agent_trading.cycle") as span:
            span.set_attributes({
                "agent.model": MODEL,
                "agent.cycle_id": cycle_id,
                "agent.run_id": run_id,
            })
            provider_state, provider_detail = _scads_model_readiness(MODEL)
            if provider_state in SCADS_UNAVAILABLE_STATES:
                duration_seconds = time.perf_counter() - started
                detail = provider_detail or "SCADS status check has temporarily paused this model."
                _finish_cycle_telemetry(
                    run_id,
                    outcome="deferred",
                    duration_ms=round(duration_seconds * 1000),
                    failure_kind="provider_paused",
                    failure_detail=detail,
                )
                span.set_attributes({
                    "outcome": "deferred",
                    "provider.state": provider_state,
                    "provider.degraded": True,
                })
                cycle_runs.add(1, {"model": MODEL, "outcome": "deferred", "failure_kind": "provider_paused"})
                cycle_duration.record(duration_seconds, {"model": MODEL, "outcome": "deferred"})
                provider_degradations.add(1, {"failure_kind": "provider_paused"})
                logger.info(
                    "agent trading cycle deferred model=%s cycle=%s provider_state=%s",
                    MODEL, cycle_id, provider_state,
                )
                return PROVIDER_DEGRADATION_EXIT_CODE
            try:
                summary = run_cycle(MODEL, cycle_id=cycle_id)
            except Exception as exc:  # noqa: BLE001
                failure_kind = _failure_kind(exc)
                duration_seconds = time.perf_counter() - started
                _finish_cycle_telemetry(
                    run_id,
                    outcome="failure",
                    duration_ms=round(duration_seconds * 1000),
                    failure_kind=failure_kind,
                    failure_detail=_failure_detail(exc),
                )
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attributes({"outcome": "failure", "failure.kind": failure_kind})
                cycle_runs.add(1, {"model": MODEL, "outcome": "failure", "failure_kind": failure_kind})
                cycle_duration.record(duration_seconds, {"model": MODEL, "outcome": "failure"})
                if failure_kind in PROVIDER_DEGRADATION_FAILURE_KINDS:
                    span.set_attribute("provider.degraded", True)
                    provider_degradations.add(1, {"failure_kind": failure_kind})
                logger.warning(
                    "agent trading cycle failed model=%s cycle=%s failure_kind=%s detail=%s",
                    MODEL, cycle_id, failure_kind, _failure_detail(exc),
                )
                raise
            duration_seconds = time.perf_counter() - started
            _finish_cycle_telemetry(
                run_id,
                outcome="success",
                duration_ms=round(duration_seconds * 1000),
                summary=summary,
            )
            span.set_attributes({
                "outcome": "success",
                "candidates.count": int(summary["candidate_count"]),
                "tool.steps": int(summary["tool_steps"]),
                "thesis.published": bool(summary["thesis_published"]),
            })
            if summary.get("provider_model"):
                span.set_attribute("gen_ai.response.model", str(summary["provider_model"]))
            cycle_runs.add(1, {"model": MODEL, "outcome": "success"})
            cycle_duration.record(duration_seconds, {"model": MODEL, "outcome": "success"})
            logger.info(
                "agent trading cycle completed model=%s cycle=%s candidates=%s steps=%s forecasts=%s",
                MODEL, cycle_id, summary["candidate_count"], summary["tool_steps"], summary["forecast_records"],
            )
    except Exception as exc:  # noqa: BLE001
        print(f"agent-trading-tick FAILED model={MODEL}: {exc}", file=sys.stderr)
        return (
            PROVIDER_DEGRADATION_EXIT_CODE
            if _failure_kind(exc) in PROVIDER_DEGRADATION_FAILURE_KINDS
            else 1
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
