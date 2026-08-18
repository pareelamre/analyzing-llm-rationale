from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import html
import ipaddress
import json
import logging
import math
import os
import queue
import random
import re
import secrets
import smtplib
import threading
import time
import traceback
import uuid
from collections import OrderedDict, defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional
from urllib.parse import quote as url_quote
from urllib.parse import urlparse

import duckdb
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from analyzing_llm_rationale import (
    agent_capabilities,
    benchmark_tools,
    crypto_5m,
    crypto_kalshi,
    pr_agent,
    rag,
    server_security,
    venue_mcp,
)
from analyzing_llm_rationale import (
    live_track_record as live_track_record_support,
)
from analyzing_llm_rationale.config import (
    scads_chat_model_options,
    scads_hosted_model_allowlist,
    scads_hosted_model_fallbacks,
)
from analyzing_llm_rationale.observability import init_observability
from analyzing_llm_rationale.pipeline import (
    _parse_json_dict,
    build_user_prompt,
    parse_model_response,
)
from analyzing_llm_rationale.server_security import RateLimiter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _REPO_ROOT / "static"
_ANALYTICS_DB = Path(os.environ.get("ANALYTICS_DB", "/tmp/foresea_analytics.duckdb"))
_CANONICAL = "https://foresea.ink"
_MCP_ENDPOINT = f"{_CANONICAL}/mcp/"
_PAGE_CONTEXT_SCRIPT_ID = "foresea-page-context"

_REQUIRED_API_KEY: Optional[str] = os.environ.get("API_KEY")
_GOOGLE_CLIENT_ID: Optional[str] = os.environ.get("GOOGLE_CLIENT_ID")
_GITHUB_CLIENT_ID: Optional[str] = os.environ.get("GITHUB_CLIENT_ID")
_GITHUB_CLIENT_SECRET: Optional[str] = os.environ.get("GITHUB_CLIENT_SECRET")
_SESSION_SECRET: str = os.environ.get("SESSION_SECRET", "change-me-in-production")
_SESSION_TTL_DAYS = 30
# The live track record is produced by a GitHub Action and committed to the repo
# (static/track_record_live.json). The server reads that committed file — at
# runtime from raw GitHub (so it tracks hourly commits without a redeploy),
# falling back to the bundled copy, then to the static backtest.
_TRACK_RECORD_LIVE_URL = os.environ.get(
    "TRACK_RECORD_LIVE_URL",
    "https://raw.githubusercontent.com/pareelamre/analyzing-llm-rationale/main/static/track_record_live.json",
)
_TRACK_RECORD_LIVE_TTL = int(os.environ.get("TRACK_RECORD_LIVE_TTL", "30"))
_TRACK_RECORD_LIVE_TIMEOUT = int(os.environ.get("TRACK_RECORD_LIVE_TIMEOUT", "20"))
_EDGE_BOARD_STALE_AFTER_S = int(os.environ.get("EDGE_BOARD_STALE_AFTER_S", "1800"))
_EDGE_BOARD_CURVE_MAX_POINTS = int(os.environ.get("EDGE_BOARD_CURVE_MAX_POINTS", "160"))
_FORECAST_EVALUATION_URL = os.environ.get(
    "FORECAST_EVALUATION_URL",
    "https://raw.githubusercontent.com/pareelamre/analyzing-llm-rationale/"
    "main/static/forecast_evaluation.json",
)
_MARK_TO_MARKET_LIVE_URL = os.environ.get(
    "MARK_TO_MARKET_LIVE_URL",
    "https://raw.githubusercontent.com/pareelamre/analyzing-llm-rationale/"
    "main/static/mark_to_market_live.json",
)
_FORECAST_EVALUATION_TTL = int(
    os.environ.get("FORECAST_EVALUATION_TTL", "60")
)
_FORECAST_EVALUATION_TIMEOUT = int(
    os.environ.get("FORECAST_EVALUATION_TIMEOUT", str(_TRACK_RECORD_LIVE_TIMEOUT))
)
_FORECAST_EVALUATION_STALE_AFTER_S = int(
    os.environ.get("FORECAST_EVALUATION_STALE_AFTER_S", "1800")
)
_MARK_TO_MARKET_LIVE_TTL = int(
    os.environ.get("MARK_TO_MARKET_LIVE_TTL", str(_TRACK_RECORD_LIVE_TTL))
)
_MARK_TO_MARKET_LIVE_TIMEOUT = int(
    os.environ.get("MARK_TO_MARKET_LIVE_TIMEOUT", str(_TRACK_RECORD_LIVE_TIMEOUT))
)
_MARK_TO_MARKET_STALE_AFTER_S = int(
    os.environ.get("MARK_TO_MARKET_STALE_AFTER_S", str(_EDGE_BOARD_STALE_AFTER_S))
)
# Agentic shadow-trading board (scripts/build_agent_trading_board.py, published by
# .github/workflows/agent-trading-board-publish.yml on its own cadence, separate
# from the deterministic-Kelly mark-to-market ledgers above). Shadow/paper only.
_AGENT_TRADING_BOARD_URL = os.environ.get(
    "AGENT_TRADING_BOARD_URL",
    "https://raw.githubusercontent.com/pareelamre/analyzing-llm-rationale/"
    "main/static/agent_trading_live.json",
)
_AGENT_TRADING_BOARD_TTL = int(
    os.environ.get("AGENT_TRADING_BOARD_TTL", str(_TRACK_RECORD_LIVE_TTL))
)
_AGENT_TRADING_BOARD_TIMEOUT = int(
    os.environ.get("AGENT_TRADING_BOARD_TIMEOUT", str(_TRACK_RECORD_LIVE_TIMEOUT))
)
_AGENT_TRADING_BOARD_STALE_AFTER_S = int(
    # The board publishes far less often than the MTM ledgers (every ~15 min,
    # best-effort under GitHub's scheduler) -- default staleness threshold is
    # correspondingly looser so a normal publish gap isn't flagged as stale.
    os.environ.get("AGENT_TRADING_BOARD_STALE_AFTER_S", "3600")
)
# Shared secret gating the evolution-loop bridge endpoints (pending-markets /
# mark-enrolled), called by the track-record GitHub Action. Unset = disabled.
_TRACK_RECORD_TOKEN: Optional[str] = os.environ.get("TRACK_RECORD_TOKEN")
# Shared secret for the bounded scheduled reconciliation trigger. It intentionally
# has a distinct capability from the forecast bridge because this path decrypts
# users' connected exchange accounts to read, never trade.
_TRADING_RECONCILIATION_TOKEN: Optional[str] = os.environ.get("TRADING_RECONCILIATION_TOKEN")
_TRADING_RECONCILIATION_MAX_ORDERS = max(
    1, min(int(os.environ.get("TRADING_RECONCILIATION_MAX_ORDERS", "25")), 100)
)
# Shared secret for the bounded scheduled AgentRun reconciliation trigger. A
# distinct token from trading reconciliation on purpose -- this path never
# touches a connected exchange account, only a signed-in user's own private
# research-run records.
_AGENT_RUN_RECONCILIATION_TOKEN: Optional[str] = os.environ.get("AGENT_RUN_RECONCILIATION_TOKEN")
_AGENT_RUN_RECONCILIATION_MAX_RUNS = max(
    1, min(int(os.environ.get("AGENT_RUN_RECONCILIATION_MAX_RUNS", "25")), 100)
)
# A run stuck at status="running" past this many minutes almost certainly means
# the request that owned it crashed or the process recycled mid-flight, not
# that it's still legitimately working -- the bounded tool loop (<=8 steps,
# each one LLM call) normally finishes in well under this.
_AGENT_RUN_STALE_MINUTES = int(os.environ.get("AGENT_RUN_STALE_MINUTES", "60"))
_state: Dict[str, Any] = {}
_trading_run_lock = threading.RLock()
_trading_guardrail_lock = threading.RLock()
_PUBLIC_MCP = None
_PUBLIC_MCP_APP = None

logger = logging.getLogger("foresea")

# ── Resilience knobs (all free; env-overridable) ──────────────────────────────
# Live forecast calls retry transient upstream failures with bounded backoff and
# a per-attempt timeout, so a flaky SCADS AI response degrades gracefully into a
# clean 503 instead of a raw 500. The batch pipeline already retries; this brings
# the live /predict + agent paths up to the same standard.
_PROVIDER_MAX_RETRIES = int(os.environ.get("PROVIDER_MAX_RETRIES", "2"))      # attempts = retries + 1
_PROVIDER_TIMEOUT_S = float(os.environ.get("PROVIDER_TIMEOUT_S", "90"))       # per-attempt wall-clock budget
_PROVIDER_BACKOFF_BASE_S = float(os.environ.get("PROVIDER_BACKOFF_BASE_S", "0.5"))
_INTERACTIVE_DEFAULT_MODEL = os.environ.get("INTERACTIVE_DEFAULT_MODEL", "gemma-4-26b-a4b-it").strip()
_INTERACTIVE_MAX_TOKENS = int(os.environ.get("INTERACTIVE_MAX_TOKENS", "768"))
_CHAT_PROVIDER_TIMEOUT_S = float(os.environ.get("CHAT_PROVIDER_TIMEOUT_S", "15"))
_CHAT_PROVIDER_MAX_RETRIES = int(os.environ.get("CHAT_PROVIDER_MAX_RETRIES", "1"))
# A "deep"-tier question is deliberately allowed to reason harder (see
# _REASONING_EFFORT_BY_TIER below), which routinely exceeds the 15s chat
# budget above -- give it a bigger one instead of just timing out more often.
_CHAT_DEEP_PROVIDER_TIMEOUT_S = float(os.environ.get("CHAT_DEEP_PROVIDER_TIMEOUT_S", "45"))
_REASONING_EFFORT_BY_TIER = {"simple": "low", "standard": None, "deep": "high"}
# Council members run concurrently, so retrying each member multiplies load
# without improving quorum availability. Bound each member independently and
# let the successful subset produce the forecast.
_COUNCIL_MEMBER_TIMEOUT_S = float(os.environ.get("COUNCIL_MEMBER_TIMEOUT_S", "35"))
_COUNCIL_MIN_SUCCESSFUL_MEMBERS = max(
    1, int(os.environ.get("COUNCIL_MIN_SUCCESSFUL_MEMBERS", "1"))
)
# Reject oversized request bodies before they reach a handler. Generous enough
# for PDF uploads on /extract, small enough to blunt memory-exhaustion abuse.
_MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(12 * 1024 * 1024)))
# Flipped to False during graceful shutdown so /ready drains in-flight traffic.
_ready = True

# Conversational system prompt for chat_mode — overrides the JSON-only forecast
# prompt so replies are natural language, not a forecast template.
_CHAT_SYSTEM_PROMPT = (
    "You are Foresea, a helpful forecasting and prediction-market assistant. "
    "Answer the user's question conversationally in clear natural language. "
    "Use light markdown — short paragraphs, **bold** for key points, and bullet or "
    "numbered lists when they help. Ground your answer in the provided evidence "
    "summaries and any prediction-market context when relevant, and be honest about "
    "uncertainty. Treat the provided Current Time as today's date/time for temporal "
    "phrases and deadlines. If the user asks for a forecast, prediction, likelihood, "
    "odds, or whether a future event will happen, give a clear conclusion and, when a "
    "market price is given, briefly note whether you lean above or below it. Foresea "
    "will render your probability as part of the reply, so do not use bracketed tags "
    "or a rigid forecast template in your prose. When evidence summaries are provided, include a short "
    "**Evidence used** section naming one to three provided sources by publisher or "
    "domain and article title, followed by the concrete facts that affected your "
    "estimate. Never refer to evidence only as \"Article 1\", \"Article 2\", or another "
    "bare item number. Never invent a source or claim that you checked current reporting "
    "when no evidence was retrieved. "
    "For every forecast, append the internal machine-readable marker on its own final "
    "line: [p:0.XX] where 0.XX is your estimate as a decimal (e.g. [p:0.65]). This "
    "marker is hidden from the user. Omit it only for non-forecast conversation. "
    "Do NOT output JSON, key/value objects, or a rigid forecast template — just talk to the user."
)

_CHAT_PROB_RE = re.compile(r"\[p:(0(?:\.\d+)?|1(?:\.0+)?)\]\s*$", re.MULTILINE)

# Defense-in-depth for the tool-loop path: build_grounding_note()'s text is
# meant as background context, never a quotable answer -- strip it if a model
# echoes it into its final turn anyway (extra_rules already tells it not to).
_SELF_CALIBRATION_ECHO_RE = re.compile(
    r"\n*(?:\[Self-calibration context\]\n)?Forecaster self-calibration\b.*?(?=\n\n|\Z)",
    re.DOTALL,
)

_TRADING_INTENT_RE = re.compile(
    r"\b(bet|order|trade|place|buy|sell|position|recommend|suggest|should i|what to|portfolio|stake|wager)\b",
    re.IGNORECASE,
)


def _detect_trading_intent(question: str) -> bool:
    return bool(_TRADING_INTENT_RE.search(question))


_GREETING_RE = re.compile(
    r"^(hi|hello|hey+|yo|sup|howdy|greetings|good\s+(morning|afternoon|evening|day)|"
    r"who are you|what are you|what is foresea|what does foresea do|"
    r"what can you do|how does this work|how do (i|you) use (this|foresea))\b",
    re.IGNORECASE,
)


def _is_greeting_or_meta(question: str) -> bool:
    """Zero-cost check for a greeting or a meta question about Foresea itself
    (e.g. "hello", "what can you do?") rather than a forecasting question.

    Requires the whole message to be short so a real forecasting question that
    happens to open with a greeting ("Hi, will the Fed cut rates?") is never
    misrouted away from evidence retrieval.
    """
    q = (question or "").strip()
    if not q or len(q.split()) > 8:
        return False
    return bool(_GREETING_RE.match(q))


_MULTI_PART_RE = re.compile(
    r"\b(and|or|versus|vs\.?|compared to|relative to|as well as)\b", re.IGNORECASE
)
_SCENARIO_RE = re.compile(
    r"\b(if|unless|assuming|suppose|in the event|either|both|conditional on|"
    r"depends on|scenario|what happens if)\b",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9&]*(?:\s+[A-Z][a-zA-Z0-9&]*)*\b")
_NUMERIC_ANCHOR_RE = re.compile(r"\b(19|20)\d{2}\b|\$\d|\d+(\.\d+)?%")
_EFFORT_STOPWORD_CAPS = {
    "I", "Will", "The", "What", "When", "Is", "Are", "Does", "Do", "How", "Why", "Should",
}


def _estimate_question_effort(question: str, history_len: int = 0) -> str:
    """Zero-cost, zero-LLM-call effort tier for a question: simple/standard/deep.

    Pure regex/string scoring, mirroring _detect_trading_intent's shape. Never
    raises; an empty question returns "standard" (today's fixed behavior),
    so a misclassification can only ever narrow or widen effort within
    already-validated bounds, never break the pipeline.
    """
    q = (question or "").strip()
    if not q:
        return "standard"
    words = q.split()
    entities = {m for m in _ENTITY_RE.findall(q) if m not in _EFFORT_STOPWORD_CAPS}
    score = (
        (2 if len(words) > 30 else 1 if len(words) > 14 else 0)
        + min(len(_MULTI_PART_RE.findall(q)), 3)
        + 2 * min(len(_SCENARIO_RE.findall(q)), 2)
        + (1 if q.count("?") > 1 else 0)
        + (1 if len(entities) >= 3 else 0)
        + (1 if len(_NUMERIC_ANCHOR_RE.findall(q)) >= 2 else 0)
        + (1 if (q.count(",") + q.count(";")) >= 2 else 0)
        + (1 if history_len >= 6 else 0)
    )
    if score <= 1 and len(words) <= 14:
        return "simple"
    if score >= 5:
        return "deep"
    return "standard"


_EFFORT_EVIDENCE_TOP_K = {"simple": 8, "standard": 20, "deep": 40}
_EFFORT_MAX_TOOL_STEPS = {"simple": 2, "standard": 5, "deep": 8}


def _apply_effort_tier(req: "AgentAnalyzeRequest", question: str) -> None:
    """Fill req.effort_tier (and the fields it controls) in place, unless the
    caller already set them explicitly — an explicit value always wins."""
    fields_set = req.model_fields_set
    if "effort_tier" not in fields_set:
        req.effort_tier = _estimate_question_effort(question, len(req.history))
    tier = req.effort_tier or "standard"
    if "evidence_top_k" not in fields_set:
        req.evidence_top_k = _EFFORT_EVIDENCE_TOP_K[tier]
    if "max_tool_steps" not in fields_set:
        req.max_tool_steps = min(req.max_tool_steps, _EFFORT_MAX_TOOL_STEPS[tier])


def _strategy_filter_edge_entry(entry: dict, strategy: str) -> bool:
    """Apply a paper-pnl strategy's filter logic to a live edge board entry."""
    entry_price = entry.get("entry_price", 0.5)
    abs_edge = entry.get("abs_edge", 0.0)
    domain = entry.get("domain", "")
    if strategy == "smart":
        if entry_price < 0.20 or entry_price > 0.80:
            return False
        if domain == "geopolitics" and abs_edge > 0.10:
            return False
        if abs_edge > 0.40:
            return False
        return True
    return True  # flat / half_kelly / crowd_baseline — no extra filter beyond min_edge


def _pick_best_strategy(paper_pnl: dict) -> tuple:
    """Return (name, data) of the highest-ROI strategy with at least 20 resolved bets."""
    industry_grade = {"smart", "half_kelly", "flat", "crowd_baseline"}
    candidates = [
        (k, v) for k, v in paper_pnl.items()
        if k in industry_grade
        and isinstance(v, dict)
        and v.get("roi") is not None
        and (v.get("n_bets") or 0) >= 20
    ]
    if not candidates:
        return ("flat", paper_pnl.get("flat") or {})
    return max(candidates, key=lambda x: x[1]["roi"])


def _edge_board_order_context(trl: dict) -> str:
    """Format the top edge board picks for the best paper strategy as chat context."""
    paper_pnl = trl.get("paper_pnl") or {}
    edge_board = trl.get("edge_board") or []
    if not edge_board or not paper_pnl:
        return ""
    strategy_name, strategy_data = _pick_best_strategy(paper_pnl)
    filtered = [e for e in edge_board if _strategy_filter_edge_entry(e, strategy_name)]
    filtered.sort(key=lambda e: e.get("abs_edge", 0.0), reverse=True)
    if not filtered:
        return ""
    roi_pct = f"{strategy_data['roi']:.1%}" if strategy_data.get("roi") is not None else "n/a"
    n_bets = strategy_data.get("n_bets", "?")
    lines = [
        "## Live order recommendations",
        f"Best back-tested strategy: **{strategy_name}** "
        f"(historical ROI {roi_pct} over {n_bets} resolved bets, paper only).",
        "",
    ]
    for i, e in enumerate(filtered[:10], 1):
        sig = e.get("track_record") or {}
        proven = "proven" if sig.get("skill_significant") else "unproven"
        model_p = f"{e.get('model_probability', 0):.0%}"
        mkt_p = f"{e.get('market_probability', 0):.0%}"
        edge_pct = f"{e.get('abs_edge', 0):.0%}"
        payout = e.get("payout_odds", "?")
        lines.append(
            f"{i}. **{e.get('question', '')}** [{e.get('platform', '')}]  "
            f"Bet {e.get('side', '?')} @ {mkt_p} | Model {model_p} | "
            f"Edge {edge_pct} | {payout}x payout | {proven}  "
            f"{e.get('market_url', '')}"
        )
    lines.append(
        "\nAll figures are paper/hypothetical. Entry prices are live at last tick."
    )
    return "\n".join(lines)


_strategy_filter_edge_entry = live_track_record_support.strategy_filter_edge_entry
_pick_best_strategy = live_track_record_support.pick_best_strategy
_edge_board_order_context = live_track_record_support.edge_board_order_context


_DESCRIPTION = """
## Overview

The **Foresea Intelligence API** turns market questions, linked evidence, and
prediction-market prices into probabilistic forecasts that can sit behind
trading dashboards, alerting systems, and research workflows.

It supports binary, multiple-choice, numeric, and date forecasts. Each response
includes a structured **rationale**, typed forecast fields, optional evidence
sources fetched from live news, and optional market-edge analysis when you pass
a market-implied probability.

---

## Quick start

```bash
curl -X POST https://foresea.ink/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "Will the Fed cut rates before September 30, 2026?",
    "question_type": "binary",
    "market_platform": "Polymarket",
    "market_probability": 0.54,
    "variant": "variant0_neutral_baseline"
  }'
```

---

## Question types

Use `question_type` when your client already knows the shape of the question.
If omitted, the model will try to infer it.

| Type | Response shape |
|---|---|
| `binary` | `predicted_answer` is `Yes`/`No`; `confidence` is 0–1 |
| `multiple_choice` | `options` contains per-option probabilities; `predicted_answer` is the top option |
| `numeric` | `range_forecast` contains `p10`, `p50`, `p90`, and optional `unit` |
| `date` | `range_forecast` contains date percentiles |

For multiple-choice forecasts, pass `options` when the answer set is known.

---

## Prediction-market context

Pass `market_probability` when you know the current market-implied probability
for the target outcome. For binary markets this defaults to the `Yes` side; set
`market_outcome` to compare against `No` or a named multiple-choice outcome.

Foresea returns `market_analysis` with model probability, market probability,
percentage-point edge, and a stance (`model_above_market`, `model_below_market`,
`in_line`, or `not_comparable`).

---

## Trading execution

Foresea can preview and submit guarded orders for Kalshi and Polymarket:

| Endpoint | Purpose |
|---|---|
| `GET /trading/accounts` | Return configured/not-configured server-account status without exposing secrets |
| `POST /trading/accounts/check` | Validate caller-supplied own-account (BYO) credentials without storing them |
| `POST /trading/preview` | Normalize and validate an order without placing it |
| `POST /trading/orders` | Submit a live order after explicit confirmation |

Live trading is disabled unless `FORESEA_ENABLE_TRADING=true`. Every live order
requires a signed-in user session, `execute=true`, and the exact confirmation
phrase `PLACE REAL ORDER`. Market/IOC/FOK-style orders are separately blocked
unless `FORESEA_ALLOW_MARKET_ORDERS=true`.

Exchange credentials come from a shared server account (environment variables /
Secret Manager mounts) or an authenticated user's **encrypted account
connection**. The connection endpoint accepts a user's credentials once, validates
them with a unique data-encryption key wrapped by Cloud KMS, and never returns
them to the browser. Inline `venue_credentials` are rejected on public preview
and order calls; encrypted own-account trading remains gated by
`FORESEA_ENABLE_BYO_TRADING=true` (independent of `FORESEA_ENABLE_TRADING`).

`/agent/analyze` may return a directional recommendation, but it never places an
order. Use `/trading/preview` first, review the normalized order, then call
`/trading/orders` only when you intend to submit a real order.

---

## Prompt variants

The `variant` field controls how the LLM is prompted. Choose the variant that
best matches the information you have available:

| Variant key | Focus |
|---|---|
| `variant0_neutral_baseline` | Control — no extra framing (default) |
| `variant1_predicted_event` | State the concrete predicted event |
| `variant2_key_attribute` | Highlight time / quantity / actor |
| `variant3_reasoning_type` | Specify reasoning type (speculation, expert forecast…) |
| `variant4_credibility` | Ground rationale in source credibility scores |
| `variant5_key_conditions` | List 2–4 conditions that must hold |
| `variant6_step_by_step_reasoning` | Produce 2–3 numbered reasoning steps |
| `variant7_uncertainty_language` | Require uncertainty hedging words |
| `variant8_temporal_anchors` | Anchor reasoning to specific dates |

---

## Rate limiting

**`RATE_LIMIT_PER_MIN` requests per minute per IP address (default 60).** Exceeding this returns `429 Too Many Requests`
with a `Retry-After: 60` header.

**`PREDICT_RATE_LIMIT_PER_MIN` LLM calls per minute per IP (default 10).** Applies to `/predict` and `/agent/analyze`
on top of the global limit — these endpoints are expensive so they get a tighter per-IP cap.

---

## Authentication

When the server is configured with `API_KEY`, all prediction endpoints require:

```
X-API-Key: <your-key>
```

The `/health` endpoint is always unauthenticated. Per-user endpoints such as
RAG, chat history, and trading require an `Authorization: Bearer <session-token>`
header returned by the sign-in endpoints.

---

## Source code

[github.com/pareelamle/analyzing-llm-rationale](https://github.com/pareelamre/analyzing-llm-rationale)
"""

_TAGS = [
    {
        "name": "Inference",
        "description": "Run forecast and market-edge analysis for binary, multiple-choice, numeric, and date questions.",
    },
    {
        "name": "Markets",
        "description": "Fetch live prices from prediction-market venues (Polymarket, Kalshi).",
    },
    {
        "name": "Trading",
        "description": "Preview and submit guarded prediction-market orders for authenticated users.",
    },
    {
        "name": "Agents",
        "description": "Autonomous analysis: orchestrate market fetch, evidence, forecast, edge, and custom skills into one report.",
    },
    {
        "name": "Knowledge",
        "description": "Per-user RAG knowledge base: ingest documents and retrieve them as evidence.",
    },
    {
        "name": "System",
        "description": "Health and liveness checks.",
    },
]


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _verify_google_token(credential: str) -> dict:
    """Verify a Google One-Tap ID token and return its claims."""
    if not _GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google auth is not configured on this server.")
    try:
        from google.auth.transport.requests import Request as _GRequest
        from google.oauth2.id_token import verify_oauth2_token as _verify
        return _verify(credential, _GRequest(), _GOOGLE_CLIENT_ID)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid Google credential: {exc}") from exc


def _exchange_github_code(code: str, redirect_uri: Optional[str]) -> dict:
    """Exchange a GitHub OAuth code for a profile (id, email, name, avatar)."""
    if not (_GITHUB_CLIENT_ID and _GITHUB_CLIENT_SECRET):
        raise HTTPException(status_code=503, detail="GitHub auth is not configured on this server.")
    import requests
    try:
        token_resp = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": _GITHUB_CLIENT_ID,
                "client_secret": _GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": redirect_uri or "",
            },
            timeout=12,
        )
        token = token_resp.json().get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="GitHub did not return an access token.")
        auth_headers = {"Authorization": f"Bearer {token}", "Accept": "application/json",
                        "User-Agent": "foresea-auth"}
        user = requests.get("https://api.github.com/user", headers=auth_headers, timeout=12).json()
        email = user.get("email")
        if not email:
            emails = requests.get("https://api.github.com/user/emails", headers=auth_headers, timeout=12).json()
            if isinstance(emails, list):
                primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
                email = (primary or (emails[0] if emails else {})).get("email", "")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"GitHub sign-in failed: {exc}") from exc
    if not user.get("id"):
        raise HTTPException(status_code=401, detail="GitHub profile could not be read.")
    return {
        "sub": f"github:{user['id']}",
        "email": email or "",
        "name": user.get("name") or user.get("login") or "",
        "picture": user.get("avatar_url") or "",
    }
def _issue_session(sub: str, email: str, name: str, picture: str) -> str:
    """Sign and return a JWT session token."""
    import jwt as _jwt
    now = datetime.now(timezone.utc)
    return _jwt.encode(
        {
            "sub": sub,
            "email": email,
            "name": name,
            "picture": picture,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=_SESSION_TTL_DAYS)).timestamp()),
        },
        _SESSION_SECRET,
        algorithm="HS256",
    )


def _decode_session(token: str) -> dict:
    return server_security.decode_session(token, _SESSION_SECRET)


def _require_session(request: Request) -> dict:
    return server_security.require_session(request, _SESSION_SECRET)


def _require_auth(request: Request) -> dict:
    return server_security.require_auth(
        request,
        _REQUIRED_API_KEY,
        _TRACK_RECORD_TOKEN,
        _SESSION_SECRET,
    )


def _optional_predict_claims(request: Request) -> Optional[dict]:
    has_auth = bool(request.headers.get("Authorization", "").startswith("Bearer "))
    has_api_key = bool(request.headers.get("X-API-Key"))
    has_track_token = bool(request.headers.get("X-Track-Token"))
    if _REQUIRED_API_KEY or has_auth or has_api_key or has_track_token:
        return _require_auth(request)
    return None


def _optional_user_id(request: Optional[Request]) -> Optional[str]:
    return server_security.optional_user_id(request, _SESSION_SECRET)


_ds_client: Any = None
_trading_kms_client: Any = None


def _get_datastore():
    global _ds_client
    if _ds_client is None:
        try:
            from google.cloud import datastore as _ds
            _ds_client = _ds.Client()
        except Exception:
            pass
    return _ds_client


def _upsert_user(sub: str, email: str, name: str, picture: str) -> str:
    """Create or update a User entity, deduped by email.

    A user can get a *new* Google/GitHub ``sub`` (re-consent, a different OAuth
    client, or Google rotating the subject) — keying on ``sub`` then forks a
    duplicate account for the same person. So we resolve by verified email first:
    if any account already exists for this email, reuse the oldest one as
    canonical (recording the new ``sub`` in ``alt_subs``) instead of creating a
    duplicate. Returns the effective user id to use for the session.
    """
    client = _get_datastore()
    if client is None:
        return sub
    from google.cloud import datastore as _ds
    now = datetime.now(timezone.utc)
    entity = None
    if email:
        try:
            q = client.query(kind="User")
            q.add_filter("email", "=", email)
            matches = list(q.fetch())
        except Exception:
            logger.warning("user email-dedup lookup failed", exc_info=True)
            matches = []
        if matches:
            matches.sort(key=lambda e: e.get("created_at") or now)
            entity = matches[0]  # oldest account for this email is canonical
            if entity.key.name != sub:
                aliases = set(entity.get("alt_subs") or [])
                aliases.add(sub)
                entity["alt_subs"] = sorted(aliases)
    if entity is None:
        key = client.key("User", sub)
        entity = client.get(key)
        if entity is None:
            entity = _ds.Entity(key=key, exclude_from_indexes=("picture",))
            entity["created_at"] = now
    entity.update(email=email, name=name, picture=picture, last_login=now)
    client.put(entity)
    _sync_user_duckdb(entity.key.name or sub, email, name, picture, entity.get("created_at") or now, now)
    return entity.key.name or sub


def _sync_user_duckdb(sub: str, email: str, name: str, picture: str, created_at: Any, last_login: Any) -> None:
    """Mirror user entity into DuckDB users table for unified analytics and local dev."""
    try:
        conn = _analytics_conn()
        try:
            conn.execute(
                """
                INSERT INTO users (sub, email, name, picture, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (sub) DO UPDATE SET
                    email = excluded.email,
                    name = excluded.name,
                    picture = excluded.picture,
                    last_login = excluded.last_login
                """,
                [sub, email, name, picture, created_at, last_login],
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


# Datastore kind: agent-forecast markets to enrol into the live track record.
_ENROLLED_MARKET_KIND = "AgentEnrolledMarket"


def _enroll_market_sync(platform: str, ident: str, market_url: str,
                        question: str, source: str) -> None:
    """Record an agent-forecast market as a *pointer* for the track-record Action to
    enrol. Writes ONLY this Datastore pointer — never the track-record store itself
    (that stays Action-owned; see scripts/track_record_tick.py). Best-effort: no-op
    if Datastore is unavailable, never raises into the request path."""
    client = _get_datastore()
    if client is None:
        return
    try:
        from google.cloud import datastore as _ds
        key = client.key(_ENROLLED_MARKET_KIND, f"{platform}:{ident}")
        entity = client.get(key)
        now = datetime.now(timezone.utc)
        if entity is None:
            entity = _ds.Entity(key=key, exclude_from_indexes=("market_url", "question"))
            entity["first_seen_ts"] = now
            entity["seen_count"] = 0
            entity["enrolled"] = False
        entity.update(
            platform=platform,
            ident=ident,
            market_url=market_url,
            question=(question or "")[:500],
            last_seen_ts=now,
            seen_count=int(entity.get("seen_count") or 0) + 1,
            request_source=source,
        )
        client.put(entity)
    except Exception:
        logger.warning("market enrollment failed", exc_info=True)


async def _enroll_market(platform: Optional[str], ident: Optional[str],
                         market_url: Optional[str], question: str, source: str) -> None:
    """Fire-and-forget enrolment off the request path (executor + swallow errors)."""
    if not platform or not ident or not market_url:
        return
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, _enroll_market_sync, platform, ident, market_url, question, source)
    except Exception:
        pass


# ── Email + password accounts ───────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PBKDF2_ITERATIONS = 200_000
# A throwaway hash so failed logins for unknown emails still pay the PBKDF2 cost,
# keeping response timing uniform whether or not the account exists.
_DUMMY_PASSWORD_HASH = (
    "pbkdf2_sha256$200000$"
    "00000000000000000000000000000000$"
    "0000000000000000000000000000000000000000000000000000000000000000"
)


def _normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def _hash_password(password: str) -> str:
    """Return a PBKDF2-HMAC-SHA256 hash string: ``algo$iters$salt_hex$hash_hex``."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of ``password`` against an encoded PBKDF2 hash."""
    try:
        _algo, iters_str, salt_hex, hash_hex = encoded.split("$")
        iterations = int(iters_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


def _get_user_record(user_id: str) -> Optional[Dict[str, Any]]:
    """Return the stored User record (dict) for ``user_id``, or None if absent."""
    client = _get_datastore()
    if client is None:
        return _state.setdefault("password_users", {}).get(user_id)
    entity = client.get(client.key("User", user_id))
    return dict(entity) if entity is not None else None


def _create_password_user(user_id: str, email: str, name: str, password_hash: str) -> None:
    """Persist a new email/password account."""
    now = datetime.now(timezone.utc)
    client = _get_datastore()
    if client is None:
        _state.setdefault("password_users", {})[user_id] = {
            "email": email,
            "name": name,
            "picture": "",
            "password_hash": password_hash,
            "auth_provider": "password",
            "created_at": now,
            "last_login": now,
        }
        return
    from google.cloud import datastore as _ds
    entity = _ds.Entity(
        key=client.key("User", user_id),
        exclude_from_indexes=("picture", "password_hash"),
    )
    entity.update(
        email=email,
        name=name,
        picture="",
        password_hash=password_hash,
        auth_provider="password",
        created_at=now,
        last_login=now,
    )
    client.put(entity)


def _touch_last_login(user_id: str) -> None:
    """Best-effort update of ``last_login`` for an existing account."""
    client = _get_datastore()
    now = datetime.now(timezone.utc)
    if client is None:
        record = _state.setdefault("password_users", {}).get(user_id)
        if record is not None:
            record["last_login"] = now
        return
    entity = client.get(client.key("User", user_id))
    if entity is not None:
        entity["last_login"] = now
        client.put(entity)


def _conversation_key(client: Any, user_id: str, conversation_id: str) -> Any:
    return client.key("User", user_id, "Conversation", conversation_id)


def _message_key(client: Any, user_id: str, conversation_id: str, message_id: str) -> Any:
    return client.key(
        "User",
        user_id,
        "Conversation",
        conversation_id,
        "Message",
        message_id,
    )


def _normalise_message(message: Dict[str, Any], index: int) -> Dict[str, Any]:
    normalised = dict(message)
    normalised.setdefault("id", f"msg_{index}")
    normalised.setdefault("createdAt", normalised.get("updatedAt") or index)
    return normalised


def _list_conversations(user_id: str) -> List[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        conversations = list(_state.setdefault("chat_conversations", {}).get(user_id, {}).values())
        return sorted(conversations, key=lambda c: c.get("updatedAt") or 0, reverse=True)
    query = client.query(kind="Conversation", ancestor=client.key("User", user_id))
    conversations = []
    for entity in query.fetch(limit=100):
        conversations.append({
            "id": entity.key.name,
            "title": entity.get("title", "New conversation"),
            "createdAt": entity.get("createdAt"),
            "updatedAt": entity.get("updatedAt"),
            "conversationSteer": entity.get("conversationSteer", ""),
            "messages": _list_messages(user_id, entity.key.name),
        })
    conversations.sort(key=lambda c: c.get("updatedAt") or 0, reverse=True)
    return conversations


def _list_messages(user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        conv = _state.setdefault("chat_conversations", {}).get(user_id, {}).get(conversation_id, {})
        return conv.get("messages", [])
    query = client.query(kind="Message", ancestor=_conversation_key(client, user_id, conversation_id))
    messages = []
    for entity in query.fetch(limit=500):
        message = dict(entity)
        message["id"] = entity.key.name
        messages.append(message)
    messages.sort(key=lambda m: (m.get("createdAt") or 0, m.get("id") or ""))
    return messages


def _unindexed_nested(_ds: Any, value: Any) -> Any:
    """Recursively rebuild dict values as Datastore Entities with every key
    excluded from indexes.

    ``exclude_from_indexes`` on a parent property only marks that property's
    own value as unindexed -- it does not cascade into a nested dict's
    properties, which the client library otherwise auto-wraps into a fresh
    Entity with no exclusions of its own (so they'd stay indexed and subject
    to Datastore's 1500-byte indexed-string limit). Rebuilding each nested
    dict as its own Entity with its own keys excluded fixes that at every
    depth, including dicts nested inside lists.
    """
    if isinstance(value, dict):
        nested = _ds.Entity(key=None, exclude_from_indexes=tuple(value.keys()))
        nested.update({k: _unindexed_nested(_ds, v) for k, v in value.items()})
        return nested
    if isinstance(value, list):
        return [_unindexed_nested(_ds, v) for v in value]
    return value


def _put_conversation(user_id: str, conversation: Dict[str, Any]) -> Dict[str, Any]:
    client = _get_datastore()
    messages = [
        _normalise_message(message, index)
        for index, message in enumerate(conversation.get("messages", []))
    ]
    conversation = dict(conversation)
    conversation["messages"] = messages
    if client is None:
        _state.setdefault("chat_conversations", {}).setdefault(user_id, {})[conversation["id"]] = conversation
        return conversation
    from google.cloud import datastore as _ds
    key = _conversation_key(client, user_id, conversation["id"])
    entity = _ds.Entity(key=key, exclude_from_indexes=("conversationSteer",))
    entity.update({
        "title": conversation.get("title", "New conversation"),
        "createdAt": conversation.get("createdAt"),
        "updatedAt": conversation.get("updatedAt"),
        "conversationSteer": conversation.get("conversationSteer", ""),
        "saved_at": datetime.now(timezone.utc),
    })
    message_keys = [
        _message_key(client, user_id, conversation["id"], message["id"])
        for message in messages
    ]
    with client.transaction():
        client.put(entity)
        existing_query = client.query(kind="Message", ancestor=key)
        existing_query.keys_only()
        existing_keys = [message_entity.key for message_entity in existing_query.fetch()]
        stale_keys = [existing_key for existing_key in existing_keys if existing_key not in message_keys]
        if stale_keys:
            client.delete_multi(stale_keys)
        # Only index the short scalar fields used for ordering/identity;
        # exclude everything else to stay under Datastore's 1500-byte limit.
        # Datastore indexes each property of an embedded (dict) entity
        # independently, so excluding the parent key (e.g. "data") from
        # indexes does NOT cascade to its nested fields (e.g.
        # "data.model_rationale") -- each nested dict must be rebuilt as its
        # own Entity with its own keys excluded, recursively.
        _MSG_INDEXED = frozenset({"id", "role", "createdAt", "updatedAt", "index"})
        message_entities = []
        for message, message_key in zip(messages, message_keys):
            exclude = tuple(k for k in message if k not in _MSG_INDEXED)
            message_entity = _ds.Entity(key=message_key, exclude_from_indexes=exclude)
            message_entity.update({
                k: (_unindexed_nested(_ds, v) if k not in _MSG_INDEXED else v)
                for k, v in message.items()
            })
            message_entities.append(message_entity)
        if message_entities:
            client.put_multi(message_entities)
    return conversation


def _delete_conversation(user_id: str, conversation_id: str) -> None:
    client = _get_datastore()
    if client is None:
        _state.setdefault("chat_conversations", {}).setdefault(user_id, {}).pop(conversation_id, None)
        return
    key = _conversation_key(client, user_id, conversation_id)
    message_query = client.query(kind="Message", ancestor=key)
    message_query.keys_only()
    message_keys = [entity.key for entity in message_query.fetch()]
    if message_keys:
        client.delete_multi(message_keys)
    client.delete(key)


# ── Favourite markets / watchlist (per-user on Datastore) ────────────────────

_FAVORITE_KIND = "FavoriteMarket"
_FAVORITE_FIELDS = (
    "key", "question", "platform", "ident", "market_url",
    "model_probability", "market_probability", "notify", "createdAt", "updatedAt",
)


def _favorite_key(client: Any, user_id: str, key: str) -> Any:
    return client.key("User", user_id, _FAVORITE_KIND, key)


def _favorite_from_entity(entity: Any) -> Dict[str, Any]:
    fav = {field: entity.get(field) for field in _FAVORITE_FIELDS}
    fav["key"] = entity.key.name
    fav["notify"] = bool(fav.get("notify"))
    return fav


def _list_favorites(user_id: str) -> List[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        favs = list(_state.setdefault("favorites", {}).get(user_id, {}).values())
        return sorted(favs, key=lambda f: f.get("updatedAt") or 0, reverse=True)
    query = client.query(kind=_FAVORITE_KIND, ancestor=client.key("User", user_id))
    favs = [_favorite_from_entity(entity) for entity in query.fetch(limit=200)]
    favs.sort(key=lambda f: f.get("updatedAt") or 0, reverse=True)
    return favs


def _put_favorite(user_id: str, favorite: Dict[str, Any]) -> Dict[str, Any]:
    favorite = {field: favorite.get(field) for field in _FAVORITE_FIELDS}
    favorite["notify"] = bool(favorite.get("notify"))
    client = _get_datastore()
    if client is None:
        _state.setdefault("favorites", {}).setdefault(user_id, {})[favorite["key"]] = favorite
        return favorite
    from google.cloud import datastore as _ds
    key = _favorite_key(client, user_id, favorite["key"])
    # Only short scalar identity/order fields need indexing; exclude free text.
    indexed = {"key", "platform", "ident", "notify", "createdAt", "updatedAt"}
    exclude = tuple(f for f in favorite if f not in indexed)
    entity = _ds.Entity(key=key, exclude_from_indexes=exclude)
    entity.update(favorite)
    client.put(entity)
    return favorite


def _delete_favorite(user_id: str, key: str) -> None:
    client = _get_datastore()
    if client is None:
        _state.setdefault("favorites", {}).setdefault(user_id, {}).pop(key, None)
        return
    client.delete(_favorite_key(client, user_id, key))


_PERSONAL_LEDGER_KIND = "PersonalLedgerEntry"
_PERSONAL_LEDGER_FIELDS = (
    "id", "conversation_id", "message_id", "question", "predicted_answer",
    "probability", "rationale", "model", "createdAt", "user_verdict", "judgedAt",
)


def _personal_ledger_key(client: Any, user_id: str, entry_id: str) -> Any:
    return client.key("User", user_id, _PERSONAL_LEDGER_KIND, entry_id)


def _personal_ledger_from_entity(entity: Any) -> Dict[str, Any]:
    entry = {field: entity.get(field) for field in _PERSONAL_LEDGER_FIELDS}
    entry["id"] = entity.key.name
    return entry


def _list_personal_ledger(user_id: str) -> List[Dict[str, Any]]:
    """List a user's explicit chat-forecast saves in newest-first order."""
    client = _get_datastore()
    if client is None:
        entries = list(_state.setdefault("personal_ledger", {}).get(user_id, {}).values())
        return sorted(entries, key=lambda entry: entry.get("createdAt") or 0, reverse=True)
    query = client.query(kind=_PERSONAL_LEDGER_KIND, ancestor=client.key("User", user_id))
    entries = [_personal_ledger_from_entity(entity) for entity in query.fetch(limit=500)]
    entries.sort(key=lambda entry: entry.get("createdAt") or 0, reverse=True)
    return entries


def _get_personal_ledger_entry(user_id: str, entry_id: str) -> Optional[Dict[str, Any]]:
    """Return one ledger entry only when it belongs to the signed-in user."""
    client = _get_datastore()
    if client is None:
        return _state.setdefault("personal_ledger", {}).setdefault(user_id, {}).get(entry_id)
    entity = client.get(_personal_ledger_key(client, user_id, entry_id))
    return _personal_ledger_from_entity(entity) if entity is not None else None


def _put_personal_ledger_entry(user_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    entry = {field: entry.get(field) for field in _PERSONAL_LEDGER_FIELDS}
    existing = _get_personal_ledger_entry(user_id, entry["id"])
    if existing is not None:
        # Adding the same chat response is idempotent; never erase the user's
        # explicit correctness judgement during an add/retry.
        for field in ("user_verdict", "judgedAt"):
            if entry.get(field) is None:
                entry[field] = existing.get(field)
    client = _get_datastore()
    if client is None:
        _state.setdefault("personal_ledger", {}).setdefault(user_id, {})[entry["id"]] = entry
        return entry
    from google.cloud import datastore as _ds
    key = _personal_ledger_key(client, user_id, entry["id"])
    indexed = {"id", "conversation_id", "message_id", "probability", "model", "createdAt"}
    entity = _ds.Entity(key=key, exclude_from_indexes=tuple(field for field in entry if field not in indexed))
    entity.update(entry)
    client.put(entity)
    return entry


def _delete_personal_ledger_entry(user_id: str, entry_id: str) -> None:
    client = _get_datastore()
    if client is None:
        _state.setdefault("personal_ledger", {}).setdefault(user_id, {}).pop(entry_id, None)
        return
    client.delete(_personal_ledger_key(client, user_id, entry_id))


# Private copied-agent recipes (per-user on Datastore).
_AGENT_PROFILE_KIND = "AgentProfile"
_AGENT_PROFILE_FIELDS = (
    "id", "name", "source_agent_id", "model", "instruction", "version",
    "execution_mode", "created_at", "updated_at",
)
_MAX_AGENT_PROFILES_PER_USER = 5

# Durable research runs are strictly private to the signed-in user. The
# request snapshot deliberately excludes browser/provider secrets and prior
# conversation content; the run stores only its public market inputs, progress
# events, and the resulting research report.
_AGENT_RUN_KIND = "AgentRun"
_AGENT_RUN_FIELDS = (
    "id", "status", "title", "question", "platform", "recommendation",
    "model_probability", "market_probability", "edge", "agent_profile",
    "request", "report", "timeline", "steps", "created_at", "updated_at",
    "completed_at", "error_code", "client_run_key",
)
_MAX_AGENT_RUNS_PER_USER = 100


def _agent_run_key(client: Any, user_id: str, run_id: str) -> Any:
    return client.key("User", user_id, _AGENT_RUN_KIND, run_id)


def _agent_run_from_entity(entity: Any) -> Dict[str, Any]:
    record = {field: entity.get(field) for field in _AGENT_RUN_FIELDS}
    record["id"] = record.get("id") or entity.key.name
    return record


def _list_agent_runs(user_id: str) -> List[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        records = list(_state.setdefault("agent_runs", {}).get(user_id, {}).values())
    else:
        query = client.query(kind=_AGENT_RUN_KIND, ancestor=client.key("User", user_id))
        records = [_agent_run_from_entity(entity) for entity in query.fetch(limit=_MAX_AGENT_RUNS_PER_USER)]
    return sorted(records, key=lambda record: record.get("updated_at") or "", reverse=True)


def _read_agent_run(user_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        record = _state.setdefault("agent_runs", {}).setdefault(user_id, {}).get(run_id)
        return dict(record) if record else None
    entity = client.get(_agent_run_key(client, user_id, run_id))
    return _agent_run_from_entity(entity) if entity is not None else None


def _put_agent_run(user_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    record = {field: record.get(field) for field in _AGENT_RUN_FIELDS}
    client = _get_datastore()
    if client is None:
        _state.setdefault("agent_runs", {}).setdefault(user_id, {})[record["id"]] = record
        return record
    from google.cloud import datastore as _ds

    entity = _ds.Entity(
        key=_agent_run_key(client, user_id, record["id"]),
        exclude_from_indexes=("title", "question", "request", "report", "timeline", "steps", "agent_profile"),
    )
    entity.update(record)
    client.put(entity)
    return record


def _list_stale_running_agent_runs(cutoff_iso: str, limit: int) -> List[tuple[str, Dict[str, Any]]]:
    """Cross-user, bounded set of AgentRun records stuck at status="running"
    with no update since `cutoff_iso` -- almost certainly orphaned by a
    crashed or recycled request, not still legitimately in flight."""
    bounded_limit = max(1, min(int(limit), _AGENT_RUN_RECONCILIATION_MAX_RUNS))
    client = _get_datastore()
    if client is None:
        candidates = [
            (user_id, dict(record))
            for user_id, records in _state.setdefault("agent_runs", {}).items()
            for record in records.values()
            if record.get("status") == "running" and (record.get("updated_at") or "") < cutoff_iso
        ]
        return sorted(candidates, key=lambda item: item[1].get("updated_at") or "")[:bounded_limit]

    query = client.query(kind=_AGENT_RUN_KIND)
    query.add_filter("status", "=", "running")
    query.add_filter("updated_at", "<", cutoff_iso)
    query.order = ["updated_at"]
    candidates: List[tuple[str, Dict[str, Any]]] = []
    for entity in query.fetch(limit=bounded_limit):
        parent = entity.key.parent
        user_id = parent.name if parent is not None else None
        if not user_id:
            continue
        candidates.append((str(user_id), _agent_run_from_entity(entity)))
    return candidates


def _agent_profile_key(client: Any, user_id: str, profile_id: str) -> Any:
    return client.key("User", user_id, _AGENT_PROFILE_KIND, profile_id)


def _agent_profile_from_entity(entity: Any) -> Dict[str, Any]:
    profile = {field: entity.get(field) for field in _AGENT_PROFILE_FIELDS}
    profile["id"] = entity.key.name
    return profile


def _list_agent_profiles(user_id: str) -> List[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        profiles = list(_state.setdefault("agent_profiles", {}).get(user_id, {}).values())
    else:
        query = client.query(kind=_AGENT_PROFILE_KIND, ancestor=client.key("User", user_id))
        profiles = [_agent_profile_from_entity(entity) for entity in query.fetch(limit=_MAX_AGENT_PROFILES_PER_USER)]
    return sorted(profiles, key=lambda profile: profile.get("created_at") or "", reverse=True)


def _read_agent_profile(user_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        profile = _state.setdefault("agent_profiles", {}).setdefault(user_id, {}).get(profile_id)
        return dict(profile) if profile else None
    entity = client.get(_agent_profile_key(client, user_id, profile_id))
    return _agent_profile_from_entity(entity) if entity is not None else None


def _put_agent_profile(user_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = {field: profile.get(field) for field in _AGENT_PROFILE_FIELDS}
    client = _get_datastore()
    if client is None:
        _state.setdefault("agent_profiles", {}).setdefault(user_id, {})[profile["id"]] = profile
        return profile
    from google.cloud import datastore as _ds

    entity = _ds.Entity(
        key=_agent_profile_key(client, user_id, profile["id"]),
        exclude_from_indexes=("instruction",),
    )
    entity.update(profile)
    client.put(entity)
    return profile


def _delete_agent_profile(user_id: str, profile_id: str) -> None:
    client = _get_datastore()
    if client is None:
        _state.setdefault("agent_profiles", {}).setdefault(user_id, {}).pop(profile_id, None)
        return
    client.delete(_agent_profile_key(client, user_id, profile_id))


# ── RAG knowledge base (per-user vector store on Datastore) ──────────────────

def _rag_fetch(user_id: str, namespace: str, limit: int = 2000) -> List[Dict[str, Any]]:
    """All stored chunks (with embeddings) for a user/namespace."""
    client = _get_datastore()
    if client is None:
        store = _state.setdefault("rag", {}).get(user_id, [])
        return [c for c in store if c.get("namespace") == namespace]
    query = client.query(kind="VectorChunk", ancestor=client.key("User", user_id))
    query.add_filter("namespace", "=", namespace)
    return [dict(e) for e in query.fetch(limit=limit)]


def _rag_add(user_id: str, namespace: str, items: List[Dict[str, Any]]) -> int:
    """Chunk, embed, and store documents. Returns the number of chunks stored."""
    from analyzing_llm_rationale import rag

    chunks: List[tuple] = []
    for item in items:
        doc_id = item.get("doc_id") or ("doc_" + secrets.token_hex(6))
        for ch in rag.chunk_text(item.get("text") or ""):
            chunks.append((ch, item.get("title") or "", item.get("url") or "", item.get("source") or "", doc_id))
    if not chunks:
        return 0
    vectors = rag.embed([c[0] for c in chunks])
    if not vectors or not vectors[0]:
        raise RuntimeError("Embeddings are unavailable on this server.")
    now = datetime.now(timezone.utc)
    records = [
        {"namespace": namespace, "doc_id": doc_id, "text": text, "title": title,
         "url": url, "source": source, "embedding": emb, "created_at": now}
        for (text, title, url, source, doc_id), emb in zip(chunks, vectors)
    ]
    client = _get_datastore()
    if client is None:
        _state.setdefault("rag", {}).setdefault(user_id, []).extend(records)
        return len(records)
    from google.cloud import datastore as _ds
    entities = []
    for r in records:
        entity = _ds.Entity(
            key=client.key("User", user_id, "VectorChunk"),
            exclude_from_indexes=("text", "embedding", "title", "url", "source"),
        )
        entity.update(r)
        entities.append(entity)
    for i in range(0, len(entities), 100):  # Datastore put_multi caps at 500
        client.put_multi(entities[i:i + 100])
    return len(records)


def _rag_search(user_id: str, namespace: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    from analyzing_llm_rationale import rag
    qvec = rag.embed_one(query)
    if not qvec:
        return []
    return rag.top_k(qvec, _rag_fetch(user_id, namespace), k=k)


def _rag_documents(user_id: str, namespace: str) -> List[Dict[str, Any]]:
    """Distinct ingested documents (grouped by doc_id) with chunk counts."""
    docs: Dict[str, Dict[str, Any]] = {}
    for c in _rag_fetch(user_id, namespace):
        d = docs.setdefault(c.get("doc_id") or "", {
            "doc_id": c.get("doc_id"), "title": c.get("title"),
            "url": c.get("url"), "source": c.get("source"), "chunks": 0,
        })
        d["chunks"] += 1
    return list(docs.values())


def _rag_delete(user_id: str, namespace: str, doc_id: Optional[str] = None) -> int:
    client = _get_datastore()
    if client is None:
        store = _state.setdefault("rag", {}).get(user_id, [])
        keep = [c for c in store if not (c.get("namespace") == namespace and (doc_id is None or c.get("doc_id") == doc_id))]
        removed = len(store) - len(keep)
        _state["rag"][user_id] = keep
        return removed
    query = client.query(kind="VectorChunk", ancestor=client.key("User", user_id))
    query.add_filter("namespace", "=", namespace)
    keys = [e.key for e in query.fetch() if doc_id is None or e.get("doc_id") == doc_id]
    for i in range(0, len(keys), 200):
        client.delete_multi(keys[i:i + 200])
    return len(keys)


# ── Shared cache / coordination (Redis, optional) ───────────────────────────
_redis_client: Any = None
_redis_initialised = False


def _get_redis() -> Any:
    """Return a shared Redis client when ``REDIS_URL`` is configured, else None.

    Redis (Cloud Memorystore in production) lets multiple Cloud Run instances
    share rate-limit counters and caches. When it is absent or unreachable,
    callers fall back to per-instance in-memory state.
    """
    global _redis_client, _redis_initialised
    if _redis_initialised:
        return _redis_client
    _redis_initialised = True
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        import redis as _redis  # optional dependency
        client = _redis.from_url(
            url,
            socket_connect_timeout=1,
            socket_timeout=1,
            decode_responses=True,
        )
        client.ping()
        _redis_client = client
    except Exception:
        _redis_client = None
    return _redis_client


# ── Cache (Redis-backed, per-instance in-memory fallback) ────────────────────
_local_cache: "OrderedDict[str, Any]" = OrderedDict()
_background_tasks: set = set()
_evidence_prefetch_inflight: set[str] = set()
_LOCAL_CACHE_MAX = int(os.environ.get("LOCAL_CACHE_MAX", "1024"))
_EVIDENCE_CACHE_TTL = int(os.environ.get("EVIDENCE_CACHE_TTL", "900"))
_EVIDENCE_PREFETCH_TOP_N = int(os.environ.get("EVIDENCE_PREFETCH_TOP_N", "3"))
_EVIDENCE_TIMEOUT_S = float(os.environ.get("EVIDENCE_TIMEOUT_S", "0"))
_EVIDENCE_MAX_CONCURRENCY = max(1, int(os.environ.get("EVIDENCE_MAX_CONCURRENCY", "4")))
_evidence_fetch_slots = threading.BoundedSemaphore(_EVIDENCE_MAX_CONCURRENCY)
_EXTRACT_CACHE_TTL = int(os.environ.get("EXTRACT_CACHE_TTL", "3600"))
_PREDICT_CACHE_TTL = int(os.environ.get("PREDICT_CACHE_TTL", "600"))


def _cache_key(*parts: Any) -> str:
    """Stable cache key from arbitrary JSON-serialisable parts."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Any:
    """Return a cached value (Redis first, then in-memory), or None on miss."""
    client = _get_redis()
    if client is not None:
        try:
            raw = client.get(f"cache:{key}")
            return json.loads(raw) if raw is not None else None
        except Exception:
            pass  # fall back to local cache
    entry = _local_cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if expires_at < time.time():
        _local_cache.pop(key, None)
        return None
    _local_cache.move_to_end(key)
    return value


def _cache_set(key: str, value: Any, ttl: int) -> None:
    """Store a JSON-serialisable value with a TTL (seconds). No-op if ttl <= 0."""
    if ttl <= 0:
        return
    client = _get_redis()
    if client is not None:
        try:
            client.set(f"cache:{key}", json.dumps(value, default=str), ex=ttl)
            return
        except Exception:
            pass  # fall back to local cache
    _local_cache[key] = (time.time() + ttl, value)
    _local_cache.move_to_end(key)
    while len(_local_cache) > _LOCAL_CACHE_MAX:
        _local_cache.popitem(last=False)


def _spawn_background(awaitable) -> None:
    """Keep best-effort async work alive without holding the response open."""
    task = asyncio.ensure_future(awaitable)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _read_live_track_record() -> Optional[Dict[str, Any]]:
    """Return the committed live track-record aggregate, or None.

    Tries (cached): raw GitHub copy → bundled file. Synchronous; call via
    ``run_in_executor`` from async handlers. Fails open to None so the caller
    falls back to the static backtest.
    """
    import requests

    cache_key = _cache_key("track_record_live", "v3")
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    payload: Optional[Dict[str, Any]] = None
    try:
        sep = "&" if "?" in _TRACK_RECORD_LIVE_URL else "?"
        cache_busted_url = f"{_TRACK_RECORD_LIVE_URL}{sep}_={int(time.time() // max(_TRACK_RECORD_LIVE_TTL, 1))}"
        resp = requests.get(
            cache_busted_url,
            timeout=6,
            headers={
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "Foresea/edge-board-live",
            },
        )
        if resp.status_code == 200:
            payload = resp.json()
    except Exception:
        logger.warning("live track record fetch failed; trying bundled copy", exc_info=True)
    if payload is None:
        bundled = _STATIC_DIR / "track_record_live.json"
        if bundled.exists():
            try:
                payload = json.loads(bundled.read_text())
            except Exception:
                logger.warning("bundled live track record unreadable", exc_info=True)
    if payload is not None:
        _cache_set(cache_key, payload, _TRACK_RECORD_LIVE_TTL)
    return payload


def _track_record_freshness(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    generated_at = (payload or {}).get("generated_at")
    age_s = None
    if generated_at:
        try:
            dt = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_s = max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
        except Exception:
            age_s = None
    stale = age_s is None or age_s > _EDGE_BOARD_STALE_AFTER_S
    return {
        "generated_at": generated_at,
        "age_seconds": age_s,
        "stale": stale,
        "stale_after_seconds": _EDGE_BOARD_STALE_AFTER_S,
    }


# ── Evolution-loop feedback: live calibration + model auto-selection ──────────
def _sample_indices(n: int, max_points: int) -> List[int]:
    """Evenly sample an ordered series while preserving both endpoints."""
    if n <= max_points:
        return list(range(n))
    if max_points <= 1:
        return [0]
    return sorted({round(i * (n - 1) / (max_points - 1)) for i in range(max_points)})


def _sample_list(values: Any, max_points: int = _EDGE_BOARD_CURVE_MAX_POINTS) -> Any:
    if not isinstance(values, list) or len(values) <= max_points:
        return values
    return [values[i] for i in _sample_indices(len(values), max_points)]


def _compact_pnl_strategy(strategy: Any) -> Any:
    if not isinstance(strategy, dict):
        return strategy
    keep = {
        "n_bets",
        "roi",
        "win_rate",
        "total_staked",
        "compound_bankroll",
        "compound_return",
        "growth_curve",
        "growth_curve_ts",
        "equity_curve",
        "equity_curve_ts",
        "compound_curve",
        "compound_curve_ts",
        "cumulative_curve",
        "cumulative_curve_ts",
    }
    compact = {k: strategy.get(k) for k in keep if k in strategy}
    for curve_key in (
        "growth_curve",
        "growth_curve_ts",
        "equity_curve",
        "equity_curve_ts",
        "compound_curve",
        "compound_curve_ts",
        "cumulative_curve",
        "cumulative_curve_ts",
    ):
        if curve_key in compact:
            compact[curve_key] = _sample_list(compact[curve_key])
    return compact


def _compact_paper_pnl(pnl: Any) -> Any:
    if not isinstance(pnl, dict):
        return pnl
    compact: Dict[str, Any] = {}
    for key, value in pnl.items():
        if isinstance(value, dict) and (
            "growth_curve" in value
            or "equity_curve" in value
            or "compound_curve" in value
            or "cumulative_curve" in value
            or "n_bets" in value
        ):
            compact[key] = _compact_pnl_strategy(value)
        elif key in {"disclaimer", "methodology", "notes"}:
            compact[key] = value
    return compact


def _compact_models_comparison(models: Any) -> List[Dict[str, Any]]:
    compact_models: List[Dict[str, Any]] = []
    for model in models or []:
        if not isinstance(model, dict):
            continue
        compact: Dict[str, Any] = {
            key: model.get(key)
            for key in (
                "model",
                "n_snapshots_resolved",
                "n_markets_resolved",
                "accuracy",
                "model_brier",
                "skill_vs_market",
                "paper_roi",
                "paper_roi_smart",
                "by_horizon",
            )
            if key in model
        }
        if "paper_pnl" in model:
            compact["paper_pnl"] = _compact_paper_pnl(model.get("paper_pnl"))
        compact_models.append(compact)
    return compact_models


def _compact_mark_to_market_account(account: Any, max_points: int = _EDGE_BOARD_CURVE_MAX_POINTS) -> Any:
    if not isinstance(account, dict):
        return account
    compact = {
        key: account.get(key)
        for key in (
            "account_value",
            "cash",
            "liquidation_value",
            "return",
            "realized_pnl",
            "unrealized_pnl",
            "n_open_positions",
            "n_illiquid_positions",
            "n_settlements",
            "n_trades",
            "notes",
            "ts",
            "value_method",
        )
        if key in account
    }
    if "value_curve" in account:
        compact["value_curve"] = _sample_list(account.get("value_curve"), max_points=max_points)
    return compact


def _compact_mark_to_market_by_model(rows: Any) -> List[Dict[str, Any]]:
    compact_rows: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        compact = {
            key: row.get(key)
            for key in (
                "model",
                "account_value",
                "cash",
                "liquidation_value",
                "return",
                "realized_pnl",
                "unrealized_pnl",
                "n_open_positions",
                "n_illiquid_positions",
                "n_settlements",
                "n_trades",
                "status",
            )
            if key in row
        }
        compact["account"] = _compact_mark_to_market_account(row.get("account"), max_points=40)
        compact_rows.append(compact)
    return compact_rows


_LIVE_TRACK_RECORD_READER = live_track_record_support.LiveTrackRecordReader(
    cache_key=_cache_key,
    cache_get=_cache_get,
    cache_set=_cache_set,
    config=live_track_record_support.LiveTrackRecordConfig(
        live_url=_TRACK_RECORD_LIVE_URL,
        ttl_seconds=_TRACK_RECORD_LIVE_TTL,
        stale_after_seconds=_EDGE_BOARD_STALE_AFTER_S,
        bundled_path=_STATIC_DIR / "track_record_live.json",
        request_timeout_seconds=_TRACK_RECORD_LIVE_TIMEOUT,
    ),
    logger=logger,
)
_read_live_track_record = _LIVE_TRACK_RECORD_READER.read
_track_record_freshness = _LIVE_TRACK_RECORD_READER.freshness
_FORECAST_EVALUATION_READER = live_track_record_support.LiveTrackRecordReader(
    cache_key=_cache_key,
    cache_get=_cache_get,
    cache_set=_cache_set,
    config=live_track_record_support.LiveTrackRecordConfig(
        live_url=_FORECAST_EVALUATION_URL,
        ttl_seconds=_FORECAST_EVALUATION_TTL,
        stale_after_seconds=_FORECAST_EVALUATION_STALE_AFTER_S,
        bundled_path=_STATIC_DIR / "forecast_evaluation.json",
        request_timeout_seconds=_FORECAST_EVALUATION_TIMEOUT,
        cache_namespace="forecast_evaluation",
        cache_version="v1",
        resource_label="forecast evaluation report",
        user_agent="Foresea/forecast-evaluation",
    ),
    logger=logger,
)
_read_forecast_evaluation = _FORECAST_EVALUATION_READER.read
_forecast_evaluation_freshness = _FORECAST_EVALUATION_READER.freshness
_MARK_TO_MARKET_READER = live_track_record_support.LiveTrackRecordReader(
    cache_key=_cache_key,
    cache_get=_cache_get,
    cache_set=_cache_set,
    config=live_track_record_support.LiveTrackRecordConfig(
        live_url=_MARK_TO_MARKET_LIVE_URL,
        ttl_seconds=_MARK_TO_MARKET_LIVE_TTL,
        stale_after_seconds=_MARK_TO_MARKET_STALE_AFTER_S,
        bundled_path=_STATIC_DIR / "mark_to_market_live.json",
        request_timeout_seconds=_MARK_TO_MARKET_LIVE_TIMEOUT,
        cache_namespace="mark_to_market_live",
        cache_version="v1",
        resource_label="mark-to-market live report",
        user_agent="Foresea/mark-to-market-live",
    ),
    logger=logger,
)
_read_mark_to_market_record = _MARK_TO_MARKET_READER.read
_AGENT_TRADING_BOARD_READER = live_track_record_support.LiveTrackRecordReader(
    cache_key=_cache_key,
    cache_get=_cache_get,
    cache_set=_cache_set,
    config=live_track_record_support.LiveTrackRecordConfig(
        live_url=_AGENT_TRADING_BOARD_URL,
        ttl_seconds=_AGENT_TRADING_BOARD_TTL,
        stale_after_seconds=_AGENT_TRADING_BOARD_STALE_AFTER_S,
        bundled_path=_STATIC_DIR / "agent_trading_live.json",
        request_timeout_seconds=_AGENT_TRADING_BOARD_TIMEOUT,
        cache_namespace="agent_trading_live",
        cache_version="v1",
        resource_label="agent trading board",
        user_agent="Foresea/agent-trading-board",
    ),
    logger=logger,
)
_read_agent_trading_board = _AGENT_TRADING_BOARD_READER.read
_agent_trading_board_freshness = _AGENT_TRADING_BOARD_READER.freshness

_MARK_TO_MARKET_MERGE_KEYS = (
    "edge_board",
    "discrepancy_monitor",
    "arbitrage_signals",
    "mark_to_market_account",
    "mark_to_market_by_model",
    "quarter_kelly_by_model",
    "growth_1pct_by_model",
    "growth_2pct_by_model",
    "mark_to_market_cycle_minutes",
    "n_markets_open",
    "n_markets_tracked",
)


def _merge_mark_to_market_record(
    resolved_payload: Optional[Dict[str, Any]],
    mtm_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = dict(resolved_payload or {})
    if not mtm_payload:
        return merged
    resolved_generated_at = merged.get("generated_at")
    mtm_generated_at = mtm_payload.get("generated_at")
    for key in _MARK_TO_MARKET_MERGE_KEYS:
        if key in mtm_payload:
            merged[key] = mtm_payload[key]
    if mtm_generated_at:
        merged["generated_at"] = mtm_generated_at
        merged["mark_to_market_generated_at"] = mtm_generated_at
        if resolved_generated_at:
            merged["resolved_generated_at"] = resolved_generated_at
    return merged


def _read_edge_board_record() -> Dict[str, Any]:
    return _merge_mark_to_market_record(
        _read_live_track_record(),
        _read_mark_to_market_record(),
    )


_AUTO_SELECT_MODEL = os.environ.get("AUTO_SELECT_MODEL", "1").lower() not in {"0", "false", "no"}
_MODEL_SWITCH_MARGIN = float(os.environ.get("MODEL_SWITCH_MARGIN", "0.02"))


# Isotonic calibration is kept as a future experiment — NOT applied to live
# predictions (prediction markets are context-dependent; a global map is too naive).
# _calibrate_probability is retained so existing tests and future callers work.
def _calibrate_probability(p: Optional[float]) -> Optional[float]:
    """Apply the isotonic calibration map from the live track record, or pass through.
    Currently a no-op in production because calibration_model.applied is always False;
    kept so the function can be wired back in when a better calibration strategy exists."""
    if p is None:
        return p
    try:
        cal = (_read_live_track_record() or {}).get("calibration_model") or {}
        if not cal.get("applied"):
            return p
        bps = cal.get("breakpoints") or []
        if len(bps) < 2:
            return p
        from analyzing_llm_rationale import track_record_live as _trl
        m = ([float(b[0]) for b in bps], [float(b[1]) for b in bps])
        return _trl._apply_isotonic(m, float(p))
    except Exception:
        return p


def _auto_selected_model() -> Optional[str]:
    """Best Smart-strategy paper-edge model from the live `models_comparison`, gated on
    a margin over the configured default (anti-thrash). Returns an allowlisted
    label to forecast with, or None to keep the default. No-op until resolved data
    produces a significant per-model edge."""
    if not _AUTO_SELECT_MODEL:
        return None
    try:
        comp = (_read_live_track_record() or {}).get("models_comparison") or []
        default = _state.get("model_key")
        inc_roi = next((m.get("paper_roi_smart") for m in comp if m.get("model") == default), None)
        if inc_roi is None:
            return None
        best, best_roi = None, None
        for m in comp:
            label, roi = m.get("model"), m.get("paper_roi_smart")
            if label in _SCADS_MODEL_ALLOWLIST and roi is not None and (best_roi is None or roi > best_roi):
                best, best_roi = label, roi
        if not best or best == default:
            return None
        if best_roi - inc_roi >= _MODEL_SWITCH_MARGIN:
            return best
        return None
    except Exception:
        return None

_rate_limiter = RateLimiter(calls=int(os.environ.get("RATE_LIMIT_PER_MIN", "60")), period=60)
_predict_rate_limiter = RateLimiter(calls=int(os.environ.get("PREDICT_RATE_LIMIT_PER_MIN", "10")), period=60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    _ready = True
    # Fail fast at startup rather than on the first form/file request. FastAPI
    # only checks for python-multipart when a matching endpoint is called, so a
    # missing package produces a runtime 500 instead of a startup error.
    try:
        import multipart  # noqa: F401
    except ImportError:
        raise RuntimeError(
            'python-multipart is required for Form/File endpoints. '
            'Install the serve extras: pip install "analyzing-llm-rationale[serve]"'
        ) from None
    logger.info("foresea server starting up")
    async with AsyncExitStack() as stack:
        if _PUBLIC_MCP is not None:
            await stack.enter_async_context(_PUBLIC_MCP.session_manager.run())
            logger.info("foresea public MCP endpoint mounted at /mcp")
        yield
        # Graceful shutdown: flip readiness so the load balancer stops routing new
        # traffic while uvicorn drains in-flight requests on SIGTERM (Cloud Run
        # instance recycle).
        _ready = False
        logger.info("foresea server shutting down (draining in-flight requests)")


app = FastAPI(
    title="Foresea Intelligence API",
    description=_DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "Pareel Amre",
        "email": "pareel.amre@gmail.com",
        "url": "https://github.com/pareelamre/analyzing-llm-rationale",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=_TAGS,
    lifespan=lifespan,
)

_cors_origins = [
    "https://foresea.ink",
    "https://www.foresea.ink",
    "http://localhost:8000",
    "http://localhost:3000",
]
if os.environ.get("CORS_EXTRA_ORIGIN"):
    _cors_origins.append(os.environ["CORS_EXTRA_ORIGIN"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Content-Type",
        "X-API-Key",
        "Authorization",
        "Mcp-Session-Id",
        "MCP-Protocol-Version",
    ],
    expose_headers=["Mcp-Session-Id", "X-Request-ID"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)



def _mount_public_mcp_endpoint() -> None:
    """Expose Foresea as a remote MCP server when the SDK is installed."""
    global _PUBLIC_MCP, _PUBLIC_MCP_APP
    if os.environ.get("DISABLE_PUBLIC_MCP", "").lower() in {"1", "true", "yes"}:
        return
    try:
        from analyzing_llm_rationale.mcp_server import create_mcp_server

        timeout_s = float(os.environ.get("FORESEA_MCP_TIMEOUT_S", "120"))
        _PUBLIC_MCP = create_mcp_server(
            base_url=os.environ.get("FORESEA_MCP_UPSTREAM_URL", _CANONICAL),
            api_key=os.environ.get("FORESEA_MCP_UPSTREAM_API_KEY")
            or os.environ.get("FORESEA_API_KEY")
            or os.environ.get("API_KEY"),
            timeout_s=timeout_s,
            host="0.0.0.0",
            streamable_http_path="/",
        )
        _PUBLIC_MCP_APP = _PUBLIC_MCP.streamable_http_app()
        app.mount("/mcp", _PUBLIC_MCP_APP)
    except Exception as exc:
        _PUBLIC_MCP = None
        _PUBLIC_MCP_APP = None
        logger.warning("public MCP endpoint disabled: %s", exc)


_mount_public_mcp_endpoint()

init_observability(app)

_tracer = otel_trace.get_tracer("foresea.server")
_meter = otel_metrics.get_meter("foresea.server")

# Business metrics
_forecast_counter = _meter.create_counter(
    "forecast.requests", unit="1", description="Total forecast requests"
)
_forecast_errors = _meter.create_counter(
    "forecast.errors", unit="1", description="Forecast requests that errored"
)
_forecast_duration = _meter.create_histogram(
    "forecast.duration", unit="s", description="Forecast end-to-end latency"
)
_forecast_stream_prepare_duration = _meter.create_histogram(
    "forecast.stream.prepare.duration",
    unit="s",
    description="Latency before a forecast stream is ready to call the model",
)
_forecast_stream_first_token_duration = _meter.create_histogram(
    "forecast.stream.first_token.duration",
    unit="s",
    description="Latency from forecast stream start to first model token",
)
_llm_calls = _meter.create_counter(
    "llm.calls", unit="1", description="LLM provider call attempts"
)
_llm_errors = _meter.create_counter(
    "llm.errors", unit="1", description="LLM provider call failures"
)
_council_member_calls = _meter.create_counter(
    "forecast.council.member.calls",
    unit="1",
    description="Council member calls by round and outcome",
)
_council_member_duration = _meter.create_histogram(
    "forecast.council.member.duration",
    unit="s",
    description="Council member call latency",
)
_council_requests = _meter.create_counter(
    "forecast.council.requests",
    unit="1",
    description="Council forecast requests by outcome",
)
_agent_counter = _meter.create_counter(
    "agent.requests", unit="1", description="Agent analyze requests"
)
_forecast_evaluation_reads = _meter.create_counter(
    "forecast_evaluation.reads",
    unit="1",
    description="Internal forecast evaluation report reads",
)
_forecast_context_packages = _meter.create_counter(
    "forecast.context.packages",
    unit="1",
    description="Forecast context packages by completeness",
)
_forecast_rag_contexts = _meter.create_counter(
    "forecast.rag.contexts",
    unit="1",
    description="Forecast knowledge-base context retrieval outcomes",
)
_forecast_evidence_requests = _meter.create_counter(
    "forecast.evidence.requests",
    unit="1",
    description="Forecast evidence retrieval cache and fetch outcomes",
)
_forecast_evidence_duration = _meter.create_histogram(
    "forecast.evidence.duration",
    unit="s",
    description="Forecast evidence retrieval latency",
)
_radar_evidence_prefetches = _meter.create_counter(
    "radar.evidence.prefetches",
    unit="1",
    description="Radar-triggered evidence prefetch outcomes",
)
_market_context_counter = _meter.create_counter(
    "market.context_enrichments",
    unit="1",
    description="Prediction-market context enrichment outcomes",
)
_personal_ledger_actions = _meter.create_counter(
    "personal_ledger.actions",
    unit="1",
    description="Personal-ledger actions by type and outcome",
)
_live_trade_intents = _meter.create_counter(
    "trading.live_trade_intents",
    unit="1",
    description="Chat research reports that produced a reviewable live-trade intent",
)
_trading_connection_actions = _meter.create_counter(
    "trading.connection.actions",
    unit="1",
    description="Secure exchange connection actions by venue, action, and outcome",
)
_trading_run_actions = _meter.create_counter(
    "trading.run.actions",
    unit="1",
    description="Durable trade-run lifecycle actions by venue, action, and outcome",
)
_trading_reconciliation_actions = _meter.create_counter(
    "trading.reconciliation.actions",
    unit="1",
    description="Trading portfolio, order, and cancellation reconciliation actions",
)
_trading_guardrail_actions = _meter.create_counter(
    "trading.guardrail.actions",
    unit="1",
    description="Trading risk-guardrail evaluations by venue, action, and outcome",
)
_trading_readiness_actions = _meter.create_counter(
    "trading.readiness.actions",
    unit="1",
    description="Operator launch-readiness checks for guarded live trading",
)
_agent_profile_actions = _meter.create_counter(
    "agent.profile.actions",
    unit="1",
    description="Private copied-agent profile actions by lifecycle outcome",
)
_agent_run_actions = _meter.create_counter(
    "agent.run.actions",
    unit="1",
    description="Durable agent-run lifecycle actions by outcome",
)
_analytics_attribution_actions = _meter.create_counter(
    "analytics.attribution.records",
    unit="1",
    description="Analytics records by authenticated or anonymous attribution",
)


# Redirect middleware: send requests from run.app hosts to the custom domain
def _should_redirect_run_app_hosts() -> bool:
    environment = (os.environ.get("ENVIRONMENT") or "").strip().lower()
    return environment in {"", "prod", "production"}


@app.middleware("http")
async def host_redirect_middleware(request: Request, call_next):
    host = (request.headers.get("host") or "").lower()
    if not _should_redirect_run_app_hosts():
        return await call_next(request)
    # Target domain can be overridden by env var CUSTOM_DOMAIN
    target_domain = os.environ.get("CUSTOM_DOMAIN", "foresea.ink").lower()
    # Redirect only run.app hosts (avoid loop when already on target domain)
    if host.endswith(".run.app") and target_domain and target_domain not in host:
        url = request.url
        new_url = f"https://{target_domain}{url.path}"
        if url.query:
            new_url = new_url + "?" + url.query
        return RedirectResponse(url=new_url, status_code=301)
    return await call_next(request)


# Middleware to catch unhandled exceptions: alert + return a scrubbed response.
# Intentional HTTPExceptions are handled by FastAPI before reaching here, so
# anything caught here is a genuine bug — never leak its traceback to the client.
@app.middleware("http")
async def exception_alert_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        request_id = getattr(request.state, "request_id", "-")
        tb = traceback.format_exc()
        logger.error("unhandled error [%s] %s %s\n%s",
                     request_id, request.method, request.url.path, tb)
        subject = f"Foresea server error: {type(exc).__name__}"
        body = (
            f"Request ID: {request_id}\n"
            f"Request: {request.method} {request.url}\n"
            f"Host: {request.headers.get('host')}\n\n"
            f"Exception:\n{tb}"
        )
        # Send email in background thread to avoid blocking the response.
        try:
            threading.Thread(target=_send_alert_email, args=(subject, body), daemon=True).start()
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )


# Reject oversized request bodies up front. Content-Length lets us reject before
# reading a request. For chunked and HTTP/2 uploads that omit it, wrap the ASGI
# receive channel so the request is rejected as soon as its running size exceeds
# the cap, before an endpoint can buffer the complete body.
@app.middleware("http")
async def body_size_limit_middleware(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > _MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds the {_MAX_BODY_BYTES // (1024 * 1024)} MB limit."},
                )
        except ValueError:
            pass
    elif request.method not in {"GET", "HEAD", "OPTIONS"}:
        receive = request._receive
        received_bytes = 0
        exceeded = False

        async def limited_receive():
            nonlocal exceeded, received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > _MAX_BODY_BYTES:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        request._receive = limited_receive
        response = await call_next(request)
        if exceeded:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body exceeds the {_MAX_BODY_BYTES // (1024 * 1024)} MB limit."},
            )
        return response
    return await call_next(request)


# Standard security headers on every response (table-stakes for a public API).
_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    # Permissive enough for the SPA's CDNs (Tailwind, jsDelivr, Google fonts/GIS)
    # while still blocking arbitrary origins, plugins, and framing.
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com "
        "https://cdn.jsdelivr.net https://accounts.google.com https://www.gstatic.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net "
        "https://accounts.google.com; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "connect-src 'self' https://api.openai.com https://openrouter.ai https://accounts.google.com "
        "https://www.gstatic.com; "
        "frame-src https://accounts.google.com; "
        "frame-ancestors 'none'; base-uri 'self'; object-src 'none'"
    ),
}


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    for key, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    return response


# Assign a request ID to every request (honouring an inbound X-Request-ID) so
# logs, error responses, and alerts can be correlated. Outermost middleware.
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _send_alert_email(subject: str, body: str) -> None:
    """Send a plain-text alert email. Configured via env vars:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_FROM, ALERT_TO
    """
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    alert_from = os.environ.get("ALERT_FROM", "noreply@foresea.ink")
    alert_to = os.environ.get("ALERT_TO", "pareel.amre@gmail.com")

    if not smtp_host:
        return

    msg = EmailMessage()
    msg["From"] = alert_from
    msg["To"] = alert_to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if smtp_user and smtp_password:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as s:
                s.starttls()
                s.login(smtp_user, smtp_password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as s:
                s.send_message(msg)
    except Exception:
        return


async def _provider_chat(
    provider,
    messages,
    temperature,
    max_tokens,
    *,
    timeout_s: Optional[float] = None,
    max_retries: Optional[int] = None,
    call_site: str = "predict",
    reasoning_effort: Optional[str] = None,
) -> str:
    """Run a blocking ``chat_completion`` in the default executor with a
    per-attempt timeout and bounded exponential backoff (+jitter) on transient
    failures.

    ``ContextLimitError`` and non-retryable ``ProviderResponseError`` propagate
    immediately — only ``RetryableProviderError`` and timeouts are retried.
    Callers map the raised exception to a clean HTTP status via
    :func:`_provider_http_error`.
    """
    from opentelemetry.trace import Status, StatusCode

    from analyzing_llm_rationale.providers import (
        ContextLimitError,
        RetryableProviderError,
    )

    model_name = getattr(provider, "model_name", "unknown")
    provider_type = type(provider).__name__
    effective_timeout_s = _PROVIDER_TIMEOUT_S if timeout_s is None else max(0.001, timeout_s)
    effective_max_retries = _PROVIDER_MAX_RETRIES if max_retries is None else max(0, max_retries)

    with _tracer.start_as_current_span("llm.chat_completion") as span:
        span.set_attributes({
            "gen_ai.request.model": model_name,
            "gen_ai.provider.name": provider_type,
            "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens,
            "foresea.llm.call_site": call_site,
        })
        _llm_calls.add(1, {
            "gen_ai.request.model": model_name,
            "gen_ai.provider.name": provider_type,
            "foresea.llm.call_site": call_site,
        })

        loop = asyncio.get_running_loop()
        attempts = effective_max_retries + 1
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: provider.chat_completion(
                            messages, temperature, max_tokens, reasoning_effort
                        ),
                    ),
                    timeout=effective_timeout_s,
                )
                span.set_attribute("outcome", "success")
                return result
            except ContextLimitError as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("outcome", "context_limit")
                _llm_errors.add(1, {"gen_ai.request.model": model_name, "error.type": "context_limit"})
                raise
            except (RetryableProviderError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    break
                delay = _PROVIDER_BACKOFF_BASE_S * (2 ** attempt)
                delay += random.uniform(0, delay * 0.25)  # jitter to avoid thundering herd
                logger.warning(
                    "provider call failed (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1, attempts, delay, type(exc).__name__,
                )
                await asyncio.sleep(delay)

        assert last_exc is not None
        span.record_exception(last_exc)
        span.set_status(Status(StatusCode.ERROR))
        span.set_attribute("outcome", "error")
        error_type = "timeout" if isinstance(last_exc, asyncio.TimeoutError) else "retryable"
        _llm_errors.add(1, {"gen_ai.request.model": model_name, "error.type": error_type})
        raise last_exc


async def _provider_stream_chat(
    provider,
    messages,
    temperature,
    max_tokens,
    *,
    first_token_timeout_s: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
):
    """Yield provider tokens from a blocking stream without blocking the event loop."""
    q: "queue.Queue[Any]" = queue.Queue()
    sentinel = object()

    def worker() -> None:
        try:
            for chunk in provider.stream_chat_completion(
                messages, temperature, max_tokens, reasoning_effort
            ):
                if chunk:
                    q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    loop = asyncio.get_running_loop()
    emitted = False
    while True:
        next_item = loop.run_in_executor(None, q.get)
        if first_token_timeout_s is not None and not emitted:
            item = await asyncio.wait_for(
                next_item,
                timeout=max(0.001, first_token_timeout_s),
            )
        else:
            item = await next_item
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        emitted = True
        yield str(item)


def _provider_http_error(exc: Exception, *, model_name: Optional[str] = None) -> HTTPException:
    """Map a provider exception to a clean, non-leaky HTTPException.

    The raw exception text (which can include upstream URLs/keys/prompts) is
    logged server-side, never returned to the client.
    """
    from analyzing_llm_rationale.providers import (
        ContextLimitError,
        RetryableProviderError,
    )

    logger.error("provider error: %s: %s", type(exc).__name__, exc)
    if isinstance(exc, ContextLimitError):
        return HTTPException(
            status_code=422,
            detail="The question plus evidence is too long for the model's context window. Try a shorter question or fewer attachments.",
        )
    if isinstance(exc, (RetryableProviderError, asyncio.TimeoutError)):
        # Structured (non-chat_mode) requests intentionally skip the chat
        # fallback chain -- see _chat_fallback_providers -- so a requested
        # model's own outage surfaces as-is instead of being silently masked
        # by a substituted model. Name the model so that's diagnosable
        # instead of reading as a generic, unexplained "forecasting is down".
        detail = "The forecasting model is temporarily unavailable. Please retry in a moment."
        if model_name:
            detail = (
                f"The '{model_name}' forecasting model is temporarily unavailable. "
                "Please retry in a moment, or pass a different `model` in the request."
            )
        return HTTPException(
            status_code=503,
            detail=detail,
            headers={"Retry-After": "10"},
        )
    return HTTPException(status_code=502, detail="The forecasting model returned an unexpected response.")


def _json_script_payload(payload: Dict[str, Any]) -> str:
    """Serialize data for an inert JSON script tag without allowing HTML breaks."""
    return (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _page_context(request: Request, page: str) -> Dict[str, Any]:
    path = request.url.path or "/"
    return {
        "page": page,
        "path": path,
        "canonical": f"{_CANONICAL}{path}",
        "api": {
            "authConfig": "/auth/config",
            "radar": "/radar",
            "trackRecord": "/track-record",
            "marketForecast": "/market/forecast",
            "marketForecastStream": "/market/forecast/stream",
            "analyticsEvent": "/analytics/event",
        },
        "auth": {
            "google": bool(_GOOGLE_CLIENT_ID),
            "github": bool(_GITHUB_CLIENT_ID),
        },
    }


def _inject_page_context(source: str, context: Dict[str, Any]) -> str:
    marker = f'id="{_PAGE_CONTEXT_SCRIPT_ID}"'
    if marker in source:
        return source
    tag = (
        f'\n  <script type="application/json" id="{_PAGE_CONTEXT_SCRIPT_ID}">'
        f"{_json_script_payload(context)}</script>\n"
    )
    if "</head>" in source:
        return source.replace("</head>", f"{tag}</head>", 1)
    return f"{tag}{source}"


def _render_static_html_page(
    filename: str,
    request: Request,
    *,
    page: str,
    cache_control: str,
) -> Response:
    source = (_STATIC_DIR / filename).read_text(encoding="utf-8")
    html_source = _inject_page_context(source, _page_context(request, page))
    return Response(
        html_source,
        media_type="text/html",
        headers={"Cache-Control": cache_control},
    )


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index(request: Request) -> Response:
    # Revalidate every load so deploys of the single-file SPA show immediately
    # (browser still gets a cheap 304 when unchanged).
    return _render_static_html_page(
        "index.html",
        request,
        page="home",
        cache_control="no-cache",
    )


async def _spa_page(request: Request, page: str) -> Response:
    """Serve the app shell for a client-side route that must survive refresh."""
    return _render_static_html_page(
        "index.html",
        request,
        page=page,
        cache_control="no-cache",
    )


@app.get("/ask", include_in_schema=False)
async def ask_page(request: Request) -> Response:
    return await _spa_page(request, "ask")


@app.get("/edge", include_in_schema=False)
async def edge_page(request: Request) -> Response:
    return await _spa_page(request, "edge")


@app.get("/edge/{panel:path}", include_in_schema=False)
async def edge_panel_page(panel: str, request: Request) -> Response:
    """Refresh-safe URL for one edge-board sub-panel (markets/mtm/agentic)."""
    if panel not in ("mtm", "agentic"):
        raise HTTPException(status_code=404)
    return await _spa_page(request, "edge")


@app.get("/track", include_in_schema=False)
async def track_page(request: Request) -> Response:
    return await _spa_page(request, "track")


@app.get("/ledger", include_in_schema=False)
async def ledger_page(request: Request) -> Response:
    return await _spa_page(request, "ledger")


@app.get("/agents", include_in_schema=False)
async def agents_page(request: Request) -> Response:
    """Human- and crawler-readable integration surface for AI agents."""
    return _render_static_html_page(
        "agents.html",
        request,
        page="agents",
        cache_control="public, max-age=300",
    )


@app.get("/forecast/{share_id}", include_in_schema=False)
async def shared_forecast_page(share_id: str, request: Request) -> Response:
    if not re.fullmatch(r"[A-Za-z0-9]{6,32}", share_id):
        raise HTTPException(status_code=404, detail="Forecast not found.")
    payload = await asyncio.get_running_loop().run_in_executor(None, _read_shared_forecast, share_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Forecast not found.")
    try:
        await asyncio.get_running_loop().run_in_executor(
            None,
            _record_analytics_event,
            AnalyticsEventRequest(event_name="share_page_view", path=f"/forecast/{share_id}", metadata={}),
            request,
        )
    except Exception:
        pass
    return Response(
        _shared_forecast_html(share_id, payload),
        media_type="text/html",
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/widget.js", include_in_schema=False)
async def widget_js() -> Response:
    """Serve Foresea embeddable drop-in widget javascript with global CORS."""
    widget_file = _STATIC_DIR / "widget.js"
    if widget_file.exists():
        content = widget_file.read_text(encoding="utf-8")
    else:
        content = "// Foresea Widget"
    return Response(
        content,
        media_type="application/javascript",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/embed/forecast/{share_id}", include_in_schema=False)
async def embed_forecast(share_id: str) -> Response:
    """Serve lightweight iframe-friendly embed page for a shared forecast."""
    if not re.fullmatch(r"[A-Za-z0-9]{6,32}", share_id):
        raise HTTPException(status_code=404, detail="Forecast not found.")
    payload = await asyncio.get_running_loop().run_in_executor(None, _read_shared_forecast, share_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Forecast not found.")
    q = html.escape(str(payload.get("question") or "Foresea Forecast"))
    prob = payload.get("probability")
    p_val = float(prob) if isinstance(prob, (int, float)) else 0.5
    ans = str(payload.get("predicted_answer") or ("YES" if p_val >= 0.5 else "NO")).upper()
    pct = round(p_val * 100)
    doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{ margin: 0; padding: 12px; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #e2e8f0; }}
    .card {{ background: #121824; border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 14px 16px; box-sizing: border-box; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; font-size: 11px; font-weight: bold; color: #00d2ff; text-transform: uppercase; margin-bottom: 8px; }}
    .q {{ font-size: 14px; font-weight: 600; line-height: 1.3; margin-bottom: 12px; color: #fff; }}
    .bar-bg {{ height: 8px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; width: {pct}%; background: linear-gradient(90deg, #00d2ff, #00f5a0); border-radius: 4px; }}
    .bar-labels {{ display: flex; justify-content: space-between; font-size: 12px; font-weight: bold; margin-bottom: 4px; }}
    .footer {{ margin-top: 10px; font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between; }}
    .footer a {{ color: inherit; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header"><span>🌊 Foresea Calibrated AI</span></div>
    <div class="q">{q}</div>
    <div class="bar-labels"><span>Probability: {pct}%</span><span style="color: {'#00f5a0' if pct >= 50 else '#ff5577'}">{ans}</span></div>
    <div class="bar-bg"><div class="bar-fill"></div></div>
    <div class="footer"><span>Transparent Track Record</span><a href="{_CANONICAL}/forecast/{share_id}" target="_blank">View Thesis →</a></div>
  </div>
</body>
</html>"""
    return Response(doc, media_type="text/html", headers={"Cache-Control": "public, max-age=300", "Access-Control-Allow-Origin": "*"})


# ── AI-agent / crawler discoverability ────────────────────────────────────────


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    """Welcome search + AI/LLM crawlers (most of these are blocked by default on
    other sites); point them at the sitemap and the machine-readable llms.txt."""
    bots = ["GPTBot", "OAI-SearchBot", "ChatGPT-User", "ClaudeBot", "anthropic-ai",
            "Claude-Web", "PerplexityBot", "Perplexity-User", "Google-Extended",
            "Applebot-Extended", "CCBot", "Bingbot", "Googlebot"]
    lines = ["# Foresea welcomes AI agents and crawlers.", "User-agent: *", "Allow: /", ""]
    for b in bots:
        lines += [f"User-agent: {b}", "Allow: /", ""]
    lines += [f"Sitemap: {_CANONICAL}/sitemap.xml",
              f"# Agent integration guide: {_CANONICAL}/agents",
              f"# Agent manifest: {_CANONICAL}/.well-known/agent.json",
              f"# Machine-readable guide for LLMs: {_CANONICAL}/llms.txt",
              f"# Remote MCP server: {_MCP_ENDPOINT}",
              f"# MCP discovery manifest: {_CANONICAL}/.well-known/mcp/server.json",
              f"# OpenAPI spec: {_CANONICAL}/openapi.json"]
    return PlainTextResponse("\n".join(lines),
                             headers={"Cache-Control": "public, max-age=86400"})


def _mcp_server_manifest() -> Dict[str, Any]:
    return {
        "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
        "name": "ink.foresea/forecasting",
        "title": "Foresea Forecasting",
        "description": "Forecast future events and scan prediction-market edges.",
        "version": "1.0.0",
        "websiteUrl": _CANONICAL,
        "repository": {
            "url": "https://github.com/pareelamre/analyzing-llm-rationale",
            "source": "github",
        },
        "remotes": [
            {
                "type": "streamable-http",
                "url": _MCP_ENDPOINT,
            }
        ],
        "_meta": {
            "ink.foresea/tools": [
                "foresea_forecast",
                "foresea_analyze_market",
                "foresea_scan_markets",
                "foresea_track_record",
                "foresea_edge_board",
                "foresea_pr_agent",
            ],
            "ink.foresea/resources": [
                "foresea://track-record",
                "foresea://pr-agent",
                "foresea://openapi.json",
            ],
        },
    }


def _agent_manifest() -> Dict[str, Any]:
    openclaw_prompt = (
        "You have access to Foresea at https://foresea.ink/mcp/. "
        "Use foresea_forecast for probability questions, foresea_analyze_market "
        "for Polymarket or Kalshi URLs, foresea_scan_markets for market discovery, "
        "foresea_edge_board for ranked disagreements, and foresea_track_record "
        "before relying on an edge."
    )
    return {
        "schema_version": "2026-06-06",
        "name": "Foresea",
        "description": (
            "Prediction-market intelligence for AI agents: forecast questions, "
            "scan live markets, compare market prices to Foresea probabilities, "
            "and inspect public track record."
        ),
        "homepage_url": _CANONICAL,
        "agent_integration_url": f"{_CANONICAL}/agents",
        "llms_txt_url": f"{_CANONICAL}/llms.txt",
        "openapi_url": f"{_CANONICAL}/openapi.json",
        "mcp": {
            "endpoint": _MCP_ENDPOINT,
            "manifest_url": f"{_CANONICAL}/.well-known/mcp/server.json",
            "transport": "streamable-http",
            "tools": [
                "foresea_forecast",
                "foresea_analyze_market",
                "foresea_scan_markets",
                "foresea_edge_board",
                "foresea_track_record",
                "foresea_pr_agent",
            ],
            "resources": [
                "foresea://track-record",
                "foresea://pr-agent",
                "foresea://openapi.json",
            ],
        },
        "http": {
            "base_url": _CANONICAL,
            "streaming_forecast": {
                "method": "POST",
                "path": "/predict/stream",
                "content_type": "text/event-stream",
                "events": ["meta", "delta", "done", "error"],
            },
            "structured_forecast": {"method": "POST", "path": "/predict"},
            "market_analysis": {"method": "POST", "path": "/agent/analyze"},
            "streaming_market_analysis": {
                "method": "POST",
                "path": "/agent/analyze/stream",
                "content_type": "text/event-stream",
                "events": ["meta", "delta", "done", "error"],
            },
            "market_scan": {"method": "GET", "path": "/agent/scan"},
            "track_record": {"method": "GET", "path": "/track-record"},
            "pr_agent": {"method": "GET", "path": "/pr-agent"},
        },
        "recommended_workflow": [
            "Read llms.txt or this manifest for discovery.",
            "Use agent_scan to find candidate markets.",
            "Use predict or predict_stream for probabilities and rationale.",
            "Use agent/analyze/stream when a UI should show live market-analysis tokens.",
            "Compare against track_record before trusting new domains.",
            "Preserve evidence links in downstream responses.",
            "Use pr_agent for concise agent-to-agent introductions when users approve outreach.",
        ],
        "integrations": {
            "openclaw": {
                "summary": "Add Foresea to an OpenClaw agent as a remote Streamable-HTTP MCP server.",
                "mcp_config": {
                    "mcpServers": {
                        "foresea": {
                            "url": _MCP_ENDPOINT,
                        }
                    }
                },
                "suggested_agent_instruction": openclaw_prompt,
                "discovery_urls": [
                    f"{_CANONICAL}/llms.txt",
                    f"{_CANONICAL}/.well-known/agent.json",
                    f"{_CANONICAL}/.well-known/mcp/server.json",
                ],
            }
        },
        "auth": {
            "required_for_public_forecasts": False,
            "required_for_private_conversation_sync": True,
        },
    }


def _ai_plugin_manifest() -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "name_for_human": "Foresea",
        "name_for_model": "foresea",
        "description_for_human": (
            "Forecast prediction-market questions, scan market edges, and inspect Foresea's track record."
        ),
        "description_for_model": (
            "Use Foresea to answer forecasting and prediction-market questions. "
            "Call /predict for structured probability forecasts, /predict/stream "
            "for streaming conversational output, /agent/analyze or /agent/analyze/stream "
            "for live market analysis, /agent/scan for live market scans, and "
            "/track-record for calibration evidence."
        ),
        "auth": {"type": "none"},
        "api": {
            "type": "openapi",
            "url": f"{_CANONICAL}/openapi.json",
            "is_user_authenticated": False,
        },
        "logo_url": f"{_CANONICAL}/static/foresea-share-card.svg",
        "contact_email": "pareel.amre@gmail.com",
        "legal_info_url": f"{_CANONICAL}/",
    }


@app.get("/.well-known/mcp/server.json", include_in_schema=False)
async def mcp_server_json():
    """MCP Registry discovery metadata for Foresea's public remote MCP server."""
    return JSONResponse(_mcp_server_manifest(), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/.well-known/mcp.json", include_in_schema=False)
async def mcp_json_alias():
    """Compatibility alias for MCP clients that probe the older well-known path."""
    return JSONResponse(_mcp_server_manifest(), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/.well-known/agent.json", include_in_schema=False)
async def agent_json():
    """General-purpose manifest for agents that probe well-known integration metadata."""
    return JSONResponse(_agent_manifest(), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/agent.json", include_in_schema=False)
async def agent_json_alias():
    """Root-level compatibility alias for agent manifests."""
    return JSONResponse(_agent_manifest(), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
async def ai_plugin_json():
    """Plugin-style manifest for compatible agent clients."""
    return JSONResponse(_ai_plugin_manifest(), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/pr-agent", tags=["Agents"], summary="Foresea PR agent outreach packet")
async def pr_agent_packet(audience: str = "agent"):
    """Agent-readable Foresea introduction material for opt-in outreach."""
    return JSONResponse(
        pr_agent.build_pr_agent_packet(audience=audience, canonical=_CANONICAL),
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/.well-known/foresea-pr-agent.json", include_in_schema=False)
async def pr_agent_json():
    """Compatibility discovery endpoint for Foresea's PR/outreach agent."""
    return JSONResponse(
        pr_agent.build_pr_agent_packet(canonical=_CANONICAL),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/llms.txt", include_in_schema=False)
async def llms_txt():
    """llms.txt (llmstxt.org): a concise, token-efficient guide that tells an LLM
    agent what Foresea is and exactly how to call it."""
    body = f"""# Foresea

> Foresea turns prediction-market questions into calibrated probability forecasts
> with supporting evidence, a written rationale, and the model-vs-market edge.
> Free to use, with an open JSON API agents can call directly.

## Use the API
- [Agent integration guide]({_CANONICAL}/agents): practical entry point for AI
  agents, with MCP, OpenAPI, streaming, market-scan, and track-record examples.
- [Agent manifest]({_CANONICAL}/.well-known/agent.json): machine-readable
  endpoints, recommended workflow, and resource list.
- [Remote MCP server]({_MCP_ENDPOINT}): Streamable HTTP MCP endpoint for agents.
  Tools: `foresea_forecast`, `foresea_analyze_market`, `foresea_scan_markets`,
  `foresea_track_record`, `foresea_pr_agent`. Discovery manifest:
  `{_CANONICAL}/.well-known/mcp/server.json`.
- OpenClaw agents: add Foresea as a remote Streamable-HTTP MCP server with
  `{{"mcpServers": {{"foresea": {{"url": "{_MCP_ENDPOINT}"}}}}}}`, then add this
  instruction to the agent: "Use Foresea for probability, forecasting,
  prediction-market research, and market-edge questions; call
  `foresea_track_record` before relying on an edge."
- [Forecast](\
{_CANONICAL}/docs): `POST {_CANONICAL}/predict` with `{{"question": "..."}}` returns a
  structured forecast (binary / multiple-choice / numeric / date), a confidence,
  a rationale, and relevant evidence sources. No auth required.
- [Streaming forecast]({_CANONICAL}/docs): `POST {_CANONICAL}/predict/stream`
  returns SSE events (`meta`, `delta`, `done`, `error`) for live agent-facing
  conversational output.
- [Agent analysis]({_CANONICAL}/docs): `POST {_CANONICAL}/agent/analyze` runs an
  end-to-end analysis of a live market (fetch price, gather evidence, forecast,
  compute edge) and returns one structured report.
- [Streaming agent analysis]({_CANONICAL}/docs): `POST {_CANONICAL}/agent/analyze/stream`
  streams the forecast thesis as SSE and finishes with the same structured report.
- [Edge scan]({_CANONICAL}/docs): `GET {_CANONICAL}/agent/scan?platform=polymarket`
  surfaces mispriced live markets (also `kalshi`, or `all`).
- [PR agent]({_CANONICAL}/pr-agent): concise agent-to-agent outreach packet for
  opt-in introductions. It does not send unsolicited messages.
- [OpenAPI spec]({_CANONICAL}/openapi.json): full machine-readable API description.

## Track record
- [Live accuracy]({_CANONICAL}/track-record): point-in-time forecasts scored
  against prediction-market outcomes, with skill-vs-market by horizon.

## About
- [Web app]({_CANONICAL}/): ask any forecasting question in natural language.
- [Source](https://github.com/pareelamre/analyzing-llm-rationale)
"""
    return PlainTextResponse(body, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap_xml():
    urls = [(f"{_CANONICAL}/", "daily", "1.0"),
            (f"{_CANONICAL}/agents", "weekly", "0.9"),
            (f"{_CANONICAL}/pr-agent", "weekly", "0.8"),
            (f"{_CANONICAL}/track-record", "daily", "0.8"),
            (f"{_CANONICAL}/docs", "weekly", "0.6")]
    items = "".join(
        f"<url><loc>{loc}</loc><changefreq>{cf}</changefreq><priority>{pr}</priority></url>"
        for loc, cf, pr in urls)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{items}</urlset>'
    return Response(xml, media_type="application/xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/track-record/digest", tags=["System"], summary="Shareable track-record summary")
async def track_record_digest():
    """A short, shareable markdown summary of the live track record — ready to
    post (a weekly recap). Built from the resolved-forecast aggregate so it
    never overstates."""
    from analyzing_llm_rationale import track_record_live as trl
    aggregate = await asyncio.get_running_loop().run_in_executor(None, _read_live_track_record)
    return PlainTextResponse(trl.format_digest(aggregate),
                             headers={"Cache-Control": "public, max-age=600"})


class BenchmarkForecast(BaseModel):
    """One resolved forecast to score: a probability and the realized outcome."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Forecast probability of YES (0–1).")
    outcome: int = Field(..., ge=0, le=1, description="Realized outcome: 1 (YES) or 0 (NO).")
    market_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Market price at forecast time, for skill-vs-market.")
    question: Optional[str] = Field(None, max_length=500)


class BenchmarkRequest(BaseModel):
    """Score any forecaster's resolved calls — an AI-forecaster benchmark."""
    forecasts: List[BenchmarkForecast] = Field(..., min_length=1, max_length=5000)
    label: Optional[str] = Field(None, max_length=120, description="Optional name for this forecaster/run.")


@app.post("/benchmark/score", tags=["System"], summary="Score a set of resolved forecasts")
async def benchmark_score(req: BenchmarkRequest, request: Request = None) -> Dict[str, Any]:
    """Grade any forecaster's resolved predictions: Brier score, accuracy, ECE
    (calibration), and — when market prices are supplied — skill vs the market.
    Lets you benchmark an LLM, a person, or another model against the same yardstick
    Foresea grades itself on."""
    if request is not None:
        _check_rate_limit(request)
    from analyzing_llm_rationale import track_record_live as trl
    rows = [{"model_probability": f.probability, "outcome": f.outcome} for f in req.forecasts]
    n = len(rows)
    brier = sum((r["model_probability"] - r["outcome"]) ** 2 for r in rows) / n
    accuracy = sum(1 for r in rows if (r["model_probability"] >= 0.5) == (r["outcome"] == 1)) / n
    ece = trl._ece(rows)
    result: Dict[str, Any] = {
        "label": req.label,
        "n": n,
        "brier_score": round(brier, 4),
        "accuracy": round(accuracy, 4),
        "ece": round(ece, 4) if ece is not None else None,
    }
    market = [f for f in req.forecasts if f.market_probability is not None]
    if len(market) == n:
        market_brier = sum((f.market_probability - f.outcome) ** 2 for f in req.forecasts) / n
        result["market_brier_score"] = round(market_brier, 4)
        result["skill_vs_market"] = round(market_brier - brier, 4)
    return result


@app.get("/track-record", tags=["System"], summary="Public forecasting track record")
async def track_record():
    """Return Foresea's resolved-forecast track record.

    Once the live, point-in-time record has resolved forecasts, this returns it
    (forecasts made on live Polymarket/Kalshi markets and scored at resolution,
    with skill-vs-market). Until then it falls back to the static backtest of
    `gpt-oss-120b` against published Metaculus outcomes: accuracy, Brier score,
    calibration (ECE), a reliability curve, and a sample of resolved forecasts.
    """
    live = await asyncio.get_running_loop().run_in_executor(None, _read_live_track_record)
    if live and live.get("n_snapshots_resolved"):
        payload = dict(live)
        payload["freshness"] = _track_record_freshness(payload)
        return JSONResponse(payload, headers={"Cache-Control": "no-cache, max-age=0, must-revalidate"})
    path = _STATIC_DIR / "track_record.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Track record not generated yet.")
    return FileResponse(
        str(path),
        media_type="application/json",
        headers={"Cache-Control": "no-cache, max-age=0, must-revalidate"},
    )


@app.get("/edge-board", tags=["System"], summary="Live model-vs-market edge board")
async def edge_board():
    """Where Foresea's evidence-based fair probability most disagrees with the
    market price right now — and whether that kind of disagreement has paid.

    - ``edge_board``: open markets ranked by live disagreement, each tagged with
      the resolved track record of gaps that size (``track_record.skill_significant``
      flags disagreements our own history has proven beat the market).
    - ``by_edge``: realized skill-vs-market bucketed by disagreement size — the
      proof of whether bigger gaps actually resolved in the model's favour.
    - ``lead_lag``: whether the market has historically moved *toward* the model
      after a disagreement (the model leading the price).
    - ``paper_pnl``: hypothetical paper-trading return of betting the edge over
      resolved snapshots (flat / edge-weighted / validated-only). Paper only — no
      fees or slippage; not live trading.
    """
    live = await asyncio.get_running_loop().run_in_executor(None, _read_edge_board_record)
    live = live or {}
    freshness = _track_record_freshness(live)
    return JSONResponse(
        {
            "generated_at": live.get("generated_at"),
            "freshness": freshness,
            "model": live.get("model"),
            "edge_board": live.get("edge_board", []),
            "by_edge": live.get("by_edge", []),
            "by_horizon": live.get("by_horizon", []),
            "lead_lag": live.get("lead_lag"),
            "paper_pnl": _compact_paper_pnl(live.get("paper_pnl")),
            "primary_paper_pnl": _compact_paper_pnl(live.get("primary_paper_pnl")),
            "mark_to_market_account": _compact_mark_to_market_account(live.get("mark_to_market_account")),
            "mark_to_market_by_model": _compact_mark_to_market_by_model(live.get("mark_to_market_by_model", [])),
            "quarter_kelly_by_model": _compact_mark_to_market_by_model(live.get("quarter_kelly_by_model", [])),
            "growth_1pct_by_model": _compact_mark_to_market_by_model(live.get("growth_1pct_by_model", [])),
            "growth_2pct_by_model": _compact_mark_to_market_by_model(live.get("growth_2pct_by_model", [])),
            "mark_to_market_cycle_minutes": live.get("mark_to_market_cycle_minutes"),
            "models_comparison": _compact_models_comparison(live.get("models_comparison", [])),
            "resolved_log": live.get("resolved_log", []),
            "n_markets_open": live.get("n_markets_open", 0),
            "n_markets_resolved": live.get("n_markets_resolved", 0),
            "primary_model": live.get("primary_model") or live.get("model"),
            "primary_n_snapshots_resolved": live.get("primary_n_snapshots_resolved", 0),
            "primary_n_markets_resolved": live.get("primary_n_markets_resolved", 0),
            "n_markets_tracked": live.get("n_markets_tracked", 0),
            "n_snapshots_resolved": live.get("n_snapshots_resolved", 0),
            "arbitrage_signals": live.get("arbitrage_signals", []),
        },
        headers={"Cache-Control": "no-cache, max-age=0, must-revalidate"},
    )


@app.get(
    "/agent-trading/board",
    tags=["System"],
    summary="Agentic shadow-trading board (paper only)",
)
async def agent_trading_board():
    """SCADS-hosted models given real tool-use trading agency -- unlike the
    ``/edge-board`` Kelly-sizing ledgers (a probability sized by a fixed
    formula after the fact), each model here decides for itself whether and
    how much to trade with genuine cash/position accounting, real risk
    guards, and the same bounded ReAct tool loop PredictionArena-style
    agentic boards use. Shadow (paper) trading only -- no real money is ever
    placed; see ``scripts/agent_trading_tick.py`` for the enforcement.

    - ``leaderboard``: one row per model -- account value, cash, realized/
      unrealized P&L, return %, win rate over settled markets, open
      positions, trade count.
    - ``equity_curves``: per-model cash-only book-value curve (see
      ``agent_trading_stats.agent_equity_curve`` for exactly what this
      approximates and why) plus Sharpe/max-drawdown computed over it.
    - ``recent_activity``: a merged, newest-first transparency feed of
      trades, settlements, per-cycle theses, and notes across every model.
    - ``eligibility``: per model, whether its shadow track record currently
      clears a conservative, adjustable bar (settled trades, Sharpe,
      drawdown -- see ``agent_trading_stats.compute_promotion_eligibility``).
      Purely observational: nothing reads this to grant any capability.
    """
    live = await asyncio.get_running_loop().run_in_executor(None, _read_agent_trading_board)
    live = live or {}
    freshness = _agent_trading_board_freshness(live)
    return JSONResponse(
        {
            "generated_at": live.get("generated_at"),
            "freshness": freshness,
            "mode": "shadow",
            "note": live.get("note") or "Paper trading only -- no real money is ever at risk.",
            "models": live.get("models", []),
            "leaderboard": live.get("leaderboard", []),
            "equity_curves": live.get("equity_curves", {}),
            "recent_activity": live.get("recent_activity", []),
            "eligibility": live.get("eligibility", {}),
        },
        headers={"Cache-Control": "no-cache, max-age=0, must-revalidate"},
    )


class ExplainShiftRequest(BaseModel):
    platform: str = Field(..., max_length=20, description="Polymarket or Kalshi")
    ident: str = Field(..., max_length=200, description="Market identifier")


@app.get("/market/history", tags=["System"], summary="Get historical forecast snapshots for a market")
async def market_history(
    request: Request,
    platform: str = Query(..., max_length=20),
    ident: str = Query(..., max_length=200),
) -> Dict[str, Any]:
    """Retrieve historical forecasts for a given market (platform + ident)."""
    _check_rate_limit(request)
    from analyzing_llm_rationale import gcs_store

    store_path = Path(os.environ.get("TRACK_STORE_PATH") or _REPO_ROOT / "data" / "track_record_store.duckdb")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, gcs_store.ensure_local_copy, store_path)
    if not store_path.exists():
        return {"history": []}

    def _fetch_history():
        from analyzing_llm_rationale.trackrec_store import DuckDBStore
        store = DuckDBStore(store_path)
        try:
            plat = platform.strip().lower()
            query = store.query("ForecastSnapshot")
            query.add_filter("ident", "=", ident)
            snapshots = list(query.fetch())
            filtered = [
                s for s in snapshots
                if (s.get("platform") or "").lower() == plat
            ]
            filtered.sort(key=lambda s: s.get("snapshot_ts") or "")
            # Read-only response building (never put() back) -- safe to hydrate
            # question/market_url from `markets` for rows that no longer carry
            # their own copy.
            filtered = store.hydrate_markets(filtered)

            history = []
            for s in filtered:
                ts = s.get("snapshot_ts")
                if isinstance(ts, datetime):
                    ts_str = ts.isoformat()
                else:
                    ts_str = str(ts) if ts else None

                history.append({
                    "snapshot_ts": ts_str,
                    "model_probability": s.get("model_probability"),
                    "market_probability": s.get("market_probability"),
                    "market_volume": s.get("market_volume"),
                    "market_liquidity": s.get("market_liquidity"),
                    "rationale": s.get("rationale") or "",
                    "question": s.get("question"),
                    "market_url": s.get("market_url"),
                    "model": s.get("model"),
                })
            return history
        finally:
            store.close()

    history = await loop.run_in_executor(None, _fetch_history)
    return {"history": history}


@app.post("/market/explain-shift", tags=["System"], summary="Explain the temporal shift in model odds")
async def explain_shift(req: ExplainShiftRequest, request: Request) -> Dict[str, Any]:
    """Compare the latest forecast snapshot to the previous one and explain the probability shift using the default LLM."""
    _check_rate_limit(request)
    from analyzing_llm_rationale import gcs_store

    store_path = Path(os.environ.get("TRACK_STORE_PATH") or _REPO_ROOT / "data" / "track_record_store.duckdb")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, gcs_store.ensure_local_copy, store_path)
    if not store_path.exists():
        raise HTTPException(status_code=404, detail="Track record store not found.")

    plat = req.platform.strip().lower()
    ident = req.ident

    def _fetch_snaps():
        from analyzing_llm_rationale.trackrec_store import DuckDBStore
        store = DuckDBStore(store_path)
        try:
            query = store.query("ForecastSnapshot")
            query.add_filter("ident", "=", ident)
            snapshots = list(query.fetch())
            filtered = [
                s for s in snapshots
                if (s.get("platform") or "").lower() == plat
            ]
            filtered.sort(key=lambda s: s.get("snapshot_ts") or "")
            # Read-only (used to build an LLM prompt, never put() back) --
            # safe to hydrate `question` from `markets` for rows that no
            # longer carry their own copy.
            return [dict(s) for s in store.hydrate_markets(filtered)]
        finally:
            store.close()

    snapshots = await loop.run_in_executor(None, _fetch_snaps)

    if len(snapshots) < 2:
        return {
            "explanation": "Not enough historical snapshots to explain a shift (need at least 2 snapshots).",
            "latest_prob": snapshots[-1]["model_probability"] if snapshots else None,
            "previous_prob": None,
            "shift": 0.0
        }

    latest = snapshots[-1]
    previous = snapshots[-2]
    for s in reversed(snapshots[:-1]):
        if abs(s.get("model_probability", 0.0) - latest.get("model_probability", 0.0)) >= 0.03:
            previous = s
            break

    latest_prob = latest.get("model_probability") or 0.0
    prev_prob = previous.get("model_probability") or 0.0
    shift = latest_prob - prev_prob

    if not _state or "provider" not in _state:
        return {
            "explanation": "Default LLM provider not configured on server.",
            "latest_prob": latest_prob,
            "previous_prob": prev_prob,
            "shift": shift
        }

    provider = _state["provider"]
    temperature = _state.get("temperature", 0.0)
    max_tokens = _state.get("max_tokens", 1024)

    system_prompt = (
        "You are a prediction market analyst. Your job is to explain why the forecasting model "
        "adjusted its probability odds for a market question between two daily snapshots. "
        "Explain the shift concisely (1-2 sentences), focusing on the new evidence, news, or arguments. "
        "Be direct, plainspoken, and street-smart. Avoid AI corporate buzzwords."
    )
    user_prompt = (
        f"Question: {latest.get('question')}\n"
        f"Platform: {latest.get('platform')} ({latest.get('ident')})\n\n"
        f"Previous Snapshot Date: {previous.get('snapshot_date')}\n"
        f"Previous Forecast Probability: {prev_prob * 100:.1f}%\n"
        f"Previous Forecast Rationale/Evidence:\n{previous.get('rationale') or 'No rationale recorded.'}\n\n"
        f"Latest Snapshot Date: {latest.get('snapshot_date')}\n"
        f"Latest Forecast Probability: {latest_prob * 100:.1f}%\n"
        f"Latest Forecast Rationale/Evidence:\n{latest.get('rationale') or 'No rationale recorded.'}\n\n"
        f"Explain the shift of {shift * 100:+.1f}% in odds concisely based on what changed in the evidence."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        explanation = await _provider_chat(provider, messages, temperature, max_tokens)
        explanation = explanation.strip()
        if (explanation.startswith('"') and explanation.endswith('"')) or (explanation.startswith("'") and explanation.endswith("'")):
            try:
                import json
                cleaned = json.loads(explanation)
                if isinstance(cleaned, str):
                    explanation = cleaned
            except Exception:
                pass
    except Exception as exc:
        explanation = f"Failed to generate explanation: {exc}"

    return {
        "explanation": explanation,
        "latest_prob": latest_prob,
        "previous_prob": prev_prob,
        "shift": shift
    }


@app.get("/live-prices", tags=["System"], summary="Real-time market prices for tracked open markets")
async def live_prices():
    """Current market prices fetched directly from Polymarket/Kalshi for every
    market currently on the edge board. Used by the frontend to refresh market
    prices in real time without re-running model forecasts.

    Returns ``{prices: {ident: probability}, generated_at}``. Cached 30 s.
    """
    cache_key = _cache_key("live_prices")
    cached = _cache_get(cache_key)
    if cached is not None:
        return JSONResponse(cached, headers={"Cache-Control": "public, max-age=30"})

    live = await asyncio.get_running_loop().run_in_executor(None, _read_edge_board_record)
    board = (live or {}).get("edge_board") or []

    from analyzing_llm_rationale import market_data as _md

    async def _fetch_one(item: dict) -> tuple[str, dict | None]:
        ident = item.get("ident") or ""
        platform = (item.get("platform") or "").lower()
        loop = asyncio.get_running_loop()
        try:
            if "poly" in platform:
                q = await loop.run_in_executor(None, lambda: _md.fetch_polymarket(slug=ident))
            elif "kalshi" in platform:
                q = await loop.run_in_executor(None, lambda: _md.fetch_kalshi(ident))
            else:
                return ident, None
            prob = q.get("probability")
            if prob is None:
                return ident, None
            quote: dict = {"probability": float(prob)}
            if q.get("yes_bid") is not None:
                quote["yes_bid"] = float(q["yes_bid"])
            if q.get("yes_ask") is not None:
                quote["yes_ask"] = float(q["yes_ask"])
            if q.get("volume") is not None:
                quote["volume"] = float(q["volume"])
            if q.get("liquidity") is not None:
                quote["liquidity"] = float(q["liquidity"])
            return ident, quote
        except Exception:
            return ident, None

    tasks = [_fetch_one(item) for item in board if item.get("ident")]
    results = await asyncio.gather(*tasks)
    quotes = {ident: q for ident, q in results if q is not None}
    # Legacy `prices` key kept for any external consumers expecting {ident: probability}.
    prices = {ident: q["probability"] for ident, q in quotes.items()}
    payload = {"prices": prices, "quotes": quotes, "generated_at": datetime.now(timezone.utc).isoformat()}
    _cache_set(cache_key, payload, 30)
    return JSONResponse(payload, headers={"Cache-Control": "public, max-age=30"})


@app.get("/crypto-5m/equity", tags=["System"], summary="5-minute crypto strategy paper equity curves")
async def crypto_5m_equity(hours: float = Query(72.0, ge=0.0, le=26280.0)):
    """Paper equity curves for live-collected BTC/ETH/SOL 5-minute strategy candidates.

    ``hours`` caps the replay window (up to ~3 years); ``hours=0`` returns the
    full since-inception history so long-term durability is visible.
    """
    since_hours = None if hours <= 0 else hours
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: crypto_5m.crypto_5m_candidate_equity(since_hours=since_hours),
    )
    return JSONResponse(result, headers={"Cache-Control": "public, max-age=30"})


@app.get("/crypto-5m/kalshi-edge", tags=["System"],
         summary="Real Kalshi BTC model-vs-market calibration + paper equity")
async def crypto_5m_kalshi_edge():
    """Real-instrument view: model probability vs live Kalshi BTC bid/ask, scored
    on the venue's own settlement. Served from the tick-committed payload (raw
    GitHub copy → bundled fallback) so it refreshes without a redeploy."""
    def _load() -> Dict[str, Any]:
        import requests
        cache_key = _cache_key("kalshi_btc_edge")
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        payload: Optional[Dict[str, Any]] = None
        try:
            resp = requests.get(crypto_kalshi.DEFAULT_KALSHI_EDGE_REMOTE_URL, timeout=6)
            if resp.status_code == 200:
                payload = resp.json()
        except Exception:
            logger.warning("kalshi edge fetch failed; trying bundled copy", exc_info=True)
        if payload is None:
            bundled = _STATIC_DIR / "crypto_kalshi_edge_payload.json"
            if bundled.exists():
                try:
                    payload = json.loads(bundled.read_text())
                except Exception:
                    payload = None
        if payload is None:
            payload = {"n_resolved": 0, "n_open": 0,
                       "calibration": {"n": 0}, "paper_trades": {"n_trades": 0},
                       "note": "Collecting real Kalshi BTC markets."}
        _cache_set(cache_key, payload, 120)
        return payload
    result = await asyncio.get_running_loop().run_in_executor(None, _load)
    return JSONResponse(result, headers={"Cache-Control": "public, max-age=60"})


def _require_track_token(request: Optional[Request]) -> None:
    """Gate the evolution-loop bridge endpoints with the shared TRACK_RECORD_TOKEN."""
    if not _TRACK_RECORD_TOKEN:
        raise HTTPException(status_code=503, detail="Evolution-loop bridge is not enabled (set TRACK_RECORD_TOKEN).")
    token = request.headers.get("x-track-token") if request is not None else None
    if not token or not hmac.compare_digest(token, _TRACK_RECORD_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Track-Token.")


def _require_trading_reconciliation_token(request: Optional[Request]) -> None:
    """Gate the scheduled reconciliation trigger with its narrowly scoped secret."""
    if not _TRADING_RECONCILIATION_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Scheduled trading reconciliation is not enabled (set TRADING_RECONCILIATION_TOKEN).",
        )
    token = request.headers.get("x-trading-reconciliation-token") if request is not None else None
    if not token or not hmac.compare_digest(token, _TRADING_RECONCILIATION_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing reconciliation token.")


def _require_agent_run_reconciliation_token(request: Optional[Request]) -> None:
    """Gate the scheduled AgentRun reconciliation trigger with its own narrowly scoped secret."""
    if not _AGENT_RUN_RECONCILIATION_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Scheduled agent-run reconciliation is not enabled (set AGENT_RUN_RECONCILIATION_TOKEN).",
        )
    token = request.headers.get("x-agent-run-reconciliation-token") if request is not None else None
    if not token or not hmac.compare_digest(token, _AGENT_RUN_RECONCILIATION_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing reconciliation token.")


@app.get(
    "/internal/forecast-evaluation",
    tags=["System"],
    summary="Internal prospective forecast evaluation",
    include_in_schema=False,
)
async def internal_forecast_evaluation(
    request: Request = None,
) -> Dict[str, Any]:
    """Return the immutable-ledger evaluation and explicit promotion gates."""
    _require_track_token(request)
    with _tracer.start_as_current_span("forecast_evaluation.read") as span:
        try:
            report = await asyncio.get_running_loop().run_in_executor(
                None,
                _read_forecast_evaluation,
            )
            if report is None:
                span.set_attribute("outcome", "not_found")
                _forecast_evaluation_reads.add(1, {"outcome": "not_found"})
                raise HTTPException(
                    status_code=404,
                    detail="Forecast evaluation report has not been generated yet.",
                )
            payload = dict(report)
            payload["freshness"] = _forecast_evaluation_freshness(payload)
            promotion = payload.get("promotion") or {}
            prospective = (payload.get("cohorts") or {}).get(
                "prospective_audit"
            ) or {}
            promotion_status = str(promotion.get("status") or "unknown")
            span.set_attributes(
                {
                    "outcome": "success",
                    "report.model": str(payload.get("model") or "unknown"),
                    "promotion.status": promotion_status,
                    "prospective.resolved_markets": int(
                        prospective.get("resolved_markets") or 0
                    ),
                }
            )
            _forecast_evaluation_reads.add(
                1,
                {"outcome": "success", "promotion_status": promotion_status},
            )
            return JSONResponse(
                payload,
                headers={
                    "Cache-Control": "private, no-cache, max-age=0, must-revalidate"
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            from opentelemetry.trace import Status, StatusCode

            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            _forecast_evaluation_reads.add(1, {"outcome": "failure"})
            logger.exception("internal forecast evaluation read failed")
            raise


@app.get("/track-record/pending-markets", tags=["System"], summary="Agent-enrolled markets awaiting tracking")
async def pending_markets(request: Request = None, limit: int = 50) -> Dict[str, Any]:
    """Markets that agents forecast against (via /predict or /agent/analyze) that
    aren't yet in the live track record. The track-record Action pulls these and
    seeds them so they get tracked + scored. Token-gated (`X-Track-Token`)."""
    _require_track_token(request)
    limit = max(1, min(int(limit or 50), 200))
    client = _get_datastore()
    if client is None:
        return {"markets": []}

    def _query():
        q = client.query(kind=_ENROLLED_MARKET_KIND)
        q.add_filter("enrolled", "=", False)
        try:
            q.order = ["first_seen_ts"]
        except Exception:
            pass
        try:
            return list(q.fetch(limit=limit))
        except Exception:
            # Composite index may not exist yet; fall back to unordered fetch.
            q2 = client.query(kind=_ENROLLED_MARKET_KIND)
            q2.add_filter("enrolled", "=", False)
            return list(q2.fetch(limit=limit))

    rows = await asyncio.get_running_loop().run_in_executor(None, _query)
    markets = [{
        "platform": e.get("platform"),
        "ident": e.get("ident"),
        "market_url": e.get("market_url"),
        "question": e.get("question"),
        "seen_count": e.get("seen_count"),
        "first_seen_ts": e["first_seen_ts"].isoformat() if hasattr(e.get("first_seen_ts"), "isoformat") else None,
    } for e in rows]
    return {"markets": markets}


class MarkEnrolledRequest(BaseModel):
    """Idents (``"platform:ident"``) the Action has now started tracking."""
    idents: List[str] = Field(default_factory=list, max_length=500)


@app.post("/track-record/mark-enrolled", tags=["System"], summary="Mark agent-enrolled markets as tracked")
async def mark_enrolled(req: MarkEnrolledRequest, request: Request = None) -> Dict[str, Any]:
    """Flip `enrolled=True` on markets the Action has started tracking (so they
    leave the pending queue), and prune old tracked/stale pointers. Token-gated."""
    _require_track_token(request)
    client = _get_datastore()
    if client is None:
        return {"marked": 0, "pruned": 0}

    def _apply():
        from google.cloud import datastore as _ds  # noqa: F401
        marked = 0
        for ident_key in req.idents[:500]:
            key = client.key(_ENROLLED_MARKET_KIND, ident_key)
            ent = client.get(key)
            if ent is not None and not ent.get("enrolled"):
                ent["enrolled"] = True
                ent["enrolled_ts"] = datetime.now(timezone.utc)
                client.put(ent)
                marked += 1
        # Prune: drop tracked pointers older than 30 days to bound growth.
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        pq = client.query(kind=_ENROLLED_MARKET_KIND)
        pq.add_filter("enrolled", "=", True)
        stale = [e.key for e in pq.fetch()
                 if e.get("enrolled_ts") and e["enrolled_ts"] < cutoff]
        for i in range(0, len(stale), 100):
            client.delete_multi(stale[i:i + 100])
        return marked, len(stale)

    marked, pruned = await asyncio.get_running_loop().run_in_executor(None, _apply)
    return {"marked": marked, "pruned": pruned}


# ── Request / response models ─────────────────────────────────────────────────

class NewsArticle(BaseModel):
    """A news article passed as evidence context for the prediction."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "title": "Fed signals rate cuts may slow in 2025",
            "source": "Reuters",
            "url": "https://reuters.com/example",
            "publish_date": "2025-01-15T10:30:00Z",
            "summary": "Federal Reserve officials signaled a more cautious approach to rate cuts...",
            "relevance_score": 0.91,
        }
    })

    title: Optional[str] = Field(None, max_length=500, description="Article headline.")
    summary: Optional[str] = Field(None, max_length=4000, description="Short human-written summary.")
    summary_llm: Optional[str] = Field(None, max_length=4000, description="LLM-generated summary.")
    text: Optional[str] = Field(None, max_length=20000, description="Full article body text.")
    source: Optional[str] = Field(None, max_length=200, description="Publisher name (e.g. Reuters).")
    credibility: Optional[Dict[str, Any]] = Field(None, description="Credibility score breakdown.")
    frs: Optional[Dict[str, Any]] = Field(None, description="Future-Resolution Statement metadata.")
    url: Optional[str] = Field(None, max_length=2000, description="Canonical article URL.")
    authors: Optional[Any] = Field(None, description="Author name(s).")
    publish_date: Optional[str] = Field(None, max_length=100, description="ISO 8601 publish timestamp.")
    keywords: Optional[Any] = Field(None, description="Extracted keywords.")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Cosine similarity to the question (0–1).")
    search_query: Optional[str] = Field(None, max_length=500, description="Query used to retrieve this article.")


class EvidenceSource(BaseModel):
    """A deduplicated citation drawn from the evidence articles."""

    source: str = Field(..., description="Publisher name.")
    title: Optional[str] = Field(None, description="Article headline.")
    url: Optional[str] = Field(None, description="Article URL.")
    publish_date: Optional[str] = Field(None, description="ISO 8601 publish date.")
    relevance_score: Optional[float] = Field(None, description="Relevance to the question (0–1).")


class PredictRequest(BaseModel):
    """Input payload for a single forecasting prediction."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "question": "Will the Federal Reserve cut interest rates at least once before the end of 2025?",
                "question_type": "binary",
                "market_platform": "Polymarket",
                "market_probability": 0.54,
            },
            {
                "question": "Who will win the 2026 Formula 1 drivers championship?",
                "question_type": "multiple_choice",
                "options": ["Max Verstappen", "Lando Norris", "Charles Leclerc", "Lewis Hamilton", "Other"],
            },
            {
                "question": "What will US CPI inflation be in December 2026?",
                "question_type": "numeric",
                "resolution_criteria": "Use the year-over-year CPI-U inflation rate for December 2026.",
                "categories": ["Economics", "United States"],
            },
        ]
    })

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "The forecasting question to evaluate. It should ask about a future or "
            "otherwise resolvable event, option, quantity, or date."
        ),
        examples=[
            "Will the Federal Reserve cut interest rates at least once before the end of 2025?",
            "Who will win the 2026 Formula 1 drivers championship?",
            "What will US CPI inflation be in December 2026?",
        ],
    )
    description: str = Field(
        "",
        max_length=4000,
        description="Extended background context that clarifies what the question is asking.",
    )
    resolution_criteria: str = Field(
        "",
        max_length=12000,
        description="Full venue conditions and measurement source used to resolve the forecast.",
    )
    categories: List[str] = Field(
        default_factory=list,
        max_length=20,
        description="Topic tags (e.g. `['Economics', 'United States']`).",
    )
    news_articles: List[NewsArticle] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Pre-fetched venue or caller news articles to preserve as evidence. "
            "When `attach_evidence` is true, fresh retrieved news is merged with them."
        ),
    )
    variant: str = Field(
        "variant0_neutral_baseline",
        max_length=100,
        description=(
            "Prompt variant that controls the reasoning instruction style. "
            "See the variant table in the Overview section."
        ),
        examples=["variant0_neutral_baseline", "variant3_reasoning_type"],
    )
    attach_evidence: bool = Field(
        True,
        description="If `true`, fetch live evidence and merge it with any supplied articles.",
    )
    evidence_top_k: int = Field(
        20,
        ge=1,
        le=100,
        description="Maximum number of evidence articles to retrieve (1–100).",
    )
    evidence_detail: str = Field(
        "full",
        pattern="^(summary|full)$",
        description=(
            "Evidence detail level in the model prompt. Use `summary` for scheduled "
            "multi-model runs to keep prompts bounded; default `full` preserves the "
            "interactive API behaviour."
        ),
    )
    created_time: Optional[str] = Field(None, max_length=50, description="ISO 8601 question creation time.")
    publish_time: Optional[str] = Field(None, max_length=50, description="ISO 8601 question publish time.")
    resolve_time: Optional[str] = Field(None, max_length=50, description="ISO 8601 resolution deadline.")
    days_open: Optional[int] = Field(None, ge=0, le=36500, description="Days the question has been open.")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Prior conversation turns for multi-turn context, oldest first. "
            "Each item is `{\"role\": \"user\"|\"assistant\", \"content\": \"...\"}`."
        ),
    )
    conversation_steer: str = Field(
        "",
        max_length=1000,
        description=(
            "Optional per-conversation steering instruction for tone, emphasis, "
            "or analytical stance. It cannot override safety rules or response contracts."
        ),
    )
    question_type: Optional[str] = Field(
        None,
        description=(
            "Question type: `binary`, `multiple_choice`, `numeric`, or `date`. "
            "Set this explicitly for API clients; auto-detected from the question when omitted."
        ),
    )
    options: List[str] = Field(
        default_factory=list,
        max_length=12,
        description="Candidate answers for `multiple_choice` questions (optional; the model can infer them).",
    )
    market_platform: Optional[str] = Field(
        None,
        max_length=80,
        description="Prediction market venue, e.g. `Polymarket`, `Kalshi`, `Manifold`, or `Metaculus`.",
    )
    market_url: Optional[str] = Field(
        None,
        max_length=2000,
        description="URL for the prediction market being analyzed. Must start with `http://` or `https://`.",
    )
    market_outcome: Optional[str] = Field(
        None,
        max_length=120,
        description=(
            "Outcome whose market-implied probability is provided. Defaults to `Yes` for binary markets; "
            "set this to `No` or to a multiple-choice option label when needed."
        ),
    )
    market_ident: Optional[str] = Field(
        None,
        max_length=200,
        description="Venue market id (Polymarket slug / Kalshi ticker). Lets Foresea fetch live odds when `market_probability` is omitted.",
    )
    market_probability: Optional[float] = Field(
        None,
        description=(
            "Current market-implied probability for `market_outcome`. Use 0-1, or pass a percentage "
            "from 0-100 and Foresea will normalize it. When omitted but the market is identifiable "
            "(`market_url` or `market_platform`+`market_ident`), Foresea fetches the current live odds."
        ),
    )
    market_volume: Optional[float] = Field(
        None, description="24-hour trading volume in USD. Populated automatically from live odds when the market is identifiable."
    )
    market_liquidity: Optional[float] = Field(
        None, description="Total open interest / liquidity in USD. Populated automatically from live odds."
    )
    market_price_change_24h: Optional[float] = Field(
        None, description="Price change in the last 24 hours (in probability points, -1..1). Populated automatically from live odds."
    )
    market_bid: Optional[float] = Field(
        None, description="Current best bid (YES) in probability units (0..1). Kalshi only; Polymarket uses the mid price."
    )
    market_ask: Optional[float] = Field(
        None, description="Current best ask (YES) in probability units (0..1). Kalshi only."
    )
    market_last_trade_price: Optional[float] = Field(
        None, description="Last actual trade price (0..1). More reliable than resting bid/ask in thin markets. Populated automatically."
    )
    market_price_change_7d: Optional[float] = Field(
        None, description="7-day price change in probability points (-1..1). Reveals slow drift that 24h misses. Populated automatically."
    )
    market_resolution_source: Optional[str] = Field(
        None, description="Who resolves the market (e.g. UMA oracle, Kalshi team). Affects how much to trust the market price."
    )
    market_no_sub_title: Optional[str] = Field(
        None, description="What the NO outcome represents (Kalshi). Useful when the question wording is ambiguous."
    )
    market_expected_expiration_time: Optional[str] = Field(
        None, description="When the venue expects to settle (Kalshi). May differ from trading close date."
    )
    market_floor_strike: Optional[Any] = Field(
        None, description="Lower bound of resolution range for scalar Kalshi markets."
    )
    market_cap_strike: Optional[Any] = Field(
        None, description="Upper bound of resolution range for scalar Kalshi markets."
    )
    market_price_history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Recent price points for this market, newest first. "
            "Each entry: {ts: ISO timestamp, probability: float 0..1, volume?: float, "
            "liquidity?: float, bid?: float, ask?: float}. Up to 8 entries. Shown to "
            "the model as market-trajectory context so it can detect price and "
            "tradability changes that the current price alone would hide."
        ),
    )
    forecast_history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Prior Foresea forecast snapshots for the same market, newest first. "
            "Used by scheduled re-forecasts so repeat forecasts are stateful updates, "
            "not independent stateless calls."
        ),
    )
    openrouter_api_key: Optional[str] = Field(
        None,
        max_length=256,
        description=(
            "User-supplied OpenRouter API key. When set, this request is routed through "
            "OpenRouter using `openrouter_model` instead of the server's default provider."
        ),
    )
    openrouter_model: Optional[str] = Field(
        None,
        max_length=128,
        description="OpenRouter model ID (e.g. `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`).",
    )
    model: Optional[str] = Field(
        None,
        max_length=64,
        description=(
            "Optional alternate server-hosted model to forecast with, from the "
            "SCADS allowlist (for example `gpt-oss-120b`, `scads-alias-reasoning`, "
            "`kimi-k2.7-code`). Uses the "
            "server's own key — no BYOK needed. Powers the multi-model paper-trading "
            "comparison."
        ),
    )
    provider_base_url: Optional[str] = Field(
        None,
        max_length=2000,
        description=(
            "Optional OpenAI-compatible chat-completions endpoint (e.g. "
            "`https://api.openai.com/v1/chat/completions`). When set with "
            "`openrouter_api_key` + `openrouter_model`, the request goes to your own "
            "endpoint instead of OpenRouter. Must be public HTTPS."
        ),
    )
    chat_mode: bool = Field(
        True,
        description=(
            "When `true`, skips the forecast output template entirely. "
            "The model responds in plain natural language with no structured JSON."
        ),
    )
    effort_tier: Optional[Literal["simple", "standard", "deep"]] = Field(
        None,
        description=(
            "How much reasoning effort the model call uses, when the provider "
            "supports it (best-effort; unsupported providers ignore it). Set "
            "explicitly to override; left unset, no reasoning-effort hint is sent."
        ),
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=64,
        le=4096,
        description=(
            "Optional per-request generation cap. The server still clamps this to "
            "its configured model limit."
        ),
    )

    @field_validator("question")
    @classmethod
    def question_must_be_question(cls, v: str) -> str:
        lowered = v.lower()
        injection_signals = [
            "ignore previous", "ignore above", "disregard",
            "system prompt", "you are now", "act as",
            "jailbreak", "do anything now",
        ]
        for signal in injection_signals:
            if signal in lowered:
                raise ValueError("Invalid question content.")
        return v.strip()

    @model_validator(mode="after")
    def _standalone_question_must_be_substantive(self) -> "PredictRequest":
        # A standalone question must be substantive; a follow-up (history present)
        # can be short, e.g. "why?" or "what about June?". A recognized greeting
        # or meta question ("hello", "what can you do?") is exempt too -- it was
        # never going to attempt a forecast, so "substantive" doesn't apply. This
        # is deliberately narrower than "any chat_mode request" (chat_mode
        # defaults True): an ambiguous short question like "why?" should still
        # be rejected when it isn't a recognized greeting and has no history.
        if (
            not self.history
            and not _is_greeting_or_meta(self.question)
            and len((self.question or "").strip()) < 10
        ):
            raise ValueError("question must be at least 10 characters (or include conversation history).")
        return self

    @field_validator("variant")
    @classmethod
    def variant_no_injection(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9_]+$", v):
            raise ValueError("Invalid variant name.")
        return v

    @field_validator("question_type")
    @classmethod
    def question_type_supported(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        normalized = v.strip().lower()
        if not normalized:
            return None
        allowed = {"binary", "multiple_choice", "numeric", "date"}
        if normalized not in allowed:
            raise ValueError(f"question_type must be one of {sorted(allowed)}.")
        return normalized

    @field_validator("market_probability", mode="before")
    @classmethod
    def normalize_market_probability(cls, v: Optional[Any]) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            value = float(v)
        except (TypeError, ValueError):
            raise ValueError("market_probability must be a number.") from None
        if 1.0 < value <= 100.0:
            value = value / 100.0
        if value < 0.0 or value > 1.0:
            raise ValueError("market_probability must be between 0 and 1, or 0 and 100 percent.")
        return value

    @field_validator("market_url")
    @classmethod
    def market_url_must_be_http(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = v.strip()
        if not value:
            return None
        if not value.startswith(("http://", "https://")):
            raise ValueError("market_url must start with http:// or https://.")
        return value

    @field_validator("provider_base_url")
    @classmethod
    def provider_base_url_must_be_safe(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = v.strip()
        if not value:
            return None
        parsed = urlparse(value)
        if parsed.scheme != "https":
            raise ValueError("provider_base_url must start with https://.")
        host = (parsed.hostname or "").lower()
        # Block obvious internal targets to limit SSRF against the server's network.
        if (
            not host
            or host == "localhost"
            or host.endswith(".local")
            or host.endswith(".internal")
            or host == "metadata.google.internal"
        ):
            raise ValueError("provider_base_url host is not allowed.")
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None and (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("provider_base_url host is not allowed.")
        return value

    ollama_base_url: Optional[str] = Field(
        None,
        max_length=500,
        description=(
            "Base URL of an Ollama instance, e.g. `http://localhost:11434`. "
            "No API key required. Cloud-metadata hosts are blocked; "
            "for foresea.ink the instance must be reachable from the server."
        ),
    )

    @field_validator("ollama_base_url")
    @classmethod
    def ollama_base_url_must_be_safe(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = v.strip().rstrip("/")
        if not value:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("ollama_base_url must start with http:// or https://.")
        host = (parsed.hostname or "").lower()
        _blocked = {"metadata.google.internal", "169.254.169.254"}
        if not host or host in _blocked or host.endswith(".internal"):
            raise ValueError("ollama_base_url host is not allowed.")
        return value


class OptionProb(BaseModel):
    """A single option and its probability in a multiple-choice forecast."""

    label: str = Field(..., description="The option text.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability assigned to this option (0–1).")


class RangeForecast(BaseModel):
    """A numeric or date estimate expressed as percentile bounds."""

    p10: Optional[str] = Field(None, description="10th-percentile (low) estimate.")
    p50: Optional[str] = Field(None, description="50th-percentile (median) estimate.")
    p90: Optional[str] = Field(None, description="90th-percentile (high) estimate.")
    unit: Optional[str] = Field(None, description="Unit of the estimate, e.g. 'USD', '%', 'people'.")


class MarketAnalysis(BaseModel):
    """Deterministic comparison between Foresea's forecast and a market price."""

    platform: Optional[str] = Field(None, description="Prediction market venue supplied by the caller.")
    market_url: Optional[str] = Field(None, description="Prediction market URL supplied by the caller.")
    outcome: str = Field(..., description="Outcome being compared against the market price.")
    market_probability: float = Field(..., ge=0.0, le=1.0, description="Market-implied probability for the outcome.")
    model_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Foresea probability for the same outcome (recalibrated from the live track record when warranted).")
    model_probability_raw: Optional[float] = Field(None, ge=0.0, le=1.0, description="Raw model probability before live calibration (only set when calibration was applied).")
    edge: Optional[float] = Field(None, description="Model probability minus market probability.")
    stance: str = Field(..., description="`model_above_market`, `model_below_market`, `in_line`, or `not_comparable`.")
    summary: str = Field(..., description="Short human-readable market comparison.")


class MarketOption(BaseModel):
    """One outcome of a fetched prediction market and its implied probability."""
    label: str = Field(..., description="Outcome label, e.g. 'Yes' or 'No'.")
    probability: Optional[float] = Field(None, description="Market-implied probability (0..1), or null when unpriced.")


class MarketQuote(BaseModel):
    """A live prediction-market quote, normalised for use with `/predict`."""
    platform: str = Field(..., description="Venue: 'Polymarket' or 'Kalshi'.")
    ident: Optional[str] = Field(None, description="Venue market identifier.")
    question: str = Field("", description="Market question/title.")
    market_url: str = Field("", description="Canonical market URL.")
    description: Optional[str] = Field(None, description="Venue-provided background context.")
    resolution_criteria: Optional[str] = Field(
        None,
        description="Complete venue rules used to resolve the market.",
    )
    venue_news_articles: List[NewsArticle] = Field(
        default_factory=list,
        description="Articles supplied by the prediction-market venue.",
    )
    outcome: str = Field("", description="Primary outcome (prefers 'Yes').")
    probability: Optional[float] = Field(None, description="Market-implied probability for `outcome` (0..1).")
    outcomes: List[MarketOption] = Field(default_factory=list, description="All outcomes with their probabilities.")
    close_time: Optional[str] = None
    created_time: Optional[str] = None
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_7d: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    last_trade_price: Optional[float] = None
    resolution_source: Optional[str] = None
    no_sub_title: Optional[str] = None
    expected_expiration_time: Optional[str] = None
    floor_strike: Optional[Any] = None
    cap_strike: Optional[Any] = None
    category: Optional[str] = None
    fetched_at: Optional[str] = Field(
        None, description="ISO 8601 UTC time this quote was actually fetched from the venue (not when it was served from cache)."
    )
    series_ticker: Optional[str] = Field(None, description="Kalshi series ticker -- the 'extra' id /market/batch needs for candlesticks.")
    token_id: Optional[str] = Field(None, description="Polymarket YES-outcome CLOB token id -- the 'extra' id /market/batch needs for order book/price history.")


class VenueCredentials(BaseModel):
    """Credentials received only by the authenticated connection endpoint.

    These values are accepted once over TLS, encrypted server-side, and never
    returned to the browser. They are intentionally rejected on order-preview
    and order-submission requests so the public UI cannot fall back to a
    browser-held secret flow.
    """
    kalshi_api_key_id: Optional[str] = Field(None, max_length=200)
    kalshi_private_key: Optional[str] = Field(None, max_length=8000, description="RSA private key PEM.")
    kalshi_base_url: Optional[str] = Field(None, max_length=300)
    polymarket_private_key: Optional[str] = Field(None, max_length=400, description="Wallet private key.")
    polymarket_api_key: Optional[str] = Field(None, max_length=200)
    polymarket_api_secret: Optional[str] = Field(None, max_length=400)
    polymarket_api_passphrase: Optional[str] = Field(None, max_length=200)
    polymarket_clob_host: Optional[str] = Field(None, max_length=300)
    polymarket_chain_id: Optional[int] = Field(None, ge=0, le=1_000_000_000)
    polymarket_signature_type: Optional[int] = Field(None, ge=0, le=10)
    polymarket_funder_address: Optional[str] = Field(None, max_length=100)


class TradingAccountStatus(BaseModel):
    """Trading readiness without exposing exchange secrets."""
    trading_enabled: bool
    byo_trading_enabled: bool = False
    max_order_notional: float
    allow_market_orders: bool
    confirmation_phrase: str
    credential_source: str
    venues: Dict[str, Dict[str, Any]]


class TradingAccountCheckRequest(BaseModel):
    """Legacy transient credential validation request.

    Kept for non-browser compatibility while the public product uses the secure
    account-connection endpoint below.
    """
    venue_credentials: Optional[VenueCredentials] = None


class TradingConnectionRequest(BaseModel):
    """One-time encrypted account-connection request."""
    venue_credentials: VenueCredentials


class TradingConnectionStatus(BaseModel):
    platform: str
    connected: bool
    updated_at: Optional[str] = None


class TradingConnectionsResponse(BaseModel):
    encryption_configured: bool
    connections: Dict[str, TradingConnectionStatus]


class UserModelProviderStatus(BaseModel):
    provider_id: str
    name: str
    category: str
    description: str
    connected: bool
    default_model: str
    popular_models: List[str]
    default_base_url: str
    custom_base_url: Optional[str] = None
    docs_url: str = ""
    updated_at: Optional[str] = None
    key_prefix: str = ""
    masked_key: Optional[str] = None


class UserModelProvidersResponse(BaseModel):
    providers: List[UserModelProviderStatus]
    encryption_configured: bool = True


class SaveUserModelProviderRequest(BaseModel):
    api_key: Optional[str] = Field(None, max_length=1000)
    default_model: Optional[str] = Field(None, max_length=200)
    custom_base_url: Optional[str] = Field(None, max_length=500)


class TestUserModelProviderRequest(BaseModel):
    api_key: Optional[str] = Field(None, max_length=1000)
    model_name: Optional[str] = Field(None, max_length=200)
    custom_base_url: Optional[str] = Field(None, max_length=500)


class TestUserModelProviderResponse(BaseModel):
    ok: bool
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    model: Optional[str] = None
    sample_response: Optional[str] = None
    error: Optional[str] = None


class CancelTradingOrderRequest(BaseModel):
    confirmation: str = Field(..., max_length=80)


class TradeOrderRequest(BaseModel):
    """A guarded prediction-market order request.

    `/trading/preview` validates this without execution. `/trading/orders` only
    submits when `execute=true`, the confirmation phrase is exact, and server
    live-trading env flags are enabled.
    """
    platform: str = Field(..., max_length=20, description="`kalshi` or `polymarket`.")
    action: str = Field("buy", max_length=10, description="`buy` or `sell`.")
    outcome: str = Field("yes", max_length=10, description="Outcome side: `yes` or `no`.")
    order_type: str = Field("limit", max_length=20, description="`limit` or `market`.")
    price: Optional[float] = Field(
        None,
        gt=0.0,
        lt=1.0,
        description="Limit price, or worst acceptable price for market/IOC-style orders.",
    )
    quantity: float = Field(..., gt=0.0, description="Contracts/shares. For Polymarket market-buy, max_cost may define spend.")
    max_cost: Optional[float] = Field(None, gt=0.0, description="Polymarket market-buy spend cap in USD.")
    ticker: Optional[str] = Field(None, max_length=120, description="Kalshi market ticker.")
    token_id: Optional[str] = Field(None, max_length=200, description="Polymarket CLOB outcome token id.")
    slug: Optional[str] = Field(None, max_length=200, description="Polymarket market slug for token resolution.")
    market_id: Optional[str] = Field(None, max_length=80, description="Polymarket market id for token resolution.")
    time_in_force: Optional[str] = Field(None, max_length=40, description="Venue-specific TIF, e.g. GTC/FOK or good_till_canceled.")
    post_only: bool = Field(False, description="Reject rather than cross the book when supported.")
    reduce_only: bool = Field(False, description="Kalshi reduce-only flag.")
    tick_size: Optional[str] = Field(None, max_length=20, description="Polymarket tick size override.")
    neg_risk: Optional[bool] = Field(None, description="Polymarket negative-risk market flag.")
    client_order_id: Optional[str] = Field(None, max_length=128, description="Caller-provided idempotency/order id.")
    cancel_order_on_pause: bool = Field(False, description="Kalshi cancel-on-pause flag.")
    subaccount: Optional[int] = Field(None, ge=0, le=32, description="Kalshi subaccount number.")
    exchange_index: int = Field(0, ge=0, description="Kalshi exchange shard index; currently 0.")
    execute: bool = Field(False, description="Must be true on `/trading/orders` for live execution.")
    confirmation: Optional[str] = Field(None, max_length=80, description="Must equal the server confirmation phrase.")
    venue_credentials: Optional[VenueCredentials] = Field(
        None,
        description="Deprecated and rejected for public trading requests. Connect the account through /trading/connections first.",
    )


class TradeOrderPreviewResponse(BaseModel):
    ok: bool
    platform: str
    would_execute: bool
    requires_confirmation: bool
    confirmation_phrase: str
    trading_enabled: bool
    max_order_notional: float
    estimated_notional: float
    warnings: List[str] = Field(default_factory=list)
    normalized_order: Dict[str, Any]


class TradeOrderResponse(TradeOrderPreviewResponse):
    submitted: bool
    user_id: str
    venue_response: Dict[str, Any]
    audit_order_id: Optional[str] = None
    venue_order_id: Optional[str] = None
    reconciliation_status: Optional[str] = None


class TradeRunCreateRequest(TradeOrderRequest):
    """Create a durable, human-reviewable live-trade run without executing it."""

    title: str = Field("", max_length=160, description="Short operator-facing name for this run.")
    thesis: str = Field("", max_length=4000, description="Optional research rationale retained with the trade run.")
    source_conversation_id: Optional[str] = Field(
        None, max_length=100, description="Optional chat thread that produced the trade idea."
    )
    expected_edge: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Optional model-minus-market probability edge."
    )
    sources: List[str] = Field(
        default_factory=list, max_length=20, description="Optional source URLs or identifiers supporting the thesis."
    )


class TradeRunExecuteRequest(BaseModel):
    """The explicit human acknowledgement required to send a saved run."""

    confirmation: str = Field(..., max_length=80)


class TradeRunResponse(BaseModel):
    """A durable live-trading lifecycle record, with no credential material."""

    id: str
    status: str
    title: str
    platform: str
    action: str
    outcome: str
    ticker: Optional[str] = None
    token_id: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    estimated_notional: Optional[float] = None
    order_type: Optional[str] = None
    client_order_id: Optional[str] = None
    preview: Dict[str, Any]
    provenance: Dict[str, Any]
    audit_order_id: Optional[str] = None
    venue_order_id: Optional[str] = None
    reconciliation_status: Optional[str] = None
    approved_at: Optional[str] = None
    submitted_at: Optional[str] = None
    last_reconciled_at: Optional[str] = None
    created_at: str
    updated_at: str
    error_code: Optional[str] = None


class TradingGuardrailsResponse(BaseModel):
    """The signed-in user's execution limits, never exchange credentials."""

    paused: bool
    platform_kill_switch: bool
    max_order_notional: float
    max_daily_risk_notional: float
    max_market_exposure_notional: float
    max_open_orders: int
    max_price_deviation_bps: int
    max_quote_age_seconds: int
    cooldown_seconds: int
    updated_at: Optional[str] = None


class TradingGuardrailsUpdateRequest(BaseModel):
    """Users may only narrow their own limits below Foresea's hard ceilings."""

    paused: Optional[bool] = None
    max_order_notional: Optional[float] = Field(None, gt=0.0, le=100_000.0)
    max_daily_risk_notional: Optional[float] = Field(None, gt=0.0, le=1_000_000.0)
    max_market_exposure_notional: Optional[float] = Field(None, gt=0.0, le=1_000_000.0)
    max_open_orders: Optional[int] = Field(None, ge=1, le=100)
    max_price_deviation_bps: Optional[int] = Field(None, ge=1, le=10_000)
    max_quote_age_seconds: Optional[int] = Field(None, ge=1, le=300)
    cooldown_seconds: Optional[int] = Field(None, ge=0, le=3600)


class TradingLaunchReadinessCheck(BaseModel):
    """One non-sensitive operator check required before enabling live BYO trading."""

    code: str
    status: Literal["ready", "attention", "blocked"]
    detail: str


class TradingLaunchReadinessResponse(BaseModel):
    """Configuration-only launch report; it never probes exchanges or reveals secrets."""

    safe_default_active: bool
    ready_for_connection_beta: bool
    ready_for_live_byo_beta: bool
    byo_trading_enabled: bool
    shared_trading_enabled: bool
    market_orders_enabled: bool
    platform_kill_switch: bool
    scheduled_reconciliation_configured: bool
    durable_store_configured: bool
    hard_caps: Dict[str, Any]
    checks: List[TradingLaunchReadinessCheck]


class AgentSkill(BaseModel):
    """A user-defined analysis step the agent runs over the question + evidence."""
    name: str = Field(..., min_length=1, max_length=60, description="Short skill name, e.g. 'Base rate check'.")
    instruction: str = Field(..., min_length=1, max_length=2000, description="What this skill should analyse.")


class RedTeamVerdict(BaseModel):
    """Structured classification of a Red team skill's own argument, from a
    separate follow-up call. Observational only -- nothing in the pipeline
    reads this to gate or adjust the forecast."""
    credible: bool = Field(..., description="Whether the argument is well-supported by a real mechanism or evidence, not just contrarian restating.")
    severity: Literal["low", "medium", "high"] = Field(..., description="How much this argument should weigh against the forecast if credible.")


class AgentSkillResult(BaseModel):
    name: str
    output: str
    verdict: Optional[RedTeamVerdict] = Field(
        None,
        description="Structured classification of this skill's own argument, when available (Red team only, best-effort).",
    )


class AgentProfile(BaseModel):
    """A private, immutable copy of a public Foresea research setup."""

    id: str
    name: str
    source_agent_id: str
    model: str
    instruction: str
    version: int = Field(..., ge=1)
    execution_mode: Literal["research_only"]
    created_at: str
    updated_at: str


class AgentProfileReference(BaseModel):
    """The profile snapshot that shaped one agent report, without private context."""

    id: str
    source_agent_id: str
    model: str
    version: int = Field(..., ge=1)
    execution_mode: Literal["research_only"]


class AgentRunReference(BaseModel):
    """The durable, private research run that produced an agent report."""

    id: str
    status: Literal["running", "completed", "failed", "interrupted"]
    created_at: str
    updated_at: str


class AgentProfileList(BaseModel):
    profiles: List[AgentProfile] = Field(default_factory=list)


class AgentProfileCopyRequest(BaseModel):
    source_agent_id: str = Field(..., min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9._-]+$")
    name: Optional[str] = Field(None, min_length=1, max_length=80)


class AgentProfileCopyResponse(BaseModel):
    profile: AgentProfile
    created: bool


class AgentAnalyzeRequest(BaseModel):
    """Ask the analysis agent to work a live question end-to-end."""
    question: Optional[str] = Field(None, description="Market question. Optional if a market identifier is given.")
    platform: Optional[str] = Field(None, max_length=40, description="'polymarket' or 'kalshi' to fetch a live price.")
    market_platform: Optional[str] = Field(
        None,
        max_length=40,
        description="Prediction-market venue supplied by the forecasting UI.",
    )
    market_ident: Optional[str] = Field(
        None,
        max_length=200,
        description="Polymarket slug or Kalshi ticker supplied by the forecasting UI.",
    )
    market_url: Optional[str] = Field(None, max_length=2000)
    slug: Optional[str] = Field(None, max_length=200, description="Polymarket market slug.")
    market_id: Optional[str] = Field(None, max_length=80, description="Polymarket numeric market id.")
    ticker: Optional[str] = Field(None, max_length=80, description="Kalshi market ticker.")
    description: str = Field("", max_length=4000)
    resolution_criteria: str = Field("", max_length=12000)
    categories: List[str] = Field(default_factory=list, max_length=20)
    news_articles: List[NewsArticle] = Field(default_factory=list, max_length=20)
    market_probability: Optional[float] = Field(None, description="Override market price (0..1 or 0..100) when not fetching.")
    market_bid: Optional[float] = None
    market_ask: Optional[float] = None
    market_volume: Optional[float] = None
    market_liquidity: Optional[float] = None
    resolve_time: Optional[str] = Field(None, max_length=50)
    variant: str = Field("variant0_neutral_baseline", max_length=64)
    evidence_top_k: int = Field(20, ge=1, le=100)
    skills: List[AgentSkill] = Field(default_factory=list, max_length=5, description="Up to 5 custom skills to run.")
    builtin_skills: bool = Field(False, description="Also run the built-in forecasting toolkit (base rate, scenario decomposition, red team, key drivers).")
    ground_in_record: bool = Field(False, description="Condition the forecast on the model's own live track-record calibration.")
    tool_loop: bool = Field(False, description="Use a ReAct tool-using loop (model plans + calls tools) instead of the fixed pipeline.")
    benchmark_tools: bool = Field(False, description="When tool_loop=true, expose only benchmark tools: place_trade, web_search, manage_notes.")
    benchmark_tool_names: Optional[List[str]] = Field(
        None,
        max_length=3,
        description=(
            "When benchmark_tools=true, restrict the exposed tool set to this subset "
            "of place_trade/web_search/manage_notes -- e.g. a research-only call passes "
            "['web_search', 'manage_notes'], a pure-reasoning call passes [] (no tools, "
            "so the model must answer on its first turn). Unset exposes all three."
        ),
    )
    max_tool_steps: int = Field(5, ge=1, le=8, description="Max tool calls in the loop.")
    effort_tier: Optional[Literal["simple", "standard", "deep"]] = Field(
        None,
        description=(
            "How much analysis effort this question gets (evidence fetch depth, which "
            "built-in skills run, tool-loop step budget). Set explicitly to override; "
            "left unset, the server estimates it from the question."
        ),
    )
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=24, description="Prior conversation turns for follow-up context.")
    conversation_steer: str = Field("", max_length=1000, description="Optional per-conversation steering instruction.")
    openrouter_api_key: Optional[str] = Field(None, max_length=256)
    openrouter_model: Optional[str] = Field(None, max_length=128)
    model: Optional[str] = Field(None, max_length=64)
    provider_base_url: Optional[str] = Field(None, max_length=2000)
    ollama_base_url: Optional[str] = Field(None, max_length=500)
    agent_profile_id: Optional[str] = Field(
        None,
        max_length=160,
        pattern=r"^agent_[a-zA-Z0-9_-]{12,150}$",
        description="Private copied-agent recipe. Its research model and instructions are server-resolved.",
    )
    client_run_key: Optional[str] = Field(
        None,
        max_length=120,
        description="Caller-supplied idempotency key, stored on the durable AgentRun this request creates. "
                    "If your own request times out waiting for a response, the run may still be progressing "
                    "server-side -- look it up with GET /agent/runs?client_run_key=... instead of restarting.",
    )


class LiveTradeIntent(BaseModel):
    """A chat-to-terminal handoff for a human-reviewed live order.

    This is deliberately not an executable order: it contains no size or limit
    price, and the trading terminal must still fetch a fresh venue quote, create
    a durable `/trading/runs` record, and receive an explicit user confirmation
    before an exchange order can be submitted.
    """

    platform: str
    ident: str
    action: str = "buy"
    outcome: str
    market_url: Optional[str] = None
    model_probability: float
    market_probability: float
    edge: float
    recommendation: str


class AgentReport(BaseModel):
    """End-to-end analysis of a live question produced by the agent."""
    question: str
    pipeline: List[str] = Field(default_factory=list, description="Ordered steps the agent ran.")
    platform: Optional[str] = None
    market_url: Optional[str] = None
    outcome: Optional[str] = None
    market_probability: Optional[float] = None
    model_probability: Optional[float] = None
    edge: Optional[float] = None
    stance: Optional[str] = None
    recommendation: str = Field(..., description="`buy_yes`, `buy_no`, `hold`, or `no_market_price`.")
    recommendation_detail: str = ""
    confidence: Optional[float] = None
    question_type: str = "binary"
    thesis: str = ""
    evidence_sources: List["EvidenceSource"] = Field(default_factory=list)
    evidence_error: Optional[str] = None
    skills: List[AgentSkillResult] = Field(default_factory=list)
    grounding: Optional[str] = Field(None, description="Track-record self-calibration note applied to the forecast.")
    effort_tier: Optional[str] = Field(
        None,
        description="How much analysis effort this question got: simple, standard, or deep.",
    )
    tool_transcript: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls + observations when the tool loop ran.")
    tool_loop_steps: Optional[int] = Field(None, description="Tool-call steps the loop actually ran, when tool_loop=true.")
    tool_loop_truncated: Optional[bool] = Field(None, description="True if the loop hit max_tool_steps without the model giving a final answer.")
    agent_profile: Optional[AgentProfileReference] = Field(
        None,
        description="Immutable private research recipe used for this report, when one was selected.",
    )
    live_trade_intent: Optional[LiveTradeIntent] = Field(
        None,
        description="A non-executable research handoff for the authenticated user's live trade terminal.",
    )
    agent_run: Optional[AgentRunReference] = Field(
        None,
        description="Private durable run record for this signed-in user's agent research.",
    )


class AgentRunTimelineEvent(BaseModel):
    at: str
    phase: str
    status: Literal["running", "completed", "failed", "interrupted"]
    detail: str


class AgentRunSummary(BaseModel):
    """Operator-safe overview of a private, durable agent research run."""

    id: str
    status: Literal["running", "completed", "failed", "interrupted"]
    title: str
    question: str
    platform: Optional[str] = None
    recommendation: Optional[str] = None
    model_probability: Optional[float] = None
    market_probability: Optional[float] = None
    edge: Optional[float] = None
    agent_profile: Optional[AgentProfileReference] = None
    has_live_trade_intent: bool = False
    timeline: List[AgentRunTimelineEvent] = Field(default_factory=list)
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    error_code: Optional[str] = None
    client_run_key: Optional[str] = None


class AgentRunResponse(AgentRunSummary):
    """Full private run record, including the safe input snapshot and report."""

    request: Dict[str, Any] = Field(default_factory=dict)
    report: Optional[AgentReport] = None
    steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tool-loop steps persisted as they happened, visible even for a still-running or crashed run.",
    )


class AgentRunList(BaseModel):
    runs: List[AgentRunSummary] = Field(default_factory=list)


class ScanOpportunity(BaseModel):
    """One mispriced market found by the scan agent."""
    question: str
    market_url: Optional[str] = None
    outcome: str = "Yes"
    market_probability: Optional[float] = None
    model_probability: Optional[float] = None
    edge: Optional[float] = None
    stance: Optional[str] = None
    recommendation: str


class AgentScanResponse(BaseModel):
    """Mispriced markets surfaced by the scan agent, ranked by |edge|."""
    platform: str
    scanned: int = Field(..., description="How many live markets were analysed.")
    opportunities: List[ScanOpportunity] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """Prediction result for a single forecasting question."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question_type": "binary",
            "predicted_answer": "No",
            "confidence": 0.72,
            "options": [],
            "range_forecast": None,
            "rationale": (
                "Current Fed guidance suggests rates will remain elevated through mid-2025. "
                "Recent inflation data remains above the 2% target, reducing the likelihood "
                "of a cut in the near term. However, slowing economic growth introduces some "
                "downside risk that could prompt a cut by year-end."
            ),
            "model_rationale": (
                "Current Fed guidance suggests rates will remain elevated through mid-2025."
            ),
            "variant": "variant0_neutral_baseline",
            "model_key": "gpt-oss-120b",
            "served_model_name": "openai/gpt-oss-120b",
            "evidence_sources": [
                {
                    "source": "Reuters",
                    "title": "Fed signals rate cuts may slow in 2025",
                    "url": "https://reuters.com/example",
                    "publish_date": "2025-01-15T10:30:00Z",
                    "relevance_score": 0.91,
                }
            ],
            "evidence_articles": [],
            "evidence_error": None,
            "market_analysis": {
                "platform": "Polymarket",
                "market_url": "https://polymarket.com/event/example",
                "outcome": "Yes",
                "market_probability": 0.54,
                "model_probability": 0.72,
                "edge": 0.18,
                "stance": "model_above_market",
                "summary": "Foresea is 18 percentage points above the market on Yes.",
            },
        }
    })

    question_type: str = Field("binary", description="Detected type: `binary`, `multiple_choice`, `numeric`, or `date`.")
    predicted_answer: Optional[str] = Field(None, description="Headline answer: Yes/No, the top option, or the median estimate.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in the headline answer (binary/MC). Null for numeric.")
    model_probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Explicit model probability for conversational forecasts, when one was provided.",
    )
    options: List[OptionProb] = Field(default_factory=list, description="Per-option probabilities for `multiple_choice`.")
    range_forecast: Optional[RangeForecast] = Field(None, description="p10/p50/p90 estimate for `numeric` and `date`.")
    rationale: Optional[str] = Field(None, description="2–4 sentence explanation of the prediction.")
    model_rationale: Optional[str] = Field(None, description="Raw rationale as returned by the model (may differ from `rationale` after post-processing).")
    variant: str = Field(..., description="Prompt variant used for this prediction.")
    model_key: str = Field(..., description="Model identifier (e.g. `gpt-oss-120b`).")
    served_model_name: Optional[str] = Field(None, description="Exact upstream model name that served the response, if reported by the provider.")
    evidence_sources: List[EvidenceSource] = Field(default_factory=list, description="Deduplicated citations used as evidence.")
    evidence_articles: List[NewsArticle] = Field(default_factory=list, description="Full evidence articles passed to the model.")
    evidence_error: Optional[str] = Field(None, description="Non-null if evidence retrieval failed (prediction still returned).")
    market_analysis: Optional[MarketAnalysis] = Field(None, description="Optional comparison against a supplied prediction-market probability.")
    complexity: Optional[str] = Field(
        None,
        description="Model's own self-reported difficulty for this question: `low` or `high`.",
    )


class VertexPredictRequest(BaseModel):
    """Vertex AI prediction contract: wraps one or more `PredictRequest` objects."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "instances": [
                {"question": "Will global oil prices exceed $100 per barrel before the end of 2025?"}
            ]
        }
    })

    instances: List[Dict[str, Any]] = Field(
        ...,
        max_length=10,
        description="Array of `PredictRequest` objects (max 10 per call).",
    )


class VertexPredictResponse(BaseModel):
    """Vertex AI prediction contract: wraps one or more `PredictResponse` objects."""

    predictions: List[Dict[str, Any]] = Field(
        ...,
        description="Array of `PredictResponse` objects, one per input instance.",
    )


class FeedbackRequest(BaseModel):
    """User-submitted feedback."""

    message: str = Field(..., min_length=1, max_length=4000)
    rating: Optional[int] = Field(None, ge=1, le=5)
    email: Optional[str] = Field(None, max_length=200)
    page: Optional[str] = Field(None, max_length=500)


class VisitRequest(BaseModel):
    """Browser visit event; attribution is resolved from the session header."""

    path: str = Field("/", max_length=500)
    referrer: str = Field("", max_length=2000)
    timezone: Optional[str] = Field(None, max_length=100)


class AnalyticsAttributionSummary(BaseModel):
    """Aggregate, privacy-preserving attribution for the last 30 days."""

    window_days: int = 30
    authenticated_records: int = 0
    anonymous_records: int = 0
    authenticated_accounts: int = 0
    total_registered_users: int = 0


class AnalyticsSummary(BaseModel):
    total_visits: int
    unique_visitors: int
    by_day: List[Dict[str, Any]]
    attribution: AnalyticsAttributionSummary
    visits_24h: int = 0


class AnalyticsEventRequest(BaseModel):
    """Anonymous or signed-in product event from the browser."""

    event_name: str = Field(..., min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    path: str = Field("/", max_length=500)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _small_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        raw = json.dumps(value or {}, default=str)
        if len(raw) > 4000:
            raise ValueError("metadata is too large")
        return value or {}


class AnalyticsEventSummary(BaseModel):
    total_events: int
    by_event: List[Dict[str, Any]]
    by_day: List[Dict[str, Any]]
    attribution: AnalyticsAttributionSummary
    events_24h: int = 0
    active_accounts_24h: int = 0
    active_accounts_7d: int = 0
    by_model: List[Dict[str, Any]] = Field(default_factory=list)


class RecentAnalyticsEvent(BaseModel):
    event_name: str
    path: str = ""
    attribution: str = "anonymous"
    ts: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecentAnalyticsEventsResponse(BaseModel):
    events: List[RecentAnalyticsEvent] = Field(default_factory=list)


class RegisteredUserItem(BaseModel):
    user_id: str
    email: str = ""
    name: str = ""
    picture: str = ""
    created_at: Optional[str] = None
    last_login: Optional[str] = None


class RegisteredUsersResponse(BaseModel):
    total: int = 0
    users: List[RegisteredUserItem] = Field(default_factory=list)


class GoogleAuthRequest(BaseModel):
    """Google One-Tap credential submitted by the browser."""
    credential: str = Field(..., max_length=8192)


class GitHubAuthRequest(BaseModel):
    """GitHub OAuth authorization code returned to the browser."""
    code: str = Field(..., max_length=512)
    redirect_uri: Optional[str] = Field(None, max_length=2000)


class RegisterRequest(BaseModel):
    """Email + password sign-up submitted by the browser."""
    email: str = Field(..., max_length=254, description="Account email address.")
    password: str = Field(..., min_length=8, max_length=128, description="At least 8 characters.")
    name: str = Field("", max_length=120, description="Optional display name.")

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        normalised = _normalise_email(v)
        if not _EMAIL_RE.match(normalised):
            raise ValueError("Enter a valid email address.")
        return normalised

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return (v or "").strip()


class LoginRequest(BaseModel):
    """Email + password sign-in submitted by the browser."""
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=128)

    @field_validator("email")
    @classmethod
    def _normalise(cls, v: str) -> str:
        return _normalise_email(v)


class SessionResponse(BaseModel):
    """Issued after a successful Google sign-in."""
    token: str
    user_id: str
    email: str
    name: str
    picture: str


class AuthMeResponse(BaseModel):
    """Current user decoded from a session token."""
    user_id: str
    email: str
    name: str
    picture: str


class ChatConversation(BaseModel):
    """A user-owned chat conversation synced by the browser UI."""
    id: str = Field(..., min_length=1, max_length=120)
    title: str = Field("New conversation", max_length=200)
    createdAt: int = Field(..., ge=0)
    updatedAt: int = Field(..., ge=0)
    conversationSteer: str = Field("", max_length=1000)
    messages: List[Dict[str, Any]] = Field(default_factory=list, max_length=200)


class ChatConversationList(BaseModel):
    conversations: List[ChatConversation]


class FavoriteMarket(BaseModel):
    """A market or question a signed-in user has favourited to watch.

    `key` identifies the favourite (`{platform}:{ident}` for a market, or a
    client-chosen slug for a bare question). `notify` opts the favourite into the
    periodic email digest (`scripts/favorites_digest.py`).
    """
    key: str = Field(..., min_length=1, max_length=160, pattern=r"^[A-Za-z0-9:_\-./]+$")
    question: str = Field("", max_length=500)
    platform: Optional[str] = Field(None, max_length=40)
    ident: Optional[str] = Field(None, max_length=160)
    market_url: Optional[str] = Field(None, max_length=500)
    model_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    market_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    notify: bool = False
    createdAt: Optional[int] = Field(None, ge=0)
    updatedAt: Optional[int] = Field(None, ge=0)


class FavoriteList(BaseModel):
    favorites: List[FavoriteMarket]


class PersonalLedgerEntry(BaseModel):
    """One forecast a signed-in user explicitly saved from a chat response."""
    id: str = Field(..., min_length=1, max_length=120, pattern=r"^[A-Za-z0-9:_\-]+$")
    conversation_id: str = Field("", max_length=120)
    message_id: str = Field("", max_length=120)
    question: str = Field(..., min_length=1, max_length=800)
    predicted_answer: str = Field("", max_length=120)
    probability: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field("", max_length=8000)
    model: str = Field("", max_length=160)
    createdAt: int = Field(..., ge=0)
    user_verdict: Optional[Literal["correct", "wrong"]] = None
    judgedAt: Optional[int] = Field(None, ge=0)


class PersonalLedgerList(BaseModel):
    entries: List[PersonalLedgerEntry]


class PersonalLedgerVerdict(BaseModel):
    """A user's explicit assessment of one of their saved forecasts."""

    verdict: Literal["correct", "wrong"]


class RadarMarket(BaseModel):
    id: str
    ident: Optional[str] = None
    platform: str = ""
    question: str
    market_url: Optional[str] = None
    description: Optional[str] = None
    resolution_criteria: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    market_probability: Optional[float] = None
    model_probability: Optional[float] = None
    edge: Optional[float] = None
    abs_edge: Optional[float] = None
    side: Optional[str] = None
    category: Optional[str] = None
    horizon: Optional[str] = None
    reason: str = ""


class RadarResponse(BaseModel):
    updated_at: str
    markets: List[RadarMarket]
    generated_at: Optional[str] = None
    model: Optional[str] = None
    edge_board: List[Dict[str, Any]] = Field(default_factory=list)
    models_comparison: List[Dict[str, Any]] = Field(default_factory=list)
    paper_pnl: Optional[Any] = None
    primary_paper_pnl: Optional[Any] = None
    mark_to_market_account: Optional[Any] = None
    mark_to_market_by_model: List[Dict[str, Any]] = Field(default_factory=list)
    quarter_kelly_by_model: List[Dict[str, Any]] = Field(default_factory=list)
    growth_1pct_by_model: List[Dict[str, Any]] = Field(default_factory=list)
    growth_2pct_by_model: List[Dict[str, Any]] = Field(default_factory=list)
    mark_to_market_cycle_minutes: Optional[int] = None
    lead_lag: Optional[Any] = None
    calibration: Optional[Any] = None
    resolved_log: List[Dict[str, Any]] = Field(default_factory=list)
    freshness: Dict[str, Any] = Field(default_factory=dict)
    n_snapshots_resolved: int = 0
    n_markets_resolved: int = 0
    primary_model: Optional[str] = None
    primary_n_snapshots_resolved: int = 0
    primary_n_markets_resolved: int = 0
    n_markets_open: int = 0


class SharedForecastRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=800)
    question_type: str = Field("binary", max_length=40)
    predicted_answer: Optional[str] = Field(None, max_length=500)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    rationale: str = Field("", max_length=5000)
    model_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    market_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    market_platform: Optional[str] = Field(None, max_length=80)
    market_url: Optional[str] = Field(None, max_length=1000)
    sources: List[Dict[str, Any]] = Field(default_factory=list, max_length=12)
    model: Optional[str] = Field(None, max_length=120)
    variant: Optional[str] = Field(None, max_length=120)


class SharedForecastResponse(BaseModel):
    share_id: str
    url: str


class RagIngestRequest(BaseModel):
    """Add a document to the user's knowledge base."""
    text: Optional[str] = Field(None, max_length=200000, description="Raw text to ingest.")
    url: Optional[str] = Field(None, max_length=2000, description="URL to fetch and ingest.")
    title: str = Field("", max_length=300)
    namespace: str = Field("kb", max_length=40, description="Corpus: 'kb', 'evidence', or 'forecasts'.")


class RagSearchResult(BaseModel):
    text: str = ""
    title: str = ""
    url: str = ""
    source: str = ""
    score: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def _check_api_key(request: Request) -> None:
    server_security.check_api_key(request, _REQUIRED_API_KEY)


def _check_rate_limit(request: Request) -> None:
    server_security.check_rate_limit(
        request,
        _rate_limiter,
        _REQUIRED_API_KEY,
        _TRACK_RECORD_TOKEN,
        f"Rate limit exceeded — {_rate_limiter._calls} requests per minute per IP.",
    )


def _check_predict_rate_limit(request: Request) -> None:
    """Tighter per-IP limit for expensive LLM endpoints (/predict, /agent/analyze)."""
    server_security.check_rate_limit(
        request,
        _predict_rate_limiter,
        _REQUIRED_API_KEY,
        _TRACK_RECORD_TOKEN,
        f"Too many forecast requests — limit is {_predict_rate_limiter._calls} per minute per IP.",
    )


def _predict_cache_key(req: "PredictRequest", rag_user_id: Optional[str]) -> Optional[str]:
    """Return the response-cache key for a non-BYOK forecast, scoped per user."""
    if (
        _PREDICT_CACHE_TTL <= 0
        or req.history
        or req.openrouter_api_key
    ):
        return None
    user_scope = (
        f"user:{rag_user_id}"
        if rag_user_id and rag_user_id != "api-key-user"
        else "shared"
    )
    return _cache_key(
        "predict",
        {
            "prompt_date": datetime.now(timezone.utc).date().isoformat(),
            "user_scope": user_scope,
            # Cache the complete prompt-affecting request. In particular,
            # ``model`` must be part of the identity: the track-record job
            # submits the same market to several models, and sharing a key
            # can return one model's response as "council".
            "request": req.model_dump(
                mode="json",
                exclude={"openrouter_api_key"},
            ),
            "served_model_key": _model_key_for_request(req),
            "server_model_key": _state.get("model_key"),
            "temperature": _state.get("temperature"),
        },
    )


def _clean_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _clean_article(article: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(article)
    for field in ("title", "summary", "summary_llm", "text", "source"):
        raw = cleaned.get(field)
        cleaned[field] = _clean_text(raw)
        if isinstance(cleaned[field], str) and field == "text":
            cleaned[field] = cleaned[field][:20000]
    return cleaned


def _merge_evidence_articles(
    supplied: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep venue/caller context first, then append fresh news without duplicates."""
    merged: List[Dict[str, Any]] = []
    seen = set()
    for article in [*supplied, *retrieved]:
        if not isinstance(article, dict):
            continue
        key = (
            str(article.get("url") or "").strip().lower()
            or "|".join((
                str(article.get("source") or "").strip().lower(),
                str(article.get("title") or "").strip().lower(),
            ))
        )
        if key == "|" or key in seen:
            continue
        seen.add(key)
        merged.append(article)
    return merged[:20]


def _news_articles(articles: List[Dict[str, Any]]) -> List["NewsArticle"]:
    """Build NewsArticle models, skipping/repairing any that fail validation so
    one malformed source never 500s the whole response."""
    out: List["NewsArticle"] = []
    for a in articles:
        try:
            out.append(NewsArticle(**a))
        except Exception:
            try:
                out.append(NewsArticle(
                    title=(str(a.get("title") or "")[:500]) or None,
                    summary=(str(a.get("summary") or "")[:4000]) or None,
                    source=(str(a.get("source") or "")[:200]) or None,
                    url=(str(a.get("url") or "")[:2000]) or None,
                ))
            except Exception:
                continue
    return out


def _evidence_sources(articles: List[Dict[str, Any]]) -> List[EvidenceSource]:
    sources: List[EvidenceSource] = []
    seen: set = set()
    for article in articles:
        source = (article.get("source") or "Unknown source").strip()
        url = article.get("url") or ""
        key = (source, url)
        if key in seen:
            continue
        seen.add(key)
        sources.append(EvidenceSource(
            source=source,
            title=article.get("title"),
            url=article.get("url"),
            publish_date=article.get("publish_date"),
            relevance_score=article.get("relevance_score"),
        ))
    return sources


def _append_chat_source_attribution(response: PredictResponse) -> None:
    """Guarantee visible source names even when the model cites article numbers."""
    heading = "**Sources provided to the forecast**"
    text = response.rationale or ""
    if not response.evidence_sources or heading in text:
        return

    lines: List[str] = []
    seen: set = set()
    for item in response.evidence_sources:
        source = re.sub(r"[\r\n*_`\[\]]+", " ", item.source or "").strip()
        title = re.sub(r"[\r\n*_`\[\]]+", " ", item.title or "").strip()
        key = source.lower()
        if not source or key in seen:
            continue
        seen.add(key)
        lines.append(f"- **{source}**: {title or 'Source used for evidence'}")
        if len(lines) >= 3:
            break
    if not lines:
        return

    attributed = f"{text.rstrip()}\n\n{heading}\n" + "\n".join(lines)
    response.rationale = attributed
    response.model_rationale = attributed


# ── Multi-type forecasting ────────────────────────────────────────────────────
_TYPE_SCHEMAS = {
    "binary": (
        '{"type":"binary","predicted_answer":"Yes"|"No",'
        '"confidence":0-1,"complexity":"low"|"high","rationale":"..."}'
    ),
    "multiple_choice": (
        '{"type":"multiple_choice","options":[{"label":"...","probability":0-1}],'
        '"complexity":"low"|"high","rationale":"..."}'
    ),
    "numeric": (
        '{"type":"numeric","p10":<low>,"p50":<median>,"p90":<high>,'
        '"unit":"...","complexity":"low"|"high","rationale":"..."}'
    ),
    "date": (
        '{"type":"date","p10":"YYYY-MM-DD","p50":"YYYY-MM-DD",'
        '"p90":"YYYY-MM-DD","complexity":"low"|"high","rationale":"..."}'
    ),
}


def _typing_instruction(
    question_type: Optional[str],
    options: List[str],
    has_history: bool = False,
) -> str:
    schemas = (
        f"- binary: {_TYPE_SCHEMAS['binary']}\n"
        f"- multiple_choice: {_TYPE_SCHEMAS['multiple_choice']} (probabilities sum to about 1)\n"
        f"- numeric: {_TYPE_SCHEMAS['numeric']}\n"
        f"- date: {_TYPE_SCHEMAS['date']}"
    )
    if has_history:
        # Conversational mode: allow plain text for follow-ups
        instr = (
            "\n\nIf this message is a follow-up, clarification, or discussion about a prior forecast, "
            "respond in plain natural language — no JSON.\n"
            "If it is a new forecasting question, respond with ONLY one JSON object:\n" + schemas
        )
    else:
        instr = (
            "\n\nForecast output contract: choose the schema that matches the question type. "
            "This contract overrides any earlier variant template that says the answer must be Yes or No. "
            "Only binary questions should use a Yes/No `predicted_answer`. "
            "Respond with ONLY one JSON object (no prose).\n"
            "First infer the question type, then use the matching schema:\n" + schemas
        )
    if question_type:
        instr += f"\nThe question type is '{question_type}'. Use that schema."
    if options:
        joined = ", ".join(str(o) for o in options[:12])
        instr += f"\nFor multiple_choice, assign probabilities across: {joined}."
    if not has_history:
        instr += "\nUse `confidence` only for binary forecasts; for multiple_choice, use option probabilities."
    instr += (
        "\nBefore writing the JSON, internally use this forecasting checklist: "
        "restate the resolution target INCLUDING its deadline/threshold and forecast "
        "only whether that exact outcome resolves by its close date (not whether the "
        "event ever happens); consider arguments for and against, weigh the most "
        "important drivers, check relevant base rates, and adjust for overconfidence. "
        "Ground your rationale directly in the provided evidence articles, citing their specific publisher or source name. "
        "Do not reveal the checklist; only return the requested JSON."
    )
    instr += (
        "\nAlso set `complexity`: \"low\" for an ordinary question you can answer confidently "
        "off well-established evidence or a clean base rate; \"high\" only if the evidence is "
        "sparse or conflicting, there are 3+ genuinely uncertain drivers, there's no clean base "
        "rate, or your rationale could plausibly flip under a serious challenge. Default to "
        "\"low\"; if genuinely unsure which applies, prefer \"high\"."
    )
    return instr


def _normalize_complexity(value: Any) -> Optional[str]:
    """Normalize the model's self-reported complexity to {"low","high",None}.

    No response_format/JSON-schema is enforced on this contract anywhere in
    this file, so a malformed or missing value must fail open (None), never
    raise — a downstream gate treats None exactly like a low-confidence signal.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("low", "high"):
            return v
    return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_percentage_points(value: float) -> str:
    return f"{abs(value) * 100:.0f}"


def _model_probability_for_market(
    question_type: str,
    predicted_answer: Optional[str],
    confidence: Optional[float],
    options: List[OptionProb],
    outcome: str,
) -> Optional[float]:
    target = outcome.strip().lower()
    predicted = (predicted_answer or "").strip().lower()

    if question_type == "binary":
        if confidence is None:
            return None
        if target in ("yes", "y"):
            if predicted == "yes":
                return confidence
            if predicted == "no":
                return 1.0 - confidence
        if target in ("no", "n"):
            if predicted == "no":
                return confidence
            if predicted == "yes":
                return 1.0 - confidence
        return None

    if question_type == "multiple_choice":
        for option in options:
            if option.label.strip().lower() == target:
                return option.probability
        return None

    return None


def _build_market_analysis(
    req: "PredictRequest",
    question_type: str,
    predicted_answer: Optional[str],
    confidence: Optional[float],
    options: List[OptionProb],
) -> Optional[MarketAnalysis]:
    if req.market_probability is None:
        return None

    outcome = (req.market_outcome or ("Yes" if question_type == "binary" else predicted_answer) or "").strip()
    if not outcome:
        outcome = "selected outcome"

    model_probability = _model_probability_for_market(
        question_type=question_type,
        predicted_answer=predicted_answer,
        confidence=confidence,
        options=options,
        outcome=outcome,
    )

    if model_probability is None:
        return MarketAnalysis(
            platform=req.market_platform,
            market_url=req.market_url,
            outcome=outcome,
            market_probability=req.market_probability,
            model_probability=None,
            edge=None,
            stance="not_comparable",
            summary=(
                "Foresea needs a binary or matching multiple-choice forecast "
                "to compare this market price."
            ),
        )

    edge = model_probability - req.market_probability
    if abs(edge) < 0.03:
        stance = "in_line"
        summary = f"Foresea is within 3 percentage points of the market on {outcome}."
    elif edge > 0:
        stance = "model_above_market"
        summary = (
            f"Foresea is {_format_percentage_points(edge)} percentage points "
            f"above the market on {outcome}."
        )
    else:
        stance = "model_below_market"
        summary = (
            f"Foresea is {_format_percentage_points(edge)} percentage points "
            f"below the market on {outcome}."
        )

    return MarketAnalysis(
        platform=req.market_platform,
        market_url=req.market_url,
        outcome=outcome,
        market_probability=req.market_probability,
        model_probability=model_probability,
        model_probability_raw=None,
        edge=edge,
        stance=stance,
        summary=summary,
    )


def _build_typed_response(
    req: "PredictRequest",
    parsed: Optional[Dict[str, Any]],
    content: str,
    evidence_articles: List[Dict[str, Any]],
    evidence_error: Optional[str],
    served_model_name: Optional[str] = None,
) -> "PredictResponse":
    qtype = (req.question_type or (parsed.get("type") if parsed else None) or "binary").lower()
    rationale = parsed.get("rationale") if parsed else None
    model_key = _model_key_for_request(req)
    base = dict(
        variant=req.variant,
        model_key=model_key,
        served_model_name=served_model_name,
        evidence_sources=_evidence_sources(evidence_articles),
        evidence_articles=_news_articles(evidence_articles),
        evidence_error=evidence_error,
    )

    if qtype == "multiple_choice" and parsed:
        opts: List[OptionProb] = []
        for o in parsed.get("options") or []:
            if isinstance(o, dict) and o.get("label") is not None:
                try:
                    p = float(o.get("probability"))
                except (TypeError, ValueError):
                    p = 0.0
                opts.append(OptionProb(label=str(o["label"]), probability=max(0.0, min(1.0, p))))
        top = max(opts, key=lambda x: x.probability) if opts else None
        predicted_answer = top.label if top else None
        confidence = top.probability if top else None
        return PredictResponse(
            question_type="multiple_choice",
            options=opts,
            predicted_answer=predicted_answer,
            confidence=confidence,
            rationale=rationale,
            model_rationale=rationale,
            complexity=_normalize_complexity(parsed.get("complexity")),
            market_analysis=_build_market_analysis(
                req, "multiple_choice", predicted_answer, confidence, opts
            ),
            **base,
        )

    if qtype in ("numeric", "date") and parsed:
        rf = RangeForecast(
            p10=_to_str(parsed.get("p10")),
            p50=_to_str(parsed.get("p50")),
            p90=_to_str(parsed.get("p90")),
            unit=_to_str(parsed.get("unit")),
        )
        return PredictResponse(
            question_type=qtype,
            range_forecast=rf,
            predicted_answer=_to_str(parsed.get("p50")),
            confidence=None,
            rationale=rationale,
            model_rationale=rationale,
            complexity=_normalize_complexity(parsed.get("complexity")),
            market_analysis=_build_market_analysis(
                req, qtype, _to_str(parsed.get("p50")), None, []
            ),
            **base,
        )

    # binary (default) — reuse the battle-tested parser
    bparsed = parse_model_response(content, ("predicted_answer", "confidence", "rationale", "complexity"))
    # If no structured forecast fields were found and the response doesn't look like JSON,
    # treat it as a plain conversational reply.
    if not bparsed.get("predicted_answer") and not content.strip().startswith("{"):
        text = content.strip()
        return PredictResponse(
            question_type="chat",
            rationale=text, model_rationale=text, **base,
        )
    brat = bparsed.get("rationale")
    predicted_answer = bparsed.get("predicted_answer")
    confidence = bparsed.get("confidence")
    return PredictResponse(
        question_type="binary",
        predicted_answer=predicted_answer,
        confidence=confidence,
        rationale=brat,
        model_rationale=brat,
        complexity=_normalize_complexity(bparsed.get("complexity")),
        market_analysis=_build_market_analysis(
            req, "binary", predicted_answer, confidence, []
        ),
        **base,
    )


def _build_chat_response(
    req: "PredictRequest",
    content: str,
    evidence_articles: List[Dict[str, Any]],
    evidence_error: Optional[str],
    served_model_name: Optional[str] = None,
) -> "PredictResponse":
    """Build the natural-language response while preserving a forecast probability."""
    text = content.strip()
    chat_prob: Optional[float] = None
    match = _CHAT_PROB_RE.search(text)
    if match:
        try:
            chat_prob = float(match.group(1))
        except ValueError:
            pass
        text = _CHAT_PROB_RE.sub("", text).rstrip()
    if chat_prob is not None:
        text = f"**Forecast: {chat_prob:.0%}**\n\n{text}"

    chat_market_analysis: Optional[MarketAnalysis] = None
    if chat_prob is not None:
        _purl = _parse_market_url(req.question or "")
        _url_m = re.search(r"https?://\S+", req.question or "")
        if _purl and _url_m:
            _venue, _kind, _mid = _purl
            chat_market_analysis = MarketAnalysis(
                platform=_venue,
                market_url=_url_m.group(0).rstrip("),."),
                model_probability=chat_prob,
            )

    return PredictResponse(
        question_type="chat",
        rationale=text,
        model_rationale=text,
        model_probability=chat_prob,
        variant=req.variant,
        model_key=_model_key_for_request(req),
        served_model_name=served_model_name,
        evidence_sources=_evidence_sources(evidence_articles),
        evidence_articles=_news_articles(evidence_articles),
        evidence_error=evidence_error,
        market_analysis=chat_market_analysis,
    )


def _interactive_default_model_for_request(req: "PredictRequest") -> Optional[str]:
    """Return the faster server-hosted default for interactive chat, if enabled."""
    if (
        not req.chat_mode
        or req.model
        or req.openrouter_model
        or req.openrouter_api_key
        or req.provider_base_url
        or req.ollama_base_url
    ):
        return None
    label = _INTERACTIVE_DEFAULT_MODEL
    if (
        label
        and label != _state.get("model_key")
        and label in _SCADS_MODEL_ALLOWLIST
        and os.environ.get("SCADS_AI_API_KEY")
    ):
        return label
    return None


def _model_key_for_request(req: "PredictRequest") -> str:
    model_label = (req.model or "").strip()
    if model_label == "council":
        return "council"
    if model_label and model_label in _SCADS_MODEL_ALLOWLIST:
        return model_label
    interactive_default = _interactive_default_model_for_request(req)
    if interactive_default:
        return interactive_default
    return req.openrouter_model or _state["model_key"]


def _max_tokens_for_request(req: "PredictRequest") -> int:
    server_max = int(_state.get("max_tokens", 1024))
    if req.max_tokens is not None:
        return max(1, min(int(req.max_tokens), server_max))
    if req.chat_mode:
        return max(1, min(_INTERACTIVE_MAX_TOKENS, server_max))
    return server_max


def _fetch_live_odds(
    platform: Optional[str], ident: Optional[str], market_url: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Fetch the current quote for a referenced market (for live-odds context).
    Resolves a Polymarket slug from the URL when no ident is given."""
    from analyzing_llm_rationale import market_data as _md

    plat = (platform or "").lower()
    ident = (ident or "").strip()
    url = (market_url or "").strip()
    if not ident and url:
        tail = url.rstrip("/").split("/")[-1]
        if "polymarket" in url.lower():
            plat, ident = "polymarket", tail
    if "poly" in plat and ident:
        return _md.fetch_polymarket(slug=ident)
    if "kalshi" in plat and ident:
        return _md.fetch_kalshi(ident)
    return None


async def _fetch_market_context(
    platform: Optional[str],
    ident: Optional[str],
    market_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Best-effort venue metadata fetch for rules, context, and live pricing."""
    venue = (platform or "").strip().lower() or "unknown"
    with _tracer.start_as_current_span("market.enrich_context") as span:
        span.set_attribute("market.venue", venue)
        try:
            quote = await asyncio.get_running_loop().run_in_executor(
                None,
                _fetch_live_odds,
                platform,
                ident,
                market_url,
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(
                otel_trace.Status(otel_trace.StatusCode.ERROR, type(exc).__name__)
            )
            _market_context_counter.add(
                1,
                {"market.venue": venue, "outcome": "failure"},
            )
            logger.warning(
                "market context enrichment failed venue=%s error=%s",
                venue,
                type(exc).__name__,
            )
            return None

        outcome = "success" if quote else "skipped"
        span.set_attribute(
            "market.rules.available",
            bool(quote and quote.get("resolution_criteria")),
        )
        _market_context_counter.add(
            1,
            {"market.venue": venue, "outcome": outcome},
        )
        return quote


async def _fetch_evidence_with_cache(
    evidence_question: str,
    top_k: int,
    *,
    source: str,
) -> tuple[List[Dict[str, Any]], Optional[str], str]:
    """Return cached or freshly retrieved evidence with bounded telemetry."""
    evidence_pipeline = _state.get("evidence_pipeline")
    attrs = {"source": source}
    started = time.monotonic()
    if evidence_pipeline is None:
        try:
            from analyzing_llm_rationale.news_pipeline import NewsPipeline
            evidence_pipeline = NewsPipeline(
                fetch_sources=("web", "gdelt", "google-news", "newsapi", "rss", "open-meteo"),
                summarize_articles=False,
                use_embeddings=False,
                min_relevance=float(os.environ.get("EVIDENCE_MIN_RELEVANCE", "0.25")),
            )
            _state["evidence_pipeline"] = evidence_pipeline
        except Exception as exc:
            _forecast_evidence_requests.add(1, {**attrs, "outcome": "unconfigured"})
            return [], f"Evidence pipeline is not configured on this server: {exc}", "unconfigured"

    top_k = max(1, min(top_k, 100))
    evidence_cache_key = _cache_key("evidence", evidence_question, top_k)
    cached = _cache_get(evidence_cache_key)
    if cached is not None:
        _forecast_evidence_requests.add(1, {**attrs, "outcome": "cache_hit"})
        _forecast_evidence_duration.record(time.monotonic() - started, {**attrs, "outcome": "cache_hit"})
        otel_trace.get_current_span().set_attribute("forecast.evidence.cache_hit", True)
        return cached, None, "cache_hit"

    _forecast_evidence_requests.add(1, {**attrs, "outcome": "cache_miss"})
    if not _evidence_fetch_slots.acquire(blocking=False):
        _forecast_evidence_requests.add(1, {**attrs, "outcome": "busy"})
        _forecast_evidence_duration.record(time.monotonic() - started, {**attrs, "outcome": "busy"})
        otel_trace.get_current_span().set_attribute("forecast.evidence.busy", True)
        return [], "Evidence retrieval is busy; forecast continued without fresh evidence.", "busy"

    try:
        def fetch_articles() -> Any:
            try:
                return evidence_pipeline.fetch_summarize_rank(
                    evidence_question,
                    top_k=top_k,
                )
            finally:
                _evidence_fetch_slots.release()

        fetch_task = asyncio.get_running_loop().run_in_executor(
            None,
            fetch_articles,
        )
        if _EVIDENCE_TIMEOUT_S and _EVIDENCE_TIMEOUT_S > 0:
            articles = await asyncio.wait_for(fetch_task, timeout=_EVIDENCE_TIMEOUT_S)
        else:
            articles = await fetch_task
        articles = articles or []
        outcome = "fresh_nonempty" if articles else "fresh_empty"
        if articles:
            _cache_set(evidence_cache_key, articles, _EVIDENCE_CACHE_TTL)
        _forecast_evidence_requests.add(1, {**attrs, "outcome": outcome})
        _forecast_evidence_duration.record(time.monotonic() - started, {**attrs, "outcome": outcome})
        span = otel_trace.get_current_span()
        span.set_attribute("forecast.evidence.cache_hit", False)
        span.set_attribute("forecast.evidence.count", len(articles))
        return articles, None, outcome
    except asyncio.TimeoutError:
        _forecast_evidence_requests.add(1, {**attrs, "outcome": "timeout"})
        _forecast_evidence_duration.record(time.monotonic() - started, {**attrs, "outcome": "timeout"})
        span = otel_trace.get_current_span()
        span.set_attribute("forecast.evidence.cache_hit", False)
        span.set_attribute("forecast.evidence.timeout_s", _EVIDENCE_TIMEOUT_S)
        logger.warning(
            "evidence retrieval timed out source=%s top_k=%s timeout_s=%.3f",
            source,
            top_k,
            _EVIDENCE_TIMEOUT_S,
        )
        return [], (
            f"Evidence retrieval timed out after {_EVIDENCE_TIMEOUT_S:g}s; "
            "forecast continued without fresh evidence."
        ), "timeout"
    except Exception as exc:
        _forecast_evidence_requests.add(1, {**attrs, "outcome": "error"})
        _forecast_evidence_duration.record(time.monotonic() - started, {**attrs, "outcome": "error"})
        return [], f"Evidence retrieval failed: {exc}", "error"


def _greeting_system_prompt_addition(req: "PredictRequest") -> str:
    """Extra system-prompt instruction for a first-turn greeting/meta question
    ("hello", "what can you do?") — introduces Foresea briefly instead of
    letting the model guess. Empty on any later turn, so an ongoing thread's
    "thanks!" just gets a brief acknowledgment, not a repeated pitch."""
    if req.history or not _is_greeting_or_meta(req.question):
        return ""
    return (
        "\n\nThe user just said hello or asked what this is, and this is the very "
        "start of a new conversation. Reply with a brief, warm introduction (2-3 "
        "sentences): Foresea turns forecasting questions into calibrated "
        "probabilities backed by evidence and, when a market price is available, "
        "how your estimate compares to it. Invite them to ask a forecasting "
        "question or paste a Polymarket/Kalshi market link to get started. Do not "
        "attempt a forecast or use the [p:0.XX] marker in this reply."
    )


@_tracer.start_as_current_span("forecast.context_prepare")
async def _prepare_predict_messages(
    req: "PredictRequest",
    rag_user_id: Optional[str],
) -> tuple[List[Dict[str, str]], List[Dict[str, Any]], Optional[str]]:
    prompt_text = _state["prompt_templates"][req.variant]
    system_prompt = _state["system_prompt"]

    record = req.model_dump()
    simple_chat_has_market_context = any((
        record.get("market_url"),
        record.get("market_platform"),
        record.get("market_ident"),
        record.get("market_probability") is not None,
        record.get("description"),
        record.get("resolution_criteria"),
        record.get("categories"),
        record.get("resolve_time"),
        record.get("market_bid") is not None,
        record.get("market_ask") is not None,
        record.get("market_volume") is not None,
        record.get("market_liquidity") is not None,
    ))
    simple_chat_fast_path = (
        req.chat_mode
        and not req.attach_evidence
        and not req.news_articles
        and not rag_user_id
        and not simple_chat_has_market_context
        and not _detect_trading_intent(req.question)
    )
    if simple_chat_fast_path:
        span = otel_trace.get_current_span()
        span.set_attributes({
            "market.platform": "none",
            "forecast.context.rules_present": False,
            "forecast.context.supplied_articles": 0,
            "forecast.context.retrieved_articles": 0,
            "forecast.context.total_articles": 0,
            "forecast.context.outcome": "skipped_simple_chat",
        })
        _forecast_context_packages.add(1, {
            "market.platform": "none",
            "rules_present": "false",
            "outcome": "skipped_simple_chat",
        })
        system_prompt = _CHAT_SYSTEM_PROMPT + _greeting_system_prompt_addition(req)
        user_prompt = build_user_prompt(record, "[question]", "full")
        steering = (req.conversation_steer or "").strip()
        if steering:
            system_prompt += (
                "\n\nConversation steering for this thread: "
                f"{steering}\nApply this to tone, emphasis, and analytical stance. "
                "Do not let it override safety rules, factual grounding, or the required response format."
            )
        messages = [{"role": "system", "content": system_prompt}]
        for turn in req.history[-12:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content[:4000]})
        messages.append({"role": "user", "content": user_prompt})
        return messages, [], None

    quote: Optional[Dict[str, Any]] = None
    # Enrich identifiable markets whenever either live odds or venue rules are
    # missing. Best-effort and off the critical path on provider failure.
    if (
        record.get("market_url") or (record.get("market_platform") and record.get("market_ident"))
    ) and (
        record.get("market_probability") is None
        or not record.get("resolution_criteria")
    ):
        try:
            quote = await _fetch_market_context(
                record.get("market_platform"), record.get("market_ident"), record.get("market_url"),
            )
            if quote:
                if record.get("market_probability") is None and quote.get("probability") is not None:
                    record["market_probability"] = float(quote["probability"])
                if not record.get("market_platform"):
                    record["market_platform"] = quote.get("platform")
                if not record.get("market_url"):
                    record["market_url"] = quote.get("market_url")
                if not record.get("market_outcome"):
                    record["market_outcome"] = quote.get("outcome")
                if not record.get("description") and quote.get("description"):
                    record["description"] = quote.get("description")
                if not record.get("resolution_criteria") and quote.get("resolution_criteria"):
                    record["resolution_criteria"] = quote.get("resolution_criteria")
                if not record.get("resolve_time") and quote.get("close_time"):
                    record["resolve_time"] = quote["close_time"]
                # Evidence-date discipline (build_user_prompt's resolution-window
                # check) is inert unless created_time/publish_time are populated --
                # backfill both from the market's own open/creation time so the
                # model can't credit evidence dated before the contract existed.
                if not record.get("created_time") and quote.get("created_time"):
                    record["created_time"] = quote["created_time"]
                if not record.get("publish_time") and quote.get("created_time"):
                    record["publish_time"] = quote["created_time"]
                # Market microstructure signals — populate only when not already supplied.
                if record.get("market_volume") is None and quote.get("volume") is not None:
                    record["market_volume"] = float(quote["volume"])
                if record.get("market_liquidity") is None and quote.get("liquidity") is not None:
                    record["market_liquidity"] = float(quote["liquidity"])
                if record.get("market_price_change_24h") is None and quote.get("price_change_24h") is not None:
                    record["market_price_change_24h"] = float(quote["price_change_24h"])
                if record.get("market_bid") is None and quote.get("yes_bid") is not None:
                    record["market_bid"] = float(quote["yes_bid"])
                if record.get("market_ask") is None and quote.get("yes_ask") is not None:
                    record["market_ask"] = float(quote["yes_ask"])
                if record.get("market_last_trade_price") is None and quote.get("last_trade_price") is not None:
                    record["market_last_trade_price"] = float(quote["last_trade_price"])
                if record.get("market_price_change_7d") is None and quote.get("price_change_7d") is not None:
                    record["market_price_change_7d"] = float(quote["price_change_7d"])
                for fld in ("resolution_source", "no_sub_title", "expected_expiration_time",
                            "floor_strike", "cap_strike"):
                    if record.get(f"market_{fld}") is None and quote.get(fld) is not None:
                        record[f"market_{fld}"] = quote[fld]
        except Exception:
            pass
    venue_articles = (
        quote.get("venue_news_articles") or []
        if quote
        else []
    )
    caller_articles = [article.model_dump() for article in req.news_articles]
    supplied_articles = _merge_evidence_articles(venue_articles, caller_articles)
    retrieved_articles: List[Dict[str, Any]] = []
    evidence_articles = list(supplied_articles)
    evidence_error = None
    evidence_question = (
        str(quote.get("question") or "").strip()
        if quote
        else ""
    ) or req.question

    # A short follow-up in an ongoing thread ("WE is 90+", "why?") is a poor
    # standalone search query. Keep retrieval on, but anchor it to the latest
    # substantive user question so the forecast still receives raw evidence.
    short_followup = bool(req.history) and len(req.question.split()) <= 6
    evidence_top_k = req.evidence_top_k
    if short_followup:
        previous_user_questions = [
            (turn.get("content") or "").strip()
            for turn in req.history
            if turn.get("role") == "user" and (turn.get("content") or "").strip()
        ]
        if previous_user_questions:
            evidence_question = (
                f"{previous_user_questions[-1][:500]}\n"
                f"Follow-up: {req.question}"
            )
            evidence_top_k = max(evidence_top_k, 5)

    if req.attach_evidence:
        retrieved_articles, evidence_error, _ = await _fetch_evidence_with_cache(
            evidence_question,
            evidence_top_k,
            source="forecast",
        )

    evidence_articles = _merge_evidence_articles(supplied_articles, retrieved_articles)
    evidence_articles = [_clean_article(a) for a in evidence_articles]
    # Personalised retrieval: prepend signed-in KB hits only when the embedder is
    # already warm. Forecasts should not pay a sentence-transformer cold start.
    if rag_user_id and rag.is_loaded():
        try:
            loop = asyncio.get_running_loop()
            kb_hits = await loop.run_in_executor(
                None, _rag_search, rag_user_id, "kb", req.question, 3
            )
            kb_articles = [
                _clean_article({
                    "title": h.get("title") or "Knowledge base",
                    "summary": h.get("text"),
                    "text": h.get("text"),
                    "source": h.get("source") or "Knowledge base",
                    "url": h.get("url") or None,
                    "relevance_score": h.get("score"),
                })
                for h in kb_hits
            ]
            evidence_articles = kb_articles + evidence_articles
            _forecast_rag_contexts.add(1, {
                "outcome": "success",
                "namespace": "kb",
                "hits": "nonzero" if kb_articles else "zero",
            })
        except Exception:
            _forecast_rag_contexts.add(1, {
                "outcome": "error",
                "namespace": "kb",
                "hits": "unknown",
            })
            pass
    elif rag_user_id:
        _forecast_rag_contexts.add(1, {
            "outcome": "skipped_cold",
            "namespace": "kb",
            "hits": "unknown",
        })
        otel_trace.get_current_span().set_attribute(
            "forecast.rag.skipped_cold_start",
            True,
        )

    if (
        req.attach_evidence
        and not evidence_articles
        and evidence_error is None
    ):
        evidence_error = (
            "No relevant live evidence sources were found after retrying retrieval."
        )
        logger.warning(
            "Evidence retrieval returned no relevant sources for question=%r",
            req.question,
        )

    rules_present = bool(str(record.get("resolution_criteria") or "").strip())
    platform = str(record.get("market_platform") or "none").strip().lower() or "none"
    context_outcome = (
        "complete" if rules_present and evidence_articles
        else "missing_rules" if not rules_present
        else "missing_evidence"
    )
    span = otel_trace.get_current_span()
    span.set_attributes({
        "market.platform": platform,
        "forecast.context.rules_present": rules_present,
        "forecast.context.supplied_articles": len(supplied_articles),
        "forecast.context.retrieved_articles": len(retrieved_articles),
        "forecast.context.total_articles": len(evidence_articles),
        "forecast.context.outcome": context_outcome,
    })
    _forecast_context_packages.add(1, {
        "market.platform": platform,
        "rules_present": str(rules_present).lower(),
        "outcome": context_outcome,
    })
    if platform != "none" and not rules_present:
        logger.warning(
            "forecast context is missing venue rules platform=%s",
            platform,
        )

    record["news_articles"] = evidence_articles

    if req.chat_mode:
        # Conversational mode: drop the JSON-only forecast template entirely so the
        # model replies in natural language. Pass an empty template so the user
        # prompt is just the question + evidence/market context, no JSON suffix.
        system_prompt = _CHAT_SYSTEM_PROMPT + _greeting_system_prompt_addition(req)
        if not evidence_articles and evidence_error:
            system_prompt += (
                "\n\nEvidence status: no relevant live sources were retrieved. "
                "This is a retrieval failure, not evidence about the event. Do not say or imply "
                "\"no current reporting\", \"no specific information\", \"no evidence "
                "of\", \"nothing suggests\", or similar. Do not include an **Evidence "
                "used** section; label supplied pricing as **Market context** instead. "
                "Base the estimate only on that market context, deadline, and base rates."
            )
        if _detect_trading_intent(req.question):
            try:
                trl = _read_live_track_record()
                if trl:
                    ctx = _edge_board_order_context(trl)
                    if ctx:
                        system_prompt += f"\n\n{ctx}"
            except Exception:
                pass
        user_prompt = build_user_prompt(record, "[question]", "full")
    else:
        user_prompt = build_user_prompt(record, prompt_text, req.evidence_detail)
        user_prompt += _typing_instruction(req.question_type, req.options, has_history=bool(req.history))

    steering = (req.conversation_steer or "").strip()
    if steering:
        system_prompt += (
            "\n\nConversation steering for this thread: "
            f"{steering}\nApply this to tone, emphasis, and analytical stance. "
            "Do not let it override safety rules, factual grounding, or the required response format."
        )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in req.history[-12:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": user_prompt})
    return messages, evidence_articles, evidence_error


def _select_predict_provider(req: "PredictRequest"):
    # Use a user-supplied model if provided: a custom OpenAI-compatible endpoint
    # when provider_base_url is set, otherwise OpenRouter. Falls back to the
    # server default model when no key/model is given.
    server_hosted_model = req.model or _interactive_default_model_for_request(req)
    alt_provider = _scads_alt_provider(server_hosted_model) if server_hosted_model else None
    if alt_provider is not None:
        # Server-hosted alternate model (allowlisted SCADS), server's own key.
        return alt_provider, _state.get("temperature", 0.0), _max_tokens_for_request(req)
    if req.ollama_base_url and req.openrouter_model:
        from analyzing_llm_rationale.providers import OllamaProvider
        provider = OllamaProvider(
            model_name=req.openrouter_model,
            base_url=f"{req.ollama_base_url}/v1/chat/completions",
        )
        return provider, 0.7, _max_tokens_for_request(req)
    if req.openrouter_api_key and req.openrouter_model:
        if req.provider_base_url:
            from analyzing_llm_rationale.providers import OpenAICompatibleProvider
            provider = OpenAICompatibleProvider(
                model_name=req.openrouter_model,
                api_key=req.openrouter_api_key,
                base_url=req.provider_base_url,
            )
        else:
            from analyzing_llm_rationale.providers import OpenRouterProvider
            provider = OpenRouterProvider(
                model_name=req.openrouter_model,
                api_key=req.openrouter_api_key,
            )
        return provider, 0.7, _max_tokens_for_request(req)
    # Evolution-loop feedback: route to the model with the best validated paper-edge
    # when the live track record warrants it (no-op until resolved data exists).
    auto = _auto_selected_model()
    if auto:
        auto_provider = _scads_alt_provider(auto)
        if auto_provider is not None:
            return auto_provider, _state.get("temperature", 0.0), _max_tokens_for_request(req)
    return _state["provider"], _state["temperature"], _max_tokens_for_request(req)


# ── Attachment extraction ─────────────────────────────────────────────────────

def _extract_pdf_bytes(content: bytes) -> str:
    try:
        import io  # noqa: PLC0415

        from pypdf import PdfReader  # noqa: PLC0415

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"PDF extraction failed: {exc}") from exc


def _extract_url_sync(url: str) -> Dict[str, Any]:
    import re
    from urllib.parse import urlparse
    try:
        import requests as _req
        from bs4 import BeautifulSoup
        resp = _req.get(url, timeout=20, headers={"User-Agent": "Foresea/1.0"}, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title = (soup.title.string or "").strip() if soup.title else urlparse(url).netloc
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        return {"title": title, "text": text[:20000], "source": urlparse(url).netloc, "url": url}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"URL extraction failed: {exc}") from exc


def _analytics_conn():
    _ANALYTICS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(_ANALYTICS_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS page_visits (
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            path TEXT,
            referrer TEXT,
            user_agent TEXT,
            timezone TEXT,
            visitor_hash TEXT,
            account_ref TEXT,
            attribution TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics_events (
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            day TEXT,
            event_name TEXT,
            path TEXT,
            user_id TEXT,
            visitor_id TEXT,
            account_ref TEXT,
            attribution TEXT,
            metadata_json TEXT
        )
        """
    )
    # Existing local analytics stores predate authenticated attribution.
    # These additive migrations keep the fallback usable across upgrades.
    conn.execute("ALTER TABLE page_visits ADD COLUMN IF NOT EXISTS account_ref TEXT")
    conn.execute("ALTER TABLE page_visits ADD COLUMN IF NOT EXISTS attribution TEXT")
    conn.execute("ALTER TABLE analytics_events ADD COLUMN IF NOT EXISTS account_ref TEXT")
    conn.execute("ALTER TABLE analytics_events ADD COLUMN IF NOT EXISTS attribution TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shared_forecasts (
            share_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT,
            payload_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            sub TEXT PRIMARY KEY,
            email TEXT,
            name TEXT,
            picture TEXT,
            created_at TIMESTAMP,
            last_login TIMESTAMP
        )
        """
    )
    return conn


def _visitor_ip_ua(request: Request) -> tuple[str, str]:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else ""
    if not ip and request.client:
        ip = request.client.host
    return ip, request.headers.get("user-agent", "")


def _visitor_hash(request: Request) -> str:
    """Per-day salted visitor hash (used by the DuckDB fallback's daily uniques)."""
    ip, user_agent = _visitor_ip_ua(request)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    salt = os.environ.get("ANALYTICS_SALT", "foresea-analytics")
    raw = f"{day}:{ip}:{user_agent}:{salt}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _visitor_id(request: Request) -> str:
    """Stable, day-independent visitor id (no raw IP stored) so cumulative
    unique-visitor counts mean *distinct people over all time*, not per-day."""
    ip, user_agent = _visitor_ip_ua(request)
    salt = os.environ.get("ANALYTICS_SALT", "foresea-analytics")
    raw = f"{ip}:{user_agent}:{salt}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _analytics_account_ref(user_id: str) -> str:
    """Return a domain-separated HMAC reference without persisting a raw account ID."""
    digest = hmac.new(
        _SESSION_SECRET.encode("utf-8"),
        f"foresea.analytics.account.v1:{user_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"acct_{digest[:32]}"


def _analytics_attribution(request: Optional[Request]) -> tuple[str, Optional[str]]:
    """Resolve an analytics attribution class and non-reversible account reference."""
    user_id = _optional_user_id(request)
    if not user_id:
        return "anonymous", None
    return "authenticated", _analytics_account_ref(user_id)


def _analytics_attribution_summary(
    records: Iterable[Dict[str, Any]],
    total_registered: Optional[int] = None,
) -> AnalyticsAttributionSummary:
    """Summarize records without exposing account references or legacy raw IDs."""
    authenticated_records = 0
    anonymous_records = 0
    accounts: set[str] = set()
    for record in records:
        account_ref = record.get("account_ref")
        if not account_ref and record.get("user_id"):
            # Legacy rows may have a raw ID; transform it only in memory so the
            # aggregate remains comparable without returning or re-persisting it.
            account_ref = _analytics_account_ref(str(record["user_id"]))
        if account_ref:
            authenticated_records += 1
            accounts.add(str(account_ref))
        else:
            anonymous_records += 1
    total_reg = total_registered if total_registered is not None else _count_registered_users()
    return AnalyticsAttributionSummary(
        authenticated_records=authenticated_records,
        anonymous_records=anonymous_records,
        authenticated_accounts=len(accounts),
        total_registered_users=int(total_reg or 0),
    )


def _record_visit_duckdb(
    event: VisitRequest,
    request: Request,
    attribution: str,
    account_ref: Optional[str],
) -> None:
    conn = _analytics_conn()
    try:
        conn.execute(
            """
            INSERT INTO page_visits
                (path, referrer, user_agent, timezone, visitor_hash, account_ref, attribution)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event.path,
                event.referrer,
                request.headers.get("user-agent", "")[:1000],
                event.timezone,
                _visitor_hash(request),
                account_ref,
                attribution,
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _record_visit_datastore(
    event: VisitRequest,
    request: Request,
    attribution: str,
    account_ref: Optional[str],
) -> None:
    """Persist a visit in Cloud Datastore so counts survive instance recycles.

    Cumulative totals live in an ``AnalyticsStats`` singleton updated in a
    transaction; ``Visitor`` entities (keyed by the stable visitor id) dedupe
    unique people; ``PageVisit`` rows back the 30-day daily breakdown. No raw
    IP is ever stored.
    """
    client = _get_datastore()
    from google.cloud import datastore as _ds

    vid = _visitor_id(request)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    now = datetime.now(timezone.utc)

    with client.transaction():
        visit = _ds.Entity(
            client.key("PageVisit"),
            exclude_from_indexes=("account_ref", "referrer", "user_agent", "timezone", "path"),
        )
        visit.update(
            ts=now, day=day, path=event.path, referrer=event.referrer,
            timezone=event.timezone, visitor_id=vid, attribution=attribution,
            account_ref=account_ref,
            user_agent=request.headers.get("user-agent", "")[:1000],
        )
        client.put(visit)

        stats_key = client.key("AnalyticsStats", "global")
        stats = client.get(stats_key) or _ds.Entity(stats_key)
        visitor_key = client.key("Visitor", vid)
        is_new_visitor = client.get(visitor_key) is None
        stats["total_visits"] = int(stats.get("total_visits", 0)) + 1
        if is_new_visitor:
            stats["unique_visitors"] = int(stats.get("unique_visitors", 0)) + 1
            client.put(_ds.Entity(visitor_key))
        client.put(stats)


def _record_visit(
    event: VisitRequest,
    request: Request,
    attribution: Optional[str] = None,
    account_ref: Optional[str] = None,
) -> str:
    """Record a visit in Datastore when available, else the local DuckDB."""
    if attribution is None:
        attribution, account_ref = _analytics_attribution(request)
    if _get_datastore() is not None:
        try:
            _record_visit_datastore(event, request, attribution, account_ref)
            return "datastore"
        except Exception:
            logger.warning("datastore visit record failed; falling back to duckdb", exc_info=True)
    _record_visit_duckdb(event, request, attribution, account_ref)
    return "duckdb"


def _record_analytics_event_duckdb(
    event: AnalyticsEventRequest,
    request: Request,
    attribution: str,
    account_ref: Optional[str],
) -> None:
    conn = _analytics_conn()
    try:
        conn.execute(
            """
            INSERT INTO analytics_events
                (day, event_name, path, user_id, visitor_id, account_ref, attribution, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                time.strftime("%Y-%m-%d", time.gmtime()),
                event.event_name,
                event.path,
                None,
                _visitor_id(request),
                account_ref,
                attribution,
                json.dumps(event.metadata or {}, default=str, sort_keys=True),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _record_analytics_event_datastore(
    event: AnalyticsEventRequest,
    request: Request,
    attribution: str,
    account_ref: Optional[str],
) -> None:
    client = _get_datastore()
    from google.cloud import datastore as _ds

    now = datetime.now(timezone.utc)
    entity = _ds.Entity(
        client.key("AnalyticsEvent"),
        exclude_from_indexes=("account_ref", "metadata", "path"),
    )
    entity.update(
        ts=now,
        day=time.strftime("%Y-%m-%d", time.gmtime()),
        event_name=event.event_name,
        path=event.path,
        account_ref=account_ref,
        attribution=attribution,
        visitor_id=_visitor_id(request),
        metadata=json.dumps(event.metadata or {}, default=str, sort_keys=True)[:4000],
    )
    client.put(entity)


def _record_analytics_event(
    event: AnalyticsEventRequest,
    request: Request,
    attribution: Optional[str] = None,
    account_ref: Optional[str] = None,
) -> str:
    if attribution is None:
        attribution, account_ref = _analytics_attribution(request)
    if _get_datastore() is not None:
        try:
            _record_analytics_event_datastore(event, request, attribution, account_ref)
            return "datastore"
        except Exception:
            logger.warning("datastore analytics event failed; falling back to duckdb", exc_info=True)
    _record_analytics_event_duckdb(event, request, attribution, account_ref)
    return "duckdb"


def _analytics_events_summary_datastore() -> "AnalyticsEventSummary":
    client = _get_datastore()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_7d = now - timedelta(days=7)
    query = client.query(kind="AnalyticsEvent")
    query.add_filter("ts", ">=", cutoff)
    records = list(query.fetch(limit=10000))
    by_event: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "authenticated": 0, "anonymous": 0})
    by_day: Dict[str, int] = defaultdict(int)
    by_model: Dict[str, int] = defaultdict(int)
    events_24h = 0
    active_accounts_24h: set = set()
    active_accounts_7d: set = set()
    for entity in records:
        ts = entity.get("ts")
        name = entity.get("event_name") or "unknown"
        is_auth = bool(entity.get("account_ref") or entity.get("user_id"))
        by_event[name]["count"] += 1
        if is_auth:
            by_event[name]["authenticated"] += 1
        else:
            by_event[name]["anonymous"] += 1
        by_day[entity.get("day") or entity["ts"].strftime("%Y-%m-%d")] += 1

        # Model usage tracking from metadata
        raw_meta = entity.get("metadata")
        if raw_meta:
            try:
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                if isinstance(meta, dict) and meta.get("model"):
                    by_model[str(meta["model"])] += 1
            except Exception:
                pass

        if ts and ts >= cutoff_24h:
            events_24h += 1
            ref = entity.get("account_ref") or (_analytics_account_ref(str(entity["user_id"])) if entity.get("user_id") else None)
            if ref:
                active_accounts_24h.add(ref)
        if ts and ts >= cutoff_7d:
            ref = entity.get("account_ref") or (_analytics_account_ref(str(entity["user_id"])) if entity.get("user_id") else None)
            if ref:
                active_accounts_7d.add(ref)
    return AnalyticsEventSummary(
        total_events=len(records),
        by_event=[
            {
                "event_name": name,
                "count": agg["count"],
                "authenticated": agg["authenticated"],
                "anonymous": agg["anonymous"],
            }
            for name, agg in sorted(by_event.items(), key=lambda kv: kv[1]["count"], reverse=True)
        ],
        by_day=[
            {"day": day, "count": count}
            for day, count in sorted(by_day.items(), reverse=True)
        ],
        by_model=[
            {"model": model, "count": count}
            for model, count in sorted(by_model.items(), key=lambda kv: kv[1], reverse=True)
        ],
        attribution=_analytics_attribution_summary(records),
        events_24h=events_24h,
        active_accounts_24h=len(active_accounts_24h),
        active_accounts_7d=len(active_accounts_7d),
    )


def _recent_analytics_events_datastore(limit: int = 20) -> List[Dict[str, Any]]:
    client = _get_datastore()
    query = client.query(kind="AnalyticsEvent")
    query.order = ["-ts"]
    entities = list(query.fetch(limit=limit))
    events = []
    for entity in entities:
        meta = {}
        if entity.get("metadata"):
            try:
                meta = json.loads(entity["metadata"]) if isinstance(entity["metadata"], str) else entity["metadata"]
            except Exception:
                meta = {}
        ts_val = entity.get("ts")
        events.append({
            "event_name": str(entity.get("event_name", "")),
            "path": str(entity.get("path", "/")),
            "attribution": str(entity.get("attribution", "anonymous")),
            "ts": ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val or ""),
            "metadata": meta,
        })
    return events


def _recent_analytics_events_duckdb(limit: int = 20) -> List[Dict[str, Any]]:
    conn = _analytics_conn()
    try:
        rows = conn.execute(
            """
            SELECT event_name, path, attribution, ts, metadata_json
            FROM analytics_events
            ORDER BY ts DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
    finally:
        conn.close()
    events = []
    for event_name, path, attribution, ts_val, metadata_json in rows:
        meta = {}
        if metadata_json:
            try:
                meta = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
            except Exception:
                meta = {}
        events.append({
            "event_name": str(event_name or ""),
            "path": str(path or "/"),
            "attribution": str(attribution or "anonymous"),
            "ts": ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val or ""),
            "metadata": meta,
        })
    return events


def _count_registered_users() -> int:
    """Return the total count of registered user accounts."""
    client = _get_datastore()
    if client is not None:
        try:
            q = client.query(kind="User")
            q.keys_only()
            return len(list(q.fetch()))
        except Exception:
            pass
    try:
        conn = _analytics_conn()
        try:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] or 0)
        finally:
            conn.close()
    except Exception:
        return 0


def _get_registered_users_datastore(limit: int = 500) -> List[Dict[str, Any]]:
    """Fetch registered users with their verified email IDs and timestamps from Datastore."""
    client = _get_datastore()
    if client is None:
        return []
    lim = int(limit) if isinstance(limit, (int, str)) and str(limit).isdigit() else 500
    query = client.query(kind="User")
    entities = list(query.fetch(limit=lim))
    users = []
    for e in entities:
        c_at = e.get("created_at")
        l_in = e.get("last_login")
        c_at_str = c_at.isoformat() if hasattr(c_at, "isoformat") else (str(c_at) if c_at else None)
        l_in_str = l_in.isoformat() if hasattr(l_in, "isoformat") else (str(l_in) if l_in else None)
        users.append({
            "user_id": e.key.name or str(e.key.id or ""),
            "email": str(e.get("email") or ""),
            "name": str(e.get("name") or ""),
            "picture": str(e.get("picture") or ""),
            "created_at": c_at_str,
            "last_login": l_in_str,
        })
    users.sort(key=lambda u: u.get("last_login") or u.get("created_at") or "", reverse=True)
    return users


def _get_registered_users_duckdb(limit: int = 500) -> List[Dict[str, Any]]:
    """Fetch registered users with their verified email IDs and timestamps from DuckDB."""
    lim = int(limit) if isinstance(limit, (int, str)) and str(limit).isdigit() else 500
    conn = _analytics_conn()
    try:
        rows = conn.execute(
            """
            SELECT sub, email, name, picture, created_at, last_login
            FROM users
            ORDER BY last_login DESC NULLS LAST
            LIMIT ?
            """,
            [lim],
        ).fetchall()
    finally:
        conn.close()
    users = []
    for sub, email, name, picture, c_at, l_in in rows:
        c_at_str = c_at.isoformat() if hasattr(c_at, "isoformat") else (str(c_at) if c_at else None)
        l_in_str = l_in.isoformat() if hasattr(l_in, "isoformat") else (str(l_in) if l_in else None)
        users.append({
            "user_id": str(sub or ""),
            "email": str(email or ""),
            "name": str(name or ""),
            "picture": str(picture or ""),
            "created_at": c_at_str,
            "last_login": l_in_str,
        })
    return users


def _radar_id(platform: str, market_url: Optional[str], question: str) -> str:
    source = f"{platform}:{market_url or question}"
    return hashlib.sha256(source.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _radar_from_track_record(limit: int = 12) -> "RadarResponse":
    radar_cache_ttl = int(os.environ.get("RADAR_CACHE_TTL", "0"))
    cache_key = _cache_key("radar", limit)
    if radar_cache_ttl > 0:
        cached = _cache_get(cache_key)
        if cached is not None:
            return RadarResponse(**cached)
    payload = _read_edge_board_record() or {}
    rows = payload.get("edge_board") or []
    markets: List[RadarMarket] = []
    seen: set[str] = set()
    for row in rows:
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        platform = str(row.get("platform") or "").strip() or "Prediction market"
        market_url = row.get("market_url")
        ident = _radar_id(platform, market_url, question)
        if ident in seen:
            continue
        seen.add(ident)
        edge = row.get("edge")
        abs_edge = row.get("abs_edge")
        try:
            edge_pp = round(float(edge) * 100)
            reason = f"Foresea is {abs(edge_pp)} pts {'above' if edge_pp > 0 else 'below'} the market."
        except Exception:
            reason = "Live market with a model-vs-market gap."
        markets.append(RadarMarket(
            id=ident,
            ident=row.get("ident"),
            platform=platform,
            question=question[:500],
            market_url=market_url,
            description=row.get("description"),
            resolution_criteria=row.get("resolution_criteria"),
            categories=row.get("categories") or [],
            market_probability=row.get("market_probability"),
            model_probability=row.get("model_probability"),
            edge=edge,
            abs_edge=abs_edge,
            side=row.get("side"),
            category=row.get("domain"),
            horizon=row.get("horizon") or row.get("lead_bucket"),
            reason=reason,
        ))
        if len(markets) >= limit:
            break
    response = RadarResponse(
        updated_at=str(payload.get("generated_at") or datetime.now(timezone.utc).isoformat()),
        markets=markets,
        generated_at=payload.get("generated_at"),
        model=payload.get("model"),
        edge_board=rows[:limit],
        models_comparison=payload.get("models_comparison") or [],
        paper_pnl=payload.get("paper_pnl"),
        primary_paper_pnl=payload.get("primary_paper_pnl"),
        mark_to_market_account=payload.get("mark_to_market_account"),
        mark_to_market_by_model=payload.get("mark_to_market_by_model") or [],
        mark_to_market_cycle_minutes=payload.get("mark_to_market_cycle_minutes"),
        lead_lag=payload.get("lead_lag"),
        calibration=payload.get("calibration"),
        resolved_log=payload.get("resolved_log") or [],
        n_snapshots_resolved=int(payload.get("n_snapshots_resolved") or 0),
        n_markets_resolved=int(payload.get("n_markets_resolved") or 0),
        primary_model=payload.get("primary_model") or payload.get("model"),
        primary_n_snapshots_resolved=int(payload.get("primary_n_snapshots_resolved") or 0),
        primary_n_markets_resolved=int(payload.get("primary_n_markets_resolved") or 0),
        n_markets_open=int(payload.get("n_markets_open") or len(rows)),
        freshness=_track_record_freshness(payload),
    )
    _cache_set(cache_key, response.model_dump(mode="json"), radar_cache_ttl)
    return response


async def _prefetch_radar_evidence(markets: List["RadarMarket"]) -> None:
    """Warm evidence cache for top radar markets without delaying /radar."""
    if _EVIDENCE_PREFETCH_TOP_N <= 0 or _state.get("evidence_pipeline") is None:
        return
    top_k = 3
    candidates = [
        str(m.question or "").strip()
        for m in markets[:_EVIDENCE_PREFETCH_TOP_N]
        if str(m.question or "").strip()
    ]
    for question in candidates:
        key = _cache_key("evidence", question, top_k)
        if _cache_get(key) is not None:
            _radar_evidence_prefetches.add(1, {"outcome": "cache_hit"})
            continue
        if key in _evidence_prefetch_inflight:
            _radar_evidence_prefetches.add(1, {"outcome": "coalesced"})
            continue
        _evidence_prefetch_inflight.add(key)
        with _tracer.start_as_current_span("radar.evidence_prefetch") as span:
            span.set_attribute("radar.evidence.top_k", top_k)
            try:
                articles, _, outcome = await _fetch_evidence_with_cache(
                    question,
                    top_k,
                    source="radar_prefetch",
                )
                _radar_evidence_prefetches.add(1, {
                    "outcome": "filled" if articles else outcome,
                })
            except Exception as exc:
                span.record_exception(exc)
                _radar_evidence_prefetches.add(1, {"outcome": "error"})
                logger.warning("radar evidence prefetch failed: %s", type(exc).__name__)
            finally:
                _evidence_prefetch_inflight.discard(key)


def _share_payload(req: SharedForecastRequest, request: Request) -> Dict[str, Any]:
    payload = req.model_dump(mode="json")
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["user_id"] = _optional_user_id(request)
    payload["sources"] = [
        {
            "title": str(src.get("title") or src.get("source") or "")[:300],
            "source": str(src.get("source") or "")[:160],
            "url": str(src.get("url") or "")[:1000],
        }
        for src in (req.sources or [])[:12]
        if isinstance(src, dict)
    ]
    return payload


def _store_shared_forecast(req: SharedForecastRequest, request: Request) -> "SharedForecastResponse":
    share_id = "".join(secrets.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(10))
    payload = _share_payload(req, request)
    client = _get_datastore()
    if client is not None:
        try:
            from google.cloud import datastore as _ds
            entity = _ds.Entity(
                key=client.key("SharedForecast", share_id),
                exclude_from_indexes=("payload",),
            )
            entity.update(
                share_id=share_id,
                created_at=datetime.now(timezone.utc),
                user_id=payload.get("user_id"),
                payload=json.dumps(payload, default=str, sort_keys=True),
            )
            client.put(entity)
            return SharedForecastResponse(share_id=share_id, url=f"{_CANONICAL}/forecast/{share_id}")
        except Exception:
            logger.warning("datastore shared forecast write failed; falling back to duckdb", exc_info=True)
    conn = _analytics_conn()
    try:
        conn.execute(
            "INSERT INTO shared_forecasts (share_id, user_id, payload_json) VALUES (?, ?, ?)",
            [share_id, payload.get("user_id"), json.dumps(payload, default=str, sort_keys=True)],
        )
        conn.commit()
    finally:
        conn.close()
    return SharedForecastResponse(share_id=share_id, url=f"{_CANONICAL}/forecast/{share_id}")


def _read_shared_forecast(share_id: str) -> Optional[Dict[str, Any]]:
    client = _get_datastore()
    if client is not None:
        try:
            entity = client.get(client.key("SharedForecast", share_id))
            if entity is not None:
                return json.loads(entity.get("payload") or "{}")
        except Exception:
            logger.warning("datastore shared forecast read failed; falling back to duckdb", exc_info=True)
    conn = _analytics_conn()
    try:
        row = conn.execute(
            "SELECT payload_json FROM shared_forecasts WHERE share_id = ?",
            [share_id],
        ).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()


def _shared_forecast_html(share_id: str, payload: Dict[str, Any]) -> str:
    q = html.escape(str(payload.get("question") or "Forecast"))
    q_param = url_quote(str(payload.get("question") or "Forecast"))
    answer = html.escape(str(payload.get("predicted_answer") or "Forecast"))
    rationale = html.escape(str(payload.get("rationale") or ""))
    model = html.escape(str(payload.get("model") or "Foresea"))
    conf = payload.get("confidence")
    model_p = payload.get("model_probability")
    market_p = payload.get("market_probability")
    platform = html.escape(str(payload.get("market_platform") or "Market"))
    market_url = html.escape(str(payload.get("market_url") or ""))
    pct_line = []
    if model_p is not None:
        pct_line.append(f"Foresea {round(float(model_p) * 100)}%")
    elif conf is not None:
        pct_line.append(f"Confidence {round(float(conf) * 100)}%")
    if market_p is not None:
        pct_line.append(f"{platform} {round(float(market_p) * 100)}%")
    sources = payload.get("sources") or []
    source_html = "".join(
        f'<a class="chip" href="{html.escape(str(s.get("url") or "#"))}" target="_blank" rel="noopener">'
        f'{html.escape(str(s.get("title") or s.get("source") or "Source"))}</a>'
        for s in sources if isinstance(s, dict)
    )
    market_link = (
        f'<a class="btn ghost" href="{market_url}" target="_blank" rel="noopener">Open market</a>'
        if market_url else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{q} | Foresea forecast</title>
  <meta name="description" content="{q}">
  <link rel="canonical" href="{_CANONICAL}/forecast/{share_id}">
  <style>
    body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#111; background:#f7f7f8; }}
    main {{ max-width:760px; margin:0 auto; padding:48px 18px; }}
    .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:28px; box-shadow:0 18px 50px rgba(15,15,16,.06); }}
    .brand {{ display:flex; align-items:center; gap:9px; font-weight:700; margin-bottom:24px; }}
    .dot {{ width:20px; height:20px; border-radius:6px; background:#111; }}
    h1 {{ font-size:30px; line-height:1.12; margin:0 0 18px; letter-spacing:0; }}
    .answer {{ display:inline-flex; background:#111827; color:#fff; border-radius:4px; padding:6px 12px; font-weight:700; margin-right:10px; }}
    .muted {{ color:#6b7280; }}
    .rationale {{ margin-top:22px; line-height:1.65; white-space:pre-wrap; }}
    .actions {{ margin-top:26px; display:flex; gap:10px; flex-wrap:wrap; }}
    .btn {{ display:inline-block; border-radius:999px; padding:9px 14px; background:#111; color:#fff; text-decoration:none; font-weight:600; font-size:14px; }}
    .btn.ghost {{ background:#fff; color:#111; border:1px solid #d1d5db; }}
    .chip {{ display:inline-block; margin:6px 6px 0 0; border:1px solid #e5e7eb; border-radius:999px; padding:6px 10px; color:#374151; text-decoration:none; font-size:13px; }}
  </style>
</head>
<body>
  <main>
    <article class="card">
      <div class="brand"><span class="dot"></span><span>Foresea</span></div>
      <h1>{q}</h1>
      <div><span class="answer">{answer}</span><span class="muted">{html.escape(" · ".join(pct_line))}</span></div>
      <div class="rationale">{rationale}</div>
      {f'<div style="margin-top:22px">{source_html}</div>' if source_html else ''}
      <div class="actions">
        <a class="btn" href="{_CANONICAL}/?q={q_param}">Ask Foresea</a>
        {market_link}
      </div>
      <p class="muted" style="margin-top:24px;font-size:12px">Shared forecast generated by {model}. Decision support only.</p>
    </article>
  </main>
</body>
</html>"""


def _analytics_summary_datastore() -> "AnalyticsSummary":
    client = _get_datastore()
    stats = client.get(client.key("AnalyticsStats", "global"))
    total = int(stats["total_visits"]) if stats else 0
    unique = int(stats["unique_visitors"]) if stats else 0

    # Daily breakdown (last 30 days) from PageVisit rows; distinct counted in
    # Python over a bounded window.
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)
    cutoff_24h = now - timedelta(hours=24)
    query = client.query(kind="PageVisit")
    query.add_filter("ts", ">=", cutoff)
    by_day: Dict[str, Dict[str, Any]] = {}
    records = list(query.fetch())
    visits_24h = 0
    for e in records:
        d = e.get("day") or e["ts"].strftime("%Y-%m-%d")
        agg = by_day.setdefault(d, {"visits": 0, "visitors": set()})
        agg["visits"] += 1
        if e.get("visitor_id"):
            agg["visitors"].add(e["visitor_id"])
        ts = e.get("ts")
        if ts and ts >= cutoff_24h:
            visits_24h += 1
    rows = sorted(by_day.items(), reverse=True)[:30]
    return AnalyticsSummary(
        total_visits=total,
        unique_visitors=unique,
        by_day=[
            {"day": d, "visits": agg["visits"], "unique_visitors": len(agg["visitors"])}
            for d, agg in rows
        ],
        attribution=_analytics_attribution_summary(records),
        visits_24h=visits_24h,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="Service status.",
    responses={200: {"content": {"application/json": {"example": {"status": "ok"}}}}},
)
async def health() -> Dict[str, str]:
    """Returns `{"status": "ok"}` when the server is running.

    Use this endpoint for liveness probes and uptime monitoring.
    It does **not** verify that the LLM provider is reachable.
    """
    return {"status": "ok"}


@app.get(
    "/ready",
    tags=["System"],
    summary="Readiness check",
    response_description="Whether the service is ready to serve traffic.",
)
async def ready() -> JSONResponse:
    """Returns 200 when the server is initialised and accepting traffic, 503
    otherwise (still booting, misconfigured, or draining on shutdown).

    Unlike `/health` (liveness), this verifies the forecasting provider is
    configured — use it for load-balancer readiness routing.
    """
    provider_ready = bool(_state.get("provider"))
    ok = _ready and provider_ready
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "ready": ok,
            "provider_configured": provider_ready,
            "draining": not _ready,
        },
    )


@app.post("/analytics/visit", tags=["System"], summary="Record a privacy-preserving page visit")
async def record_visit(event: VisitRequest, request: Request) -> Dict[str, str]:
    """Record one page visit with optional, non-reversible account attribution.

    Stores no raw IP address. Unique visitors are estimated with a salted hash
    of IP address and user agent.
    """
    attribution, account_ref = _analytics_attribution(request)
    with _tracer.start_as_current_span("analytics.visit.record") as span:
        span.set_attribute("analytics.attribution", attribution)
        try:
            # Datastore/DuckDB I/O is blocking — keep it off the event loop.
            store = await asyncio.get_running_loop().run_in_executor(
                None, _record_visit, event, request, attribution, account_ref
            )
            span.set_attributes({"analytics.store": store, "outcome": "success"})
            _analytics_attribution_actions.add(
                1, {"kind": "visit", "attribution": attribution, "outcome": "success"}
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, type(exc).__name__))
            span.set_attribute("outcome", "error")
            _analytics_attribution_actions.add(
                1, {"kind": "visit", "attribution": attribution, "outcome": "error"}
            )
            logger.exception("analytics visit record failed")
            raise
    return {"status": "ok"}


@app.post("/analytics/event", tags=["System"], summary="Record product engagement event")
async def record_analytics_event(event: AnalyticsEventRequest, request: Request) -> Dict[str, str]:
    """Record a lightweight product event with optional private attribution."""
    attribution, account_ref = _analytics_attribution(request)
    with _tracer.start_as_current_span("analytics.event.record") as span:
        span.set_attribute("analytics.attribution", attribution)
        try:
            store = await asyncio.get_running_loop().run_in_executor(
                None, _record_analytics_event, event, request, attribution, account_ref
            )
            span.set_attributes({"analytics.store": store, "outcome": "success"})
            _analytics_attribution_actions.add(
                1, {"kind": "event", "attribution": attribution, "outcome": "success"}
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, type(exc).__name__))
            span.set_attribute("outcome", "error")
            _analytics_attribution_actions.add(
                1, {"kind": "event", "attribution": attribution, "outcome": "error"}
            )
            logger.exception("analytics event record failed")
            raise
    return {"status": "ok"}


@app.get("/analytics/summary", tags=["System"], response_model=AnalyticsSummary)
async def analytics_summary(request: Request) -> AnalyticsSummary:
    """Return basic page-visit counts.

    If `API_KEY` is configured, this endpoint requires `X-API-Key`.

    Counts are persisted in Cloud Datastore when available (survive deploys /
    instance recycles); otherwise they come from the local ephemeral DuckDB.
    """
    _check_api_key(request)
    if _get_datastore() is not None:
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, _analytics_summary_datastore
            )
        except Exception:
            logger.warning("datastore summary failed; falling back to duckdb", exc_info=True)
    conn = _analytics_conn()
    try:
        total, unique_visitors = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT visitor_hash) FROM page_visits"
        ).fetchone()
        visits_24h = conn.execute(
            "SELECT COUNT(*) FROM page_visits WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 24 HOUR"
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT
                CAST(ts AS DATE) AS day,
                COUNT(*) AS visits,
                COUNT(DISTINCT visitor_hash) AS unique_visitors
            FROM page_visits
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 30
            """
        ).fetchall()
        attribution_rows = conn.execute(
            """
            SELECT account_ref, attribution
            FROM page_visits
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 30 DAY
            """
        ).fetchall()
    finally:
        conn.close()

    return AnalyticsSummary(
        total_visits=int(total or 0),
        unique_visitors=int(unique_visitors or 0),
        by_day=[
            {
                "day": str(day),
                "visits": int(visits),
                "unique_visitors": int(unique_count),
            }
            for day, visits, unique_count in rows
        ],
        attribution=_analytics_attribution_summary(
            [{"account_ref": account_ref, "attribution": attribution}
             for account_ref, attribution in attribution_rows]
        ),
        visits_24h=int(visits_24h or 0),
    )


@app.get("/analytics/events/summary", tags=["System"], response_model=AnalyticsEventSummary)
async def analytics_events_summary(request: Request) -> AnalyticsEventSummary:
    """Return product-event counts for the last 30 days.

    If `API_KEY` is configured, this endpoint requires `X-API-Key`.
    """
    _check_api_key(request)
    if _get_datastore() is not None:
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, _analytics_events_summary_datastore
            )
        except Exception:
            logger.warning("datastore event summary failed; falling back to duckdb", exc_info=True)
    conn = _analytics_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM analytics_events "
            "WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 30 DAY"
        ).fetchone()[0]
        events_24h = conn.execute(
            "SELECT COUNT(*) FROM analytics_events "
            "WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 24 HOUR"
        ).fetchone()[0]
        by_event = conn.execute(
            """
            SELECT event_name, COUNT(*) AS count,
                   COUNT(CASE WHEN account_ref IS NOT NULL OR user_id IS NOT NULL THEN 1 END) AS authenticated,
                   COUNT(CASE WHEN account_ref IS NULL AND user_id IS NULL THEN 1 END) AS anonymous
            FROM analytics_events
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 30 DAY
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 50
            """
        ).fetchall()
        by_day = conn.execute(
            """
            SELECT day, COUNT(*) AS count
            FROM analytics_events
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 30 DAY
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 30
            """
        ).fetchall()
        attribution_rows = conn.execute(
            """
            SELECT account_ref, user_id, attribution
            FROM analytics_events
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 30 DAY
            """
        ).fetchall()
        active_24h_rows = conn.execute(
            """
            SELECT account_ref, user_id
            FROM analytics_events
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 24 HOUR
              AND (account_ref IS NOT NULL OR user_id IS NOT NULL)
            """
        ).fetchall()
        active_7d_rows = conn.execute(
            """
            SELECT account_ref, user_id
            FROM analytics_events
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 7 DAY
              AND (account_ref IS NOT NULL OR user_id IS NOT NULL)
            """
        ).fetchall()
        meta_rows = conn.execute(
            """
            SELECT metadata_json
            FROM analytics_events
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 30 DAY
              AND metadata_json IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()
    active_accounts_24h = {
        (row[0] or _analytics_account_ref(str(row[1])))
        for row in active_24h_rows
        if row[0] or row[1]
    }
    active_accounts_7d = {
        (row[0] or _analytics_account_ref(str(row[1])))
        for row in active_7d_rows
        if row[0] or row[1]
    }
    by_model: Dict[str, int] = defaultdict(int)
    for (mjson,) in meta_rows:
        if mjson:
            try:
                m = json.loads(mjson) if isinstance(mjson, str) else mjson
                if isinstance(m, dict) and m.get("model"):
                    by_model[str(m["model"])] += 1
            except Exception:
                pass
    return AnalyticsEventSummary(
        total_events=int(total or 0),
        by_event=[
            {
                "event_name": name,
                "count": int(count),
                "authenticated": int(auth),
                "anonymous": int(anon),
            }
            for name, count, auth, anon in by_event
        ],
        by_day=[{"day": str(day), "count": int(count)} for day, count in by_day],
        by_model=[
            {"model": model, "count": count}
            for model, count in sorted(by_model.items(), key=lambda kv: kv[1], reverse=True)
        ],
        attribution=_analytics_attribution_summary(
            [
                {"account_ref": account_ref, "user_id": user_id, "attribution": attribution}
                for account_ref, user_id, attribution in attribution_rows
            ]
        ),
        events_24h=int(events_24h or 0),
        active_accounts_24h=len(active_accounts_24h),
        active_accounts_7d=len(active_accounts_7d),
    )


@app.get(
    "/analytics/events/recent",
    tags=["System"],
    response_model=RecentAnalyticsEventsResponse,
    summary="Recent privacy-preserving product activity feed",
)
async def recent_analytics_events(
    request: Request,
    limit: int = Query(20, ge=1, le=100, description="Max events to return"),
) -> RecentAnalyticsEventsResponse:
    """Return the most recent product events without exposing PII."""
    _check_api_key(request)
    if _get_datastore() is not None:
        try:
            events = await asyncio.get_running_loop().run_in_executor(
                None, _recent_analytics_events_datastore, limit
            )
            return RecentAnalyticsEventsResponse(events=events)
        except Exception:
            logger.warning("datastore recent events failed; falling back to duckdb", exc_info=True)
    events = await asyncio.get_running_loop().run_in_executor(
        None, _recent_analytics_events_duckdb, limit
    )
    return RecentAnalyticsEventsResponse(events=events)


@app.get(
    "/analytics/users",
    tags=["System"],
    response_model=RegisteredUsersResponse,
    summary="List registered users and verified email IDs for operators",
)
async def list_registered_users(
    request: Request,
    limit: int = Query(500, ge=1, le=1000, description="Max users to return"),
) -> RegisteredUsersResponse:
    """Return the list of registered accounts and verified email IDs for operators."""
    _check_api_key(request)
    lim = int(limit) if isinstance(limit, (int, str)) and str(limit).isdigit() else 500
    if _get_datastore() is not None:
        try:
            users = await asyncio.get_running_loop().run_in_executor(
                None, _get_registered_users_datastore, lim
            )
            return RegisteredUsersResponse(total=len(users), users=users)
        except Exception:
            logger.warning("datastore users list failed; falling back to duckdb", exc_info=True)
    users = await asyncio.get_running_loop().run_in_executor(
        None, _get_registered_users_duckdb, lim
    )
    return RegisteredUsersResponse(total=len(users), users=users)


@app.get(
    "/analytics/export",
    tags=["System"],
    summary="Export aggregate privacy-preserving analytics summary and registered users in CSV format",
)
async def export_analytics_csv(request: Request) -> Response:
    """Export 30-day aggregate traffic, event counts, and registered users in CSV format."""
    _check_api_key(request)
    visits_task = analytics_summary(request)
    events_task = analytics_events_summary(request)
    users_task = list_registered_users(request, limit=500)
    visits, events, users = await asyncio.gather(visits_task, events_task, users_task)

    lines = ["# Foresea Analytics Summary Export", ""]
    lines.append("Metric,Value")
    lines.append(f"Total Visits (30d),{visits.total_visits}")
    lines.append(f"Unique Visitors (30d),{visits.unique_visitors}")
    lines.append(f"Visits (24h),{visits.visits_24h}")
    lines.append(f"Total Events (30d),{events.total_events}")
    lines.append(f"Events (24h),{events.events_24h}")
    lines.append(f"Active Accounts (24h),{events.active_accounts_24h}")
    lines.append(f"Active Accounts (7d),{events.active_accounts_7d}")
    lines.append(f"Total Registered Users,{events.attribution.total_registered_users}")
    lines.append(f"Authenticated Records (30d),{events.attribution.authenticated_records}")
    lines.append(f"Anonymous Records (30d),{events.attribution.anonymous_records}")
    lines.append("")
    lines.append("Day,Visits,Unique Visitors")
    for row in visits.by_day:
        lines.append(f"{row.get('day')},{row.get('visits', 0)},{row.get('unique_visitors', 0)}")
    lines.append("")
    lines.append("Event Name,Total Count,Authenticated,Anonymous")
    for row in events.by_event:
        lines.append(f"{row.get('event_name')},{row.get('count', 0)},{row.get('authenticated', 0)},{row.get('anonymous', 0)}")
    lines.append("")
    lines.append("Model,Forecast Count")
    for row in events.by_model:
        lines.append(f"{row.get('model')},{row.get('count', 0)}")
    lines.append("")
    lines.append("User ID,Email,Name,Created At,Last Login")
    for u in users.users:
        lines.append(f'"{u.user_id}","{u.email}","{u.name}","{u.created_at or ""}","{u.last_login or ""}"')

    csv_content = "\n".join(lines)
    filename = f"foresea_analytics_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/analytics/dashboard", response_class=HTMLResponse, tags=["System"], summary="Live Operator Activity & Attribution Dashboard")
async def analytics_dashboard(request: Request) -> HTMLResponse:
    """Operator dashboard for real-time Foresea activity and attribution monitoring."""
    _check_api_key(request)
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Foresea • Activity & Attribution Desk</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔮</text></svg>">
  <style>
    :root {
      --bg: #090d16;
      --card: #111827;
      --card-hover: #172033;
      --border: #1f2937;
      --border-subtle: #2d3748;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --text-dim: #6b7280;
      --accent: #6366f1;
      --emerald: #10b981;
      --emerald-dim: rgba(16, 185, 129, 0.15);
      --amber: #f59e0b;
      --indigo: #6366f1;
      --indigo-dim: rgba(99, 102, 241, 0.15);
      --slate: #64748b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 28px 20px 60px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    .title-group { display: flex; align-items: center; gap: 12px; }
    .logo-badge {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, #4f46e5, #06b6d4);
      border-radius: 9px;
      display: grid;
      place-items: center;
      font-weight: 700;
      font-size: 18px;
      color: white;
    }
    h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
    .subtitle { font-size: 13px; color: var(--text-muted); }
    .controls { display: flex; align-items: center; gap: 12px; }
    .live-indicator {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--emerald-dim);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
    }
    .pulse-dot {
      width: 7px;
      height: 7px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .btn-refresh {
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 14px;
      border-radius: 7px;
      font-size: 13px;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .btn-refresh:hover { background: var(--card-hover); border-color: var(--border-subtle); }
    .grid-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .stat-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .stat-card:hover { border-color: var(--border-subtle); }
    .stat-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); font-weight: 600; margin-bottom: 6px; }
    .stat-val { font-size: 30px; font-weight: 700; letter-spacing: -0.03em; color: var(--text); margin-bottom: 6px; line-height: 1.1; }
    .stat-sub { font-size: 13px; color: var(--text-muted); }
    .badge-auth { color: var(--emerald); font-weight: 600; }
    .badge-anon { color: var(--slate); font-weight: 600; }
    
    .grid-panels {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }
    @media (max-width: 900px) { .grid-panels { grid-template-columns: 1fr; } }
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .panel-title { font-size: 15px; font-weight: 600; color: var(--text); }
    .panel-sub { font-size: 12px; color: var(--text-dim); }

    /* Funnel Flow */
    .funnel-step {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 8px;
    }
    .funnel-name { font-size: 13px; font-weight: 500; }
    .funnel-counts { display: flex; align-items: center; gap: 12px; }
    .funnel-count { font-size: 14px; font-weight: 700; }
    .funnel-conv { font-size: 12px; color: #a5b4fc; background: var(--indigo-dim); padding: 2px 7px; border-radius: 4px; font-weight: 600; }

    /* Event Table */
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { text-align: left; padding: 8px 10px; color: var(--text-dim); font-weight: 600; font-size: 11px; text-transform: uppercase; border-bottom: 1px solid var(--border); }
    td { padding: 10px 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); }
    tr:last-child td { border-bottom: none; }
    .event-bar-wrap { width: 100%; height: 6px; background: #334155; border-radius: 3px; overflow: hidden; display: flex; margin-top: 4px; }
    .bar-auth { background: var(--emerald); height: 100%; }
    .bar-anon { background: var(--slate); height: 100%; }

    /* Activity Stream */
    .activity-list { list-style: none; display: flex; flex-direction: column; gap: 8px; max-height: 290px; overflow-y: auto; }
    .activity-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 13px;
    }
    .activity-tag {
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
    }
    .tag-auth { background: var(--emerald-dim); color: #34d399; }
    .tag-anon { background: rgba(100, 116, 139, 0.2); color: var(--slate); }

    /* Privacy Banner */
    .privacy-banner {
      background: rgba(16, 185, 129, 0.05);
      border: 1px solid rgba(16, 185, 129, 0.2);
      border-radius: 10px;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      gap: 14px;
      margin-top: 24px;
      font-size: 13px;
      color: #a7f3d0;
    }
    .privacy-icon { font-size: 20px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title-group">
        <div class="logo-badge">🔮</div>
        <div>
          <h1>Foresea Activity Desk</h1>
          <div class="subtitle">Real-time user engagement & privacy-preserving attribution telemetry</div>
        </div>
      </div>
      <div class="controls">
        <div class="live-indicator"><span class="pulse-dot"></span> Live Telemetry</div>
        <a href="/analytics/export" class="btn-refresh">📥 Export CSV</a>
        <button class="btn-refresh" id="refreshBtn" onclick="loadTelemetry()">Refresh</button>
      </div>
    </div>

    <!-- Stat Cards -->
    <div class="grid-stats">
      <div class="stat-card">
        <div class="stat-label">Registered Accounts</div>
        <div class="stat-val" id="registeredAccounts">--</div>
        <div class="stat-sub"><span id="activeAccounts24h">--</span> active in 24h / <span id="activeAccounts7d">--</span> in 7d</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Events Velocity (24h)</div>
        <div class="stat-val" id="events24h">--</div>
        <div class="stat-sub"><span id="totalEvents">--</span> total product events (30d)</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Visits Velocity (24h)</div>
        <div class="stat-val" id="visits24h">--</div>
        <div class="stat-sub"><span id="totalVisits">--</span> visits (<span id="uniqueVisitors">--</span> unique)</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Authenticated Cohort</div>
        <div class="stat-val" id="authRatio">--%</div>
        <div class="stat-sub"><span class="badge-auth" id="authRecords">--</span> signed-in / <span class="badge-anon" id="anonRecords">--</span> anon</div>
      </div>
    </div>

    <!-- Main Content Panels -->
    <div class="grid-panels">
      <!-- Funnel Flow -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Forecasting & Product Funnel</div>
            <div class="panel-sub">Trailing 30-day conversion flow</div>
          </div>
        </div>
        <div id="funnelFlow">
          <div class="funnel-step">
            <span class="funnel-name">1. Page Visits</span>
            <div class="funnel-counts"><span class="funnel-count" id="fnVisits">--</span><span class="funnel-conv">100%</span></div>
          </div>
          <div class="funnel-step">
            <span class="funnel-name">2. Forecast Started</span>
            <div class="funnel-counts"><span class="funnel-count" id="fnStarted">--</span><span class="funnel-conv" id="fnConvStarted">--%</span></div>
          </div>
          <div class="funnel-step">
            <span class="funnel-name">3. Forecast Completed</span>
            <div class="funnel-counts"><span class="funnel-count" id="fnCompleted">--</span><span class="funnel-conv" id="fnConvCompleted">--%</span></div>
          </div>
          <div class="funnel-step">
            <span class="funnel-name">4. Desk / Radar Opened</span>
            <div class="funnel-counts"><span class="funnel-count" id="fnDesk">--</span><span class="funnel-conv" id="fnConvDesk">--%</span></div>
          </div>
          <div class="funnel-step">
            <span class="funnel-name">5. Watchlist / Ledger Actions</span>
            <div class="funnel-counts"><span class="funnel-count" id="fnActions">--</span><span class="funnel-conv" id="fnConvActions">--%</span></div>
          </div>
        </div>
      </div>

      <!-- Events Breakdown -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Product Event Attribution</div>
            <div class="panel-sub">Signed-in (<span style="color:var(--emerald)">■</span>) vs Anonymous (<span style="color:var(--slate)">■</span>)</div>
          </div>
        </div>
        <div style="max-height: 290px; overflow-y: auto;">
          <table>
            <thead>
              <tr>
                <th>Event</th>
                <th style="text-align:right">Volume</th>
                <th style="text-align:right">Split</th>
              </tr>
            </thead>
            <tbody id="eventsTableBody">
              <tr><td colspan="3" style="text-align:center; color:var(--text-dim)">Loading events...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Row 2 Panels: Activity Stream & Model Preference -->
    <div class="grid-panels">
      <!-- Live Recent Activity Stream -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Live Activity Stream</div>
            <div class="panel-sub">Most recent privacy-preserving product events</div>
          </div>
        </div>
        <div id="recentActivityList" class="activity-list">
          <div style="text-align:center; color:var(--text-dim); padding: 16px;">Loading recent events...</div>
        </div>
      </div>

      <!-- Model Preference -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Model Preference & Utilization</div>
            <div class="panel-sub">Forecast model selection share</div>
          </div>
        </div>
        <div style="max-height: 290px; overflow-y: auto;">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th style="text-align:right">Forecasts</th>
                <th style="text-align:right">Share</th>
              </tr>
            </thead>
            <tbody id="modelsTableBody">
              <tr><td colspan="3" style="text-align:center; color:var(--text-dim)">Loading models...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Row 3 Panel: Registered User Accounts & Email Directory -->
    <div class="panel" style="margin-bottom: 24px;">
      <div class="panel-header">
        <div>
          <div class="panel-title">Registered User Directory</div>
          <div class="panel-sub">Verified accounts created via Google One-Tap / OAuth</div>
        </div>
        <div id="userCountBadge" class="panel-sub" style="font-weight:600; color:var(--emerald);">-- users</div>
      </div>
      <div style="max-height: 320px; overflow-y: auto;">
        <table>
          <thead>
            <tr>
              <th>User</th>
              <th>Verified Email</th>
              <th style="text-align:right">Registered Date</th>
              <th style="text-align:right">Last Active</th>
            </tr>
          </thead>
          <tbody id="usersTableBody">
            <tr><td colspan="4" style="text-align:center; color:var(--text-dim)">Loading registered users...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Privacy Banner -->
    <div class="privacy-banner">
      <span class="privacy-icon">🛡️</span>
      <div>
        <strong>Operator Privacy Notice:</strong> User email directory is available exclusively on operator-authenticated dashboard endpoints. Public forecast and search streams remain strictly zero-PII with HMAC-SHA256 account reference separation.
      </div>
    </div>
  </div>

  <script>
    function esc(s) {
      if (!s) return '';
      return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    async function loadTelemetry() {
      const btn = document.getElementById('refreshBtn');
      if (btn) btn.innerText = 'Refreshing...';
      try {
        const [vRes, eRes, rRes, uRes] = await Promise.all([
          fetch('/analytics/summary'),
          fetch('/analytics/events/summary'),
          fetch('/analytics/events/recent?limit=15'),
          fetch('/analytics/users?limit=200')
        ]);
        if (!vRes.ok || !eRes.ok) throw new Error('Analytics endpoints unreachable');
        const visits = await vRes.json();
        const events = await eRes.json();
        const recents = rRes.ok ? await rRes.json() : { events: [] };
        const usersData = uRes.ok ? await uRes.json() : { total: 0, users: [] };

        // 1. Stats
        const regTotal = events.attribution && events.attribution.total_registered_users != null
          ? events.attribution.total_registered_users
          : (usersData.total || 0);
        document.getElementById('registeredAccounts').innerText = regTotal.toLocaleString();
        document.getElementById('activeAccounts24h').innerText = (events.active_accounts_24h || 0).toLocaleString();
        document.getElementById('activeAccounts7d').innerText = (events.active_accounts_7d || events.active_accounts_24h || 0).toLocaleString();
        document.getElementById('events24h').innerText = (events.events_24h || 0).toLocaleString();
        document.getElementById('totalEvents').innerText = (events.total_events || 0).toLocaleString();
        document.getElementById('visits24h').innerText = (visits.visits_24h || 0).toLocaleString();
        document.getElementById('totalVisits').innerText = (visits.total_visits || 0).toLocaleString();
        document.getElementById('uniqueVisitors').innerText = (visits.unique_visitors || 0).toLocaleString();

        const attr = events.attribution || {};
        const authCount = attr.authenticated_records || 0;
        const anonCount = attr.anonymous_records || 0;
        const totalAttr = authCount + anonCount;
        const ratio = totalAttr > 0 ? Math.round((authCount / totalAttr) * 100) : 0;
        document.getElementById('authRatio').innerText = ratio + '%';
        document.getElementById('authRecords').innerText = authCount.toLocaleString() + ' auth';
        document.getElementById('anonRecords').innerText = anonCount.toLocaleString() + ' anon';

        // 2. Funnel
        const evMap = {};
        (events.by_event || []).forEach(e => { evMap[e.event_name] = e.count; });
        const vCount = visits.total_visits || 1;
        const fnStarted = (evMap['forecast_started'] || 0) + (evMap['first_forecast'] || 0);
        const fnCompleted = evMap['forecast_completed'] || 0;
        const fnDesk = (evMap['track_record_opened'] || 0) + (evMap['edge_board_opened'] || 0);
        const fnActions = (evMap['watchlist_add'] || 0) + (evMap['personal_ledger_add'] || 0) + (evMap['share_created'] || 0);

        document.getElementById('fnVisits').innerText = vCount.toLocaleString();
        document.getElementById('fnStarted').innerText = fnStarted.toLocaleString();
        document.getElementById('fnConvStarted').innerText = Math.round((fnStarted / vCount) * 100) + '%';
        document.getElementById('fnCompleted').innerText = fnCompleted.toLocaleString();
        document.getElementById('fnConvCompleted').innerText = (fnStarted > 0 ? Math.round((fnCompleted / fnStarted) * 100) : 0) + '%';
        document.getElementById('fnDesk').innerText = fnDesk.toLocaleString();
        document.getElementById('fnConvDesk').innerText = (vCount > 0 ? Math.round((fnDesk / vCount) * 100) : 0) + '%';
        document.getElementById('fnActions').innerText = fnActions.toLocaleString();
        document.getElementById('fnConvActions').innerText = (fnCompleted > 0 ? Math.round((fnActions / fnCompleted) * 100) : 0) + '%';

        // 3. Events Table
        const tbody = document.getElementById('eventsTableBody');
        if (events.by_event && events.by_event.length) {
          tbody.innerHTML = events.by_event.map(e => {
            const auth = e.authenticated || 0;
            const anon = e.anonymous != null ? e.anonymous : (e.count - auth);
            const total = e.count || 1;
            const authPct = Math.round((auth / total) * 100);
            const anonPct = 100 - authPct;
            return `<tr>
              <td>
                <div style="font-weight:600">${esc(e.event_name)}</div>
                <div class="event-bar-wrap">
                  <div class="bar-auth" style="width:${authPct}%"></div>
                  <div class="bar-anon" style="width:${anonPct}%"></div>
                </div>
              </td>
              <td style="text-align:right; font-weight:700">${e.count.toLocaleString()}</td>
              <td style="text-align:right; font-size:12px; color:var(--text-muted)">
                <span class="badge-auth">${auth}</span> / <span class="badge-anon">${anon}</span>
              </td>
            </tr>`;
          }).join('');
        } else {
          tbody.innerHTML = '<tr><td colspan="3" style="text-align:center; color:var(--text-dim)">No events recorded yet</td></tr>';
        }

        // 4. Live Recent Activity Stream
        const streamContainer = document.getElementById('recentActivityList');
        if (recents.events && recents.events.length) {
          streamContainer.innerHTML = recents.events.map(ev => {
            const isAuth = ev.attribution === 'authenticated';
            const tagClass = isAuth ? 'tag-auth' : 'tag-anon';
            const tagLabel = isAuth ? 'Signed In' : 'Anonymous';
            const tsStr = ev.ts ? new Date(ev.ts).toLocaleTimeString() : '';
            return `<div class="activity-item">
              <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-weight:600">${esc(ev.event_name)}</span>
                <span class="activity-tag ${tagClass}">${tagLabel}</span>
              </div>
              <span style="color:var(--text-dim); font-size:12px;">${tsStr}</span>
            </div>`;
          }).join('');
        } else {
          streamContainer.innerHTML = '<div style="text-align:center; color:var(--text-dim); padding: 16px;">No recent events recorded yet</div>';
        }

        // 5. Model Preference Table
        const modelTbody = document.getElementById('modelsTableBody');
        if (events.by_model && events.by_model.length) {
          const totalModelForecasts = events.by_model.reduce((acc, m) => acc + (m.count || 0), 0) || 1;
          modelTbody.innerHTML = events.by_model.map(m => {
            const mCount = m.count || 0;
            const pct = Math.round((mCount / totalModelForecasts) * 100);
            return `<tr>
              <td>
                <div style="font-weight:600">${esc(m.model)}</div>
                <div class="event-bar-wrap">
                  <div style="background:#6366f1; width:${pct}%; height:100%;"></div>
                </div>
              </td>
              <td style="text-align:right; font-weight:700">${mCount.toLocaleString()}</td>
              <td style="text-align:right; font-size:12px; color:#a5b4fc; font-weight:600">${pct}%</td>
            </tr>`;
          }).join('');
        } else {
          modelTbody.innerHTML = '<tr><td colspan="3" style="text-align:center; color:var(--text-dim)">No model forecasts recorded yet</td></tr>';
        }

        // 6. Registered User Directory Table
        const userTbody = document.getElementById('usersTableBody');
        const countBadge = document.getElementById('userCountBadge');
        if (usersData.users && usersData.users.length) {
          countBadge.innerText = usersData.users.length + ' account' + (usersData.users.length === 1 ? '' : 's');
          userTbody.innerHTML = usersData.users.map(u => {
            const nameStr = u.name || 'Anonymous User';
            const emailStr = u.email || '<span style="color:var(--text-dim)">No email recorded</span>';
            const cDate = u.created_at ? new Date(u.created_at).toLocaleDateString() : '--';
            const lDate = u.last_login ? new Date(u.last_login).toLocaleString() : '--';
            return `<tr>
              <td>
                <div style="display:flex; align-items:center; gap:8px;">
                  <div style="width:26px; height:26px; border-radius:50%; background:var(--indigo-dim); color:#a5b4fc; display:grid; place-items:center; font-weight:700; font-size:11px;">${esc(nameStr.charAt(0).toUpperCase())}</div>
                  <div style="font-weight:600">${esc(nameStr)}</div>
                </div>
              </td>
              <td><span style="font-family:monospace; color:#60a5fa; font-size:12px;">${esc(emailStr)}</span></td>
              <td style="text-align:right; color:var(--text-muted); font-size:12px;">${cDate}</td>
              <td style="text-align:right; color:var(--emerald); font-size:12px; font-weight:500;">${lDate}</td>
            </tr>`;
          }).join('');
        } else {
          countBadge.innerText = '0 accounts';
          userTbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-dim)">No registered users found</td></tr>';
        }
      } catch (err) {
        console.error(err);
      } finally {
        if (btn) btn.innerText = 'Refresh';
      }
    }

    loadTelemetry();
    setInterval(loadTelemetry, 10000);
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content, media_type="text/html")


@app.post("/forecasts/share", tags=["System"], response_model=SharedForecastResponse)
async def share_forecast(req: SharedForecastRequest, request: Request) -> SharedForecastResponse:
    """Create a public, intentionally shared forecast page."""
    response = await asyncio.get_running_loop().run_in_executor(None, _store_shared_forecast, req, request)
    try:
        await asyncio.get_running_loop().run_in_executor(
            None,
            _record_analytics_event,
            AnalyticsEventRequest(event_name="share_created", path="/forecasts/share", metadata={"share_id": response.share_id}),
            request,
        )
    except Exception:
        pass
    return response


@app.post("/feedback", tags=["System"], summary="Submit user feedback")
async def submit_feedback(fb: FeedbackRequest, request: Request) -> Dict[str, str]:
    """Accept feedback from the UI and forward it by email."""
    stars = f"{fb.rating}/5" if fb.rating else "unrated"
    reply_to = fb.email or "anonymous"
    page = fb.page or "/"
    ip = request.client.host if request.client else "unknown"
    subject = f"[Foresea feedback] {stars} from {reply_to}"
    body = (
        f"Rating: {stars}\n"
        f"From: {reply_to}\n"
        f"Page: {page}\n"
        f"IP: {ip}\n\n"
        f"{fb.message}"
    )
    threading.Thread(target=_send_alert_email, args=(subject, body), daemon=True).start()
    logger.info("feedback received rating=%s from=%s", fb.rating, reply_to)
    return {"status": "ok"}


_MARKET_CACHE_TTL = int(os.environ.get("MARKET_CACHE_TTL", "30"))


async def _fetch_market_quote(
    venue: str, *, force_refresh: bool = False, **kwargs: Any
) -> "MarketQuote":
    """Shared market fetch; execution guardrails may explicitly bypass the cache."""
    from analyzing_llm_rationale import market_data

    cache_key = _cache_key("market", venue, kwargs)
    cached = None if force_refresh else _cache_get(cache_key)
    if cached is not None:
        return MarketQuote(**cached)

    loop = asyncio.get_running_loop()
    try:
        if venue == "polymarket":
            quote = await loop.run_in_executor(
                None,
                lambda: market_data.fetch_polymarket(
                    slug=kwargs.get("slug"), market_id=kwargs.get("market_id")
                ),
            )
        else:
            quote = await loop.run_in_executor(
                None, lambda: market_data.fetch_kalshi(kwargs["ticker"])
            )
    except market_data.MarketDataError as exc:
        code = 404 if "not found" in str(exc).lower() else 502
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Market provider error: {exc}") from exc

    # Stamp before caching so a cache hit still reports when the data was
    # actually pulled from the venue, not when it was served -- the staleness
    # signal callers (e.g. /market/batch) need to make their own trust call.
    quote["fetched_at"] = datetime.now(timezone.utc).isoformat()
    _cache_set(cache_key, quote, _MARKET_CACHE_TTL)
    return MarketQuote(**quote)


@app.get("/markets/polymarket", tags=["Markets"], summary="Fetch a Polymarket market", response_model=MarketQuote)
async def market_polymarket(
    request: Request,
    slug: Optional[str] = None,
    id: Optional[str] = None,
) -> "MarketQuote":
    """Fetch a live Polymarket quote by market `slug` (or numeric `id`).

    The result is normalised so `probability` can be passed to `/predict` as
    `market_probability` (with `market_platform="Polymarket"`) to compute an edge.
    """
    _check_rate_limit(request)
    if not slug and not id:
        raise HTTPException(status_code=422, detail="Provide a Polymarket 'slug' or 'id'.")
    return await _fetch_market_quote("polymarket", slug=slug, market_id=id)


@app.get("/markets/kalshi", tags=["Markets"], summary="Fetch a Kalshi market", response_model=MarketQuote)
async def market_kalshi(request: Request, ticker: str) -> "MarketQuote":
    """Fetch a live Kalshi quote by market `ticker` (e.g. `KXFED-26SEP-C`)."""
    _check_rate_limit(request)
    if not ticker.strip():
        raise HTTPException(status_code=422, detail="Provide a Kalshi 'ticker'.")
    return await _fetch_market_quote("kalshi", ticker=ticker)


class BatchQuoteItem(BaseModel):
    platform: str
    ident: str
    question: Optional[str] = None
    probability: Optional[float] = None
    market_url: Optional[str] = None
    volume: Optional[float] = None
    close_time: Optional[str] = None
    order_book: Optional[Dict[str, Any]] = Field(
        None, description="Present only when the ref included its extra id (Kalshi series_ticker / Polymarket token_id)."
    )
    candles: Optional[List[Dict[str, Any]]] = Field(
        None, description="Present only when the ref included its extra id."
    )
    fetched_at: Optional[str] = Field(None, description="ISO 8601 UTC time the price/volume fields were fetched.")
    age_seconds: Optional[float] = Field(None, description="How old fetched_at is, in seconds -- the trust signal.")
    error: Optional[str] = Field(None, description="Set (with other fields empty) when this one ref failed.")


class BatchQuoteResponse(BaseModel):
    quotes: List[BatchQuoteItem]
    count: int
    truncated: bool = Field(False, description="True if more than _MAX_BATCH_REFS refs were requested; the extras were dropped, not silently merged.")


_MAX_BATCH_REFS = 50


def _parse_batch_ref(raw: str) -> tuple[str, str, Optional[str]]:
    """"platform:ident[:extra]" -> (platform, ident, extra). extra is the Kalshi
    series_ticker or Polymarket CLOB token_id that unlocks order book/candles;
    Kalshi tickers and Polymarket slugs/token ids never contain ":"."""
    parts = raw.split(":", 2)
    platform = (parts[0] if parts else "").strip().lower()
    ident = parts[1].strip() if len(parts) > 1 else ""
    extra = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    return platform, ident, extra


async def _one_batch_quote(platform: str, ident: str) -> "BatchQuoteItem":
    item = BatchQuoteItem(platform=platform, ident=ident)
    if not ident:
        item.error = f"missing market identifier for platform {platform!r}"
        return item
    try:
        if platform == "kalshi":
            quote = await _fetch_market_quote("kalshi", ticker=ident)
        elif platform == "polymarket":
            quote = await _fetch_market_quote("polymarket", slug=ident)
        else:
            item.error = 'platform must be "kalshi" or "polymarket"'
            return item
    except HTTPException as exc:
        item.error = str(exc.detail)
        return item
    item.question = quote.question
    item.probability = quote.probability
    item.market_url = quote.market_url
    item.volume = quote.volume
    item.close_time = quote.close_time
    item.fetched_at = quote.fetched_at
    fetched = _parse_trading_timestamp(quote.fetched_at)
    if fetched:
        item.age_seconds = max(0.0, (datetime.now(timezone.utc) - fetched).total_seconds())
    return item


@app.get("/market/batch", tags=["Markets"], summary="Current state for N markets in one call", response_model=BatchQuoteResponse)
async def market_batch(
    request: Request,
    refs: List[str] = Query(  # noqa: B008
        ...,
        description='Repeat this param per market: "platform:ident" or "platform:ident:extra" '
                    '(extra = Kalshi series_ticker or Polymarket CLOB token_id, adds order book + candles). '
                    "Up to 50 refs.",
    ),
) -> "BatchQuoteResponse":
    """Fetch current price/volume for up to 50 markets in one call, so a watchlist
    check doesn't need N sequential tool calls. Every quote carries `fetched_at`
    and `age_seconds` -- the freshness signal to make your own trust call on,
    since both venues rate-limit hard and a quote can otherwise look fresh when
    it isn't. One bad ref returns an error on that entry only; the rest of the
    batch still succeeds."""
    _check_rate_limit(request)
    truncated = False
    if len(refs) > _MAX_BATCH_REFS:
        refs = refs[:_MAX_BATCH_REFS]
        truncated = True

    parsed = [_parse_batch_ref(r) for r in refs]
    items = await asyncio.gather(*(_one_batch_quote(p, i) for p, i, _e in parsed))
    by_key: Dict[tuple[str, str], BatchQuoteItem] = {(it.platform, it.ident): it for it in items}

    # Depth enrichment: one marketd call carrying every ref that supplied an
    # extra id -- marketd fans these out concurrently itself, so this is one
    # round trip total, not one per market.
    depth_refs = [f"{p}:{i}:{e}" for p, i, e in parsed if e and (p, i) in by_key and not by_key[(p, i)].error]
    if depth_refs:
        loop = asyncio.get_running_loop()
        depth_quotes = await loop.run_in_executor(None, _marketd_quotes_sync, depth_refs)
        for dq in depth_quotes or []:
            if not isinstance(dq, dict) or dq.get("error"):
                continue  # depth failed but the base quote is still useful -- don't overwrite it
            key = (str(dq.get("platform") or "").lower(), str(dq.get("ident") or ""))
            item = by_key.get(key)
            if item is None:
                continue
            item.order_book = dq.get("order_book")
            item.candles = dq.get("candles")

    return BatchQuoteResponse(quotes=list(by_key.values()), count=len(by_key), truncated=truncated)


@app.get("/radar", tags=["Markets"], summary="Live Foresea market radar", response_model=RadarResponse)
async def radar(limit: int = Query(12, ge=1, le=30)) -> JSONResponse:
    """Return a cached list of live markets with notable model-vs-market gaps."""
    payload = await asyncio.get_running_loop().run_in_executor(None, _radar_from_track_record, limit)
    if payload.markets:
        _spawn_background(_prefetch_radar_evidence(payload.markets))
    return JSONResponse(
        payload.model_dump(mode="json"),
        headers={"Cache-Control": "no-cache, max-age=0, must-revalidate"},
    )


@app.get("/market/exchange-status", tags=["Markets"], summary="Operational status and schedule of exchanges")
async def market_exchange_status() -> Dict[str, Any]:
    """Get Kalshi and prediction exchange operational status and schedule."""
    from analyzing_llm_rationale import market_data as _md

    loop = asyncio.get_running_loop()
    status = await loop.run_in_executor(None, _md.fetch_kalshi_exchange_status)
    schedule = await loop.run_in_executor(None, _md.fetch_kalshi_exchange_schedule)
    return {"status": status, "schedule": schedule}


@app.get("/market/orderbook", tags=["Markets"], summary="Fetch real-time orderbook bids and asks")
async def market_orderbook_route(
    platform: str = Query("kalshi", description="Venue: 'kalshi' or 'polymarket'"),
    ident: str = Query(..., description="Market ticker or token_id"),
) -> Dict[str, Any]:
    """Fetch live bids/asks orderbook depth."""
    from analyzing_llm_rationale import market_data as _md

    loop = asyncio.get_running_loop()
    if "poly" in platform.lower():
        ob = await loop.run_in_executor(None, lambda: _md.fetch_polymarket_orderbook(ident))
    else:
        ob = await loop.run_in_executor(None, lambda: _md.fetch_kalshi_orderbook(ident))
    return ob or {"error": "Orderbook unavailable", "platform": platform, "ident": ident}


@app.get("/market/trades", tags=["Markets"], summary="Recent executed trade tape")
async def market_trades_route(
    platform: str = Query("kalshi", description="Venue: 'kalshi' or 'polymarket'"),
    ident: str = Query(..., description="Market ticker or token_id"),
    limit: int = Query(20, ge=1, le=100, description="Max trades to return"),
) -> Dict[str, Any]:
    """Fetch executed prints/trade tape."""
    from analyzing_llm_rationale import market_data as _md

    loop = asyncio.get_running_loop()
    trades = await loop.run_in_executor(None, lambda: _md.fetch_recent_trades(platform, ident, limit))
    return trades or {"trades": [], "platform": platform, "ident": ident}


@app.get("/market/leaderboard", tags=["Markets"], summary="Top trader rankings and volume leaders")
async def market_leaderboard_route(
    limit: int = Query(20, ge=1, le=100, description="Max leaders to return"),
) -> Dict[str, Any]:
    """Fetch trader rankings and volume leaders."""
    from analyzing_llm_rationale import market_data as _md

    loop = asyncio.get_running_loop()
    lb = await loop.run_in_executor(None, lambda: _md.fetch_trader_leaderboard(limit))
    return lb or {"leaderboard": []}


@app.get("/market/tags", tags=["Markets"], summary="Categorization tags for a market")
async def market_tags_route() -> Dict[str, Any]:
    """Fetch market tags and categorization metadata."""
    from analyzing_llm_rationale import market_data as _md

    loop = asyncio.get_running_loop()
    tags = await loop.run_in_executor(None, _md.fetch_polymarket_tags)
    return tags or {"tags": []}


@app.get("/market/price-history", tags=["Markets"], summary="Candlestick OHLCV price history")
async def market_price_history_route(
    platform: str = Query("kalshi", description="Venue: 'kalshi' or 'polymarket'"),
    ident: str = Query(..., description="Market ticker or token_id"),
    series_ticker: Optional[str] = Query(None, description="Series ticker for Kalshi"),
) -> Dict[str, Any]:
    """Fetch historical OHLCV candlestick bars."""
    from analyzing_llm_rationale import market_data as _md

    loop = asyncio.get_running_loop()
    if "poly" in platform.lower():
        candles = await loop.run_in_executor(None, lambda: _md.fetch_polymarket_price_history(ident))
    else:
        candles = await loop.run_in_executor(None, lambda: _md.fetch_kalshi_candlesticks(ident, series_ticker))
    return candles or {"candlesticks": []}


_TRADING_CONNECTION_KIND = "TradingConnection"
_TRADING_ORDER_KIND = "TradingOrder"
_TRADING_RUN_KIND = "TradingRun"
_TRADING_GUARDRAILS_KIND = "TradingGuardrails"
_TRADING_RISK_EVENT_KIND = "TradingRiskEvent"
_TRADING_ORDER_FIELDS = (
    "id",
    "trade_run_id",
    "platform",
    "venue_order_id",
    "status",
    "venue_status",
    "action",
    "outcome",
    "ticker",
    "token_id",
    "quantity",
    "price",
    "estimated_notional",
    "order_type",
    "subaccount",
    "exchange_index",
    "filled_quantity",
    "remaining_quantity",
    "created_at",
    "updated_at",
    "last_reconciled_at",
    "canceled_at",
)
_TRADING_RUN_FIELDS = (
    "id",
    "status",
    "title",
    "platform",
    "action",
    "outcome",
    "ticker",
    "token_id",
    "quantity",
    "price",
    "estimated_notional",
    "order_type",
    "client_order_id",
    "order_request",
    "preview",
    "provenance",
    "audit_order_id",
    "venue_order_id",
    "reconciliation_status",
    "approved_at",
    "submitted_at",
    "last_reconciled_at",
    "created_at",
    "updated_at",
    "error_code",
)
_TRADING_GUARDRAIL_FIELDS = (
    "paused",
    "max_order_notional",
    "max_daily_risk_notional",
    "max_market_exposure_notional",
    "max_open_orders",
    "max_price_deviation_bps",
    "max_quote_age_seconds",
    "cooldown_seconds",
    "updated_at",
)
_TRADING_RISK_EVENT_FIELDS = (
    "id",
    "created_at",
    "venue",
    "event",
    "outcome",
    "reason_code",
    "trade_run_id",
    "audit_order_id",
)


class SecureTradingConnectionError(RuntimeError):
    """The server is unable to safely persist or read a connected account."""


class TradingGuardrailError(RuntimeError):
    """A real-order request crossed a deliberate Foresea risk boundary."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_TRADING_KMS_KEY_ENV = "FORESEA_TRADING_KMS_KEY_NAME"
_TRADING_KMS_KEY_PATTERN = re.compile(
    r"^projects/[^/]+/locations/[^/]+/keyRings/[^/]+/cryptoKeys/[^/]+$"
)
_TRADING_CONNECTION_ENVELOPE_VERSION = 2
_TRADING_LEGACY_ENCRYPTION_KEY_ENV = "FORESEA_CREDENTIALS_ENCRYPTION_KEY"


def _clean_trading_platform(value: Any) -> str:
    platform = str(value or "").strip().lower()
    if platform in {"kalshi"}:
        return "kalshi"
    if platform in {"poly", "polymarket"}:
        return "polymarket"
    raise HTTPException(status_code=422, detail="platform must be 'kalshi' or 'polymarket'.")


def _trading_connection_key(client: Any, user_id: str, platform: str) -> Any:
    return client.key("User", user_id, _TRADING_CONNECTION_KIND, platform)


def _trading_order_key(client: Any, user_id: str, order_id: str) -> Any:
    return client.key("User", user_id, _TRADING_ORDER_KIND, order_id)


def _trading_run_key(client: Any, user_id: str, run_id: str) -> Any:
    return client.key("User", user_id, _TRADING_RUN_KIND, run_id)


def _iso_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _trading_kms_key_name() -> str:
    """Return the dedicated Cloud KMS key name without accepting a fallback."""
    raw_name = (os.environ.get(_TRADING_KMS_KEY_ENV) or "").strip()
    if not raw_name:
        raise SecureTradingConnectionError(
            f"Secure exchange connections are unavailable until {_TRADING_KMS_KEY_ENV} is configured."
        )
    if not _TRADING_KMS_KEY_PATTERN.fullmatch(raw_name):
        raise SecureTradingConnectionError(
            f"{_TRADING_KMS_KEY_ENV} must be a Cloud KMS CryptoKey resource name."
        )
    return raw_name


def _get_trading_kms_client() -> Any:
    """Create one ADC-authenticated KMS client for envelope-key operations."""
    global _trading_kms_client
    if _trading_kms_client is not None:
        return _trading_kms_client
    try:
        from google.cloud import kms_v1

        _trading_kms_client = kms_v1.KeyManagementServiceClient()
    except Exception as exc:
        raise SecureTradingConnectionError(
            "Cloud KMS is unavailable. Check the deployment dependency and service-account permissions."
        ) from exc
    return _trading_kms_client


def _trading_kms_aad(user_id: str, platform: str) -> bytes:
    """Bind a wrapped data key to exactly one user and venue without storing secrets."""
    return f"foresea:trading-connection:v2:{user_id}:{platform}".encode("utf-8")


def _legacy_trading_fernet():
    """Load only the retired root key needed while migrating version-1 records."""
    raw_key = (os.environ.get(_TRADING_LEGACY_ENCRYPTION_KEY_ENV) or "").strip()
    if not raw_key:
        raise SecureTradingConnectionError(
            "This legacy exchange connection requires its retired encryption key to migrate. "
            "Restore it temporarily or have the user reconnect."
        )
    try:
        from cryptography.fernet import Fernet

        return Fernet(raw_key.encode("utf-8"))
    except Exception as exc:
        raise SecureTradingConnectionError(
            f"{_TRADING_LEGACY_ENCRYPTION_KEY_ENV} is invalid; this connection must be reconnected."
        ) from exc


def _secure_connection_configured() -> bool:
    try:
        _trading_kms_key_name()
    except SecureTradingConnectionError:
        return False
    return True


def _trading_launch_readiness() -> TradingLaunchReadinessResponse:
    """Report deploy-local prerequisites without exercising KMS or an exchange.

    A green report means this revision is configured for an invite-only account
    connection beta. It deliberately does *not* claim that Cloud KMS IAM,
    GitHub's mirrored scheduler secret, or a venue account have been proven;
    those require their own least-privilege smoke checks.
    """
    from analyzing_llm_rationale import trading

    checks: List[TradingLaunchReadinessCheck] = []

    def add(code: str, status: Literal["ready", "attention", "blocked"], detail: str) -> None:
        checks.append(TradingLaunchReadinessCheck(code=code, status=status, detail=detail))

    secure_connections = _secure_connection_configured()
    add(
        "secure_connections",
        "ready" if secure_connections else "blocked",
        (
            "A dedicated Cloud KMS key resource is configured for encrypted user connections."
            if secure_connections
            else "A valid Cloud KMS key resource is required before user exchange accounts can connect."
        ),
    )

    durable_store = _get_datastore() is not None
    add(
        "durable_store",
        "ready" if durable_store else "blocked",
        (
            "A durable store client is available for trade runs, audit orders, and guardrail events."
            if durable_store
            else "A durable Datastore client is required; in-memory trade state is not launch-safe."
        ),
    )

    scheduler_configured = bool(_TRADING_RECONCILIATION_TOKEN)
    add(
        "scheduled_reconciliation",
        "ready" if scheduler_configured else "blocked",
        (
            "The service has a reconciliation token; confirm the matching GitHub Actions secret separately."
            if scheduler_configured
            else "Set the reconciliation token in the service and its matching GitHub Actions secret."
        ),
    )

    try:
        hard_caps = _platform_trading_guardrail_caps()
        caps_valid = True
        add("hard_caps", "ready", "Server-side order, exposure, freshness, and cooldown caps are valid.")
    except TradingGuardrailError:
        hard_caps = {}
        caps_valid = False
        add("hard_caps", "blocked", "One or more server-side trading cap values are invalid.")

    byo_trading_enabled = trading._env_bool("FORESEA_ENABLE_BYO_TRADING", False)
    shared_trading_enabled = trading._env_bool("FORESEA_ENABLE_TRADING", False)
    market_orders_enabled = trading._env_bool("FORESEA_ALLOW_MARKET_ORDERS", False)
    kill_switch = _trading_guardrail_env_bool("FORESEA_TRADING_KILL_SWITCH", False)
    safe_default_active = not byo_trading_enabled and not shared_trading_enabled
    if safe_default_active:
        add("execution_gate", "ready", "Live execution is disabled by default.")
    elif kill_switch:
        add("execution_gate", "attention", "Execution is enabled, but the platform kill switch currently blocks new orders.")
    else:
        add("execution_gate", "attention", "Live execution is enabled; keep the rollout invite-only and within hard caps.")

    legacy_key_present = bool((os.environ.get(_TRADING_LEGACY_ENCRYPTION_KEY_ENV) or "").strip())
    add(
        "legacy_encryption_key",
        "attention" if legacy_key_present else "ready",
        (
            "A retired connection-encryption key remains for lazy migration; remove it after all version-1 records migrate."
            if legacy_key_present
            else "No retired shared connection-encryption key is configured."
        ),
    )

    connection_beta_ready = secure_connections and durable_store and scheduler_configured and caps_valid
    live_byo_beta_ready = connection_beta_ready and byo_trading_enabled and not kill_switch
    return TradingLaunchReadinessResponse(
        safe_default_active=safe_default_active,
        ready_for_connection_beta=connection_beta_ready,
        ready_for_live_byo_beta=live_byo_beta_ready,
        byo_trading_enabled=byo_trading_enabled,
        shared_trading_enabled=shared_trading_enabled,
        market_orders_enabled=market_orders_enabled,
        platform_kill_switch=kill_switch,
        scheduled_reconciliation_configured=scheduler_configured,
        durable_store_configured=durable_store,
        hard_caps=hard_caps,
        checks=checks,
    )


def _read_trading_connection(user_id: str, platform: str) -> Optional[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        record = _state.setdefault("trading_connections", {}).get(user_id, {}).get(platform)
        return dict(record) if record else None
    entity = client.get(_trading_connection_key(client, user_id, platform))
    return dict(entity) if entity is not None else None


def _connection_status(user_id: str, platform: str) -> TradingConnectionStatus:
    record = _read_trading_connection(user_id, platform)
    return TradingConnectionStatus(
        platform=platform,
        connected=bool(record and record.get("encrypted_credentials")),
        updated_at=_iso_timestamp(record.get("updated_at")) if record else None,
    )


def _put_trading_connection(
    user_id: str, platform: str, credentials: Dict[str, Any]
) -> TradingConnectionStatus:
    """Encrypt credentials with a unique DEK and wrap that DEK in Cloud KMS."""
    try:
        from cryptography.fernet import Fernet

        data_key = Fernet.generate_key()
        plaintext = json.dumps(
            {
                "version": _TRADING_CONNECTION_ENVELOPE_VERSION,
                "user_id": user_id,
                "platform": platform,
                "credentials": credentials,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        ciphertext = Fernet(data_key).encrypt(plaintext).decode("utf-8")
        kms_key_name = _trading_kms_key_name()
        wrapped_key = _get_trading_kms_client().encrypt(
            request={
                "name": kms_key_name,
                "plaintext": data_key,
                "additional_authenticated_data": _trading_kms_aad(user_id, platform),
            }
        )
    except Exception as exc:
        raise SecureTradingConnectionError(
            "Could not encrypt exchange credentials with Cloud KMS. Check the KMS key and service-account permissions."
        ) from exc
    now = datetime.now(timezone.utc)
    record = {
        "platform": platform,
        "encrypted_credentials": ciphertext,
        "wrapped_data_key": base64.urlsafe_b64encode(wrapped_key.ciphertext).decode("ascii"),
        "kms_key_name": kms_key_name,
        "kms_key_version": str(getattr(wrapped_key, "name", "") or ""),
        "updated_at": now,
        "credential_version": _TRADING_CONNECTION_ENVELOPE_VERSION,
    }
    client = _get_datastore()
    if client is None:
        _state.setdefault("trading_connections", {}).setdefault(user_id, {})[platform] = record
    else:
        from google.cloud import datastore as _ds

        entity = _ds.Entity(
            key=_trading_connection_key(client, user_id, platform),
            exclude_from_indexes=("encrypted_credentials", "wrapped_data_key"),
        )
        entity.update(record)
        client.put(entity)
    return TradingConnectionStatus(platform=platform, connected=True, updated_at=now.isoformat())


def _delete_trading_connection(user_id: str, platform: str) -> None:
    client = _get_datastore()
    if client is None:
        _state.setdefault("trading_connections", {}).setdefault(user_id, {}).pop(platform, None)
        return
    client.delete(_trading_connection_key(client, user_id, platform))


def _stored_trading_credentials(user_id: str, platform: str) -> Optional[Dict[str, Any]]:
    record = _read_trading_connection(user_id, platform)
    if not record or not record.get("encrypted_credentials"):
        return None
    try:
        from cryptography.fernet import Fernet

        credential_version = int(record.get("credential_version") or 1)
        if credential_version == 1:
            # Migration keeps the previous root only long enough to rewrap this
            # connection under a new, per-record KMS-backed data key.
            legacy_plaintext = _legacy_trading_fernet().decrypt(
                str(record["encrypted_credentials"]).encode("utf-8")
            )
            legacy_credentials = json.loads(legacy_plaintext.decode("utf-8"))
            if not isinstance(legacy_credentials, dict):
                raise SecureTradingConnectionError(
                    f"The {platform} connection is malformed. Disconnect and reconnect it."
                )
            _put_trading_connection(user_id, platform, legacy_credentials)
            return legacy_credentials
        if credential_version != _TRADING_CONNECTION_ENVELOPE_VERSION:
            raise SecureTradingConnectionError(
                f"The {platform} connection uses an unsupported encryption version. Disconnect and reconnect it."
            )
        wrapped_data_key = base64.urlsafe_b64decode(
            str(record["wrapped_data_key"]).encode("ascii")
        )
        data_key = _get_trading_kms_client().decrypt(
            request={
                "name": str(record.get("kms_key_name") or _trading_kms_key_name()),
                "ciphertext": wrapped_data_key,
                "additional_authenticated_data": _trading_kms_aad(user_id, platform),
            }
        ).plaintext
        plaintext = Fernet(data_key).decrypt(str(record["encrypted_credentials"]).encode("utf-8"))
        envelope = json.loads(plaintext.decode("utf-8"))
    except SecureTradingConnectionError:
        raise
    except Exception as exc:
        raise SecureTradingConnectionError(
            f"The {platform} connection can no longer be decrypted. Disconnect and reconnect it."
        ) from exc
    if (
        not isinstance(envelope, dict)
        or envelope.get("version") != _TRADING_CONNECTION_ENVELOPE_VERSION
        or envelope.get("user_id") != user_id
        or envelope.get("platform") != platform
        or not isinstance(envelope.get("credentials"), dict)
    ):
        raise SecureTradingConnectionError(
            f"The {platform} connection is malformed. Disconnect and reconnect it."
        )
    return envelope["credentials"]


def _trading_order_from_entity(entity: Any) -> Dict[str, Any]:
    record = {field: entity.get(field) for field in _TRADING_ORDER_FIELDS}
    record["id"] = record.get("id") or entity.key.name
    return record


def _list_trading_orders(user_id: str, platform: Optional[str] = None) -> List[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        records = list(_state.setdefault("trading_orders", {}).get(user_id, {}).values())
    else:
        query = client.query(kind=_TRADING_ORDER_KIND, ancestor=client.key("User", user_id))
        records = [_trading_order_from_entity(entity) for entity in query.fetch(limit=250)]
    if platform:
        records = [record for record in records if record.get("platform") == platform]
    return sorted(records, key=lambda record: record.get("created_at") or "", reverse=True)


_RECONCILABLE_TRADING_ORDER_STATUSES = ("submitted", "open")


def _list_reconcilable_trading_orders(limit: int) -> List[tuple[str, Dict[str, Any]]]:
    """Return a bounded cross-account set of non-terminal audited orders.

    Credentials are deliberately not joined here.  The reconciliation worker
    decrypts each user's connection only for the venue being read, and never
    returns account, order, or credential identifiers to its scheduler.
    """
    bounded_limit = max(1, min(int(limit), _TRADING_RECONCILIATION_MAX_ORDERS))
    client = _get_datastore()
    if client is None:
        candidates = [
            (user_id, dict(record))
            for user_id, records in _state.setdefault("trading_orders", {}).items()
            for record in records.values()
            if record.get("status") in _RECONCILABLE_TRADING_ORDER_STATUSES
            and record.get("venue_order_id")
        ]
        return sorted(candidates, key=lambda item: item[1].get("updated_at") or "")[:bounded_limit]

    query = client.query(kind=_TRADING_ORDER_KIND)
    query.add_filter("status", "IN", list(_RECONCILABLE_TRADING_ORDER_STATUSES))
    # The composite index in index.yaml makes this a fair, oldest-first queue
    # rather than repeatedly reading whichever live orders sort first by key.
    query.order = ["last_reconciled_at"]
    candidates: List[tuple[str, Dict[str, Any]]] = []
    for entity in query.fetch(limit=bounded_limit):
        parent = entity.key.parent
        user_id = parent.name if parent is not None else None
        if not user_id or not entity.get("venue_order_id"):
            continue
        candidates.append((str(user_id), _trading_order_from_entity(entity)))
    return candidates


def _read_trading_order(user_id: str, order_id: str) -> Optional[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        record = _state.setdefault("trading_orders", {}).get(user_id, {}).get(order_id)
        return dict(record) if record else None
    entity = client.get(_trading_order_key(client, user_id, order_id))
    return _trading_order_from_entity(entity) if entity is not None else None


def _put_trading_order(user_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    record = {field: record.get(field) for field in _TRADING_ORDER_FIELDS}
    client = _get_datastore()
    if client is None:
        _state.setdefault("trading_orders", {}).setdefault(user_id, {})[record["id"]] = record
        return record
    from google.cloud import datastore as _ds

    entity = _ds.Entity(
        key=_trading_order_key(client, user_id, record["id"]),
        exclude_from_indexes=("ticker", "token_id"),
    )
    entity.update(record)
    client.put(entity)
    return record


def _trading_run_from_entity(entity: Any) -> Dict[str, Any]:
    record = {field: entity.get(field) for field in _TRADING_RUN_FIELDS}
    record["id"] = record.get("id") or entity.key.name
    return record


def _list_trading_runs(user_id: str, platform: Optional[str] = None) -> List[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        records = list(_state.setdefault("trading_runs", {}).get(user_id, {}).values())
    else:
        query = client.query(kind=_TRADING_RUN_KIND, ancestor=client.key("User", user_id))
        records = [_trading_run_from_entity(entity) for entity in query.fetch(limit=250)]
    if platform:
        records = [record for record in records if record.get("platform") == platform]
    return sorted(records, key=lambda record: record.get("created_at") or "", reverse=True)


def _read_trading_run(user_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        record = _state.setdefault("trading_runs", {}).get(user_id, {}).get(run_id)
        return dict(record) if record else None
    entity = client.get(_trading_run_key(client, user_id, run_id))
    return _trading_run_from_entity(entity) if entity is not None else None


def _put_trading_run(user_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    record = {field: record.get(field) for field in _TRADING_RUN_FIELDS}
    client = _get_datastore()
    if client is None:
        _state.setdefault("trading_runs", {}).setdefault(user_id, {})[record["id"]] = record
        return record
    from google.cloud import datastore as _ds

    entity = _ds.Entity(
        key=_trading_run_key(client, user_id, record["id"]),
        exclude_from_indexes=("order_request", "preview", "provenance"),
    )
    entity.update(record)
    client.put(entity)
    return record


def _trading_guardrails_key(client: Any, user_id: str) -> Any:
    return client.key("User", user_id, _TRADING_GUARDRAILS_KIND, "current")


def _trading_risk_event_key(client: Any, user_id: str, event_id: str) -> Any:
    return client.key("User", user_id, _TRADING_RISK_EVENT_KIND, event_id)


def _trading_guardrail_env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise TradingGuardrailError("invalid_platform_config", f"{name} must be numeric.") from exc
    if not math.isfinite(value) or value <= 0:
        raise TradingGuardrailError("invalid_platform_config", f"{name} must be greater than zero.")
    return value


def _trading_guardrail_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _trading_guardrail_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise TradingGuardrailError("invalid_platform_config", f"{name} must be an integer.") from exc
    if value < minimum:
        raise TradingGuardrailError("invalid_platform_config", f"{name} must be at least {minimum}.")
    return value


def _platform_trading_guardrail_caps() -> Dict[str, Any]:
    """Hard ceilings. Users may choose lower limits but never raise these caps."""
    from analyzing_llm_rationale import trading

    max_order = float(trading._max_order_notional())
    return {
        "max_order_notional": max_order,
        "max_daily_risk_notional": _trading_guardrail_env_float(
            "FORESEA_MAX_DAILY_RISK_NOTIONAL", max(max_order, 100.0)
        ),
        "max_market_exposure_notional": _trading_guardrail_env_float(
            "FORESEA_MAX_MARKET_EXPOSURE_NOTIONAL", max(max_order, 50.0)
        ),
        "max_open_orders": _trading_guardrail_env_int("FORESEA_MAX_OPEN_ORDERS", 5),
        "max_price_deviation_bps": _trading_guardrail_env_int(
            "FORESEA_MAX_PRICE_DEVIATION_BPS", 300
        ),
        "max_quote_age_seconds": _trading_guardrail_env_int("FORESEA_MAX_QUOTE_AGE_SECONDS", 20),
        "cooldown_seconds": _trading_guardrail_env_int(
            "FORESEA_ORDER_COOLDOWN_SECONDS", 60, minimum=0
        ),
    }


def _default_trading_guardrails() -> Dict[str, Any]:
    caps = _platform_trading_guardrail_caps()
    return {"paused": False, **caps, "updated_at": None}


def _read_trading_guardrails(user_id: str) -> Optional[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        record = _state.setdefault("trading_guardrails", {}).get(user_id)
        return dict(record) if record else None
    entity = client.get(_trading_guardrails_key(client, user_id))
    if entity is None:
        return None
    return {field: entity.get(field) for field in _TRADING_GUARDRAIL_FIELDS}


def _effective_trading_guardrails(user_id: str) -> Dict[str, Any]:
    policy = _default_trading_guardrails()
    stored = _read_trading_guardrails(user_id)
    if stored:
        policy.update({field: stored.get(field) for field in _TRADING_GUARDRAIL_FIELDS if stored.get(field) is not None})
    caps = _platform_trading_guardrail_caps()
    for field, hard_cap in caps.items():
        policy[field] = min(policy[field], hard_cap)
    return policy


def _put_trading_guardrails(user_id: str, policy: Dict[str, Any]) -> Dict[str, Any]:
    record = {field: policy.get(field) for field in _TRADING_GUARDRAIL_FIELDS}
    client = _get_datastore()
    if client is None:
        _state.setdefault("trading_guardrails", {})[user_id] = record
        return record
    from google.cloud import datastore as _ds

    entity = _ds.Entity(key=_trading_guardrails_key(client, user_id))
    entity.update(record)
    client.put(entity)
    return record


def _update_trading_guardrails(user_id: str, update: TradingGuardrailsUpdateRequest) -> Dict[str, Any]:
    caps = _platform_trading_guardrail_caps()
    changes = update.model_dump(exclude_none=True)
    for field, value in changes.items():
        if field == "paused":
            continue
        if value > caps[field]:
            raise TradingGuardrailError(
                "limit_above_platform_cap",
                f"{field} may not exceed Foresea's hard ceiling of {caps[field]}.",
            )
    with _trading_guardrail_lock:
        policy = _effective_trading_guardrails(user_id)
        policy.update(changes)
        policy["updated_at"] = datetime.now(timezone.utc).isoformat()
        return _put_trading_guardrails(user_id, policy)


def _trading_guardrails_response(user_id: str) -> TradingGuardrailsResponse:
    policy = _effective_trading_guardrails(user_id)
    return TradingGuardrailsResponse(
        **policy,
        platform_kill_switch=_trading_guardrail_env_bool("FORESEA_TRADING_KILL_SWITCH", False),
    )


def _record_trading_risk_event(
    user_id: str,
    *,
    venue: str,
    event: str,
    outcome: str,
    reason_code: Optional[str] = None,
    trade_run_id: Optional[str] = None,
    audit_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a safe, immutable audit record without order payloads or secrets."""
    record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "venue": venue,
        "event": event,
        "outcome": outcome,
        "reason_code": reason_code,
        "trade_run_id": trade_run_id,
        "audit_order_id": audit_order_id,
    }
    client = _get_datastore()
    if client is None:
        events = _state.setdefault("trading_risk_events", {}).setdefault(user_id, [])
        events.append(record)
        del events[:-250]
        return record
    from google.cloud import datastore as _ds

    entity = _ds.Entity(key=_trading_risk_event_key(client, user_id, record["id"]))
    entity.update(record)
    client.put(entity)
    return record


def _list_trading_risk_events(user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    bounded_limit = max(1, min(int(limit), 250))
    client = _get_datastore()
    if client is None:
        records = list(_state.setdefault("trading_risk_events", {}).get(user_id, []))
    else:
        query = client.query(kind=_TRADING_RISK_EVENT_KIND, ancestor=client.key("User", user_id))
        records = [
            {field: entity.get(field) for field in _TRADING_RISK_EVENT_FIELDS}
            for entity in query.fetch(limit=bounded_limit)
        ]
    return sorted(records, key=lambda record: record.get("created_at") or "", reverse=True)[:bounded_limit]


def _notify_trading_safety_event(
    *, venue: str, event: str, reason_code: str, trade_run_id: Optional[str] = None
) -> None:
    """Alert an operator only for states that require human follow-up."""
    if event not in {"submission_unknown", "venue_rejected", "platform_kill_switch", "order_filled"}:
        return
    logger.error("trading safety event venue=%s event=%s reason=%s", venue, event, reason_code)
    try:
        threading.Thread(
            target=_send_alert_email,
            args=(
                f"Foresea trading safety: {event}",
                "\n".join(
                    part
                    for part in (
                        f"Venue: {venue}",
                        f"Event: {event}",
                        f"Reason: {reason_code}",
                        f"Trade run: {trade_run_id}" if trade_run_id else "",
                    )
                    if part
                ),
            ),
            daemon=True,
        ).start()
    except Exception:
        logger.exception("could not enqueue trading safety alert")


def _record_terminal_trading_order_event(
    user_id: str, *, venue: str, record: Dict[str, Any], previous_status: Any
) -> None:
    status = str(record.get("status") or "").lower()
    if status not in {"filled", "canceled", "rejected"} or status == str(previous_status or "").lower():
        return
    event = {"filled": "order_filled", "canceled": "order_canceled", "rejected": "venue_rejected"}[status]
    _record_trading_risk_event(
        user_id,
        venue=venue,
        event=event,
        outcome=status,
        reason_code=status,
        trade_run_id=record.get("trade_run_id"),
        audit_order_id=record.get("id"),
    )
    _notify_trading_safety_event(
        venue=venue, event=event, reason_code=status, trade_run_id=record.get("trade_run_id")
    )


def _parse_trading_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _trading_market_key(payload: Dict[str, Any], normalized: Dict[str, Any]) -> str:
    venue = _clean_trading_platform(normalized.get("platform") or payload.get("platform"))
    identifier = normalized.get("ticker") or normalized.get("token_id")
    if not identifier:
        identifier = payload.get("ticker") or payload.get("token_id") or payload.get("slug") or payload.get("market_id")
    cleaned = str(identifier or "").strip().lower()
    if not cleaned:
        raise TradingGuardrailError("market_identifier_missing", "A stable market identifier is required for risk checks.")
    return f"{venue}:{cleaned}"


def _trading_order_market_key(record: Dict[str, Any]) -> str:
    try:
        return _trading_market_key(record, record)
    except TradingGuardrailError:
        # Older audit rows may predate stable market identifiers. They cannot
        # safely contribute to a market-specific match, but still count in the
        # global daily/open-order limits.
        return ""


def _trading_risk_usage(user_id: str, market_key: str, now: datetime) -> Dict[str, float]:
    risk_window_start = now - timedelta(hours=24)
    daily_risk = 0.0
    open_orders = 0.0
    local_market_exposure = 0.0
    for order in _list_trading_orders(user_id):
        status = str(order.get("status") or "").lower()
        notional = float(order.get("estimated_notional") or 0.0)
        created_at = _parse_trading_timestamp(order.get("created_at"))
        if status not in {"rejected", "canceled"} and created_at and created_at >= risk_window_start:
            daily_risk += notional
        if status in {"submitted", "open"}:
            open_orders += 1
            if _trading_order_market_key(order) == market_key:
                local_market_exposure += notional
    for run in _list_trading_runs(user_id):
        status = str(run.get("status") or "").lower()
        if status in {"submitting", "submission_unknown"}:
            open_orders += 1
            if _trading_order_market_key(run) == market_key:
                local_market_exposure += float(run.get("estimated_notional") or 0.0)
    return {
        "daily_risk_notional": round(daily_risk, 6),
        "open_orders": open_orders,
        "local_market_exposure_notional": round(local_market_exposure, 6),
    }


def _has_recent_duplicate_trade(
    user_id: str,
    *,
    market_key: str,
    action: str,
    outcome: str,
    now: datetime,
    cooldown_seconds: int,
    exclude_run_id: Optional[str] = None,
) -> bool:
    if cooldown_seconds <= 0:
        return False
    threshold = now - timedelta(seconds=cooldown_seconds)
    candidate_records = [
        *(_list_trading_orders(user_id)),
        *(_list_trading_runs(user_id)),
    ]
    for record in candidate_records:
        if exclude_run_id and record.get("id") == exclude_run_id:
            continue
        if str(record.get("status") or "").lower() in {"blocked", "rejected", "canceled"}:
            continue
        if str(record.get("action") or "").lower() != action or str(record.get("outcome") or "").lower() != outcome:
            continue
        if _trading_order_market_key(record) != market_key:
            continue
        created_at = _parse_trading_timestamp(record.get("created_at"))
        if created_at and created_at >= threshold:
            return True
    return False


def _quote_probability_for_outcome(quote: MarketQuote, outcome: str) -> Optional[float]:
    desired = outcome.strip().lower()
    for option in quote.outcomes:
        if option.label.strip().lower() == desired and option.probability is not None:
            return float(option.probability)
    if quote.outcome.strip().lower() == desired and quote.probability is not None:
        return float(quote.probability)
    if desired == "no" and quote.outcome.strip().lower() == "yes" and quote.probability is not None:
        return 1.0 - float(quote.probability)
    return None


async def _fresh_trade_guard_quote(payload: Dict[str, Any], normalized: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a no-cache quote immediately before live execution; failures block safely."""
    venue = _clean_trading_platform(normalized.get("platform") or payload.get("platform"))
    if venue == "kalshi":
        ticker = str(normalized.get("ticker") or payload.get("ticker") or "").strip()
        if not ticker:
            raise TradingGuardrailError("market_identifier_missing", "Kalshi ticker is required for a live quote check.")
        quote = await _fetch_market_quote("kalshi", ticker=ticker, force_refresh=True)
    else:
        slug = str(payload.get("slug") or "").strip() or None
        market_id = str(payload.get("market_id") or "").strip() or None
        if not (slug or market_id):
            raise TradingGuardrailError(
                "live_quote_identifier_required",
                "Polymarket live execution requires the market slug or market_id, not only a token ID.",
            )
        quote = await _fetch_market_quote("polymarket", slug=slug, market_id=market_id, force_refresh=True)
    probability = _quote_probability_for_outcome(quote, str(normalized.get("outcome") or payload.get("outcome") or "yes"))
    if probability is None or not (0.0 < probability < 1.0):
        raise TradingGuardrailError("quote_unpriced", "The selected outcome has no actionable live quote.")
    return {
        "outcome_probability": probability,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "market_ident": quote.ident,
    }


def _portfolio_available_usd(snapshot: Dict[str, Any]) -> Optional[float]:
    balance = snapshot.get("balance") if isinstance(snapshot.get("balance"), dict) else {}
    value = balance.get("available")
    if not isinstance(value, (int, float)):
        return None
    return float(value) / 100.0 if str(balance.get("unit") or "").lower() == "cents" else float(value)


def _portfolio_market_exposure(snapshot: Dict[str, Any], normalized: Dict[str, Any]) -> float:
    venue = _clean_trading_platform(normalized.get("platform"))
    identifier = str(normalized.get("ticker") if venue == "kalshi" else normalized.get("token_id") or "")
    exposure = 0.0
    for position in snapshot.get("positions") or []:
        if not isinstance(position, dict):
            continue
        position_id = str(position.get("ticker") if venue == "kalshi" else position.get("token_id") or "")
        if position_id != identifier:
            continue
        raw_value = position.get("exposure") if venue == "kalshi" else position.get("current_value")
        if isinstance(raw_value, (int, float)):
            exposure += abs(float(raw_value))
    return exposure


async def _validate_live_trade_guardrails(
    user_id: str,
    *,
    payload: Dict[str, Any],
    preview: Dict[str, Any],
    credentials: Dict[str, Any],
    trade_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Fail closed on policy, quote, portfolio, exposure, and duplicate checks."""
    now = datetime.now(timezone.utc)
    normalized = dict(preview.get("normalized_order") or {})
    venue = _clean_trading_platform(normalized.get("platform") or payload.get("platform"))
    policy = _effective_trading_guardrails(user_id)
    if _trading_guardrail_env_bool("FORESEA_TRADING_KILL_SWITCH", False):
        raise TradingGuardrailError("platform_kill_switch", "Trading is temporarily paused by Foresea's platform kill switch.")
    if policy["paused"]:
        raise TradingGuardrailError("user_paused", "Trading is paused in your risk controls. Resume it before submitting a new order.")
    estimated_notional = float(preview.get("estimated_notional") or 0.0)
    if estimated_notional <= 0:
        raise TradingGuardrailError("invalid_notional", "The order did not produce a positive risk notional.")
    if estimated_notional > float(policy["max_order_notional"]):
        raise TradingGuardrailError(
            "max_order_notional",
            f"Order risk ${estimated_notional:.2f} exceeds your ${float(policy['max_order_notional']):.2f} per-order limit.",
        )
    market_key = _trading_market_key(payload, normalized)
    if _has_recent_duplicate_trade(
        user_id,
        market_key=market_key,
        action=str(normalized.get("action") or ""),
        outcome=str(normalized.get("outcome") or ""),
        now=now,
        cooldown_seconds=int(policy["cooldown_seconds"]),
        exclude_run_id=trade_run_id,
    ):
        raise TradingGuardrailError(
            "duplicate_cooldown",
            f"An equivalent order is already active or was created within the {int(policy['cooldown_seconds'])}-second cooldown.",
        )

    quote = await _fresh_trade_guard_quote(payload, normalized)
    quote_at = _parse_trading_timestamp(quote.get("fetched_at"))
    quote_age_seconds = (datetime.now(timezone.utc) - quote_at).total_seconds() if quote_at else float("inf")
    if quote_age_seconds > int(policy["max_quote_age_seconds"]):
        raise TradingGuardrailError("stale_quote", "The live market quote became stale before the order could be submitted.")
    current_price = float(quote["outcome_probability"])
    if current_price <= 0.0:
        raise TradingGuardrailError("invalid_quote_price", "The live market quote has a non-positive probability and cannot be traded.")
    limit_price = float(normalized.get("price") or 0.0)
    action = str(normalized.get("action") or "buy")
    adverse_move = ((limit_price - current_price) / current_price) if action == "buy" else ((current_price - limit_price) / current_price)
    adverse_bps = max(0.0, adverse_move * 10_000.0)
    if adverse_bps > float(policy["max_price_deviation_bps"]):
        raise TradingGuardrailError(
            "price_deviation",
            f"Limit price is {adverse_bps:.0f} bps worse than the fresh quote; your cap is {int(policy['max_price_deviation_bps'])} bps.",
        )

    from analyzing_llm_rationale import trading

    snapshot = await asyncio.get_running_loop().run_in_executor(
        None, lambda: trading.reconcile_portfolio(venue, credentials, limit=100)
    )
    available = _portfolio_available_usd(snapshot)
    if available is None:
        raise TradingGuardrailError("balance_unavailable", "Available exchange balance could not be confirmed; no order was sent.")
    if available + 1e-9 < estimated_notional:
        raise TradingGuardrailError(
            "insufficient_available_balance",
            f"Available balance ${available:.2f} is below this order's ${estimated_notional:.2f} worst-case notional.",
        )
    usage = _trading_risk_usage(user_id, market_key, now)
    if usage["daily_risk_notional"] + estimated_notional > float(policy["max_daily_risk_notional"]):
        raise TradingGuardrailError(
            "daily_risk_limit",
            "This order would exceed your trailing-day worst-case risk budget.",
        )
    if usage["open_orders"] + 1 > int(policy["max_open_orders"]):
        raise TradingGuardrailError(
            "max_open_orders",
            f"This would exceed your limit of {int(policy['max_open_orders'])} outstanding orders.",
        )
    market_exposure = _portfolio_market_exposure(snapshot, normalized) + usage["local_market_exposure_notional"]
    if market_exposure + estimated_notional > float(policy["max_market_exposure_notional"]):
        raise TradingGuardrailError(
            "market_exposure_limit",
            "This order would exceed your per-market exposure cap after current positions and open orders.",
        )
    return {
        "policy": {
            key: policy[key]
            for key in (
                "max_order_notional",
                "max_daily_risk_notional",
                "max_market_exposure_notional",
                "max_open_orders",
                "max_price_deviation_bps",
                "max_quote_age_seconds",
                "cooldown_seconds",
            )
        },
        "quote": {**quote, "age_seconds": round(quote_age_seconds, 3), "adverse_bps": round(adverse_bps, 2)},
        "portfolio": {
            "available": round(available, 6),
            "market_exposure_notional": round(market_exposure, 6),
        },
        "usage": usage,
    }


def _new_trading_run(
    req: TradeRunCreateRequest,
    payload: Dict[str, Any],
    preview: Dict[str, Any],
) -> Dict[str, Any]:
    """Capture a validated order plan before a human can send it to a venue."""
    normalized = dict(preview.get("normalized_order") or {})
    run_id = str(uuid.uuid4())
    client_order_id = str(payload.get("client_order_id") or f"foresea-run-{run_id}")
    payload = {**payload, "client_order_id": client_order_id}
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": run_id,
        "status": "awaiting_approval",
        "title": req.title.strip() or f"{normalized.get('action', 'buy').title()} {normalized.get('outcome', 'yes').upper()}",
        "platform": normalized.get("platform"),
        "action": normalized.get("action"),
        "outcome": normalized.get("outcome"),
        "ticker": normalized.get("ticker"),
        "token_id": normalized.get("token_id"),
        "quantity": normalized.get("quantity"),
        "price": normalized.get("price"),
        "estimated_notional": preview.get("estimated_notional"),
        "order_type": normalized.get("order_type"),
        "client_order_id": client_order_id,
        "order_request": payload,
        "preview": preview,
        "provenance": {
            "thesis": req.thesis.strip(),
            "source_conversation_id": req.source_conversation_id,
            "expected_edge": req.expected_edge,
            "sources": [source.strip() for source in req.sources if source.strip()],
        },
        "audit_order_id": None,
        "venue_order_id": None,
        "reconciliation_status": None,
        "approved_at": None,
        "submitted_at": None,
        "last_reconciled_at": None,
        "created_at": now,
        "updated_at": now,
        "error_code": None,
    }


def _claim_trading_run_for_execution(
    user_id: str, run_id: str, preview: Dict[str, Any]
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Atomically acquire a saved run for one exchange-submission attempt.

    The Datastore transaction is the cross-instance idempotency boundary.  The
    in-memory lock gives development and tests the same single-process guarantee.
    """
    now = datetime.now(timezone.utc).isoformat()

    def claim(record: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        if record.get("status") != "awaiting_approval":
            return record, False
        record["preview"] = preview
        record["estimated_notional"] = preview.get("estimated_notional")
        record["status"] = "submitting"
        record["approved_at"] = now
        record["updated_at"] = now
        record["error_code"] = None
        return record, True

    client = _get_datastore()
    if client is None:
        with _trading_run_lock:
            record = _state.setdefault("trading_runs", {}).get(user_id, {}).get(run_id)
            if record is None:
                return None, False
            claimed, acquired = claim(dict(record))
            if acquired:
                _state["trading_runs"][user_id][run_id] = claimed
            return claimed, acquired

    with client.transaction():
        entity = client.get(_trading_run_key(client, user_id, run_id))
        if entity is None:
            return None, False
        record, acquired = claim(_trading_run_from_entity(entity))
        if acquired:
            _put_trading_run(user_id, record)
        return record, acquired


def _trade_run_status_from_order(order_status: Any) -> str:
    status = str(order_status or "").strip().lower()
    if status in {"filled", "canceled", "rejected"}:
        return status
    return "submitted"


def _sync_trade_run_from_order(user_id: str, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Project venue reconciliation back onto its parent run, when one exists."""
    run_id = str(order.get("trade_run_id") or "").strip()
    if not run_id:
        return None
    run = _read_trading_run(user_id, run_id)
    if run is None:
        return None
    run["audit_order_id"] = order.get("id") or run.get("audit_order_id")
    run["venue_order_id"] = order.get("venue_order_id") or run.get("venue_order_id")
    run["reconciliation_status"] = order.get("status") or run.get("reconciliation_status")
    run["last_reconciled_at"] = order.get("last_reconciled_at") or run.get("last_reconciled_at")
    run["status"] = _trade_run_status_from_order(order.get("status"))
    run["updated_at"] = datetime.now(timezone.utc).isoformat()
    run["error_code"] = None
    return _put_trading_run(user_id, run)


def _status_from_venue(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"filled", "executed", "matched", "complete", "completed"}:
        return "filled"
    if raw in {"canceled", "cancelled", "cancel"}:
        return "canceled"
    if raw in {"rejected", "failed", "error", "unmatched"}:
        return "rejected"
    if raw in {"resting", "live", "open", "active", "pending", "delayed"}:
        return "open"
    return "submitted"


def _submitted_trading_order(
    result: Dict[str, Any], *, trade_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Keep a local audit row without persisting exchange responses or secrets."""
    normalized = result.get("normalized_order") or {}
    venue_response = result.get("venue_response") or {}
    source = venue_response.get("body") if isinstance(venue_response, dict) else {}
    if isinstance(source, list):
        source = source[0] if source and isinstance(source[0], dict) else {}
    if not isinstance(source, dict):
        source = {}
    venue_order_id = next(
        (
            str(source[key])
            for key in ("order_id", "orderID", "id", "orderId")
            if source.get(key) not in (None, "")
        ),
        None,
    )
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": str(uuid.uuid4()),
        "trade_run_id": trade_run_id,
        "platform": normalized.get("platform"),
        "venue_order_id": venue_order_id,
        "status": _status_from_venue(source.get("status")),
        "venue_status": str(source.get("status") or "submitted"),
        "action": normalized.get("action"),
        "outcome": normalized.get("outcome"),
        "ticker": normalized.get("ticker"),
        "token_id": normalized.get("token_id"),
        "quantity": normalized.get("quantity"),
        "price": normalized.get("price"),
        "estimated_notional": result.get("estimated_notional"),
        "order_type": normalized.get("order_type"),
        "subaccount": normalized.get("subaccount"),
        "exchange_index": normalized.get("exchange_index"),
        "filled_quantity": None,
        "remaining_quantity": None,
        "created_at": now,
        "updated_at": now,
        "last_reconciled_at": None,
        "canceled_at": None,
    }


def _merge_order_reconciliation(record: Dict[str, Any], reconciliation: Dict[str, Any]) -> Dict[str, Any]:
    for field in (
        "venue_order_id",
        "status",
        "venue_status",
        "filled_quantity",
        "remaining_quantity",
    ):
        if reconciliation.get(field) is not None:
            record[field] = reconciliation[field]
    now = datetime.now(timezone.utc).isoformat()
    record["last_reconciled_at"] = now
    record["updated_at"] = now
    return record


def _scheduled_reconcile_open_trading_orders(limit: int) -> Dict[str, Any]:
    """Reconcile a bounded batch of submitted orders without placing trades.

    This is intentionally serial and rate-limited by batch size: it is a
    read-only venue operation that moves audit state from venue evidence only.
    """
    from analyzing_llm_rationale import trading

    result: Dict[str, Any] = {
        "checked": 0,
        "updated": 0,
        "terminal": 0,
        "connection_missing": 0,
        "errors": 0,
        "by_venue": {"kalshi": 0, "polymarket": 0},
    }
    credentials_cache: Dict[tuple[str, str], Optional[Dict[str, Any]]] = {}
    for user_id, record in _list_reconcilable_trading_orders(limit):
        try:
            venue = _clean_trading_platform(record.get("platform"))
        except Exception:
            result["errors"] += 1
            continue
        result["checked"] += 1
        result["by_venue"][venue] += 1
        cache_key = (user_id, venue)
        if cache_key not in credentials_cache:
            try:
                credentials_cache[cache_key] = _stored_trading_credentials(user_id, venue)
            except Exception:
                # Do not expose the user, order, or decrypt error to logs or the
                # scheduler response. The owner sees connection health in their UI.
                credentials_cache[cache_key] = None
        credentials = credentials_cache[cache_key]
        if credentials is None:
            result["connection_missing"] += 1
            _trading_reconciliation_actions.add(
                1, {"venue": venue, "action": "scheduled_order", "outcome": "connection_missing"}
            )
            continue
        try:
            venue_state = trading.reconcile_order(venue, str(record["venue_order_id"]), credentials)
            previous_status = record.get("status")
            record = _merge_order_reconciliation(record, venue_state)
            _put_trading_order(user_id, record)
            _sync_trade_run_from_order(user_id, record)
            _record_terminal_trading_order_event(
                user_id, venue=venue, record=record, previous_status=previous_status
            )
            result["updated"] += 1
            if record.get("status") in {"filled", "canceled", "rejected"}:
                result["terminal"] += 1
            _trading_reconciliation_actions.add(
                1, {"venue": venue, "action": "scheduled_order", "outcome": "success"}
            )
        except Exception:
            result["errors"] += 1
            _trading_reconciliation_actions.add(
                1, {"venue": venue, "action": "scheduled_order", "outcome": "error"}
            )
            logger.warning("scheduled trading reconciliation could not update one %s order", venue)
    return result


def _trading_http_exception(exc: Exception) -> HTTPException:
    from analyzing_llm_rationale import trading

    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, SecureTradingConnectionError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, TradingGuardrailError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, trading.TradingValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (trading.TradingDisabledError, trading.TradingNotConfiguredError)):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, trading.TradingExecutionError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Trading error: {exc}")


@app.get(
    "/trading/accounts",
    tags=["Trading"],
    summary="Check configured trading venues",
    response_model=TradingAccountStatus,
)
async def trading_accounts(request: Request) -> TradingAccountStatus:
    """Return configured/not-configured status for server-side exchange credentials.

    This endpoint never returns private keys, API secrets, or passphrases. Live
    order submission still requires `/trading/orders` confirmation.
    """
    _check_rate_limit(request)
    _require_session(request)
    from analyzing_llm_rationale import trading

    try:
        return TradingAccountStatus(**trading.account_status())
    except Exception as exc:
        raise _trading_http_exception(exc) from exc


@app.post(
    "/trading/accounts/check",
    tags=["Trading"],
    summary="Validate own-account (BYO) trading credentials",
    response_model=TradingAccountStatus,
)
async def trading_accounts_check(
    req: TradingAccountCheckRequest, request: Request
) -> TradingAccountStatus:
    """Report readiness for caller-supplied own-account credentials, never stored.

    Lets the UI confirm a connected Kalshi/Polymarket account (e.g. the RSA key
    parses) before placing an order. Credentials are used transiently and never
    echoed back — only boolean readiness is returned.
    """
    _check_rate_limit(request)
    _require_session(request)
    from analyzing_llm_rationale import trading

    # NB: `creds` carries funds-moving secrets — never log it.
    creds = req.venue_credentials.model_dump(exclude_none=True) if req.venue_credentials else None
    try:
        return TradingAccountStatus(**trading.account_status(creds))
    except Exception as exc:
        raise _trading_http_exception(exc) from exc


@app.get(
    "/trading/guardrails",
    tags=["Trading"],
    summary="Read the signed-in user's real-money trading guardrails",
    response_model=TradingGuardrailsResponse,
)
async def get_trading_guardrails(request: Request) -> TradingGuardrailsResponse:
    _check_rate_limit(request)
    claims = _require_session(request)
    return _trading_guardrails_response(claims["sub"])


@app.put(
    "/trading/guardrails",
    tags=["Trading"],
    summary="Narrow the signed-in user's real-money trading guardrails",
    response_model=TradingGuardrailsResponse,
)
async def update_trading_guardrails(
    req: TradingGuardrailsUpdateRequest, request: Request
) -> TradingGuardrailsResponse:
    _check_rate_limit(request)
    claims = _require_session(request)
    with _tracer.start_as_current_span("trading.guardrail.update") as span:
        span.set_attribute("trading.user_id", claims["sub"])
        try:
            policy = _update_trading_guardrails(claims["sub"], req)
            _record_trading_risk_event(
                claims["sub"], venue="all", event="guardrail_updated", outcome="success"
            )
            _trading_guardrail_actions.add(1, {"venue": "all", "action": "update", "outcome": "success"})
            return TradingGuardrailsResponse(
                **policy,
                platform_kill_switch=_trading_guardrail_env_bool("FORESEA_TRADING_KILL_SWITCH", False),
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            _trading_guardrail_actions.add(1, {"venue": "all", "action": "update", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


@app.get(
    "/trading/guardrails/events",
    tags=["Trading"],
    summary="List safe audit events for real-money trading controls",
)
async def list_trading_guardrail_events(
    request: Request, limit: int = Query(50, ge=1, le=250)
) -> Dict[str, Any]:
    _check_rate_limit(request)
    claims = _require_session(request)
    return {"events": _list_trading_risk_events(claims["sub"], limit)}


@app.get(
    "/trading/connections",
    tags=["Trading"],
    summary="List encrypted exchange connections",
    response_model=TradingConnectionsResponse,
)
async def trading_connections(request: Request) -> TradingConnectionsResponse:
    """Return only connection metadata for the signed-in user, never secrets."""
    _check_rate_limit(request)
    claims = _require_session(request)
    return TradingConnectionsResponse(
        encryption_configured=_secure_connection_configured(),
        connections={
            platform: _connection_status(claims["sub"], platform)
            for platform in ("kalshi", "polymarket")
        },
    )


@app.put(
    "/trading/connections/{platform}",
    tags=["Trading"],
    summary="Encrypt and save one exchange connection",
    response_model=TradingConnectionStatus,
)
async def save_trading_connection(
    platform: str, req: TradingConnectionRequest, request: Request
) -> TradingConnectionStatus:
    """Validate and store one account connection with server-side encryption."""
    _check_rate_limit(request)
    claims = _require_session(request)
    venue = _clean_trading_platform(platform)
    with _tracer.start_as_current_span("trading.connection.save") as span:
        span.set_attribute("trading.venue", venue)
        try:
            from analyzing_llm_rationale import trading

            # Keep this mapping in local scope only. Do not log it or attach it
            # to telemetry; it contains funds-moving secrets.
            credentials = trading.connection_credentials(
                venue, req.venue_credentials.model_dump(exclude_none=True)
            )
            status = _put_trading_connection(claims["sub"], venue, credentials)
            _trading_connection_actions.add(1, {"venue": venue, "action": "save", "outcome": "success"})
            return status
        except Exception as exc:
            span.record_exception(exc)
            _trading_connection_actions.add(1, {"venue": venue, "action": "save", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


@app.delete(
    "/trading/connections/{platform}",
    tags=["Trading"],
    summary="Delete one encrypted exchange connection",
    response_model=TradingConnectionStatus,
)
async def delete_trading_connection(platform: str, request: Request) -> TradingConnectionStatus:
    """Delete only the encrypted credential blob; trading audit history remains."""
    _check_rate_limit(request)
    claims = _require_session(request)
    venue = _clean_trading_platform(platform)
    with _tracer.start_as_current_span("trading.connection.delete") as span:
        span.set_attribute("trading.venue", venue)
        try:
            _delete_trading_connection(claims["sub"], venue)
            _trading_connection_actions.add(1, {"venue": venue, "action": "delete", "outcome": "success"})
            return TradingConnectionStatus(platform=venue, connected=False)
        except Exception as exc:
            span.record_exception(exc)
            _trading_connection_actions.add(1, {"venue": venue, "action": "delete", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


# ── BYO Model Providers Endpoints ─────────────────────────────────────────────

@app.get(
    "/api/user/model-providers",
    tags=["Model Providers"],
    summary="List safe BYO model provider connection statuses",
    response_model=UserModelProvidersResponse,
)
async def list_user_model_providers(request: Request) -> UserModelProvidersResponse:
    """Return configured model provider statuses without leaking secret API keys."""
    _check_rate_limit(request)
    claims = _require_session(request)
    from analyzing_llm_rationale import model_providers

    try:
        statuses = model_providers.get_user_provider_status_list(claims["sub"])
        return UserModelProvidersResponse(
            providers=[UserModelProviderStatus(**s.__dict__) for s in statuses],
            encryption_configured=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model provider statuses: {exc}") from exc


@app.put(
    "/api/user/model-providers/{provider_id}",
    tags=["Model Providers"],
    summary="Encrypt and save credentials for one AI model provider",
    response_model=UserModelProviderStatus,
)
async def save_user_model_provider(
    provider_id: str, req: SaveUserModelProviderRequest, request: Request
) -> UserModelProviderStatus:
    """Validate and store one model provider connection with server-side envelope encryption."""
    _check_rate_limit(request)
    claims = _require_session(request)
    from analyzing_llm_rationale import model_providers

    clean_pid = str(provider_id).strip().lower()
    if clean_pid not in model_providers.CREDIBLE_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported model provider '{provider_id}'.")

    try:
        # If user is updating model/URL without retyping the API key, reuse existing key
        api_key = req.api_key or ""
        if not api_key:
            existing = model_providers.read_user_model_provider(claims["sub"], clean_pid)
            if existing and existing.get("encrypted_secret"):
                secret = model_providers._decrypt_provider_secret(claims["sub"], clean_pid, existing)
                api_key = secret.get("api_key", "")

        status = model_providers.put_user_model_provider(
            user_id=claims["sub"],
            provider_id=clean_pid,
            api_key=api_key,
            default_model=req.default_model,
            custom_base_url=req.custom_base_url,
        )
        return UserModelProviderStatus(**status.__dict__)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete(
    "/api/user/model-providers/{provider_id}",
    tags=["Model Providers"],
    summary="Delete encrypted credentials for one AI model provider",
    response_model=UserModelProviderStatus,
)
async def delete_user_model_provider(provider_id: str, request: Request) -> UserModelProviderStatus:
    """Delete encrypted credentials for the specified provider."""
    _check_rate_limit(request)
    claims = _require_session(request)
    from analyzing_llm_rationale import model_providers

    clean_pid = str(provider_id).strip().lower()
    desc = model_providers.CREDIBLE_PROVIDERS.get(clean_pid)
    if not desc:
        raise HTTPException(status_code=400, detail=f"Unsupported model provider '{provider_id}'.")

    try:
        model_providers.delete_user_model_provider(claims["sub"], clean_pid)
        return UserModelProviderStatus(
            provider_id=desc.id,
            name=desc.name,
            category=desc.category,
            description=desc.description,
            connected=False,
            default_model=desc.default_model,
            popular_models=desc.popular_models,
            default_base_url=desc.default_base_url,
            custom_base_url=None,
            docs_url=desc.docs_url,
            updated_at=None,
            key_prefix=desc.key_prefix,
            masked_key=None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/api/user/model-providers/{provider_id}/test",
    tags=["Model Providers"],
    summary="Live-test credentials against the provider API",
    response_model=TestUserModelProviderResponse,
)
async def test_user_model_provider(
    provider_id: str, req: TestUserModelProviderRequest, request: Request
) -> TestUserModelProviderResponse:
    """Send a minimal test request to verify API key validity and measure response latency."""
    _check_rate_limit(request)
    from analyzing_llm_rationale import model_providers

    clean_pid = str(provider_id).strip().lower()
    if clean_pid not in model_providers.CREDIBLE_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported model provider '{provider_id}'.")

    api_key = req.api_key or ""
    if not api_key:
        try:
            claims = _require_session(request)
            existing = model_providers.read_user_model_provider(claims["sub"], clean_pid)
            if existing and existing.get("encrypted_secret"):
                secret = model_providers._decrypt_provider_secret(claims["sub"], clean_pid, existing)
                api_key = secret.get("api_key", "")
        except Exception:
            pass

    res = model_providers.test_provider_credentials(
        provider_id=clean_pid,
        api_key=api_key,
        model_name=req.model_name,
        custom_base_url=req.custom_base_url,
    )
    return TestUserModelProviderResponse(**res)


def _resolve_order_credentials(user_id: str, req: TradeOrderRequest) -> Optional[Dict[str, Any]]:
    """Prefer a secure per-user connection and block deprecated inline secrets."""
    if req.venue_credentials is not None:
        raise HTTPException(
            status_code=422,
            detail="Inline exchange credentials are no longer accepted. Connect the account securely first.",
        )
    platform = _clean_trading_platform(req.platform)
    return _stored_trading_credentials(user_id, platform)


@app.post(
    "/trading/preview",
    tags=["Trading"],
    summary="Preview a guarded prediction-market order",
    response_model=TradeOrderPreviewResponse,
)
async def trading_preview(req: TradeOrderRequest, request: Request) -> TradeOrderPreviewResponse:
    """Validate and normalize a Kalshi/Polymarket order without placing it."""
    _check_rate_limit(request)
    claims = _require_session(request)
    from analyzing_llm_rationale import trading

    try:
        creds = _resolve_order_credentials(claims["sub"], req)
        payload = req.model_dump(exclude_none=True, exclude={"venue_credentials"})
        return TradeOrderPreviewResponse(**trading.preview_order(payload, creds))
    except Exception as exc:
        raise _trading_http_exception(exc) from exc


@app.post(
    "/trading/runs",
    tags=["Trading"],
    summary="Create a durable, reviewable live-trade run",
    response_model=TradeRunResponse,
)
async def create_trading_run(req: TradeRunCreateRequest, request: Request) -> TradeRunResponse:
    """Save a validated order plan. This endpoint cannot submit an order."""
    _check_rate_limit(request)
    claims = _require_session(request)
    from analyzing_llm_rationale import trading

    if req.execute or req.confirmation is not None:
        raise HTTPException(
            status_code=422,
            detail="Trade runs are created without execution or confirmation. Execute the saved run explicitly.",
        )
    venue = _clean_trading_platform(req.platform)
    with _tracer.start_as_current_span("trading.run.create") as span:
        span.set_attribute("trading.venue", venue)
        try:
            credentials = _resolve_order_credentials(claims["sub"], req)
            payload = req.model_dump(
                exclude_none=True,
                exclude={
                    "title", "thesis", "source_conversation_id", "expected_edge", "sources",
                    "execute", "confirmation", "venue_credentials",
                },
            )
            preview = trading.preview_order(payload, credentials)
            policy = _effective_trading_guardrails(claims["sub"])
            if float(preview.get("estimated_notional") or 0.0) > float(policy["max_order_notional"]):
                raise TradingGuardrailError(
                    "max_order_notional",
                    f"Order risk ${float(preview['estimated_notional']):.2f} exceeds your ${float(policy['max_order_notional']):.2f} per-order limit.",
                )
            preview["guardrails"] = TradingGuardrailsResponse(
                **policy,
                platform_kill_switch=_trading_guardrail_env_bool("FORESEA_TRADING_KILL_SWITCH", False),
            ).model_dump()
            preview.setdefault("warnings", []).append(
                "Foresea rechecks a fresh quote, available balance, exposure, duplicate cooldown, and risk limits immediately before submission."
            )
            run = _put_trading_run(claims["sub"], _new_trading_run(req, payload, preview))
            span.set_attribute("trading.run_id", run["id"])
            span.set_attribute("trading.run.status", run["status"])
            _record_trading_risk_event(
                claims["sub"], venue=venue, event="trade_run_created", outcome="success", trade_run_id=run["id"]
            )
            _trading_run_actions.add(1, {"venue": venue, "action": "create", "outcome": "success"})
            logger.info("trading run created id=%s venue=%s", run["id"], venue)
            return TradeRunResponse(**run)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            _trading_run_actions.add(1, {"venue": venue, "action": "create", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


@app.get("/trading/runs", tags=["Trading"], summary="List the signed-in user's durable trade runs")
async def list_trading_runs(
    request: Request, platform: Optional[str] = Query(None, max_length=20)
) -> Dict[str, Any]:
    _check_rate_limit(request)
    claims = _require_session(request)
    venue = _clean_trading_platform(platform) if platform else None
    return {"runs": [TradeRunResponse(**run).model_dump() for run in _list_trading_runs(claims["sub"], venue)]}


@app.get("/trading/runs/{run_id}", tags=["Trading"], summary="Read one durable trade run", response_model=TradeRunResponse)
async def read_trading_run(run_id: str, request: Request) -> TradeRunResponse:
    _check_rate_limit(request)
    claims = _require_session(request)
    run = _read_trading_run(claims["sub"], run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Trade run was not found.")
    return TradeRunResponse(**run)


@app.post(
    "/trading/runs/{run_id}/execute",
    tags=["Trading"],
    summary="Execute one saved run after explicit confirmation",
    response_model=TradeRunResponse,
)
async def execute_trading_run(
    run_id: str, req: TradeRunExecuteRequest, request: Request
) -> TradeRunResponse:
    """Submit exactly one approved run and attach the resulting order audit row."""
    _check_rate_limit(request)
    claims = _require_session(request)
    from analyzing_llm_rationale import trading

    run: Optional[Dict[str, Any]] = None
    claimed_for_execution = False
    venue = "unknown"
    with _tracer.start_as_current_span("trading.run.execute") as span:
        span.set_attribute("trading.run_id", run_id)
        try:
            run = _read_trading_run(claims["sub"], run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Trade run was not found.")
            venue = _clean_trading_platform(run.get("platform"))
            span.set_attribute("trading.venue", venue)
            if run.get("status") != "awaiting_approval":
                raise HTTPException(
                    status_code=409,
                    detail=f"Trade run is {run.get('status')}; it cannot be submitted again.",
                )
            if req.confirmation != trading.CONFIRMATION_PHRASE:
                raise HTTPException(
                    status_code=422,
                    detail=f"confirmation must be exactly '{trading.CONFIRMATION_PHRASE}'.",
                )
            payload = run.get("order_request")
            if not isinstance(payload, dict):
                raise HTTPException(status_code=409, detail="Trade run does not contain a valid order plan.")
            credentials = _stored_trading_credentials(claims["sub"], venue)
            if credentials is None:
                raise HTTPException(status_code=409, detail=f"Connect a {venue} account before executing this run.")

            # Revalidate guardrails immediately before the user-approved request,
            # then atomically claim the run before touching the exchange.  A
            # second browser tab or Cloud Run instance will receive a conflict
            # rather than submitting a duplicate venue order.
            preview = trading.preview_order(payload, credentials)
            if not preview.get("trading_enabled"):
                trading.place_order(
                    {**payload, "execute": True, "confirmation": req.confirmation},
                    user_id=claims["sub"],
                    creds=credentials,
                )
            guardrail_snapshot = await _validate_live_trade_guardrails(
                claims["sub"],
                payload=payload,
                preview=preview,
                credentials=credentials,
                trade_run_id=run_id,
            )
            preview["guardrails"] = guardrail_snapshot
            run, claimed_for_execution = _claim_trading_run_for_execution(
                claims["sub"], run_id, preview
            )
            if run is None:
                raise HTTPException(status_code=404, detail="Trade run was not found.")
            if not claimed_for_execution:
                raise HTTPException(
                    status_code=409,
                    detail=f"Trade run is {run.get('status')}; it cannot be submitted again.",
                )

            result = trading.place_order(
                {**payload, "execute": True, "confirmation": req.confirmation},
                user_id=claims["sub"],
                creds=credentials,
            )
            audit = _put_trading_order(
                claims["sub"], _submitted_trading_order(result, trade_run_id=run_id)
            )
            _record_terminal_trading_order_event(
                claims["sub"], venue=venue, record=audit, previous_status="submitted"
            )
            submitted_at = datetime.now(timezone.utc).isoformat()
            run.update({
                "status": _trade_run_status_from_order(audit.get("status")),
                "audit_order_id": audit["id"],
                "venue_order_id": audit.get("venue_order_id"),
                "reconciliation_status": audit.get("status"),
                "submitted_at": submitted_at,
                "updated_at": submitted_at,
                "error_code": None,
            })
            _put_trading_run(claims["sub"], run)
            span.set_attribute("trading.audit_order_id", audit["id"])
            span.set_attribute("trading.run.status", run["status"])
            _record_trading_risk_event(
                claims["sub"],
                venue=venue,
                event="order_submitted",
                outcome="success",
                trade_run_id=run_id,
                audit_order_id=audit["id"],
            )
            _trading_guardrail_actions.add(1, {"venue": venue, "action": "execute", "outcome": "passed"})
            _trading_run_actions.add(1, {"venue": venue, "action": "execute", "outcome": "success"})
            logger.info("trading run submitted id=%s venue=%s audit_order_id=%s", run_id, venue, audit["id"])
            return TradeRunResponse(**run)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            if isinstance(exc, TradingGuardrailError):
                _record_trading_risk_event(
                    claims["sub"],
                    venue=venue,
                    event="guardrail_blocked",
                    outcome="blocked",
                    reason_code=exc.code,
                    trade_run_id=run_id,
                )
                _trading_guardrail_actions.add(1, {"venue": venue, "action": "execute", "outcome": "blocked"})
                _notify_trading_safety_event(
                    venue=venue, event=exc.code, reason_code=exc.code, trade_run_id=run_id
                )
            if claimed_for_execution and run is not None and run.get("status") == "submitting":
                if isinstance(exc, trading.TradingExecutionError):
                    # A transport failure can occur after the venue accepts the
                    # request. Preserve the idempotency key and force a later
                    # reconciliation instead of issuing a duplicate order.
                    run["status"] = "submission_unknown"
                    run["error_code"] = "venue_submission_uncertain"
                    logger.error("trading run submission uncertain id=%s venue=%s", run_id, venue)
                    _record_trading_risk_event(
                        claims["sub"],
                        venue=venue,
                        event="submission_unknown",
                        outcome="error",
                        reason_code="venue_submission_uncertain",
                        trade_run_id=run_id,
                    )
                    _notify_trading_safety_event(
                        venue=venue,
                        event="submission_unknown",
                        reason_code="venue_submission_uncertain",
                        trade_run_id=run_id,
                    )
                else:
                    run["status"] = "blocked"
                    run["error_code"] = "execution_guard_failed"
                    logger.warning("trading run blocked before submission id=%s venue=%s", run_id, venue)
                run["updated_at"] = datetime.now(timezone.utc).isoformat()
                _put_trading_run(claims["sub"], run)
            _trading_run_actions.add(1, {"venue": venue, "action": "execute", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


@app.post(
    "/trading/orders",
    tags=["Trading"],
    summary="Submit a confirmed prediction-market order",
    response_model=TradeOrderResponse,
)
async def trading_order(req: TradeOrderRequest, request: Request) -> TradeOrderResponse:
    """Submit a live order after preview guardrails and exact confirmation.

    Live execution is disabled unless `FORESEA_ENABLE_BYO_TRADING=true`. Market/IOC
    style orders are separately disabled unless `FORESEA_ALLOW_MARKET_ORDERS=true`.
    """
    _check_rate_limit(request)
    claims = _require_session(request)
    from analyzing_llm_rationale import trading

    venue = _clean_trading_platform(req.platform)
    with _tracer.start_as_current_span("trading.order.submit") as span:
        span.set_attribute("trading.venue", venue)
        try:
            creds = _resolve_order_credentials(claims["sub"], req)
            payload = req.model_dump(exclude_none=True, exclude={"venue_credentials"})
            preview = trading.preview_order(payload, creds)
            if not preview.get("trading_enabled"):
                # Preserve the explicit server-side live-trading gate before
                # reporting account-specific readiness details.
                trading.place_order(payload, user_id=claims["sub"], creds=creds)
            if creds is None:
                raise HTTPException(status_code=409, detail=f"Connect a {venue} account before submitting an order.")
            await _validate_live_trade_guardrails(
                claims["sub"], payload=payload, preview=preview, credentials=creds
            )
            result = trading.place_order(payload, user_id=claims["sub"], creds=creds)
            audit = _put_trading_order(claims["sub"], _submitted_trading_order(result))
            _record_terminal_trading_order_event(
                claims["sub"], venue=venue, record=audit, previous_status="submitted"
            )
            span.set_attribute("trading.audit_order_id", audit["id"])
            _record_trading_risk_event(
                claims["sub"],
                venue=venue,
                event="order_submitted",
                outcome="success",
                audit_order_id=audit["id"],
            )
            _trading_guardrail_actions.add(1, {"venue": venue, "action": "direct_submit", "outcome": "passed"})
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "submit", "outcome": "success"})
            return TradeOrderResponse(
                **result,
                audit_order_id=audit["id"],
                venue_order_id=audit.get("venue_order_id"),
                reconciliation_status=audit.get("status"),
            )
        except Exception as exc:
            span.record_exception(exc)
            if isinstance(exc, TradingGuardrailError):
                _record_trading_risk_event(
                    claims["sub"],
                    venue=venue,
                    event="guardrail_blocked",
                    outcome="blocked",
                    reason_code=exc.code,
                )
                _trading_guardrail_actions.add(1, {"venue": venue, "action": "direct_submit", "outcome": "blocked"})
                _notify_trading_safety_event(venue=venue, event=exc.code, reason_code=exc.code)
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "submit", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


@app.get("/trading/orders", tags=["Trading"], summary="List the signed-in user's submitted order audit trail")
async def list_trading_orders(
    request: Request, platform: Optional[str] = Query(None, max_length=20)
) -> Dict[str, Any]:
    _check_rate_limit(request)
    claims = _require_session(request)
    venue = _clean_trading_platform(platform) if platform else None
    return {"orders": _list_trading_orders(claims["sub"], venue)}


@app.get("/trading/portfolio", tags=["Trading"], summary="Reconcile an exchange portfolio, orders, and fills")
async def trading_portfolio(
    request: Request,
    platform: str = Query(..., max_length=20),
    limit: int = Query(50, ge=1, le=100),
) -> Dict[str, Any]:
    """Fetch live venue state and safely merge known order status/fill progress."""
    _check_rate_limit(request)
    claims = _require_session(request)
    venue = _clean_trading_platform(platform)
    with _tracer.start_as_current_span("trading.portfolio.reconcile") as span:
        span.set_attribute("trading.venue", venue)
        try:
            from analyzing_llm_rationale import trading

            credentials = _stored_trading_credentials(claims["sub"], venue)
            if credentials is None:
                raise HTTPException(status_code=409, detail=f"Connect a {venue} account before reconciling it.")
            snapshot = trading.reconcile_portfolio(venue, credentials, limit=limit)
            remote_orders = {
                record.get("venue_order_id"): record
                for record in snapshot.get("orders") or []
                if record.get("venue_order_id")
            }
            remote_fills: Dict[str, float] = defaultdict(float)
            for fill in snapshot.get("fills") or []:
                order_id = fill.get("order_id")
                quantity = fill.get("quantity")
                if order_id and isinstance(quantity, (int, float)):
                    remote_fills[str(order_id)] += float(quantity)
            reconciled = []
            for record in _list_trading_orders(claims["sub"], venue):
                venue_order_id = record.get("venue_order_id")
                remote = remote_orders.get(venue_order_id)
                if remote:
                    previous_status = record.get("status")
                    record = _merge_order_reconciliation(record, remote)
                    _put_trading_order(claims["sub"], record)
                    _sync_trade_run_from_order(claims["sub"], record)
                    _record_terminal_trading_order_event(
                        claims["sub"], venue=venue, record=record, previous_status=previous_status
                    )
                elif venue_order_id and venue_order_id in remote_fills:
                    previous_status = record.get("status")
                    record["filled_quantity"] = remote_fills[venue_order_id]
                    if record.get("quantity") is not None and remote_fills[venue_order_id] >= float(record["quantity"]):
                        record["status"] = "filled"
                    record["last_reconciled_at"] = datetime.now(timezone.utc).isoformat()
                    record["updated_at"] = record["last_reconciled_at"]
                    _put_trading_order(claims["sub"], record)
                    _sync_trade_run_from_order(claims["sub"], record)
                    _record_terminal_trading_order_event(
                        claims["sub"], venue=venue, record=record, previous_status=previous_status
                    )
                reconciled.append(record)
            snapshot["audit_orders"] = reconciled
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "portfolio", "outcome": "success"})
            return snapshot
        except HTTPException:
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "portfolio", "outcome": "error"})
            raise
        except Exception as exc:
            span.record_exception(exc)
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "portfolio", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


@app.post("/trading/orders/{audit_order_id}/reconcile", tags=["Trading"], summary="Refresh a submitted order's fill/status")
async def reconcile_trading_order(audit_order_id: str, request: Request) -> Dict[str, Any]:
    _check_rate_limit(request)
    claims = _require_session(request)
    record = _read_trading_order(claims["sub"], audit_order_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Submitted order was not found.")
    if not record.get("venue_order_id"):
        raise HTTPException(status_code=409, detail="The exchange did not return an order ID to reconcile.")
    venue = _clean_trading_platform(record.get("platform"))
    with _tracer.start_as_current_span("trading.order.reconcile") as span:
        span.set_attribute("trading.venue", venue)
        span.set_attribute("trading.audit_order_id", audit_order_id)
        try:
            from analyzing_llm_rationale import trading

            credentials = _stored_trading_credentials(claims["sub"], venue)
            if credentials is None:
                raise HTTPException(status_code=409, detail=f"Reconnect {venue} before reconciling this order.")
            venue_state = trading.reconcile_order(venue, record["venue_order_id"], credentials)
            previous_status = record.get("status")
            record = _merge_order_reconciliation(record, venue_state)
            _put_trading_order(claims["sub"], record)
            _sync_trade_run_from_order(claims["sub"], record)
            _record_terminal_trading_order_event(
                claims["sub"], venue=venue, record=record, previous_status=previous_status
            )
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "order", "outcome": "success"})
            return record
        except HTTPException:
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "order", "outcome": "error"})
            raise
        except Exception as exc:
            span.record_exception(exc)
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "order", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


@app.post(
    "/trading/runs/{run_id}/reconcile",
    tags=["Trading"],
    summary="Reconcile the exchange order linked to a durable trade run",
    response_model=TradeRunResponse,
)
async def reconcile_trading_run(run_id: str, request: Request) -> TradeRunResponse:
    _check_rate_limit(request)
    claims = _require_session(request)
    venue = "unknown"
    with _tracer.start_as_current_span("trading.run.reconcile") as span:
        span.set_attribute("trading.run_id", run_id)
        try:
            run = _read_trading_run(claims["sub"], run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Trade run was not found.")
            audit_order_id = run.get("audit_order_id")
            if not audit_order_id:
                raise HTTPException(status_code=409, detail="Trade run has not submitted an exchange order yet.")
            record = _read_trading_order(claims["sub"], str(audit_order_id))
            if record is None or not record.get("venue_order_id"):
                raise HTTPException(status_code=409, detail="Trade run has no reconcilable exchange order.")
            venue = _clean_trading_platform(record.get("platform"))
            span.set_attribute("trading.venue", venue)
            span.set_attribute("trading.audit_order_id", str(audit_order_id))
            from analyzing_llm_rationale import trading

            credentials = _stored_trading_credentials(claims["sub"], venue)
            if credentials is None:
                raise HTTPException(status_code=409, detail=f"Reconnect {venue} before reconciling this run.")
            venue_state = trading.reconcile_order(venue, record["venue_order_id"], credentials)
            previous_status = record.get("status")
            record = _merge_order_reconciliation(record, venue_state)
            _put_trading_order(claims["sub"], record)
            synced = _sync_trade_run_from_order(claims["sub"], record)
            _record_terminal_trading_order_event(
                claims["sub"], venue=venue, record=record, previous_status=previous_status
            )
            if synced is None:
                raise HTTPException(status_code=409, detail="Trade run lost its order linkage; reconnect and investigate.")
            span.set_attribute("trading.run.status", synced["status"])
            _trading_run_actions.add(1, {"venue": venue, "action": "reconcile", "outcome": "success"})
            return TradeRunResponse(**synced)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            _trading_run_actions.add(1, {"venue": venue, "action": "reconcile", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


@app.delete("/trading/orders/{audit_order_id}", tags=["Trading"], summary="Cancel the remaining quantity of a submitted order")
async def cancel_trading_order(
    audit_order_id: str, req: CancelTradingOrderRequest, request: Request
) -> Dict[str, Any]:
    _check_rate_limit(request)
    claims = _require_session(request)
    if req.confirmation != "CANCEL OPEN ORDER":
        raise HTTPException(status_code=422, detail="confirmation must be exactly 'CANCEL OPEN ORDER'.")
    record = _read_trading_order(claims["sub"], audit_order_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Submitted order was not found.")
    if not record.get("venue_order_id"):
        raise HTTPException(status_code=409, detail="The exchange did not return an order ID to cancel.")
    if record.get("status") in {"filled", "canceled", "rejected"}:
        raise HTTPException(status_code=409, detail="This order is already in a terminal state.")
    venue = _clean_trading_platform(record.get("platform"))
    with _tracer.start_as_current_span("trading.order.cancel") as span:
        span.set_attribute("trading.venue", venue)
        span.set_attribute("trading.audit_order_id", audit_order_id)
        try:
            from analyzing_llm_rationale import trading

            credentials = _stored_trading_credentials(claims["sub"], venue)
            if credentials is None:
                raise HTTPException(status_code=409, detail=f"Reconnect {venue} before cancelling this order.")
            venue_state = trading.cancel_order(
                venue,
                record["venue_order_id"],
                credentials,
                subaccount=record.get("subaccount"),
                exchange_index=int(record.get("exchange_index") or 0),
            )
            previous_status = record.get("status")
            record = _merge_order_reconciliation(record, venue_state)
            record["canceled_at"] = datetime.now(timezone.utc).isoformat()
            _put_trading_order(claims["sub"], record)
            _sync_trade_run_from_order(claims["sub"], record)
            _record_terminal_trading_order_event(
                claims["sub"], venue=venue, record=record, previous_status=previous_status
            )
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "cancel", "outcome": "success"})
            return record
        except HTTPException:
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "cancel", "outcome": "error"})
            raise
        except Exception as exc:
            span.record_exception(exc)
            _trading_reconciliation_actions.add(1, {"venue": venue, "action": "cancel", "outcome": "error"})
            raise _trading_http_exception(exc) from exc


@app.post(
    "/internal/trading/reconcile",
    tags=["System"],
    summary="Run a bounded, read-only reconciliation of open live-trading orders",
    include_in_schema=False,
)
async def scheduled_trading_reconciliation(
    request: Request,
    limit: int = Query(25, ge=1, le=100),
) -> Dict[str, Any]:
    """Scheduler-only trigger; it cannot create, modify, or cancel a trade."""
    _require_trading_reconciliation_token(request)
    effective_limit = min(limit, _TRADING_RECONCILIATION_MAX_ORDERS)
    with _tracer.start_as_current_span("trading.reconciliation.scheduled") as span:
        span.set_attribute("trading.reconciliation.limit", effective_limit)
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, _scheduled_reconcile_open_trading_orders, effective_limit
            )
            span.set_attributes(
                {
                    "trading.reconciliation.checked": result["checked"],
                    "trading.reconciliation.updated": result["updated"],
                    "trading.reconciliation.errors": result["errors"],
                }
            )
            logger.info(
                "scheduled trading reconciliation completed checked=%s updated=%s errors=%s",
                result["checked"], result["updated"], result["errors"],
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.error("scheduled trading reconciliation failed")
            raise HTTPException(status_code=503, detail="Scheduled reconciliation could not complete.") from exc


@app.post(
    "/internal/agent-runs/reconcile",
    tags=["System"],
    summary="Mark stale, orphaned AgentRun records as interrupted",
    include_in_schema=False,
)
async def scheduled_agent_run_reconciliation(
    request: Request,
    limit: int = Query(25, ge=1, le=100),
) -> Dict[str, Any]:
    """Scheduler-only trigger; it never resumes a run, only marks a run stuck
    at status="running" past AGENT_RUN_STALE_MINUTES as interrupted."""
    _require_agent_run_reconciliation_token(request)
    effective_limit = min(limit, _AGENT_RUN_RECONCILIATION_MAX_RUNS)
    with _tracer.start_as_current_span("agent.run.reconciliation.scheduled") as span:
        span.set_attribute("agent.run.reconciliation.limit", effective_limit)
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, _scheduled_reconcile_stale_agent_runs, _AGENT_RUN_STALE_MINUTES, effective_limit
            )
            span.set_attributes(
                {
                    "agent.run.reconciliation.checked": result["checked"],
                    "agent.run.reconciliation.interrupted": result["interrupted"],
                    "agent.run.reconciliation.errors": result["errors"],
                }
            )
            logger.info(
                "scheduled agent run reconciliation completed checked=%s interrupted=%s errors=%s",
                result["checked"], result["interrupted"], result["errors"],
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.error("scheduled agent run reconciliation failed")
            raise HTTPException(status_code=503, detail="Scheduled reconciliation could not complete.") from exc


@app.get(
    "/internal/trading/readiness",
    tags=["System"],
    summary="Read non-sensitive launch prerequisites for guarded live trading",
    response_model=TradingLaunchReadinessResponse,
    include_in_schema=False,
)
async def trading_launch_readiness(request: Request) -> TradingLaunchReadinessResponse:
    """Operator-only configuration check; this endpoint never contacts a venue."""
    _require_trading_reconciliation_token(request)
    with _tracer.start_as_current_span("trading.readiness.check") as span:
        try:
            readiness = _trading_launch_readiness()
            outcome = "ready" if readiness.ready_for_connection_beta else "blocked"
            span.set_attributes(
                {
                    "trading.readiness.outcome": outcome,
                    "trading.readiness.check_count": len(readiness.checks),
                    "trading.readiness.byo_enabled": readiness.byo_trading_enabled,
                    "trading.readiness.kill_switch": readiness.platform_kill_switch,
                }
            )
            _trading_readiness_actions.add(1, {"outcome": outcome})
            logger.info(
                "trading launch readiness checked outcome=%s checks=%s",
                outcome,
                len(readiness.checks),
            )
            return readiness
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            _trading_readiness_actions.add(1, {"outcome": "error"})
            logger.error("trading launch readiness check failed")
            raise HTTPException(status_code=503, detail="Trading launch readiness could not be evaluated.") from exc


@app.get("/providers/models", tags=["System"], summary="List models from a provider base URL")
async def list_provider_models(base_url: str = Query(..., max_length=500)):
    """Fetch available models from an OpenAI-compatible or Ollama endpoint.

    Pass `base_url` (e.g. `http://localhost:11434`). Returns `{models: [...]}`.
    Cloud-metadata hosts are blocked.
    """
    import ipaddress as _ip
    from urllib.parse import urlparse as _up

    parsed = _up(base_url.strip().rstrip("/"))
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=422, detail="base_url must be http:// or https://")
    host = (parsed.hostname or "").lower()
    blocked = {"metadata.google.internal", "169.254.169.254"}
    if not host or host in blocked or host.endswith(".internal"):
        raise HTTPException(status_code=422, detail="base_url host is not allowed")
    try:
        ip = _ip.ip_address(host)
        if ip == _ip.ip_address("169.254.169.254"):
            raise HTTPException(status_code=422, detail="base_url host is not allowed")
    except ValueError:
        pass

    import httpx
    errors = []
    for path in ("/api/tags", "/v1/models"):
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(f"{base_url.rstrip('/')}{path}")
            if resp.status_code == 200:
                data = resp.json()
                if path == "/api/tags":
                    models = [m.get("name", "") for m in data.get("models", [])]
                else:
                    models = [m.get("id", "") for m in data.get("data", [])]
                return {"models": [m for m in models if m]}
        except Exception as exc:
            errors.append(str(exc))
    raise HTTPException(status_code=502, detail=f"Could not reach provider: {'; '.join(errors)}")


@app.post("/extract", tags=["System"], summary="Extract text from a PDF file or URL")
async def extract_attachment(  # noqa: B008
    request: Request,
    file: Optional[UploadFile] = File(None),  # noqa: B008
    url: Optional[str] = Form(None),  # noqa: B008
) -> Dict[str, Any]:
    """Extract plain text from an uploaded PDF or a web URL.

    Returns a dict compatible with the `news_articles` field of `/predict`:
    `{title, text, source, url}`. Pass one of `file` or `url`, not both.
    """
    _check_rate_limit(request)
    loop = asyncio.get_running_loop()
    if file is not None:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Only PDF files are supported.")
        content = await file.read()
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="PDF exceeds 20 MB limit.")
        text = await loop.run_in_executor(None, _extract_pdf_bytes, content)
        return {"title": file.filename, "text": text[:20000], "source": file.filename, "url": None}
    if url:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=422, detail="URL must start with http:// or https://")
        url_cache_key = _cache_key("extract_url", url)
        cached = _cache_get(url_cache_key)
        if cached is not None:
            return cached
        extracted = await loop.run_in_executor(None, _extract_url_sync, url)
        _cache_set(url_cache_key, extracted, _EXTRACT_CACHE_TTL)
        return extracted
    raise HTTPException(status_code=400, detail="Provide either a PDF file or a url.")


@app.get("/auth/config", tags=["Auth"], include_in_schema=False)
async def auth_config() -> Dict[str, str]:
    """Return public OAuth client IDs so the browser can offer sign-in options."""
    return {
        "google_client_id": _GOOGLE_CLIENT_ID or "",
        "github_client_id": _GITHUB_CLIENT_ID or "",
    }


@app.post("/auth/google", tags=["Auth"], summary="Sign in with Google", response_model=SessionResponse)
async def auth_google(req: GoogleAuthRequest) -> SessionResponse:
    """Verify a Google One-Tap ID token, create or update the user account, and return a session token.

    The browser should send the `credential` string returned by
    `google.accounts.id.initialize({ callback })` after the user grants consent.
    Store the returned `token` in `localStorage` and include it as
    `Authorization: Bearer <token>` on subsequent authenticated requests.
    """
    claims = _verify_google_token(req.credential)
    sub = str(claims["sub"])
    email = claims.get("email", "")
    name = claims.get("name", "")
    picture = claims.get("picture", "")
    loop = asyncio.get_running_loop()
    uid = await loop.run_in_executor(None, _upsert_user, sub, email, name, picture)
    token = _issue_session(uid, email, name, picture)
    return SessionResponse(token=token, user_id=uid, email=email, name=name, picture=picture)


@app.post("/auth/github", tags=["Auth"], summary="Sign in with GitHub", response_model=SessionResponse)
async def auth_github(req: GitHubAuthRequest) -> SessionResponse:
    """Exchange a GitHub OAuth `code` for the user's profile and return a session token.

    The browser sends users to GitHub's authorize URL, then posts the returned
    `code` (and the same `redirect_uri`) here. Store the returned `token` like the
    Google flow.
    """
    loop = asyncio.get_running_loop()
    profile = await loop.run_in_executor(None, _exchange_github_code, req.code, req.redirect_uri)
    sub, email, name, picture = profile["sub"], profile["email"], profile["name"], profile["picture"]
    uid = await loop.run_in_executor(None, _upsert_user, sub, email, name, picture)
    token = _issue_session(uid, email, name, picture)
    return SessionResponse(token=token, user_id=uid, email=email, name=name, picture=picture)


@app.post("/auth/register", tags=["Auth"], summary="Create an account", response_model=SessionResponse)
async def auth_register(req: RegisterRequest) -> SessionResponse:
    """Create an email/password account and return a session token.

    The email becomes the account ID. Passwords are stored as salted
    PBKDF2-HMAC-SHA256 hashes; the plaintext is never persisted.
    """
    user_id = req.email
    loop = asyncio.get_running_loop()
    existing = await loop.run_in_executor(None, _get_user_record, user_id)
    if existing is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    name = req.name or req.email.split("@")[0]
    password_hash = _hash_password(req.password)
    await loop.run_in_executor(None, _create_password_user, user_id, req.email, name, password_hash)
    token = _issue_session(user_id, req.email, name, "")
    return SessionResponse(token=token, user_id=user_id, email=req.email, name=name, picture="")


@app.post("/auth/login", tags=["Auth"], summary="Sign in with email and password", response_model=SessionResponse)
async def auth_login(req: LoginRequest) -> SessionResponse:
    """Verify an email/password account and return a session token."""
    user_id = req.email
    loop = asyncio.get_running_loop()
    record = await loop.run_in_executor(None, _get_user_record, user_id)
    stored_hash = record.get("password_hash") if record else None
    # Always run a verification so timing does not reveal whether the email exists.
    if not _verify_password(req.password, stored_hash or _DUMMY_PASSWORD_HASH) or not stored_hash:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    name = record.get("name") or req.email.split("@")[0]
    picture = record.get("picture", "")
    await loop.run_in_executor(None, _touch_last_login, user_id)
    token = _issue_session(user_id, req.email, name, picture)
    return SessionResponse(token=token, user_id=user_id, email=req.email, name=name, picture=picture)


@app.get("/auth/me", tags=["Auth"], summary="Get current user", response_model=AuthMeResponse)
async def auth_me(request: Request) -> AuthMeResponse:
    """Return the authenticated user decoded from the `Authorization: Bearer` session token."""
    claims = _require_session(request)
    return AuthMeResponse(
        user_id=claims["sub"],
        email=claims["email"],
        name=claims["name"],
        picture=claims["picture"],
    )


@app.get("/chat/conversations", tags=["Auth"], response_model=ChatConversationList, include_in_schema=False)
async def chat_conversations(request: Request) -> ChatConversationList:
    """List conversations for the signed-in user."""
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    conversations = await loop.run_in_executor(None, _list_conversations, claims["sub"])
    return ChatConversationList(conversations=[ChatConversation(**c) for c in conversations])


@app.put("/chat/conversations/{conversation_id}", tags=["Auth"], response_model=ChatConversation, include_in_schema=False)
async def save_chat_conversation(
    conversation_id: str,
    conversation: ChatConversation,
    request: Request,
) -> ChatConversation:
    """Create or replace one conversation for the signed-in user."""
    claims = _require_session(request)
    if conversation.id != conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID path/body mismatch.")
    loop = asyncio.get_running_loop()
    saved = await loop.run_in_executor(None, _put_conversation, claims["sub"], conversation.model_dump())
    return ChatConversation(**saved)


@app.delete("/chat/conversations/{conversation_id}", tags=["Auth"], include_in_schema=False)
async def delete_chat_conversation(conversation_id: str, request: Request) -> Dict[str, bool]:
    """Delete one conversation for the signed-in user."""
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _delete_conversation, claims["sub"], conversation_id)
    return {"ok": True}


@app.get(
    "/personal-ledger",
    tags=["Auth"],
    response_model=PersonalLedgerList,
    response_model_exclude_none=True,
)
async def list_personal_ledger(request: Request) -> PersonalLedgerList:
    """Return the current user's explicitly saved chat forecasts."""
    claims = _require_session(request)
    with _tracer.start_as_current_span("personal_ledger.list") as span:
        span.set_attribute("user.id", claims["sub"])
        loop = asyncio.get_running_loop()
        entries = await loop.run_in_executor(None, _list_personal_ledger, claims["sub"])
        span.set_attribute("personal_ledger.entries.count", len(entries))
        return PersonalLedgerList(entries=[PersonalLedgerEntry(**entry) for entry in entries])


@app.put(
    "/personal-ledger/{entry_id}",
    tags=["Auth"],
    response_model=PersonalLedgerEntry,
    response_model_exclude_none=True,
)
async def save_personal_ledger_entry(
    entry_id: str,
    entry: PersonalLedgerEntry,
    request: Request,
) -> PersonalLedgerEntry:
    """Save a forecast to the current user's private personal ledger."""
    claims = _require_session(request)
    if entry.id != entry_id:
        raise HTTPException(status_code=400, detail="Ledger entry path/body mismatch.")
    with _tracer.start_as_current_span("personal_ledger.add") as span:
        span.set_attribute("user.id", claims["sub"])
        span.set_attribute("personal_ledger.action", "add")
        try:
            loop = asyncio.get_running_loop()
            saved = await loop.run_in_executor(
                None, _put_personal_ledger_entry, claims["sub"], entry.model_dump()
            )
            _personal_ledger_actions.add(1, {"action": "add", "outcome": "success"})
            logger.info("personal ledger entry saved")
            return PersonalLedgerEntry(**saved)
        except Exception as exc:
            _personal_ledger_actions.add(1, {"action": "add", "outcome": "failure"})
            span.record_exception(exc)
            span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, type(exc).__name__))
            logger.exception("personal ledger entry could not be saved")
            raise


@app.patch(
    "/personal-ledger/{entry_id}/verdict",
    tags=["Auth"],
    response_model=PersonalLedgerEntry,
    response_model_exclude_none=True,
)
async def judge_personal_ledger_entry(
    entry_id: str,
    verdict: PersonalLedgerVerdict,
    request: Request,
) -> PersonalLedgerEntry:
    """Mark one of the current user's saved forecasts correct or wrong."""
    claims = _require_session(request)
    with _tracer.start_as_current_span("personal_ledger.feedback") as span:
        span.set_attribute("user.id", claims["sub"])
        span.set_attribute("personal_ledger.action", "feedback")
        span.set_attribute("personal_ledger.verdict", verdict.verdict)
        try:
            loop = asyncio.get_running_loop()
            entry = await loop.run_in_executor(
                None, _get_personal_ledger_entry, claims["sub"], entry_id
            )
            if entry is None:
                _personal_ledger_actions.add(
                    1, {"action": "feedback", "verdict": verdict.verdict, "outcome": "missing"}
                )
                raise HTTPException(status_code=404, detail="Ledger entry not found.")
            entry["user_verdict"] = verdict.verdict
            entry["judgedAt"] = int(time.time() * 1000)
            saved = await loop.run_in_executor(None, _put_personal_ledger_entry, claims["sub"], entry)
            _personal_ledger_actions.add(
                1, {"action": "feedback", "verdict": verdict.verdict, "outcome": "success"}
            )
            logger.info("personal ledger entry feedback saved")
            return PersonalLedgerEntry(**saved)
        except HTTPException:
            raise
        except Exception as exc:
            _personal_ledger_actions.add(
                1, {"action": "feedback", "verdict": verdict.verdict, "outcome": "failure"}
            )
            span.record_exception(exc)
            span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, type(exc).__name__))
            logger.exception("personal ledger entry feedback could not be saved")
            raise


@app.delete("/personal-ledger/{entry_id}", tags=["Auth"])
async def delete_personal_ledger_entry(entry_id: str, request: Request) -> Dict[str, bool]:
    """Remove one saved forecast from the current user's personal ledger."""
    claims = _require_session(request)
    with _tracer.start_as_current_span("personal_ledger.remove") as span:
        span.set_attribute("user.id", claims["sub"])
        span.set_attribute("personal_ledger.action", "remove")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _delete_personal_ledger_entry, claims["sub"], entry_id)
            _personal_ledger_actions.add(1, {"action": "remove", "outcome": "success"})
            return {"ok": True}
        except Exception as exc:
            _personal_ledger_actions.add(1, {"action": "remove", "outcome": "failure"})
            span.record_exception(exc)
            span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, type(exc).__name__))
            logger.exception("personal ledger entry could not be removed")
            raise


@app.get("/agent-profiles", tags=["Agents"], response_model=AgentProfileList)
async def list_agent_profiles(request: Request) -> AgentProfileList:
    """List the current user's private copied-agent recipes."""
    _check_rate_limit(request)
    claims = _require_session(request)
    with _tracer.start_as_current_span("agent.profile.list") as span:
        span.set_attribute("user.id", claims["sub"])
        profiles = _list_agent_profiles(claims["sub"])
        span.set_attribute("agent.profile.count", len(profiles))
        _agent_profile_actions.add(1, {"action": "list", "outcome": "success"})
        return AgentProfileList(profiles=[AgentProfile(**profile) for profile in profiles])


@app.post(
    "/agent-profiles/copy",
    tags=["Agents"],
    response_model=AgentProfileCopyResponse,
    status_code=201,
)
async def copy_agent_profile(req: AgentProfileCopyRequest, request: Request) -> AgentProfileCopyResponse:
    """Copy a public model's research setup into a private, research-only recipe."""
    _check_rate_limit(request)
    claims = _require_session(request)
    source_agent_id = req.source_agent_id.strip()
    with _tracer.start_as_current_span("agent.profile.copy") as span:
        span.set_attribute("user.id", claims["sub"])
        try:
            if source_agent_id not in _SCADS_MODEL_ALLOWLIST:
                raise HTTPException(status_code=422, detail="That public Foresea agent is not available to copy.")
            span.set_attribute("agent.profile.source", source_agent_id)
            profiles = _list_agent_profiles(claims["sub"])
            existing = next(
                (profile for profile in profiles if profile.get("source_agent_id") == source_agent_id),
                None,
            )
            if existing is not None:
                span.set_attribute("outcome", "existing")
                _agent_profile_actions.add(1, {"action": "copy", "outcome": "existing"})
                return AgentProfileCopyResponse(profile=AgentProfile(**existing), created=False)
            if len(profiles) >= _MAX_AGENT_PROFILES_PER_USER:
                raise HTTPException(
                    status_code=409,
                    detail=f"You can keep up to {_MAX_AGENT_PROFILES_PER_USER} copied agents. Remove one first.",
                )
            now = datetime.now(timezone.utc).isoformat()
            name = (req.name or f"Copy of {source_agent_id}").strip()
            profile = _put_agent_profile(
                claims["sub"],
                {
                    "id": f"agent_{secrets.token_urlsafe(12).replace('-', '_')}",
                    "name": name,
                    "source_agent_id": source_agent_id,
                    "model": source_agent_id,
                    "instruction": (
                        f"Use {source_agent_id}'s public Foresea research setup: resolve the exact contract "
                        "and current venue quote, gather independent evidence, state the model-versus-market "
                        "edge, then list the strongest disconfirming case and the condition that would invalidate "
                        "the thesis. This is research only: do not create, size, or submit a trade."
                    ),
                    "version": 1,
                    "execution_mode": "research_only",
                    "created_at": now,
                    "updated_at": now,
                },
            )
            span.set_attributes({"agent.profile.id": profile["id"], "outcome": "created"})
            _agent_profile_actions.add(1, {"action": "copy", "outcome": "created"})
            logger.info("agent profile copied id=%s source=%s", profile["id"], source_agent_id)
            return AgentProfileCopyResponse(profile=AgentProfile(**profile), created=True)
        except HTTPException:
            _agent_profile_actions.add(1, {"action": "copy", "outcome": "rejected"})
            raise
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            _agent_profile_actions.add(1, {"action": "copy", "outcome": "error"})
            logger.error("agent profile copy failed")
            raise


@app.delete("/agent-profiles/{profile_id}", tags=["Agents"])
async def delete_agent_profile(profile_id: str, request: Request) -> Dict[str, bool]:
    """Delete one of the current user's private copied-agent recipes."""
    _check_rate_limit(request)
    claims = _require_session(request)
    with _tracer.start_as_current_span("agent.profile.delete") as span:
        span.set_attributes({"user.id": claims["sub"], "agent.profile.id": profile_id})
        profile = _read_agent_profile(claims["sub"], profile_id)
        if profile is None:
            _agent_profile_actions.add(1, {"action": "delete", "outcome": "missing"})
            raise HTTPException(status_code=404, detail="Copied agent was not found.")
        try:
            _delete_agent_profile(claims["sub"], profile_id)
            _agent_profile_actions.add(1, {"action": "delete", "outcome": "success"})
            logger.info("agent profile deleted id=%s", profile_id)
            return {"ok": True}
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            _agent_profile_actions.add(1, {"action": "delete", "outcome": "error"})
            logger.error("agent profile deletion failed")
            raise


@app.get("/agent/runs", tags=["Agents"], response_model=AgentRunList)
async def list_agent_runs(
    request: Request,
    client_run_key: Optional[str] = Query(
        None, max_length=120,
        description="Filter to the run created with this caller-supplied idempotency key -- "
                    "how an API-key client looks its own run back up by client_run_key after a timeout.",
    ),
) -> AgentRunList:
    """List the signed-in user's durable research runs, newest activity first."""
    _check_rate_limit(request)
    claims = _require_auth(request)
    with _tracer.start_as_current_span("agent.run.list") as span:
        span.set_attribute("user.id", claims["sub"])
        runs = _list_agent_runs(claims["sub"])
        if client_run_key:
            runs = [r for r in runs if r.get("client_run_key") == client_run_key]
        span.set_attribute("agent.run.count", len(runs))
        _agent_run_actions.add(1, {"action": "list", "outcome": "success"})
        return AgentRunList(runs=[_agent_run_summary(run) for run in runs])


@app.get("/agent/runs/{run_id}", tags=["Agents"], response_model=AgentRunResponse)
async def read_agent_run(run_id: str, request: Request) -> AgentRunResponse:
    """Read one private research run, including its completed report when available."""
    _check_rate_limit(request)
    claims = _require_auth(request)
    with _tracer.start_as_current_span("agent.run.read") as span:
        span.set_attributes({"user.id": claims["sub"], "agent.run.id": run_id})
        run = _read_agent_run(claims["sub"], run_id)
        if run is None:
            _agent_run_actions.add(1, {"action": "read", "outcome": "missing"})
            raise HTTPException(status_code=404, detail="Agent run was not found.")
        _agent_run_actions.add(1, {"action": "read", "outcome": "success"})
        return _agent_run_response(run)


@app.get("/chat/models", tags=["System"], include_in_schema=False)
async def chat_models(request: Request) -> Dict[str, Any]:
    """Chat-capable server-hosted models and connected user BYO models exposed in the prompt selector."""
    _check_rate_limit(request)
    default_key = _state.get("model_key") or "gpt-oss-120b"
    models = [
        {
            "key": cfg.name,
            "label": cfg.result_label,
            "model": cfg.router_model_name,
            "fallbacks": list(cfg.fallback_model_chain),
            "default": cfg.name == default_key,
        }
        for cfg in _SCADS_CHAT_MODEL_OPTIONS
    ]
    user_models = []
    user_id = _optional_user_id(request)
    if user_id:
        try:
            from analyzing_llm_rationale import model_providers
            statuses = model_providers.get_user_provider_status_list(user_id)
            for s in statuses:
                if s.connected:
                    user_models.append({
                        "key": f"byo:{s.provider_id}:{s.default_model}",
                        "label": f"{s.name} ({s.default_model})",
                        "model": s.default_model,
                        "provider": s.provider_id,
                        "custom": True,
                    })
        except Exception:
            pass
    return {"default_model": default_key, "models": models, "user_models": user_models}


@app.get("/chat/{conversation_id}", include_in_schema=False)
async def chat_page(conversation_id: str, request: Request) -> Response:
    """Serve a refresh-safe, shareable app URL for one local conversation."""
    return await _spa_page(request, "chat")


@app.get("/favorites", tags=["Auth"], response_model=FavoriteList, summary="List favourited markets")
async def list_favorites(request: Request) -> FavoriteList:
    """Return the signed-in user's favourited markets/questions (their watchlist)."""
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    favs = await loop.run_in_executor(None, _list_favorites, claims["sub"])
    return FavoriteList(favorites=[FavoriteMarket(**f) for f in favs])


@app.put("/favorites/{key:path}", tags=["Auth"], response_model=FavoriteMarket, summary="Add/update a favourite")
async def save_favorite(key: str, favorite: FavoriteMarket, request: Request) -> FavoriteMarket:
    """Create or replace one favourite for the signed-in user."""
    claims = _require_session(request)
    if favorite.key != key:
        raise HTTPException(status_code=400, detail="Favourite key path/body mismatch.")
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    payload = favorite.model_dump()
    payload["updatedAt"] = now
    payload.setdefault("createdAt", None)
    if not payload.get("createdAt"):
        payload["createdAt"] = now
    loop = asyncio.get_running_loop()
    saved = await loop.run_in_executor(None, _put_favorite, claims["sub"], payload)
    # Enrol the market into the forecasting loop so the tick keeps a fresh model
    # forecast for it — that's what powers the live model/edge in the watchlist.
    await _enroll_market(saved.get("platform"), saved.get("ident"),
                         saved.get("market_url"), saved.get("question") or "", "favorite")
    return FavoriteMarket(**saved)


def _norm_url(url: Optional[str]) -> str:
    """Normalise a market URL for joining favourites to forecasts (all surfaces
    build URLs with the same _polymarket_quote/_kalshi_quote helpers)."""
    return (url or "").strip().rstrip("/").lower()


def _forecast_by_url() -> Dict[str, float]:
    """Latest model forecast per market URL from the committed live track record."""
    out: Dict[str, float] = {}
    live = _read_edge_board_record() or {}
    for item in live.get("edge_board", []):
        url = _norm_url(item.get("market_url"))
        mp = item.get("model_probability")
        if url and mp is not None:
            out[url] = float(mp)
    return out


@app.delete("/favorites/{key:path}", tags=["Auth"], summary="Remove a favourite")
async def remove_favorite(key: str, request: Request) -> Dict[str, bool]:
    """Delete one favourite for the signed-in user."""
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _delete_favorite, claims["sub"], key)
    return {"ok": True}


@app.get("/favorites/prices", tags=["Auth"], summary="Live prices for favourited markets")
async def favorite_prices(request: Request) -> Dict[str, Any]:
    """Fetch current market prices for the user's favourited markets (cheap, no LLM)."""
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    favs = await loop.run_in_executor(None, _list_favorites, claims["sub"])
    from analyzing_llm_rationale import market_data as _md

    async def _one(fav: Dict[str, Any]) -> tuple[str, Optional[Dict[str, Any]]]:
        platform = (fav.get("platform") or "").lower()
        ident = fav.get("ident") or ""
        if not ident:
            return fav["key"], None
        # Short per-market cache so near-real-time polling (every few seconds,
        # across all users) hits each venue at most ~once per cache window.
        qkey = f"favquote:{platform}:{ident}"
        cached = _cache_get(qkey)
        if cached is not None:
            return fav["key"], cached
        try:
            if "poly" in platform:
                q = await loop.run_in_executor(None, lambda: _md.fetch_polymarket(slug=ident))
            elif "kalshi" in platform:
                q = await loop.run_in_executor(None, lambda: _md.fetch_kalshi(ident))
            else:
                return fav["key"], None
            prob = q.get("probability")
            if prob is None:
                return fav["key"], None
            v = {"probability": float(prob), "change_24h": q.get("price_change_24h")}
            _cache_set(qkey, v, 6)
            return fav["key"], v
        except Exception:
            return fav["key"], None

    results = await asyncio.gather(*[_one(f) for f in favs if f.get("ident")])
    quotes = {k: v for k, v in results if v is not None}
    # Attach the latest model forecast by market URL.
    fcast = await loop.run_in_executor(None, _forecast_by_url)
    favs_by_key = {f["key"]: f for f in favs}
    for key, v in quotes.items():
        fav = favs_by_key.get(key) or {}
        # Prefer the freshest tick forecast; fall back to the
        # add-time forecast stored on the favourite so it persists across reloads.
        model = fcast.get(_norm_url(fav.get("market_url")))
        if model is None:
            model = fav.get("model_probability")
        if model is not None:
            v["model"] = model
    # `prices` kept for backwards-compat; `quotes` carries price + 24h change + model.
    prices = {k: v["probability"] for k, v in quotes.items()}
    return {"prices": prices, "quotes": quotes,
            "generated_at": datetime.now(timezone.utc).isoformat()}


@app.get("/market/quote", tags=["Markets"], summary="Quote a single market by platform+ident")
async def market_quote(
    request: Request,
    platform: str = Query(..., max_length=20),
    ident: str = Query(..., max_length=200),
) -> Dict[str, Any]:
    """Fetch one market's current quote (question, price, 24h change, close time)
    so the watchlist can add a market pasted as a URL. Public + cached, no LLM."""
    _check_rate_limit(request)
    plat = platform.strip().lower()
    cache_key = f"market_quote:{plat}:{ident}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return JSONResponse(cached, headers={"Cache-Control": "public, max-age=30"})
    from analyzing_llm_rationale import market_data as _md

    loop = asyncio.get_running_loop()
    try:
        if "poly" in plat:
            q = await loop.run_in_executor(None, lambda: _md.fetch_polymarket(slug=ident))
        elif "kalshi" in plat:
            q = await loop.run_in_executor(None, lambda: _md.fetch_kalshi(ident))
        else:
            raise HTTPException(status_code=422, detail="platform must be 'polymarket' or 'kalshi'.")
    except _md.MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = {
        "platform": q.get("platform"),
        "ident": q.get("ident") or ident,
        "question": q.get("question"),
        "market_url": q.get("market_url"),
        "probability": q.get("probability"),
        "change_24h": q.get("price_change_24h"),
        "close_time": q.get("close_time"),
    }
    _cache_set(cache_key, payload, 30)
    return JSONResponse(payload, headers={"Cache-Control": "public, max-age=30"})


class MarketForecastRequest(BaseModel):
    """Forecast one market immediately (e.g. just-watchlisted)."""
    question: str = Field(..., max_length=600)
    market_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    market_platform: Optional[str] = Field(None, max_length=40)
    market_ident: Optional[str] = Field(None, max_length=200)
    market_url: Optional[str] = Field(None, max_length=500)


def _market_forecast_predict_request(req: MarketForecastRequest) -> PredictRequest:
    return PredictRequest(
        question=req.question,
        market_probability=req.market_probability,
        market_platform=req.market_platform,
        market_ident=req.market_ident,
        market_url=req.market_url,
        market_outcome="Yes",
        chat_mode=False,
    )


def _market_forecast_payload(resp: PredictResponse) -> Dict[str, Any]:
    model_p = _model_probability_from_prediction(resp)
    return {"model_probability": model_p}


@app.post("/market/forecast", tags=["Markets"], summary="Forecast one market now")
async def market_forecast(req: MarketForecastRequest, request: Request) -> Dict[str, Any]:
    """Run the model on a single market right now and return its probability, so a
    freshly-watchlisted market shows a forecast without waiting for the next tick.
    Sign-in gated (the watchlist is). Reuses the `/predict` pipeline."""
    _require_session(request)
    pr = _market_forecast_predict_request(req)
    resp = await predict(pr, request)
    return _market_forecast_payload(resp)


@app.post("/market/forecast/stream", tags=["Markets"], summary="Stream one market forecast now")
async def market_forecast_stream(req: MarketForecastRequest, request: Request) -> StreamingResponse:
    """Stream the underlying LLM forecast for a newly-watchlisted market."""
    _require_session(request)
    _check_rate_limit(request)
    _check_api_key(request)
    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    async def events():
        pr = _market_forecast_predict_request(req)
        try:
            messages, evidence_articles, evidence_error = await _prepare_predict_messages(
                pr, _optional_user_id(request)
            )
            provider, temperature, max_tokens = _select_predict_provider(pr)
        except HTTPException as exc:
            yield _sse_event("error", {"status_code": exc.status_code, "detail": exc.detail})
            return
        except Exception:
            logger.exception("market forecast stream setup failed")
            yield _sse_event("error", {
                "status_code": 500,
                "detail": "The streaming market forecast could not be prepared.",
            })
            return

        yield _sse_event("meta", {"status": "streaming"})
        chunks: List[str] = []
        try:
            async for chunk in _provider_stream_chat(provider, messages, temperature, max_tokens):
                if await request.is_disconnected():
                    return
                chunks.append(chunk)
                yield _sse_event("delta", {"text": chunk})
        except Exception as exc:
            http_exc = _provider_http_error(exc)
            yield _sse_event("error", {
                "status_code": http_exc.status_code,
                "detail": http_exc.detail,
            })
            return
        text = "".join(chunks).strip()
        parsed = parse_model_response(text, ("type", "predicted_answer", "confidence",
                                             "rationale", "options", "p10", "p50", "p90", "unit"))
        resp = _build_typed_response(pr, parsed, text, evidence_articles, evidence_error)
        await _finalize_predict_response(pr, resp, _optional_user_id(request))
        yield _sse_event("done", {
            "response": resp.model_dump(mode="json"),
            **_market_forecast_payload(resp),
        })

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# marketd — the Go market-data ingestion microservice. When MARKETD_URL is set,
# /markets/search is served by it (concurrent venue ingestion + normalization in
# Go); the in-process Python path stays as a fallback.
_MARKETD_URL = (os.environ.get("MARKETD_URL") or "").rstrip("/")


def _marketd_token(audience: str) -> Optional[str]:
    """Mint a Cloud Run identity token for the authenticated call to marketd."""
    try:
        import google.auth.transport.requests as _greq
        import google.oauth2.id_token as _idt
        return _idt.fetch_id_token(_greq.Request(), audience)
    except Exception:
        return None


def _marketd_search_sync(query: str, category: str, limit: int) -> Optional[List[Dict[str, Any]]]:
    """Fetch normalized markets from marketd. Returns None on any failure so the
    caller falls back to the in-process Python ingestion."""
    if not _MARKETD_URL:
        return None
    import requests

    headers = {}
    token = _marketd_token(_MARKETD_URL)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(
            f"{_MARKETD_URL}/markets",
            params={"q": query, "category": category, "limit": limit},
            headers=headers, timeout=10,
        )
        if resp.status_code != 200:
            return None
        markets = resp.json().get("markets")
    except Exception:
        return None
    if not isinstance(markets, list):
        return None
    out: List[Dict[str, Any]] = []
    for m in markets:
        ident = m.get("ident") or ""
        if not ident or m.get("probability") is None:
            continue
        out.append({
            "platform": m.get("platform"),
            "ident": ident,
            "question": m.get("question"),
            "market_url": m.get("market_url"),
            "probability": m.get("probability"),
            "close_time": m.get("close_time"),
            "volume": m.get("volume"),
            "category": m.get("category"),
        })
    return out


def _marketd_quotes_sync(refs: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Fetch depth/candle data for a batch of "platform:ident:extra" refs from
    marketd's /quotes. Returns None if marketd is unconfigured or the whole
    request fails; a single bad ref is already carried as that item's own
    "error" field by marketd itself, not a None here."""
    if not _MARKETD_URL or not refs:
        return None
    import requests

    headers = {}
    token = _marketd_token(_MARKETD_URL)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(
            f"{_MARKETD_URL}/quotes",
            params=[("ref", r) for r in refs],
            headers=headers, timeout=12,
        )
        if resp.status_code != 200:
            return None
        quotes = resp.json().get("quotes")
    except Exception:
        return None
    return quotes if isinstance(quotes, list) else None


@app.get("/markets/search", tags=["Markets"], summary="Search markets to add to a watchlist")
async def markets_search(
    request: Request,
    q: str = Query("", max_length=120),
    category: str = Query("", max_length=40),
    limit: int = Query(24, ge=1, le=40),
) -> Dict[str, Any]:
    """Search open Polymarket + Kalshi markets by keyword and/or category (empty
    `q`+`category` = trending by volume) so the watchlist can browse-and-add with
    one click. Public + cached."""
    _check_rate_limit(request)
    query = q.strip()
    cat = category.strip()
    cache_key = f"markets_search:{query.lower()}:{cat.lower()}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return JSONResponse(cached, headers={"Cache-Control": "public, max-age=60"})

    loop = asyncio.get_running_loop()

    # Primary path: the marketd Go microservice (concurrent ingestion + normalize).
    md_results = await loop.run_in_executor(None, _marketd_search_sync, query, cat, limit)
    if md_results is not None:
        payload = {"results": md_results[:limit], "query": query, "category": cat, "source": "marketd"}
        _cache_set(cache_key, payload, 60)
        return JSONResponse(payload, headers={"Cache-Control": "public, max-age=60"})

    # Fallback: in-process Python ingestion (also used by the tick and agent paths).
    from analyzing_llm_rationale import market_data as _md

    per_venue = max(1, limit // 2 + 1)

    def _list(lister) -> List[Dict[str, Any]]:
        try:
            return lister(limit=per_venue, query=query or None, contested_only=False,
                          category=cat or None)
        except _md.MarketDataError:
            return []

    poly, kalshi = await asyncio.gather(
        loop.run_in_executor(None, lambda: _list(_md.list_polymarket)),
        loop.run_in_executor(None, lambda: _list(_md.list_kalshi)),
    )
    results = []
    for quote in (*poly, *kalshi):
        ident = quote.get("ident") or ""
        if not ident or quote.get("probability") is None:
            continue
        results.append({
            "platform": quote.get("platform"),
            "ident": ident,
            "question": quote.get("question"),
            "market_url": quote.get("market_url"),
            "probability": quote.get("probability"),
            "close_time": quote.get("close_time"),
            "volume": quote.get("volume"),
            "category": quote.get("category"),
        })
    payload = {"results": results[:limit], "query": query, "category": cat, "source": "python"}
    _cache_set(cache_key, payload, 60)
    return JSONResponse(payload, headers={"Cache-Control": "public, max-age=60"})


@app.get("/watchlist", include_in_schema=False)
async def watchlist_page(request: Request) -> Response:
    """Serve the SPA at a real URL so the watchlist can open in its own window."""
    return _render_static_html_page(
        "index.html",
        request,
        page="watchlist",
        cache_control="no-cache",
    )


@app.get("/trade", include_in_schema=False)
async def trade_page(request: Request) -> Response:
    """Serve a dedicated professional trading dashboard in its own window."""
    return _render_static_html_page(
        "trade.html",
        request,
        page="trade",
        cache_control="no-cache",
    )


@app.post("/rag/ingest", tags=["Knowledge"], summary="Add a document to your knowledge base")
async def rag_ingest(req: RagIngestRequest, request: Request) -> Dict[str, Any]:
    """Chunk, embed, and store a document (text or URL) in the signed-in user's
    knowledge base, so the agent can retrieve it as evidence."""
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    title, text, url = req.title, (req.text or "").strip(), (req.url or "").strip()
    source = "Knowledge base"
    if not text and url:
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=422, detail="url must start with http:// or https://")
        extracted = await loop.run_in_executor(None, _extract_url_sync, url)
        text = (extracted or {}).get("text") or ""
        title = title or (extracted or {}).get("title") or url
        source = (extracted or {}).get("source") or url
    if not text:
        raise HTTPException(status_code=422, detail="Provide text or a fetchable url.")
    item = {"text": text[:200000], "title": title, "url": url, "source": source}
    try:
        added = await loop.run_in_executor(None, _rag_add, claims["sub"], req.namespace, [item])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"chunks_added": added, "namespace": req.namespace}


@app.get("/rag/search", tags=["Knowledge"], summary="Search your knowledge base", response_model=List[RagSearchResult])
async def rag_search(request: Request, q: str, namespace: str = "kb", top_k: int = 5) -> List[RagSearchResult]:
    claims = _require_session(request)
    if not q.strip():
        raise HTTPException(status_code=422, detail="Provide a query 'q'.")
    loop = asyncio.get_running_loop()
    hits = await loop.run_in_executor(None, _rag_search, claims["sub"], namespace, q, max(1, min(top_k, 20)))
    return [RagSearchResult(**h) for h in hits]


@app.get("/rag/documents", tags=["Knowledge"], summary="List your knowledge-base documents")
async def rag_documents(request: Request, namespace: str = "kb") -> Dict[str, Any]:
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(None, _rag_documents, claims["sub"], namespace)
    return {"namespace": namespace, "documents": docs}


@app.delete("/rag/documents", tags=["Knowledge"], summary="Delete knowledge-base documents")
async def rag_delete(request: Request, namespace: str = "kb", doc_id: Optional[str] = None) -> Dict[str, int]:
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    removed = await loop.run_in_executor(None, _rag_delete, claims["sub"], namespace, doc_id)
    return {"removed": removed}


def _council_provider(label: str):
    """Return the LLM provider for a council-member model label."""
    if label == _state.get("model_key"):
        if os.environ.get("SCADS_AI_API_KEY") and label in _SCADS_MODEL_ALLOWLIST:
            return _scads_alt_provider(
                label,
                allow_default=True,
                request_timeout_s=_COUNCIL_MEMBER_TIMEOUT_S,
            )
        return _state.get("provider")
    return _scads_alt_provider(
        label,
        request_timeout_s=_COUNCIL_MEMBER_TIMEOUT_S,
    )


async def _council_forecast(
    messages: List[Dict[str, str]],
    req: "PredictRequest",
    evidence_articles: List[Dict[str, Any]],
    evidence_error: Optional[str],
) -> "PredictResponse":
    """Two-round debate between all SCADS LLM models; returns the consensus.

    Round 1: all models forecast independently from the same evidence.
    Round 2: each model sees the others' Round-1 probability + rationale and
             gives a revised estimate, correcting for groupthink or stale evidence.
    Final: median of Round-2 probabilities (robust to one outlier)."""
    temperature = _state.get("temperature", 0.0)
    max_tokens = _state.get("max_tokens", 1024)
    council_models = list(_SCADS_MODEL_ALLOWLIST.keys())

    async def _call(label: str, msgs: List[Dict], round_number: int) -> tuple:
        from opentelemetry.trace import Status, StatusCode

        started = time.perf_counter()
        outcome = "error"
        attributes = {
            "forecast.council.member": label,
            "forecast.council.round": round_number,
        }
        with _tracer.start_as_current_span("forecast.council.member") as span:
            span.set_attributes(attributes)
            try:
                provider = _council_provider(label)
                if provider is None:
                    raise RuntimeError("council provider is unavailable")
                content = await _provider_chat(
                    provider,
                    msgs,
                    temperature,
                    max_tokens,
                    timeout_s=_COUNCIL_MEMBER_TIMEOUT_S,
                    max_retries=0,
                    call_site=f"council_round_{round_number}",
                )
                outcome = "success"
                span.set_attribute("outcome", outcome)
                return label, content
            except Exception as exc:
                outcome = "timeout" if isinstance(exc, asyncio.TimeoutError) else "error"
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("outcome", outcome)
                logger.warning(
                    "council member failed round=%d model=%s error=%s",
                    round_number,
                    label,
                    type(exc).__name__,
                )
                return label, None
            finally:
                metric_attributes = {**attributes, "outcome": outcome}
                _council_member_calls.add(1, metric_attributes)
                _council_member_duration.record(
                    time.perf_counter() - started,
                    metric_attributes,
                )

    def _extract_prob(content: Optional[str]) -> Optional[Dict]:
        if not content:
            return None
        parsed = _parse_json_dict(content)
        if not parsed:
            return None
        conf = parsed.get("confidence")
        if conf is None:
            return None
        answer = (parsed.get("predicted_answer") or "").strip().lower()
        if answer not in ("yes", "y", "no", "n"):
            return None
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            return None
        if not 0.0 <= confidence <= 1.0:
            return None
        prob = confidence if answer in ("yes", "y") else 1.0 - confidence
        return {
            "probability": max(0.01, min(0.99, prob)),
            "rationale": (parsed.get("rationale") or "")[:300],
        }

    # Round 1: independent forecasts ─────────────────────────────────────────
    r1_raw = dict(await asyncio.gather(*[_call(m, messages, 1) for m in council_models]))
    r1 = {m: _extract_prob(c) for m, c in r1_raw.items()}
    r1 = {m: v for m, v in r1.items() if v}

    required_members = max(
        1, min(_COUNCIL_MIN_SUCCESSFUL_MEMBERS, len(council_models))
    )
    if len(r1) < required_members:
        from analyzing_llm_rationale.providers import RetryableProviderError

        _council_requests.add(1, {"outcome": "insufficient_members"})
        logger.error(
            "council quorum unavailable successful=%d required=%d configured=%d",
            len(r1),
            required_members,
            len(council_models),
        )
        raise RetryableProviderError("council quorum unavailable")

    r2: Dict[str, Dict] = {}
    if len(r1) >= 2:
        # A second round is useful only when members can react to another model.
        lines = ["Independent forecasters' initial estimates:\n"]
        for label, v in r1.items():
            pct = round(v["probability"] * 100)
            snippet = v["rationale"].replace("\n", " ")
            lines.append(f"• {label} ({pct}% YES): \"{snippet}\"")
        lines.append(
            "\nReview each estimate carefully. Flag stale evidence, past events "
            "misread as ongoing, or reasoning errors you spot in ANY of the above. "
            "Provide your REVISED probability in the same JSON format — update "
            "substantially if you identify a clear error."
        )
        debate_msg: Dict[str, str] = {"role": "user", "content": "\n".join(lines)}
        r2_messages = messages + [debate_msg]

        r2_raw = dict(await asyncio.gather(*[_call(m, r2_messages, 2) for m in r1]))
        r2 = {m: _extract_prob(c) for m, c in r2_raw.items()}
        r2 = {m: v for m, v in r2.items() if v}

    # Consensus: median of revised probabilities (fall back to R1 if R2 missing)
    final = {**r1, **r2}
    probs = sorted(v["probability"] for v in final.values())
    consensus = probs[len(probs) // 2]

    # Build rationale showing the debate ─────────────────────────────────────
    debate_lines = ["[Council debate]"]
    for label in council_models:
        v1, v2 = r1.get(label), r2.get(label)
        if not v1:
            continue
        if v2 and abs(v2["probability"] - v1["probability"]) >= 0.05:
            debate_lines.append(
                f"{label}: {round(v1['probability']*100)}% → {round(v2['probability']*100)}% "
                f"— {v2['rationale']}"
            )
        else:
            v = v2 or v1
            debate_lines.append(f"{label}: {round(v['probability']*100)}% — {v['rationale']}")
    rationale = "\n".join(debate_lines)

    consensus_answer = "Yes" if consensus >= 0.5 else "No"
    consensus_conf = consensus if consensus >= 0.5 else 1.0 - consensus
    synthetic = {
        "type": "binary",
        "predicted_answer": consensus_answer,
        "confidence": round(consensus_conf, 4),
        "rationale": rationale,
    }
    _council_requests.add(1, {"outcome": "success"})
    req_binary = req.model_copy(update={"question_type": "binary"})
    return _build_typed_response(
        req_binary, synthetic, json.dumps(synthetic), evidence_articles, evidence_error
    )


@app.post(
    "/predict",
    tags=["Inference"],
    summary="Run a single forecasting prediction",
    response_description="Prediction result with confidence score, rationale, and evidence sources.",
    responses={
        200: {"description": "Prediction returned successfully."},
        400: {"description": "Invalid request — unknown variant or malformed input."},
        401: {"description": "Missing or invalid `X-API-Key` header (only when API key is configured)."},
        429: {"description": "Rate limit exceeded. Retry after 60 seconds."},
        503: {"description": "Server not yet initialised — LLM provider not loaded."},
    },
    response_model=PredictResponse,
)
async def predict(req: PredictRequest, request: Request = None, kb_user_id: Optional[str] = None) -> PredictResponse:
    """Submit a forecasting question and receive a typed structured prediction.

    The model returns:
    - **`question_type`** — `binary`, `multiple_choice`, `numeric`, or `date`
    - **`predicted_answer`** — `Yes`/`No`, top option, or median estimate
    - **`confidence`** — probability (0–1) for binary and multiple-choice answers
    - **`options`** — per-option probabilities for multiple-choice questions
    - **`range_forecast`** — p10/p50/p90 bounds for numeric and date questions
    - **`rationale`** — 2–4 sentence explanation
    - **`evidence_sources`** — news articles used as context
    - **`market_analysis`** — model-vs-market edge when `market_probability` is provided

    ### Choosing a variant

    Start with `variant0_neutral_baseline` (the default). Switch to other variants
    to inject additional structure into the prompt — for example, `variant3_reasoning_type`
    asks the model to identify whether the prediction is based on speculation, an expert
    forecast, or a stated plan.

    ### Providing your own evidence

    Pass pre-fetched articles in `news_articles` to use them directly.
    Leave the list empty to trigger automatic news retrieval (if the evidence
    pipeline is configured on the server).

    ### Example — binary request

    ```bash
    curl -X POST /predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "Will oil prices exceed $100 per barrel in 2026?",
        "question_type": "binary",
        "market_platform": "Kalshi",
        "market_probability": 0.31
      }'
    ```

    ### Example — multiple choice

    ```bash
    curl -X POST /predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "Who will win the 2026 Formula 1 drivers championship?",
        "question_type": "multiple_choice",
        "options": ["Max Verstappen", "Lando Norris", "Charles Leclerc", "Lewis Hamilton", "Other"],
        "attach_evidence": false
      }'
    ```

    ### Example — numeric forecast with context

    ```bash
    curl -X POST /predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "What will US CPI inflation be in December 2026?",
        "question_type": "numeric",
        "resolution_criteria": "Use the year-over-year CPI-U inflation rate for December 2026.",
        "categories": ["Economics", "United States"],
        "variant": "variant5_key_conditions"
      }'
    ```
    """
    claims = None
    if request is not None:
        _check_rate_limit(request)
        _check_predict_rate_limit(request)
        claims = _optional_predict_claims(request)

    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    variants = _state["variants"]
    if req.variant not in variants:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant '{req.variant}'. Valid: {sorted(variants)}",
        )

    # Signed-in users get personalised (knowledge-base) evidence, so their
    # responses must never be shared via the cache.
    rag_user_id = kb_user_id or (claims.get("sub") if claims else None)

    # Serve identical forecasts from cache to cut latency and model spend.
    # Signed-in users use a per-user scope, so personalised context is not shared.
    predict_cache_key = _predict_cache_key(req, rag_user_id)
    if predict_cache_key is not None:
        cached = _cache_get(predict_cache_key)
        if cached is not None:
            return PredictResponse(**cached)

    messages, evidence_articles, evidence_error = await _prepare_predict_messages(req, rag_user_id)

    if req.model == "council":
        try:
            response = await _council_forecast(messages, req, evidence_articles, evidence_error)
        except Exception as exc:
            _forecast_errors.add(1, {
                "forecast.variant": req.variant or "unknown",
                "error.type": type(exc).__name__,
            })
            raise _provider_http_error(exc) from exc
        await _finalize_predict_response(req, response, rag_user_id, predict_cache_key)
        return response

    provider, temperature, max_tokens = _select_predict_provider(req)

    try:
        content, served_provider = await _provider_chat_with_chat_fallbacks(
            req, provider, messages, temperature, max_tokens
        )
    except Exception as exc:
        _forecast_errors.add(1, {
            "forecast.variant": req.variant or "unknown",
            "error.type": type(exc).__name__,
        })
        raise _provider_http_error(exc, model_name=getattr(provider, "model_name", None)) from exc

    if req.chat_mode:
        response = _build_chat_response(
            req,
            content,
            evidence_articles,
            evidence_error,
            _provider_served_model_name(served_provider),
        )
    else:
        parsed = _parse_json_dict(content)
        response = _build_typed_response(
            req,
            parsed,
            content,
            evidence_articles,
            evidence_error,
            _provider_served_model_name(served_provider),
        )

    await _finalize_predict_response(req, response, rag_user_id, predict_cache_key)
    return response


async def _finalize_predict_response(
    req: PredictRequest,
    response: PredictResponse,
    rag_user_id: Optional[str],
    predict_cache_key: Optional[str] = None,
) -> None:
    """Apply best-effort side effects shared by blocking and streaming forecasts."""
    _append_chat_source_attribution(response)

    _forecast_counter.add(1, {
        "forecast.variant": req.variant or "unknown",
        "forecast.question_type": response.question_type or "unknown",
        "forecast.model": response.model_key or "default",
    })

    if predict_cache_key is not None:
        _cache_set(predict_cache_key, response.model_dump(), _PREDICT_CACHE_TTL)

    # Evolution loop: enrol the market this forecast was made against so the
    # track-record Action tracks + scores it (a Datastore pointer only — never the
    # track-record store). Prefer the structured market_analysis; fall back to a
    # market URL pasted into the question text.
    from analyzing_llm_rationale import track_record_live as _trl
    _ma = getattr(response, "market_analysis", None)
    if _ma is not None and _ma.market_url and _ma.model_probability is not None:
        _ident = _trl.ident_from_url(_ma.platform or "", _ma.market_url)
        await _enroll_market(_ma.platform, _ident, _ma.market_url, req.question, "predict")
    else:
        _purl = _parse_market_url(req.question or "")
        _m = re.search(r"https?://\S+", req.question or "")
        if _purl and _m:
            _venue, _kind, _id = _purl
            await _enroll_market(_venue, _id, _m.group(0).rstrip(").,"), req.question, "predict")

    # Best-effort: index the forecast for "search my past forecasts", but only if
    # the embedder is already loaded — never pay a cold start on the forecast path.
    if rag_user_id and rag.is_loaded():
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _rag_add, rag_user_id, "forecasts", [{
                "text": f"Q: {req.question}\nForecast: {response.predicted_answer or response.question_type} — "
                        f"{response.model_rationale or response.rationale or ''}",
                "title": req.question[:300], "source": "Past forecast",
            }])
        except Exception:
            pass


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post(
    "/predict/stream",
    tags=["Inference"],
    summary="Stream a conversational forecasting response",
    response_description="Server-sent events containing model text deltas and a final PredictResponse payload.",
)
async def predict_stream(req: PredictRequest, request: Request) -> StreamingResponse:
    """Stream model output as server-sent events, for all question types.

    Delta events carry raw text chunks as they arrive so the UI can render the
    rationale progressively. The final ``done`` event contains the same fully-
    parsed ``PredictResponse`` shape that ``/predict`` returns, including
    structured fields (``predicted_answer``, ``confidence``, etc.) for binary,
    numeric, and multiple-choice questions.
    """
    _check_rate_limit(request)
    _check_predict_rate_limit(request)
    claims = _optional_predict_claims(request)

    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    variants = _state["variants"]
    if req.variant not in variants:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant '{req.variant}'. Valid: {sorted(variants)}",
        )

    rag_user_id = claims.get("sub") if claims else None

    async def events():
        stream_started = time.monotonic()
        stream_attrs = {
            "forecast.variant": req.variant or "unknown",
            "forecast.model": _model_key_for_request(req),
        }
        span = otel_trace.get_current_span()
        yield _sse_event("meta", {
            "status": "preparing",
            "variant": req.variant,
            "model_key": _model_key_for_request(req),
            "elapsed_ms": 0,
        })
        prepare_started = time.monotonic()
        predict_cache_key = _predict_cache_key(req, rag_user_id)
        if predict_cache_key is not None:
            cached = _cache_get(predict_cache_key)
            if cached is not None:
                response = PredictResponse(**cached)
                prepare_elapsed = time.monotonic() - prepare_started
                first_delta_ms = round((time.monotonic() - stream_started) * 1000)
                _forecast_stream_prepare_duration.record(
                    prepare_elapsed,
                    {**stream_attrs, "outcome": "cache_hit"},
                )
                _forecast_stream_first_token_duration.record(
                    time.monotonic() - stream_started,
                    {**stream_attrs, "outcome": "cache_hit"},
                )
                span.set_attribute("forecast.stream.cache_hit", True)
                span.set_attribute("forecast.stream.prepare_ms", round(prepare_elapsed * 1000))
                span.set_attribute("forecast.stream.first_delta_ms", first_delta_ms)
                yield _sse_event("meta", {
                    "status": "streaming",
                    "variant": req.variant,
                    "model_key": response.model_key,
                    "prepare_ms": round((time.monotonic() - stream_started) * 1000),
                    "cache_hit": True,
                    "evidence_sources": [s.model_dump(mode="json") for s in response.evidence_sources],
                    "evidence_articles": [a.model_dump(mode="json") for a in response.evidence_articles],
                    "evidence_error": response.evidence_error,
                })
                text = (
                    response.rationale
                    or response.model_rationale
                    or response.predicted_answer
                    or ""
                )
                if text:
                    yield _sse_event("delta", {
                        "text": text,
                        "first_delta_ms": first_delta_ms,
                        "provider_first_delta_ms": 0,
                        "cache_hit": True,
                    })
                yield _sse_event("done", {"response": response.model_dump(mode="json")})
                return

        try:
            messages, evidence_articles, evidence_error = await _prepare_predict_messages(
                req, rag_user_id
            )
            if req.model != "council":
                provider, temperature, max_tokens = _select_predict_provider(req)
            prepare_elapsed = time.monotonic() - prepare_started
            _forecast_stream_prepare_duration.record(
                prepare_elapsed,
                {**stream_attrs, "outcome": "success"},
            )
            span.set_attribute("forecast.stream.prepare_ms", round(prepare_elapsed * 1000))
        except HTTPException as exc:
            _forecast_stream_prepare_duration.record(
                time.monotonic() - prepare_started,
                {**stream_attrs, "outcome": "error"},
            )
            yield _sse_event("error", {"status_code": exc.status_code, "detail": exc.detail})
            return
        except Exception:
            _forecast_stream_prepare_duration.record(
                time.monotonic() - prepare_started,
                {**stream_attrs, "outcome": "error"},
            )
            logger.exception("predict stream setup failed")
            yield _sse_event("error", {
                "status_code": 500,
                "detail": "The streaming request could not be prepared.",
            })
            return

        yield _sse_event("meta", {
            "status": "streaming",
            "variant": req.variant,
            "model_key": _model_key_for_request(req),
            "prepare_ms": round((time.monotonic() - stream_started) * 1000),
            "evidence_sources": [s.model_dump(mode="json") for s in _evidence_sources(evidence_articles)],
            "evidence_articles": [a.model_dump(mode="json") for a in _news_articles(evidence_articles)],
            "evidence_error": evidence_error,
        })

        # Council forecasts are an orchestration path, not a stream-capable
        # provider. Selecting a provider here silently ran the server default
        # while labelling the response "council".
        if req.model == "council":
            try:
                response = await _council_forecast(
                    messages,
                    req,
                    evidence_articles,
                    evidence_error,
                )
                await _finalize_predict_response(req, response, rag_user_id, predict_cache_key)
            except Exception as exc:
                http_exc = _provider_http_error(exc)
                yield _sse_event("error", {
                    "status_code": http_exc.status_code,
                    "detail": http_exc.detail,
                })
                return
            yield _sse_event("done", {"response": response.model_dump(mode="json")})
            return

        chunks: List[str] = []
        first_delta_sent = False
        used_provider_ref: Dict[str, Any] = {"provider": provider}
        provider_stream_started = time.monotonic()
        try:
            async for chunk in _provider_stream_chat_with_chat_fallbacks(
                req,
                provider,
                messages,
                temperature,
                max_tokens,
                used_provider_ref,
            ):
                if await request.is_disconnected():
                    return
                chunks.append(chunk)
                payload = {"text": chunk}
                if not first_delta_sent:
                    first_delta_sent = True
                    first_token_elapsed = time.monotonic() - stream_started
                    _forecast_stream_first_token_duration.record(
                        first_token_elapsed,
                        {**stream_attrs, "outcome": "success"},
                    )
                    first_delta_ms = round(first_token_elapsed * 1000)
                    provider_first_delta_ms = round((time.monotonic() - provider_stream_started) * 1000)
                    span.set_attribute("forecast.stream.first_delta_ms", first_delta_ms)
                    span.set_attribute("forecast.stream.provider_first_delta_ms", provider_first_delta_ms)
                    payload["first_delta_ms"] = first_delta_ms
                    payload["provider_first_delta_ms"] = provider_first_delta_ms
                yield _sse_event("delta", payload)
            if not chunks:
                fallback_started = time.monotonic()
                text, served_provider = await _provider_chat_with_chat_fallbacks(
                    req,
                    provider,
                    messages,
                    temperature,
                    max_tokens,
                )
                used_provider_ref["provider"] = served_provider
                text = text.strip()
                if text:
                    chunks.append(text)
                    first_delta_sent = True
                    first_token_elapsed = time.monotonic() - stream_started
                    _forecast_stream_first_token_duration.record(
                        first_token_elapsed,
                        {**stream_attrs, "outcome": "stream_empty_fallback"},
                    )
                    first_delta_ms = round(first_token_elapsed * 1000)
                    provider_first_delta_ms = round((time.monotonic() - fallback_started) * 1000)
                    span.set_attribute("forecast.stream.first_delta_ms", first_delta_ms)
                    span.set_attribute("forecast.stream.provider_first_delta_ms", provider_first_delta_ms)
                    span.set_attribute("forecast.stream.empty_fallback", True)
                    yield _sse_event("delta", {
                        "text": text,
                        "first_delta_ms": first_delta_ms,
                        "provider_first_delta_ms": provider_first_delta_ms,
                        "stream_empty_fallback": True,
                    })
        except Exception as exc:
            if not first_delta_sent:
                _forecast_stream_first_token_duration.record(
                    time.monotonic() - stream_started,
                    {**stream_attrs, "outcome": "error"},
                )
            http_exc = _provider_http_error(exc)
            yield _sse_event("error", {
                "status_code": http_exc.status_code,
                "detail": http_exc.detail,
            })
            return

        text = "".join(chunks).strip()
        served_model_name = _provider_served_model_name(used_provider_ref.get("provider"))
        if req.chat_mode:
            response = _build_chat_response(
                req,
                text,
                evidence_articles,
                evidence_error,
                served_model_name,
            )
        else:
            parsed = parse_model_response(text, ("type", "predicted_answer", "confidence",
                                                 "rationale", "options", "p10", "p50", "p90", "unit"))
            response = _build_typed_response(
                req,
                parsed,
                text,
                evidence_articles,
                evidence_error,
                served_model_name,
            )
        await _finalize_predict_response(req, response, rag_user_id, predict_cache_key)
        yield _sse_event("done", {"response": response.model_dump(mode="json")})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/vertex-predict",
    tags=["Inference"],
    summary="Vertex AI batch prediction (instances wrapper)",
    response_description="Array of prediction results, one per input instance.",
    responses={
        200: {"description": "All predictions returned successfully."},
        400: {"description": "One or more instances are malformed."},
        429: {"description": "Rate limit exceeded."},
        503: {"description": "Server not yet initialised."},
    },
    response_model=VertexPredictResponse,
    include_in_schema=False,
)
async def vertex_predict(req: VertexPredictRequest, request: Request = None) -> VertexPredictResponse:
    """Vertex AI-compatible prediction endpoint.

    Wraps `/predict` in the Vertex AI contract:
    - **Request**: `{"instances": [PredictRequest, ...]}`
    - **Response**: `{"predictions": [PredictResponse, ...]}`

    Instances are processed sequentially. Maximum 10 instances per call.

    This endpoint is called automatically by the Vertex AI SDK and REST API.
    For direct use, prefer `/predict` instead.

    ```python
    from google.cloud import aiplatform

    endpoint = aiplatform.Endpoint("projects/.../endpoints/7325853011580813312")
    response = endpoint.predict(instances=[
        {"question": "Will oil prices exceed $100 per barrel in 2025?"}
    ])
    ```
    """
    if request is not None:
        _check_rate_limit(request)
        _check_predict_rate_limit(request)
        _check_api_key(request)
    predictions = []
    for instance in req.instances:
        result = await predict(PredictRequest(**instance))
        predictions.append(result.model_dump())
    return VertexPredictResponse(predictions=predictions)


# ── Agent: orchestrated, autonomous analysis ─────────────────────────────────

_AGENT_SKILL_SYSTEM = (
    "You are one analysis skill inside a forecasting agent. Apply the requested "
    "skill to the question, the agent's current forecast, and the evidence. "
    "Respond in 2-5 sentences of clear natural language (light markdown is fine). "
    "Do not output JSON or restate the whole forecast — add the specific insight "
    "the skill asks for."
)

# Kept as a separate, small follow-up call rather than folded into
# _AGENT_SKILL_SYSTEM above so the other three built-in skills' plain-prose
# contract (and their "do not output JSON" instruction) stays untouched.
_RED_TEAM_VERDICT_SYSTEM = (
    "Classify a red-team argument made against a forecast. Respond with ONLY a "
    "JSON object: {\"credible\": true|false, \"severity\": \"low\"|\"medium\"|\"high\"}. "
    "\"credible\" means the argument rests on a real mechanism or evidence, not "
    "just a contrarian restatement. \"severity\" is how much this argument should "
    "weigh against the forecast if it is credible. No other text."
)


# Alternate server-hosted SCADS models the public API may forecast with (using
# the server's own key) — for the multi-model paper-trading comparison.
_SCADS_BASE_URL = os.environ.get("SCADS_BASE_URL", "https://llm.scads.ai/v1/chat/completions")
try:
    _SCADS_MODEL_ALLOWLIST = scads_hosted_model_allowlist(_REPO_ROOT / "configs" / "models.yaml")
    _SCADS_CHAT_MODEL_OPTIONS = scads_chat_model_options(_REPO_ROOT / "configs" / "models.yaml")
    _SCADS_MODEL_FALLBACKS = scads_hosted_model_fallbacks(_REPO_ROOT / "configs" / "models.yaml")
except Exception as exc:  # pragma: no cover - defensive production fallback.
    logger.warning("failed to load SCADS model allowlist from config: %s", exc)
    _SCADS_MODEL_ALLOWLIST = {
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gemma-4-26b-a4b-it": "google/gemma-4-26B-A4B-it",
        "gemma-4-31b-it": "google/gemma-4-31B-it",
        "scads-alias-reasoning": "alias-reasoning",
        "kimi-k3": "moonshotai/Kimi-K3",
        "kimi-k2.7-code": "moonshotai/Kimi-K2.7-Code",
    }
    _SCADS_CHAT_MODEL_OPTIONS = ()
    _SCADS_MODEL_FALLBACKS = {"gpt-oss-120b": ("google/gemma-4-26B-A4B-it",)}


def _scads_alt_provider(
    model_label: str,
    *,
    allow_default: bool = False,
    request_timeout_s: Optional[float] = None,
):
    """Build a provider for an allowlisted alternate SCADS model using the server's
    own key. Returns None if the label is the server default or not allowlisted."""
    label = (model_label or "").strip()
    if (
        not label
        or (label == _state.get("model_key") and not allow_default)
        or label not in _SCADS_MODEL_ALLOWLIST
    ):
        return None
    scads_key = os.environ.get("SCADS_AI_API_KEY")
    if not scads_key:
        raise HTTPException(status_code=503, detail="Alternate models are not configured on this server.")
    from analyzing_llm_rationale.providers import OpenAICompatibleProvider
    return OpenAICompatibleProvider(
        model_name=_SCADS_MODEL_ALLOWLIST[label],
        api_key=scads_key,
        base_url=_SCADS_BASE_URL,
        request_timeout_s=(
            max(0.001, request_timeout_s)
            if request_timeout_s is not None
            else 120.0
        ),
    )


def _agent_run_event(phase: str, status: str, detail: str) -> Dict[str, str]:
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "status": status,
        "detail": detail[:240],
    }


def _agent_run_request_snapshot(req: AgentAnalyzeRequest) -> Dict[str, Any]:
    """Persist only bounded, non-secret inputs needed to understand a run."""
    return {
        "question": (req.question or "").strip(),
        "platform": (req.platform or req.market_platform or "").strip().lower() or None,
        "market_ident": req.market_ident or req.ticker or req.slug or req.market_id,
        "market_url": req.market_url or None,
        "agent_profile_id": req.agent_profile_id or None,
        "model": req.model or None,
        "builtin_skills": bool(req.builtin_skills),
        "ground_in_record": bool(req.ground_in_record),
        "tool_loop": bool(req.tool_loop),
        "evidence_top_k": int(req.evidence_top_k),
        "benchmark_tools": bool(req.benchmark_tools),
        "max_tool_steps": int(req.max_tool_steps),
    }


def _agent_run_title(question: str) -> str:
    normalized = " ".join((question or "Agent research").split())
    return normalized[:157] + "..." if len(normalized) > 160 else normalized


def _new_agent_run(user_id: str, req: AgentAnalyzeRequest) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    snapshot = _agent_run_request_snapshot(req)
    return _put_agent_run(
        user_id,
        {
            "id": f"agent_run_{secrets.token_urlsafe(12).replace('-', '_')}",
            "status": "running",
            "title": _agent_run_title(snapshot["question"]),
            "question": snapshot["question"],
            "platform": snapshot["platform"],
            "recommendation": None,
            "model_probability": None,
            "market_probability": None,
            "edge": None,
            "agent_profile": None,
            "request": snapshot,
            "report": None,
            "timeline": [_agent_run_event("created", "running", "Research run created")],
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "error_code": None,
            "client_run_key": req.client_run_key,
        },
    )


def _advance_agent_run(
    user_id: str, run: Dict[str, Any], *, phase: str, detail: str, question: Optional[str] = None,
    platform: Optional[str] = None,
) -> Dict[str, Any]:
    if question:
        run["question"] = question
        run["title"] = _agent_run_title(question)
    if platform:
        run["platform"] = platform
    timeline = list(run.get("timeline") or [])
    timeline.append(_agent_run_event(phase, "running", detail))
    run["timeline"] = timeline[-12:]
    run["updated_at"] = datetime.now(timezone.utc).isoformat()
    return _put_agent_run(user_id, run)


async def _persist_agent_run_step(user_id: str, run: Dict[str, Any], step: Dict[str, Any]) -> None:
    """Best-effort incremental persistence of one tool-loop step, keyed by
    step["index"]. A step already present at that index is merged into --
    the start-phase write ({index, thought, action, args, started_at}) is
    later filled in by the completion-phase write ({observation, error,
    completed_at}) on the *same* record, rather than appending a second,
    disconnected entry. This is what lets a still-"started_at"-only step
    show up as visibly stuck (crashed mid-tool-call) instead of silently
    missing. Mutates `run` in place (the same object the caller holds) so
    partial progress survives even if this specific write fails -- the next
    successful step's write carries it forward. Never raises: a persistence
    hiccup must not interrupt the tool loop itself."""
    try:
        steps = list(run.get("steps") or [])
        index = step.get("index")
        existing = next((i for i, s in enumerate(steps) if s.get("index") == index), None)
        if existing is not None:
            steps[existing] = {**steps[existing], **step}
        else:
            steps.append(dict(step))
        run["steps"] = steps
        run["updated_at"] = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _put_agent_run, user_id, run)
        _agent_run_actions.add(1, {"action": "step", "outcome": "success"})
    except Exception:
        _agent_run_actions.add(1, {"action": "step", "outcome": "error"})
        logger.warning("failed to persist agent run step id=%s", run.get("id"), exc_info=True)


def _agent_run_reference(run: Dict[str, Any]) -> AgentRunReference:
    return AgentRunReference(
        id=str(run["id"]),
        status=str(run["status"]),
        created_at=str(run["created_at"]),
        updated_at=str(run["updated_at"]),
    )


def _complete_agent_run(user_id: str, run: Dict[str, Any], report: AgentReport) -> AgentReport:
    now = datetime.now(timezone.utc).isoformat()
    report_snapshot = report.model_dump(mode="json", exclude={"agent_run"})
    timeline = list(run.get("timeline") or [])
    timeline.append(_agent_run_event("completed", "completed", f"Research complete: {report.recommendation}"))
    run.update(
        {
            "status": "completed",
            "title": _agent_run_title(report.question),
            "question": report.question,
            "platform": report.platform,
            "recommendation": report.recommendation,
            "model_probability": report.model_probability,
            "market_probability": report.market_probability,
            "edge": report.edge,
            "agent_profile": report.agent_profile.model_dump(mode="json") if report.agent_profile else None,
            "report": report_snapshot,
            "timeline": timeline[-12:],
            "updated_at": now,
            "completed_at": now,
            "error_code": None,
        }
    )
    stored = _put_agent_run(user_id, run)
    report.agent_run = _agent_run_reference(stored)
    _agent_run_actions.add(1, {"action": "complete", "outcome": "success"})
    logger.info("agent run completed id=%s recommendation=%s", stored["id"], report.recommendation)
    return report


def _fail_agent_run(
    user_id: str, run: Dict[str, Any], *, status: str, error_code: str, detail: str
) -> Dict[str, Any]:
    if run.get("status") != "running":
        return run
    now = datetime.now(timezone.utc).isoformat()
    timeline = list(run.get("timeline") or [])
    timeline.append(_agent_run_event("failed", status, detail))
    run.update(
        {
            "status": status,
            "timeline": timeline[-12:],
            "updated_at": now,
            "completed_at": now,
            "error_code": error_code,
        }
    )
    stored = _put_agent_run(user_id, run)
    _agent_run_actions.add(1, {"action": "complete", "outcome": status})
    logger.warning("agent run ended id=%s status=%s code=%s", stored["id"], status, error_code)
    return stored


def _scheduled_reconcile_stale_agent_runs(stale_minutes: int, limit: int) -> Dict[str, Any]:
    """Mark AgentRun records stuck at status="running" past `stale_minutes` as
    interrupted. Idempotent: _fail_agent_run only acts on a record whose own
    status is still "running", so a run that legitimately completes between
    being listed here and being processed is left untouched, not clobbered."""
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)).isoformat()
    result: Dict[str, Any] = {"checked": 0, "interrupted": 0, "errors": 0}
    for user_id, record in _list_stale_running_agent_runs(cutoff_iso, limit):
        result["checked"] += 1
        try:
            _fail_agent_run(
                user_id, record, status="interrupted", error_code="stale_reconciled",
                detail=f"No progress for over {stale_minutes} minutes; marked interrupted by scheduled reconciliation.",
            )
            result["interrupted"] += 1
        except Exception:
            result["errors"] += 1
            logger.warning("agent run reconciliation could not update run id=%s", record.get("id"))
    return result


def _agent_run_summary(record: Dict[str, Any]) -> AgentRunSummary:
    report = record.get("report") if isinstance(record.get("report"), dict) else {}
    return AgentRunSummary(
        id=str(record["id"]),
        status=str(record["status"]),
        title=str(record.get("title") or "Agent research"),
        question=str(record.get("question") or ""),
        platform=record.get("platform"),
        recommendation=record.get("recommendation"),
        model_probability=record.get("model_probability"),
        market_probability=record.get("market_probability"),
        edge=record.get("edge"),
        agent_profile=record.get("agent_profile"),
        has_live_trade_intent=bool(report.get("live_trade_intent")),
        timeline=list(record.get("timeline") or []),
        created_at=str(record["created_at"]),
        updated_at=str(record["updated_at"]),
        completed_at=record.get("completed_at"),
        error_code=record.get("error_code"),
        client_run_key=record.get("client_run_key"),
    )


def _agent_run_response(record: Dict[str, Any]) -> AgentRunResponse:
    report_payload = record.get("report") if isinstance(record.get("report"), dict) else None
    if report_payload is not None:
        report_payload = {**report_payload, "agent_run": _agent_run_reference(record).model_dump(mode="json")}
    return AgentRunResponse(
        **_agent_run_summary(record).model_dump(mode="json"),
        request=dict(record.get("request") or {}),
        report=AgentReport(**report_payload) if report_payload is not None else None,
        steps=list(record.get("steps") or []),
    )


def _resolve_agent_profile_request(
    req: AgentAnalyzeRequest, user_id: str
) -> tuple[AgentAnalyzeRequest, Optional[AgentProfileReference]]:
    """Resolve an owned copied-agent recipe and strip unsafe client overrides."""
    profile_id = req.agent_profile_id
    if not profile_id:
        return req, None
    with _tracer.start_as_current_span("agent.profile.resolve") as span:
        span.set_attributes({"user.id": user_id, "agent.profile.id": profile_id})
        profile = _read_agent_profile(user_id, profile_id)
        if profile is None:
            _agent_profile_actions.add(1, {"action": "resolve", "outcome": "missing"})
            raise HTTPException(status_code=404, detail="Copied agent was not found.")
        source_agent_id = str(profile.get("source_agent_id") or "")
        model = str(profile.get("model") or "")
        if (
            profile.get("execution_mode") != "research_only"
            or not source_agent_id
            or model not in _SCADS_MODEL_ALLOWLIST
        ):
            _agent_profile_actions.add(1, {"action": "resolve", "outcome": "unavailable"})
            raise HTTPException(status_code=409, detail="This copied agent is no longer available for research.")
        profile_skill = AgentSkill(name=str(profile["name"]), instruction=str(profile["instruction"]))
        effective = req.model_copy(
            update={
                "model": model,
                "skills": [profile_skill, *req.skills[:4]],
                "builtin_skills": True,
                "ground_in_record": True,
                "tool_loop": False,
                "benchmark_tools": False,
                "openrouter_api_key": None,
                "openrouter_model": None,
                "provider_base_url": None,
                "ollama_base_url": None,
            }
        )
        reference = AgentProfileReference(
            id=str(profile["id"]),
            source_agent_id=source_agent_id,
            model=model,
            version=int(profile["version"]),
            execution_mode="research_only",
        )
        span.set_attribute("agent.profile.version", reference.version)
        _agent_profile_actions.add(1, {"action": "resolve", "outcome": "success"})
        return effective, reference


def _scads_provider_for_model_name(
    model_name: str,
    *,
    request_timeout_s: Optional[float] = None,
):
    """Build a direct SCADS provider for a concrete upstream model name."""
    name = (model_name or "").strip()
    if not name:
        return None
    scads_key = os.environ.get("SCADS_AI_API_KEY")
    if not scads_key:
        return None
    from analyzing_llm_rationale.providers import OpenAICompatibleProvider
    return OpenAICompatibleProvider(
        model_name=name,
        api_key=scads_key,
        base_url=_SCADS_BASE_URL,
        request_timeout_s=(
            max(0.001, request_timeout_s)
            if request_timeout_s is not None
            else 120.0
        ),
    )


def _provider_served_model_name(provider) -> Optional[str]:
    value = getattr(provider, "last_response_model", None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _chat_fallback_providers(req: "PredictRequest", primary_provider) -> List[Any]:
    """Concrete fallback providers for interactive chat only.

    Structured forecast jobs keep their requested model identity stable; the
    chat UI can tolerate and surface fallback model use via served_model_name.
    """
    if (
        not req.chat_mode
        or req.openrouter_model
        or req.openrouter_api_key
        or req.provider_base_url
        or req.ollama_base_url
        or req.model == "council"
    ):
        return []
    label = _model_key_for_request(req)
    chain = _SCADS_MODEL_FALLBACKS.get(label, ())
    if not chain:
        return []
    primary_name = getattr(primary_provider, "model_name", None)
    providers = []
    for model_name in chain:
        if model_name == primary_name:
            continue
        provider = _scads_provider_for_model_name(model_name)
        if provider is not None:
            providers.append(provider)
    return providers


def _chat_timeout_s(req: "PredictRequest") -> Optional[float]:
    if not req.chat_mode:
        return None
    return _CHAT_DEEP_PROVIDER_TIMEOUT_S if req.effort_tier == "deep" else _CHAT_PROVIDER_TIMEOUT_S


async def _provider_chat_with_chat_fallbacks(
    req: "PredictRequest",
    provider,
    messages,
    temperature,
    max_tokens,
) -> tuple[str, Any]:
    providers = [provider, *_chat_fallback_providers(req, provider)]
    reasoning_effort = _REASONING_EFFORT_BY_TIER.get(req.effort_tier or "standard")
    last_exc: Optional[Exception] = None
    for idx, candidate in enumerate(providers):
        try:
            content = await _provider_chat(
                candidate,
                messages,
                temperature,
                max_tokens,
                timeout_s=_chat_timeout_s(req),
                max_retries=(_CHAT_PROVIDER_MAX_RETRIES if req.chat_mode else None),
                call_site="predict" if idx == 0 else "predict_fallback",
                reasoning_effort=reasoning_effort,
            )
            return content, candidate
        except Exception as exc:
            last_exc = exc
            if idx == len(providers) - 1:
                break
            logger.warning(
                "chat model failed; trying fallback requested=%s failed_model=%s fallback_model=%s error=%s",
                _model_key_for_request(req),
                getattr(candidate, "model_name", "unknown"),
                getattr(providers[idx + 1], "model_name", "unknown"),
                type(exc).__name__,
            )
    assert last_exc is not None
    raise last_exc


async def _provider_stream_chat_with_chat_fallbacks(
    req: "PredictRequest",
    provider,
    messages,
    temperature,
    max_tokens,
    used_provider_ref: Dict[str, Any],
):
    providers = [provider, *_chat_fallback_providers(req, provider)]
    reasoning_effort = _REASONING_EFFORT_BY_TIER.get(req.effort_tier or "standard")
    for idx, candidate in enumerate(providers):
        used_provider_ref["provider"] = candidate
        emitted = False
        try:
            async for chunk in _provider_stream_chat(
                candidate,
                messages,
                temperature,
                max_tokens,
                first_token_timeout_s=_chat_timeout_s(req),
                reasoning_effort=reasoning_effort,
            ):
                emitted = True
                yield chunk
            return
        except Exception as exc:
            if emitted or idx == len(providers) - 1:
                raise
            logger.warning(
                "chat stream model failed before first token; trying fallback requested=%s failed_model=%s fallback_model=%s error=%s",
                _model_key_for_request(req),
                getattr(candidate, "model_name", "unknown"),
                getattr(providers[idx + 1], "model_name", "unknown"),
                type(exc).__name__,
            )


def _select_provider(
    openrouter_api_key: Optional[str],
    openrouter_model: Optional[str],
    provider_base_url: Optional[str],
    ollama_base_url: Optional[str] = None,
):
    """Return (provider, temperature, max_tokens): BYOK model if given, else the server default."""
    max_tokens = _state.get("max_tokens", 1024)
    if ollama_base_url and openrouter_model:
        from analyzing_llm_rationale.providers import OllamaProvider
        return OllamaProvider(
            model_name=openrouter_model,
            base_url=f"{ollama_base_url}/v1/chat/completions",
        ), 0.7, max_tokens
    if openrouter_api_key and openrouter_model:
        if provider_base_url:
            from analyzing_llm_rationale.providers import OpenAICompatibleProvider
            return (
                OpenAICompatibleProvider(
                    model_name=openrouter_model, api_key=openrouter_api_key, base_url=provider_base_url
                ),
                0.7,
                max_tokens,
            )
        from analyzing_llm_rationale.providers import OpenRouterProvider
        return OpenRouterProvider(model_name=openrouter_model, api_key=openrouter_api_key), 0.7, max_tokens
    return _state["provider"], _state["temperature"], _state["max_tokens"]


def _parse_market_url(text: str) -> Optional[tuple]:
    """Extract (venue, kind, identifier) from a Polymarket/Kalshi URL in `text`.

    Polymarket: the slug is the last path segment
    (e.g. .../lpl/lol-al-we-2026-06-01 -> slug). Kalshi: the segment after
    /markets/ is the ticker. Returns None if no recognised URL is present.
    """
    for token in re.findall(r"https?://\S+", text or ""):
        parsed = urlparse(token)
        host = (parsed.hostname or "").lower()
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            continue
        if "polymarket.com" in host:
            return ("polymarket", "slug", parts[-1])
        if "kalshi.com" in host and "markets" in parts:
            i = parts.index("markets")
            if i + 1 < len(parts):
                return ("kalshi", "ticker", parts[i + 1])
    return None


def _agent_recommendation(edge: Optional[float], outcome: str) -> tuple[str, str]:
    """Deterministic call from the model-vs-market edge."""
    out = outcome or "Yes"
    if edge is None:
        return "no_market_price", "No market price supplied, so no edge could be computed."
    pts = round(abs(edge) * 100)
    if abs(edge) < 0.05:
        return "hold", f"Model is within {pts} pts of the market on {out} — roughly fair value."
    if edge > 0:
        return "buy_yes", f"Model is {pts} pts above the market on {out}; {out} looks underpriced."
    return "buy_no", f"Model is {pts} pts below the market on {out}; {out} looks overpriced."


def _live_trade_intent(
    *,
    platform: Optional[str],
    ident: Optional[str],
    market_url: Optional[str],
    question_type: str,
    recommendation: str,
    model_probability: Optional[float],
    market_probability: Optional[float],
    edge: Optional[float],
) -> Optional[LiveTradeIntent]:
    """Create a *review-only* live-order handoff from a binary agent report.

    The returned probabilities are always for the suggested contract. A
    ``buy_no`` recommendation therefore complements the report's YES-side
    probabilities, which keeps the terminal's research context unambiguous.
    """
    venue = str(platform or "").strip().lower()
    market_ident = str(ident or "").strip()
    with _tracer.start_as_current_span("trading.live_trade_intent.prepare") as span:
        span.set_attributes({
            "market.venue": venue or "unknown",
            "market.id": market_ident[:200] if market_ident else "unknown",
            "trading.recommendation": recommendation or "unknown",
        })
        if (
            question_type != "binary"
            or venue not in {"kalshi", "polymarket"}
            or not market_ident
            or recommendation not in {"buy_yes", "buy_no"}
            or model_probability is None
            or market_probability is None
            or edge is None
        ):
            span.set_attribute("outcome", "skipped")
            _live_trade_intents.add(1, {"outcome": "skipped", "platform": venue or "unknown"})
            return None

        try:
            model_p = float(model_probability)
            market_p = float(market_probability)
            raw_edge = float(edge)
        except (TypeError, ValueError):
            span.set_attribute("outcome", "skipped")
            _live_trade_intents.add(1, {"outcome": "skipped", "platform": venue})
            return None
        if not (0.0 <= model_p <= 1.0 and 0.0 <= market_p <= 1.0):
            span.set_attribute("outcome", "skipped")
            _live_trade_intents.add(1, {"outcome": "skipped", "platform": venue})
            return None

        outcome = "yes" if recommendation == "buy_yes" else "no"
        if outcome == "no":
            model_p, market_p, raw_edge = 1.0 - model_p, 1.0 - market_p, -raw_edge
        intent = LiveTradeIntent(
            platform=venue,
            ident=market_ident,
            action="buy",
            outcome=outcome,
            market_url=market_url or None,
            model_probability=model_p,
            market_probability=market_p,
            edge=raw_edge,
            recommendation=recommendation,
        )
        span.set_attributes({
            "trading.outcome_contract": outcome,
            "outcome": "ready",
        })
        _live_trade_intents.add(1, {"outcome": "ready", "platform": venue})
        logger.info("live trade intent prepared: platform=%s outcome=%s", venue, outcome)
        return intent


def _model_probability_from_prediction(resp: PredictResponse) -> Optional[float]:
    analysis = getattr(resp, "market_analysis", None)
    if analysis is not None and analysis.model_probability is not None:
        return analysis.model_probability
    if getattr(resp, "question_type", None) == "binary" and resp.confidence is not None:
        ans = (resp.predicted_answer or "").strip().lower()
        if ans == "yes":
            return resp.confidence
        if ans == "no":
            return 1.0 - resp.confidence
    return resp.confidence


async def _classify_red_team_argument(
    argument: str, provider, temperature: float, max_tokens: int
) -> Optional[RedTeamVerdict]:
    """Best-effort structured classification of the Red team skill's own
    output, via a small separate follow-up call. Never raises -- a
    classification failure just means no verdict, not a broken skill run."""
    try:
        raw = await _provider_chat(
            provider,
            [
                {"role": "system", "content": _RED_TEAM_VERDICT_SYSTEM},
                {"role": "user", "content": argument[:2000]},
            ],
            temperature,
            min(max_tokens, 150),
            call_site="red_team_verdict",
        )
        obj = json.loads((raw or "").strip())
        return RedTeamVerdict(credible=bool(obj["credible"]), severity=str(obj["severity"]))
    except Exception:
        logger.warning("red team verdict classification failed", exc_info=True)
        return None


async def _run_agent_skill(skill: AgentSkill, context: str, provider, temperature, max_tokens) -> AgentSkillResult:
    messages = [
        {"role": "system", "content": _AGENT_SKILL_SYSTEM},
        {"role": "user", "content": f"{context}\n\nSkill: {skill.name}\nInstruction: {skill.instruction}"},
    ]
    try:
        output = (await _provider_chat(provider, messages, temperature, max_tokens) or "").strip()
    except Exception as exc:
        logger.warning("agent skill %r failed: %s", skill.name, type(exc).__name__)
        return AgentSkillResult(name=skill.name, output="(this analysis step is temporarily unavailable)")
    verdict = None
    if skill.name == "Red team" and output:
        verdict = await _classify_red_team_argument(output, provider, temperature, max_tokens)
    return AgentSkillResult(name=skill.name, output=output, verdict=verdict)


async def _resolve_agent_question(
    req: AgentAnalyzeRequest,
) -> tuple[str, Optional[MarketQuote], Optional[str], List[str]]:
    pipeline: List[str] = []
    quote: Optional[MarketQuote] = None
    venue = (req.platform or req.market_platform or "").strip().lower()
    slug, market_id, ticker = req.slug, req.market_id, req.ticker
    if req.market_ident:
        if "poly" in venue and not (slug or market_id):
            slug = req.market_ident
        elif "kalshi" in venue and not ticker:
            ticker = req.market_ident
    from_url = False
    if not (venue and (slug or market_id or ticker)):
        parsed = _parse_market_url(req.market_url or req.question or "")
        if parsed:
            venue, kind, ident = parsed
            slug = ident if kind == "slug" else None
            ticker = ident if kind == "ticker" else None
            from_url = True
    if venue and (slug or market_id or ticker):
        try:
            if "poly" in venue:
                quote = await _fetch_market_quote("polymarket", slug=slug, market_id=market_id)
            elif "kalshi" in venue:
                quote = await _fetch_market_quote("kalshi", ticker=ticker)
        except HTTPException:
            if not from_url:
                raise
        if quote is not None:
            pipeline.append("resolve_market")

    question = (req.question or "").strip()
    if quote is not None and (not question or _parse_market_url(question)):
        question = quote.question or question
    if not question:
        raise HTTPException(status_code=422, detail="Provide a question, or a platform plus market identifier.")

    grounding_note = None
    if req.ground_in_record:
        agg = await asyncio.get_running_loop().run_in_executor(None, _read_live_track_record)
        grounding_note = agent_capabilities.build_grounding_note(agg) or None
        if grounding_note:
            pipeline.append("ground_in_record")

    return question, quote, grounding_note, pipeline


def _agent_prediction_request(
    req: AgentAnalyzeRequest,
    question: str,
    quote: Optional[MarketQuote],
    grounding_note: Optional[str],
) -> PredictRequest:
    has_market_context = bool(
        quote
        or req.market_url
        or req.market_platform
        or req.platform
        or req.market_ident
        or req.market_probability is not None
    )
    is_greeting = not has_market_context and _is_greeting_or_meta(question)
    history = list(req.history)
    # The self-calibration note is forecast-specific (Brier/ECE, longshot bias) and
    # irrelevant to a greeting -- injecting it as a fake prior "user" turn gives a
    # weaker model nothing to do but echo it back, leaking internal track-record
    # data into what should be a plain reply.
    if grounding_note and not is_greeting:
        history = history + [{
            "role": "user",
            "content": f"[Self-calibration context — apply as a prior, not a hard rule]\n{grounding_note}",
        }]
    # AgentAnalyzeRequest.history allows up to 24 turns but PredictRequest.history
    # caps at 12 -- truncate here (keeping the most recent turns, so an appended
    # grounding_note always survives) or construction below raises a ValidationError.
    history = history[-12:]
    return PredictRequest(
        question=question,
        description=(
            req.description
            or (quote.description if quote and quote.description else "")
        ),
        resolution_criteria=(
            req.resolution_criteria
            or (quote.resolution_criteria if quote and quote.resolution_criteria else "")
        ),
        categories=req.categories,
        news_articles=[
            *(quote.venue_news_articles if quote else []),
            *req.news_articles,
        ],
        # A greeting or meta question ("hello", "what can you do?") with no
        # market attached skips evidence retrieval and the JSON-forecast
        # contract entirely -- otherwise the model is force-fed a "respond
        # with ONLY one JSON forecast object" instruction with no escape
        # hatch, and either hallucinates a fake Yes/No forecast about "hello"
        # or breaks format in a way nothing downstream is designed to expect.
        attach_evidence=not is_greeting,
        chat_mode=is_greeting,
        effort_tier=req.effort_tier,
        evidence_top_k=req.evidence_top_k,
        variant=req.variant,
        history=history,
        conversation_steer=req.conversation_steer,
        market_platform=(quote.platform if quote else (req.platform or req.market_platform)),
        market_ident=(quote.ident if quote else req.market_ident),
        market_url=(quote.market_url if quote else req.market_url),
        market_outcome=(quote.outcome if quote else None),
        market_probability=(quote.probability if quote else req.market_probability),
        market_bid=(quote.yes_bid if quote and quote.yes_bid is not None else req.market_bid),
        market_ask=(quote.yes_ask if quote and quote.yes_ask is not None else req.market_ask),
        market_volume=(quote.volume if quote and quote.volume is not None else req.market_volume),
        market_liquidity=(
            quote.liquidity if quote and quote.liquidity is not None else req.market_liquidity
        ),
        resolve_time=(quote.close_time if quote and quote.close_time else req.resolve_time),
        created_time=(quote.created_time if quote and quote.created_time else None),
        publish_time=(quote.created_time if quote and quote.created_time else None),
        openrouter_api_key=req.openrouter_api_key,
        openrouter_model=req.openrouter_model,
        model=req.model,
        provider_base_url=req.provider_base_url,
    )


def _select_agent_provider(req: "AgentAnalyzeRequest"):
    """Provider selection shared by skills and the tool loop. Mirrors
    _select_predict_provider's branch order (explicit model -> BYOK/custom
    endpoint -> ROI-based auto-selection -> server default), so an
    /agent/analyze request without an explicit model benefits from the same
    evolution-loop auto-routing the main forecast already gets, instead of
    always falling straight to the server's static default."""
    alt_provider = _scads_alt_provider(req.model) if req.model else None
    if alt_provider is not None:
        return alt_provider, _state.get("temperature", 0.0), _state.get("max_tokens", 1024)
    if (req.ollama_base_url and req.openrouter_model) or (req.openrouter_api_key and req.openrouter_model):
        return _select_provider(
            req.openrouter_api_key, req.openrouter_model, req.provider_base_url,
            getattr(req, "ollama_base_url", None),
        )
    auto = _auto_selected_model()
    if auto:
        auto_provider = _scads_alt_provider(auto)
        if auto_provider is not None:
            return auto_provider, _state.get("temperature", 0.0), _state.get("max_tokens", 1024)
    return _select_provider(
        req.openrouter_api_key, req.openrouter_model, req.provider_base_url,
        getattr(req, "ollama_base_url", None),
    )


def _should_reduce_builtin_skills(req: "AgentAnalyzeRequest", result: "PredictResponse") -> bool:
    """True only when the heuristic tier says "simple" and nothing about the
    forecast itself contradicts that. This can only ever narrow the simple
    tier back toward standard — it never downgrades a "standard"/"deep" tier,
    since a shallow answer and a shallow self-report share a common cause and
    are not a trustworthy basis for reducing work on their own."""
    if req.effort_tier != "simple":
        return False
    if result.complexity == "high":
        return False
    if (
        result.question_type == "binary"
        and result.confidence is not None
        and 0.45 <= result.confidence <= 0.55
    ):
        return False
    return True


async def _run_agent_skills(
    req: AgentAnalyzeRequest,
    question: str,
    pred_req: PredictRequest,
    result: PredictResponse,
) -> tuple[List[AgentSkillResult], bool, Optional[str]]:
    if pred_req.chat_mode:
        # A greeting/meta reply never tried to forecast anything -- running a
        # research skill (base rate, red team, ...) against "hello" makes no
        # sense and would be pure wasted LLM spend.
        return [], False, None
    skills_to_run: List[AgentSkill] = []
    skill_marker: Optional[str] = None
    if req.builtin_skills:
        builtins = [AgentSkill(**s) for s in agent_capabilities.builtin_skills()]
        if _should_reduce_builtin_skills(req, result):
            builtins = [s for s in builtins if s.name == "Key drivers"]
            skill_marker = "skills_reduced_low_complexity"
        elif req.effort_tier == "simple":
            skill_marker = "skills_escalated_from_self_report"
        skills_to_run.extend(builtins)
    skills_to_run.extend(req.skills)
    if not skills_to_run:
        return [], False, skill_marker

    provider, temperature, max_tokens = _select_agent_provider(req)
    sources_txt = "\n".join(
        f"- {s.source}: {s.title}" for s in result.evidence_sources[:8]
    ) or "(no evidence retrieved)"
    context = (
        f"Question: {question}\n"
        f"Forecast: {result.predicted_answer} "
        f"(confidence {result.confidence if result.confidence is not None else 'n/a'})\n"
        f"Market-implied probability: {pred_req.market_probability}\n"
        f"Thesis: {result.model_rationale or result.rationale or ''}\n"
        f"Evidence:\n{sources_txt}"
    )
    skill_results = await asyncio.gather(
        *(_run_agent_skill(s, context, provider, temperature, max_tokens) for s in skills_to_run)
    )
    return list(skill_results), True, skill_marker


async def _agent_report_from_prediction(
    req: AgentAnalyzeRequest,
    question: str,
    quote: Optional[MarketQuote],
    grounding_note: Optional[str],
    pipeline: List[str],
    pred_req: PredictRequest,
    result: PredictResponse,
    agent_profile: Optional[AgentProfileReference] = None,
) -> AgentReport:
    pipeline = list(pipeline) + ["gather_evidence", "forecast"]
    analysis = result.market_analysis
    edge = analysis.edge if analysis else None
    model_probability = analysis.model_probability if analysis else _model_probability_from_prediction(result)
    stance = analysis.stance if analysis else None
    outcome = (quote.outcome if quote else None) or "Yes"
    recommendation, detail = _agent_recommendation(edge, outcome)
    if analysis is not None:
        pipeline.append("price_edge")

    skill_results, ran_skills, skill_marker = await _run_agent_skills(req, question, pred_req, result)
    if ran_skills:
        pipeline.append("skills")
    if skill_marker:
        pipeline.append(skill_marker)
    pipeline.append("recommend")

    market_platform = quote.platform if quote else (req.platform or req.market_platform)
    market_ident = quote.ident if quote else (
        req.market_ident or req.ticker or req.slug or req.market_id
    )
    market_url = quote.market_url if quote else req.market_url
    live_trade_intent = _live_trade_intent(
        platform=market_platform,
        ident=market_ident,
        market_url=market_url,
        question_type=result.question_type,
        recommendation=recommendation,
        model_probability=model_probability,
        market_probability=(analysis.market_probability if analysis else pred_req.market_probability),
        edge=edge,
    )
    report = AgentReport(
        question=question,
        pipeline=pipeline,
        platform=market_platform,
        market_url=market_url,
        outcome=outcome,
        market_probability=(analysis.market_probability if analysis else pred_req.market_probability),
        model_probability=model_probability,
        edge=edge,
        stance=stance,
        recommendation=recommendation,
        recommendation_detail=detail,
        confidence=result.confidence,
        question_type=result.question_type,
        thesis=result.model_rationale or result.rationale or "",
        evidence_sources=result.evidence_sources,
        evidence_error=result.evidence_error,
        skills=list(skill_results),
        grounding=grounding_note,
        effort_tier=req.effort_tier,
        agent_profile=agent_profile,
        live_trade_intent=live_trade_intent,
    )
    if report.market_url and report.model_probability is not None:
        from analyzing_llm_rationale import track_record_live as _trl
        _ident = _trl.ident_from_url(report.platform or "", report.market_url)
        await _enroll_market(report.platform, _ident, report.market_url, question, "agent_analyze")
    return report


async def _agent_analyze_durable(
    req: AgentAnalyzeRequest, request: Request, claims: Dict[str, Any]
) -> AgentReport:
    """Run one authenticated analysis while recording its private lifecycle."""
    user_id = str(claims["sub"])
    run = _new_agent_run(user_id, req)
    with _tracer.start_as_current_span("agent.run.execute") as span:
        span.set_attributes({"user.id": user_id, "agent.run.id": run["id"]})
        _agent_run_actions.add(1, {"action": "create", "outcome": "success"})
        try:
            agent_profile = None
            if req.agent_profile_id:
                req, agent_profile = _resolve_agent_profile_request(req, user_id)

            question, quote, grounding_note, pipeline = await _resolve_agent_question(req)
            _apply_effort_tier(req, question)
            if agent_profile is not None:
                pipeline.insert(0, "agent_profile")
            run = _advance_agent_run(
                user_id,
                run,
                phase="context_ready",
                detail="Resolved a live market quote" if quote is not None else "Prepared research question",
                question=question,
                platform=(quote.platform if quote else (req.platform or req.market_platform)),
            )

            if req.tool_loop:
                report = await _agent_tool_loop(
                    req, request, question, quote, grounding_note, run=run, user_id=user_id
                )
            else:
                pred_req = _agent_prediction_request(req, question, quote, grounding_note)
                _agent_counter.add(1, {"agent.platform": req.platform or "unknown"})
                result = await predict(pred_req, kb_user_id=user_id)
                report = await _agent_report_from_prediction(
                    req, question, quote, grounding_note, pipeline, pred_req, result, agent_profile
                )
            report = _complete_agent_run(user_id, run, report)
            span.set_attributes({"agent.run.status": "completed", "outcome": "success"})
            return report
        except HTTPException as exc:
            _fail_agent_run(
                user_id, run, status="failed", error_code=f"http_{exc.status_code}", detail="Research run was rejected"
            )
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "rejected")
            raise
        except Exception as exc:
            _fail_agent_run(
                user_id, run, status="failed", error_code="analysis_failed", detail="Research run could not complete"
            )
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "error")
            logger.exception("agent run failed")
            raise


@app.post("/agent/analyze", tags=["Agents"], summary="Run the analysis agent on a live question", response_model=AgentReport)
async def agent_analyze(req: AgentAnalyzeRequest, request: Request = None) -> AgentReport:
    """Orchestrate an end-to-end analysis of a live market question.

    Pipeline: resolve the market (fetch a live Polymarket/Kalshi price when an
    identifier is given) → gather evidence and forecast → compute the model-vs-market
    edge → run any custom **skills** → recommend. Returns one structured report.
    """
    claims = None
    if request is not None:
        _check_rate_limit(request)
        _check_predict_rate_limit(request)
        claims = _require_auth(request)
    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")
    if claims is not None:
        return await _agent_analyze_durable(req, request, claims)

    agent_profile = None
    if req.agent_profile_id:
        if claims is None:
            raise HTTPException(status_code=401, detail="Sign in to use a copied agent.")
        req, agent_profile = _resolve_agent_profile_request(req, claims["sub"])

    question, quote, grounding_note, pipeline = await _resolve_agent_question(req)
    _apply_effort_tier(req, question)
    if agent_profile is not None:
        pipeline.insert(0, "agent_profile")

    # Optional: ReAct tool-using loop instead of the fixed pipeline below.
    if req.tool_loop:
        return await _agent_tool_loop(req, request, question, quote, grounding_note)

    # 2. Evidence + forecast + edge — reuse the /predict pipeline.
    pred_req = _agent_prediction_request(req, question, quote, grounding_note)
    _agent_counter.add(1, {"agent.platform": req.platform or "unknown"})
    result = await predict(pred_req, kb_user_id=(claims.get("sub") if claims else None))
    return await _agent_report_from_prediction(
        req, question, quote, grounding_note, pipeline, pred_req, result, agent_profile
    )


@app.post(
    "/agent/analyze/stream",
    tags=["Agents"],
    summary="Stream the analysis agent on a live question",
    response_description="Server-sent events with forecast deltas and a final AgentReport payload.",
)
async def agent_analyze_stream(req: AgentAnalyzeRequest, request: Request) -> StreamingResponse:
    """Stream the agent's LLM forecast thesis while preserving the final report shape.

    The fixed pipeline streams the underlying forecast generation as `delta` events,
    then emits the same structured `AgentReport` as `/agent/analyze` in `done`.
    Tool-loop mode remains available through the blocking endpoint because its model
    turns are interleaved with tool calls rather than one continuous answer.
    """
    _check_rate_limit(request)
    claims = _require_auth(request)
    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")
    agent_profile = None
    if req.agent_profile_id:
        req, agent_profile = _resolve_agent_profile_request(req, claims["sub"])
    if req.tool_loop:
        raise HTTPException(status_code=400, detail="Streaming is not supported for tool_loop agent mode.")
    with _tracer.start_as_current_span("agent.run.stream") as span:
        run = _new_agent_run(claims["sub"], req)
        span.set_attributes({"user.id": claims["sub"], "agent.run.id": run["id"], "outcome": "started"})
        _agent_run_actions.add(1, {"action": "create", "outcome": "success"})
        logger.info("agent streaming run created id=%s", run["id"])

    async def events():
        nonlocal run
        yield _sse_event("meta", {"status": "resolving", "agent_run": _agent_run_reference(run).model_dump(mode="json")})
        try:
            question, quote, grounding_note, pipeline = await _resolve_agent_question(req)
            _apply_effort_tier(req, question)
            if agent_profile is not None:
                pipeline.insert(0, "agent_profile")
            run = _advance_agent_run(
                claims["sub"],
                run,
                phase="context_ready",
                detail="Resolved a live market quote" if quote is not None else "Prepared research question",
                question=question,
                platform=(quote.platform if quote else (req.platform or req.market_platform)),
            )
            yield _sse_event("meta", {
                "status": "forecasting",
                "question": question,
                "pipeline": pipeline,
            })
            pred_req = _agent_prediction_request(req, question, quote, grounding_note)
            rag_user_id = claims.get("sub")
            messages, evidence_articles, evidence_error = await _prepare_predict_messages(
                pred_req, rag_user_id
            )
            provider, temperature, max_tokens = _select_predict_provider(pred_req)
        except HTTPException as exc:
            _fail_agent_run(
                claims["sub"], run, status="failed", error_code=f"http_{exc.status_code}", detail="Research run was rejected"
            )
            yield _sse_event("error", {"status_code": exc.status_code, "detail": exc.detail})
            return
        except Exception:
            logger.exception("agent stream setup failed")
            _fail_agent_run(
                claims["sub"], run, status="failed", error_code="analysis_failed", detail="Research run could not be prepared"
            )
            yield _sse_event("error", {
                "status_code": 500,
                "detail": "The streaming agent request could not be prepared.",
            })
            return

        yield _sse_event("meta", {
            "status": "streaming",
            "pipeline": pipeline + ["gather_evidence", "forecast"],
            "evidence_sources": [s.model_dump(mode="json") for s in _evidence_sources(evidence_articles)],
            "evidence_articles": [a.model_dump(mode="json") for a in _news_articles(evidence_articles)],
            "evidence_error": evidence_error,
        })

        chunks: List[str] = []
        try:
            async for chunk in _provider_stream_chat(provider, messages, temperature, max_tokens):
                if await request.is_disconnected():
                    _fail_agent_run(
                        claims["sub"], run, status="interrupted", error_code="client_disconnected", detail="Client disconnected during research"
                    )
                    return
                chunks.append(chunk)
                yield _sse_event("delta", {"text": chunk, "phase": "forecast"})
        except Exception as exc:
            http_exc = _provider_http_error(exc)
            _fail_agent_run(
                claims["sub"], run, status="failed", error_code=f"http_{http_exc.status_code}", detail="Forecasting could not complete"
            )
            yield _sse_event("error", {
                "status_code": http_exc.status_code,
                "detail": http_exc.detail,
            })
            return

        text = "".join(chunks).strip()
        # chat_mode (e.g. a greeting) must go through _build_chat_response, same as
        # predict() and /predict/stream -- otherwise the raw [p:0.XX] marker and
        # conversational prose get force-parsed as a JSON forecast, question_type
        # never becomes "chat", and the frontend renders full report chrome
        # (probability tag, grounding note) around what should be a plain reply.
        if pred_req.chat_mode:
            result = _build_chat_response(pred_req, text, evidence_articles, evidence_error)
        else:
            parsed = parse_model_response(text, ("type", "predicted_answer", "confidence",
                                                 "rationale", "options", "p10", "p50", "p90", "unit"))
            result = _build_typed_response(pred_req, parsed, text, evidence_articles, evidence_error)
        try:
            await _finalize_predict_response(pred_req, result, _optional_user_id(request))
        except Exception:
            logger.exception("agent stream forecast finalisation failed")
            _fail_agent_run(
                claims["sub"], run, status="failed", error_code="analysis_failed", detail="Forecast result could not be finalized"
            )
            yield _sse_event("error", {
                "status_code": 500,
                "detail": "The streaming agent forecast could not be finalized.",
            })
            return

        yield _sse_event("meta", {"status": "skills"})
        try:
            report = await _agent_report_from_prediction(
                req, question, quote, grounding_note, pipeline, pred_req, result, agent_profile
            )
        except Exception:
            logger.exception("agent stream finalisation failed")
            _fail_agent_run(
                claims["sub"], run, status="failed", error_code="analysis_failed", detail="Research report could not be finalized"
            )
            yield _sse_event("error", {
                "status_code": 500,
                "detail": "The streaming agent report could not be finalized.",
            })
            return
        report = _complete_agent_run(claims["sub"], run, report)
        yield _sse_event("done", {"report": report.model_dump(mode="json")})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _agent_tool_loop(req: "AgentAnalyzeRequest", request, question: str,
                           quote: "Optional[MarketQuote]", grounding_note: Optional[str],
                           *, run: Optional[Dict[str, Any]] = None,
                           user_id: Optional[str] = None) -> "AgentReport":
    """ReAct tool-using loop: the model plans and calls tools (forecast, market
    fetch, evidence search, venue scan, track record), then answers. Falls back
    cleanly to a no-edge report if no forecast tool was used.

    When `run`/`user_id` are given (the durable /agent/analyze path), each
    step is persisted to the AgentRun record as it happens via on_step."""
    from analyzing_llm_rationale import market_data

    async def _on_step_start(step: Dict[str, Any]) -> None:
        if run is None or user_id is None:
            return
        await _persist_agent_run_step(
            user_id, run, {**step, "started_at": datetime.now(timezone.utc).isoformat()}
        )

    async def _on_step(step: Dict[str, Any]) -> None:
        if run is None or user_id is None:
            return
        await _persist_agent_run_step(
            user_id, run, {**step, "completed_at": datetime.now(timezone.utc).isoformat()}
        )

    provider, temperature, max_tokens = _select_agent_provider(req)
    loop = asyncio.get_running_loop()
    last: Dict[str, Any] = {}
    agent_id = str(req.model or req.openrouter_model or _state.get("model_key") or "agent")
    tool_ctx = benchmark_tools.ToolContext(
        agent_id=agent_id,
        user_id=_optional_user_id(request),
        model=agent_id,
    )

    async def _tool_place_trade(args):
        return benchmark_tools.observation(
            await loop.run_in_executor(
                None,
                lambda: benchmark_tools.place_trade(args, tool_ctx),
            )
        )

    async def _tool_web_search(args):
        return benchmark_tools.observation(
            await loop.run_in_executor(None, lambda: benchmark_tools.web_search(args))
        )

    async def _tool_manage_notes(args):
        return benchmark_tools.observation(
            await loop.run_in_executor(
                None,
                lambda: benchmark_tools.manage_notes(args, tool_ctx),
            )
        )

    async def _tool_forecast(args):
        q = str(args.get("question") or question)
        mp = args.get("market_probability")
        pred_req = _agent_prediction_request(req, q, quote, grounding_note)
        if mp is not None:
            pred_req = pred_req.model_copy(update={"market_probability": mp})
        r = await predict(pred_req, kb_user_id=_optional_user_id(request))
        a = r.market_analysis
        last.update(answer=r.predicted_answer, confidence=r.confidence,
                    model_probability=(a.model_probability if a else r.confidence),
                    market_probability=(a.market_probability if a else mp),
                    edge=(a.edge if a else None), stance=(a.stance if a else None),
                    thesis=r.model_rationale or r.rationale or "", question_type=r.question_type,
                    evidence_sources=list(r.evidence_sources),
                    evidence_error=r.evidence_error)
        msg = f"Forecast: {r.predicted_answer} (confidence {r.confidence})."
        if a and a.edge is not None:
            msg += (f" Model {round((a.model_probability or 0) * 100)}% vs market "
                    f"{round((a.market_probability or 0) * 100)}%, edge {round(a.edge * 100):+d} pts.")
        return msg + " " + (r.model_rationale or r.rationale or "")[:600]

    async def _tool_get_market(args):
        plat = str(args.get("platform", "")).lower()
        try:
            if "poly" in plat:
                q = await _fetch_market_quote("polymarket", slug=args.get("slug"), market_id=args.get("market_id"))
            elif "kalshi" in plat:
                q = await _fetch_market_quote("kalshi", ticker=args.get("ticker"))
            else:
                return "Specify platform 'polymarket' or 'kalshi' plus a slug/ticker."
        except HTTPException as exc:
            return f"(market fetch failed: {exc.detail})"
        return f"{q.platform}: {q.question} — {q.outcome} at {round((q.probability or 0) * 100)}% ({q.market_url})"

    async def _tool_search_evidence(args):
        ep = _state.get("evidence_pipeline")
        if ep is None:
            return "(evidence retrieval not configured)"
        query = str(args.get("query") or question)
        try:
            arts = await loop.run_in_executor(None, lambda: ep.fetch_summarize_rank(query, top_k=5))
        except Exception:
            return "(evidence retrieval failed)"
        return "\n".join(f"- {a.get('source', '?')}: {a.get('title', '')}" for a in (arts or [])[:6]) or "No evidence found."

    async def _tool_scan(args):
        plat = str(args.get("platform", "polymarket")).lower()
        fn = market_data.list_polymarket if "poly" in plat else market_data.list_kalshi if "kalshi" in plat else None
        if fn is None:
            return "platform must be 'polymarket' or 'kalshi'"
        try:
            qs = await loop.run_in_executor(None, lambda: fn(limit=5, query=args.get("query")))
        except Exception:
            return "(listing failed)"
        return "\n".join(f"- [{q.get('platform')}] {(q.get('question') or '')[:70]} @ "
                         f"{round((q.get('probability') or 0) * 100)}%" for q in qs) or "No markets found."

    async def _tool_track_record(args):
        agg = await loop.run_in_executor(None, _read_live_track_record)
        return agent_capabilities.build_grounding_note(agg) or "No resolved forecasts yet."

    async def _tool_edge_board(args):
        live = await loop.run_in_executor(None, _read_edge_board_record)
        board = (live or {}).get("edge_board") or []
        limit = min(max(1, int(args.get("limit", 5))), 20)
        lines = []
        for item in board[:limit]:
            ident = item.get("ticker") or item.get("slug") or item.get("ident") or "?"
            q = item.get("question") or "?"
            m_prob = item.get("market_probability")
            f_prob = item.get("foresea_probability") or item.get("model_probability")
            edge = item.get("edge")
            m_pct = f"{round(m_prob * 100)}%" if m_prob is not None else "n/a"
            f_pct = f"{round(f_prob * 100)}%" if f_prob is not None else "n/a"
            edge_str = f"{edge:+.1%}" if edge is not None else "n/a"
            lines.append(f"- [{item.get('platform', '?')}] {ident}: \"{q[:60]}\" (mkt: {m_pct}, model: {f_pct}, edge: {edge_str})")
        return "\n".join(lines) or "No edge board opportunities found."

    async def _tool_batch_quotes(args):
        refs = args.get("refs") or args.get("tickers") or args.get("slugs") or []
        if isinstance(refs, str):
            refs = [r.strip() for r in refs.split(",") if r.strip()]
        if not isinstance(refs, list) or not refs:
            return "Pass a list of market tickers/slugs or comma-separated string in 'refs'."
        results = []
        for ref in refs[:10]:
            ref_str = str(ref).strip()
            try:
                if ref_str.startswith("KX") or "-" in ref_str:
                    q = await _fetch_market_quote("kalshi", ticker=ref_str)
                else:
                    q = await _fetch_market_quote("polymarket", slug=ref_str)
                results.append(f"- [{q.platform}] {ref_str}: {q.outcome} at {round((q.probability or 0) * 100)}%")
            except Exception:
                results.append(f"- {ref_str}: fetch failed")
        return "\n".join(results) or "No quotes fetched."

    async def _tool_fetch_api(args):
        url = str(args.get("url") or args.get("endpoint") or "").strip()
        if not url:
            return "url argument required."
        if url.startswith("/"):
            url = f"http://127.0.0.1:8080{url}"
        method = str(args.get("method", "GET")).upper()
        if method not in ("GET", "POST"):
            return "method must be 'GET' or 'POST'."
        payload = args.get("data") or args.get("json") or args.get("body")
        try:
            def _http_call():
                import requests
                headers = {"User-Agent": "Foresea-Agent-Tool/1.0", "Accept": "application/json, text/plain"}
                if method == "POST":
                    resp = requests.post(url, json=payload if isinstance(payload, dict) else None, data=payload if isinstance(payload, str) else None, headers=headers, timeout=15)
                else:
                    resp = requests.get(url, headers=headers, timeout=15)
                return f"HTTP {resp.status_code}\n{resp.text[:3500]}"
            return await loop.run_in_executor(None, _http_call)
        except Exception as exc:
            return f"(API request failed: {exc})"
    async def _tool_exchange_status(args):
        try:
            status = await loop.run_in_executor(None, market_data.fetch_kalshi_exchange_status)
            schedule = await loop.run_in_executor(None, market_data.fetch_kalshi_exchange_schedule)
            active = status.get("exchange_active", True) and status.get("trading_active", True)
            res = f"Kalshi Exchange Status: {'ACTIVE' if active else 'PAUSED/INACTIVE'}\nRaw Status: {status}\nSchedule: {schedule}"
            return res[:3500]
        except Exception as exc:
            return f"(Exchange status fetch failed: {exc})"

    async def _tool_orderbook(args):
        ticker = str(args.get("ticker") or args.get("symbol") or "").strip()
        token_id = str(args.get("token_id") or "").strip()
        if not ticker and not token_id:
            return "Specify 'ticker' for Kalshi or 'token_id' for Polymarket."
        try:
            if ticker:
                ob = await loop.run_in_executor(None, lambda: market_data.fetch_kalshi_orderbook(ticker))
                return f"Kalshi Orderbook for {ticker}:\n{json.dumps(ob, indent=2)[:3500]}"
            ob = await loop.run_in_executor(None, lambda: market_data.fetch_polymarket_orderbook(token_id))
            return f"Polymarket Orderbook for token {token_id}:\n{json.dumps(ob, indent=2)[:3500]}"
        except Exception as exc:
            return f"(Orderbook fetch failed: {exc})"

    async def _tool_market_tags(args):
        try:
            tags = await loop.run_in_executor(None, market_data.fetch_polymarket_tags)
            names = [f"- {t.get('label') or t.get('name') or t.get('id', '?')}" for t in (tags or [])[:20]]
            return f"Polymarket Market Categories/Tags ({len(tags)} total):\n" + "\n".join(names)
        except Exception as exc:
            return f"(Market tags fetch failed: {exc})"

    async def _tool_price_history(args):
        ticker = str(args.get("ticker") or args.get("market") or "").strip()
        series_ticker = str(args.get("series_ticker") or "").strip()
        if not ticker:
            return "ticker or market argument required."
        try:
            if ticker.startswith("KX") or "-" in ticker:
                candles = await loop.run_in_executor(None, lambda: market_data.fetch_kalshi_candlesticks(ticker, series_ticker))
                return f"Kalshi Candlesticks for {ticker}:\n{json.dumps(candles[:10], indent=2)[:3500]}"
            hist = await loop.run_in_executor(None, lambda: market_data.fetch_polymarket_price_history(ticker))
            return f"Polymarket Price History for {ticker}:\n{json.dumps(hist[:10], indent=2)[:3500]}"
        except Exception as exc:
            return f"(Price history fetch failed: {exc})"

    async def _tool_live_data(args):
        event_ticker = str(args.get("event_ticker") or args.get("ticker") or "").strip()
        data_type = str(args.get("type") or "").strip()
        try:
            res = await loop.run_in_executor(None, lambda: market_data.fetch_kalshi_live_data(event_ticker, data_type))
            return f"Kalshi Live Data:\n{json.dumps(res, indent=2)[:3500]}"
        except Exception as exc:
            return f"(Live data fetch failed: {exc})"

    async def _tool_polymarket_meta(args):
        target = str(args.get("target") or args.get("resource") or "series").strip().lower()
        market_id = str(args.get("market_id") or args.get("market") or "").strip()
        try:
            if target == "comments":
                res = await loop.run_in_executor(None, lambda: market_data.fetch_polymarket_comments(market_id))
                return f"Polymarket Comments:\n{json.dumps(res[:10], indent=2)[:3500]}"
            if target in ("sports", "teams"):
                res = await loop.run_in_executor(None, market_data.fetch_polymarket_sports)
                return f"Polymarket Sports Metadata:\n{json.dumps(res[:10], indent=2)[:3500]}"
            res = await loop.run_in_executor(None, market_data.fetch_polymarket_series)
            return f"Polymarket Event Series:\n{json.dumps(res[:10], indent=2)[:3500]}"
        except Exception as exc:
            return f"(Polymarket metadata fetch failed: {exc})"

    async def _tool_recent_trades(args):
        platform = str(args.get("platform") or "kalshi").strip()
        ticker = str(args.get("ticker") or args.get("token_id") or args.get("market") or "").strip()
        limit = int(args.get("limit") or 20)
        try:
            res = await loop.run_in_executor(None, lambda: market_data.fetch_recent_trades(platform, ticker, limit))
            return f"Recent Trades ({platform}):\n{json.dumps(res[:limit], indent=2)[:3500]}"
        except Exception as exc:
            return f"(Recent trades fetch failed: {exc})"

    async def _tool_market_leaderboard(args):
        limit = int(args.get("limit") or 20)
        try:
            res = await loop.run_in_executor(None, lambda: market_data.fetch_trader_leaderboard(limit))
            return f"Trader Leaderboard:\n{json.dumps(res[:limit], indent=2)[:3500]}"
        except Exception as exc:
            return f"(Leaderboard fetch failed: {exc})"

    benchmark_tool_map = {
        "place_trade": _tool_place_trade,
        "web_search": _tool_web_search,
        "manage_notes": _tool_manage_notes,
        "get_market": _tool_get_market,
        "scan_markets": _tool_scan,
        "forecast": _tool_forecast,
        "search_evidence": _tool_search_evidence,
        "track_record": _tool_track_record,
        "edge_board": _tool_edge_board,
        "batch_quotes": _tool_batch_quotes,
        "fetch_api": _tool_fetch_api,
        "exchange_status": _tool_exchange_status,
        "orderbook": _tool_orderbook,
        "market_tags": _tool_market_tags,
        "price_history": _tool_price_history,
        "live_data": _tool_live_data,
        "polymarket_meta": _tool_polymarket_meta,
        "recent_trades": _tool_recent_trades,
        "market_leaderboard": _tool_market_leaderboard,
    }
    benchmark_specs = [
        {"name": "place_trade", "args": "ticker, side, price, quantity, platform?", "description": "Buy YES or NO contracts on Kalshi or Polymarket using immediate-or-cancel execution only; unfilled quantity is cancelled and no order rests. Pass platform='kalshi' or platform='polymarket' (defaults to kalshi if omitted) -- ticker is the Kalshi ticker or the Polymarket market slug, matching whichever venue a candidate line came from. There is no sell tool; exiting is represented by buying the opposite side. This tool runs in shadow (paper) mode: no real order ever reaches an exchange and no real money is ever at risk, but every call that passes the guards below DOES execute and permanently update your persistent positions/actions tables with weighted-average entry, netting PnL, settlements, cash, and realized PnL -- it is never a no-op, a preview, or a dry run, and there is no separate 'confirm' step. If you've decided to trade, calling this tool is the only way to actually do it. Trades are guarded by account solvency, a 15% single-market cost-basis cap, and a per-cycle spend limit -- a rejection means one of those guards tripped, not that trading itself is unavailable."},
        {"name": "web_search", "args": "query", "description": "Research market events with OpenAI web search. CoinMarketCap and other blacklisted domains are excluded from results."},
        {"name": "manage_notes", "args": "action, id?, text?, query?, tags?", "description": "Store, search, edit, list, or delete persistent notes. Max 50 notes per agent, 1200 characters each."},
        {"name": "get_market", "args": "platform, slug|ticker", "description": "Fetch a live Polymarket/Kalshi price."},
        {"name": "scan_markets", "args": "platform, query?", "description": "List live markets on a venue (optionally filtered by keyword)."},
        {"name": "forecast", "args": "question, market_probability?", "description": "Produce a probability forecast (with evidence) for a question; pass market_probability to get the edge."},
        {"name": "search_evidence", "args": "query", "description": "Retrieve recent news headlines relevant to a query."},
        {"name": "track_record", "args": "", "description": "Get the model's own live calibration / skill-vs-market."},
        {"name": "edge_board", "args": "limit?", "description": "Get top open prediction market opportunities ranked by model-vs-market edge."},
        {"name": "batch_quotes", "args": "refs", "description": "Fetch batch market quotes for a comma-separated list of tickers or slugs."},
        {"name": "fetch_api", "args": "url, method?, json?", "description": "Execute a GET or POST HTTP API request to an API endpoint URL."},
        {"name": "exchange_status", "args": "", "description": "Get Kalshi exchange operational status (trading_active) and schedule."},
        {"name": "orderbook", "args": "ticker|token_id", "description": "Fetch live orderbook bids/asks for Kalshi ticker or Polymarket token."},
        {"name": "market_tags", "args": "", "description": "Fetch active market categories and tags on Polymarket."},
        {"name": "price_history", "args": "ticker|market, series_ticker?", "description": "Fetch historical prices or OHLC candlesticks for a market."},
        {"name": "live_data", "args": "event_ticker?, type?", "description": "Fetch real-time sports game stats and live event feeds from Kalshi."},
        {"name": "polymarket_meta", "args": "target?, market_id?", "description": "Fetch Polymarket series listings, comments, or sports metadata (target: series|comments|sports)."},
        {"name": "recent_trades", "args": "platform?, ticker?, limit?", "description": "Fetch recent public executed trades / trade tape for Kalshi or Polymarket."},
        {"name": "market_leaderboard", "args": "limit?", "description": "Fetch top profitable prediction market trader leaderboard."},
    ]
    if req.benchmark_tools:
        allowed_names = req.benchmark_tool_names
        if allowed_names is None:
            tools = dict(benchmark_tool_map)
            specs = list(benchmark_specs)
        else:
            allowed = set(allowed_names)
            tools = {name: fn for name, fn in benchmark_tool_map.items() if name in allowed}
            specs = [s for s in benchmark_specs if s["name"] in allowed]
    else:
        tools = {"forecast": _tool_forecast, "get_market": _tool_get_market,
                 "search_evidence": _tool_search_evidence, "scan_markets": _tool_scan,
                 "batch_quotes": _tool_batch_quotes, "fetch_api": _tool_fetch_api,
                 "exchange_status": _tool_exchange_status, "orderbook": _tool_orderbook,
                 "market_tags": _tool_market_tags, "price_history": _tool_price_history,
                 "live_data": _tool_live_data, "polymarket_meta": _tool_polymarket_meta,
                 "recent_trades": _tool_recent_trades, "market_leaderboard": _tool_market_leaderboard}
        specs = [
            {"name": "forecast", "args": "question, market_probability?", "description": "Produce a probability forecast (with evidence) for a question; pass market_probability to get the edge."},
            {"name": "get_market", "args": "platform, slug|ticker", "description": "Fetch a live Polymarket/Kalshi price."},
            {"name": "search_evidence", "args": "query", "description": "Retrieve recent news headlines relevant to a query."},
            {"name": "scan_markets", "args": "platform, query?", "description": "List live markets on a venue (optionally filtered by keyword)."},
            {"name": "track_record", "args": "", "description": "Get the model's own live calibration / skill-vs-market."},
            {"name": "edge_board", "args": "limit?", "description": "Get top open prediction market opportunities ranked by model-vs-market edge."},
            {"name": "batch_quotes", "args": "refs", "description": "Fetch batch market quotes for a comma-separated list of tickers or slugs."},
            {"name": "fetch_api", "args": "url, method?, json?", "description": "Execute a GET or POST HTTP API request to an API endpoint URL."},
            {"name": "exchange_status", "args": "", "description": "Get Kalshi exchange operational status (trading_active) and schedule."},
            {"name": "orderbook", "args": "ticker|token_id", "description": "Fetch live orderbook bids/asks for Kalshi ticker or Polymarket token."},
            {"name": "market_tags", "args": "", "description": "Fetch active market categories and tags on Polymarket."},
            {"name": "price_history", "args": "ticker|market, series_ticker?", "description": "Fetch historical prices or OHLC candlesticks for a market."},
            {"name": "live_data", "args": "event_ticker?, type?", "description": "Fetch real-time sports game stats and live event feeds from Kalshi."},
            {"name": "polymarket_meta", "args": "target?, market_id?", "description": "Fetch Polymarket series listings, comments, or sports metadata (target: series|comments|sports)."},
            {"name": "recent_trades", "args": "platform?, ticker?, limit?", "description": "Fetch recent public executed trades / trade tape for Kalshi or Polymarket."},
            {"name": "market_leaderboard", "args": "limit?", "description": "Fetch top profitable prediction market trader leaderboard."},
        ]

    # Optional: proxy the venues' own MCP tools (orderbook/depth/etc.) when
    # POLYMARKET_MCP_URL / KALSHI_MCP_URL are set. Additive + best-effort; venue
    # output is untrusted context (Foresea's forecast stays the source of truth).
    if venue_mcp.configured_venues():
        try:
            for _ns, _meta in (await venue_mcp.discover_tools()).items():
                tools[_ns] = venue_mcp.make_tool_fn(_meta["url"], _meta["name"])
                specs.append({"name": _ns, "args": "venue-specific JSON",
                              "description": _meta["description"][:200]})
        except Exception:
            logger.warning("venue MCP discovery failed", exc_info=True)

    async def chat_fn(messages):
        return await _provider_chat(provider, messages, temperature, max_tokens)

    q = question
    rule = ("You MUST call the `forecast` tool before your final answer — it produces "
            "the probability and edge the report needs. Never state a probability "
            "without having called `forecast` for it. For example, after finding a "
            "market with scan_markets/get_market, call forecast with that market's "
            "question and its market_probability, then base your answer on that result.")
    if req.benchmark_tools:
        active = set(tools.keys())
        if not active:
            # A specialist-pipeline stage that deliberately hands over zero
            # tools (e.g. a pure sizing/reasoning turn) -- run_tool_loop still
            # expects extra_rules to make sense standing alone.
            rule = "No tools are available this turn. Give your final answer directly."
        else:
            rule_parts = [
                "Benchmark mode: you have {n} tool{s} available this turn: {names}.".format(
                    n=len(active), s="" if len(active) == 1 else "s",
                    names=", ".join(f"`{name}`" for name in sorted(active)),
                )
            ]
            if "place_trade" in active:
                rule_parts.append(
                    "Use `place_trade` for Kalshi or Polymarket trade decisions -- pass "
                    "platform='kalshi' or platform='polymarket' matching whichever venue the "
                    "candidate came from (defaults to kalshi if omitted); ticker is the Kalshi "
                    "ticker or the Polymarket market slug. There is no sell tool; "
                    "exit by buying the opposite side. `place_trade` uses immediate-or-cancel "
                    "execution only: unfilled quantity is cancelled and no order rests. Cash, "
                    "positions, actions, weighted-average entry price, netting PnL, and market "
                    "settlements persist across cycles; settlements are checked before a new "
                    "cycle's trade. This account is shadow (paper) mode: no real order ever "
                    "reaches an exchange and no real money is ever at risk -- but calling "
                    "`place_trade` always actually executes and permanently records the paper "
                    "trade against your persistent account when it passes the guards below. It "
                    "is never disabled, never a no-op, and never waiting on some other 'live "
                    "trading' switch. If you decide to trade this cycle, calling `place_trade` "
                    "is the only way to do it -- do not conclude no trade can happen because "
                    "this is a shadow account. `place_trade` is guarded by account solvency "
                    "including fees/netting payouts, a 15% single-market cost-basis cap, and a "
                    "per-cycle spend limit; a rejection names which specific guard tripped, not "
                    "that trading is unavailable."
                )
            if "web_search" in active:
                rule_parts.append("Use `web_search` for current evidence.")
            if "manage_notes" in active:
                rule_parts.append("Use `manage_notes` for memory across cycles.")
            rule = " ".join(rule_parts)
    if grounding_note:
        # Background context for calibration only -- goes in the system rules,
        # not the user-visible question, and is explicitly marked
        # non-quotable so the model doesn't echo it into its final answer
        # (which the frontend and API both treat as the visible response).
        rule = (
            f"{rule}\n\n[Internal self-calibration context — use this to calibrate "
            "your probability, but do not quote, repeat, or reference this note "
            f"in your final answer]\n{grounding_note}"
        )
    backstopped = False
    try:
        res = await agent_capabilities.run_tool_loop(
            q, tools, specs, chat_fn, max_steps=req.max_tool_steps, extra_rules=rule,
            on_step=_on_step, on_step_start=_on_step_start)
        # Deterministic backstop: if the model answered without ever calling
        # `forecast`, run it ourselves so edge/recommendation always populate.
        if not last and not req.benchmark_tools:
            backstopped = True
            await _tool_forecast({"question": question,
                                  "market_probability": (quote.probability if quote else req.market_probability)})
    except Exception as exc:
        raise _provider_http_error(exc) from exc

    outcome = (quote.outcome if quote else None) or "Yes"
    edge = last.get("edge")
    recommendation, detail = _agent_recommendation(edge, outcome)
    pipeline = ["tool_loop"]
    if req.benchmark_tools:
        pipeline.append("benchmark_tools")
    if last:
        pipeline.append("forecast")
    if grounding_note:
        pipeline.insert(0, "ground_in_record")
    market_platform = quote.platform if quote else (req.platform or req.market_platform)
    market_ident = quote.ident if quote else (
        req.market_ident or req.ticker or req.slug or req.market_id
    )
    market_url = quote.market_url if quote else req.market_url
    live_trade_intent = _live_trade_intent(
        platform=market_platform,
        ident=market_ident,
        market_url=market_url,
        question_type=last.get("question_type", "binary"),
        recommendation=recommendation,
        model_probability=last.get("model_probability"),
        market_probability=last.get("market_probability"),
        edge=edge,
    )
    raw_answer = _SELF_CALIBRATION_ECHO_RE.sub("", res.get("answer", "")).strip()
    use_backstop_thesis = backstopped and bool(last.get("thesis"))
    tool_transcript = list(res.get("transcript", []))
    if use_backstop_thesis and raw_answer:
        # The backstop's forecast becomes the thesis below (so the number it
        # cites stays consistent with edge/recommendation), which would
        # otherwise silently drop the model's own final-turn text -- keep it
        # in the transcript so a malformed or off-contract tool call still
        # leaves a trace in the durable run record.
        tool_transcript.append({
            "action": "final_text_before_backstop",
            "args": {},
            "observation": raw_answer[:4000],
        })
    report = AgentReport(
        question=question, pipeline=pipeline,
        platform=market_platform,
        market_url=market_url,
        outcome=outcome, market_probability=last.get("market_probability"),
        model_probability=last.get("model_probability"), edge=edge, stance=last.get("stance"),
        recommendation=recommendation, recommendation_detail=detail, confidence=last.get("confidence"),
        question_type=last.get("question_type", "binary"),
        # When we had to backstop the forecast, the model's prose can cite a
        # different number than the structured forecast that drives the
        # recommendation — so use the forecast's own rationale to stay consistent.
        thesis=(last.get("thesis") if use_backstop_thesis else raw_answer),
        evidence_sources=last.get("evidence_sources", []),
        evidence_error=last.get("evidence_error"),
        grounding=grounding_note, tool_transcript=tool_transcript,
        tool_loop_steps=res.get("steps"), tool_loop_truncated=res.get("truncated"),
        effort_tier=req.effort_tier,
        live_trade_intent=live_trade_intent,
    )
    # Evolution loop: enrol the analysed market into the live track record (pointer only).
    if report.market_url and report.model_probability is not None:
        from analyzing_llm_rationale import track_record_live as _trl
        _ident = _trl.ident_from_url(report.platform or "", report.market_url)
        await _enroll_market(report.platform, _ident, report.market_url, question, "agent_analyze")
    return report


@app.get("/agent/scan", tags=["Agents"], summary="Scan a venue for mispriced markets", response_model=AgentScanResponse)
async def agent_scan(
    request: Request = None,
    platform: str = "polymarket",
    limit: int = 4,
    min_edge: float = 0.1,
    evidence_top_k: int = 3,
    query: Optional[str] = None,
) -> AgentScanResponse:
    """Scan live markets on one or both venues and surface the most mispriced.

    Lists active markets (Polymarket and/or Kalshi), forecasts each, compares
    against the market price, and returns those whose model-vs-market gap is at
    least `min_edge`. `platform` accepts `polymarket`, `kalshi`, or `all`/`both`.
    `query` optionally filters to markets whose question contains that keyword
    (e.g. `nba`). Bounded by `limit` per venue (max 8) since each market runs a
    full forecast; results are cached briefly.
    """
    claims = None
    if request is not None:
        _check_rate_limit(request)
        _check_predict_rate_limit(request)
        claims = _require_auth(request)
    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    venue = (platform or "polymarket").strip().lower()
    kw = (query or "").strip() or None
    limit = max(1, min(limit, 8))
    evidence_top_k = max(1, evidence_top_k)

    if venue in ("all", "both") or ("poly" in venue and "kalshi" in venue):
        venues = ["polymarket", "kalshi"]
    elif "kalshi" in venue:
        venues = ["kalshi"]
    elif "poly" in venue:
        venues = ["polymarket"]
    else:
        raise HTTPException(status_code=422, detail="platform must be 'polymarket', 'kalshi', or 'all'.")

    cache_key = _cache_key("scan", ",".join(venues), kw or "", limit, round(min_edge, 3),
                           evidence_top_k, _state.get("model_key"))
    cached = _cache_get(cache_key)
    if cached is not None:
        return AgentScanResponse(**cached)

    from analyzing_llm_rationale import market_data
    loop = asyncio.get_running_loop()
    _listers = {"polymarket": market_data.list_polymarket, "kalshi": market_data.list_kalshi}
    quotes: List[Dict[str, Any]] = []
    list_errors: List[str] = []
    for v in venues:
        try:
            vq = await loop.run_in_executor(None, lambda fn=_listers[v]: fn(limit=limit, query=kw))
            quotes.extend(vq[:limit])
        except market_data.MarketDataError as exc:
            list_errors.append(f"{v}: {exc}")
    if not quotes:
        if list_errors and len(list_errors) == len(venues):
            raise HTTPException(status_code=502, detail=f"Could not list markets: {'; '.join(list_errors)}")
        return AgentScanResponse(platform=" + ".join(p.title() for p in venues), scanned=0, opportunities=[])
    platform_name = " + ".join(p.title() for p in venues)

    async def _score(quote: Dict[str, Any]) -> Optional[ScanOpportunity]:
        try:
            res = await predict(PredictRequest(
                question=quote["question"],
                description=quote.get("description") or "",
                resolution_criteria=quote.get("resolution_criteria") or "",
                categories=[quote["category"]] if quote.get("category") else [],
                news_articles=_news_articles(quote.get("venue_news_articles") or []),
                attach_evidence=True,
                evidence_top_k=evidence_top_k,
                market_platform=quote.get("platform"),
                market_ident=quote.get("ident"),
                market_url=quote.get("market_url"),
                market_outcome=quote.get("outcome"),
                market_probability=quote.get("probability"),
                created_time=quote.get("created_time"),
                publish_time=quote.get("created_time"),
                chat_mode=False,
            ), kb_user_id=(claims.get("sub") if claims else None))
        except Exception:
            return None
        analysis = res.market_analysis
        if analysis is None or analysis.edge is None:
            return None
        recommendation, _ = _agent_recommendation(analysis.edge, quote.get("outcome") or "Yes")
        return ScanOpportunity(
            question=quote["question"],
            market_url=quote.get("market_url"),
            outcome=quote.get("outcome") or "Yes",
            market_probability=analysis.market_probability,
            model_probability=analysis.model_probability,
            edge=analysis.edge,
            stance=analysis.stance,
            recommendation=recommendation,
        )

    scored = await asyncio.gather(*(_score(q) for q in quotes))
    opportunities = [s for s in scored if s is not None and s.edge is not None and abs(s.edge) >= min_edge]
    opportunities.sort(key=lambda o: abs(o.edge), reverse=True)
    response = AgentScanResponse(platform=platform_name, scanned=len(quotes), opportunities=opportunities)
    _cache_set(cache_key, response.model_dump(), _MARKET_CACHE_TTL * 10)
    return response


# The live track record is advanced by a GitHub Action (see
# .github/workflows/track-record-tick.yml + scripts/track_record_tick.py), which
# commits the public aggregate to static/track_record_live.json. The server only
# *serves* that result (see _read_live_track_record + GET /track-record) — no
# batch work runs on Cloud Run, which is what kept OOM/timeout-failing.
