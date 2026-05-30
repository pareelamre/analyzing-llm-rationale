from __future__ import annotations

import asyncio
import hashlib
import html
import os
import re
import smtplib
import threading
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from analyzing_llm_rationale.pipeline import (
    _parse_json_dict,
    build_user_prompt,
    parse_model_response,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _REPO_ROOT / "static"
_ANALYTICS_DB = Path(os.environ.get("ANALYTICS_DB", "/tmp/foresea_analytics.duckdb"))

_REQUIRED_API_KEY: Optional[str] = os.environ.get("API_KEY")
_GOOGLE_CLIENT_ID: Optional[str] = os.environ.get("GOOGLE_CLIENT_ID")
_SESSION_SECRET: str = os.environ.get("SESSION_SECRET", "change-me-in-production")
_SESSION_TTL_DAYS = 30
_state: Dict[str, Any] = {}

_DESCRIPTION = """
## Overview

The **LLM Forecasting API** runs probabilistic forecasts for
[Metaculus](https://www.metaculus.com)-style forecasting questions using
**GPT-OSS-120B** via the SCADS AI inference cluster.

It supports binary, multiple-choice, numeric, and date forecasts. Each response
includes a structured **rationale**, typed forecast fields, and optional evidence
sources fetched from live news when the evidence pipeline is configured.

---

## Quick start

```bash
curl -X POST https://foresea.ink/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What will US CPI inflation be in December 2026?",
    "question_type": "numeric",
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

**20 requests per minute per IP address.** Exceeding this returns `429 Too Many Requests`
with a `Retry-After: 60` header.

---

## Authentication

When the server is configured with `API_KEY`, all prediction endpoints require:

```
X-API-Key: <your-key>
```

The `/health` endpoint is always unauthenticated.

---

## Source code

[github.com/pareelamle/analyzing-llm-rationale](https://github.com/pareelamre/analyzing-llm-rationale)
"""

_TAGS = [
    {
        "name": "Inference",
        "description": "Run probabilistic forecasts on binary, multiple-choice, numeric, and date forecasting questions.",
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
    """Verify a session JWT and return its claims."""
    import jwt as _jwt
    try:
        return _jwt.decode(token, _SESSION_SECRET, algorithms=["HS256"])
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired.") from None
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token.") from None


_ds_client: Any = None


def _get_datastore():
    global _ds_client
    if _ds_client is None:
        try:
            from google.cloud import datastore as _ds
            _ds_client = _ds.Client()
        except Exception:
            pass
    return _ds_client


def _upsert_user(sub: str, email: str, name: str, picture: str) -> None:
    """Create or update a User entity in Cloud Datastore."""
    client = _get_datastore()
    if client is None:
        return
    from google.cloud import datastore as _ds
    key = client.key("User", sub)
    entity = client.get(key)
    if entity is None:
        entity = _ds.Entity(key=key, exclude_from_indexes=("picture",))
        entity["created_at"] = datetime.now(timezone.utc)
    entity.update(
        email=email,
        name=name,
        picture=picture,
        last_login=datetime.now(timezone.utc),
    )
    client.put(entity)


# ── Rate limiter ──────────────────────────────────────────────────────────────
class _RateLimiter:
    def __init__(self, calls: int = 20, period: int = 60):
        self._calls = calls
        self._period = period
        self._log: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        window = now - self._period
        log = self._log[key]
        while log and log[0] < window:
            log.pop(0)
        if len(log) >= self._calls:
            return False
        log.append(now)
        return True


_rate_limiter = _RateLimiter(calls=20, period=60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="LLM Forecasting API",
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://foresea.ink",
        "https://www.foresea.ink",
        "https://analyzing-llm-rationale-hy7gvnvt4a-uc.a.run.app",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# Redirect middleware: send requests from run.app hosts to the custom domain
@app.middleware("http")
async def host_redirect_middleware(request: Request, call_next):
    host = (request.headers.get("host") or "").lower()
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


# Middleware to catch unhandled exceptions and send an alert email
@app.middleware("http")
async def exception_alert_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        # Build alert
        tb = traceback.format_exc()
        subject = f"Foresea server error: {type(exc).__name__}"
        body = (
            f"Request: {request.method} {request.url}\n"
            f"Host: {request.headers.get('host')}\n\n"
            f"Exception:\n{tb}"
        )
        # Send email in background thread to avoid blocking
        try:
            threading.Thread(target=_send_alert_email, args=(subject, body), daemon=True).start()
        except Exception:
            pass
        raise


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

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/track-record", tags=["System"], summary="Public forecasting track record")
async def track_record():
    """Return Foresea's resolved-forecast track record.

    A backtest of `gpt-oss-120b` predictions scored against published Metaculus
    outcomes: accuracy, Brier score, calibration (ECE), a reliability curve, and
    a sample of individual resolved forecasts. Only questions with a known
    real-world outcome are included.
    """
    path = _STATIC_DIR / "track_record.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Track record not generated yet.")
    return FileResponse(str(path), media_type="application/json")


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
        min_length=10,
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
        max_length=2000,
        description="Exact conditions or measurement source used to resolve the forecast.",
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
            "Pre-fetched news articles to use as evidence. "
            "If empty and the evidence pipeline is configured, articles are fetched automatically."
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
        description="If `true` and `news_articles` is empty, fetch live evidence automatically.",
    )
    evidence_top_k: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum number of evidence articles to retrieve (1–10).",
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
        }
    })

    question_type: str = Field("binary", description="Detected type: `binary`, `multiple_choice`, `numeric`, or `date`.")
    predicted_answer: Optional[str] = Field(None, description="Headline answer: Yes/No, the top option, or the median estimate.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in the headline answer (binary/MC). Null for numeric.")
    options: List[OptionProb] = Field(default_factory=list, description="Per-option probabilities for `multiple_choice`.")
    range_forecast: Optional[RangeForecast] = Field(None, description="p10/p50/p90 estimate for `numeric` and `date`.")
    rationale: Optional[str] = Field(None, description="2–4 sentence explanation of the prediction.")
    model_rationale: Optional[str] = Field(None, description="Raw rationale as returned by the model (may differ from `rationale` after post-processing).")
    variant: str = Field(..., description="Prompt variant used for this prediction.")
    model_key: str = Field(..., description="Model identifier (e.g. `gpt-oss-120b`).")
    evidence_sources: List[EvidenceSource] = Field(default_factory=list, description="Deduplicated citations used as evidence.")
    evidence_articles: List[NewsArticle] = Field(default_factory=list, description="Full evidence articles passed to the model.")
    evidence_error: Optional[str] = Field(None, description="Non-null if evidence retrieval failed (prediction still returned).")


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


class VisitRequest(BaseModel):
    """Anonymous browser visit event."""

    path: str = Field("/", max_length=500)
    referrer: str = Field("", max_length=2000)
    timezone: Optional[str] = Field(None, max_length=100)


class AnalyticsSummary(BaseModel):
    total_visits: int
    unique_visitors: int
    by_day: List[Dict[str, Any]]


class GoogleAuthRequest(BaseModel):
    """Google One-Tap credential submitted by the browser."""
    credential: str = Field(..., max_length=8192)


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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _check_api_key(request: Request) -> None:
    if not _REQUIRED_API_KEY:
        return
    if request.headers.get("X-API-Key", "") != _REQUIRED_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def _check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded — 20 requests per minute per IP.",
            headers={"Retry-After": "60"},
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


# ── Multi-type forecasting ────────────────────────────────────────────────────
_TYPE_SCHEMAS = {
    "binary": (
        '{"type":"binary","predicted_answer":"Yes"|"No",'
        '"confidence":0-1,"rationale":"..."}'
    ),
    "multiple_choice": (
        '{"type":"multiple_choice","options":[{"label":"...","probability":0-1}],'
        '"rationale":"..."}'
    ),
    "numeric": (
        '{"type":"numeric","p10":<low>,"p50":<median>,"p90":<high>,'
        '"unit":"...","rationale":"..."}'
    ),
    "date": (
        '{"type":"date","p10":"YYYY-MM-DD","p50":"YYYY-MM-DD",'
        '"p90":"YYYY-MM-DD","rationale":"..."}'
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
    return instr


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _build_typed_response(
    req: "PredictRequest",
    parsed: Optional[Dict[str, Any]],
    content: str,
    evidence_articles: List[Dict[str, Any]],
    evidence_error: Optional[str],
) -> "PredictResponse":
    qtype = (req.question_type or (parsed.get("type") if parsed else None) or "binary").lower()
    rationale = parsed.get("rationale") if parsed else None
    model_key = req.openrouter_model or _state["model_key"]
    base = dict(
        variant=req.variant,
        model_key=model_key,
        evidence_sources=_evidence_sources(evidence_articles),
        evidence_articles=[NewsArticle(**a) for a in evidence_articles],
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
        return PredictResponse(
            question_type="multiple_choice",
            options=opts,
            predicted_answer=top.label if top else None,
            confidence=top.probability if top else None,
            rationale=rationale, model_rationale=rationale, **base,
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
            rationale=rationale, model_rationale=rationale, **base,
        )

    # binary (default) — reuse the battle-tested parser
    bparsed = parse_model_response(content, ("predicted_answer", "confidence", "rationale"))
    # If no structured forecast fields were found and the response doesn't look like JSON,
    # treat it as a plain conversational reply.
    if not bparsed.get("predicted_answer") and not content.strip().startswith("{"):
        text = content.strip()
        return PredictResponse(
            question_type="chat",
            rationale=text, model_rationale=text, **base,
        )
    brat = bparsed.get("rationale")
    return PredictResponse(
        question_type="binary",
        predicted_answer=bparsed.get("predicted_answer"),
        confidence=bparsed.get("confidence"),
        rationale=brat, model_rationale=brat, **base,
    )


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
            visitor_hash TEXT
        )
        """
    )
    return conn


def _visitor_hash(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else ""
    if not ip and request.client:
        ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    day = time.strftime("%Y-%m-%d", time.gmtime())
    salt = os.environ.get("ANALYTICS_SALT", "foresea-analytics")
    raw = f"{day}:{ip}:{user_agent}:{salt}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _record_visit(event: VisitRequest, request: Request) -> None:
    conn = _analytics_conn()
    try:
        conn.execute(
            """
            INSERT INTO page_visits (path, referrer, user_agent, timezone, visitor_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                event.path,
                event.referrer,
                request.headers.get("user-agent", "")[:1000],
                event.timezone,
                _visitor_hash(request),
            ],
        )
    finally:
        conn.close()


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


@app.post("/analytics/visit", tags=["System"], summary="Record anonymous page visit")
async def record_visit(event: VisitRequest, request: Request) -> Dict[str, str]:
    """Record one anonymous page visit.

    Stores no raw IP address. Unique visitors are estimated with a daily salted
    hash of IP address and user agent.
    """
    _record_visit(event, request)
    return {"status": "ok"}


@app.get("/analytics/summary", tags=["System"], response_model=AnalyticsSummary)
async def analytics_summary(request: Request) -> AnalyticsSummary:
    """Return basic page-visit counts.

    If `API_KEY` is configured, this endpoint requires `X-API-Key`.
    """
    _check_api_key(request)
    conn = _analytics_conn()
    try:
        total, unique_visitors = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT visitor_hash) FROM page_visits"
        ).fetchone()
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
    )


@app.get("/auth/config", tags=["Auth"], include_in_schema=False)
async def auth_config() -> Dict[str, str]:
    """Return the public Google OAuth client ID so the browser can initialise GIS."""
    return {"google_client_id": _GOOGLE_CLIENT_ID or ""}


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
    await loop.run_in_executor(None, _upsert_user, sub, email, name, picture)
    token = _issue_session(sub, email, name, picture)
    return SessionResponse(token=token, user_id=sub, email=email, name=name, picture=picture)


@app.get("/auth/me", tags=["Auth"], summary="Get current user", response_model=AuthMeResponse)
async def auth_me(request: Request) -> AuthMeResponse:
    """Return the authenticated user decoded from the `Authorization: Bearer` session token."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer header.")
    claims = _decode_session(auth[7:])
    return AuthMeResponse(
        user_id=claims["sub"],
        email=claims["email"],
        name=claims["name"],
        picture=claims["picture"],
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
async def predict(req: PredictRequest, request: Request = None) -> PredictResponse:
    """Submit a forecasting question and receive a typed structured prediction.

    The model returns:
    - **`question_type`** — `binary`, `multiple_choice`, `numeric`, or `date`
    - **`predicted_answer`** — `Yes`/`No`, top option, or median estimate
    - **`confidence`** — probability (0–1) for binary and multiple-choice answers
    - **`options`** — per-option probabilities for multiple-choice questions
    - **`range_forecast`** — p10/p50/p90 bounds for numeric and date questions
    - **`rationale`** — 2–4 sentence explanation
    - **`evidence_sources`** — news articles used as context

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
        "question_type": "binary"
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
    if request is not None:
        _check_rate_limit(request)
        _check_api_key(request)

    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    variants = _state["variants"]
    if req.variant not in variants:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant '{req.variant}'. Valid: {sorted(variants)}",
        )

    prompt_text = _state["prompt_templates"][req.variant]
    system_prompt = _state["system_prompt"]

    record = req.model_dump()
    evidence_articles = [article.model_dump() for article in req.news_articles]
    evidence_error = None

    if req.attach_evidence and not evidence_articles:
        evidence_pipeline = _state.get("evidence_pipeline")
        if evidence_pipeline is None:
            evidence_error = "Evidence pipeline is not configured on this server."
        else:
            top_k = max(1, min(req.evidence_top_k, 10))
            loop = asyncio.get_running_loop()
            try:
                evidence_articles = await loop.run_in_executor(
                    None,
                    lambda: evidence_pipeline.fetch_summarize_rank(req.question, top_k=top_k),
                )
            except Exception as exc:
                evidence_error = f"Evidence retrieval failed: {exc}"

    evidence_articles = [_clean_article(a) for a in evidence_articles]
    record["news_articles"] = evidence_articles

    user_prompt = build_user_prompt(record, prompt_text, "full")
    user_prompt += _typing_instruction(req.question_type, req.options, has_history=bool(req.history))
    messages = [{"role": "system", "content": system_prompt}]
    for turn in req.history[-12:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": user_prompt})

    # Use user-supplied OpenRouter key/model if provided, otherwise fall back to server default.
    if req.openrouter_api_key and req.openrouter_model:
        from analyzing_llm_rationale.providers import OpenRouterProvider
        provider = OpenRouterProvider(
            model_name=req.openrouter_model,
            api_key=req.openrouter_api_key,
        )
        temperature = 0.7
        max_tokens = _state.get("max_tokens", 1024)
    else:
        provider = _state["provider"]
        temperature = _state["temperature"]
        max_tokens = _state["max_tokens"]

    loop = asyncio.get_running_loop()
    try:
        content = await loop.run_in_executor(
            None,
            lambda: provider.chat_completion(messages, temperature, max_tokens),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Provider error: {exc}") from exc

    parsed = _parse_json_dict(content)
    return _build_typed_response(req, parsed, content, evidence_articles, evidence_error)


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
        _check_api_key(request)
    predictions = []
    for instance in req.instances:
        result = await predict(PredictRequest(**instance))
        predictions.append(result.model_dump())
    return VertexPredictResponse(predictions=predictions)
