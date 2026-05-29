from __future__ import annotations

import asyncio
import hashlib
import html
import os
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from analyzing_llm_rationale.pipeline import build_user_prompt, parse_model_response

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _REPO_ROOT / "static"
_ANALYTICS_DB = Path(os.environ.get("ANALYTICS_DB", "/tmp/foresea_analytics.duckdb"))

_REQUIRED_API_KEY: Optional[str] = os.environ.get("API_KEY")
_state: Dict[str, Any] = {}

_DESCRIPTION = """
## Overview

The **LLM Forecasting API** runs probabilistic binary (yes/no) predictions on
[Metaculus](https://www.metaculus.com)-style forecasting questions using
**GPT-OSS-120B** via the SCADS AI inference cluster.

Each prediction includes a **confidence score** (0–1), a structured **rationale**,
and optional evidence sources fetched from live news when the evidence pipeline is
configured.

---

## Quick start

```bash
curl -X POST https://analyzing-llm-rationale-hy7gvnvt4a-uc.a.run.app/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "Will the Federal Reserve cut interest rates at least once before the end of 2025?",
    "variant": "variant0_neutral_baseline"
  }'
```

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
        "description": "Run probabilistic predictions on binary forecasting questions.",
    },
    {
        "name": "System",
        "description": "Health and liveness checks.",
    },
]


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
        "https://analyzing-llm-rationale-hy7gvnvt4a-uc.a.run.app",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


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
            },
            {
                "question": "Will the Federal Reserve cut interest rates at least once before the end of 2025?",
                "description": "The question resolves YES if the FOMC reduces the federal funds rate target by at least 25 basis points from its current level at any point before 31 December 2025.",
                "resolution_criteria": "Resolves YES if the Federal Reserve lowers the federal funds rate at least once before 2026-01-01.",
                "categories": ["Economics", "Finance", "United States"],
                "variant": "variant3_reasoning_type",
                "resolve_time": "2025-12-31T23:59:00Z",
                "days_open": 180,
            },
        ]
    })

    question: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The binary forecasting question to evaluate. Must be answerable with Yes or No.",
        examples=["Will the Federal Reserve cut interest rates at least once before the end of 2025?"],
    )
    description: str = Field(
        "",
        max_length=4000,
        description="Extended background context that clarifies what the question is asking.",
    )
    resolution_criteria: str = Field(
        "",
        max_length=2000,
        description="Exact conditions under which the question resolves Yes or No.",
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


class PredictResponse(BaseModel):
    """Prediction result for a single forecasting question."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "predicted_answer": "No",
            "confidence": 0.72,
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

    predicted_answer: Optional[str] = Field(None, description="`Yes` or `No`.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence (0 = certain No, 1 = certain Yes).")
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
    """Submit a binary forecasting question and receive a structured prediction.

    The model returns:
    - **`predicted_answer`** — `Yes` or `No`
    - **`confidence`** — probability (0–1) the answer is Yes
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

    ### Example — minimal request

    ```bash
    curl -X POST /predict \\
      -H "Content-Type: application/json" \\
      -d '{"question": "Will oil prices exceed $100 per barrel in 2025?"}'
    ```

    ### Example — with context and variant

    ```bash
    curl -X POST /predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "Will the EU impose new sanctions on Russia before July 2025?",
        "description": "Focuses on economic sanctions, not diplomatic measures.",
        "resolution_criteria": "Resolves YES if any new EU economic sanction package is announced.",
        "categories": ["Geopolitics", "Europe"],
        "variant": "variant5_key_conditions",
        "resolve_time": "2025-07-01T00:00:00Z"
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

    variant = variants[req.variant]
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

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

    parsed = parse_model_response(content, variant.output_fields)
    rationale = parsed.get("rationale")
    return PredictResponse(
        predicted_answer=parsed.get("predicted_answer"),
        confidence=parsed.get("confidence"),
        rationale=rationale,
        model_rationale=rationale,
        variant=req.variant,
        model_key=_state["model_key"],
        evidence_sources=_evidence_sources(evidence_articles),
        evidence_articles=[NewsArticle(**a) for a in evidence_articles],
        evidence_error=evidence_error,
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
