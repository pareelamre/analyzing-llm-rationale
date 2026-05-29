from __future__ import annotations

import asyncio
import html
import os
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from analyzing_llm_rationale.pipeline import build_user_prompt, parse_model_response

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _REPO_ROOT / "static"

# Optional API key protection — set API_KEY env var to require it on /predict
_REQUIRED_API_KEY: Optional[str] = os.environ.get("API_KEY")

# Module-level state populated by serve_command() before uvicorn.run()
_state: Dict[str, Any] = {}


# ── Rate limiter (sliding window, per IP, in-process) ─────────────────────────
class _RateLimiter:
    def __init__(self, calls: int = 20, period: int = 60):
        self._calls = calls
        self._period = period
        self._log: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        window = now - self._period
        log = self._log[key]
        # Evict old entries
        while log and log[0] < window:
            log.pop(0)
        if len(log) >= self._calls:
            return False
        log.append(now)
        return True


_rate_limiter = _RateLimiter(calls=20, period=60)  # 20 req/min per IP


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="LLM Forecasting API",
    description="Single-record prediction endpoint for the analyzing-llm-rationale pipeline.",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow same-origin browser requests and the deployed Cloud Run origin.
# Cross-origin API callers must send a valid API_KEY header.
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

# ── Static / UI ───────────────────────────────────────────────────────────────
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ── Models ────────────────────────────────────────────────────────────────────
class NewsArticle(BaseModel):
    title: Optional[str] = Field(None, max_length=500)
    summary: Optional[str] = Field(None, max_length=4000)
    summary_llm: Optional[str] = Field(None, max_length=4000)
    text: Optional[str] = Field(None, max_length=20000)
    source: Optional[str] = Field(None, max_length=200)
    credibility: Optional[Dict[str, Any]] = None
    frs: Optional[Dict[str, Any]] = None
    url: Optional[str] = Field(None, max_length=2000)
    authors: Optional[Any] = None
    publish_date: Optional[str] = Field(None, max_length=100)
    keywords: Optional[Any] = None
    relevance_score: Optional[float] = None
    search_query: Optional[str] = Field(None, max_length=500)


class EvidenceSource(BaseModel):
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    publish_date: Optional[str] = None
    relevance_score: Optional[float] = None


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=2000)
    description: str = Field("", max_length=4000)
    resolution_criteria: str = Field("", max_length=2000)
    categories: List[str] = Field(default_factory=list, max_length=20)
    news_articles: List[NewsArticle] = Field(default_factory=list, max_length=20)
    variant: str = Field("variant0_neutral_baseline", max_length=100)
    attach_evidence: bool = True
    evidence_top_k: int = Field(5, ge=1, le=10)
    created_time: Optional[str] = Field(None, max_length=50)
    publish_time: Optional[str] = Field(None, max_length=50)
    resolve_time: Optional[str] = Field(None, max_length=50)
    days_open: Optional[int] = Field(None, ge=0, le=36500)

    @field_validator("question")
    @classmethod
    def question_must_be_question(cls, v: str) -> str:
        # Block obvious prompt injection attempts
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
    predicted_answer: Optional[str]
    confidence: Optional[float]
    rationale: Optional[str]
    model_rationale: Optional[str]
    variant: str
    model_key: str
    evidence_sources: List[EvidenceSource] = Field(default_factory=list)
    evidence_articles: List[NewsArticle] = Field(default_factory=list)
    evidence_error: Optional[str] = None


class VertexPredictRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(..., max_length=10)


class VertexPredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


# ── Auth helper ───────────────────────────────────────────────────────────────
def _check_api_key(request: Request) -> None:
    if not _REQUIRED_API_KEY:
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != _REQUIRED_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key header.",
        )


# ── Rate limit helper ─────────────────────────────────────────────────────────
def _check_rate_limit(request: Request) -> None:
    ip = (request.client.host if request.client else "unknown")
    if not _rate_limiter.is_allowed(ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded — 20 requests per minute per IP.",
            headers={"Retry-After": "60"},
        )


# ── Utilities ─────────────────────────────────────────────────────────────────
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


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request = None) -> PredictResponse:
    if request is not None:
        _check_rate_limit(request)
        _check_api_key(request)

    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    variants = _state["variants"]
    if req.variant not in variants:
        valid = sorted(variants)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant '{req.variant}'. Valid: {valid}",
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


@app.post("/vertex-predict", response_model=VertexPredictResponse)
async def vertex_predict(req: VertexPredictRequest, request: Request = None) -> VertexPredictResponse:
    if request is not None:
        _check_rate_limit(request)
        _check_api_key(request)
    predictions = []
    for instance in req.instances:
        result = await predict(PredictRequest(**instance))
        predictions.append(result.model_dump())
    return VertexPredictResponse(predictions=predictions)
