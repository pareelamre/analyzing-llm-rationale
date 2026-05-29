from __future__ import annotations

import asyncio
import html
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _REPO_ROOT / "static"

from analyzing_llm_rationale.pipeline import build_user_prompt, parse_model_response

# Module-level state populated by serve_command() before uvicorn.run()
_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="LLM Forecasting API",
    description="Single-record prediction endpoint for the analyzing-llm-rationale pipeline.",
    lifespan=lifespan,
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


class NewsArticle(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    summary_llm: Optional[str] = None
    text: Optional[str] = None
    source: Optional[str] = None
    credibility: Optional[Dict[str, Any]] = None
    frs: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    authors: Optional[Any] = None
    publish_date: Optional[str] = None
    keywords: Optional[Any] = None
    relevance_score: Optional[float] = None
    search_query: Optional[str] = None


class EvidenceSource(BaseModel):
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    publish_date: Optional[str] = None
    relevance_score: Optional[float] = None


class PredictRequest(BaseModel):
    question: str
    description: str = ""
    resolution_criteria: str = ""
    categories: List[str] = Field(default_factory=list)
    news_articles: List[NewsArticle] = Field(default_factory=list)
    variant: str = "variant0_neutral_baseline"
    attach_evidence: bool = True
    evidence_top_k: int = 5
    created_time: Optional[str] = None
    publish_time: Optional[str] = None
    resolve_time: Optional[str] = None
    days_open: Optional[int] = None


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
    instances: List[Dict[str, Any]]

class VertexPredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def _clean_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _clean_article(article: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(article)
    for field in ("title", "summary", "summary_llm", "text", "source"):
        cleaned[field] = _clean_text(cleaned.get(field))
    return cleaned


def _evidence_sources(articles: List[Dict[str, Any]]) -> List[EvidenceSource]:
    sources: List[EvidenceSource] = []
    seen: set[tuple[str, str]] = set()
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


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
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

    evidence_articles = [_clean_article(article) for article in evidence_articles]
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
        evidence_articles=[NewsArticle(**article) for article in evidence_articles],
        evidence_error=evidence_error,
    )


@app.post("/vertex-predict", response_model=VertexPredictResponse)
async def vertex_predict(req: VertexPredictRequest) -> VertexPredictResponse:
    predictions = []
    for instance in req.instances:
        result = await predict(PredictRequest(**instance))
        predictions.append(result.model_dump())
    return VertexPredictResponse(predictions=predictions)
