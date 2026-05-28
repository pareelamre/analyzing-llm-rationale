from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


class NewsArticle(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    summary_llm: Optional[str] = None
    text: Optional[str] = None
    credibility: Optional[Dict[str, Any]] = None
    frs: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    authors: Optional[Any] = None
    publish_date: Optional[str] = None
    keywords: Optional[Any] = None


class PredictRequest(BaseModel):
    question: str
    description: str = ""
    resolution_criteria: str = ""
    categories: List[str] = []
    news_articles: List[NewsArticle] = []
    variant: str = "variant0_neutral_baseline"
    created_time: Optional[str] = None
    publish_time: Optional[str] = None
    resolve_time: Optional[str] = None
    days_open: Optional[int] = None


class PredictResponse(BaseModel):
    predicted_answer: Optional[str]
    confidence: Optional[float]
    rationale: Optional[str]
    variant: str
    model_key: str


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


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
    return PredictResponse(
        predicted_answer=parsed.get("predicted_answer"),
        confidence=parsed.get("confidence"),
        rationale=parsed.get("rationale"),
        variant=req.variant,
        model_key=_state["model_key"],
    )
