"""
Prefect flow: fetch news → summarize/rank → run inference → store in DuckDB.

Usage (on-demand):
    python flows/forecasting_flow.py --question-id 124

Usage (local Prefect server + schedule):
    prefect server start          # in a separate terminal
    python flows/forecasting_flow.py --deploy

Environment variables required:
    SCADS_AI_API_KEY   — SCADS AI API key (same as the rest of the project)

Optional:
    NEWSAPI_KEY        — NewsAPI free-tier key for richer article retrieval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefect import flow, task, get_run_logger
from prefect.schedules import CronSchedule

from analyzing_llm_rationale.db import (
    get_connection,
    ingest_dataset,
    store_news_articles,
)
from analyzing_llm_rationale.news_pipeline import NewsPipeline

DATASET_PATH = next(
    Path(__file__).resolve().parents[1].glob(
        "forecasting_qa_news_metaculus_*.json"
    ),
    None,
)
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_question(question_id: int) -> dict:
    if DATASET_PATH is None:
        raise FileNotFoundError("Dataset file not found.")
    records = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    for r in records:
        if int(r.get("id", -1)) == question_id:
            return r
    raise ValueError(f"Question ID {question_id} not found in dataset.")


@task(name="fetch-and-rank-news", retries=1, retry_delay_seconds=10)
def fetch_and_rank_news(question: str, top_k: int = 5) -> list[dict]:
    logger = get_run_logger()
    logger.info("Fetching news for: %s", question[:80])
    pipeline = NewsPipeline()
    articles = pipeline.fetch_summarize_rank(question, top_k=top_k)
    logger.info("Retrieved %d ranked articles.", len(articles))
    return articles


@task(name="run-inference")
def run_inference(record: dict, articles: list[dict]) -> dict:
    logger = get_run_logger()
    from analyzing_llm_rationale.cli import build_provider, repo_root
    from analyzing_llm_rationale.config import load_model_configs, load_variant_configs
    from analyzing_llm_rationale.pipeline import build_user_prompt, parse_model_response

    models = load_model_configs(CONFIGS_DIR / "models.yaml")
    model_cfg = models["gpt-oss-120b"]
    variants = load_variant_configs(CONFIGS_DIR / "variants.yaml")
    variant = variants["variant0_neutral_baseline"]

    class _Args:
        model = "gpt-oss-120b"
        provider = model_cfg.provider
        local_model_name = model_cfg.local_model_name
        router_model_name = model_cfg.router_model_name
        api_base_url = model_cfg.api_base_url
        api_key_env_var = model_cfg.api_key_env_var
        api_key_file = model_cfg.api_key_file
        device = "cpu"
        request_timeout_s = 120.0
        model_label = model_cfg.result_label

    provider = build_provider(_Args())

    enriched = dict(record)
    enriched["news_articles"] = articles

    prompt_template = (repo_root() / variant.prompt_path).read_text(encoding="utf-8")
    system_prompt = (repo_root() / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
    user_prompt = build_user_prompt(enriched, prompt_template, "summary")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    content = provider.chat_completion(messages, temperature=0.0, max_tokens=2048)
    parsed = parse_model_response(content, variant.output_fields)
    logger.info(
        "Prediction: %s (confidence=%.2f)",
        parsed.get("predicted_answer"),
        parsed.get("confidence") or 0.0,
    )
    return {
        "question_id": int(record.get("id", 0)),
        "predicted_answer": parsed.get("predicted_answer"),
        "confidence": parsed.get("confidence"),
        "rationale": parsed.get("rationale"),
        "model": "gpt-oss-120b",
        "variant": "variant0_neutral_baseline",
        "temperature": 0.0,
    }


@task(name="store-results")
def store_results(prediction: dict, articles: list[dict]) -> None:
    logger = get_run_logger()
    conn = get_connection()
    run_id = str(uuid.uuid4())

    conn.execute(
        """
        INSERT INTO predictions
          (run_id, model, temperature, variant, question_id,
           predicted_answer, confidence, rationale)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            prediction["model"],
            prediction["temperature"],
            prediction["variant"],
            prediction["question_id"],
            prediction["predicted_answer"],
            prediction["confidence"],
            prediction["rationale"],
        ],
    )
    store_news_articles(prediction["question_id"], articles, run_id=run_id, conn=conn)
    conn.close()
    logger.info("Stored prediction and %d articles in DuckDB.", len(articles))


@flow(name="forecasting-pipeline")
def forecasting_pipeline(question_id: int, top_k: int = 5) -> dict:
    record = _load_question(question_id)
    question = record["question"]

    articles = fetch_and_rank_news(question, top_k=top_k)
    prediction = run_inference(record, articles)
    store_results(prediction, articles)

    return prediction


def main():
    parser = argparse.ArgumentParser(description="Run forecasting pipeline for a question.")
    parser.add_argument("--question-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=5, help="Number of articles to fetch.")
    parser.add_argument(
        "--deploy", action="store_true",
        help="Deploy as a scheduled Prefect flow (daily at 06:00 UTC).",
    )
    args = parser.parse_args()

    if args.deploy:
        forecasting_pipeline.serve(
            name="daily-forecasting",
            schedules=[CronSchedule(cron="0 6 * * *", timezone="UTC")],
        )
    else:
        result = forecasting_pipeline(
            question_id=args.question_id, top_k=args.top_k
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
