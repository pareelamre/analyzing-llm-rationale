"""Measure Foresea server-side request preparation latency without HTTP/model calls.

This benchmark isolates the app-owned part of forecast latency: context assembly,
market enrichment dispatch, evidence retrieval dispatch, and prompt construction.
It intentionally replaces network-dependent fetches with controlled delays so
the result is repeatable on a developer machine.

Examples:
  py scripts/measure_prepare_latency.py
  py scripts/measure_prepare_latency.py --iterations 100 --simulated-delay-ms 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import server  # noqa: E402


def _install_test_state() -> None:
    server._state.clear()
    server._local_cache.clear()
    server._state.update({
        "provider": SimpleNamespace(model_name="benchmark-provider"),
        "evidence_pipeline": SimpleNamespace(),
        "variants": {
            "variant0_neutral_baseline": SimpleNamespace(
                output_fields=("predicted_answer", "confidence", "rationale")
            )
        },
        "system_prompt": "System",
        "prompt_templates": {
            "variant0_neutral_baseline": "[question]\nReturn JSON.",
        },
        "temperature": 0.0,
        "max_tokens": 384,
        "model_key": "benchmark-model",
    })


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def _stats_ms(values: list[float]) -> dict[str, float]:
    return {
        "avg_ms": round(statistics.mean(values), 3),
        "p50_ms": round(_percentile(values, 0.50), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


async def _run_case(
    name: str,
    req: server.PredictRequest,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    timings: list[float] = []
    for index in range(warmup + iterations):
        started = time.perf_counter()
        messages, articles, evidence_error = await server._prepare_predict_messages(req, None)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if index >= warmup:
            timings.append(elapsed_ms)
    return {
        "case": name,
        "iterations": iterations,
        "message_count": len(messages),
        "article_count": len(articles),
        "evidence_error": evidence_error,
        **_stats_ms(timings),
    }


async def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _install_test_state()
    delay_s = max(0.0, args.simulated_delay_ms / 1000)
    calls = {"market_context": 0, "evidence": 0}

    async def fake_fetch_market_context(platform, ident, url):  # noqa: ANN001
        calls["market_context"] += 1
        await asyncio.sleep(delay_s)
        return {
            "platform": platform or "kalshi",
            "market_url": url,
            "probability": 0.42,
            "question": "Will the benchmark market resolve yes?",
            "description": "Benchmark market description.",
            "resolution_criteria": "Resolves yes if the benchmark condition is met.",
            "venue_news_articles": [],
        }

    async def fake_fetch_evidence_with_cache(question, top_k, source="forecast"):  # noqa: ANN001
        calls["evidence"] += 1
        await asyncio.sleep(delay_s)
        return [
            {
                "title": "Benchmark evidence",
                "source": "Benchmark News",
                "url": "https://example.com/benchmark",
                "summary": f"Controlled evidence for {question}",
                "relevance_score": 1.0,
            }
        ], None, "success"

    original_market_context = server._fetch_market_context
    original_evidence = server._fetch_evidence_with_cache
    server._fetch_market_context = fake_fetch_market_context
    server._fetch_evidence_with_cache = fake_fetch_evidence_with_cache
    try:
        cases = [
            (
                "simple_chat_fast_path",
                server.PredictRequest(
                    question="Can you explain what changed?",
                    chat_mode=True,
                    attach_evidence=False,
                    max_tokens=384,
                ),
            ),
            (
                "supplied_market_context_no_fetch",
                server.PredictRequest(
                    question="Will the benchmark market resolve yes?",
                    chat_mode=True,
                    attach_evidence=False,
                    market_platform="kalshi",
                    market_ident="BENCHMARK",
                    market_probability=0.42,
                    resolution_criteria="Resolves yes if the benchmark condition is met.",
                    max_tokens=384,
                ),
            ),
            (
                "market_enrichment_fetch",
                server.PredictRequest(
                    question="Will the benchmark market resolve yes?",
                    chat_mode=True,
                    attach_evidence=False,
                    market_url="https://kalshi.com/markets/BENCHMARK",
                    max_tokens=384,
                ),
            ),
            (
                "evidence_fetch",
                server.PredictRequest(
                    question="Will the benchmark market resolve yes?",
                    chat_mode=True,
                    attach_evidence=True,
                    evidence_top_k=3,
                    max_tokens=384,
                ),
            ),
        ]
        results = [
            await _run_case(name, req, args.iterations, args.warmup)
            for name, req in cases
        ]
    finally:
        server._fetch_market_context = original_market_context
        server._fetch_evidence_with_cache = original_evidence
        server._state.clear()
        server._local_cache.clear()

    return {
        "iterations": args.iterations,
        "warmup": args.warmup,
        "simulated_delay_ms": args.simulated_delay_ms,
        "calls": calls,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Foresea request preparation latency.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--simulated-delay-ms", type=float, default=25.0)
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    print(json.dumps(asyncio.run(_benchmark(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
