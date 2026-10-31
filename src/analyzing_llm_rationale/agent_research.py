"""Bounded public research for the Edge Board's scheduled agent, without inference."""
from __future__ import annotations

import json
import logging
import math
import re
from contextvars import copy_context
from datetime import datetime, timezone
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from time import monotonic
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from . import market_data
from .benchmark_tools import _domain_blacklisted
from .news_pipeline import _QUERY_AGNOSTIC_CHANNELS, NewsPipeline

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
prefetches = meter.create_counter("agent.research.prefetches", unit="1")
NEWS_SOURCES = ("web", "gdelt", "google-news", "bing-news")
MAX_MARKETS = 2
MIN_RELEVANCE = 0.25
PREFETCH_TIMEOUT_S = 20.0
_research_slots = BoundedSemaphore(MAX_MARKETS)


@tracer.start_as_current_span("agent.research.news")
def search_market_evidence(query: str) -> dict[str, Any]:
    """Return source-provided snippets; never plan queries or summarize with an LLM."""
    try:
        pipeline = NewsPipeline(use_query_planner=False, summarize_articles=False,
                                use_embeddings=False, fetch_sources=NEWS_SOURCES,
                                min_relevance=MIN_RELEVANCE)
        # One query, no nested executor or fallback-query retry: daemon workers
        # below must be able to stop delaying inference at the shared deadline.
        articles = pipeline.rank(str(query)[:500], pipeline.fetch(str(query)[:500], top_k=5))
        articles = pipeline.select_diverse_sources(articles, top_k=3)
        sources = []
        for article in articles:
            if not isinstance(article, Mapping):
                continue
            url = str(article.get("url") or "")
            try:
                parts = urlsplit(url)
                relevance = float(article.get("relevance") or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if (parts.scheme not in {"http", "https"} or not parts.hostname
                    or _domain_blacklisted(url)
                    or not math.isfinite(relevance) or relevance < MIN_RELEVANCE
                    or article.get("source_channel") in _QUERY_AGNOSTIC_CHANNELS):
                continue
            # Do not truncate a citation into a different (or invalid) URL.
            if len(url) > 250:
                continue
            sources.append({
                "title": str(article.get("title") or "")[:100], "url": url,
                "published_at": str(article.get("publish_date") or article.get("published_at") or "")[:50],
                "snippet": str(article.get("summary") or article.get("text") or "")[:250],
            })
            if len(sources) == 2:
                break
        return {"evidence_status": "available" if sources else "empty", "sources": sources}
    except Exception as exc:  # aqg: top-level boundary -- a source outage must not invent evidence
        trace.get_current_span().record_exception(exc)
        trace.get_current_span().set_status(Status(StatusCode.ERROR))
        logger.warning("Agent research news unavailable (%s)", type(exc).__name__)
        return {"evidence_status": "unavailable", "sources": []}


def _book_levels(book: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for side in ("yes", "no", "yes_dollars", "no_dollars", "bids", "asks"):
        levels = []
        for row in book.get(side) or []:
            try:
                price = float(row["price"] if isinstance(row, dict) else row[0])
                size = float(row["size"] if isinstance(row, dict) else row[1])
                if not math.isfinite(price) or not math.isfinite(size) or price < 0 or size <= 0:
                    continue
                levels.append((price, {"price": str(row["price"])[:32], "size": str(row["size"])[:32]}
                               if isinstance(row, dict) else [price, size]))
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                continue
        if levels:
            result[side] = [row for _, row in sorted(levels, key=lambda pair: pair[0],
                                                    reverse=side != "asks")[:3]]
    return result


@tracer.start_as_current_span("agent.research.market")
def _market_packet(quote: Mapping[str, Any]) -> dict[str, Any]:
    venue, ident = str(quote["platform"]).lower(), str(quote["ident"])
    question = str(quote.get("question") or ident)[:500]
    packet: dict[str, Any] = {
        "platform": venue, "ident": ident, "question": question,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "resolution_criteria": str(quote.get("resolution_criteria") or "")[:700],
        "orderbook_status": "unavailable", "orderbook": {},
    }
    for name in ("probability", "yes_bid", "yes_ask", "close_time"):
        value = quote.get(name)
        if isinstance(value, (int, float)) and math.isfinite(value):
            packet[name] = value
        elif isinstance(value, str):
            packet[name] = value[:60]
    try:
        if venue == "kalshi":
            book = market_data.fetch_kalshi_orderbook(ident)
        else:
            token = str(quote.get("token_id") or "")
            if not re.fullmatch(r"[0-9]{1,200}", token):
                raise ValueError("YES token unavailable")
            packet["orderbook_outcome"] = "YES"
            book = market_data.fetch_polymarket_orderbook(token)
        packet["orderbook"] = _book_levels(book)
        packet["orderbook_status"] = "available" if packet["orderbook"] else "empty"
        if venue == "kalshi":
            packet["orderbook_convention"] = (
                "yes/no are BID ladders in cents; yes_dollars/no_dollars are BID ladders in dollars. "
                "Buy YES at 1 minus the best NO bid in dollars; buy NO at 1 minus the best YES bid. "
                "Quote yes_bid/yes_ask and probabilities are dollar fractions. Recheck before execution."
            )
            for side, ask_name in (("no", "buy_yes_ask_dollars"), ("yes", "buy_no_ask_dollars")):
                levels = packet["orderbook"].get(side + "_dollars")
                divisor = 1 if levels else 100
                levels = levels or packet["orderbook"].get(side)
                if levels:
                    row = levels[0]
                    bid = float(row["price"] if isinstance(row, dict) else row[0]) / divisor
                    if 0 <= bid <= 1:
                        packet[ask_name] = round(1 - bid, 6)
    except Exception as exc:  # aqg: top-level boundary -- partial public-data outage
        trace.get_current_span().record_exception(exc)
        trace.get_current_span().set_status(Status(StatusCode.ERROR))
        logger.warning("Agent research book unavailable venue=%s (%s)", venue, type(exc).__name__)
    packet.update(search_market_evidence(question))
    return {"action": "research_brief", "args": {"platform": venue, "ticker": ident},
            "observation": json.dumps(packet, ensure_ascii=False), "initiated_by": "system_prefetch",
            "research_available": packet["orderbook_status"] == "available" or packet["evidence_status"] == "available"}


def _unavailable_packet(quote: Mapping[str, Any]) -> dict[str, Any]:
    return {"action": "research_brief", "args": {"platform": quote["platform"], "ticker": quote["ident"]},
            "observation": json.dumps({"ident": quote["ident"], "orderbook_status": "unavailable",
                                       "evidence_status": "unavailable", "reason": "research_deadline_or_capacity"}),
            "initiated_by": "system_prefetch", "research_available": False}


def _prepare_packet(quote: Mapping[str, Any], index: int, completed: Queue) -> None:
    try:
        completed.put((index, _market_packet(quote)))
    except Exception as exc:  # aqg: top-level boundary -- isolate one malformed upstream packet
        logger.warning("Agent research packet unavailable (%s)", type(exc).__name__)
        completed.put((index, _unavailable_packet(quote)))
    finally:
        _research_slots.release()


@tracer.start_as_current_span("agent.research.prefetch")
def prefetch_market_research(quotes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prepare at most two distinct markets using fixed venue adapters, never arbitrary URLs."""
    selected, seen = [], set()
    for quote in quotes:
        if not isinstance(quote, Mapping):
            continue
        venue, ident = str(quote.get("platform") or "").lower(), str(quote.get("ident") or "")
        if (venue not in {"kalshi", "polymarket"}
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", ident)
                or (venue, ident) in seen):
            continue
        seen.add((venue, ident))
        selected.append(dict(quote))
        if len(selected) == MAX_MARKETS:
            break
    results = [_unavailable_packet(quote) for quote in selected]
    completed: Queue = Queue()
    deadline = monotonic() + PREFETCH_TIMEOUT_S
    pending = 0
    for index, quote in enumerate(selected):
        if _research_slots.acquire(blocking=False):
            # At most two active daemon workers, even across retries. No executor
            # shutdown/join can extend the deadline or process exit on an outage.
            Thread(target=copy_context().run, args=(_prepare_packet, quote, index, completed),
                   daemon=True).start()
            pending += 1
    while pending and (remaining := deadline - monotonic()) > 0:
        try:
            index, packet = completed.get(timeout=remaining)
        except Empty:
            break
        results[index] = packet
        pending -= 1
    trace.get_current_span().set_attribute("research.markets", len(results))
    prefetches.add(1, {"outcome": "prepared" if results else "empty",
                      "research_available": any(item["research_available"] for item in results)})
    logger.info("Agent initial research prepared markets=%d model_calls=0", len(results))
    return results
