#!/usr/bin/env python3
"""LLM relevance ranking + summarization for articles in kg_dataset.json.

Implements the paper's pipeline (Halawi et al. 2024, §4.1 steps 3-4):
  Step 3 — Rate article relevance wrt the question on a 1-6 scale.
           Discard articles with rating ≤ 3.
  Step 4 — Summarize each relevant article in 2-3 sentences focused on the question.
           Adds `relevance_score` and `summary_llm` fields to each article.

Uses the SCADS AI endpoint (same as the live server) via the existing provider
infrastructure.  Processes questions that have articles but lack `summary_llm`
on at least one of them.

Usage:
    python scripts/summarize_articles.py
    python scripts/summarize_articles.py --source kalshi --limit 100
    python scripts/summarize_articles.py --workers 8 --dry-run

Environment:
    SCADS_AI_API_KEY  — required (or --api-key)
    SCADS_AI_BASE_URL — default: https://llm.scads.ai/v1
    SUMMARIZE_MODEL   — default: gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_INPUT = Path("data/kg_dataset.json")
DEFAULT_BASE_URL = "https://llm.scads.ai/v1"
DEFAULT_MODEL = "gpt-oss-120b"
CHECKPOINT_EVERY = 50
MIN_RELEVANCE = 4       # paper: discard articles with rating <= 3
MAX_ARTICLE_CHARS = 1500  # feed at most this many chars of article text per rating/summary call

_RELEVANCE_PROMPT = """\
You are evaluating a news article's relevance to a forecasting question.

Question: {question}

Article title: {title}
Article text: {text}

Rate the relevance of this article to the question on a scale of 1–6:
1 = completely irrelevant
2 = mostly irrelevant
3 = slightly relevant
4 = moderately relevant
5 = highly relevant
6 = directly addresses the question

Reply with only a single integer (1–6) and nothing else."""

_SUMMARY_PROMPT = """\
You are summarizing a news article to help answer a forecasting question.

Question: {question}

Article title: {title}
Article text: {text}

Write a 2–3 sentence summary that captures only the information in this article \
that is relevant to answering the question. Be concise and factual.
Reply with the summary text only."""


def _chat(messages: List[Dict], model: str, api_key: str, base_url: str,
          max_tokens: int = 256, temperature: float = 0.0) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _rate_relevance(question: str, title: str, text: str,
                    model: str, api_key: str, base_url: str) -> int:
    snippet = text[:MAX_ARTICLE_CHARS] if text else title
    prompt = _RELEVANCE_PROMPT.format(question=question, title=title, text=snippet)
    try:
        reply = _chat([{"role": "user", "content": prompt}], model, api_key, base_url,
                      max_tokens=4, temperature=0.0)
        m = re.search(r"[1-6]", reply)
        return int(m.group()) if m else 3
    except Exception:
        return 3


def _summarize(question: str, title: str, text: str,
               model: str, api_key: str, base_url: str) -> str:
    snippet = text[:MAX_ARTICLE_CHARS] if text else title
    prompt = _SUMMARY_PROMPT.format(question=question, title=title, text=snippet)
    try:
        return _chat([{"role": "user", "content": prompt}], model, api_key, base_url,
                     max_tokens=200, temperature=0.2)
    except Exception:
        return ""


def process_question(record: Dict[str, Any], model: str, api_key: str,
                     base_url: str) -> Tuple[int, int]:
    """Rate + summarize articles in a record. Returns (rated, summarized) counts."""
    question = record.get("question", "")
    articles = record.get("news_articles") or []
    rated = summarized = 0

    for art in articles:
        if not isinstance(art, dict):
            continue
        if art.get("summary_llm"):
            continue  # already done

        title = art.get("title", "")
        text = art.get("text") or art.get("summary") or title

        score = _rate_relevance(question, title, text, model, api_key, base_url)
        art["relevance_score"] = score
        rated += 1

        if score >= MIN_RELEVANCE:
            summary = _summarize(question, title, text, model, api_key, base_url)
            if summary:
                art["summary_llm"] = summary
                summarized += 1
        else:
            art["summary_llm"] = None  # mark as processed but below threshold

    return rated, summarized


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None)
    ap.add_argument("--source", default=None,
                    help="kalshi|polymarket|manifold|metaculus")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--api-key", default=os.environ.get("SCADS_AI_API_KEY"))
    ap.add_argument("--base-url",
                    default=os.environ.get("SCADS_AI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--model",
                    default=os.environ.get("SUMMARIZE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.api_key and not args.dry_run:
        ap.error("SCADS_AI_API_KEY not set — pass --api-key or set the env var")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    records: List[Dict[str, Any]] = json.loads(input_path.read_text())
    print(f"Loaded {len(records)} records from {input_path}")

    needs: List[Tuple[int, Dict]] = []
    for i, r in enumerate(records):
        arts = r.get("news_articles") or []
        if not arts:
            continue
        if args.source and r.get("source") != args.source:
            continue
        if any(isinstance(a, dict) and not a.get("summary_llm") for a in arts):
            needs.append((i, r))

    if args.limit:
        needs = needs[:args.limit]

    total_arts = sum(
        sum(1 for a in r.get("news_articles", []) if isinstance(a, dict) and not a.get("summary_llm"))
        for _, r in needs
    )
    print(f"{len(needs)} questions need article summarization ({total_arts} articles)")

    if args.dry_run:
        est = total_arts * 2 * 1.5 / args.workers  # 2 calls per article, ~1.5s each
        print(f"Estimated runtime: ~{est/60:.0f} min with {args.workers} workers")
        return

    _lock = threading.Lock()
    done_q = [0]
    rated_total = [0]
    summ_total = [0]

    def process(seq: int, idx: int, record: Dict) -> Tuple[int, int, int]:
        r, s = process_question(record, args.model, args.api_key, args.base_url)
        return seq, r, s

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, seq, idx, r): seq
                   for seq, (idx, r) in enumerate(needs)}

        for future in concurrent.futures.as_completed(futures):
            _, r, s = future.result()
            with _lock:
                done_q[0] += 1
                rated_total[0] += r
                summ_total[0] += s
                seq = done_q[0]

            src = needs[futures[future]][1].get("source", "?")
            q_text = needs[futures[future]][1].get("question", "")[:55]
            print(f"  [{seq}/{len(needs)}] {src}: {q_text}… rated={r} summ={s}", flush=True)

            if seq % CHECKPOINT_EVERY == 0:
                with _lock:
                    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
                print(f"  ── checkpoint {seq} (rated={rated_total[0]} summ={summ_total[0]}) ──",
                      flush=True)

    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\nDone: {rated_total[0]} articles rated · {summ_total[0]} summarized → {output_path}")


if __name__ == "__main__":
    main()
