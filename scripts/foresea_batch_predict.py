#!/usr/bin/env python3
"""Batch-call Foresea /predict for all questions in kg_dataset.json.

Sends each question (plus any fetched news_articles) to the live Foresea
/predict endpoint and saves structured results to a JSON file.

Usage:
    python scripts/foresea_batch_predict.py
    python scripts/foresea_batch_predict.py --base-url http://localhost:8000
    python scripts/foresea_batch_predict.py --workers 20 --limit 100
    python scripts/foresea_batch_predict.py --api-key sk-xxx

Output: results/foresea_predict/results.json
        (incrementally checkpointed every --checkpoint-every records)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_BASE_URL = "https://foresea.ink"
DEFAULT_INPUT = Path("data/kg_dataset.json")
DEFAULT_OUTPUT = Path("results/foresea_predict/results.json")
DEFAULT_WORKERS = 20
CHECKPOINT_EVERY = 100
REQUEST_TIMEOUT = 90  # seconds per /predict call
MAX_RETRIES = 4


def predict(
    question: str,
    news_articles: List[Dict],
    base_url: str,
    api_key: Optional[str],
    variant: str,
) -> Optional[Dict[str, Any]]:
    """Call POST /predict and return the parsed response, or None on failure."""
    url = base_url.rstrip("/") + "/predict"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload: Dict[str, Any] = {
        "question": question,
        "variant": variant,
        "news_articles": news_articles,
        "chat_mode": False,
    }
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", backoff))
                time.sleep(retry_after + 1)
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES - 1:
                return {"_error": "timeout"}
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except Exception as e:
            return {"_error": str(e)}
    return {"_error": "max retries exceeded"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--base-url", default=os.environ.get("FORESEA_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api-key", default=os.environ.get("FORESEA_API_KEY"))
    ap.add_argument("--variant", default="variant0_neutral_baseline")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--source", default=None, help="kalshi|polymarket|manifold|metaculus")
    ap.add_argument("--rate-limit", type=float, default=RATE_LIMIT_PER_MIN,
                    help="Max requests per minute (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = json.loads(input_path.read_text())
    print(f"Loaded {len(records)} records from {input_path}")

    # Load existing results to resume
    existing: Dict[str, Dict] = {}
    if output_path.exists():
        try:
            for row in json.loads(output_path.read_text()):
                if isinstance(row, dict) and row.get("id") and not row.get("_error"):
                    existing[row["id"]] = row
            print(f"Resuming: {len(existing)} already done")
        except Exception:
            pass

    needs = [
        r for r in records
        if r.get("id") not in existing
        and (args.source is None or r.get("source") == args.source)
    ]
    if args.limit:
        needs = needs[:args.limit]

    print(f"{len(needs)} questions to forecast (skipping {len(existing)} done)")
    if args.dry_run:
        avg_s = 20  # typical /predict latency
        est = len(needs) * avg_s / args.workers / 60
        print(f"Estimated runtime: ~{est:.0f} min @ {args.workers} workers / {args.rate_limit}/min")
        return

    global _bucket
    _bucket = _TokenBucket(args.rate_limit)

    _lock = threading.Lock()
    results: Dict[str, Dict] = dict(existing)
    done = [0]
    errors = [0]

    def process(record: Dict) -> Tuple[str, Optional[Dict]]:
        qid = record.get("id", "")
        q = record.get("question", "")
        arts = record.get("news_articles") or []
        resp = predict(q, arts, args.base_url, args.api_key, args.variant)
        return qid, resp

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, r): r.get("id") for r in needs}

        for future in concurrent.futures.as_completed(futures):
            qid, resp = future.result()
            src = futures[future]

            with _lock:
                done[0] += 1
                seq = done[0]

                if resp and not resp.get("_error"):
                    # Merge result with original record metadata
                    record = next((r for r in needs if r.get("id") == qid), {})
                    row = {
                        "id": qid,
                        "question": record.get("question", ""),
                        "answer": record.get("answer"),
                        "source": record.get("source"),
                        "resolve_time": record.get("resolve_time"),
                    }
                    # Extract key fields from response
                    ma = resp.get("market_analysis") or {}
                    row["model_probability"] = ma.get("model_probability")
                    row["model_probability_raw"] = ma.get("model_probability_raw")
                    row["edge"] = ma.get("edge")
                    row["stance"] = ma.get("stance")
                    row["confidence"] = resp.get("confidence")
                    row["predicted_answer"] = resp.get("predicted_answer")
                    row["rationale"] = resp.get("rationale")
                    row["question_type"] = resp.get("question_type")
                    row["_raw"] = resp  # keep full response
                    results[qid] = row
                else:
                    errors[0] += 1
                    err_msg = resp.get("_error", "unknown") if resp else "null response"
                    results[qid] = {"id": qid, "_error": err_msg}

            prob = results[qid].get("model_probability")
            prob_str = f"{prob:.2f}" if prob is not None else "err"
            rec = next((r for r in needs if r.get("id") == qid), {})
            print(f"  [{seq}/{len(needs)}] {rec.get('source','?')}: "
                  f"{rec.get('question','')[:55]}… → {prob_str}", flush=True)

            if seq % args.checkpoint_every == 0:
                with _lock:
                    _save(output_path, results, records)
                print(f"  ── checkpoint {seq} (errors={errors[0]}) ──", flush=True)

    _save(output_path, results, records)
    good = sum(1 for v in results.values() if not v.get("_error"))
    print(f"\nDone: {good} forecasts saved · {errors[0]} errors → {output_path}")


def _save(path: Path, results: Dict[str, Dict], records: List[Dict]) -> None:
    ordered = [results[r["id"]] for r in records if r.get("id") in results]
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
