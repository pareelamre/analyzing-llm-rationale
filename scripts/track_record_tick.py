#!/usr/bin/env python3
"""Advance the live track record — runs in a GitHub Action, not on Cloud Run.

This is the loop that used to live behind ``POST /track-record/tick``. Doing it
here (on a 16 GB / 6 h runner instead of a 1 GB / 900 s Cloud Run request)
removes the OOM/timeout class entirely: Cloud Run only *serves* the result.

Each run:
  1. scores snapshots whose markets resolved since last time,
  2. appends a cheap hourly price point per open market,
  3. takes the daily LLM forecast on tracked-open + newly-discovered markets
     (one HTTP call to ``/predict`` per market — inference stays server-side,
     so no model is held in this runner's memory),
  4. recomputes the public aggregate.

Outputs (committed back to the repo by the workflow):
  - ``data/track_record_store.json``   — full entity store (source of truth)
  - ``static/track_record_live.json``  — the public aggregate served at
    ``GET /track-record`` once it has resolved forecasts.

Env:
  FORESEA_BASE_URL  forecast endpoint base   (default https://foresea.ink)
  TRACK_MODEL       model label for metadata (default gpt-oss-120b)
  TRACK_VARIANT     variant label            (default variant0_neutral_baseline)
  TRACK_TEMPERATURE temperature label        (default 0.0)
  PER_VENUE         new markets per venue     (default 3, clamped 1..5)
  PREDICT_API_KEY   sent as X-API-Key if /predict is protected (optional)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import market_data  # noqa: E402
from analyzing_llm_rationale import track_record_live as trl  # noqa: E402
from analyzing_llm_rationale.trackrec_store import FileStore  # noqa: E402

STORE_PATH = Path(os.environ.get("TRACK_STORE_PATH") or ROOT / "data" / "track_record_store.json")
PUBLIC_PATH = Path(os.environ.get("TRACK_PUBLIC_PATH") or ROOT / "static" / "track_record_live.json")

BASE_URL = os.environ.get("FORESEA_BASE_URL", "https://foresea.ink").rstrip("/")
MODEL = os.environ.get("TRACK_MODEL", "gpt-oss-120b")
# Models to forecast each market with, for the paper-trading comparison. The
# first is the primary (the public track record); the rest are graded alongside.
# Each must be in the server's /predict allowlist.
TRACK_MODELS = [m.strip() for m in os.environ.get(
    "TRACK_MODELS", "gpt-oss-120b,gemma-4-31b-it,kimi-k2.6").split(",") if m.strip()]
VARIANT = os.environ.get("TRACK_VARIANT", "variant0_neutral_baseline")
TEMPERATURE = float(os.environ.get("TRACK_TEMPERATURE", "0.0") or 0.0)
PER_VENUE = max(1, min(int(os.environ.get("PER_VENUE", "3") or 3), 5))
PREDICT_API_KEY = os.environ.get("PREDICT_API_KEY") or None

_PREDICT_TIMEOUT_S = 120
_PREDICT_RETRIES = 3


def _post_predict(payload: dict) -> dict | None:
    """POST /predict with a few retries on transient (5xx) failures."""
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if PREDICT_API_KEY:
        headers["X-API-Key"] = PREDICT_API_KEY
    last_err = None
    for attempt in range(_PREDICT_RETRIES):
        req = urllib.request.Request(f"{BASE_URL}/predict", data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=_PREDICT_TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code < 500:  # client error — retrying won't help
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
        if attempt < _PREDICT_RETRIES - 1:
            time.sleep(2 ** attempt)
    print(f"  predict failed: {last_err}", file=sys.stderr)
    return None


async def forecast_fn(quote: dict, evidence_top_k: int, model: str | None = None) -> dict | None:
    """Forecast one market via /predict with a chosen model; mirror
    server._track_record_forecast. ``model`` selects an allowlisted server-hosted
    model (server's own key) for the paper-trading comparison."""
    payload = {
        "question": quote["question"],
        "attach_evidence": True,
        "evidence_top_k": evidence_top_k,
        "market_platform": quote.get("platform"),
        "market_url": quote.get("market_url"),
        "market_outcome": quote.get("outcome"),
        "market_probability": quote.get("probability"),
    }
    if model:
        payload["model"] = model
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, _post_predict, payload)
    if not res:
        return None
    analysis = res.get("market_analysis")
    if not analysis or analysis.get("model_probability") is None:
        return None
    return {
        "model_probability": analysis["model_probability"],
        "market_probability": analysis.get("market_probability"),
        "evidence_count": len(res.get("evidence_sources") or []),
    }


async def main() -> int:
    store = FileStore(STORE_PATH)

    newly_resolved = trl.resolve_open_snapshots(store, market_data)
    price_points = trl.record_price_points(store, market_data)
    recorded = await trl.record_snapshots(
        store, market_data, forecast_fn,
        models=TRACK_MODELS, default_model=TRACK_MODELS[0], per_venue=PER_VENUE)
    agg = trl.aggregate(store, model=TRACK_MODELS[0], variant=VARIANT, temperature=TEMPERATURE)

    PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_PATH.write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n")

    summary = {
        "snapshots_recorded": recorded,
        "price_points_recorded": price_points,
        "snapshots_resolved": newly_resolved,
        "n_markets_resolved": agg.get("n_markets_resolved"),
        "n_markets_open": agg.get("n_markets_open"),
        "n_snapshots_resolved": agg.get("n_snapshots_resolved"),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
