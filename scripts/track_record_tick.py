#!/usr/bin/env python3
"""Advance the live track record — runs in a GitHub Action, not on Cloud Run.

This is the loop that used to live behind ``POST /track-record/tick``. Doing it
here (on a 16 GB / 6 h runner instead of a 1 GB / 900 s Cloud Run request)
removes the OOM/timeout class entirely: Cloud Run only *serves* the result.

Each run:
  1. scores snapshots whose markets resolved since last time,
  2. appends a cheap hourly price point per open market,
  3. takes LLM forecast snapshots on tracked-open + newly-discovered markets
     (daily for slow markets; intraday slots for short-horizon markets)
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
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import market_data  # noqa: E402
from analyzing_llm_rationale import track_record_live as trl  # noqa: E402
from analyzing_llm_rationale.trackrec_store import DuckDBStore  # noqa: E402

STORE_PATH = Path(os.environ.get("TRACK_STORE_PATH") or ROOT / "data" / "track_record_store.duckdb")
PUBLIC_PATH = Path(os.environ.get("TRACK_PUBLIC_PATH") or ROOT / "static" / "track_record_live.json")

BASE_URL = os.environ.get("FORESEA_BASE_URL", "https://foresea.ink").rstrip("/")
MODEL = os.environ.get("TRACK_MODEL", "gpt-oss-120b")
# Models to forecast each market with, for the paper-trading comparison. The
# first is the primary (the public track record); the rest are graded alongside.
# Each must be in the server's /predict allowlist.
TRACK_MODELS = [m.strip() for m in os.environ.get(
    "TRACK_MODELS", "gpt-oss-120b,gemma-4-31b-it,kimi-k2.6,council,crowd-follow").split(",") if m.strip()]
VARIANT = os.environ.get("TRACK_VARIANT", "variant0_neutral_baseline")
TEMPERATURE = float(os.environ.get("TRACK_TEMPERATURE", "0.0") or 0.0)
PER_VENUE = max(1, min(int(os.environ.get("PER_VENUE", "3") or 3), 5))
PREDICT_API_KEY = os.environ.get("PREDICT_API_KEY") or None
# Gates the evolution-loop bridge (pending-markets / mark-enrolled). When unset,
# the tick simply doesn't pull agent-enrolled seeds (discovery still runs).
TRACK_RECORD_TOKEN = os.environ.get("TRACK_RECORD_TOKEN") or None
# Re-forecast a market if its live price has moved more than this many pp since
# today's snapshot. Prevents stale model probability from being paired with a
# current price on the edge board. Set to 1.0 to disable drift re-forecasting.
PRICE_DRIFT_THRESHOLD = float(os.environ.get("PRICE_DRIFT_THRESHOLD") or trl.PRICE_DRIFT_THRESHOLD)
SHORT_HORIZON_REFORECAST_LEAD_DAYS = float(
    os.environ.get("SHORT_HORIZON_REFORECAST_LEAD_DAYS") or 90.0)
SHORT_HORIZON_SLOT_HOURS = int(
    os.environ.get("SHORT_HORIZON_SLOT_HOURS")
    or trl.SHORT_HORIZON_SLOT_HOURS)
EXPIRY_REFORECAST_LEAD_DAYS = float(
    os.environ.get("EXPIRY_REFORECAST_LEAD_DAYS") or trl.EXPIRY_REFORECAST_LEAD_DAYS)
EXPIRY_SLOT_HOURS = int(
    os.environ.get("EXPIRY_SLOT_HOURS") or trl.EXPIRY_SLOT_HOURS)
# Re-run the LLM forecast for every tracked-open market on every tick (not just
# the daily first pass / price-drift), so the edge board always reflects the
# model's current opinion and matches live /predict. Default on. Each tick then
# costs one /predict per (open market × model) — same load as today's daily pass.
REFORECAST_EACH_TICK = (os.environ.get("REFORECAST_EACH_TICK", "1").strip().lower()
                        in ("1", "true", "yes", "on"))
# How many LLM /predict calls may be in-flight simultaneously.
# crowd-follow is instant (no LLM) and doesn't consume a slot.
PREDICT_CONCURRENCY = int(os.environ.get("PREDICT_CONCURRENCY", "4"))
# Minimum wall-clock gap between any two calls dispatched to the executor.
# With PREDICT_CONCURRENCY=4 and 1s spacing the peak outbound rate is 4 req/s,
# comfortably under SCADS AI limits. Raise PREDICT_CONCURRENCY to go faster;
# lower PREDICT_MIN_INTERVAL_S if the server proves tolerant.
_PREDICT_MIN_INTERVAL_S = float(os.environ.get("PREDICT_MIN_INTERVAL_S", "1.0"))
_PREDICT_TIMEOUT_S = 120
_PREDICT_RETRIES = 3
_predict_rate_lock = threading.Lock()
_last_predict_ts: float = 0.0


def _post_predict(payload: dict) -> dict | None:
    """POST /predict with thread-safe rate-limit pacing + retries.

    Multiple executor threads may call this concurrently; the lock guarantees
    the minimum inter-call interval is enforced globally across all threads."""
    global _last_predict_ts
    with _predict_rate_lock:
        elapsed = time.time() - _last_predict_ts
        if elapsed < _PREDICT_MIN_INTERVAL_S:
            time.sleep(_PREDICT_MIN_INTERVAL_S - elapsed)
        _last_predict_ts = time.time()

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
            if exc.code == 429:
                # Hard rate-limit hit — back off longer before retry.
                time.sleep(30 * (attempt + 1))
            elif exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
        if attempt < _PREDICT_RETRIES - 1:
            time.sleep(2 ** attempt)
    print(f"  predict failed: {last_err}", file=sys.stderr)
    return None


def _get_pending_markets() -> list[tuple[str, str]]:
    """Pull agent-enrolled markets from the server's bridge. Fail-open to []."""
    if not TRACK_RECORD_TOKEN:
        return []
    req = urllib.request.Request(
        f"{BASE_URL}/track-record/pending-markets?limit=50",
        headers={"X-Track-Token": TRACK_RECORD_TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return [(m.get("platform"), m.get("ident")) for m in data.get("markets", [])
                if m.get("platform") and m.get("ident")]
    except Exception as exc:  # noqa: BLE001
        print(f"  pending-markets fetch failed: {exc}", file=sys.stderr)
        return []


def _mark_enrolled(idents: list[str]) -> None:
    """Tell the server which agent-enrolled markets are now tracked. Fail-open."""
    if not TRACK_RECORD_TOKEN or not idents:
        return
    body = json.dumps({"idents": idents}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/track-record/mark-enrolled", data=body,
        headers={"Content-Type": "application/json", "X-Track-Token": TRACK_RECORD_TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  mark-enrolled failed: {exc}", file=sys.stderr)


async def forecast_fn(quote: dict, evidence_top_k: int, model: str | None = None) -> dict | None:
    """Forecast one market via /predict with a chosen model; mirror
    server._track_record_forecast. ``model`` selects an allowlisted server-hosted
    model (server's own key) for the paper-trading comparison."""
    categories = [quote["category"]] if quote.get("category") else []
    payload = {
        "question": quote["question"],
        "description": quote.get("description") or "",
        "resolution_criteria": quote.get("resolution_criteria") or "",
        "attach_evidence": True,
        "evidence_top_k": evidence_top_k,
        "market_platform": quote.get("platform"),
        "market_url": quote.get("market_url"),
        "market_outcome": quote.get("outcome"),
        "market_probability": quote.get("probability"),
        "resolve_time": quote.get("close_time"),
        "publish_time": quote.get("created_time"),
        "categories": categories,
        "market_volume": quote.get("volume"),
        "market_liquidity": quote.get("liquidity"),
        "market_price_change_24h": quote.get("price_change_24h"),
        "market_bid": quote.get("yes_bid"),
        "market_ask": quote.get("yes_ask"),
        "market_price_history": quote.get("market_price_history") or quote.get("price_history") or [],
        "forecast_history": quote.get("forecast_history") or [],
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
        "rationale": res.get("model_rationale") or res.get("rationale") or "",
    }


PRICE_ONLY = "--price-only" in sys.argv


async def main() -> int:
    store = DuckDBStore(STORE_PATH)

    newly_resolved = trl.resolve_open_snapshots(store, market_data)
    price_points = trl.record_price_points(store, market_data)

    recorded = 0
    backfilled = 0
    if not PRICE_ONLY:
        seeds = _get_pending_markets()
        recorded = await trl.record_snapshots(
            store, market_data, forecast_fn,
            models=TRACK_MODELS, default_model=TRACK_MODELS[0], per_venue=PER_VENUE,
            seed_idents=seeds, price_drift_threshold=PRICE_DRIFT_THRESHOLD,
            reforecast_each_tick=REFORECAST_EACH_TICK,
            short_horizon_reforecast_lead_days=SHORT_HORIZON_REFORECAST_LEAD_DAYS,
            short_horizon_slot_hours=SHORT_HORIZON_SLOT_HOURS,
            expiry_reforecast_lead_days=EXPIRY_REFORECAST_LEAD_DAYS,
            expiry_slot_hours=EXPIRY_SLOT_HOURS,
            concurrency=PREDICT_CONCURRENCY)
        backfilled = await trl.backfill_missing_model_snapshots(
            store, forecast_fn,
            models=TRACK_MODELS, default_model=TRACK_MODELS[0],
            concurrency=PREDICT_CONCURRENCY)
        _mark_enrolled([f"{p}:{i}" for p, i in seeds])

    agg = trl.aggregate(store, model=TRACK_MODELS[0], variant=VARIANT, temperature=TEMPERATURE)

    PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_PATH.write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n")

    summary = {
        "snapshots_recorded": recorded,
        "snapshots_backfilled": backfilled,
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
