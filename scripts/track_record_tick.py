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
    "TRACK_MODELS", "gpt-oss-120b,gemma-4-31b-it,kimi-k2.6,crowd-follow").split(",") if m.strip()]
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
# Re-run the LLM forecast for every tracked-open market on every tick (not just
# the daily first pass / price-drift), so the edge board always reflects the
# model's current opinion and matches live /predict. Default on. Each tick then
# costs one /predict per (open market × model) — same load as today's daily pass.
REFORECAST_EACH_TICK = (os.environ.get("REFORECAST_EACH_TICK", "1").strip().lower()
                        in ("1", "true", "yes", "on"))
# Minimum seconds between successive LLM /predict calls to avoid SCADS AI 429s.
# crowd-follow calls are skipped (no LLM). 2s pacing keeps 4 models × 21 markets
# = ~84 calls within ~10 min on the daily first pass, well under rate limits.
_PREDICT_INTER_CALL_SLEEP_S = float(os.environ.get("PREDICT_INTER_CALL_SLEEP_S", "2"))

_PREDICT_TIMEOUT_S = 120
_PREDICT_RETRIES = 3
_last_predict_ts: float = 0.0


def _post_predict(payload: dict) -> dict | None:
    """POST /predict with rate-limit pacing + retries on transient (5xx) failures."""
    global _last_predict_ts
    # Pace calls to avoid SCADS AI 429s. crowd-follow sends no model field so
    # the server handles it cheaply; rate-limit pacing still applies because the
    # server itself must respond.
    elapsed = time.time() - _last_predict_ts
    if elapsed < _PREDICT_INTER_CALL_SLEEP_S:
        time.sleep(_PREDICT_INTER_CALL_SLEEP_S - elapsed)

    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if PREDICT_API_KEY:
        headers["X-API-Key"] = PREDICT_API_KEY
    last_err = None
    for attempt in range(_PREDICT_RETRIES):
        req = urllib.request.Request(f"{BASE_URL}/predict", data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=_PREDICT_TIMEOUT_S) as resp:
                result = json.loads(resp.read())
                _last_predict_ts = time.time()
                return result
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
    _last_predict_ts = time.time()
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

    seeds = _get_pending_markets()  # agent-enrolled markets from the evolution-loop bridge
    newly_resolved = trl.resolve_open_snapshots(store, market_data)
    price_points = trl.record_price_points(store, market_data)
    recorded = await trl.record_snapshots(
        store, market_data, forecast_fn,
        models=TRACK_MODELS, default_model=TRACK_MODELS[0], per_venue=PER_VENUE,
        seed_idents=seeds, price_drift_threshold=PRICE_DRIFT_THRESHOLD,
        reforecast_each_tick=REFORECAST_EACH_TICK)
    # Flip enrolled markets out of the pending queue (and let the server prune).
    _mark_enrolled([f"{p}:{i}" for p, i in seeds])
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
