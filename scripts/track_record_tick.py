#!/usr/bin/env python3
"""Advance the live track record — runs in a GitHub Action, not on Cloud Run.

This is the loop that used to live behind ``POST /track-record/tick``. Doing it
here (on a 16 GB / 6 h runner instead of a 1 GB / 900 s Cloud Run request)
removes the OOM/timeout class entirely: Cloud Run only *serves* the result.

Each run:
  1. scores snapshots whose markets resolved since last time,
  2. appends a cheap hourly price point per open market,
  3. takes hourly LLM forecast snapshots on tracked-open + newly-discovered markets
     (one HTTP call to ``/predict`` per market — inference stays server-side,
     so no model is held in this runner's memory),
  4. mirrors snapshots and resolutions into the append-only forecast ledger,
  5. recomputes the public aggregate.

Outputs (committed back to the repo by the workflow):
  - ``data/track_record_store.duckdb``: snapshots plus immutable ledger events
  - ``static/track_record_live.json``  — the public aggregate served at
    ``GET /track-record`` once it has resolved forecasts.
  - ``static/forecast_evaluation.json``: the provenance-separated internal
    evaluation report and strategy promotion gates.

Env:
  FORESEA_BASE_URL  forecast endpoint base   (default https://foresea.ink)
  TRACK_MODEL       stable primary model for public metrics (default council)
  TRACK_VARIANT     variant label            (default variant0_neutral_baseline)
  TRACK_TEMPERATURE temperature label        (default 0.0)
  PER_VENUE         new markets per venue     (default 3, clamped 1..5)
  TRACK_RECORD_PREDICT_MODE  remote|local     (default remote)
  SCADS_AI_API_KEY   model credential; used by Cloud Run in remote mode and by
                    GitHub Actions in local mode
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import market_data  # noqa: E402
from analyzing_llm_rationale import track_record_live as trl  # noqa: E402
from analyzing_llm_rationale.config import scads_track_model_labels  # noqa: E402
from analyzing_llm_rationale.forecast_evaluation import ResolvedForecast  # noqa: E402
from analyzing_llm_rationale.forecast_evaluation_report import (  # noqa: E402
    EvaluationPolicy,
    build_evaluation_artifact,
    compact_evaluation_summary,
)
from analyzing_llm_rationale.forecast_ledger import (  # noqa: E402
    ForecastLedger,
    sync_snapshot_ledger,
)
from analyzing_llm_rationale.observability import init_observability  # noqa: E402
from analyzing_llm_rationale.trackrec_store import DuckDBStore  # noqa: E402

logger = logging.getLogger("foresea.track_record")
_context_tracer = otel_trace.get_tracer("foresea.track_record")
_context_meter = otel_metrics.get_meter("foresea.track_record")
_context_packages = _context_meter.create_counter(
    "forecast.context.packages",
    unit="1",
    description="Track-record forecast context packages by completeness",
)
_predict_calls = _context_meter.create_counter(
    "track_record.predict.calls",
    unit="1",
    description="Track-record prediction calls by dispatch mode and outcome",
)
_forecast_shard_runs = _context_meter.create_counter(
    "track_record.forecast.shard_runs",
    unit="1",
    description="Track-record forecast shard runs by outcome",
)

STORE_PATH = Path(os.environ.get("TRACK_STORE_PATH") or ROOT / "data" / "track_record_store.duckdb")
PUBLIC_PATH = Path(os.environ.get("TRACK_PUBLIC_PATH") or ROOT / "static" / "track_record_live.json")
MARK_TO_MARKET_PATH = Path(
    os.environ.get("TRACK_MTM_PATH")
    or ROOT / "static" / "mark_to_market_live.json"
)
EVALUATION_PATH = Path(
    os.environ.get("TRACK_EVALUATION_PATH")
    or ROOT / "static" / "forecast_evaluation.json"
)

BASE_URL = os.environ.get("FORESEA_BASE_URL", "https://foresea.ink").rstrip("/")
MODEL = os.environ.get("TRACK_MODEL", "council").strip()
SCADS_TRACK_MODELS = scads_track_model_labels(ROOT / "configs" / "models.yaml")
DEFAULT_TRACK_MODELS = ("council", *SCADS_TRACK_MODELS, "crowd-follow")


def _csv_list(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _resolve_model_shard_index(
    *,
    count: int,
    raw_index: str | None,
    slot_minutes: int,
    now_s: float | None = None,
) -> int:
    if count <= 1:
        return 0
    if raw_index is not None and raw_index.strip():
        index = int(raw_index)
        if index < 0 or index >= count:
            raise RuntimeError(
                f"TRACK_MODEL_SHARD_INDEX must be between 0 and {count - 1}"
            )
        return index
    slot_s = max(1, slot_minutes) * 60
    return int((now_s if now_s is not None else time.time()) // slot_s) % count


def _select_model_shard(
    models: list[str],
    *,
    count: int,
    index: int,
    always: list[str] | None = None,
) -> list[str]:
    if count <= 1:
        return list(models)
    always_set = set(always or [])
    selected: list[str] = []
    shardable = [model for model in models if model not in always_set]
    selected.extend(
        model
        for position, model in enumerate(shardable)
        if position % count == index
    )
    selected.extend(
        model
        for model in models
        if model in always_set and model not in selected
    )
    return selected


# Models to forecast each market with, for the paper-trading comparison. The
# first is the primary (the public track record); the rest are graded alongside.
# Each must be in the server's /predict allowlist.
ALL_TRACK_MODELS = _csv_list(os.environ.get(
    "TRACK_MODELS",
    ",".join(DEFAULT_TRACK_MODELS),
))
if MODEL not in ALL_TRACK_MODELS:
    raise RuntimeError(f"TRACK_MODEL {MODEL!r} must be included in TRACK_MODELS")
TRACK_MODEL_SHARD_COUNT = max(
    1, int(os.environ.get("TRACK_MODEL_SHARD_COUNT", "1") or 1)
)
TRACK_MODEL_SHARD_SLOT_MINUTES = max(
    1, int(os.environ.get("TRACK_MODEL_SHARD_SLOT_MINUTES", "15") or 15)
)
TRACK_MODEL_SHARD_INDEX = _resolve_model_shard_index(
    count=TRACK_MODEL_SHARD_COUNT,
    raw_index=os.environ.get("TRACK_MODEL_SHARD_INDEX"),
    slot_minutes=TRACK_MODEL_SHARD_SLOT_MINUTES,
)
TRACK_MODEL_SHARD_ALWAYS = _csv_list(
    os.environ.get("TRACK_MODEL_SHARD_ALWAYS", "crowd-follow")
)
TRACK_MODELS = _select_model_shard(
    ALL_TRACK_MODELS,
    count=TRACK_MODEL_SHARD_COUNT,
    index=TRACK_MODEL_SHARD_INDEX,
    always=TRACK_MODEL_SHARD_ALWAYS,
)
TRACK_TARGET_SHARD_COUNT = max(
    1, int(os.environ.get("TRACK_TARGET_SHARD_COUNT", "1") or 1)
)
TRACK_TARGET_SHARD_SLOT_MINUTES = max(
    1, int(os.environ.get("TRACK_TARGET_SHARD_SLOT_MINUTES", "15") or 15)
)
TRACK_TARGET_SHARD_INDEX = _resolve_model_shard_index(
    count=TRACK_TARGET_SHARD_COUNT,
    raw_index=os.environ.get("TRACK_TARGET_SHARD_INDEX"),
    slot_minutes=TRACK_TARGET_SHARD_SLOT_MINUTES,
)
TRACK_FORECAST_MAX_TARGETS = int(os.environ.get("TRACK_FORECAST_MAX_TARGETS", "0") or 0)
if TRACK_FORECAST_MAX_TARGETS <= 0:
    TRACK_FORECAST_MAX_TARGETS = None
VARIANT = os.environ.get("TRACK_VARIANT", "variant0_neutral_baseline")
TEMPERATURE = float(os.environ.get("TRACK_TEMPERATURE", "0.0") or 0.0)
PER_VENUE = max(1, min(int(os.environ.get("PER_VENUE", "3") or 3), 5))
# Dedicated convergence-window discovery: markets specifically at 7-14d lead time.
# Higher limit than PER_VENUE since this is the target data-collection window.
CONVERGENCE_PER_VENUE = max(0, int(os.environ.get("CONVERGENCE_PER_VENUE", "20") or 20))
PREDICT_API_KEY = os.environ.get("PREDICT_API_KEY") or None
# Gates the evolution-loop bridge (pending-markets / mark-enrolled). When unset,
# the tick simply doesn't pull agent-enrolled seeds (discovery still runs).
TRACK_RECORD_TOKEN = os.environ.get("TRACK_RECORD_TOKEN") or None
TRACK_RECORD_EVIDENCE_DETAIL = os.environ.get("TRACK_RECORD_EVIDENCE_DETAIL", "summary").strip().lower()
if TRACK_RECORD_EVIDENCE_DETAIL not in {"summary", "full"}:
    raise RuntimeError("TRACK_RECORD_EVIDENCE_DETAIL must be 'summary' or 'full'")
# Re-forecast a market if its live price has moved more than this many pp since
# today's snapshot. Prevents stale model probability from being paired with a
# current price on the edge board. Set to 1.0 to disable drift re-forecasting.
PRICE_DRIFT_THRESHOLD = float(os.environ.get("PRICE_DRIFT_THRESHOLD") or trl.PRICE_DRIFT_THRESHOLD)
SHORT_HORIZON_REFORECAST_LEAD_DAYS = float(
    os.environ.get("SHORT_HORIZON_REFORECAST_LEAD_DAYS") or 90.0)
SHORT_HORIZON_SLOT_HOURS = int(
    os.environ.get("SHORT_HORIZON_SLOT_HOURS")
    or 1)
EXPIRY_REFORECAST_LEAD_DAYS = float(
    os.environ.get("EXPIRY_REFORECAST_LEAD_DAYS") or trl.EXPIRY_REFORECAST_LEAD_DAYS)
EXPIRY_SLOT_HOURS = int(
    os.environ.get("EXPIRY_SLOT_HOURS") or trl.EXPIRY_SLOT_HOURS)
CYCLE_INTERVAL_MINUTES = max(1, int(os.environ.get("CYCLE_INTERVAL_MINUTES", "15") or 15))
SNAPSHOT_SLOT_MINUTES = int(os.environ.get("SNAPSHOT_SLOT_MINUTES", "0") or 0) or None
# Re-run the LLM forecast for every tracked-open market on every tick (not just
# the daily first pass / price-drift), so the edge board always reflects the
# model's current opinion and matches live /predict. Default on. Each tick then
# costs one /predict per (open market × model) — same load as today's daily pass.
REFORECAST_EACH_TICK = (os.environ.get("REFORECAST_EACH_TICK", "1").strip().lower()
                        in ("1", "true", "yes", "on"))
# Production public track record should only score forecasts captured before
# resolution. Resolved-market backfill is kept as an explicit diagnostic tool,
# but is excluded from the normal live board by default.
ALLOW_RESOLVED_BACKFILL = (
    os.environ.get("ALLOW_RESOLVED_BACKFILL", "0").strip().lower()
    in ("1", "true", "yes", "on")
)
# How many LLM /predict calls may be in-flight simultaneously.
# crowd-follow is instant (no LLM) and doesn't consume a slot.
PREDICT_CONCURRENCY = int(os.environ.get("PREDICT_CONCURRENCY", "4"))
# Minimum wall-clock gap between any two calls dispatched to the executor.
# With PREDICT_CONCURRENCY=4 and 1s spacing the peak outbound rate is 4 req/s,
# comfortably under SCADS AI limits. Raise PREDICT_CONCURRENCY to go faster;
# lower PREDICT_MIN_INTERVAL_S if the server proves tolerant.
_PREDICT_MIN_INTERVAL_S = float(os.environ.get("PREDICT_MIN_INTERVAL_S", "1.0"))
_PREDICT_TIMEOUT_S = max(
    1.0, float(os.environ.get("PREDICT_TIMEOUT_S", "120") or 120)
)
_PREDICT_RETRIES = max(1, int(os.environ.get("PREDICT_RETRIES", "3") or 3))
_PREDICT_RETRY_BACKOFF_BASE_S = max(
    0.0, float(os.environ.get("PREDICT_RETRY_BACKOFF_BASE_S", "5") or 5.0)
)
_PREDICT_RETRY_BACKOFF_MAX_S = max(
    0.0, float(os.environ.get("PREDICT_RETRY_BACKOFF_MAX_S", "120") or 120.0)
)
_PREDICT_RETRY_JITTER_FRACTION = max(
    0.0, float(os.environ.get("PREDICT_RETRY_JITTER_FRACTION", "0.25") or 0.25)
)
PREDICT_MODE = os.environ.get("TRACK_RECORD_PREDICT_MODE", "remote").strip().lower()
_LOCAL_PREDICT_MODES = {"local", "direct", "github"}
LOCAL_PREDICT_MODEL = os.environ.get("TRACK_RECORD_LOCAL_MODEL", "gpt-oss-120b").strip()
LOCAL_PREDICT_DISABLE_EVIDENCE = (
    os.environ.get("TRACK_RECORD_LOCAL_DISABLE_EVIDENCE", "0").strip().lower()
    in ("1", "true", "yes", "on")
)
_PREDICT_FAILURE_CIRCUIT_THRESHOLD = max(
    1,
    int(os.environ.get("PREDICT_FAILURE_CIRCUIT_THRESHOLD", "8") or 8),
)
_SNAPSHOT_PASS_RETRIES = max(1, int(os.environ.get("TRACK_RECORD_SNAPSHOT_PASS_RETRIES", "3") or 3))
_SNAPSHOT_PASS_RETRY_SLEEP_S = float(
    os.environ.get("TRACK_RECORD_SNAPSHOT_PASS_RETRY_SLEEP_S", "30") or 30.0
)
EVALUATION_POLICY = EvaluationPolicy(
    min_resolved_markets=max(
        1, int(os.environ.get("EVALUATION_MIN_RESOLVED_MARKETS", "100") or 100)
    ),
    min_paper_trades=max(
        1, int(os.environ.get("EVALUATION_MIN_PAPER_TRADES", "30") or 30)
    ),
    min_skill_lower_bound=float(
        os.environ.get("EVALUATION_MIN_SKILL_LOWER_BOUND", "0.0") or 0.0
    ),
    max_drawdown=float(
        os.environ.get("EVALUATION_MAX_DRAWDOWN", "0.20") or 0.20
    ),
    min_edge=float(os.environ.get("EVALUATION_MIN_EDGE", "0.05") or 0.05),
    requested_fraction=float(
        os.environ.get("EVALUATION_REQUESTED_FRACTION", "0.02") or 0.02
    ),
    fee_fraction=float(
        os.environ.get("EVALUATION_FEE_FRACTION", "0.01") or 0.01
    ),
    max_total_exposure=float(
        os.environ.get("EVALUATION_MAX_TOTAL_EXPOSURE", "0.25") or 0.25
    ),
)
_predict_rate_lock = threading.Lock()
_predict_circuit_lock = threading.Lock()
_last_predict_ts: float = 0.0
_predict_consecutive_failures: Counter = Counter()
_predict_circuit_open_models: set[str] = set()
_predict_stats = Counter()
_local_predict_lock = threading.Lock()
_local_predict_ready = False


def _predict_mode() -> str:
    return "local" if PREDICT_MODE in _LOCAL_PREDICT_MODES else "remote"


def _local_predict_credentials_available() -> bool:
    return bool(os.environ.get("SCADS_AI_API_KEY") or (ROOT / "SCADS_AI_API_KEY.txt").exists())


def _record_predict_call(model_label: str, outcome: str) -> None:
    _predict_calls.add(1, {
        "forecast.model": model_label,
        "predict.mode": _predict_mode(),
        "outcome": outcome,
    })


def _reset_predict_circuit() -> None:
    with _predict_circuit_lock:
        _predict_consecutive_failures.clear()
        _predict_circuit_open_models.clear()


def _predict_model_label(payload: dict) -> str:
    return str(payload.get("model") or MODEL or "default").strip() or "default"


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Return a bounded provider error detail without making logging itself fail."""
    try:
        raw = exc.read(2048).decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        return ""
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return str(detail)[:500]
    except (TypeError, ValueError):
        pass
    return raw[:500]


def _retry_after_seconds(headers: object | None) -> float | None:
    if not headers:
        return None
    try:
        value = headers.get("Retry-After")  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _predict_retry_delay(
    attempt: int,
    *,
    status_code: int | None = None,
    retry_after_s: float | None = None,
) -> float:
    if retry_after_s is not None:
        base = retry_after_s
    elif status_code == 429:
        base = max(30.0 * (attempt + 1), _PREDICT_RETRY_BACKOFF_BASE_S)
    else:
        base = _PREDICT_RETRY_BACKOFF_BASE_S * (2 ** attempt)
    delay = min(base, _PREDICT_RETRY_BACKOFF_MAX_S) if _PREDICT_RETRY_BACKOFF_MAX_S else base
    if _PREDICT_RETRY_JITTER_FRACTION:
        delay += random.uniform(0.0, delay * _PREDICT_RETRY_JITTER_FRACTION)
    return delay


def _init_local_predict() -> None:
    """Initialise server prediction state inside the GitHub runner process."""
    global _local_predict_ready
    if _local_predict_ready:
        return
    with _local_predict_lock:
        if _local_predict_ready:
            return
        from analyzing_llm_rationale.cli import init_server_state

        init_server_state(SimpleNamespace(
            model=LOCAL_PREDICT_MODEL,
            variant=VARIANT,
            variants_config=ROOT / "configs" / "variants.yaml",
            models_config=ROOT / "configs" / "models.yaml",
            temperature=TEMPERATURE,
            max_tokens=int(os.environ.get("MAX_TOKENS", "2048")),
            provider=None,
            local_model_name=None,
            router_model_name=None,
            api_base_url=None,
            api_key_env_var=None,
            api_key_file=None,
            device=os.environ.get("MODEL_DEVICE", "cpu"),
            request_timeout_s=_PREDICT_TIMEOUT_S,
            model_label=None,
            disable_evidence=LOCAL_PREDICT_DISABLE_EVIDENCE,
            newsapi_key_env_var="NEWSAPI_KEY",
            evidence_source=None,
            disable_query_planner=True,
        ))
        _local_predict_ready = True
        print(
            "track-record direct predict initialized "
            f"model={LOCAL_PREDICT_MODEL} evidence_disabled={LOCAL_PREDICT_DISABLE_EVIDENCE}"
        )


@_context_tracer.start_as_current_span("track_record.predict.local")
def _local_predict(payload: dict) -> dict:
    """Run the same prediction code in-process instead of posting to Cloud Run."""
    _init_local_predict()
    from analyzing_llm_rationale.server import PredictRequest, predict

    span = otel_trace.get_current_span()
    span.set_attributes({
        "forecast.model": _predict_model_label(payload),
        "predict.mode": "local",
    })
    try:
        response = asyncio.run(predict(PredictRequest(**payload), request=None))
    except Exception as exc:
        cause_parts: list[str] = []
        cause = getattr(exc, "__cause__", None)
        while cause is not None and len(cause_parts) < 3:
            cause_parts.append(f"{type(cause).__name__}: {cause}")
            cause = getattr(cause, "__cause__", None)
        if cause_parts:
            print(
                "  local predict cause "
                f"[model={_predict_model_label(payload)}]: "
                + " <- ".join(cause_parts),
                file=sys.stderr,
            )
        raise
    return response.model_dump(mode="json")


def _remote_predict(payload: dict, headers: dict[str, str]) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE_URL}/predict", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=_PREDICT_TIMEOUT_S) as resp:
        return json.loads(resp.read())


def _post_predict(payload: dict) -> dict | None:
    """Run /predict with thread-safe rate-limit pacing + retries.

    Multiple executor threads may call this concurrently; the lock guarantees
    the minimum inter-call interval is enforced globally across all threads."""
    global _last_predict_ts
    model_label = _predict_model_label(payload)
    with _predict_circuit_lock:
        if model_label in _predict_circuit_open_models:
            _predict_stats["circuit_skipped"] += 1
            _predict_stats[f"circuit_skipped_model:{model_label}"] += 1
            return None

    with _predict_rate_lock:
        elapsed = time.time() - _last_predict_ts
        if elapsed < _PREDICT_MIN_INTERVAL_S:
            time.sleep(_PREDICT_MIN_INTERVAL_S - elapsed)
        _last_predict_ts = time.time()

    headers = {"Content-Type": "application/json"}
    if PREDICT_API_KEY:
        headers["X-API-Key"] = PREDICT_API_KEY
    if TRACK_RECORD_TOKEN:
        headers["X-Track-Token"] = TRACK_RECORD_TOKEN
    last_err = None
    _predict_stats["attempts"] += 1
    for attempt in range(_PREDICT_RETRIES):
        retry_delay_s: float | None = None
        try:
            result = (
                _local_predict(payload)
                if PREDICT_MODE in _LOCAL_PREDICT_MODES
                else _remote_predict(payload, headers)
            )
            _predict_stats["successes"] += 1
            _predict_stats[f"success_model:{model_label}"] += 1
            _record_predict_call(model_label, "success")
            with _predict_circuit_lock:
                _predict_consecutive_failures[model_label] = 0
            return result
        except urllib.error.HTTPError as exc:
            detail = _http_error_detail(exc)
            last_err = f"HTTP {exc.code}" + (f": {detail}" if detail else "")
            _predict_stats[f"http_{exc.code}"] += 1
            if exc.code == 429:
                # Hard rate-limit hit — back off longer before retry.
                retry_delay_s = _predict_retry_delay(
                    attempt,
                    status_code=exc.code,
                    retry_after_s=_retry_after_seconds(exc.headers),
                )
            elif exc.code < 500:
                break
            else:
                retry_delay_s = _predict_retry_delay(
                    attempt,
                    status_code=exc.code,
                    retry_after_s=_retry_after_seconds(exc.headers),
                )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
            retry_delay_s = _predict_retry_delay(attempt)
        except Exception as exc:  # noqa: BLE001
            status_code = getattr(exc, "status_code", None)
            if status_code:
                detail = getattr(exc, "detail", "")
                last_err = f"HTTP {status_code}" + (f": {detail}" if detail else "")
                status_int = int(status_code)
                _predict_stats[f"http_{status_int}"] += 1
                if status_int < 500 and status_int != 429:
                    break
                retry_delay_s = _predict_retry_delay(
                    attempt,
                    status_code=status_int,
                    retry_after_s=_retry_after_seconds(getattr(exc, "headers", None)),
                )
            else:
                last_err = f"{type(exc).__name__}: {exc}"
                retry_delay_s = _predict_retry_delay(attempt)
        if attempt < _PREDICT_RETRIES - 1:
            delay = retry_delay_s if retry_delay_s is not None else _predict_retry_delay(attempt)
            _predict_stats["retry_sleeps"] += 1
            _predict_stats[f"retry_sleep_model:{model_label}"] += 1
            time.sleep(delay)
    _predict_stats["failures"] += 1
    _predict_stats[f"failure_model:{model_label}"] += 1
    _record_predict_call(model_label, "failure")
    circuit_opened = False
    with _predict_circuit_lock:
        _predict_consecutive_failures[model_label] += 1
        if (
            _predict_consecutive_failures[model_label]
            >= _PREDICT_FAILURE_CIRCUIT_THRESHOLD
            and model_label not in _predict_circuit_open_models
        ):
            _predict_circuit_open_models.add(model_label)
            _predict_stats["circuit_opened"] += 1
            _predict_stats[f"circuit_opened_model:{model_label}"] += 1
            circuit_opened = True
    print(f"  predict failed [model={model_label}]: {last_err}", file=sys.stderr)
    if circuit_opened:
        print(
            f"track-record forecast circuit opened for {model_label!r} after "
            f"{_predict_consecutive_failures[model_label]} consecutive failures; "
            "skipping only that model's queued markets for this pass.",
            file=sys.stderr,
        )
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


@_context_tracer.start_as_current_span("forecast.context_deliver")
async def forecast_fn(quote: dict, evidence_top_k: int, model: str | None = None) -> dict | None:
    """Forecast one market via /predict with a chosen model; mirror
    server._track_record_forecast. ``model`` selects an allowlisted server-hosted
    model (server's own key) for the paper-trading comparison."""
    categories = [quote["category"]] if quote.get("category") else []
    platform = str(quote.get("platform") or "unknown").strip().lower()
    ident = str(quote.get("ident") or "").strip()
    resolution_criteria = str(quote.get("resolution_criteria") or "").strip()
    venue_articles = [
        article for article in (quote.get("venue_news_articles") or [])
        if isinstance(article, dict)
    ]
    span = otel_trace.get_current_span()
    span.set_attributes({
        "market.platform": platform,
        "market.id": ident,
        "forecast.model": model or MODEL,
        "forecast.context.rules_present": bool(resolution_criteria),
        "forecast.context.venue_articles": len(venue_articles),
    })
    if not resolution_criteria:
        logger.warning(
            "track-record forecast missing market rules platform=%s ident=%s",
            platform,
            ident,
        )
    payload = {
        "question": quote["question"],
        "description": quote.get("description") or "",
        "resolution_criteria": resolution_criteria,
        "news_articles": venue_articles,
        # Force the structured forecast template. Without this the server's
        # always-on chat mode answers conversationally (question_type="chat",
        # no market_analysis) and every LLM snapshot is silently dropped.
        "chat_mode": False,
        "attach_evidence": True,
        "evidence_top_k": evidence_top_k,
        "evidence_detail": TRACK_RECORD_EVIDENCE_DETAIL,
        "market_platform": quote.get("platform"),
        "market_ident": quote.get("ident"),
        "market_url": quote.get("market_url"),
        "market_outcome": quote.get("outcome"),
        "market_probability": quote.get("probability"),
        "market_resolution_source": quote.get("resolution_source"),
        "market_no_sub_title": quote.get("no_sub_title"),
        "market_expected_expiration_time": quote.get("expected_expiration_time"),
        "market_floor_strike": quote.get("floor_strike"),
        "market_cap_strike": quote.get("cap_strike"),
        "resolve_time": quote.get("close_time"),
        "publish_time": quote.get("created_time"),
        "categories": categories,
        "market_volume": quote.get("volume"),
        "market_liquidity": quote.get("liquidity"),
        "market_price_change_24h": quote.get("price_change_24h"),
        "market_price_change_7d": quote.get("price_change_7d"),
        "market_bid": quote.get("yes_bid"),
        "market_ask": quote.get("yes_ask"),
        "market_last_trade_price": quote.get("last_trade_price"),
        "market_price_history": quote.get("market_price_history") or quote.get("price_history") or [],
        "forecast_history": quote.get("forecast_history") or [],
    }
    if model:
        payload["model"] = model
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, _post_predict, payload)
    if not res:
        span.set_attribute("forecast.context.outcome", "delivery_failed")
        _context_packages.add(1, {
            "market.platform": platform,
            "rules_present": str(bool(resolution_criteria)).lower(),
            "outcome": "delivery_failed",
        })
        return None
    returned_model = str(res.get("model_key") or "").strip()
    if model and returned_model != model:
        _predict_stats["model_mismatches"] += 1
        print(
            f"  predict model mismatch: requested {model!r}, "
            f"received {returned_model or '<missing>'!r}",
            file=sys.stderr,
        )
        span.set_attribute("forecast.context.outcome", "model_mismatch")
        _context_packages.add(1, {
            "market.platform": platform,
            "rules_present": str(bool(resolution_criteria)).lower(),
            "outcome": "model_mismatch",
        })
        return None
    analysis = res.get("market_analysis")
    if not analysis or analysis.get("model_probability") is None:
        span.set_attribute("forecast.context.outcome", "invalid_response")
        _context_packages.add(1, {
            "market.platform": platform,
            "rules_present": str(bool(resolution_criteria)).lower(),
            "outcome": "invalid_response",
        })
        return None
    evidence_count = len(res.get("evidence_sources") or [])
    context_complete = bool(resolution_criteria) and evidence_count > 0
    context_outcome = "complete" if context_complete else "incomplete"
    span.set_attributes({
        "forecast.context.evidence_articles": evidence_count,
        "forecast.context.complete": context_complete,
        "forecast.context.outcome": context_outcome,
    })
    _context_packages.add(1, {
        "market.platform": platform,
        "rules_present": str(bool(resolution_criteria)).lower(),
        "outcome": context_outcome,
    })
    return {
        "model_probability": analysis["model_probability"],
        "market_probability": analysis.get("market_probability"),
        "evidence_count": evidence_count,
        "venue_news_count": len(venue_articles),
        "rules_present": bool(resolution_criteria),
        "context_complete": context_complete,
        "rationale": res.get("model_rationale") or res.get("rationale") or "",
    }


PRICE_ONLY = "--price-only" in sys.argv
MARK_TO_MARKET_ONLY = "--mtm-only" in sys.argv
RESOLVED_ONLY = "--resolved-only" in sys.argv
SNAPSHOT_ONLY = "--snapshot-only" in sys.argv
_MODE_FLAGS = [PRICE_ONLY, MARK_TO_MARKET_ONLY, RESOLVED_ONLY, SNAPSHOT_ONLY]
if sum(1 for flag in _MODE_FLAGS if flag) > 1:
    raise RuntimeError(
        "Use only one of --price-only, --mtm-only, --resolved-only, or --snapshot-only"
    )

_LIVE_MARKET_KEYS = (
    "source",
    "generated_at",
    "model",
    "primary_model",
    "n_markets_open",
    "n_markets_tracked",
    "edge_board",
    "discrepancy_monitor",
    "arbitrage_signals",
    "mark_to_market_account",
    "mark_to_market_by_model",
    "half_kelly_20pct_by_model",
    "growth_1pct_by_model",
    "growth_2pct_by_model",
    "mark_to_market_cycle_minutes",
)
_RESOLVED_PUBLIC_OMIT_KEYS = {
    "edge_board",
    "discrepancy_monitor",
    "arbitrage_signals",
    "mark_to_market_account",
    "mark_to_market_by_model",
    "half_kelly_20pct_by_model",
    "growth_1pct_by_model",
    "growth_2pct_by_model",
    "mark_to_market_cycle_minutes",
    "n_markets_open",
    "n_markets_tracked",
}


def _mark_to_market_payload(aggregate: dict) -> dict:
    payload = {
        key: aggregate.get(key)
        for key in _LIVE_MARKET_KEYS
        if key in aggregate
    }
    payload["source"] = "mark_to_market_live"
    payload["generated_at"] = aggregate.get("generated_at")
    return payload


def _resolved_public_payload(aggregate: dict) -> dict:
    return {
        key: value
        for key, value in aggregate.items()
        if key not in _RESOLVED_PUBLIC_OMIT_KEYS
    }


def _model_progress(store: DuckDBStore, model: str) -> dict:
    row = store._con.execute(
        """
        SELECT
            COUNT(*) AS snapshots,
            COUNT(*) FILTER (WHERE resolved = true) AS resolved_snapshots,
            MAX(snapshot_ts) AS latest_snapshot_ts,
            MAX(resolved_ts) FILTER (WHERE resolved = true) AS latest_resolved_ts
        FROM forecast_snapshot
        WHERE model = ?
        """,
        [model],
    ).fetchone()
    return {
        "snapshots": int(row[0] or 0),
        "resolved_snapshots": int(row[1] or 0),
        "latest_snapshot_ts": row[2],
        "latest_resolved_ts": row[3],
    }


def _evaluate_ledger(store: DuckDBStore, *, model: str) -> dict:
    rows = ForecastLedger(store).resolved_forecasts()
    snapshot_mirror = [
        ResolvedForecast.from_ledger(row)
        for row in rows
        if row.get("model") == model
    ]
    prospective_audit = [
        ResolvedForecast.from_ledger(row)
        for row in rows
        if row.get("model") == model and row.get("ledger_audit_grade")
    ]

    return build_evaluation_artifact(
        model=model,
        snapshot_mirror=snapshot_mirror,
        prospective_audit=prospective_audit,
        policy=EVALUATION_POLICY,
    )


async def _record_snapshots_with_retries(
    store: DuckDBStore,
    *,
    seeds: list[tuple[str, str]],
) -> int:
    """Retry the forecast pass when every /predict call fails transiently.

    A short upstream outage should not turn the whole workflow red if a second
    pass a minute later would succeed. We still fail the job after exhausting
    the pass-level retries so prolonged outages remain visible.
    """
    with _context_tracer.start_as_current_span("track_record.forecast_shard") as span:
        span.set_attributes({
            "forecast.shard.count": TRACK_MODEL_SHARD_COUNT,
            "forecast.shard.index": TRACK_MODEL_SHARD_INDEX,
            "forecast.models.count": len(TRACK_MODELS),
            "forecast.models": ",".join(TRACK_MODELS),
            "forecast.seeds.count": len(seeds),
            "forecast.target_shard.count": TRACK_TARGET_SHARD_COUNT,
            "forecast.target_shard.index": TRACK_TARGET_SHARD_INDEX,
            "forecast.targets.max": TRACK_FORECAST_MAX_TARGETS or 0,
        })
        recorded_total = 0
        try:
            for attempt in range(1, _SNAPSHOT_PASS_RETRIES + 1):
                _reset_predict_circuit()
                attempts_before = _predict_stats["attempts"]
                successes_before = _predict_stats["successes"]
                recorded_total += await trl.record_snapshots(
                    store, market_data, forecast_fn,
                    models=TRACK_MODELS, default_model=MODEL, per_venue=PER_VENUE,
                    seed_idents=seeds, price_drift_threshold=PRICE_DRIFT_THRESHOLD,
                    reforecast_each_tick=REFORECAST_EACH_TICK,
                    short_horizon_reforecast_lead_days=SHORT_HORIZON_REFORECAST_LEAD_DAYS,
                    short_horizon_slot_hours=SHORT_HORIZON_SLOT_HOURS,
                    expiry_reforecast_lead_days=EXPIRY_REFORECAST_LEAD_DAYS,
                    expiry_slot_hours=EXPIRY_SLOT_HOURS,
                    snapshot_slot_minutes=SNAPSHOT_SLOT_MINUTES,
                    concurrency=PREDICT_CONCURRENCY,
                    convergence_per_venue=CONVERGENCE_PER_VENUE,
                    target_shard_count=TRACK_TARGET_SHARD_COUNT,
                    target_shard_index=TRACK_TARGET_SHARD_INDEX,
                    max_targets=TRACK_FORECAST_MAX_TARGETS)
                pass_attempts = _predict_stats["attempts"] - attempts_before
                pass_successes = _predict_stats["successes"] - successes_before
                if pass_successes > 0 or pass_attempts == 0:
                    span.set_attributes({
                        "outcome": "success",
                        "forecast.snapshots.recorded": recorded_total,
                        "predict.calls.attempted": _predict_stats["attempts"],
                        "predict.calls.succeeded": _predict_stats["successes"],
                    })
                    _forecast_shard_runs.add(1, {
                        "outcome": "success",
                        "forecast.shard.count": str(TRACK_MODEL_SHARD_COUNT),
                        "forecast.shard.index": str(TRACK_MODEL_SHARD_INDEX),
                    })
                    return recorded_total
                if attempt < _SNAPSHOT_PASS_RETRIES:
                    delay_s = _SNAPSHOT_PASS_RETRY_SLEEP_S * (2 ** (attempt - 1))
                    print(
                        f"track-record forecast warning: all /predict calls failed on "
                        f"pass {attempt}/{_SNAPSHOT_PASS_RETRIES}; retrying in "
                        f"{delay_s:.0f}s.",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(delay_s)
            span.set_attributes({
                "outcome": "exhausted",
                "forecast.snapshots.recorded": recorded_total,
                "predict.calls.attempted": _predict_stats["attempts"],
                "predict.calls.succeeded": _predict_stats["successes"],
            })
            _forecast_shard_runs.add(1, {
                "outcome": "exhausted",
                "forecast.shard.count": str(TRACK_MODEL_SHARD_COUNT),
                "forecast.shard.index": str(TRACK_MODEL_SHARD_INDEX),
            })
            return recorded_total
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            _forecast_shard_runs.add(1, {
                "outcome": "failure",
                "forecast.shard.count": str(TRACK_MODEL_SHARD_COUNT),
                "forecast.shard.index": str(TRACK_MODEL_SHARD_INDEX),
            })
            raise


async def main() -> int:
    print(f"track-record predict mode: {_predict_mode()}")
    run_snapshots = not (PRICE_ONLY or MARK_TO_MARKET_ONLY or RESOLVED_ONLY)
    run_prices = not (RESOLVED_ONLY or SNAPSHOT_ONLY)
    run_resolutions = not (MARK_TO_MARKET_ONLY or SNAPSHOT_ONLY)
    run_ledger = not MARK_TO_MARKET_ONLY
    run_evaluation = not (MARK_TO_MARKET_ONLY or SNAPSHOT_ONLY)
    write_public = not (MARK_TO_MARKET_ONLY or SNAPSHOT_ONLY)
    write_mtm = MARK_TO_MARKET_ONLY or not (RESOLVED_ONLY or SNAPSHOT_ONLY)

    if run_snapshots and _predict_mode() == "local" and not _local_predict_credentials_available():
        print(
            "track-record forecast failed: TRACK_RECORD_PREDICT_MODE=local requires "
            "SCADS_AI_API_KEY as a GitHub Actions secret before model forecasts can run.",
            file=sys.stderr,
        )
        return 1
    store = DuckDBStore(STORE_PATH)
    primary_model = MODEL
    before_primary = _model_progress(store, primary_model)

    newly_resolved = (
        trl.resolve_open_snapshots(store, market_data)
        if run_resolutions
        else 0
    )
    price_points = trl.record_price_points(store, market_data) if run_prices else 0
    convergence_written = (
        trl.record_convergence_trades(store)
        if run_resolutions
        else 0
    )

    recorded = 0
    backfilled = 0
    if run_snapshots:
        seeds = _get_pending_markets()
        recorded = await _record_snapshots_with_retries(store, seeds=seeds)
        if ALLOW_RESOLVED_BACKFILL:
            backfilled = await trl.backfill_missing_model_snapshots(
                store, forecast_fn,
                models=TRACK_MODELS, default_model=MODEL,
                concurrency=PREDICT_CONCURRENCY)
        _mark_enrolled([f"{p}:{i}" for p, i in seeds])

    if run_ledger:
        ledger_summary = sync_snapshot_ledger(store)
    else:
        ledger_summary = {
            "snapshots_scanned": 0,
            "forecast_events_appended": 0,
            "resolution_events_appended": 0,
        }
    ledger_evaluation = (
        _evaluate_ledger(store, model=primary_model)
        if run_evaluation
        else {}
    )
    ledger_evaluation_summary = (
        compact_evaluation_summary(ledger_evaluation)
        if ledger_evaluation
        else {}
    )
    agg = {}
    if write_public or write_mtm:
        agg = trl.aggregate(
            store,
            model=primary_model,
            variant=VARIANT,
            temperature=TEMPERATURE,
            cycle_minutes=CYCLE_INTERVAL_MINUTES,
            # Forecasting can be sharded, but the public board must retain the
            # complete roster across every shard and tick.
            tracked_models=ALL_TRACK_MODELS,
        )
    after_primary = _model_progress(store, primary_model)

    if write_public:
        PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        public_payload = (
            _resolved_public_payload(agg)
            if RESOLVED_ONLY
            else agg
        )
        PUBLIC_PATH.write_text(
            json.dumps(public_payload, indent=2, sort_keys=True) + "\n"
        )
    if run_evaluation:
        EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVALUATION_PATH.write_text(
            json.dumps(ledger_evaluation, indent=2, sort_keys=True) + "\n"
        )
    if write_mtm:
        MARK_TO_MARKET_PATH.parent.mkdir(parents=True, exist_ok=True)
        MARK_TO_MARKET_PATH.write_text(
            json.dumps(_mark_to_market_payload(agg), indent=2, sort_keys=True) + "\n"
        )

    summary = {
        "snapshots_recorded": recorded,
        "snapshots_backfilled": backfilled,
        "price_points_recorded": price_points,
        "snapshots_resolved": newly_resolved,
        "convergence_trades_written": convergence_written,
        "ledger_snapshots_scanned": ledger_summary["snapshots_scanned"],
        "ledger_forecast_events_appended": ledger_summary["forecast_events_appended"],
        "ledger_resolution_events_appended": ledger_summary["resolution_events_appended"],
        "ledger_evaluation": ledger_evaluation_summary,
        "n_markets_resolved": agg.get("n_markets_resolved"),
        "n_markets_open": agg.get("n_markets_open"),
        "n_snapshots_resolved": agg.get("n_snapshots_resolved"),
        "predict_attempts": _predict_stats["attempts"],
        "predict_mode": _predict_mode(),
        "predict_successes": _predict_stats["successes"],
        "predict_failures": _predict_stats["failures"],
        "predict_retry_sleeps": _predict_stats["retry_sleeps"],
        "predict_http_401": _predict_stats["http_401"],
        "predict_circuit_opened": _predict_stats["circuit_opened"],
        "predict_circuit_skipped": _predict_stats["circuit_skipped"],
        "predict_failures_by_model": {
            key.removeprefix("failure_model:"): value
            for key, value in sorted(_predict_stats.items())
            if key.startswith("failure_model:")
        },
        "predict_circuit_open_models": sorted(_predict_circuit_open_models),
        "predict_model_mismatches": _predict_stats["model_mismatches"],
        "track_models": TRACK_MODELS,
        "track_model_shard_count": TRACK_MODEL_SHARD_COUNT,
        "track_model_shard_index": TRACK_MODEL_SHARD_INDEX,
        "track_model_shard_always": TRACK_MODEL_SHARD_ALWAYS,
        "track_target_shard_count": TRACK_TARGET_SHARD_COUNT,
        "track_target_shard_index": TRACK_TARGET_SHARD_INDEX,
        "track_forecast_max_targets": TRACK_FORECAST_MAX_TARGETS,
        "mode": (
            "snapshot-only" if SNAPSHOT_ONLY else
            "mtm-only" if MARK_TO_MARKET_ONLY else
            "resolved-only" if RESOLVED_ONLY else
            "price-only" if PRICE_ONLY else
            "full"
        ),
        "primary_model": primary_model,
        "primary_snapshots_before": before_primary["snapshots"],
        "primary_snapshots_after": after_primary["snapshots"],
        "primary_latest_snapshot_before": before_primary["latest_snapshot_ts"],
        "primary_latest_snapshot_after": after_primary["latest_snapshot_ts"],
        "primary_resolved_before": before_primary["resolved_snapshots"],
        "primary_resolved_after": after_primary["resolved_snapshots"],
    }
    print(json.dumps(summary))
    if run_snapshots:
        if _predict_stats["http_401"]:
            print(
                "track-record forecast failed: predict returned HTTP 401. "
                "Remote mode needs the Cloud Run auth headers; local mode needs "
                "SCADS_AI_API_KEY configured as a GitHub Actions secret.",
                file=sys.stderr,
            )
            return 1
        if _predict_stats["attempts"] and not _predict_stats["successes"]:
            print(
                "track-record forecast failed: all /predict calls failed.",
                file=sys.stderr,
            )
            return 1
        primary_progressed = (
            after_primary["snapshots"] > before_primary["snapshots"]
            or after_primary["resolved_snapshots"] > before_primary["resolved_snapshots"]
            or after_primary["latest_snapshot_ts"] != before_primary["latest_snapshot_ts"]
            or after_primary["latest_resolved_ts"] != before_primary["latest_resolved_ts"]
        )
        if _predict_stats["successes"] and not primary_progressed:
            print(
                f"track-record forecast warning: /predict succeeded but primary "
                f"model {primary_model!r} did not write or backfill any snapshots. "
                "This can happen when successful forecasts are intentionally "
                "skipped, for example because they returned no fresh evidence.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    init_observability()
    raise SystemExit(asyncio.run(main()))
