"""Live, point-in-time track record that grows each day.

Unlike the retrospective backtest in ``scripts/build_track_record.py`` (which
scores the model on a fixed historical Metaculus set), this records a forecast
on a *live* market the first time we see it open — capturing only information
available at that moment — then scores it once the market resolves. That makes
it genuinely out-of-sample, and because each forecast is paired with the market
price at forecast time, we can measure **skill vs the market** (Brier of model
minus Brier of market): positive means the model beat the market.

Storage is Cloud Datastore (persistent + backed up). Functions take their
dependencies (datastore client, ``market_data`` module, an async forecast
callable) as arguments so they can be unit-tested with fakes.

Datastore kinds:
- ``LiveForecast`` keyed ``"{platform}:{ident}"`` — one forecast per market, ever.
- ``TrackRecordLive`` singleton ``"global"`` — the precomputed public aggregate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

LIVE_KIND = "LiveForecast"
AGG_KIND = "TrackRecordLive"
AGG_ID = "global"

# Async callable: (quote_dict, evidence_top_k) -> (model_prob, market_prob) or None.
ForecastFn = Callable[[Dict[str, Any], int], Awaitable[Optional[tuple]]]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return _now().strftime("%Y-%m-%d")


def ident_from_url(platform: str, url: str) -> str:
    """Stable per-market identifier parsed from its venue URL."""
    url = (url or "").rstrip("/")
    p = (platform or "").lower()
    if "/market/" in url:
        return url.split("/market/")[-1]
    if "/markets/" in url:
        return url.split("/markets/")[-1]
    # Fall back to the whole URL so distinct markets never collide.
    return url or (p + ":unknown")


def brier(prob: float, outcome: int) -> float:
    return (float(prob) - float(outcome)) ** 2


async def record_new_forecasts(
    client,
    market_data,
    forecast_fn: ForecastFn,
    *,
    per_venue: int = 3,
    evidence_top_k: int = 3,
) -> int:
    """Forecast newly-seen open markets (one forecast per market, ever)."""
    from google.cloud import datastore as _ds

    candidates: List[Dict[str, Any]] = []
    for lister in (market_data.list_polymarket, market_data.list_kalshi):
        try:
            candidates.extend(lister(limit=per_venue)[:per_venue])
        except market_data.MarketDataError:
            continue

    recorded = 0
    for q in candidates:
        ident = ident_from_url(q.get("platform", ""), q.get("market_url", ""))
        market_prob = q.get("probability")
        if not ident or market_prob is None:
            continue
        key = client.key(LIVE_KIND, f"{q.get('platform')}:{ident}")
        if client.get(key) is not None:
            continue  # already tracked this market — never double-record
        try:
            scored = await forecast_fn(q, evidence_top_k)
        except Exception:
            scored = None
        if not scored or scored[0] is None:
            continue
        model_prob, mkt_prob = scored
        mkt_prob = mkt_prob if mkt_prob is not None else market_prob
        entity = _ds.Entity(key, exclude_from_indexes=("question", "market_url"))
        entity.update(
            platform=q.get("platform"),
            question=q.get("question"),
            market_url=q.get("market_url"),
            ident=ident,
            model_probability=float(model_prob),
            market_probability=float(mkt_prob),
            forecast_ts=_now(),
            forecast_date=_today(),
            resolved=False,
            outcome=None,
        )
        client.put(entity)
        recorded += 1
    return recorded


def resolve_open_forecasts(client, market_data) -> int:
    """Score every still-open forecast whose market has now resolved."""
    query = client.query(kind=LIVE_KIND)
    query.add_filter("resolved", "=", False)
    resolved_count = 0
    for entity in query.fetch():
        platform = (entity.get("platform") or "").lower()
        ident = entity.get("ident") or ""
        try:
            if "poly" in platform:
                outcome = market_data.resolve_polymarket(ident)
            elif "kalshi" in platform:
                outcome = market_data.resolve_kalshi(ident)
            else:
                outcome = None
        except market_data.MarketDataError:
            outcome = None
        if outcome is None:
            continue
        model_prob = float(entity.get("model_probability") or 0.0)
        market_prob = float(entity.get("market_probability") or 0.0)
        entity["resolved"] = True
        entity["outcome"] = int(outcome)
        entity["resolved_ts"] = _now()
        entity["model_brier"] = brier(model_prob, outcome)
        entity["market_brier"] = brier(market_prob, outcome)
        entity["model_correct"] = (model_prob >= 0.5) == (outcome == 1)
        client.put(entity)
        resolved_count += 1
    return resolved_count


def _calibration(rows: List[Dict[str, Any]], bins: int = 10) -> List[Dict[str, Any]]:
    buckets: List[Dict[str, Any]] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows
               if r["model_probability"] >= lo
               and (r["model_probability"] < hi or (b == bins - 1 and r["model_probability"] <= hi))]
        mid = round((lo + hi) / 2, 2)
        if not sel:
            buckets.append({"bucket": mid, "n": 0, "predicted": mid, "actual": None})
            continue
        pred = sum(r["model_probability"] for r in sel) / len(sel)
        act = sum(r["outcome"] for r in sel) / len(sel)
        buckets.append({"bucket": mid, "n": len(sel),
                        "predicted": round(pred, 3), "actual": round(act, 3)})
    return buckets


def aggregate(client, *, model: str, variant: str, temperature: float,
              log_limit: int = 50) -> Dict[str, Any]:
    """Recompute the public aggregate from resolved forecasts and persist it."""
    from google.cloud import datastore as _ds

    resolved: List[Dict[str, Any]] = []
    n_open = 0
    for e in client.query(kind=LIVE_KIND).fetch():
        if e.get("resolved") and e.get("outcome") is not None:
            resolved.append(dict(e))
        elif not e.get("resolved"):
            n_open += 1

    payload: Dict[str, Any] = {
        "source": "live",
        "generated_at": _now().isoformat(),
        "model": model,
        "variant": variant,
        "temperature": temperature,
        "methodology": ("Point-in-time forecasts on live Polymarket/Kalshi markets, "
                        "recorded when first seen open and scored at resolution. "
                        "Skill vs market = market Brier − model Brier (positive = model beats the market price)."),
        "n_resolved": len(resolved),
        "n_open": n_open,
    }

    if not resolved:
        payload.update({
            "accuracy": None, "brier_score": None, "market_brier_score": None,
            "skill_vs_market": None, "ece": None, "calibration": [],
            "period_start": None, "period_end": None,
            "log_sample_size": 0, "log": [],
        })
    else:
        n = len(resolved)
        model_brier = sum(r["model_brier"] for r in resolved) / n
        market_brier = sum(r["market_brier"] for r in resolved) / n
        accuracy = sum(1 for r in resolved if r["model_correct"]) / n
        calibration = _calibration(resolved)
        ece_num = sum(abs(b["predicted"] - b["actual"]) * b["n"]
                      for b in calibration if b["actual"] is not None)
        ece = ece_num / n if n else None
        dates = sorted(r["forecast_date"] for r in resolved if r.get("forecast_date"))
        log = sorted(resolved, key=lambda r: r.get("resolved_ts") or _now(), reverse=True)[:log_limit]
        payload.update({
            "accuracy": round(accuracy, 4),
            "brier_score": round(model_brier, 4),
            "market_brier_score": round(market_brier, 4),
            "skill_vs_market": round(market_brier - model_brier, 4),
            "ece": round(ece, 4) if ece is not None else None,
            "calibration": calibration,
            "period_start": dates[0] if dates else None,
            "period_end": dates[-1] if dates else None,
            "log_sample_size": len(log),
            "log": [{
                "question": r.get("question"),
                "platform": r.get("platform"),
                "market_url": r.get("market_url"),
                "forecast_date": r.get("forecast_date"),
                "model_probability": round(float(r["model_probability"]), 3),
                "market_probability": round(float(r["market_probability"]), 3),
                "outcome": int(r["outcome"]),
                "model_correct": bool(r["model_correct"]),
            } for r in log],
        })

    entity = _ds.Entity(client.key(AGG_KIND, AGG_ID),
                        exclude_from_indexes=("payload",))
    entity["payload"] = payload
    entity["generated_at"] = _now()
    client.put(entity)
    return payload


def read_aggregate(client) -> Optional[Dict[str, Any]]:
    """Return the persisted public aggregate, or None if no tick has run."""
    entity = client.get(client.key(AGG_KIND, AGG_ID))
    return dict(entity["payload"]) if entity and entity.get("payload") else None
