"""Live, point-in-time track record built from forecast *trajectories*.

For each live market we take a fresh forecast snapshot on every daily tick —
each snapshot uses only the price and news available at that moment (no
look-ahead) — and keep going until the market resolves. At resolution every
snapshot of that market is scored against the outcome.

Because each snapshot is paired with the market price *at the same instant*, we
can report **skill vs market** (market Brier − model Brier; positive = the model
beat the price) **broken out by how far ahead the forecast was made**. Skill at
long horizons is the credible signal; near-resolution snapshots naturally
converge to ~0 skill (both the model and the market are nearly certain) — that's
expected and shown honestly rather than passed off as edge.

Storage is Cloud Datastore (persistent + backed up). Functions take their
dependencies (datastore client, ``market_data`` module, an async forecast
callable) as arguments so they can be unit-tested with fakes.

Datastore kinds:
- ``ForecastSnapshot`` keyed ``"{platform}:{ident}:{date}"`` — one snapshot per
  market per UTC day.
- ``TrackRecordLive`` singleton ``"global"`` — the precomputed public aggregate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

SNAPSHOT_KIND = "ForecastSnapshot"
AGG_KIND = "TrackRecordLive"
AGG_ID = "global"

# (quote_dict, evidence_top_k) -> (model_prob, market_prob) | None
ForecastFn = Callable[[Dict[str, Any], int], Awaitable[Optional[Tuple[float, float]]]]

# Horizon buckets (days-to-resolution at the moment the forecast was made).
# Ordered long → short; the long buckets carry the credible skill signal.
_HORIZONS = [
    ("30d+", 30.0, float("inf")),
    ("14-30d", 14.0, 30.0),
    ("7-14d", 7.0, 14.0),
    ("3-7d", 3.0, 7.0),
    ("1-3d", 1.0, 3.0),
    ("<1d", 0.0, 1.0),
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return _now().strftime("%Y-%m-%d")


def ident_from_url(platform: str, url: str) -> str:
    url = (url or "").rstrip("/")
    if "/market/" in url:
        return url.split("/market/")[-1]
    if "/markets/" in url:
        return url.split("/markets/")[-1]
    return url or f"{(platform or '').lower()}:unknown"


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _lead_time_days(close_time: Any, ref: Optional[datetime] = None) -> Optional[float]:
    close = _parse_dt(close_time)
    if close is None:
        return None
    ref = ref or _now()
    return (close - ref).total_seconds() / 86400.0


def _horizon_label(lead_days: Optional[float]) -> str:
    if lead_days is None:
        return "unknown"
    for label, lo, hi in _HORIZONS:
        if lo <= lead_days < hi:
            return label
    return "<1d" if lead_days < 0 else "30d+"


def brier(prob: float, outcome: int) -> float:
    return (float(prob) - float(outcome)) ** 2


def _fetch_current_quote(market_data, platform: str, ident: str) -> Optional[Dict[str, Any]]:
    """Re-fetch a tracked market's *current* quote so we capture the live price."""
    try:
        if "poly" in platform.lower():
            return market_data.fetch_polymarket(slug=ident)
        if "kalshi" in platform.lower():
            return market_data.fetch_kalshi(ident)
    except market_data.MarketDataError:
        return None
    return None


def _open_idents(client) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Distinct (platform, ident) of markets we're tracking that haven't resolved."""
    query = client.query(kind=SNAPSHOT_KIND)
    query.add_filter("resolved", "=", False)
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in query.fetch():
        key = (e.get("platform") or "", e.get("ident") or "")
        seen.setdefault(key, {"platform": e.get("platform"), "ident": e.get("ident"),
                              "question": e.get("question"), "market_url": e.get("market_url")})
    return seen


async def record_snapshots(
    client,
    market_data,
    forecast_fn: ForecastFn,
    *,
    per_venue: int = 3,
    evidence_top_k: int = 3,
    max_active: int = 20,
    min_discovery_lead_days: float = 2.0,
    max_discovery_lead_days: float = 365.0,
) -> int:
    """Take today's forecast snapshot for every tracked-still-open market, plus
    newly-discovered markets, capturing the live price + a fresh forecast."""
    from google.cloud import datastore as _ds

    today = _today()

    # 1) Markets we're already tracking that are still open → re-fetch live quote.
    targets: List[Dict[str, Any]] = []
    for meta in _open_idents(client).values():
        quote = _fetch_current_quote(market_data, meta["platform"] or "", meta["ident"] or "")
        if quote and quote.get("probability") is not None:
            targets.append(quote)

    # 2) Discover new markets within the resolution-horizon window (skip
    #    ultra-short ones — no room for a trajectory — and multi-year ones that
    #    would never resolve during the experiment).
    discovered: List[Dict[str, Any]] = []
    for lister in (market_data.list_polymarket, market_data.list_kalshi):
        try:
            discovered.extend(lister(
                limit=per_venue,
                min_close_days=min_discovery_lead_days,
                max_close_days=max_discovery_lead_days,
            )[:per_venue])
        except market_data.MarketDataError:
            continue
    known = {(q.get("platform"), ident_from_url(q.get("platform", ""), q.get("market_url", ""))) for q in targets}
    for q in discovered:
        ident = ident_from_url(q.get("platform", ""), q.get("market_url", ""))
        if (q.get("platform"), ident) in known or q.get("probability") is None:
            continue
        lead = _lead_time_days(q.get("close_time"))
        if lead is not None and not (min_discovery_lead_days <= lead <= max_discovery_lead_days):
            continue  # outside the useful resolution window
        targets.append(q)
        known.add((q.get("platform"), ident))

    recorded = 0
    for quote in targets[:max_active]:
        ident = ident_from_url(quote.get("platform", ""), quote.get("market_url", ""))
        if not ident:
            continue
        key = client.key(SNAPSHOT_KIND, f"{quote.get('platform')}:{ident}:{today}")
        if client.get(key) is not None:
            continue  # already snapshotted this market today
        market_prob = quote.get("probability")
        try:
            scored = await forecast_fn(quote, evidence_top_k)
        except Exception:
            scored = None
        if not scored or scored[0] is None:
            continue
        model_prob, mkt_prob = scored
        mkt_prob = mkt_prob if mkt_prob is not None else market_prob
        lead = _lead_time_days(quote.get("close_time"))
        entity = _ds.Entity(key, exclude_from_indexes=("question", "market_url", "close_time"))
        entity.update(
            platform=quote.get("platform"),
            ident=ident,
            question=quote.get("question"),
            market_url=quote.get("market_url"),
            snapshot_ts=_now(),
            snapshot_date=today,
            model_probability=float(model_prob),
            market_probability=float(mkt_prob),
            close_time=quote.get("close_time"),
            lead_time_days=lead,
            horizon=_horizon_label(lead),
            resolved=False,
            outcome=None,
        )
        client.put(entity)
        recorded += 1
    return recorded


def resolve_open_snapshots(client, market_data) -> int:
    """Score every still-open snapshot whose market has now resolved."""
    query = client.query(kind=SNAPSHOT_KIND)
    query.add_filter("resolved", "=", False)
    open_snaps = list(query.fetch())

    # Resolve each distinct market once.
    outcomes: Dict[Tuple[str, str], Optional[int]] = {}
    for e in open_snaps:
        platform = (e.get("platform") or "").lower()
        ident = e.get("ident") or ""
        k = (platform, ident)
        if k in outcomes:
            continue
        try:
            if "poly" in platform:
                outcomes[k] = market_data.resolve_polymarket(ident)
            elif "kalshi" in platform:
                outcomes[k] = market_data.resolve_kalshi(ident)
            else:
                outcomes[k] = None
        except market_data.MarketDataError:
            outcomes[k] = None

    scored = 0
    for e in open_snaps:
        outcome = outcomes.get(((e.get("platform") or "").lower(), e.get("ident") or ""))
        if outcome is None:
            continue
        model_prob = float(e.get("model_probability") or 0.0)
        market_prob = float(e.get("market_probability") or 0.0)
        e["resolved"] = True
        e["outcome"] = int(outcome)
        e["resolved_ts"] = _now()
        e["model_brier"] = brier(model_prob, outcome)
        e["market_brier"] = brier(market_prob, outcome)
        e["model_correct"] = (model_prob >= 0.5) == (outcome == 1)
        client.put(e)
        scored += 1
    return scored


def _bucket_stats(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    n = len(rows)
    model_b = sum(r["model_brier"] for r in rows) / n
    market_b = sum(r["market_brier"] for r in rows) / n
    acc = sum(1 for r in rows if r["model_correct"]) / n
    return {
        "n": n,
        "model_brier": round(model_b, 4),
        "market_brier": round(market_b, 4),
        "skill_vs_market": round(market_b - model_b, 4),
        "accuracy": round(acc, 4),
    }


def aggregate(client, *, model: str, variant: str, temperature: float,
              trajectory_samples: int = 8) -> Dict[str, Any]:
    """Recompute the public aggregate (overall + by-horizon) and persist it."""
    from google.cloud import datastore as _ds

    resolved: List[Dict[str, Any]] = []
    open_idents: set = set()
    for e in client.query(kind=SNAPSHOT_KIND).fetch():
        if e.get("resolved") and e.get("outcome") is not None:
            resolved.append(dict(e))
        else:
            open_idents.add((e.get("platform"), e.get("ident")))

    n_markets_resolved = len({(r.get("platform"), r.get("ident")) for r in resolved})

    payload: Dict[str, Any] = {
        "source": "live",
        "generated_at": _now().isoformat(),
        "model": model,
        "variant": variant,
        "temperature": temperature,
        "methodology": (
            "Forecast trajectories on live Polymarket/Kalshi markets: a fresh "
            "point-in-time forecast each day (latest price + latest news, no "
            "look-ahead) until the market resolves, then every snapshot is scored. "
            "Skill vs market = market Brier − model Brier, reported by forecast "
            "horizon; long-horizon skill is the meaningful signal."
        ),
        "n_snapshots_resolved": len(resolved),
        "n_markets_resolved": n_markets_resolved,
        "n_markets_open": len(open_idents),
    }

    overall = _bucket_stats(resolved)
    by_horizon = []
    for label, _lo, _hi in _HORIZONS:
        stats = _bucket_stats([r for r in resolved if r.get("horizon") == label])
        if stats:
            stats["horizon"] = label
            by_horizon.append(stats)

    # Trajectory samples: a few resolved markets with their snapshot series.
    by_market: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in resolved:
        by_market.setdefault((r.get("platform"), r.get("ident")), []).append(r)
    trajectories = []
    for snaps in sorted(by_market.values(),
                        key=lambda s: max((x.get("resolved_ts") or _now()) for x in s),
                        reverse=True)[:trajectory_samples]:
        snaps_sorted = sorted(snaps, key=lambda x: x.get("snapshot_ts") or _now())
        first = snaps_sorted[0]
        trajectories.append({
            "question": first.get("question"),
            "platform": first.get("platform"),
            "market_url": first.get("market_url"),
            "outcome": int(first["outcome"]),
            "points": [{
                "date": s.get("snapshot_date"),
                "lead_days": round(s["lead_time_days"], 1) if s.get("lead_time_days") is not None else None,
                "model": round(float(s["model_probability"]), 3),
                "market": round(float(s["market_probability"]), 3),
            } for s in snaps_sorted],
        })

    payload.update({
        "overall": overall,
        "by_horizon": by_horizon,
        "trajectories": trajectories,
    })

    entity = _ds.Entity(client.key(AGG_KIND, AGG_ID), exclude_from_indexes=("payload",))
    entity["payload"] = payload
    entity["generated_at"] = _now()
    client.put(entity)
    return payload


def read_aggregate(client) -> Optional[Dict[str, Any]]:
    entity = client.get(client.key(AGG_KIND, AGG_ID))
    return dict(entity["payload"]) if entity and entity.get("payload") else None
