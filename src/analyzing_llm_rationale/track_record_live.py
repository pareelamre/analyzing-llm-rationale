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

Storage is a file-backed, Datastore-compatible store (``trackrec_store``): the
daily tick runs in a GitHub Action and commits the JSON store + public aggregate
back to the repo (git-scraping), so no batch work runs on Cloud Run. Functions
take their dependencies (store client, ``market_data`` module, an async forecast
callable) as arguments so they can be unit-tested with fakes.

Store kinds:
- ``ForecastSnapshot`` keyed ``"{platform}:{ident}:{date}"`` — one snapshot per
  market per UTC day.
- ``TrackRecordLive`` singleton ``"global"`` — the precomputed public aggregate.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from analyzing_llm_rationale import trackrec_store as _ds
from analyzing_llm_rationale.entity_tagger import tag_question

SNAPSHOT_KIND = "ForecastSnapshot"
PRICE_KIND = "MarketPricePoint"
AGG_KIND = "TrackRecordLive"
AGG_ID = "global"

# (quote_dict, evidence_top_k, model_label) -> {"model_probability",
# "market_probability", "evidence_count", ...} | None
ForecastFn = Callable[[Dict[str, Any], int, Optional[str]], Awaitable[Optional[Dict[str, Any]]]]

# Re-forecast a market mid-day when the live price has moved more than this many
# probability points since the snapshot was taken. Keeps the edge board accurate
# after large price swings without re-forecasting every stable market every hour.
PRICE_DRIFT_THRESHOLD = 0.05



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


OPEN_SNAPSHOT_MAX_AGE_DAYS = 2


def _drop_stale_open(rows: List[Dict[str, Any]],
                     max_age_days: int = OPEN_SNAPSHOT_MAX_AGE_DAYS) -> List[Dict[str, Any]]:
    """Keep only open snapshots (re)forecast within ``max_age_days`` of the most
    recent open snapshot. Orphaned readings — e.g. left behind when a venue
    changes a market's ident/ticker — go stale and must not surface as the current
    forecast. Self-relative (to the newest snapshot) so it's clock-independent."""
    import datetime as _dt

    def _date(r: Dict[str, Any]) -> Optional["_dt.date"]:
        try:
            return _dt.date.fromisoformat(str(r.get("snapshot_date"))[:10])
        except Exception:
            return None

    dates = [d for d in (_date(r) for r in rows) if d is not None]
    if not dates:
        return rows
    cutoff = max(dates) - _dt.timedelta(days=max_age_days)
    return [r for r in rows if (_date(r) is not None and _date(r) >= cutoff)]


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


def _get_price_history(client, ident: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Return recent price points for a market (newest first), fail-open to []."""
    try:
        if hasattr(client, "_con"):  # DuckDBStore
            rows = client._con.execute(
                "SELECT ts, market_probability FROM market_price_point "
                "WHERE ident = ? ORDER BY ts DESC LIMIT ?",
                [ident, limit],
            ).fetchall()
            return [{"ts": r[0], "probability": r[1]} for r in rows]
    except Exception:
        pass
    return []


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
    models: Optional[List[str]] = None,
    default_model: str = "gpt-oss-120b",
    per_venue: int = 3,
    evidence_top_k: int = 5,
    min_discovery_lead_days: float = 2.0,
    max_discovery_lead_days: float = 365.0,
    seed_idents: Optional[List[Tuple[str, str]]] = None,
    price_drift_threshold: float = PRICE_DRIFT_THRESHOLD,
    reforecast_each_tick: bool = False,
) -> int:
    """Take today's forecast snapshot for every tracked-still-open market, plus
    agent-enrolled ``seed_idents`` and newly-discovered markets, capturing the live
    price + a fresh forecast.

    With multiple ``models`` (labels passed to ``forecast_fn``), each market is
    forecast once per model per day — the per-model snapshots back the
    paper-trading comparison. One snapshot per (market, model, day).

    ``price_drift_threshold``: if the live market price has moved more than this
    many probability points since today's snapshot was taken, discard it and
    re-forecast so the edge board reflects the new information rather than a
    stale model opinion paired with a current price."""
    model_list = list(models) if models else [default_model]
    today = _today()

    # 1) Markets we're already tracking that are still open → re-fetch live quote.
    targets: List[Dict[str, Any]] = []
    for meta in _open_idents(client).values():
        quote = _fetch_current_quote(market_data, meta["platform"] or "", meta["ident"] or "")
        if quote and quote.get("probability") is not None:
            targets.append(quote)

    # 1.5) Agent-enrolled seeds (explicit agent forecasts via the evolution-loop
    #      bridge). Added before discovery so user-driven markets are included,
    #      and tracked even if short-dated — an agent explicitly asked.
    seen = {(q.get("platform"), ident_from_url(q.get("platform", ""), q.get("market_url", "")))
            for q in targets}
    for plat, ident in (seed_idents or []):
        if not ident:
            continue
        quote = _fetch_current_quote(market_data, plat or "", ident or "")
        if not quote or quote.get("probability") is None:
            continue
        key = (quote.get("platform"), ident_from_url(quote.get("platform", ""), quote.get("market_url", "")))
        if key in seen:
            continue
        targets.append(quote)
        seen.add(key)

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
        tags = tag_question(q.get("question") or "", q.get("category"))
        # Sports markets: LLM has no real-time edge, but track them for the
        # crowd-follow benchmark. Discovered only when crowd-follow is in the
        # model list; LLM models are skipped for sports in the snapshot loop below.
        if tags.get("domain") == "sports" and "crowd-follow" not in model_list:
            continue
        targets.append(q)
        known.add((q.get("platform"), ident))

    recorded = 0
    for quote in targets:
        ident = ident_from_url(quote.get("platform", ""), quote.get("market_url", ""))
        if not ident:
            continue
        market_prob = quote.get("probability")
        lead = _lead_time_days(quote.get("close_time"))
        quote_tags = tag_question(quote.get("question") or "", quote.get("category"))
        is_sports = quote_tags.get("domain") == "sports"
        for model in model_list:
            # LLM models have no real-time edge on sports — only crowd-follow runs there.
            if is_sports and model != "crowd-follow":
                continue
            key = client.key(SNAPSHOT_KIND, f"{quote.get('platform')}:{ident}:{model}:{today}")
            existing = client.get(key)
            if existing is not None and not reforecast_each_tick:
                # Skip unless the live price has drifted significantly since the snapshot.
                # A large move signals new information the original forecast didn't see;
                # re-forecasting keeps the edge board paired: current model vs current price.
                last_market_prob = float(existing.get("market_probability") or 0.0)
                current_prob = float(market_prob) if market_prob is not None else last_market_prob
                if abs(current_prob - last_market_prob) <= price_drift_threshold:
                    continue  # price stable — today's snapshot is still good
                # Fall through: price drifted — re-forecast and overwrite today's snapshot.
            # reforecast_each_tick=True: always re-run the forecast so the edge board
            # reflects the model's *current* opinion (matches live /predict), not a
            # snapshot taken earlier today. Overwrites today's snapshot for this model.
            # crowd-follow: record the current market price as the "model"
            # prediction — no LLM call. Lets us paper-trade a pure crowd-following
            # strategy alongside LLM models for comparison (especially useful for
            # sports markets where the crowd has real-time information the model lacks).
            if model == "crowd-follow":
                if market_prob is None:
                    continue
                scored = {
                    "model_probability": float(market_prob),
                    "market_probability": float(market_prob),
                    "evidence_count": 0,
                }
            else:
                try:
                    quote_with_history = {
                        **quote,
                        "price_history": _get_price_history(client, ident),
                    }
                    scored = await forecast_fn(quote_with_history, evidence_top_k, model)
                except Exception:
                    scored = None
                if not scored or scored.get("model_probability") is None:
                    continue
                # Without evidence the model has no informational edge over the
                # current market price and can produce extreme calls for no reason
                # (observed: 15% vs 80% market on Knicks, 15% vs 100% on Project
                # Freedom — both with evidence_count=0, both wrong). Skip the write;
                # the previous snapshot stays as the current view for this market.
                if (scored.get("evidence_count") or 0) == 0 and existing is not None:
                    continue
            mkt_prob = scored.get("market_probability")
            mkt_prob = mkt_prob if mkt_prob is not None else market_prob
            tags = tag_question(quote.get("question") or "", quote.get("category"))
            entity = _ds.Entity(key, exclude_from_indexes=("question", "market_url", "close_time", "category", "entities"))
            entity.update(
                platform=quote.get("platform"),
                ident=ident,
                model=model,
                question=quote.get("question"),
                market_url=quote.get("market_url"),
                snapshot_ts=_now(),
                snapshot_date=today,
                model_probability=float(scored["model_probability"]),
                market_probability=float(mkt_prob),
                close_time=quote.get("close_time"),
                lead_time_days=lead,
                horizon=_horizon_label(lead),
                # Training features captured at forecast time (cheap now, lost if not stored).
                category=quote.get("category"),
                market_volume=quote.get("volume"),
                evidence_count=scored.get("evidence_count"),
                resolved=False,
                outcome=None,
                drift_reforecast=existing is not None,
                # Knowledge-graph seed fields.
                domain=tags["domain"],
                entities=tags["entities"],
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


def record_price_points(client, market_data) -> int:
    """Append an hourly *price* point for each tracked-still-open market.

    This is the cheap, high-frequency half of the trajectory: it re-fetches the
    live market price only (no LLM forecast, no evidence), so we track price
    movement up to the last moment without paying inference cost every hour. One
    point per market per UTC hour.
    """

    now = _now()
    hour_key = now.strftime("%Y-%m-%dT%H")
    recorded = 0
    for meta in _open_idents(client).values():
        platform = meta.get("platform") or ""
        ident = meta.get("ident") or ""
        quote = _fetch_current_quote(market_data, platform, ident)
        if not quote or quote.get("probability") is None:
            continue
        key = client.key(PRICE_KIND, f"{platform}:{ident}:{hour_key}")
        if client.get(key) is not None:
            continue  # already recorded this market this hour
        entity = _ds.Entity(key)
        entity.update(
            platform=platform,
            ident=ident,
            market_probability=float(quote["probability"]),
            ts=now,
            close_time=quote.get("close_time"),
            lead_time_days=_lead_time_days(quote.get("close_time")),
        )
        client.put(entity)
        recorded += 1
    return recorded


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


# Disagreement (edge) buckets: |model − market| at forecast time. Ordered big →
# small because the credible question is whether *bigger* disagreement actually
# resolved in the model's favour — small gaps are noise either way.
_EDGE_BUCKETS = [
    ("20pp+", 0.20, float("inf")),
    ("10-20pp", 0.10, 0.20),
    ("5-10pp", 0.05, 0.10),
    ("0-5pp", 0.0, 0.05),
]

# No minimum disagreement gate: the organizing axis is *how early* the forecast was
# made (lead time / horizon), not edge size — so every forecast counts, and the
# board/lead-lag/PnL stratify by lead time rather than filtering by |model − market|.
_EDGE_MIN = 0.0

# Trading costs deducted per paper bet so ROI is net-of-fees (the "real" ROI).
# Kalshi charges a price-dependent taker fee ≈ coeff·contracts·p·(1−p); since a
# stake of `s` buys s/p contracts, that simplifies to coeff·s·(1−p). Polymarket
# charges no trading fee. _EXTRA_FEE_RATE adds a flat per-stake cost on every
# venue (a slippage/spread assumption). All env-overridable; defaults are
# venue-accurate (Polymarket fee-free, no slippage) so ROI stays truthful.
_FEE_COEFF = {
    "kalshi": float(os.environ.get("KALSHI_FEE_COEFF", "0.07")),
    "polymarket": float(os.environ.get("POLYMARKET_FEE_COEFF", "0.0")),
}
_DEFAULT_FEE_COEFF = float(os.environ.get("DEFAULT_FEE_COEFF", "0.0"))
_EXTRA_FEE_RATE = float(os.environ.get("PAPER_EXTRA_FEE_RATE", "0.0"))


def _bet_fee(platform: Any, stake: float, p_side: float) -> float:
    """Trading cost for one paper bet: venue taker fee (price-dependent) plus a
    flat slippage assumption, both as a fraction of stake."""
    coeff = _FEE_COEFF.get(str(platform or "").lower(), _DEFAULT_FEE_COEFF)
    return coeff * stake * (1.0 - p_side) + _EXTRA_FEE_RATE * stake

# Market-volume (USD) buckets for the niche-vs-liquid skill breakdown. The edge
# thesis: thin/niche markets have the least-informed crowd, so that's where the
# model should beat it; deep markets are efficient.
_LIQUIDITY_BUCKETS = [
    ("niche (<$1k)", 0.0, 1_000.0),
    ("small ($1k–25k)", 1_000.0, 25_000.0),
    ("liquid (>$25k)", 25_000.0, float("inf")),
]


def _edge(row: Dict[str, Any]) -> float:
    """Absolute model-vs-market disagreement for a snapshot."""
    return abs(float(row.get("model_probability") or 0.0)
               - float(row.get("market_probability") or 0.0))


def _edge_label(edge: float) -> str:
    for label, lo, hi in _EDGE_BUCKETS:
        if lo <= edge < hi:
            return label
    return "20pp+" if edge >= 0.20 else "0-5pp"


def _skill_ci(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """95% CI on skill-vs-market (mean of per-snapshot market_brier − model_brier).

    ``significant`` is True only when the lower bound clears 0 — i.e. the
    disagreement beat the market by more than sampling noise. This is the gate
    that turns "we disagree" into "this disagreement has proven edge".
    """
    n = len(rows)
    if n == 0:
        return {"skill_ci_low": None, "skill_ci_high": None, "skill_significant": False}
    diffs = [float(r["market_brier"]) - float(r["model_brier"]) for r in rows]
    mean = sum(diffs) / n
    if n < 2:
        return {"skill_ci_low": None, "skill_ci_high": None, "skill_significant": False}
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    low, high = mean - 1.96 * se, mean + 1.96 * se
    return {"skill_ci_low": round(low, 4), "skill_ci_high": round(high, 4),
            "skill_significant": low > 0}


def edge_calibration(resolved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Realized skill-vs-market bucketed by disagreement size — the proof that
    (or whether) a bigger model-vs-market gap actually paid at resolution."""
    out: List[Dict[str, Any]] = []
    for label, lo, hi in _EDGE_BUCKETS:
        bucket = [r for r in resolved if lo <= _edge(r) < hi]
        stats = _bucket_stats(bucket)
        if stats:
            stats["edge_bucket"] = label
            stats.update(_skill_ci(bucket))
            out.append(stats)
    return out


def lead_lag(by_market: Dict[Tuple[str, str], List[Dict[str, Any]]],
             min_edge: float = _EDGE_MIN) -> Optional[Dict[str, Any]]:
    """Did the *market* move toward the model after a disagreement?

    For each resolved market, take the first snapshot where the model disagreed
    with the price by ≥ ``min_edge`` and measure how far the price subsequently
    travelled toward the model's value (and whether the model's side won). High
    convergence = the model led and the market followed — edge evidence that
    shows up before resolution, using the dense price trajectory.
    """
    fracs: List[float] = []
    converged = right = 0
    n = 0
    for snaps in by_market.values():
        ordered = sorted(snaps, key=lambda s: s.get("snapshot_ts") or _now())
        first = next((s for s in ordered if _edge(s) >= min_edge), None)
        if first is None:
            continue
        m0 = float(first["model_probability"])
        p0 = float(first["market_probability"])
        direction = m0 - p0
        if abs(direction) < 1e-9:
            continue
        p_final = float(ordered[-1]["market_probability"])
        outcome = int(first["outcome"])
        n += 1
        if (p_final - p0) * direction > 0:
            converged += 1
        if (outcome - p0) * direction > 0:
            right += 1
        # Fraction of the price→model gap the market closed (clamped for overshoot).
        fracs.append(max(-1.0, min(2.0, (p_final - p0) / direction)))
    if not n:
        return None
    return {
        "n_markets": n,
        "min_edge": min_edge,
        "market_converged_to_model_pct": round(converged / n, 4),
        "model_right_pct": round(right / n, 4),
        "avg_convergence_fraction": round(sum(fracs) / len(fracs), 4),
    }


def paper_pnl(resolved: List[Dict[str, Any]],
              edge_calib: Optional[List[Dict[str, Any]]] = None,
              *, min_edge: float = _EDGE_MIN, stake_cap: float = 0.25) -> Optional[Dict[str, Any]]:
    """Hypothetical paper PnL of *following the model's own call* over resolved
    snapshots — the edge is how far ahead of time the model calls the right shot.

    For each resolved snapshot, place a hypothetical bet on the model's own
    predicted side (its >50% answer — the same side ``accuracy`` scores, never
    against the model) at that day's market price; at resolution a winning $1 of
    exposure returns ``(1 − p)/p``, a loser returns −1. Because every daily
    snapshot is a bet, a model that locks the correct answer early wins on more
    days — and usually at better (less-settled) prices — so calling right *early*
    is rewarded. ``win_rate`` therefore equals the model's accuracy. Three
    sizings: flat (pure signal), edge-weighted (stake the model-vs-market gap),
    and validated-only (flat, but only in disagreement buckets whose resolved
    track record is statistically significant).

    Returns are **net of venue trading fees** (``_bet_fee``: Kalshi's
    price-dependent taker fee; Polymarket fee-free), so ``roi`` is the real,
    cost-adjusted return. **Paper only** otherwise — excludes slippage/liquidity
    (unless ``PAPER_EXTRA_FEE_RATE`` is set) and correlation across snapshots of
    the same market. A signal check, the evidence that would justify ever
    enabling the guarded live executor in ``trading.py``, not a live PnL.
    """
    sig_buckets = {b["edge_bucket"] for b in (edge_calib or []) if b.get("skill_significant")}

    def _ts(v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return v.get("__dt__")
        return v.isoformat() if hasattr(v, "isoformat") else str(v)

    # Collect all eligible bets once; encode both sizings per record so the log
    # is self-contained and future analysis doesn't need to re-run the loop.
    all_bets: List[Dict[str, Any]] = []
    for r in sorted(resolved, key=lambda x: x.get("resolved_ts") or _now()):
        edge = _edge(r)
        if edge < min_edge:
            continue
        model_p, market_p = float(r["model_probability"]), float(r["market_probability"])
        if model_p == 0.5:
            continue
        side_yes = model_p > 0.5
        p_side = market_p if side_yes else (1.0 - market_p)
        if not (0.0 < p_side < 1.0):
            continue
        win = (int(r["outcome"]) == 1) == side_yes
        s_edge = min(edge, stake_cap)
        fee_flat = _bet_fee(r.get("platform"), 1.0, p_side)
        fee_edge = _bet_fee(r.get("platform"), s_edge, p_side)
        payout = (1.0 - p_side) / p_side
        all_bets.append({
            "question": (r.get("question") or r.get("ident") or "")[:120],
            "platform": r.get("platform"),
            "ident": r.get("ident"),
            "market_url": r.get("market_url"),
            "snapshot_ts": _ts(r.get("snapshot_ts")),
            "model": r.get("model"),
            "domain": r.get("domain"),
            "model_probability": round(model_p, 4),
            "market_probability": round(market_p, 4),
            "edge": round(edge, 4),
            "side": "YES" if side_yes else "NO",
            "win": win,
            "outcome": int(r["outcome"]),
            "stake_flat": 1.0,
            "stake_edge": round(s_edge, 4),
            "in_validated": _edge_label(edge) in sig_buckets,
            "profit_flat": round(1.0 * (payout if win else -1.0) - fee_flat, 4),
            "profit_edge": round(s_edge * (payout if win else -1.0) - fee_edge, 4),
            "fee_flat": round(fee_flat, 4),
            "fee_edge": round(fee_edge, 4),
            "resolved_ts": _ts(r.get("resolved_ts")),
        })

    def _run(sizing, *, validated: bool = False) -> Optional[Dict[str, Any]]:
        staked = pnl = wins = fees = 0.0
        n = 0
        cum = 0.0
        curve: List[float] = []
        for b in all_bets:
            if validated and not b["in_validated"]:
                continue
            stake = sizing(b["edge"])
            fee = _bet_fee(b["platform"], stake, b["market_probability"] if b["side"] == "YES" else (1.0 - b["market_probability"]))
            p_side = b["market_probability"] if b["side"] == "YES" else (1.0 - b["market_probability"])
            payout = (1.0 - p_side) / p_side
            profit = stake * (payout if b["win"] else -1.0) - fee
            staked += stake
            pnl += profit
            fees += fee
            wins += 1 if b["win"] else 0
            n += 1
            cum += profit
            curve.append(round(cum, 4))
        if not n:
            return None
        return {
            "n_bets": n,
            "total_staked": round(staked, 4),
            "fees": round(fees, 4),
            "pnl": round(pnl, 4),
            "roi": round(pnl / staked, 4) if staked else None,
            "win_rate": round(wins / n, 4),
            "equity_curve": curve[-60:],
        }

    flat = _run(lambda e: 1.0)
    if flat is None:
        return None
    return {
        "min_edge": min_edge,
        "disclaimer": "Hypothetical/paper, net of venue trading fees (Kalshi price-based; "
                      "Polymarket fee-free). Excludes slippage/liquidity unless configured. "
                      "Signal check, not live PnL.",
        "flat": flat,
        "edge_weighted": _run(lambda e: min(e, stake_cap)),
        "validated_only": _run(lambda e: 1.0, validated=True),
        "bets": all_bets,
    }


def build_models_comparison(resolved: List[Dict[str, Any]], *,
                            default_model: str) -> List[Dict[str, Any]]:
    """Per-model leaderboard over resolved snapshots: accuracy, skill-vs-market,
    and hypothetical paper-trading ROI (flat + validated-only) — so gpt-oss-120b,
    Gemma, and Kimi are graded on the same markets. Ranked best-paper-edge first."""
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in resolved:
        by_model.setdefault(r.get("model") or default_model, []).append(r)
    out: List[Dict[str, Any]] = []
    for mlabel, rows in by_model.items():
        ov = _bucket_stats(rows) or {}
        pp = paper_pnl(rows, edge_calibration(rows))
        model_by_horizon = []
        for label, _lo, _hi in _HORIZONS:
            h_rows = [r for r in rows if r.get("horizon") == label]
            stats = _bucket_stats(h_rows)
            if stats:
                stats["horizon"] = label
                stats.update(_skill_ci(h_rows))
                model_by_horizon.append(stats)
        out.append({
            "model": mlabel,
            "n_snapshots_resolved": len(rows),
            "n_markets_resolved": len({(r.get("platform"), r.get("ident")) for r in rows}),
            "accuracy": ov.get("accuracy"),
            "model_brier": ov.get("model_brier"),
            "skill_vs_market": ov.get("skill_vs_market"),
            "paper_roi": ((pp or {}).get("flat") or {}).get("roi"),
            "paper_roi_validated": ((pp or {}).get("validated_only") or {}).get("roi"),
            "paper_pnl": pp,
            "by_horizon": model_by_horizon,
        })
    out.sort(key=lambda m: (m["paper_roi_validated"] if m["paper_roi_validated"] is not None else -9.0,
                            m["paper_roi"] if m["paper_roi"] is not None else -9.0,
                            m["skill_vs_market"] if m["skill_vs_market"] is not None else -9.0),
             reverse=True)
    return out


def build_edge_board(open_rows: List[Dict[str, Any]],
                     latest_price: Dict[str, float],
                     edge_calib: List[Dict[str, Any]],
                     horizon_calib: Optional[List[Dict[str, Any]]] = None,
                     *,
                     min_abs_edge: float = 0.0,
                     max_per_close_window: int = 10,
                     limit: int = 50) -> List[Dict[str, Any]]:
    """Current open markets, each annotated with the resolved track record of
    forecasts made at a similar *lead time* (``lead_track_record`` from
    ``horizon_calib``) — interlinking the live board with the by-horizon
    calibration so a disagreement is shown *with* the earned credibility of
    forecasts made this early. ``min_abs_edge`` defaults to 0: every open forecast
    is shown regardless of edge size, since the organizing axis is how early the
    forecast was made, not the gap. The edge-size bucket's record is kept too
    (``track_record``) for continuity."""
    by_edge = {b["edge_bucket"]: b for b in (edge_calib or [])}
    by_horizon = {b["horizon"]: b for b in (horizon_calib or [])}
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in open_rows:
        key = (r.get("platform"), r.get("ident"))
        cur = latest.get(key)
        if cur is None or (r.get("snapshot_ts") or _now()) > (cur.get("snapshot_ts") or _now()):
            latest[key] = r

    board: List[Dict[str, Any]] = []
    for (platform, ident), r in latest.items():
        if r.get("model_probability") is None:
            continue
        current_lead = _lead_time_days(r.get("close_time"))
        # Only show markets the venue API is still actively pricing —
        # if latest_price has no entry the market is gone from the venue.
        market_p = latest_price.get(ident)
        if market_p is None:
            continue
        if not r.get("market_url"):
            continue
        model_p, market_p = float(r["model_probability"]), float(market_p)
        signed = model_p - market_p
        if abs(signed) < min_abs_edge:
            continue
        label = _edge_label(abs(signed))
        tr = by_edge.get(label)
        # Primary link: resolved skill of forecasts made at a similar lead time —
        # how early this call is, tied to our track record at that earliness.
        lead_label = _horizon_label(current_lead)
        lead_tr = by_horizon.get(lead_label)
        # The directional trade: buy the model's side at that side's price. Buying
        # YES is the same position as fading NO (binary markets are symmetric); the
        # payout is asymmetric, though — a $1 winner returns (1 − price)/price.
        side = "YES" if signed > 0 else "NO" if signed < 0 else None
        entry = market_p if signed > 0 else (1.0 - market_p) if signed < 0 else None
        payout_odds = round((1.0 - entry) / entry, 1) if entry and 0.0 < entry < 1.0 else None
        board.append({
            "question": r.get("question"),
            "platform": platform,
            "market_url": r.get("market_url"),
            "domain": r.get("domain") or "other",
            "horizon": r.get("horizon"),
            "lead_days": round(current_lead, 1) if current_lead is not None else None,
            "model_probability": round(model_p, 3),
            "market_probability": round(market_p, 3),
            "edge": round(signed, 3),
            "abs_edge": round(abs(signed), 3),
            "stance": ("model_above_market" if signed > 0
                       else "model_below_market" if signed < 0 else "agree"),
            "side": side,
            "entry_price": round(entry, 3) if entry is not None else None,
            "payout_odds": payout_odds,
            "edge_bucket": label,
            "lead_bucket": lead_label,
            "close_time": r.get("close_time"),
            "track_record": {
                "n": tr.get("n"),
                "skill_vs_market": tr.get("skill_vs_market"),
                "skill_ci_low": tr.get("skill_ci_low"),
                "skill_significant": tr.get("skill_significant"),
            } if tr else None,
            "lead_track_record": {
                "horizon": lead_label,
                "n": lead_tr.get("n"),
                "skill_vs_market": lead_tr.get("skill_vs_market"),
                "skill_ci_low": lead_tr.get("skill_ci_low"),
                "skill_significant": lead_tr.get("skill_significant"),
            } if lead_tr else None,
        })
    board.sort(key=lambda x: x["abs_edge"], reverse=True)

    # Deduplicate correlated markets: cap entries sharing the same platform and
    # close-week (e.g. 7 World Cup team markets → keep the top max_per_close_window).
    seen_windows: Dict[Tuple[str, str], int] = {}
    deduped: List[Dict[str, Any]] = []
    for item in board:
        close_t = item.get("close_time")
        close_dt = _parse_dt(close_t)
        if close_dt is not None:
            # Bucket by platform + ISO year-week (Mon-Sun window)
            window = (item["platform"] or "", close_dt.strftime("%G-W%V"))
        else:
            window = (item["platform"] or "", str(close_t or "")[:7])
        count = seen_windows.get(window, 0)
        if count < max_per_close_window:
            seen_windows[window] = count + 1
            deduped.append(item)

    # Strip the close_time helper field before returning
    for item in deduped:
        item.pop("close_time", None)

    return deduped[:limit]


def _ece(rows: List[Dict[str, Any]], bins: int = 10) -> Optional[float]:
    """Expected calibration error: avg |confidence − accuracy| weighted by bin size."""
    n = len(rows)
    if not n:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows
               if r["model_probability"] >= lo
               and (r["model_probability"] < hi or (b == bins - 1 and r["model_probability"] <= hi))]
        if not sel:
            continue
        conf = sum(r["model_probability"] for r in sel) / len(sel)
        acc = sum(r["outcome"] for r in sel) / len(sel)
        total += abs(conf - acc) * len(sel)
    return total / n


# ── Isotonic calibration (future experiment) ──────────────────────────────────
# Kept for reference but NOT applied to live predictions — prediction markets
# work on context and a global probability→outcome map is too naive.
# To re-enable: wire _calibration_map/_calibrate_probability back into server.py.

MIN_CALIBRATION_SAMPLES = 30
CALIBRATION_ECE_THRESHOLD = 0.05
CALIBRATION_CV_FOLDS = 5


def _pav(ys: List[float], ws: List[float]) -> List[float]:
    """Pool-adjacent-violators → non-decreasing fit (per input point)."""
    vals: List[float] = []
    wts: List[float] = []
    cnts: List[int] = []
    for y, w in zip(ys, ws):
        v, ww, c = float(y), float(w), 1
        while vals and vals[-1] > v:
            pv, pw, pc = vals.pop(), wts.pop(), cnts.pop()
            v = (pv * pw + v * ww) / (pw + ww)
            ww += pw
            c += pc
        vals.append(v)
        wts.append(ww)
        cnts.append(c)
    out: List[float] = []
    for v, c in zip(vals, cnts):
        out.extend([v] * c)
    return out


def _fit_isotonic(pairs: List[Tuple[float, int]]) -> Tuple[List[float], List[float]]:
    """Fit isotonic regression (model prob → outcome); returns dedup'd breakpoints."""
    pts = sorted(pairs, key=lambda p: p[0])
    xs = [float(p[0]) for p in pts]
    fitted = _pav([float(p[1]) for p in pts], [1.0] * len(pts))
    bx: List[float] = []
    by: List[float] = []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        bx.append(xs[i])
        by.append(sum(fitted[i:j + 1]) / (j + 1 - i))
        i = j + 1
    return bx, by


def _apply_isotonic(breakpoints: Tuple[List[float], List[float]], x: float) -> float:
    xs, ys = breakpoints
    if not xs:
        return x
    if x <= xs[0]:
        return max(0.0, min(1.0, ys[0]))
    if x >= xs[-1]:
        return max(0.0, min(1.0, ys[-1]))
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            if span == 0:
                return max(0.0, min(1.0, ys[i + 1]))
            t = (x - xs[i]) / span
            return max(0.0, min(1.0, ys[i] + t * (ys[i + 1] - ys[i])))
    return max(0.0, min(1.0, ys[-1]))


def _cv_calibrated_brier(rows: List[Dict[str, Any]], folds: int) -> Optional[float]:
    """Honest (out-of-fold) calibrated Brier so we don't report in-sample optimism."""
    n = len(rows)
    folds = min(folds, n)
    if folds < 2:
        return None
    order = sorted(range(n), key=lambda i: rows[i]["model_probability"])
    fold_of = {i: k % folds for k, i in enumerate(order)}
    se, cnt = 0.0, 0
    for f in range(folds):
        train = [(rows[i]["model_probability"], rows[i]["outcome"]) for i in range(n) if fold_of[i] != f]
        test = [i for i in range(n) if fold_of[i] == f]
        if not train or not test:
            continue
        bp = _fit_isotonic(train)
        for i in test:
            cal = _apply_isotonic(bp, rows[i]["model_probability"])
            se += (cal - rows[i]["outcome"]) ** 2
            cnt += 1
    return se / cnt if cnt else None


def _calibration_report(resolved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit + evaluate a calibration map. Reports calibrated-vs-raw skill for
    monitoring purposes; result is stored in the aggregate but NOT applied."""
    n = len(resolved)
    if n < MIN_CALIBRATION_SAMPLES:
        return {"applied": False, "reason": "insufficient_data",
                "n_resolved": n, "min_required": MIN_CALIBRATION_SAMPLES}
    raw_ece = _ece(resolved)
    if raw_ece is None or raw_ece < CALIBRATION_ECE_THRESHOLD:
        return {"applied": False, "reason": "already_well_calibrated",
                "n_resolved": n, "raw_ece": round(raw_ece, 4) if raw_ece is not None else None,
                "threshold": CALIBRATION_ECE_THRESHOLD}
    raw_brier = sum(brier(r["model_probability"], r["outcome"]) for r in resolved) / n
    market_brier = sum(r["market_brier"] for r in resolved) / n
    cal_brier_cv = _cv_calibrated_brier(resolved, CALIBRATION_CV_FOLDS)
    xs, ys = _fit_isotonic([(r["model_probability"], r["outcome"]) for r in resolved])
    report = {
        "applied": True,
        "method": "isotonic",
        "n_resolved": n,
        "raw_ece": round(raw_ece, 4),
        "raw_brier": round(raw_brier, 4),
        "calibrated_brier_cv": round(cal_brier_cv, 4) if cal_brier_cv is not None else None,
        "raw_skill_vs_market": round(market_brier - raw_brier, 4),
        "breakpoints": [[round(x, 4), round(y, 4)] for x, y in zip(xs, ys)],
    }
    if cal_brier_cv is not None:
        report["calibrated_skill_vs_market"] = round(market_brier - cal_brier_cv, 4)
    return report


def aggregate(client, *, model: str, variant: str, temperature: float,
              trajectory_samples: int = 8) -> Dict[str, Any]:
    """Recompute the public aggregate (overall + by-horizon) and persist it."""

    resolved: List[Dict[str, Any]] = []
    open_rows: List[Dict[str, Any]] = []
    for e in client.query(kind=SNAPSHOT_KIND).fetch():
        if e.get("resolved") and e.get("outcome") is not None:
            resolved.append(dict(e))
        else:
            open_rows.append(dict(e))
    # Drop stale open snapshots (not re-forecast recently) so orphaned readings
    # from an old ident don't surface on the edge board as live disagreements.
    open_rows = _drop_stale_open(open_rows)
    open_idents = {(r.get("platform"), r.get("ident")) for r in open_rows}

    # Latest live price per market (for the Edge Board's current disagreement).
    latest_price: Dict[str, float] = {}
    _latest_price_ts: Dict[str, Any] = {}
    for p in client.query(kind=PRICE_KIND).fetch():
        ident = p.get("ident")
        ts = p.get("ts") or _now()
        if ident not in _latest_price_ts or ts > _latest_price_ts[ident]:
            _latest_price_ts[ident] = ts
            latest_price[ident] = float(p.get("market_probability") or 0.0)

    # The public sections describe the primary model; the comparison spans all.
    # Snapshots predating the multi-model split have no `model` field → primary.
    resolved_primary = [r for r in resolved if (r.get("model") or model) == model]
    open_primary = [r for r in open_rows if (r.get("model") or model) == model]
    open_idents = {(r.get("platform"), r.get("ident")) for r in open_primary}
    n_markets_resolved = len({(r.get("platform"), r.get("ident")) for r in resolved_primary})

    # Skill metrics use all resolved primary snapshots — sports included.
    # The crowd-follow model benchmarks against LLM on sports specifically.
    resolved_skill = resolved_primary

    # Lead-time (how-early) calibration: resolved skill bucketed by forecast
    # horizon, with significance — the axis the edge board links against.
    by_horizon = []
    for label, _lo, _hi in _HORIZONS:
        stats = _bucket_stats([r for r in resolved_primary if r.get("horizon") == label])
        if stats:
            stats["horizon"] = label
            stats.update(_skill_ci([r for r in resolved_primary if r.get("horizon") == label]))
            by_horizon.append(stats)

    # Compute edge board early so the stat can reflect its actual length.
    # Use non-sports resolved snapshots for the skill calibration so sports
    _res_skill_early = [r for r in resolved if (r.get("model") or model) == model]
    by_edge_early = edge_calibration(_res_skill_early)
    edge_board_result = build_edge_board(open_primary, latest_price, by_edge_early, by_horizon)

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
        "n_snapshots_resolved": len(resolved_primary),
        "n_markets_resolved": n_markets_resolved,
        "n_markets_open": len(edge_board_result),
        "n_markets_tracked": len(open_idents),
    }

    overall = _bucket_stats(resolved_primary)
    # by_horizon computed above (with significance) so the edge board can link to it.

    # Trajectory samples: a few resolved markets with their snapshot series.
    by_market: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in resolved_primary:
        by_market.setdefault((r.get("platform"), r.get("ident")), []).append(r)
    trajectories = []
    for snaps in sorted(by_market.values(),
                        key=lambda s: max((x.get("resolved_ts") or _now()) for x in s),
                        reverse=True)[:trajectory_samples]:
        snaps_sorted = sorted(snaps, key=lambda x: x.get("snapshot_ts") or _now())
        first = snaps_sorted[0]
        # Dense hourly price line for this market (the cheap high-frequency half).
        price_q = client.query(kind=PRICE_KIND)
        price_q.add_filter("ident", "=", first.get("ident"))
        price_points = sorted(price_q.fetch(), key=lambda p: p.get("ts") or _now())
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
            "price_points": [{
                "ts": p["ts"].isoformat() if hasattr(p.get("ts"), "isoformat") else str(p.get("ts")),
                "market": round(float(p["market_probability"]), 3),
            } for p in price_points],
        })

    # Category-level skill-vs-market: which categories Foresea actually beats the
    # market in. Significance-gated (skill_significant) so a category is only
    # claimed as an edge when its lower CI bound clears 0 — never on a couple of
    # lucky resolutions. Empty / not-yet-significant until enough markets settle.
    by_domain: List[Dict[str, Any]] = []
    domain_rows: Dict[str, List[Dict[str, Any]]] = {}
    for r in resolved_primary:
        d = r.get("domain") or "other"
        domain_rows.setdefault(d, []).append(r)
    for domain, rows in domain_rows.items():
        stats = _bucket_stats(rows)
        if stats:
            stats["domain"] = domain
            stats.update(_skill_ci(rows))
            by_domain.append(stats)
    by_domain.sort(key=lambda x: (x.get("skill_significant", False), x.get("skill_vs_market") or 0), reverse=True)

    # Liquidity-level skill: the "edge lives in thin/niche markets" thesis — the
    # crowd is smallest and least informed where volume is low, so that's where
    # the model should beat it. Bucketed by market volume (USD) at forecast time.
    by_liquidity: List[Dict[str, Any]] = []
    for label, lo, hi in _LIQUIDITY_BUCKETS:
        rows = [r for r in resolved_primary if lo <= float(r.get("market_volume") or 0.0) < hi]
        stats = _bucket_stats(rows)
        if stats:
            stats["liquidity"] = label
            stats.update(_skill_ci(rows))
            by_liquidity.append(stats)

    # Open-market entity index: unique entities across all open snapshots
    # (primary model only), sorted by frequency — seeds the knowledge graph.
    entity_freq: Dict[str, int] = {}
    domain_open: Dict[str, int] = {}
    for r in open_primary:
        for ent in (r.get("entities") or []):
            entity_freq[ent] = entity_freq.get(ent, 0) + 1
        d = r.get("domain") or "other"
        domain_open[d] = domain_open.get(d, 0) + 1
    kg_entities = sorted(entity_freq.items(), key=lambda x: -x[1])
    kg_summary = {
        "top_entities": [{"entity": e, "count": c} for e, c in kg_entities[:30]],
        "domain_distribution": dict(sorted(domain_open.items(), key=lambda x: -x[1])),
        "n_tagged": sum(1 for r in open_primary if r.get("domain")),
    }

    by_edge_skill = edge_calibration(resolved_skill)
    by_market_skill: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in resolved_skill:
        by_market_skill.setdefault((r.get("platform"), r.get("ident")), []).append(r)

    payload.update({
        "overall": overall,
        "by_horizon": by_horizon,
        "by_edge": by_edge_skill,
        "by_domain": by_domain,
        "by_liquidity": by_liquidity,
        "lead_lag": lead_lag(by_market_skill),
        "paper_pnl": paper_pnl(resolved_skill, by_edge_skill),
        "edge_board": edge_board_result,
        "models_comparison": build_models_comparison(resolved, default_model=model),
        "trajectories": trajectories,
        "calibration_model": _calibration_report(resolved_skill),
        "knowledge_graph": kg_summary,
        "n_snapshots_skill": len(resolved_skill),
    })

    entity = _ds.Entity(client.key(AGG_KIND, AGG_ID), exclude_from_indexes=("payload",))
    entity["payload"] = payload
    entity["generated_at"] = _now()
    client.put(entity)
    return payload


def read_aggregate(client) -> Optional[Dict[str, Any]]:
    entity = client.get(client.key(AGG_KIND, AGG_ID))
    return dict(entity["payload"]) if entity and entity.get("payload") else None


def format_digest(aggregate: Optional[Dict[str, Any]]) -> str:
    """A shareable, honest markdown summary of the live track record — content
    for a weekly post. Built from the aggregate so it never overstates."""
    cta = "Tracked live, scored at resolution — no cherry-picking. https://foresea.ink/track-record"
    n = (aggregate or {}).get("n_snapshots_resolved") or 0
    if not aggregate or not n:
        n_open = (aggregate or {}).get("n_markets_open") or 0
        return ("**Foresea forecast track record**\n\n"
                f"Now tracking {n_open} live market(s) point-in-time; none have resolved yet, "
                "so there are no scores to report. Every forecast is logged before resolution "
                "and graded when the market settles.\n\n" + cta)

    overall = aggregate.get("overall") or {}
    n_markets = aggregate.get("n_markets_resolved") or 0
    n_open = aggregate.get("n_markets_open") or 0
    lines = [f"**Foresea forecast track record** — {n_markets} resolved, {n_open} open"]
    acc = overall.get("accuracy")
    mb, kb = overall.get("model_brier"), overall.get("market_brier")
    skill = overall.get("skill_vs_market")
    if acc is not None:
        lines.append(f"\nHit rate: {round(acc * 100)}% across {n} forecasts.")
    if mb is not None and kb is not None:
        lines.append(f"Brier: {mb} (model) vs {kb} (market).")
    if skill is not None:
        verb = "beating the market" if skill > 0 else "trailing the market" if skill < 0 else "level with the market"
        lines.append(f"Skill vs market: {skill:+.3f} — {verb}.")

    horizons = [b for b in (aggregate.get("by_horizon") or []) if b.get("n")]
    if horizons:
        lines.append("\nBy how far ahead the call was made:")
        for b in horizons:
            s = b.get("skill_vs_market")
            s_txt = f"{s:+.3f}" if s is not None else "n/a"
            lines.append(f"- {b.get('horizon')}: skill {s_txt} (n={b.get('n')})")

    cal = aggregate.get("calibration_model") or {}
    if cal.get("applied"):
        lines.append(f"\nCalibration in progress (raw ECE {cal.get('raw_ece')}).")

    lines.append("\n" + cta)
    return "\n".join(lines)
