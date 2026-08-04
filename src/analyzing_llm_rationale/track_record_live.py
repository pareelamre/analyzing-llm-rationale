"""Live, point-in-time track record built from forecast *trajectories*.

For each live market we take point-in-time forecast snapshots: daily for slower
markets and intraday for short-horizon markets near resolution. Each snapshot
uses only the price and news available at that moment (no look-ahead) and keeps
going until the market resolves. At resolution every snapshot of that market is
scored against the outcome.

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
- ``ForecastSnapshot`` keyed by market/model/snapshot slot. Slow markets use a
  daily slot; short-horizon markets use an intraday slot.
- ``TrackRecordLive`` singleton ``"global"`` — the precomputed public aggregate.
"""
from __future__ import annotations

import hashlib
import json as _json
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from analyzing_llm_rationale import trackrec_store as _ds
from analyzing_llm_rationale.accounting import (
    simulate_market_follow_mark_to_market_account,
    simulate_shadow_mark_to_market_account,
    simulate_validated_kelly_account,
)
from analyzing_llm_rationale.entity_tagger import tag_question

# ── DuckDB SQL helpers ────────────────────────────────────────────────────────

def _sql_con(client):
    """Return the raw DuckDB connection if the store is DuckDBStore, else None."""
    return getattr(client, "_con", None)


def _sql_rows(con, sql: str, params: Optional[list] = None) -> List[Dict[str, Any]]:
    """Execute SQL and return list of dicts.  Coerces the JSON-stored ``entities``
    field back to a Python list so callers see the same shape as the ORM path."""
    result = con.execute(sql, params or [])
    cols = [d[0] for d in result.description]
    rows = []
    for r in result.fetchall():
        row = dict(zip(cols, r))
        if isinstance(row.get("entities"), str):
            try:
                row["entities"] = _json.loads(row["entities"])
            except (ValueError, TypeError):
                row["entities"] = []
        rows.append(row)
    return rows


def _is_backfill_snapshot(row: Dict[str, Any]) -> bool:
    key = row.get("key")
    if key is None:
        ent_key = getattr(row, "key", None)
        key = getattr(ent_key, "id", None) or getattr(ent_key, "name", None)
    return str(key or "").endswith("_backfill")


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
SHORT_HORIZON_REFORECAST_LEAD_DAYS = 3.0
SHORT_HORIZON_SLOT_HOURS = 6
EXPIRY_REFORECAST_LEAD_DAYS = 3.0   # markets within this many days get hourly slots
EXPIRY_SLOT_HOURS = 1



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


def _snapshot_slot(lead_days: Optional[float],
                   *,
                   now: Optional[datetime] = None,
                   short_lead_days: float = SHORT_HORIZON_REFORECAST_LEAD_DAYS,
                   short_slot_hours: int = SHORT_HORIZON_SLOT_HOURS,
                   expiry_lead_days: float = EXPIRY_REFORECAST_LEAD_DAYS,
                   expiry_slot_hours: int = EXPIRY_SLOT_HOURS,
                   slot_minutes: Optional[int] = None) -> str:
    """Storage slot for a forecast snapshot.

    Three tiers by days-to-resolution:
      > short_lead_days        → one slot per UTC day
      ≤ short_lead_days        → one slot per short_slot_hours (default 6h)
      ≤ expiry_lead_days       → one slot per expiry_slot_hours (default 1h)
    """
    now = now or _now()
    if slot_minutes and slot_minutes > 0:
        total = now.hour * 60 + now.minute
        floored = (total // slot_minutes) * slot_minutes
        hour, minute = divmod(floored, 60)
        return now.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        ).strftime("%Y-%m-%dT%H:%M")
    if lead_days is None or lead_days > short_lead_days or short_slot_hours <= 0:
        return now.strftime("%Y-%m-%d")
    slot_h = (expiry_slot_hours if (lead_days <= expiry_lead_days and expiry_slot_hours > 0)
              else short_slot_hours)
    hour = (now.hour // slot_h) * slot_h
    return now.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H")


OPEN_SNAPSHOT_MAX_AGE_DAYS = 2


def _drop_stale_open(rows: List[Dict[str, Any]],
                     max_age_days: int = OPEN_SNAPSHOT_MAX_AGE_DAYS) -> List[Dict[str, Any]]:
    """Keep only open snapshots (re)forecast within ``max_age_days`` of the most
    recent open snapshot *for the same model*. Orphaned readings — e.g. left
    behind when a venue changes a market's ident/ticker — go stale and must not
    surface as the current forecast.

    The window is per-model (relative to each model's own newest snapshot), not
    a single global cutoff. A cheap heartbeat model (e.g. ``crowd-follow``) that
    snapshots every tick would otherwise pin the global cutoff to "today" and
    evict a less-frequently-forecast model (the primary LLM) from the board the
    moment it falls a couple of days behind. Self-relative so it's
    clock-independent."""
    import datetime as _dt

    def _date(r: Dict[str, Any]) -> Optional["_dt.date"]:
        try:
            return _dt.date.fromisoformat(str(r.get("snapshot_date"))[:10])
        except Exception:
            return None

    newest_by_model: Dict[Any, "_dt.date"] = {}
    for r in rows:
        d = _date(r)
        if d is None:
            continue
        m = r.get("model")
        if m not in newest_by_model or d > newest_by_model[m]:
            newest_by_model[m] = d
    if not newest_by_model:
        return rows
    return [
        r for r in rows
        if (_date(r) is not None and r.get("model") in newest_by_model
            and _date(r) >= newest_by_model[r.get("model")] - _dt.timedelta(days=max_age_days))
    ]


def ident_from_url(platform: str, url: str) -> str:
    url = (url or "").rstrip("/")
    if "/market/" in url:
        return url.split("/market/")[-1]
    if "/markets/" in url:
        return url.split("/markets/")[-1]
    return url or f"{(platform or '').lower()}:unknown"


def _quote_ident(quote: Dict[str, Any]) -> str:
    platform = quote.get("platform", "")
    ident = str(quote.get("ident") or "").strip()
    if ident and "kalshi" in platform.lower():
        return ident
    return ident_from_url(platform, quote.get("market_url", "")) or ident


def _target_shard_index(quote: Dict[str, Any], count: int) -> int:
    if count <= 1:
        return 0
    platform = str(quote.get("platform") or "").strip().lower()
    ident = str(_quote_ident(quote)).strip()
    digest = hashlib.sha256(f"{platform}:{ident}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % count


def _select_target_shard(
    targets: List[Dict[str, Any]],
    *,
    count: int,
    index: int,
    max_targets: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if count > 1:
        targets = [
            quote for quote in targets
            if _target_shard_index(quote, count) == index
        ]
    if max_targets is not None and max_targets > 0:
        targets = targets[:max_targets]
    return targets


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


def _iso_ts(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("__dt__")
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


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


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _get_price_history(client, ident: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Return recent market context points (newest first), fail-open to []."""
    try:
        if hasattr(client, "_con"):  # DuckDBStore
            rows = client._con.execute(
                """
                SELECT ts, market_probability, market_bid, market_ask,
                       market_volume, market_liquidity, last_trade_price
                FROM market_price_point
                WHERE ident = ? ORDER BY ts DESC LIMIT ?
                """,
                [ident, limit],
            ).fetchall()
            return [
                {
                    "ts": r[0],
                    "probability": r[1],
                    "bid": r[2],
                    "ask": r[3],
                    "volume": r[4],
                    "liquidity": r[5],
                    "last_trade_price": r[6],
                }
                for r in rows
            ]
    except Exception:
        pass
    try:
        q = client.query(kind=PRICE_KIND)
        q.add_filter("ident", "=", ident)
        points = sorted(q.fetch(), key=lambda p: p.get("ts") or _now(), reverse=True)
        return [
            {
                "ts": p.get("ts"),
                "probability": p.get("market_probability"),
                "bid": p.get("market_bid"),
                "ask": p.get("market_ask"),
                "volume": p.get("market_volume"),
                "liquidity": p.get("market_liquidity"),
                "last_trade_price": p.get("last_trade_price"),
            }
            for p in points[:limit]
        ]
    except Exception:
        pass
    return []


def _get_forecast_history(
    client,
    platform: str,
    ident: str,
    model: Optional[str] = None,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """Return prior forecast snapshots for this market/model, newest first."""
    try:
        if hasattr(client, "_con"):  # DuckDBStore
            params: List[Any] = [ident]
            platform_clause = ""
            if platform:
                platform_clause = " AND lower(platform) = ?"
                params.append(platform.lower())
            model_clause = ""
            if model:
                model_clause = " AND model = ?"
                params.append(model)
            params.append(limit)
            rows = client._con.execute(
                f"""
                SELECT snapshot_ts, snapshot_date, model, model_probability,
                       market_probability, market_volume, market_liquidity,
                       evidence_count, rationale
                FROM forecast_snapshot
                WHERE ident = ?{platform_clause}{model_clause}
                ORDER BY snapshot_ts DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                {
                    "snapshot_ts": r[0],
                    "snapshot_date": r[1],
                    "model": r[2],
                    "model_probability": r[3],
                    "market_probability": r[4],
                    "market_volume": r[5],
                    "market_liquidity": r[6],
                    "evidence_count": r[7],
                    "rationale": r[8],
                }
                for r in rows
            ]
    except Exception:
        pass
    try:
        q = client.query(kind=SNAPSHOT_KIND)
        q.add_filter("ident", "=", ident)
        snaps = [
            s for s in q.fetch()
            if (not platform or (s.get("platform") or "").lower() == platform.lower())
            and (not model or s.get("model") == model)
        ]
        snaps.sort(key=lambda s: s.get("snapshot_ts") or _now(), reverse=True)
        return [
            {
                "snapshot_ts": s.get("snapshot_ts"),
                "snapshot_date": s.get("snapshot_date"),
                "model": s.get("model"),
                "model_probability": s.get("model_probability"),
                "market_probability": s.get("market_probability"),
                "market_volume": s.get("market_volume"),
                "market_liquidity": s.get("market_liquidity"),
                "evidence_count": s.get("evidence_count"),
                "rationale": s.get("rationale"),
            }
            for s in snaps[:limit]
        ]
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


def _refresh_quote_context(market_data, quote: Dict[str, Any]) -> Dict[str, Any]:
    """Hydrate a listing result from the venue's detail endpoint.

    Listing payloads can omit long rule text or related articles. Every newly
    discovered question gets one detail fetch before it is fanned out to models,
    keeping rules/news identical across council members and avoiding one venue
    request per model.
    """
    refreshed = _fetch_current_quote(
        market_data,
        str(quote.get("platform") or ""),
        _quote_ident(quote),
    )
    if not refreshed:
        return quote
    merged = {**quote, **refreshed}
    if not refreshed.get("category") and quote.get("category"):
        merged["category"] = quote["category"]
    return merged


def record_convergence_trades(store) -> int:
    """Match resolved 7-14d snapshots with their ~3d price point and write ConvergenceTrade rows.

    For each resolved snapshot originally taken at 7-14d lead time, find the
    market price point closest to 3 days before resolution (the exit), compute
    the directional PnL, and upsert a row into convergence_trade. Skips markets
    with no price data near 3d out, and markets where model == market at entry.
    Only runs on DuckDBStore (requires raw SQL join).
    """
    con = _sql_con(store)
    if con is None:
        return 0

    rows = con.execute("""
        SELECT fs.platform, fs.ident, fs.model, fs.question, fs.market_url,
               fs.snapshot_ts, fs.lead_time_days,
               fs.model_probability, fs.market_probability,
               fs.outcome, fs.resolved_ts
        FROM forecast_snapshot fs
        LEFT JOIN convergence_trade ct
            ON ct.key = fs.platform || ':' || fs.ident || ':' || fs.model
        WHERE fs.horizon = '7-14d'
          AND fs.resolved = TRUE
          AND fs.outcome IS NOT NULL
          AND ct.key IS NULL
    """).fetchall()

    written = 0
    for row in rows:
        (platform, ident, model, question, market_url,
         snap_ts, lead_days, model_prob, market_prob,
         outcome, resolved_ts) = row

        exit_row = con.execute("""
            SELECT ts, lead_time_days, market_probability
            FROM market_price_point
            WHERE ident = ?
              AND lead_time_days BETWEEN 1.5 AND 5.0
            ORDER BY ABS(lead_time_days - 3.0)
            LIMIT 1
        """, [ident]).fetchone()

        if not exit_row:
            continue

        exit_ts, exit_lead, exit_price = exit_row
        edge = (model_prob or 0.0) - (market_prob or 0.0)
        if edge == 0:
            continue

        side = "YES" if edge > 0 else "NO"
        pnl = (exit_price - market_prob) if side == "YES" else (market_prob - exit_price)

        key = f"{platform}:{ident}:{model}"
        con.execute("""
            INSERT OR REPLACE INTO convergence_trade
            (key, platform, ident, model, question, market_url,
             entry_ts, entry_lead_days, entry_model_probability, entry_market_probability,
             exit_ts, exit_lead_days, exit_market_probability,
             outcome, resolved_ts, side, pnl_flat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [key, platform, ident, model, question, market_url,
              snap_ts, lead_days, model_prob, market_prob,
              exit_ts, exit_lead, exit_price,
              outcome, resolved_ts, side, pnl])
        written += 1

    return written


def _open_idents(client) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Distinct (platform, ident) of markets we're tracking that haven't resolved."""
    con = _sql_con(client)
    if con is not None:
        rows = con.execute("""
            SELECT platform, ident,
                   MAX(question)   AS question,
                   MAX(market_url) AS market_url
            FROM forecast_snapshot
            WHERE resolved = false
            GROUP BY platform, ident
        """).fetchall()
        return {
            (r[0] or "", r[1] or ""): {
                "platform": r[0], "ident": r[1],
                "question": r[2], "market_url": r[3],
            }
            for r in rows if r[1]
        }
    # ORM fallback (FileStore / Datastore)
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
    short_horizon_reforecast_lead_days: float = SHORT_HORIZON_REFORECAST_LEAD_DAYS,
    short_horizon_slot_hours: int = SHORT_HORIZON_SLOT_HOURS,
    expiry_reforecast_lead_days: float = EXPIRY_REFORECAST_LEAD_DAYS,
    expiry_slot_hours: int = EXPIRY_SLOT_HOURS,
    snapshot_slot_minutes: Optional[int] = None,
    concurrency: int = 4,
    convergence_per_venue: int = 0,
    target_shard_count: int = 1,
    target_shard_index: int = 0,
    max_targets: Optional[int] = None,
) -> int:
    """Take today's forecast snapshot for every tracked-still-open market, plus
    agent-enrolled ``seed_idents`` and newly-discovered markets, capturing the live
    price + a fresh forecast.

    With multiple ``models`` (labels passed to ``forecast_fn``), each market is
    forecast once per model per day — the per-model snapshots back the
    paper-trading comparison. Slow markets use one snapshot per
    (market, model, UTC day); short-horizon markets use intraday slots.

    ``price_drift_threshold``: if the live market price has moved more than this
    many probability points since today's snapshot was taken, discard it and
    re-forecast so the edge board reflects the new information rather than a
    stale model opinion paired with a current price."""
    model_list = list(models) if models else [default_model]
    tick_now = _now()
    today = tick_now.strftime("%Y-%m-%d")

    # 1) Markets we're already tracking that are still open → re-fetch live quote.
    targets: List[Dict[str, Any]] = []
    for meta in _open_idents(client).values():
        quote = _fetch_current_quote(market_data, meta["platform"] or "", meta["ident"] or "")
        if quote and quote.get("probability") is not None:
            targets.append(quote)

    # 1.5) Agent-enrolled seeds (explicit agent forecasts via the evolution-loop
    #      bridge). Added before discovery so user-driven markets are included,
    #      and tracked even if short-dated — an agent explicitly asked.
    seen = {(q.get("platform"), _quote_ident(q)) for q in targets}
    for plat, ident in (seed_idents or []):
        if not ident:
            continue
        quote = _fetch_current_quote(market_data, plat or "", ident or "")
        if not quote or quote.get("probability") is None:
            continue
        key = (quote.get("platform"), _quote_ident(quote))
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
    known = {(q.get("platform"), _quote_ident(q)) for q in targets}
    for q in discovered:
        ident = _quote_ident(q)
        if (q.get("platform"), ident) in known or q.get("probability") is None:
            continue
        lead = _lead_time_days(q.get("close_time"))
        if lead is not None and not (min_discovery_lead_days <= lead <= max_discovery_lead_days):
            continue  # outside the useful resolution window
        targets.append(_refresh_quote_context(market_data, q))
        known.add((q.get("platform"), ident))

    # 2b) Convergence-window discovery: targeted 7-14d pass with a higher limit.
    #     These markets are the data-collection target for the convergence trade study.
    if convergence_per_venue > 0:
        known = {(q.get("platform"), _quote_ident(q)) for q in targets}
        for lister in (market_data.list_polymarket, market_data.list_kalshi):
            try:
                kwargs: Dict[str, Any] = dict(
                    limit=convergence_per_venue,
                    min_close_days=7.0,
                    max_close_days=14.0,
                )
                if lister is market_data.list_kalshi:
                    kwargs["paginate"] = True
                conv_markets = lister(**kwargs)[:convergence_per_venue]
            except market_data.MarketDataError:
                continue
            for q in conv_markets:
                ident = _quote_ident(q)
                if (q.get("platform"), ident) in known or q.get("probability") is None:
                    continue
                lead = _lead_time_days(q.get("close_time"))
                if lead is None or not (7.0 <= lead <= 14.0):
                    continue
                targets.append(_refresh_quote_context(market_data, q))
                known.add((q.get("platform"), ident))

    # 2c) Short-dated discovery: all models, no lead-time floor. Captures intraday
    #     and same-day markets that the standard discovery window skips.
    if "crowd-follow" in model_list:
        intraday: List[Dict[str, Any]] = []
        for lister in (market_data.list_polymarket, market_data.list_kalshi):
            try:
                intraday.extend(lister(
                    limit=per_venue * 2,
                    min_close_days=0,
                    max_close_days=min_discovery_lead_days,
                )[:per_venue * 2])
            except market_data.MarketDataError:
                continue
        for q in intraday:
            ident = _quote_ident(q)
            if (q.get("platform"), ident) in known or q.get("probability") is None:
                continue
            lead = _lead_time_days(q.get("close_time"))
            if lead is None or lead < 0:
                continue  # already past close
            targets.append(_refresh_quote_context(market_data, q))
            known.add((q.get("platform"), ident))

    targets = _select_target_shard(
        targets,
        count=target_shard_count,
        index=target_shard_index,
        max_targets=max_targets,
    )

    import asyncio as _asyncio

    def _write_snapshot(key, quote, ident, model, scored, lead, market_prob, existing, today):
        mkt_prob = scored.get("market_probability")
        mkt_prob = mkt_prob if mkt_prob is not None else market_prob
        tags = tag_question(quote.get("question") or "", quote.get("category"))
        entity = _ds.Entity(key, exclude_from_indexes=(
            "question", "market_url", "close_time", "category", "entities",
            "description", "resolution_criteria", "rationale"))
        entity.update(
            platform=quote.get("platform"),
            ident=ident,
            model=model,
            question=quote.get("question"),
            market_url=quote.get("market_url"),
            description=quote.get("description"),
            resolution_criteria=quote.get("resolution_criteria"),
            publish_time=quote.get("created_time"),
            snapshot_ts=_now(),
            snapshot_date=today,
            model_probability=float(scored["model_probability"]),
            market_probability=float(mkt_prob),
            market_bid=quote.get("yes_bid"),
            market_ask=quote.get("yes_ask"),
            close_time=quote.get("close_time"),
            lead_time_days=lead,
            horizon=_horizon_label(lead),
            category=quote.get("category"),
            market_volume=quote.get("volume"),
            market_liquidity=quote.get("liquidity"),
            evidence_count=scored.get("evidence_count"),
            venue_news_count=scored.get("venue_news_count"),
            rules_present=scored.get("rules_present"),
            context_complete=scored.get("context_complete"),
            resolved=False,
            outcome=None,
            drift_reforecast=existing is not None,
            domain=tags["domain"],
            entities=tags["entities"],
            rationale=scored.get("rationale"),
        )
        client.put(entity)

    # Phase 1: instant crowd-follow writes + collect LLM tasks.
    recorded = 0
    llm_tasks: List[Any] = []  # list of coroutines

    for quote in targets:
        ident = _quote_ident(quote)
        if not ident:
            continue
        market_prob = quote.get("probability")
        lead = _lead_time_days(quote.get("close_time"), ref=tick_now)
        if lead is not None and lead <= 0:
            continue
        slot = _snapshot_slot(
            lead,
            now=tick_now,
            short_lead_days=short_horizon_reforecast_lead_days,
            short_slot_hours=short_horizon_slot_hours,
            expiry_lead_days=expiry_reforecast_lead_days,
            expiry_slot_hours=expiry_slot_hours,
            slot_minutes=snapshot_slot_minutes,
        )
        for model in model_list:
            key = client.key(SNAPSHOT_KIND, f"{quote.get('platform')}:{ident}:{model}:{slot}")
            existing = client.get(key)
            if existing is not None and not reforecast_each_tick:
                last_market_prob = float(existing.get("market_probability") or 0.0)
                current_prob = float(market_prob) if market_prob is not None else last_market_prob
                if abs(current_prob - last_market_prob) <= price_drift_threshold:
                    continue

            if model == "crowd-follow":
                if market_prob is None:
                    continue
                _write_snapshot(key, quote, ident, model,
                                {"model_probability": float(market_prob),
                                 "market_probability": float(market_prob),
                                 "evidence_count": 0},
                                lead, market_prob, existing, today)
                recorded += 1
            else:
                # Capture loop variables for the async closure.
                # Enrich with stored context fields (description/resolution_criteria) if
                # not already in the live quote — avoids re-fetching on re-forecasts.
                stored_ctx: Dict[str, Any] = {}
                if existing is not None:
                    for _f in ("description", "resolution_criteria", "publish_time"):
                        if not quote.get(_f) and existing.get(_f):
                            stored_ctx[_f] = existing[_f]
                quote_with_history = {**quote, **stored_ctx,
                                      "market_price_history": _get_price_history(client, ident),
                                      "forecast_history": _get_forecast_history(
                                          client, quote.get("platform") or "", ident, model)}
                llm_tasks.append((key, quote, ident, model, lead, market_prob, existing,
                                  quote_with_history))

    # Phase 2: run all LLM forecasts in parallel, bounded by the semaphore.
    sem = _asyncio.Semaphore(concurrency)

    async def _run_one(key, quote, ident, model, lead, market_prob, existing, qwh):
        async with sem:
            try:
                scored = await forecast_fn(qwh, evidence_top_k, model)
            except Exception:
                return False
        if not scored or scored.get("model_probability") is None:
            return False
        # Without evidence the model has no informational edge — skip the write
        # so the previous snapshot stays as the current view for this market.
        if (scored.get("evidence_count") or 0) == 0 and existing is not None:
            return False
        _write_snapshot(key, quote, ident, model, scored, lead, market_prob, existing, today)
        return True

    results = await _asyncio.gather(
        *[_run_one(*t) for t in llm_tasks],
        return_exceptions=True,
    )
    recorded += sum(1 for r in results if r is True)
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
        lead = _lead_time_days(quote.get("close_time"), ref=now)
        if lead is not None and lead <= 0:
            continue
        key = client.key(PRICE_KIND, f"{platform}:{ident}:{hour_key}")
        if client.get(key) is not None:
            continue  # already recorded this market this hour
        entity = _ds.Entity(key)
        entity.update(
            platform=platform,
            ident=ident,
            market_url=quote.get("market_url"),
            market_probability=float(quote["probability"]),
            market_bid=quote.get("yes_bid"),
            market_ask=quote.get("yes_ask"),
            market_volume=quote.get("volume"),
            market_liquidity=quote.get("liquidity"),
            last_trade_price=quote.get("last_trade_price"),
            ts=now,
            hour=hour_key,
            close_time=quote.get("close_time"),
            lead_time_days=lead,
        )
        client.put(entity)
        recorded += 1
    return recorded


async def backfill_missing_model_snapshots(
    client,
    forecast_fn: ForecastFn,
    *,
    models: List[str],
    default_model: str,
    evidence_top_k: int = 5,
    concurrency: int = 4,
) -> int:
    """Retroactively forecast resolved markets for diagnostic comparisons.

    For each resolved market with any reference snapshot, calls forecast_fn for
    missing tracked LLM/council models. These snapshots are keyed with a
    ``_backfill`` suffix and are excluded from the public real-time aggregate.
    Production ticks should leave this disabled so the track record only scores
    forecasts captured before resolution."""
    con = _sql_con(client)
    if con is None:
        return 0

    llm_models = [m for m in models if m != "crowd-follow"]
    if not llm_models:
        return 0

    # Reference resolved snapshots keyed by (platform, ident). Prefer the
    # primary model, then any LLM/council model, then crowd-follow. Keep the
    # earliest snapshot in that priority class to stay closest to the original
    # forecast time.
    refs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    ref_priority: Dict[Tuple[str, str], int] = {}
    for r in _sql_rows(con, """
        SELECT *,
               CASE
                 WHEN model = ? THEN 0
                 WHEN model != 'crowd-follow' THEN 1
                 ELSE 2
               END AS ref_priority
        FROM forecast_snapshot
        WHERE resolved = true AND outcome IS NOT NULL AND model = ?
        ORDER BY ref_priority, snapshot_ts
    """, [default_model, default_model]):
        key = (r.get("platform") or "", r.get("ident") or "")
        if key not in refs:
            refs[key] = r
            ref_priority[key] = int(r.get("ref_priority") or 0)

    for r in _sql_rows(con, """
        SELECT *,
               CASE
                 WHEN model = ? THEN 0
                 WHEN model != 'crowd-follow' THEN 1
                 ELSE 2
               END AS ref_priority
        FROM forecast_snapshot
        WHERE resolved = true AND outcome IS NOT NULL
        ORDER BY ref_priority, snapshot_ts
    """, [default_model]):
        key = (r.get("platform") or "", r.get("ident") or "")
        pri = int(r.get("ref_priority") or 0)
        if key not in refs or pri < ref_priority[key]:
            refs[key] = r
            ref_priority[key] = pri

    if not refs:
        return 0

    # Find which (model, market) pairs are missing entirely.
    import asyncio as _asyncio

    tasks: List[Tuple[str, Tuple[str, str], Dict[str, Any]]] = []
    for model in llm_models:
        model_markets = {
            (r[0] or "", r[1] or "")
            for r in con.execute(
                "SELECT platform, ident FROM forecast_snapshot WHERE model = ?", [model]
            ).fetchall()
        }
        for mkey, ref in refs.items():
            if mkey not in model_markets:
                tasks.append((model, mkey, ref))

    if not tasks:
        return 0

    print(f"  backfill: {len(tasks)} missing (model, market) pairs")

    sem = _asyncio.Semaphore(concurrency)
    today = _now().strftime("%Y-%m-%d")

    async def _backfill_one(model: str, ref: Dict[str, Any]) -> bool:
        async with sem:
            try:
                quote = {
                    "platform": ref.get("platform"),
                    "ident": ref.get("ident"),
                    "question": ref.get("question"),
                    "description": ref.get("description"),
                    "resolution_criteria": ref.get("resolution_criteria"),
                    "market_url": ref.get("market_url"),
                    "close_time": ref.get("close_time"),
                    "created_time": ref.get("publish_time"),
                    "category": ref.get("category"),
                    "volume": ref.get("market_volume"),
                    "liquidity": ref.get("market_liquidity"),
                    "probability": ref.get("market_probability"),
                    "price_history": [],
                }
                scored = await forecast_fn(quote, evidence_top_k, model)
            except Exception:
                return False
        if not scored or scored.get("model_probability") is None:
            return False

        # Key the snapshot to the original date with a _backfill suffix so it
        # never collides with a real future snapshot for this market+model.
        slot = (ref.get("snapshot_date") or today) + "_backfill"
        snap_key = client.key(
            SNAPSHOT_KIND,
            f"{ref.get('platform')}:{ref.get('ident')}:{model}:{slot}",
        )
        if client.get(snap_key) is not None:
            return False  # idempotent

        mp = float(scored["model_probability"])
        mkt_prob = ref.get("market_probability")
        lead = ref.get("lead_time_days")
        outcome = ref.get("outcome")
        tags = tag_question(ref.get("question") or "", ref.get("category"))

        entity = _ds.Entity(snap_key, exclude_from_indexes=(
            "question", "market_url", "close_time", "category", "entities",
            "description", "resolution_criteria", "rationale"))
        entity.update(
            platform=ref.get("platform"),
            ident=ref.get("ident"),
            model=model,
            question=ref.get("question"),
            market_url=ref.get("market_url"),
            description=ref.get("description"),
            resolution_criteria=ref.get("resolution_criteria"),
            publish_time=ref.get("publish_time"),
            snapshot_ts=ref.get("snapshot_ts") or _now(),
            snapshot_date=ref.get("snapshot_date") or today,
            model_probability=mp,
            market_probability=float(mkt_prob) if mkt_prob is not None else None,
            close_time=ref.get("close_time"),
            lead_time_days=lead,
            horizon=ref.get("horizon") or _horizon_label(lead),
            category=ref.get("category"),
            market_volume=ref.get("market_volume"),
            market_liquidity=ref.get("market_liquidity"),
            evidence_count=scored.get("evidence_count"),
            venue_news_count=scored.get("venue_news_count"),
            rules_present=scored.get("rules_present"),
            context_complete=scored.get("context_complete"),
            resolved=True,
            outcome=outcome,
            resolved_ts=ref.get("resolved_ts"),
            domain=tags["domain"],
            entities=tags["entities"],
            rationale=scored.get("rationale"),
        )
        if outcome is not None and mkt_prob is not None:
            entity["model_brier"] = brier(mp, outcome)
            entity["market_brier"] = brier(float(mkt_prob), outcome)
            entity["model_correct"] = (mp >= 0.5) == bool(outcome)
        client.put(entity)
        return True

    results = await _asyncio.gather(
        *[_backfill_one(model, ref) for model, _key, ref in tasks],
        return_exceptions=True,
    )
    recorded = sum(1 for r in results if r is True)
    print(f"  backfill: wrote {recorded}/{len(tasks)} snapshots")
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
_COMPOUND_STARTING_BANKROLL = 10_000.0

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

# Models with an established, non-degenerate resolved history for the public
# "Model comparison" table. Newer forecast-matrix additions start out with all
# their resolved snapshots coming from a single backlog market
# (n_markets_resolved == 1) — that reads like a real track record but isn't.
# Keep the comparison to models with real accumulated history until a new
# model actually builds one; MTM is intentionally NOT filtered by this list.
_RESOLVED_QUALITY_MODELS = {
    "council", "crowd-follow", "gpt-oss-120b", "gemma-4-31b-it", "kimi-k2.6",
}


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
    markets.

    A resolved snapshot becomes an eligible paper signal only after the market's
    outcome is known. The PnL/account curves then deduplicate those signals to
    one settled trade per platform+market, using the earliest eligible
    point-in-time call before close. Profit/loss is booked once at
    ``resolved_ts``: a winning $1 of exposure returns ``(1 - p)/p``, a loser
    returns ``-1``. This avoids double-counting repeated snapshots of the same
    real question while still preserving the public signal log for analysis.
    Sizing modes include a flat diagnostic, a calibrated half-Kelly diagnostic,
    and compounded growth strategies. Every curve is one settled trade per
    market, not one return per forecast snapshot.

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

    # Walk-forward calibration: for each bet, calculate a calibrated model_probability
    # based only on bets resolved prior to this one.
    history = []
    for b in all_bets:
        raw_model_p = b["model_probability"]
        if len(history) >= 30:
            # Fit isotonic regression on prior resolved outcomes
            pairs = [(float(x["model_probability"]), int(x["outcome"])) for x in history]
            bp = _fit_isotonic(pairs)
            cal_p = _apply_isotonic(bp, raw_model_p)
        else:
            cal_p = raw_model_p
        b["calibrated_model_probability"] = round(cal_p, 4)
        history.append(b)

    def _public_bets() -> List[Dict[str, Any]]:
        """Public bet log: one row per market/model, preserving the earliest
        point-in-time call. Metrics above still use every trajectory snapshot."""
        by_market_model: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
        for b in all_bets:
            key = (b.get("platform"), b.get("ident"), b.get("model"))
            cur = by_market_model.get(key)
            if cur is None or (b.get("snapshot_ts") or "") < (cur.get("snapshot_ts") or ""):
                by_market_model[key] = b
        return sorted(
            by_market_model.values(),
            key=lambda b: (b.get("resolved_ts") or "", b.get("snapshot_ts") or ""),
        )

    def _run(
        sizing,
        *,
        validated: bool = False,
        filter_fn=None,
        fade: bool = False,
        compound_actual_bankroll: bool = False,
        drawdown_throttle: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Run a paper-PnL simulation over all resolved bets.

        Args:
            sizing: callable(bet_dict) -> stake amount.  Receives the full bet
                    dict so Kelly-style functions can read model_probability,
                    market_probability, edge, etc.
            validated: if True only include bets in statistically-significant
                       edge buckets.
            filter_fn: optional callable(bet_dict) -> bool, skip bet if False.
            fade: if True, bet the *opposite* side to the model's call.
        """
        # Deduplicate to one trade per market using its earliest forecast call before close.
        # Prediction market trades are placed before close and settled ONCE at resolution time.
        _scoped_bets_raw = [
            _b for _b in all_bets
            if not (validated and not _b["in_validated"])
            and not (filter_fn is not None and not filter_fn(_b))
        ]
        _first_idx_map: Dict[Tuple[Any, Any], int] = {}
        for _i, _b in enumerate(_scoped_bets_raw):
            _p = _b.get("platform")
            _id = _b.get("ident")
            if _p and _id:
                _k = (_p, _id)
                if _k not in _first_idx_map:
                    _first_idx_map[_k] = _i
            else:
                _first_idx_map[(_i, _i)] = _i

        _scoped_bets = [
            _b for _i, _b in enumerate(_scoped_bets_raw)
            if (not _b.get("platform") or not _b.get("ident")) or _i == _first_idx_map.get((_b.get("platform"), _b.get("ident")))
        ]

        # Precount planned exposure.
        _pre_staked = 0.0
        for b in _scoped_bets:
            _s = sizing(b)
            if _s > 0.0:
                _pre_staked += _s

        staked = pnl = wins = fees = 0.0
        n = 0
        cum = 0.0
        curve: List[float] = []
        curve_ts: List[Any] = []
        growth_curve: List[float] = []
        growth_curve_ts: List[Any] = []
        bankroll = _COMPOUND_STARTING_BANKROLL
        peak_bankroll = bankroll
        for _idx, b in enumerate(_scoped_bets):
            try:
                stake = sizing(b, bankroll)
            except TypeError:
                stake = sizing(b)
            if stake <= 0.0:
                continue
            if drawdown_throttle:
                drawdown = 1.0 - (bankroll / peak_bankroll) if peak_bankroll else 0.0
                if drawdown >= 0.25:
                    continue
                if drawdown >= 0.15:
                    stake *= 0.5

            mkt_p = b["market_probability"]
            side = b["side"]
            win_val = b["win"]

            if fade:
                fade_side = "NO" if side == "YES" else "YES"
                p_side = (1.0 - mkt_p) if fade_side == "NO" else mkt_p
                win = (b["outcome"] == 1) if fade_side == "YES" else (b["outcome"] == 0)
            else:
                p_side = mkt_p if side == "YES" else (1.0 - mkt_p)
                win = win_val

            if p_side <= 0.0 or p_side >= 1.0:
                continue

            payout = (1.0 - p_side) / p_side
            fee = _bet_fee(b["platform"], stake, p_side)
            profit = stake * (payout if win else -1.0) - fee
            staked += stake
            pnl += profit
            fees += fee
            wins += 1 if win else 0
            n += 1
            cum += profit
            curve.append(round(cum, 4))
            curve_ts.append(b["resolved_ts"])
            if compound_actual_bankroll:
                bankroll = max(0.0, bankroll + profit)
                peak_bankroll = max(peak_bankroll, bankroll)
            elif _pre_staked > 0:
                stake_fraction = stake / _pre_staked
                bankroll *= max(0.0, 1.0 + stake_fraction * (profit / stake))
            if compound_actual_bankroll or _pre_staked > 0:
                growth_curve.append(round(bankroll, 6))
                growth_curve_ts.append(b["resolved_ts"])
        if not n:
            return None
        return {
            "n_bets": n,
            "total_staked": round(staked, 4),
            "fees": round(fees, 4),
            "pnl": round(pnl, 4),
            "roi": round(pnl / staked, 4) if staked else None,
            "win_rate": round(wins / n, 4),
            "equity_curve": curve,
            "equity_curve_ts": curve_ts,
            "growth_curve": growth_curve,
            # One timestamp per growth_curve point (deduped to one per market,
            # unlike equity_curve_ts which has one per snapshot) -- without
            # this, the frontend's times.length !== points.length check fails
            # silently, the chart falls back to synthetic even-spacing, and
            # hover tooltips read the wrong index against the wrong curve.
            "growth_curve_ts": growth_curve_ts,
            "compound_bankroll": round(bankroll, 6),
            "compound_return": round((bankroll / _COMPOUND_STARTING_BANKROLL) - 1.0, 4),
        }

    flat = _run(lambda b: 1.0)
    if flat is None:
        return None

    def _smart_filter(b):
        """Data-driven composite filter:
        1. Entry price between 20¢-80¢ (avoid heavy-fav trap + extreme longshots)
        2. Skip geopolitics when edge is large (model can't forecast geopolitics)
        3. Cap edge at 40% (above that the model is hallucinating)
        """
        mkt_p = b["market_probability"]
        p_side = mkt_p if b["side"] == "YES" else (1.0 - mkt_p)
        # Price guard: no extreme favourites or extreme longshots
        if p_side < 0.20 or p_side > 0.80:
            return False
        # Domain guard: geopolitics is a coin-flip; only allow tiny-edge geo bets
        if b.get("domain") == "geopolitics" and b["edge"] > 0.10:
            return False
        # Overconfidence guard: extreme edge = hallucination
        if b["edge"] > 0.40:
            return False
        return True

    # ── Kelly & Growth sizing ─────────────────────────────────
    def _half_kelly_fraction(b):
        """Return the calibrated half-Kelly fraction before any stake cap."""
        mkt_p = b["market_probability"]
        model_p = b["calibrated_model_probability"]
        side = b["side"]
        if side == "YES":
            p_win = model_p
            p_side = mkt_p
        else:
            p_win = 1.0 - model_p
            p_side = 1.0 - mkt_p
        if p_side <= 0.0 or p_side >= 1.0:
            return 0.0
        odds = (1.0 - p_side) / p_side
        if odds <= 0:
            return 0.0
        return max(0.0, 0.5 * ((p_win * odds - (1.0 - p_win)) / odds))

    def _half_kelly(b, bk=_COMPOUND_STARTING_BANKROLL):
        """Half-Kelly criterion: f* = 0.5 * (p*b - q) / b
        where p = model's walk-forward calibrated probability of winning,
              b = payout odds (decimal, e.g. 1.5 means you get $1.50 on $1),
              q = 1 - p.
        Capped at 1% of the reference bankroll per market.
        """
        return min(_half_kelly_fraction(b), 0.01) * max(0.0, bk)

    def _half_kelly_20pct(b, bk=_COMPOUND_STARTING_BANKROLL):
        """Conservative Kelly: quarter-Kelly, market shrinkage, 5% cap."""
        return _conservative_kelly(b, bk)

    def _conservative_kelly(b, bk=_COMPOUND_STARTING_BANKROLL):
        """Quarter-Kelly with 50% market shrinkage and a 5% risk cap."""
        market_p = b["market_probability"]
        model_p = b["calibrated_model_probability"]
        if b["side"] == "NO":
            market_p, model_p = 1.0 - market_p, 1.0 - model_p
        p_win = market_p + 0.5 * (model_p - market_p)
        if not (0.0 < market_p < 1.0):
            return 0.0
        odds = (1.0 - market_p) / market_p
        full_kelly = (p_win * odds - (1.0 - p_win)) / odds
        return min(max(0.0, 0.25 * full_kelly), 0.05) * max(0.0, bk)

    def _growth_1pct(b, bk=_COMPOUND_STARTING_BANKROLL):
        """Dynamic 1% bankroll sizing per trade for capital growth."""
        return max(0.0, 0.01 * bk)

    def _growth_2pct(b, bk=_COMPOUND_STARTING_BANKROLL):
        """Dynamic 2% bankroll sizing per trade for aggressive capital growth."""
        return max(0.0, 0.02 * bk)

    return {
        "min_edge": min_edge,
        "disclaimer": "Hypothetical/paper, net of venue trading fees (Kalshi price-based; "
                      "Polymarket fee-free). Excludes slippage/liquidity unless configured. "
                      "Signal check, not live PnL.",
        "flat": flat,
        "half_kelly": _run(_half_kelly),
        "half_kelly_20pct": _run(_half_kelly_20pct, compound_actual_bankroll=True, drawdown_throttle=True),
        "smart": _run(_growth_1pct, filter_fn=_smart_filter, compound_actual_bankroll=True),
        "growth_1pct": _run(_growth_1pct, compound_actual_bankroll=True),
        "growth_2pct": _run(_growth_2pct, compound_actual_bankroll=True),
        "bets": _public_bets(),
    }


def crowd_baseline_equity(resolved: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Equity curve for a crowd-follow strategy: always bet the market's own
    direction (model_p = market_p), flat $1 stake, no edge threshold.

    By definition the crowd-follow model has zero edge vs the market, so it
    never passes the standard paper_pnl min_edge filter.  This separate
    computation shows what pure crowd-following earns on the same resolved
    market set — the zero-edge baseline the LLM models are judged against."""
    if not resolved:
        return None

    first_mkt_idx: Dict[Tuple[Any, Any], int] = {}
    sorted_resolved = sorted(resolved, key=lambda x: x.get("resolved_ts") or _now())
    for _i, r in enumerate(sorted_resolved):
        _p = r.get("platform")
        _id = r.get("ident")
        if _p and _id:
            _k = (_p, _id)
            if _k not in first_mkt_idx:
                first_mkt_idx[_k] = _i
        else:
            first_mkt_idx[(_i, _i)] = _i

    dedup_resolved = [
        r for _i, r in enumerate(sorted_resolved)
        if (not r.get("platform") or not r.get("ident")) or _i == first_mkt_idx.get((r.get("platform"), r.get("ident")))
    ]

    cum = 0.0
    curve: List[float] = []
    curve_ts: List[Any] = []
    staked = pnl = wins = 0.0
    n = 0
    growth_curve: List[float] = []
    growth_curve_ts: List[Any] = []
    bankroll = _COMPOUND_STARTING_BANKROLL
    dedup_staked = float(len(dedup_resolved))

    for r in dedup_resolved:
        market_p = float(r.get("market_probability") or 0.5)
        if market_p == 0.5:
            continue
        side_yes = market_p > 0.5
        p_side = market_p if side_yes else (1.0 - market_p)
        if not (0.0 < p_side < 1.0):
            continue
        outcome = int(r.get("outcome") or 0)
        win = (outcome == 1) == side_yes
        payout = (1.0 - p_side) / p_side
        fee = _bet_fee(r.get("platform"), 1.0, p_side)
        profit = 1.0 * (payout if win else -1.0) - fee
        staked += 1.0
        pnl += profit
        wins += 1 if win else 0
        n += 1
        cum += profit
        curve.append(round(cum, 4))
        ts = r.get("resolved_ts")
        if isinstance(ts, dict):
            ts = ts.get("__dt__")
        elif hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        curve_ts.append(ts)
        if dedup_staked > 0:
            bankroll *= max(0.0, 1.0 + (1.0 / dedup_staked) * profit)
            growth_curve.append(round(bankroll, 6))
            growth_curve_ts.append(ts)

    if not n:
        return None

    return {
        "n_bets": n,
        "total_staked": round(staked, 4),
        "pnl": round(pnl, 4),
        "roi": round(pnl / staked, 4) if staked else None,
        "win_rate": round(wins / n, 4),
        "equity_curve": curve,
        "equity_curve_ts": curve_ts,
        "growth_curve": growth_curve,
        "growth_curve_ts": growth_curve_ts,
        "compound_bankroll": round(bankroll, 6),
        "compound_return": round((bankroll / _COMPOUND_STARTING_BANKROLL) - 1.0, 4),
    }


def _crowd_baseline_reference_rows(resolved: List[Dict[str, Any]], *,
                                   default_model: str) -> List[Dict[str, Any]]:
    """Use the largest single LLM cohort for the crowd-follow comparison row.

    The headline/equity curve can aggregate all tracked LLM forecasts, but the
    market-following baseline must stay comparable to one model's opportunity
    set. Otherwise Council, GPT, Gemma, and Kimi duplicates inflate the baseline
    count.
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in resolved:
        mlabel = r.get("model") or default_model
        if mlabel == "crowd-follow":
            continue
        by_model.setdefault(mlabel, []).append(r)
    if not by_model:
        return []
    return max(by_model.items(), key=lambda item: (len(item[1]), item[0]))[1]


def build_models_comparison(resolved: List[Dict[str, Any]], *,
                            default_model: str,
                            crowd_baseline: Optional[Dict[str, Any]] = None,
                            crowd_reference_rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Per-model leaderboard over resolved snapshots: accuracy, skill-vs-market,
    and hypothetical paper-trading ROI (flat + validated-only). Ranked
    best-paper-edge first.

    crowd_baseline: pre-computed result of crowd_baseline_equity() on the
    largest single LLM cohort.
    When supplied, the crowd-follow row uses this data instead of its own snapshots
    so its numbers match the equity curve's crowd baseline line exactly."""
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in resolved:
        by_model.setdefault(r.get("model") or default_model, []).append(r)
    out: List[Dict[str, Any]] = []
    for mlabel, rows in by_model.items():
        if mlabel == "crowd-follow" and crowd_baseline is not None:
            cb = crowd_baseline
            crowd_market_count = len({
                (r.get("platform"), r.get("ident"))
                for r in (crowd_reference_rows or [])
            }) or len({(r.get("platform"), r.get("ident")) for r in rows})
            out.append({
                "model": mlabel,
                "n_snapshots_resolved": cb.get("n_bets"),
                "n_markets_resolved": crowd_market_count,
                "accuracy": cb.get("win_rate"),
                "model_brier": None,
                "skill_vs_market": 0.0,
                "paper_roi": cb.get("roi"),
                "paper_roi_smart": None,
                "paper_pnl": {"flat": cb},
                "by_horizon": [],
            })
            continue
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
            "paper_roi_smart": ((pp or {}).get("smart") or {}).get("roi"),
            "paper_pnl": pp,
            "by_horizon": model_by_horizon,
        })
    out.sort(key=lambda m: (m["paper_roi_smart"] if m["paper_roi_smart"] is not None else -9.0,
                            m["paper_roi"] if m["paper_roi"] is not None else -9.0,
                            m["skill_vs_market"] if m["skill_vs_market"] is not None else -9.0),
             reverse=True)
    return out


def _sharpe_and_max_drawdown(value_curve: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Per-step Sharpe ratio and max drawdown from an account's mark-to-market
    value curve (the same curve the equity chart plots).

    Unannualized: value_curve points are event-driven (one per trade/
    settlement), not fixed-interval, so there's no clean bar count to
    annualize against. This is comparable across models scored the same way,
    not meant to match a textbook annualized Sharpe."""
    values = [
        float(p["account_value"]) for p in value_curve
        if isinstance(p, dict) and p.get("account_value") is not None
    ]
    if len(values) < 2:
        return {"sharpe": None, "max_drawdown": None}

    returns = [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(1, len(values))
        if values[i - 1] > 0
    ]
    sharpe = None
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = var ** 0.5
        if std > 0:
            sharpe = round(mean / std, 4)

    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)

    return {"sharpe": sharpe, "max_drawdown": round(max_dd, 4)}


def build_mark_to_market_accounts(
    rows: List[Dict[str, Any]],
    latest_quotes: Dict[Any, Dict[str, Any]],
    *,
    default_model: str,
    tracked_models: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Bid-liquidation shadow strategy ledgers for every active tracked model."""
    all_rows = list(rows)
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        label = str(row.get("model") or default_model)
        by_model.setdefault(label, []).append(row)
    for label in tracked_models or []:
        if label:
            by_model.setdefault(str(label), [])

    accounts: List[Dict[str, Any]] = []
    for label, rows in by_model.items():
        account = _public_mark_to_market_account(
            simulate_shadow_mark_to_market_account(
                rows,
                latest_quotes=latest_quotes,
                fee_fn=_bet_fee,
            )
        )
        risk = _sharpe_and_max_drawdown(account.get("value_curve") or [])
        accounts.append({
            "model": label,
            "account": account,
            "account_value": account.get("account_value"),
            "return": account.get("return"),
            "n_trades": account.get("n_trades"),
            "n_settlements": account.get("n_settlements"),
            "n_open_positions": account.get("n_open_positions"),
            "n_illiquid_positions": account.get("n_illiquid_positions"),
            "sharpe": risk["sharpe"],
            "max_drawdown": risk["max_drawdown"],
        })
    baseline_rows = _crowd_baseline_reference_rows(all_rows, default_model=default_model)
    if baseline_rows:
        account = _public_mark_to_market_account(
            simulate_market_follow_mark_to_market_account(
                baseline_rows,
                latest_quotes=latest_quotes,
                fee_fn=_bet_fee,
            )
        )
        risk = _sharpe_and_max_drawdown(account.get("value_curve") or [])
        accounts.append({
            "model": "market-follow",
            "account": account,
            "account_value": account.get("account_value"),
            "return": account.get("return"),
            "n_trades": account.get("n_trades"),
            "n_settlements": account.get("n_settlements"),
            "n_open_positions": account.get("n_open_positions"),
            "n_illiquid_positions": account.get("n_illiquid_positions"),
            "sharpe": risk["sharpe"],
            "max_drawdown": risk["max_drawdown"],
            "baseline": True,
        })
    accounts.sort(
        key=lambda item: (
            int((item.get("n_trades") or 0) > 0),
            item.get("account_value") is not None,
            float(item.get("account_value") or 0.0),
        ),
        reverse=True,
    )
    return accounts


def _attach_walk_forward_calibration(rows: List[Dict[str, Any]], *, min_history: int = 30) -> List[Dict[str, Any]]:
    """Attach 'calibrated_model_probability' to each row, fit only on PRIOR
    resolved rows in resolved_ts order (walk-forward, no lookahead) -- mirrors
    paper_pnl's calibration exactly. Returns rows sorted by resolved_ts
    (mutated in place and returned for convenience)."""
    ordered = sorted(rows, key=lambda x: x.get("resolved_ts") or _now())
    history: List[Dict[str, Any]] = []
    for r in ordered:
        raw_model_p = float(r.get("model_probability") or 0.5)
        if len(history) >= min_history:
            pairs = [(float(x["model_probability"]), int(x["outcome"])) for x in history]
            bp = _fit_isotonic(pairs)
            cal_p = _apply_isotonic(bp, raw_model_p)
        else:
            cal_p = raw_model_p
        r["calibrated_model_probability"] = round(cal_p, 4)
        history.append(r)
    return ordered


def build_validated_kelly_accounts(
    rows: List[Dict[str, Any]],
    latest_quotes: Dict[Any, Dict[str, Any]],
    *,
    default_model: str,
    tracked_models: Optional[List[str]] = None,
    kelly_fraction: float = 0.5,
    max_concentration: float = 0.15,
    market_shrinkage: float = 0.0,
    max_drawdown: Optional[float] = None,
    require_validated: bool = True,
    follow_model_call: bool = False,
    include_collecting_models: bool = False,
) -> List[Dict[str, Any]]:
    """Per-model deployable strategy: a real position ledger (one settle per
    market, per simulate_validated_kelly_account) sized by fractional Kelly on
    walk-forward calibrated probability, gated on statistically-proven edge,
    and capped per-market so no single miscalibrated call can dominate the
    book. Unlike build_mark_to_market_accounts' fixed $1 diagnostic sizing,
    this is meant to be the strategy actually traded going forward.

    crowd-follow is skipped: it has no model-vs-market disagreement to
    validate or size against (see build_models_comparison's identical
    handling). Each account also gets 'recommended_weight': 0 for any model
    without a validated positive return, otherwise proportional to its
    realized return among the models that do -- capital only goes to models
    with actual proven edge, not to whichever one merely looks best."""
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get("model") or default_model)
        by_model.setdefault(label, []).append(row)

    accounts: List[Dict[str, Any]] = []
    # Keep the configured roster visible, but also retain historical labels
    # when a model has since been retired from the forecasting rotation.
    labels = list(dict.fromkeys([*(tracked_models or []), *by_model.keys()]))
    for label in labels:
        if label == "crowd-follow":
            continue
        model_rows = by_model.get(label) or []
        resolved_rows = [r for r in model_rows if r.get("resolved") and r.get("outcome") is not None]
        if not resolved_rows:
            if include_collecting_models:
                account = simulate_validated_kelly_account(
                    [],
                    latest_quotes=latest_quotes,
                    kelly_fraction=kelly_fraction,
                    max_concentration=max_concentration,
                    market_shrinkage=market_shrinkage,
                    max_drawdown=max_drawdown,
                    require_validated=require_validated,
                    follow_model_call=follow_model_call,
                    fee_fn=_bet_fee,
                )
                accounts.append({
                    "model": label,
                    "account": account,
                    "account_value": account.get("account_value"),
                    "return": account.get("return"),
                    "n_trades": 0,
                    "n_settlements": 0,
                    "n_skipped_not_validated": 0,
                    "n_validated_buckets": 0,
                    "sharpe": None,
                    "max_drawdown": None,
                    "status": "collecting_history",
                })
            continue
        calibrated_rows = _attach_walk_forward_calibration([dict(r) for r in resolved_rows])
        sig_buckets = {
            b["edge_bucket"] for b in edge_calibration(calibrated_rows)
            if b.get("skill_significant")
        }
        for r in calibrated_rows:
            r["_edge_validated"] = _edge_label(_edge(r)) in sig_buckets

        # Open rows occur after the resolved calibration history. Apply the
        # current calibration so their account value can be marked to market;
        # historical rows above remain strictly walk-forward.
        history_pairs = [
            (float(r["model_probability"]), int(r["outcome"]))
            for r in calibrated_rows
        ]
        calibration = _fit_isotonic(history_pairs) if len(history_pairs) >= 30 else None
        for row in model_rows:
            if row.get("resolved") and row.get("outcome") is not None:
                continue
            live_row = dict(row)
            raw_p = float(live_row.get("model_probability") or 0.5)
            live_row["calibrated_model_probability"] = round(
                _apply_isotonic(calibration, raw_p) if calibration else raw_p,
                4,
            )
            live_row["_edge_validated"] = _edge_label(_edge(live_row)) in sig_buckets
            calibrated_rows.append(live_row)

        account = simulate_validated_kelly_account(
            calibrated_rows,
            latest_quotes=latest_quotes,
            kelly_fraction=kelly_fraction,
            max_concentration=max_concentration,
            market_shrinkage=market_shrinkage,
            max_drawdown=max_drawdown,
            require_validated=require_validated,
            follow_model_call=follow_model_call,
            fee_fn=_bet_fee,
        )
        risk = _sharpe_and_max_drawdown(account.get("value_curve") or [])
        accounts.append({
            "model": label,
            "account": account,
            "account_value": account.get("account_value"),
            "return": account.get("return"),
            "n_trades": account.get("n_trades"),
            "n_settlements": account.get("n_settlements"),
            "n_skipped_not_validated": account.get("n_skipped_not_validated"),
            "n_validated_buckets": len(sig_buckets),
            "sharpe": risk["sharpe"],
            "max_drawdown": risk["max_drawdown"],
            "status": "active",
        })

    positive = [a for a in accounts if (a.get("return") or 0.0) > 0]
    total_positive_return = sum(a["return"] for a in positive)
    for a in accounts:
        a["recommended_weight"] = (
            round(a["return"] / total_positive_return, 4)
            if a in positive and total_positive_return > 0
            else 0.0
        )

    accounts.sort(key=lambda a: (a.get("return") if a.get("return") is not None else -1.0), reverse=True)
    return accounts


def build_half_kelly_20pct_accounts(
    rows: List[Dict[str, Any]],
    latest_quotes: Dict[Any, Dict[str, Any]],
    *,
    default_model: str,
    tracked_models: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Replay every executable model call with walk-forward half-Kelly sizing.

    This is the MTM counterpart to the historical growth views: it starts from
    the same June forecast history, books each market once at settlement, and
    caps every position at 5% of the current account, shrinks model
    probabilities 50% toward market pricing, and reserves a 30% maximum
    high-watermark loss budget. Unlike the guarded
    deployable strategy, it does not require a significance-approved edge
    bucket, so it can show the full historical sizing trajectory.
    """
    return build_validated_kelly_accounts(
        rows,
        latest_quotes,
        default_model=default_model,
        tracked_models=tracked_models,
        kelly_fraction=0.25,
        max_concentration=0.05,
        market_shrinkage=0.5,
        max_drawdown=0.30,
        require_validated=False,
        follow_model_call=True,
        include_collecting_models=True,
    )


def build_growth_accounts(
    rows: List[Dict[str, Any]],
    latest_quotes: Dict[Any, Dict[str, Any]],
    *,
    default_model: str,
    tracked_models: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build live 1% and 2% fixed-fraction model accounts.

    Each model follows its executable calls into a real shadow ledger. Open
    positions are bid-marked between entry and resolution, while realized P&L
    is still booked exactly once when the market settles.
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get("model") or default_model)
        by_model.setdefault(label, []).append(row)

    labels = list(dict.fromkeys([*(tracked_models or []), *by_model.keys()]))
    accounts = {"growth_1pct": [], "growth_2pct": []}
    for label in labels:
        if label == "crowd-follow":
            continue
        model_rows = [dict(row) for row in (by_model.get(label) or [])]
        for row in model_rows:
            row["calibrated_model_probability"] = float(row.get("model_probability") or 0.5)
            row["_edge_validated"] = True
        for strategy_key, fraction in (("growth_1pct", 0.01), ("growth_2pct", 0.02)):
            values = accounts[strategy_key]
            account = simulate_validated_kelly_account(
                model_rows,
                latest_quotes=latest_quotes,
                fixed_fraction=fraction,
                strategy_name=f"live_{strategy_key}_ledger",
                max_concentration=fraction,
                require_validated=False,
                follow_model_call=True,
                min_price=0.2,
                max_price=0.8,
                fee_fn=_bet_fee,
            )
            risk = _sharpe_and_max_drawdown(account.get("value_curve") or [])
            values.append({
                "model": label,
                "account": account,
                "account_value": account.get("account_value"),
                "return": account.get("return"),
                "n_trades": account.get("n_trades"),
                "n_settlements": account.get("n_settlements"),
                "sharpe": risk["sharpe"],
                "max_drawdown": risk["max_drawdown"],
                "status": "active" if model_rows else "collecting_history",
            })

    for values in accounts.values():
        values.sort(key=lambda item: float(item.get("account_value") or 0.0), reverse=True)
    return accounts


_FADE_BUCKET = "20pp+"


def _attach_fade_calibration(
    rows: List[Dict[str, Any]], *, min_history: int = 10, fade_bucket: str = _FADE_BUCKET,
) -> List[Dict[str, Any]]:
    """Attach 'calibrated_model_probability' (calibrated P(YES), for whichever
    side the fade strategy ends up trading) and '_edge_validated' for
    simulate_validated_kelly_account(fade=True): it trades the *opposite* of
    the model's implied side, restricted to fade_bucket -- the disagreement
    size proven, across every tracked model, to have statistically
    significant NEGATIVE model skill (edge_calibration's 20pp+ bucket).

    Walk-forward, no lookahead: the fade win-rate used for a given row comes
    only from this model's PRIOR resolved fade_bucket history. _edge_validated
    is only True once min_history prior observations exist in that bucket --
    below that the estimate isn't trusted yet and the row is skipped, exactly
    like the 'not enough proven history' case in the follow strategy."""
    ordered = sorted(rows, key=lambda x: x.get("resolved_ts") or _now())
    fade_outcomes: List[bool] = []
    for r in ordered:
        model_p = float(r.get("model_probability") or 0.5)
        market_p = float(r.get("market_probability") or 0.5)
        in_bucket = _edge_label(abs(model_p - market_p)) == fade_bucket
        # Matches accounting._row_edge_side's own direction convention exactly
        # (model_p vs market_p, not model_p vs 0.5) so the "fade" side here is
        # the same side simulate_validated_kelly_account(fade=True) will hold.
        model_side_yes = model_p > market_p
        fade_side_yes = not model_side_yes

        r["_edge_validated"] = False
        if in_bucket and len(fade_outcomes) >= min_history:
            fade_win_rate = sum(fade_outcomes) / len(fade_outcomes)
            r["calibrated_model_probability"] = round(
                fade_win_rate if fade_side_yes else (1.0 - fade_win_rate), 4
            )
            r["_edge_validated"] = True

        if in_bucket:
            outcome = int(r.get("outcome") or 0)
            fade_outcomes.append((outcome == 1) == fade_side_yes)
    return ordered


def build_fade_kelly_accounts(
    rows: List[Dict[str, Any]],
    latest_quotes: Dict[Any, Dict[str, Any]],
    *,
    default_model: str,
    tracked_models: Optional[List[str]] = None,
    kelly_fraction: float = 0.5,
    max_concentration: float = 0.15,
) -> List[Dict[str, Any]]:
    """Per-model fade strategy: bet *against* the model's implied side, sized
    by fractional Kelly, specifically in the disagreement regime proven --
    for every tracked model, not just one -- to have significantly negative
    model skill (see build_validated_kelly_accounts' docstring for why that
    regime exists: models are consistently worse than the market once they
    disagree with it by more than 20 percentage points). Complements, doesn't
    replace, build_validated_kelly_accounts: that one only activates once a
    model *proves* positive edge (currently: none has, yet); this one is
    already actionable today, on the one edge the data has actually proven."""
    resolved_only = [r for r in rows if r.get("resolved") and r.get("outcome") is not None]
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in resolved_only:
        label = str(row.get("model") or default_model)
        by_model.setdefault(label, []).append(row)

    accounts: List[Dict[str, Any]] = []
    for label in (tracked_models or list(by_model.keys())):
        if label == "crowd-follow":
            continue
        model_rows = by_model.get(label) or []
        if not model_rows:
            continue
        calibrated_rows = _attach_fade_calibration([dict(r) for r in model_rows])
        n_eligible = sum(1 for r in calibrated_rows if r.get("_edge_validated"))

        account = simulate_validated_kelly_account(
            calibrated_rows,
            latest_quotes=latest_quotes,
            kelly_fraction=kelly_fraction,
            max_concentration=max_concentration,
            fade=True,
            fee_fn=_bet_fee,
        )
        risk = _sharpe_and_max_drawdown(account.get("value_curve") or [])
        accounts.append({
            "model": label,
            "account": account,
            "account_value": account.get("account_value"),
            "return": account.get("return"),
            "n_trades": account.get("n_trades"),
            "n_settlements": account.get("n_settlements"),
            "n_skipped_not_validated": account.get("n_skipped_not_validated"),
            "n_eligible_fade_rows": n_eligible,
            "sharpe": risk["sharpe"],
            "max_drawdown": risk["max_drawdown"],
        })

    positive = [a for a in accounts if (a.get("return") or 0.0) > 0]
    total_positive_return = sum(a["return"] for a in positive)
    for a in accounts:
        a["recommended_weight"] = (
            round(a["return"] / total_positive_return, 4)
            if a in positive and total_positive_return > 0
            else 0.0
        )

    accounts.sort(key=lambda a: (a.get("return") if a.get("return") is not None else -1.0), reverse=True)
    return accounts


_MTM_PUBLIC_ACCOUNT_FIELDS = (
    "ts",
    "strategy",
    "value_method",
    "starting_cash",
    "cash",
    "liquidation_value",
    "account_value",
    "return",
    "unrealized_pnl",
    "realized_pnl",
    "fees_paid",
    "n_trades",
    "n_settlements",
    "n_open_positions",
    "n_illiquid_positions",
    "n_skipped_no_edge",
    "n_skipped_no_direction",
    "n_skipped_unexecutable",
    "notes",
)
_MTM_PUBLIC_CURVE_FIELDS = (
    "ts",
    "value_method",
    "cash",
    "liquidation_value",
    "account_value",
    "unrealized_pnl",
    "realized_pnl",
    "fees_paid",
    "event_type",
    "trade_status",
)


def _public_mark_to_market_account(account: Dict[str, Any]) -> Dict[str, Any]:
    """Compact MTM account payload for public JSON and chart rendering."""

    def _position_count(snapshot: Dict[str, Any], key: str) -> int:
        positions = snapshot.get(key)
        if isinstance(positions, list):
            return len(positions)
        return int(snapshot.get(f"n_{key}") or 0)

    public = {
        key: account.get(key)
        for key in _MTM_PUBLIC_ACCOUNT_FIELDS
        if key in account
    }
    public.setdefault("n_open_positions", _position_count(account, "open_positions"))
    public.setdefault("n_illiquid_positions", _position_count(account, "illiquid_positions"))

    curve = []
    for point in account.get("value_curve") or []:
        if not isinstance(point, dict):
            continue
        item = {
            key: point.get(key)
            for key in _MTM_PUBLIC_CURVE_FIELDS
            if key in point
        }
        item["n_open_positions"] = _position_count(point, "open_positions")
        item["n_illiquid_positions"] = _position_count(point, "illiquid_positions")
        curve.append(item)
    final_point = {
        key: account.get(key)
        for key in _MTM_PUBLIC_CURVE_FIELDS
        if key in account
    }
    if final_point.get("ts") and final_point.get("account_value") is not None:
        final_point.setdefault("event_type", "heartbeat")
        final_point["n_open_positions"] = public["n_open_positions"]
        final_point["n_illiquid_positions"] = public["n_illiquid_positions"]
        last = curve[-1] if curve else {}
        if (
            last.get("ts") != final_point.get("ts")
            or last.get("account_value") != final_point.get("account_value")
        ):
            curve.append(final_point)
    public["value_curve"] = curve
    return public


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
        forecast_market_p = _optional_float(r.get("market_probability"))
        crowd_move = (
            market_p - forecast_market_p
            if forecast_market_p is not None
            else None
        )
        snapshot_dt = _parse_dt(r.get("snapshot_ts"))
        forecast_age_hours = (
            max(0.0, (_now() - snapshot_dt).total_seconds() / 3600.0)
            if snapshot_dt is not None
            else None
        )
        huge_gap = abs(signed) >= 0.20
        rules_present = bool(
            str(r.get("resolution_criteria") or "").strip()
            or r.get("rules_present")
        )
        evidence_count = int(_optional_float(r.get("evidence_count")) or 0)
        venue_news_count = int(_optional_float(r.get("venue_news_count")) or 0)
        context_complete = bool(
            r.get("context_complete")
            or (rules_present and evidence_count > 0)
        )
        thin_market = max(
            _optional_float(r.get("market_volume")) or 0.0,
            _optional_float(r.get("market_liquidity")) or 0.0,
        ) < 1000.0
        review_reasons: List[str] = []
        if huge_gap and crowd_move is not None and abs(crowd_move) >= 0.10:
            review_reasons.append("crowd_moved_since_forecast")
        if huge_gap and (model_p <= 0.10 or model_p >= 0.90):
            review_reasons.append("extreme_model_probability")
        if huge_gap and thin_market:
            review_reasons.append("thin_market")
        if huge_gap and not rules_present:
            review_reasons.append("missing_market_rules")
        if huge_gap and evidence_count == 0:
            review_reasons.append("missing_news_context")
        if not huge_gap:
            discrepancy_status = "normal"
        elif "crowd_moved_since_forecast" in review_reasons:
            discrepancy_status = "stale_forecast"
        elif not context_complete:
            discrepancy_status = "context_incomplete"
        elif "thin_market" in review_reasons:
            discrepancy_status = "thin_market"
        elif "extreme_model_probability" in review_reasons:
            discrepancy_status = "extreme_forecast"
        else:
            discrepancy_status = "genuine_candidate"
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
            "ident": ident,
            "market_url": r.get("market_url"),
            "forecasted_at": _iso_ts(r.get("snapshot_ts")),
            "description": r.get("description"),
            "resolution_criteria": r.get("resolution_criteria"),
            "categories": [r.get("category")] if r.get("category") else [],
            "domain": r.get("domain") or "other",
            "horizon": r.get("horizon"),
            "lead_days": round(current_lead, 1) if current_lead is not None else None,
            "resolve_time": _iso_ts(r.get("close_time")),
            "market_bid": r.get("market_bid"),
            "market_ask": r.get("market_ask"),
            "market_volume": r.get("market_volume"),
            "market_liquidity": r.get("market_liquidity"),
            "evidence_count": evidence_count,
            "venue_news_count": venue_news_count,
            "rules_present": rules_present,
            "context_complete": context_complete,
            "model_probability": round(model_p, 3),
            "market_probability": round(market_p, 3),
            "forecast_market_probability": (
                round(forecast_market_p, 3)
                if forecast_market_p is not None
                else None
            ),
            "crowd_move_since_forecast": (
                round(crowd_move, 3) if crowd_move is not None else None
            ),
            "forecast_age_hours": (
                round(forecast_age_hours, 1)
                if forecast_age_hours is not None
                else None
            ),
            "edge": round(signed, 3),
            "abs_edge": round(abs(signed), 3),
            "needs_discrepancy_review": huge_gap,
            "discrepancy_status": discrepancy_status,
            "review_reasons": review_reasons,
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
                "accuracy": lead_tr.get("accuracy"),
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


def _is_similar_tokens(tokens1: set[str], tokens2: set[str]) -> bool:
    stopwords = {"will", "the", "a", "an", "is", "are", "on", "before", "by", "to", "in", "of", "at", "for", "with", "that", "and", "or", "happen", "be"}
    words1 = tokens1 - stopwords
    words2 = tokens2 - stopwords
    
    if not words1 or not words2:
        return False
        
    # Check numbers/digits (years, percentages, dates)
    digits1 = {w for w in tokens1 if w.isdigit()}
    digits2 = {w for w in tokens2 if w.isdigit()}
    if digits1 != digits2:
        return False
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    jaccard = len(intersection) / len(union) if union else 0.0
    overlap = len(intersection) / min(len(words1), len(words2)) if words1 and words2 else 0.0
    
    # We want either high Jaccard or high overlap
    return jaccard >= 0.65 or overlap >= 0.80


def _is_similar_question(q1: str, q2: str) -> bool:
    import re
    # Lowercase and split into alphanumeric tokens
    tokens1 = set(re.findall(r'[a-z0-9]+', q1.lower()))
    tokens2 = set(re.findall(r'[a-z0-9]+', q2.lower()))
    return _is_similar_tokens(tokens1, tokens2)


def build_arbitrage_board(open_rows: List[Dict[str, Any]], latest_price: Dict[str, float]) -> List[Dict[str, Any]]:
    """Find price discrepancies between platforms for identical/similar questions."""
    # Group latest open snapshots by key (platform, ident)
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in open_rows:
        key = (r.get("platform"), r.get("ident"))
        cur = latest.get(key)
        if cur is None or (r.get("snapshot_ts") or _now()) > (cur.get("snapshot_ts") or _now()):
            latest[key] = r

    # Filter only those that are actively priced in latest_price
    active_markets = []
    for (platform, ident), r in latest.items():
        market_p = latest_price.get(ident)
        if market_p is not None and r.get("question") and r.get("market_url"):
            active_markets.append({
                "question": r["question"],
                "platform": platform,
                "market_url": r["market_url"],
                "market_probability": float(market_p),
                "ident": ident
            })

    # Pre-tokenize all active markets' questions for O(N) regex performance
    import re
    for m in active_markets:
        m["_tokens"] = set(re.findall(r'[a-z0-9]+', m["question"].lower()))

    # Find cross-platform pairs
    pairs = []
    for i in range(len(active_markets)):
        m1 = active_markets[i]
        for j in range(i + 1, len(active_markets)):
            m2 = active_markets[j]
            # Must be different platforms
            if m1["platform"] == m2["platform"]:
                continue
            if _is_similar_tokens(m1["_tokens"], m2["_tokens"]):
                p1, p2 = m1["market_probability"], m2["market_probability"]
                # Price gap (arbitrage spread)
                gap = abs(p1 - p2)
                if gap >= 0.02: # at least 2 cents difference
                    # Determine which is cheaper
                    if p1 < p2:
                        cheaper = m1
                        dearer = m2
                    else:
                        cheaper = m2
                        dearer = m1
                    
                    p_cheap = cheaper["market_probability"]
                    p_dear = dearer["market_probability"]
                    
                    # Buying YES on cheaper platform, buying NO on dearer platform
                    total_cost = p_cheap + (1.0 - p_dear)
                    
                    if total_cost < 1.0:
                        net_profit = 1.0 - total_cost
                        roi = net_profit / total_cost if total_cost > 0 else 0.0
                        pairs.append({
                            "question1": cheaper["question"],
                            "platform1": cheaper["platform"],
                            "market_url1": cheaper["market_url"],
                            "price1": round(p_cheap, 3),
                            
                            "question2": dearer["question"],
                            "platform2": dearer["platform"],
                            "market_url2": dearer["market_url"],
                            "price2": round(p_dear, 3),
                            
                            "arbitrage_gap": round(gap, 3),
                            "total_cost": round(total_cost, 3),
                            "net_profit": round(net_profit, 3),
                            "roi": round(roi, 4)
                        })
    # Sort by ROI descending
    pairs.sort(key=lambda x: x["roi"], reverse=True)
    return pairs


def aggregate(client, *, model: str, variant: str, temperature: float,
              trajectory_samples: int = 8,
              cycle_minutes: Optional[int] = None,
              tracked_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Recompute the public aggregate (overall + by-horizon) and persist it."""

    con = _sql_con(client)
    if con is not None:
        # DuckDB fast path: two targeted queries instead of a full scan + Python split.
        resolved = _sql_rows(con, """
            SELECT * FROM forecast_snapshot
            WHERE resolved = true AND outcome IS NOT NULL
              AND key NOT LIKE '%_backfill'
            ORDER BY resolved_ts
        """)
        open_rows = _sql_rows(con, """
            SELECT * FROM forecast_snapshot
            WHERE resolved = false
            ORDER BY snapshot_ts DESC
        """)
        # Latest price per ident: arg_max picks the market_probability at max(ts).
        latest_price: Dict[str, float] = {
            r[0]: float(r[1])
            for r in con.execute("""
                SELECT ident, arg_max(market_probability, ts)
                FROM market_price_point
                GROUP BY ident
            """).fetchall()
            if r[0] and r[1] is not None
        }
        latest_quotes: Dict[Tuple[str, str], Dict[str, Any]] = {
            (r[0] or "", r[1] or ""): {
                "platform": r[0],
                "ident": r[1],
                "market_probability": r[2],
                "market_bid": r[3],
                "market_ask": r[4],
                "ts": r[5],
            }
            for r in con.execute("""
                SELECT platform, ident,
                       arg_max(market_probability, ts),
                       arg_max(market_bid, ts),
                       arg_max(market_ask, ts),
                       max(ts)
                FROM market_price_point
                GROUP BY platform, ident
            """).fetchall()
            if r[1]
        }
    else:
        # ORM fallback (FileStore / Datastore)
        resolved = []
        open_rows = []
        for e in client.query(kind=SNAPSHOT_KIND).fetch():
            if _is_backfill_snapshot(e):
                continue
            if e.get("resolved") and e.get("outcome") is not None:
                resolved.append(dict(e))
            else:
                open_rows.append(dict(e))
        latest_price = {}
        latest_quotes = {}
        _latest_price_ts: Dict[str, Any] = {}
        for p in client.query(kind=PRICE_KIND).fetch():
            ident = p.get("ident")
            ts = p.get("ts") or _now()
            if ident not in _latest_price_ts or ts > _latest_price_ts[ident]:
                _latest_price_ts[ident] = ts
                latest_price[ident] = float(p.get("market_probability") or 0.0)
                latest_quotes[(p.get("platform") or "", ident or "")] = {
                    "platform": p.get("platform"),
                    "ident": ident,
                    "market_probability": p.get("market_probability"),
                    "market_bid": p.get("market_bid"),
                    "market_ask": p.get("market_ask"),
                    "ts": ts,
                }

    # Drop stale open snapshots (not re-forecast recently) so orphaned readings
    # from an old ident don't surface on the edge board as live disagreements.
    open_rows = _drop_stale_open(open_rows)
    open_idents = {(r.get("platform"), r.get("ident")) for r in open_rows}

    # The edge board itself is driven by the primary model, but public headline
    # counts and the default equity curve should not double-count the same
    # resolved opportunities across Council, GPT, Gemma, and Kimi. Use the
    # largest single tracked LLM cohort as the comparable headline universe.
    # Snapshots predating the multi-model split have no `model` field → primary.
    resolved_primary = [r for r in resolved if (r.get("model") or model) == model]
    resolved_llm = [r for r in resolved if (r.get("model") or model) != "crowd-follow"]
    resolved_skill = _crowd_baseline_reference_rows(resolved, default_model=model) or resolved_llm
    open_primary = [r for r in open_rows if (r.get("model") or model) == model]
    open_idents = {(r.get("platform"), r.get("ident")) for r in open_primary}
    n_markets_resolved = len({(r.get("platform"), r.get("ident")) for r in resolved_skill})
    primary_n_markets_resolved = len({(r.get("platform"), r.get("ident")) for r in resolved_primary})

    # Skill metrics use this comparable cohort — sports included.

    # Lead-time (how-early) calibration: resolved skill bucketed by forecast
    # horizon, with significance — the axis the edge board links against.
    by_horizon = []
    for label, _lo, _hi in _HORIZONS:
        stats = _bucket_stats([r for r in resolved_skill if r.get("horizon") == label])
        if stats:
            stats["horizon"] = label
            stats.update(_skill_ci([r for r in resolved_skill if r.get("horizon") == label]))
            by_horizon.append(stats)

    # Compute edge board early so the stat can reflect its actual length.
    # Use non-sports resolved snapshots for the skill calibration so sports
    _res_skill_early = [r for r in resolved if (r.get("model") or model) == model]
    by_edge_early = edge_calibration(_res_skill_early)
    edge_board_result = build_edge_board(open_primary, latest_price, by_edge_early, by_horizon)
    mark_to_market_rows_all = resolved + open_rows
    mark_to_market_by_model = build_mark_to_market_accounts(
        mark_to_market_rows_all,
        latest_quotes,
        default_model=model,
        tracked_models=tracked_models,
    )
    half_kelly_20pct_by_model = build_half_kelly_20pct_accounts(
        mark_to_market_rows_all,
        latest_quotes,
        default_model=model,
        tracked_models=tracked_models,
    )
    growth_by_model = build_growth_accounts(
        mark_to_market_rows_all,
        latest_quotes,
        default_model=model,
        tracked_models=tracked_models,
    )
    primary_mark_to_market = next(
        (row.get("account") for row in mark_to_market_by_model if row.get("model") == model),
        None,
    )
    discrepancy_rows = [
        row for row in edge_board_result
        if row.get("needs_discrepancy_review")
    ]
    discrepancy_counts: Dict[str, int] = {}
    for row in discrepancy_rows:
        status = str(row.get("discrepancy_status") or "unknown")
        discrepancy_counts[status] = discrepancy_counts.get(status, 0) + 1

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
        "n_snapshots_resolved": len(resolved_skill),
        "n_markets_resolved": n_markets_resolved,
        "primary_model": model,
        "primary_n_snapshots_resolved": len(resolved_primary),
        "primary_n_markets_resolved": primary_n_markets_resolved,
        "n_markets_open": len(edge_board_result),
        "n_markets_tracked": len(open_idents),
    }

    overall = _bucket_stats(resolved_skill)
    # by_horizon computed above (with significance) so the edge board can link to it.

    # Keep only the latest snapshot for each unique question text (most recently resolved first)
    resolved_sorted = sorted(resolved_skill, key=lambda x: x.get("resolved_ts") or "", reverse=True)
    seen_questions = set()
    _log_rows = []
    for r in resolved_sorted:
        q_text = (r.get("question") or "").strip().lower()
        if not q_text or q_text in seen_questions:
            continue
        seen_questions.add(q_text)
        _log_rows.append(r)
        if len(_log_rows) >= 30:
            break
    resolved_log = []
    for _r in _log_rows:
        _prob = float(_r.get("model_probability") or 0.5)
        _outcome = _r.get("outcome")
        _correct = _r.get("model_correct")
        if _correct is None and _outcome is not None:
            _correct = (_prob >= 0.5) == (_outcome == 1)
        resolved_log.append({
            "correct": bool(_correct),
            "question": _r.get("question") or "",
            "category": _r.get("domain") or _r.get("category") or "",
            "resolve_time": str(_r.get("resolved_ts") or _r.get("close_time") or "")[:10],
            "predicted": "Yes" if _prob >= 0.5 else "No",
            "confidence": round(_prob, 3),
            "actual": "Yes" if _outcome == 1 else "No",
            "model": _r.get("model") or model,
        })

    # Reliability diagram bins (10pp wide) for the calibration chart.
    _calib_bins: List[Dict[str, Any]] = []
    for _lo in (i / 10 for i in range(10)):
        _hi = _lo + 0.1
        _bin = [r for r in resolved_skill if _lo <= float(r.get("model_probability") or 0) < _hi]
        if _bin:
            _calib_bins.append({
                "avg_predicted": round(sum(float(r.get("model_probability") or 0) for r in _bin) / len(_bin), 3),
                "observed_yes_rate": round(sum(1 for r in _bin if r.get("outcome") == 1) / len(_bin), 3),
                "n": len(_bin),
            })

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
        _ident = first.get("ident")
        if con is not None:
            _pp = con.execute(
                "SELECT ts, market_probability, market_volume, market_liquidity,"
                " market_bid, market_ask FROM market_price_point"
                " WHERE ident = ? ORDER BY ts",
                [_ident],
            ).fetchall()
            price_points = [
                {
                    "ts": r[0],
                    "market_probability": r[1],
                    "market_volume": r[2],
                    "market_liquidity": r[3],
                    "market_bid": r[4],
                    "market_ask": r[5],
                }
                for r in _pp
            ]
        else:
            price_q = client.query(kind=PRICE_KIND)
            price_q.add_filter("ident", "=", _ident)
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
    for r in resolved_skill:
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
        rows = [r for r in resolved_skill if lo <= float(r.get("market_volume") or 0.0) < hi]
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
    _crowd_ref_rows = _crowd_baseline_reference_rows(resolved, default_model=model)
    _crowd_base = crowd_baseline_equity(_crowd_ref_rows)

    payload.update({
        "overall": overall,
        "primary_overall": _bucket_stats(resolved_primary),
        "by_horizon": by_horizon,
        "by_edge": by_edge_skill,
        "by_domain": by_domain,
        "by_liquidity": by_liquidity,
        "lead_lag": lead_lag(by_market_skill),
        "paper_pnl": {**(paper_pnl(resolved_skill, by_edge_skill) or {}),
                      "crowd_baseline": _crowd_base},
        "primary_paper_pnl": paper_pnl(resolved_primary, edge_calibration(resolved_primary)),
        "mark_to_market_account": primary_mark_to_market,
        "mark_to_market_by_model": mark_to_market_by_model,
        "half_kelly_20pct_by_model": half_kelly_20pct_by_model,
        "growth_1pct_by_model": growth_by_model["growth_1pct"],
        "growth_2pct_by_model": growth_by_model["growth_2pct"],
        "mark_to_market_cycle_minutes": cycle_minutes,
        "edge_board": edge_board_result,
        "discrepancy_monitor": {
            "huge_gap_threshold": 0.20,
            "n_huge_gaps": len(discrepancy_rows),
            "by_status": discrepancy_counts,
            "market_keys": [
                f"{row.get('platform')}:{row.get('ident')}"
                for row in discrepancy_rows
            ],
        },
        "arbitrage_signals": build_arbitrage_board(open_primary, latest_price),
        "models_comparison": build_models_comparison(
            [r for r in resolved if (r.get("model") or model) in _RESOLVED_QUALITY_MODELS],
            default_model=model,
            crowd_baseline=_crowd_base,
            crowd_reference_rows=_crowd_ref_rows,
        ),
        "trajectories": trajectories,
        "resolved_log": resolved_log,
        "log_sample_size": len(resolved_log),
        "calibration": _calib_bins,
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
