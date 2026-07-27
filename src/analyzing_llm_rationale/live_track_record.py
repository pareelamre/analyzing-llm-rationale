from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import requests

CacheKeyFn = Callable[[str, str], str]
CacheGetFn = Callable[[str], Any]
CacheSetFn = Callable[[str, Any, int], None]
RequestsGetFn = Callable[..., Any]
TimeFn = Callable[[], float]


@dataclass(frozen=True)
class LiveTrackRecordConfig:
    live_url: str
    ttl_seconds: int
    stale_after_seconds: int
    bundled_path: Path
    cache_namespace: str = "track_record_live"
    cache_version: str = "v3"
    resource_label: str = "live track record"
    user_agent: str = "Foresea/edge-board-live"


class LiveTrackRecordReader:
    """Read the committed live track-record aggregate with shared cache hooks."""

    def __init__(
        self,
        *,
        cache_key: CacheKeyFn,
        cache_get: CacheGetFn,
        cache_set: CacheSetFn,
        config: LiveTrackRecordConfig,
        logger: Any,
        requests_get: RequestsGetFn = requests.get,
        time_fn: TimeFn = time.time,
    ) -> None:
        self._cache_key = cache_key
        self._cache_get = cache_get
        self._cache_set = cache_set
        self._config = config
        self._logger = logger
        self._requests_get = requests_get
        self._time = time_fn

    def read(self) -> Optional[Dict[str, Any]]:
        """Return the committed live track-record aggregate, or None.

        Tries (cached): raw GitHub copy -> bundled file. Synchronous; call via
        ``run_in_executor`` from async handlers. Fails open to None so callers
        can fall back to the static backtest.
        """
        cache_key = self._cache_key(
            self._config.cache_namespace,
            self._config.cache_version,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        payload: Optional[Dict[str, Any]] = None
        try:
            ttl = max(self._config.ttl_seconds, 1)
            sep = "&" if "?" in self._config.live_url else "?"
            cache_busted_url = f"{self._config.live_url}{sep}_={int(self._time() // ttl)}"
            resp = self._requests_get(
                cache_busted_url,
                timeout=6,
                headers={
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                    "User-Agent": self._config.user_agent,
                },
            )
            if getattr(resp, "status_code", None) == 200:
                payload = resp.json()
        except Exception:
            self._logger.warning(
                f"{self._config.resource_label} fetch failed; trying bundled copy",
                exc_info=True,
            )

        if payload is None:
            payload = self._read_bundled()

        if payload is not None:
            self._cache_set(cache_key, payload, self._config.ttl_seconds)
        return payload

    def freshness(
        self,
        payload: Optional[Dict[str, Any]],
        *,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        generated_at = (payload or {}).get("generated_at")
        age_seconds = None
        if generated_at:
            try:
                dt = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ref = now or datetime.now(timezone.utc)
                age_seconds = max(0, int((ref - dt).total_seconds()))
            except Exception:
                age_seconds = None
        stale = age_seconds is None or age_seconds > self._config.stale_after_seconds
        return {
            "generated_at": generated_at,
            "age_seconds": age_seconds,
            "stale": stale,
            "stale_after_seconds": self._config.stale_after_seconds,
        }

    def _read_bundled(self) -> Optional[Dict[str, Any]]:
        bundled = self._config.bundled_path
        if not bundled.exists():
            return None
        try:
            return json.loads(bundled.read_text())
        except Exception:
            self._logger.warning(
                f"bundled {self._config.resource_label} unreadable",
                exc_info=True,
            )
            return None


def strategy_filter_edge_entry(entry: Dict[str, Any], strategy: str) -> bool:
    """Apply a paper-PnL strategy's filter logic to a live edge board entry."""
    entry_price = entry.get("entry_price", 0.5)
    abs_edge = entry.get("abs_edge", 0.0)
    domain = entry.get("domain", "")
    if strategy == "smart":
        if entry_price < 0.20 or entry_price > 0.80:
            return False
        if domain == "geopolitics" and abs_edge > 0.10:
            return False
        if abs_edge > 0.40:
            return False
        return True
    return True


def pick_best_strategy(paper_pnl: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return the highest-ROI strategy with at least 20 resolved bets."""
    industry_grade = {"smart", "half_kelly", "flat", "crowd_baseline"}
    candidates = [
        (name, data)
        for name, data in paper_pnl.items()
        if name in industry_grade
        and isinstance(data, dict)
        and data.get("roi") is not None
        and (data.get("n_bets") or 0) >= 20
    ]
    if not candidates:
        return ("flat", paper_pnl.get("flat") or {})
    return max(candidates, key=lambda item: item[1]["roi"])


def edge_board_order_context(track_record: Dict[str, Any]) -> str:
    """Format top live edge-board picks as chat context."""
    paper_pnl = track_record.get("paper_pnl") or {}
    edge_board = track_record.get("edge_board") or []
    if not edge_board or not paper_pnl:
        return ""

    strategy_name, strategy_data = pick_best_strategy(paper_pnl)
    filtered = [
        entry
        for entry in edge_board
        if strategy_filter_edge_entry(entry, strategy_name)
    ]
    filtered.sort(key=lambda entry: entry.get("abs_edge", 0.0), reverse=True)
    if not filtered:
        return ""

    roi_pct = (
        f"{strategy_data['roi']:.1%}"
        if strategy_data.get("roi") is not None
        else "n/a"
    )
    n_bets = strategy_data.get("n_bets", "?")
    lines = [
        "## Live order recommendations",
        f"Best back-tested strategy: **{strategy_name}** "
        f"(historical ROI {roi_pct} over {n_bets} resolved bets, paper only).",
        "",
    ]
    for index, entry in enumerate(filtered[:10], 1):
        sig = entry.get("track_record") or {}
        proven = "proven" if sig.get("skill_significant") else "unproven"
        model_p = f"{entry.get('model_probability', 0):.0%}"
        market_p = f"{entry.get('market_probability', 0):.0%}"
        edge_pct = f"{entry.get('abs_edge', 0):.0%}"
        payout = entry.get("payout_odds", "?")
        lines.append(
            f"{index}. **{entry.get('question', '')}** [{entry.get('platform', '')}]  "
            f"Bet {entry.get('side', '?')} @ {market_p} | Model {model_p} | "
            f"Edge {edge_pct} | {payout}x payout | {proven}  "
            f"{entry.get('market_url', '')}"
        )
    lines.append(
        "\nAll figures are paper/hypothetical. Entry prices are live at last tick."
    )
    return "\n".join(lines)
