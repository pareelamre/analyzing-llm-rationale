"""Time-series downsampling and compression algorithms for long-tail forecasting data.

Implements Largest-Triangle-Three-Buckets (LTTB) and time-decay bucketed downsampling
to ensure time-series payloads (equity curves, mark-to-market history) remain O(1) in size.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


def downsample_lttb(
    data: Sequence[Tuple[float, float]],
    threshold: int,
) -> List[Tuple[float, float]]:
    """Downsample a 2D time-series using the Largest-Triangle-Three-Buckets (LTTB) algorithm.

    Preserves peaks, valleys, and volatility dynamics while reducing point count.
    `data` must be a sequence of (timestamp_or_x, value_or_y) sorted by x.
    """
    data_len = len(data)
    if threshold >= data_len or threshold <= 2:
        return list(data)

    sampled: List[Tuple[float, float]] = []
    bucket_size = (data_len - 2) / (threshold - 2)

    # First point is always selected
    a = 0
    sampled.append(data[a])

    for i in range(threshold - 2):
        # Calculate point average for next bucket (range c)
        avg_x = 0.0
        avg_y = 0.0
        avg_range_start = int(math.floor((i + 1) * bucket_size)) + 1
        avg_range_end = int(math.floor((i + 2) * bucket_size)) + 1
        avg_range_end = min(avg_range_end, data_len)

        avg_range_len = avg_range_end - avg_range_start
        if avg_range_len > 0:
            for j in range(avg_range_start, avg_range_end):
                avg_x += data[j][0]
                avg_y += data[j][1]
            avg_x /= avg_range_len
            avg_y /= avg_range_len
        else:
            avg_x = data[min(avg_range_start, data_len - 1)][0]
            avg_y = data[min(avg_range_start, data_len - 1)][1]

        # Get the range for this bucket (range b)
        range_offs = int(math.floor(i * bucket_size)) + 1
        range_to = int(math.floor((i + 1) * bucket_size)) + 1
        range_to = min(range_to, data_len)

        # Point a
        point_a_x = data[a][0]
        point_a_y = data[a][1]

        max_area = -1.0
        next_a = range_offs

        for j in range(range_offs, range_to):
            # Calculate triangle area over points a, data[j], and average point
            area = abs(
                (point_a_x - avg_x) * (data[j][1] - point_a_y)
                - (point_a_x - data[j][0]) * (avg_y - point_a_y)
            ) * 0.5
            if area > max_area:
                max_area = area
                next_a = j

        sampled.append(data[next_a])
        a = next_a

    # Last point is always selected
    sampled.append(data[data_len - 1])
    return sampled


def _parse_ts(val: Any) -> float:
    """Parse integer, float, or ISO-8601 string timestamp to float epoch seconds."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            pass
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise ValueError(f"Unparseable timestamp: {val!r}")


def downsample_time_decay(
    records: Sequence[Dict[str, Any]],
    *,
    ts_key: str = "timestamp",
    val_key: str = "value",
    now_ts: Optional[float] = None,
    recent_window_s: float = 86400.0,       # 24 hours: full fidelity (up to 1440 pts)
    mid_window_s: float = 2592000.0,         # 30 days: 15-minute buckets
    mid_bucket_s: float = 900.0,             # 15 minutes
    old_bucket_s: float = 86400.0,           # 1 day for older than 30 days
) -> List[Dict[str, Any]]:
    """Downsample time-series dictionaries with decaying resolution as age increases.

    - Records < 24h old: Retained as-is (100% fidelity).
    - Records 24h - 30d old: Sampled at 15-minute intervals.
    - Records > 30d old: Sampled at 1-day intervals.
    """
    if not records:
        return []

    if now_ts is None:
        now_ts = datetime.now(timezone.utc).timestamp()

    # Parse and sort by timestamp
    parsed: List[Tuple[float, Dict[str, Any]]] = []
    for r in records:
        try:
            t = _parse_ts(r.get(ts_key))
            parsed.append((t, r))
        except Exception:
            parsed.append((now_ts, r))

    parsed.sort(key=lambda x: x[0])

    result: List[Dict[str, Any]] = []
    last_mid_bucket = -1
    last_old_bucket = -1

    for t, rec in parsed:
        age_s = now_ts - t
        if age_s <= recent_window_s:
            # Recent: keep every point
            result.append(rec)
        elif age_s <= mid_window_s:
            # Intermediate age: 15-minute bucketing
            bucket_idx = int(t // mid_bucket_s)
            if bucket_idx != last_mid_bucket:
                result.append(rec)
                last_mid_bucket = bucket_idx
        else:
            # Long-tail age: 1-day bucketing
            bucket_idx = int(t // old_bucket_s)
            if bucket_idx != last_old_bucket:
                result.append(rec)
                last_old_bucket = bucket_idx

    # Ensure the latest record is preserved if not already in result
    if parsed and (not result or result[-1] != parsed[-1][1]):
        result.append(parsed[-1][1])

    return result
