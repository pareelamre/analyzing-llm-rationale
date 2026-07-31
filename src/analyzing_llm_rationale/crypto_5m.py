"""Conservative 5-minute crypto UP/DOWN probability model.

This is a market micro-model, not an LLM forecast. It prices the probability
that the selected symbol finishes above a target price after a short horizon
using recent 1-minute Binance candles:

    P(S_T > K) under a shrunken drift + realized-volatility log-return model.

For prediction-market UP/DOWN contracts, pass the contract's start/strike price
as ``target_price``. If omitted, the current price is used, which answers
"will the next 5 minutes be up from right now?".
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_KLINES_FALLBACK_URL = "https://api.binance.com/api/v3/klines"
_HEADERS = {"User-Agent": "foresea-crypto-5m/0.1"}
_TIMEOUT_S = 10
_ONE_MINUTE_MS = 60_000
DEFAULT_BENCHMARK_LOG_PATH = Path("data/crypto_5m_benchmark_runs.jsonl")
DEFAULT_SIGNAL_LOG_PATH = Path("data/crypto_5m_signal_log.jsonl")
DEFAULT_SIGNAL_DB_PATH = Path("data/crypto_5m_signals.sqlite")
DEFAULT_EQUITY_FALLBACK_PATH = Path("static/crypto_5m_equity_payload.json")
DEFAULT_EQUITY_REMOTE_URL = (
    "https://raw.githubusercontent.com/pareelamre/analyzing-llm-rationale/"
    "main/static/crypto_5m_equity_payload.json"
)
_ALLOWED_QUOTES = ("USDT", "USDC", "FDUSD", "USD")
_FEATURE_NAMES = (
    "momentum_3m_z",
    "momentum_5m_z",
    "momentum_15m_z",
    "reversal_1m_z",
    "vol_regime_z",
    "range_position",
    "volume_imbalance",
)


def _cache_busted_url(url: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}_ts={int(time.time() * 1000)}"


class CryptoModelError(RuntimeError):
    """Raised when live crypto data cannot be fetched or modelled."""


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def normalize_symbol(symbol: str) -> str:
    """Normalize common symbols to Binance spot symbols."""
    raw = (symbol or "").strip().upper().replace("-", "").replace("/", "")
    if not raw:
        raise CryptoModelError("Provide a crypto symbol, e.g. BTCUSDT or BTC.")
    for quote in _ALLOWED_QUOTES:
        if raw.endswith(quote) and len(raw) > len(quote):
            return raw
    return f"{raw}USDT"


def _fetch_klines(
    symbol: str,
    *,
    limit: int,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> List[Any]:
    import requests

    params = {"symbol": symbol, "interval": "1m", "limit": max(20, min(int(limit), 1000))}
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    errors: List[str] = []
    for url in (BINANCE_KLINES_URL, BINANCE_KLINES_FALLBACK_URL):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            continue
        if resp.status_code != 200:
            errors.append(f"{url} returned HTTP {resp.status_code}")
            continue
        try:
            payload = resp.json()
        except ValueError as exc:
            errors.append(f"{url} returned invalid JSON: {exc}")
            continue
        if not isinstance(payload, list):
            errors.append(f"{url} returned a non-list payload")
            continue
        return payload
    raise CryptoModelError("Could not fetch Binance klines: " + "; ".join(errors))


def fetch_klines_history(
    symbol: str = "BTCUSDT",
    *,
    days: float = 1.0,
    end_time_ms: Optional[int] = None,
    max_candles: int = 10_000,
) -> List[Any]:
    """Fetch paginated 1-minute klines for a recent historical window."""
    symbol_norm = normalize_symbol(symbol)
    max_rows = max(20, min(int(max_candles or 10_000), 50_000))
    window_ms = max(_ONE_MINUTE_MS * 20, int(float(days or 1.0) * 24 * 60 * _ONE_MINUTE_MS))
    end_ms = int(end_time_ms) if end_time_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - window_ms
    rows: List[Any] = []
    seen_open_times = set()
    cursor = start_ms
    while cursor < end_ms and len(rows) < max_rows:
        batch = _fetch_klines(
            symbol_norm,
            limit=min(1000, max_rows - len(rows)),
            start_time_ms=cursor,
            end_time_ms=end_ms,
        )
        if not batch:
            break
        last_open_time = None
        added = 0
        for row in batch:
            if not isinstance(row, (list, tuple)) or not row:
                continue
            open_time = _to_float(row[0])
            if open_time is None:
                continue
            open_ms = int(open_time)
            if open_ms in seen_open_times:
                continue
            seen_open_times.add(open_ms)
            rows.append(row)
            added += 1
            last_open_time = open_ms
            if len(rows) >= max_rows:
                break
        if last_open_time is None or added == 0:
            break
        next_cursor = last_open_time + _ONE_MINUTE_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
    return rows


def _parse_closes(klines: Iterable[Any]) -> List[float]:
    closes: List[float] = []
    for row in klines:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        close = _to_float(row[4])
        if close is not None and close > 0.0:
            closes.append(close)
    return closes


def _parse_candles(klines: Iterable[Any]) -> List[Dict[str, float]]:
    candles: List[Dict[str, float]] = []
    for row in klines:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        open_price = _to_float(row[1])
        high = _to_float(row[2])
        low = _to_float(row[3])
        close = _to_float(row[4])
        volume = _to_float(row[5])
        if (
            open_price is None or high is None or low is None
            or close is None or volume is None
            or open_price <= 0.0 or high <= 0.0 or low <= 0.0 or close <= 0.0
        ):
            continue
        candles.append({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": max(volume, 0.0),
        })
    return candles


def _parse_timed_candles(klines: Iterable[Any]) -> List[Dict[str, float]]:
    candles: List[Dict[str, float]] = []
    for row in klines:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        open_time = _to_float(row[0])
        open_price = _to_float(row[1])
        high = _to_float(row[2])
        low = _to_float(row[3])
        close = _to_float(row[4])
        volume = _to_float(row[5])
        close_time = _to_float(row[6]) if len(row) > 6 else None
        if (
            open_time is None or open_price is None or high is None or low is None
            or close is None or volume is None
            or open_price <= 0.0 or high <= 0.0 or low <= 0.0 or close <= 0.0
        ):
            continue
        candles.append({
            "open_time_ms": float(open_time),
            "close_time_ms": float(close_time) if close_time is not None else float(open_time) + _ONE_MINUTE_MS - 1,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": max(volume, 0.0),
        })
    return candles


def _ott_var_ma(closes: List[float], length: int) -> List[float]:
    """Variable index dynamic average used by the Pine OTT default."""
    period = max(1, int(length or 1))
    alpha = 2.0 / (period + 1.0)
    ma_values: List[float] = []
    up_moves: List[float] = []
    down_moves: List[float] = []
    prev_ma = 0.0
    prev_close: Optional[float] = None
    for close in closes:
        if prev_close is None:
            up = 0.0
            down = 0.0
        elif close > prev_close:
            up = close - prev_close
            down = 0.0
        elif close < prev_close:
            up = 0.0
            down = prev_close - close
        else:
            up = 0.0
            down = 0.0
        up_moves.append(up)
        down_moves.append(down)
        vud = sum(up_moves[-9:])
        vdd = sum(down_moves[-9:])
        cmo = ((vud - vdd) / (vud + vdd)) if (vud + vdd) else 0.0
        weight = alpha * abs(cmo)
        prev_ma = weight * close + (1.0 - weight) * prev_ma
        ma_values.append(prev_ma)
        prev_close = close
    return ma_values


def compute_ott(
    candles: List[Dict[str, float]],
    *,
    length: int = 2,
    percent: float = 1.4,
) -> List[Dict[str, Any]]:
    """Compute the Pine v4 OTT trend state over parsed 1-minute candles.

    This ports only the default VAR-based OTT calculation. Plotting, alerts,
    strategy orders, and screener labels stay out of the research code.
    """
    if not candles:
        return []
    closes = [float(candle["close"]) for candle in candles]
    ma_values = _ott_var_ma(closes, max(1, int(length or 1)))
    pct = max(0.0, float(percent or 0.0)) * 0.01
    direction = 1
    long_stop_prev: Optional[float] = None
    short_stop_prev: Optional[float] = None
    out: List[Dict[str, Any]] = []
    for index, ma_value in enumerate(ma_values):
        fark = ma_value * pct
        long_stop = ma_value - fark
        short_stop = ma_value + fark
        if long_stop_prev is None:
            long_stop_prev = long_stop
        if short_stop_prev is None:
            short_stop_prev = short_stop
        if ma_value > long_stop_prev:
            long_stop = max(long_stop, long_stop_prev)
        if ma_value < short_stop_prev:
            short_stop = min(short_stop, short_stop_prev)
        prev_direction = direction
        if direction == -1 and ma_value > short_stop_prev:
            direction = 1
        elif direction == 1 and ma_value < long_stop_prev:
            direction = -1
        mt = long_stop if direction == 1 else short_stop
        ott = mt * (200.0 + percent) / 200.0 if ma_value > mt else mt * (200.0 - percent) / 200.0
        out.append({
            "index": index,
            "support": round(ma_value, 8),
            "ott": round(ott, 8),
            "trend": direction,
            "trend_side": "up" if direction == 1 else "down",
            "reversal": index > 0 and direction != prev_direction,
        })
        long_stop_prev = long_stop
        short_stop_prev = short_stop
    return out


def _log_returns(closes: List[float]) -> List[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


def resolve_crypto_5m_outcome(
    symbol: str = "BTCUSDT",
    *,
    target_price: float,
    start_time_ms: Optional[int] = None,
    expiry_time_ms: Optional[int] = None,
    horizon_minutes: int = 5,
    predicted_outcome: Optional[str] = None,
    klines: Optional[List[Any]] = None,
    now_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve a 5-minute UP/DOWN market from 1-minute Binance candles.

    `actual_outcome` is based on the final candle close at expiry. If expiry is
    in the future, the result is `pending`; the true outcome is not knowable yet.
    """
    symbol_norm = normalize_symbol(symbol)
    if target_price <= 0:
        raise CryptoModelError("target_price must be positive.")
    horizon = max(1, min(int(horizon_minutes or 5), 15))
    if expiry_time_ms is None:
        if start_time_ms is None:
            raise CryptoModelError("Provide start_time_ms or expiry_time_ms to resolve a 5-minute market.")
        expiry_ms = int(start_time_ms) + horizon * _ONE_MINUTE_MS
    else:
        expiry_ms = int(expiry_time_ms)
    start_ms = int(start_time_ms) if start_time_ms is not None else expiry_ms - horizon * _ONE_MINUTE_MS
    current_ms = int(now_ms) if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    if current_ms < expiry_ms:
        return {
            "symbol": symbol_norm,
            "status": "pending",
            "start_time_ms": start_ms,
            "expiry_time_ms": expiry_ms,
            "target_price": round(float(target_price), 8),
            "actual_outcome": None,
            "resolved_price": None,
            "predicted_outcome": predicted_outcome,
            "prediction_correct": None,
            "reason": "Market has not reached expiry yet.",
        }

    raw = klines if klines is not None else _fetch_klines(
        symbol_norm,
        limit=max(20, horizon + 3),
        start_time_ms=start_ms,
        end_time_ms=expiry_ms,
    )
    timed = _parse_timed_candles(raw)
    if not timed:
        raise CryptoModelError("No valid candles available for outcome resolution.")
    final_candle = None
    for candle in timed:
        if candle["close_time_ms"] <= expiry_ms:
            if final_candle is None or candle["close_time_ms"] > final_candle["close_time_ms"]:
                final_candle = candle
    if final_candle is None:
        raise CryptoModelError("No completed candle found at or before expiry.")

    resolved_price = final_candle["close"]
    if math.isclose(resolved_price, float(target_price), rel_tol=0.0, abs_tol=1e-12):
        actual = "tie"
    else:
        actual = "up" if resolved_price > float(target_price) else "down"
    normalized_prediction = (predicted_outcome or "").strip().lower() or None
    prediction_correct = None
    if normalized_prediction in {"up", "down", "tie"}:
        prediction_correct = normalized_prediction == actual
    return {
        "symbol": symbol_norm,
        "status": "resolved",
        "start_time_ms": start_ms,
        "expiry_time_ms": expiry_ms,
        "target_price": round(float(target_price), 8),
        "resolved_price": round(resolved_price, 8),
        "actual_outcome": actual,
        "predicted_outcome": normalized_prediction,
        "prediction_correct": prediction_correct,
        "resolution_candle": {
            "open_time_ms": int(final_candle["open_time_ms"]),
            "close_time_ms": int(final_candle["close_time_ms"]),
            "open": round(final_candle["open"], 8),
            "high": round(final_candle["high"], 8),
            "low": round(final_candle["low"], 8),
            "close": round(final_candle["close"], 8),
        },
    }


def _normalize_market_probability(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    p = float(value)
    if p > 1.0:
        p = p / 100.0
    return max(0.0, min(1.0, p))


def _safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _robust_z(value: float, scale: float) -> float:
    return _clip(value / max(scale, 1e-9), -4.0, 4.0)


def _ewma_volatility(values: List[float], *, decay: float = 0.94) -> float:
    if not values:
        return 1e-6
    variance = values[0] * values[0]
    lam = _clip(decay, 0.5, 0.99)
    for value in values[1:]:
        variance = lam * variance + (1.0 - lam) * value * value
    return max(math.sqrt(max(variance, 0.0)), 1e-6)


def _ar1_ewma_probability(
    returns: List[float],
    *,
    current: float,
    target: float,
    horizon: int,
) -> tuple[float, Dict[str, float]]:
    """Short-horizon AR(1) return forecast with EWMA innovation volatility."""
    window = returns[-min(240, len(returns)):]
    if len(window) < 20:
        return 0.5, {
            "ts_ar1_phi": 0.0,
            "ts_mean_return_1m": 0.0,
            "ts_expected_log_move": 0.0,
            "ts_ewma_vol_1m": 1e-6,
            "ts_z_score": 0.0,
        }

    mu = mean(window)
    xs = [value - mu for value in window[:-1]]
    ys = [value - mu for value in window[1:]]
    denom = sum(x * x for x in xs)
    phi = sum(x * y for x, y in zip(xs, ys)) / denom if denom > 1e-12 else 0.0
    phi = _clip(phi, -0.65, 0.65)
    residuals = [
        window[i] - (mu + phi * (window[i - 1] - mu))
        for i in range(1, len(window))
    ]
    sigma = _ewma_volatility(residuals or window)
    last_deviation = window[-1] - mu
    expected_log_move = 0.0
    variance_scale = 0.0
    for step in range(1, max(1, horizon) + 1):
        expected_log_move += mu + (phi ** step) * last_deviation
        variance_scale += step
    # Inflate a little because 5-minute crypto residuals are heavy-tailed.
    sd_move = sigma * math.sqrt(max(variance_scale, 1.0)) * 1.15
    z = (math.log(current / target) + expected_log_move) / max(sd_move, 1e-9)
    probability = max(0.01, min(0.99, _normal_cdf(z)))
    return probability, {
        "ts_ar1_phi": round(phi, 6),
        "ts_mean_return_1m": round(mu, 8),
        "ts_expected_log_move": round(expected_log_move, 8),
        "ts_ewma_vol_1m": round(sigma, 8),
        "ts_z_score": round(z, 4),
    }


def _ml_signal_features(candles: List[Dict[str, float]], returns: List[float], sigma_1m: float) -> Dict[str, float]:
    """Feature set for the short-horizon logistic overlay.

    The coefficients below are intentionally fixed and conservative rather than
    fitted in-process. They encode common intraday crypto effects: short
    momentum, weaker mean reversion after sharp bursts, volatility regime, range
    position, and directional volume imbalance.
    """
    closes = [c["close"] for c in candles]
    current = closes[-1]
    momentum_3m = math.log(closes[-1] / closes[-4]) if len(closes) >= 4 else 0.0
    momentum_5m = math.log(closes[-1] / closes[-6]) if len(closes) >= 6 else 0.0
    momentum_15m = math.log(closes[-1] / closes[-16]) if len(closes) >= 16 else momentum_5m
    reversal_1m = -(returns[-1] if returns else 0.0)
    recent_returns = returns[-30:] if len(returns) >= 30 else returns
    older_returns = returns[-120:-30] if len(returns) >= 120 else returns[:-30]
    recent_vol = max(pstdev(recent_returns), 1e-6) if len(recent_returns) >= 2 else sigma_1m
    older_vol = max(pstdev(older_returns), 1e-6) if len(older_returns) >= 2 else sigma_1m
    vol_regime = math.log(recent_vol / older_vol)
    range_window = candles[-30:] if len(candles) >= 30 else candles
    range_high = max(c["high"] for c in range_window)
    range_low = min(c["low"] for c in range_window)
    range_pos = ((current - range_low) / max(range_high - range_low, 1e-9)) - 0.5

    signed_volume = 0.0
    total_volume = 0.0
    for c in candles[-20:]:
        body = c["close"] - c["open"]
        signed_volume += (1.0 if body > 0 else -1.0 if body < 0 else 0.0) * c["volume"]
        total_volume += c["volume"]
    volume_imbalance = signed_volume / max(total_volume, 1e-9)

    return {
        "momentum_3m_z": _robust_z(momentum_3m, sigma_1m * math.sqrt(3.0)),
        "momentum_5m_z": _robust_z(momentum_5m, sigma_1m * math.sqrt(5.0)),
        "momentum_15m_z": _robust_z(momentum_15m, sigma_1m * math.sqrt(15.0)),
        "reversal_1m_z": _robust_z(reversal_1m, sigma_1m),
        "vol_regime_z": _clip(vol_regime, -3.0, 3.0),
        "range_position": _clip(range_pos, -0.5, 0.5),
        "volume_imbalance": _clip(volume_imbalance, -1.0, 1.0),
    }


def _ml_probability(features: Dict[str, float]) -> tuple[float, float]:
    score = _fixed_ml_score(features)
    # Temperature calibration keeps the overlay from dominating the volatility
    # model. It can still move the fair value when multiple features agree.
    calibrated_score = 0.42 * score
    return _sigmoid(calibrated_score), calibrated_score


def _fixed_ml_score(features: Dict[str, float]) -> float:
    return (
        0.34 * features["momentum_3m_z"]
        + 0.24 * features["momentum_5m_z"]
        - 0.10 * features["momentum_15m_z"]
        + 0.16 * features["reversal_1m_z"]
        - 0.11 * features["vol_regime_z"]
        + 0.12 * features["range_position"]
        + 0.18 * features["volume_imbalance"]
    )


def _feature_vector(features: Dict[str, float]) -> List[float]:
    return [float(features.get(name, 0.0)) for name in _FEATURE_NAMES]


def _logistic_predict(weights: List[float], vector: List[float]) -> float:
    score = weights[0]
    for weight, value in zip(weights[1:], vector):
        score += weight * value
    return _sigmoid(score)


def _fit_logistic(
    vectors: List[List[float]],
    labels: List[int],
    *,
    epochs: int = 180,
    learning_rate: float = 0.05,
    l2: float = 0.35,
) -> List[float]:
    if not vectors:
        return [0.0] * (len(_FEATURE_NAMES) + 1)
    base_rate = _clip(sum(labels) / len(labels), 0.02, 0.98)
    weights = [math.log(base_rate / (1.0 - base_rate))] + [0.0] * len(_FEATURE_NAMES)
    n = float(len(vectors))
    for _ in range(max(1, epochs)):
        grads = [0.0] * len(weights)
        for vector, label in zip(vectors, labels):
            pred = _logistic_predict(weights, vector)
            err = pred - label
            grads[0] += err
            for j, value in enumerate(vector, start=1):
                grads[j] += err * value
        for j in range(len(weights)):
            penalty = 0.0 if j == 0 else l2 * weights[j]
            weights[j] -= learning_rate * ((grads[j] / n) + penalty / n)
    return weights


def _score_probability_rows(preds: List[float], labels: List[int]) -> Dict[str, Any]:
    if not preds:
        return {"n": 0, "brier_score": None, "log_loss": None, "accuracy": None}
    eps = 1e-6
    brier = sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(preds)
    log_loss = sum(
        -(y * math.log(_clip(p, eps, 1.0 - eps))
          + (1 - y) * math.log(_clip(1.0 - p, eps, 1.0 - eps)))
        for p, y in zip(preds, labels)
    ) / len(preds)
    accuracy = _safe_mean([
        1.0 if (p >= 0.5) == (y == 1) else 0.0
        for p, y in zip(preds, labels)
    ])
    return {
        "n": len(preds),
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "accuracy": round(accuracy, 4),
    }


def _build_ml_training_examples(
    candles: List[Dict[str, float]],
    *,
    horizon: int,
    min_history: int,
    max_examples: int,
) -> tuple[List[List[float]], List[int]]:
    stop = len(candles) - horizon
    if stop <= min_history:
        return [], []
    start = max(min_history - 1, stop - max(1, max_examples))
    vectors: List[List[float]] = []
    labels: List[int] = []
    for i in range(start, stop):
        history = candles[:i + 1]
        closes = [c["close"] for c in history]
        returns = _log_returns(closes)
        if len(returns) < 19:
            continue
        sigma_1m = max(pstdev(returns[-min(180, len(returns)):]), 1e-6)
        features = _ml_signal_features(history, returns, sigma_1m)
        vectors.append(_feature_vector(features))
        labels.append(1 if candles[i + horizon]["close"] > candles[i]["close"] else 0)
    return vectors, labels


def _adaptive_ml_probability(
    candles: List[Dict[str, float]],
    current_features: Dict[str, float],
    *,
    horizon: int,
    lookback: int,
    training_window: int,
    include_validation: bool = True,
) -> tuple[float, float, Dict[str, Any]]:
    vectors, labels = _build_ml_training_examples(
        candles,
        horizon=horizon,
        min_history=max(30, min(lookback, 240)),
        max_examples=max(40, min(training_window, 900)),
    )
    return _adaptive_ml_probability_from_examples(
        current_features,
        vectors,
        labels,
        include_validation=include_validation,
    )


def _adaptive_ml_probability_from_examples(
    current_features: Dict[str, float],
    vectors: List[List[float]],
    labels: List[int],
    *,
    include_validation: bool = True,
) -> tuple[float, float, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "ml_mode": "adaptive",
        "ml_training_examples": len(labels),
        "ml_feature_names": list(_FEATURE_NAMES),
    }
    if len(labels) < 40 or len(set(labels)) < 2:
        probability, score = _ml_probability(current_features)
        diagnostics.update({
            "ml_mode_effective": "fixed_fallback",
            "ml_fallback_reason": "not enough balanced training examples",
        })
        return probability, score, diagnostics

    split = int(len(labels) * 0.75)
    if include_validation and len(labels) >= 80 and 20 <= split < len(labels):
        validation_weights = _fit_logistic(vectors[:split], labels[:split])
        validation_preds = [_logistic_predict(validation_weights, v) for v in vectors[split:]]
        diagnostics["ml_validation"] = _score_probability_rows(validation_preds, labels[split:])

    weights = _fit_logistic(vectors, labels)
    vector = _feature_vector(current_features)
    raw_probability = _logistic_predict(weights, vector)
    # Shrink toward 50% because recent 1m candle history is short and nonstationary.
    probability = 0.5 + 0.55 * (raw_probability - 0.5)
    score = math.log(_clip(probability, 1e-6, 1.0 - 1e-6) / _clip(1.0 - probability, 1e-6, 1.0))
    diagnostics.update({
        "ml_mode_effective": "adaptive",
        "ml_base_up_rate": round(sum(labels) / len(labels), 4),
        "ml_coefficients": {
            "intercept": round(weights[0], 5),
            **{name: round(value, 5) for name, value in zip(_FEATURE_NAMES, weights[1:])},
        },
        "ml_train": _score_probability_rows(
            [_logistic_predict(weights, v) for v in vectors],
            labels,
        ),
    })
    return _clip(probability, 0.01, 0.99), score, diagnostics


def _precompute_ml_training_examples(
    candles: List[Dict[str, float]],
    *,
    horizon: int,
    min_history: int,
) -> List[tuple[int, List[float], int]]:
    """Build one labelled ML example per candle index for walk-forward reuse."""
    stop = len(candles) - horizon
    if stop <= min_history:
        return []
    examples: List[tuple[int, List[float], int]] = []
    closes: List[float] = []
    returns: List[float] = []
    for i, candle in enumerate(candles[:stop]):
        close = candle["close"]
        if closes:
            returns.append(math.log(close / closes[-1]))
        closes.append(close)
        if i < min_history - 1 or len(returns) < 19:
            continue
        sigma_1m = max(pstdev(returns[-min(180, len(returns)):]), 1e-6)
        features = _ml_signal_features(candles[:i + 1], returns, sigma_1m)
        label = 1 if candles[i + horizon]["close"] > candles[i]["close"] else 0
        examples.append((i, _feature_vector(features), label))
    return examples


def _strategy_from_edge(
    edge: Optional[float],
    probability_up: float,
    sigma_1m: float,
    horizon: float,
    edge_threshold: float = 0.03,
    fee_bps: float = 0.0,
) -> Dict[str, Any]:
    fee = max(0.0, float(fee_bps or 0.0)) / 10000.0
    if edge is None:
        return {
            "side": "none",
            "recommendation": "no_market_price",
            "confidence": "none",
            "kelly_fraction": 0.0,
            "max_position_fraction": 0.0,
            "expected_value_per_contract": None,
            "fee_bps": fee_bps,
            "reason": "No market probability supplied, so only fair value is available.",
        }
    abs_edge = abs(edge)
    threshold = max(0.0, min(float(edge_threshold or 0.0), 0.5))
    min_edge = max(threshold, fee)
    side = "up" if edge > 0 else "down"
    expected_value = abs_edge - fee
    if abs_edge < min_edge:
        confidence = "low"
    elif abs_edge < 0.08:
        confidence = "medium"
    else:
        confidence = "high"
    fair = probability_up if side == "up" else 1.0 - probability_up
    market = probability_up - edge if side == "up" else 1.0 - (probability_up - edge)
    # Binary-contract Kelly for buying the underpriced side, capped hard for
    # noisy 5-minute crypto. Positive only when fair > market.
    raw_kelly = max(0.0, (fair - market) / max(1.0 - market, 1e-9))
    volatility_penalty = 1.0 / (1.0 + 18.0 * sigma_1m * math.sqrt(horizon))
    kelly = min(0.05, raw_kelly * 0.25 * volatility_penalty)
    should_trade = abs_edge >= min_edge and expected_value > 0.0
    max_position = 0.0 if not should_trade else kelly
    return {
        "side": side,
        "recommendation": f"buy_{side}" if should_trade else "hold",
        "confidence": confidence,
        "kelly_fraction": round(kelly, 4),
        "max_position_fraction": round(max_position, 4),
        "edge_threshold": round(threshold, 4),
        "fee_bps": fee_bps,
        "min_trade_edge": round(min_edge, 4),
        "expected_value_per_contract": round(expected_value, 4),
        "reason": (
            "Net expected value clears threshold and fee."
            if should_trade else
            "Net expected value is inside the no-trade band after fees."
        ),
    }


def _fade_5m_momentum_strategy(
    momentum_5m: Optional[float],
    *,
    threshold: float = 0.002,
    fee_bps: float = 0.0,
) -> tuple[Optional[str], Dict[str, Any]]:
    """Contrarian strategy: fade large 5-minute crypto moves."""
    threshold = max(0.0, float(threshold or 0.0))
    momentum = _to_float(momentum_5m)
    if momentum is None or abs(momentum) < threshold:
        return None, {
            "side": "none",
            "recommendation": "hold",
            "confidence": "low",
            "strategy_mode": "fade_5m_momentum",
            "momentum_threshold": round(threshold, 8),
            "fee_bps": fee_bps,
            "reason": "5-minute momentum is inside the no-trade band.",
        }
    side = "down" if momentum > 0 else "up"
    return side, {
        "side": side,
        "recommendation": f"buy_{side}",
        "confidence": "medium",
        "strategy_mode": "fade_5m_momentum",
        "momentum_threshold": round(threshold, 8),
        "fee_bps": fee_bps,
        "reason": "Fade a large 5-minute move; paper-only until out-of-sample evidence holds.",
    }


def _follow_5m_momentum_strategy(
    momentum_5m: Optional[float],
    *,
    threshold: float = 0.0025,
    fee_bps: float = 0.0,
) -> tuple[Optional[str], Dict[str, Any]]:
    """Momentum strategy: follow large 5-minute crypto moves."""
    threshold = max(0.0, float(threshold or 0.0))
    momentum = _to_float(momentum_5m)
    if momentum is None or abs(momentum) < threshold:
        return None, {
            "side": "none",
            "recommendation": "hold",
            "confidence": "low",
            "strategy_mode": "follow_5m_momentum",
            "momentum_threshold": round(threshold, 8),
            "fee_bps": fee_bps,
            "reason": "5-minute momentum is inside the no-trade band.",
        }
    side = "up" if momentum > 0 else "down"
    return side, {
        "side": side,
        "recommendation": f"buy_{side}",
        "confidence": "medium",
        "strategy_mode": "follow_5m_momentum",
        "momentum_threshold": round(threshold, 8),
        "fee_bps": fee_bps,
        "reason": "Follow a large 5-minute move; paper-only until out-of-sample evidence holds.",
    }


def _inverse_edge_strategy(
    edge: Optional[float],
    *,
    threshold: float = 0.04,
    fee_bps: float = 0.0,
    strategy_mode: str = "inverse_edge_004",
) -> tuple[Optional[str], Dict[str, Any]]:
    """Buy against large model-vs-market edge when the model is anti-predictive."""
    threshold = max(0.0, float(threshold or 0.0))
    edge_value = _to_float(edge)
    if edge_value is None or abs(edge_value) < threshold:
        return None, {
            "side": "none",
            "recommendation": "hold",
            "confidence": "low",
            "strategy_mode": strategy_mode,
            "edge_threshold": round(threshold, 8),
            "fee_bps": fee_bps,
            "reason": "Model edge is inside the inverse-edge no-trade band.",
        }
    side = "down" if edge_value > 0 else "up"
    return side, {
        "side": side,
        "recommendation": f"buy_{side}",
        "confidence": "medium",
        "strategy_mode": strategy_mode,
        "edge_threshold": round(threshold, 8),
        "fee_bps": fee_bps,
        "reason": "Inverse a large model edge; paper-only until per-asset evidence holds.",
    }


def _per_asset_regime_strategy(
    symbol: str,
    forecast: Dict[str, Any],
    *,
    momentum_threshold: float = 0.0025,
    fee_bps: float = 0.0,
) -> tuple[Optional[str], Dict[str, Any]]:
    """Conservative per-asset selector derived from paper evidence."""
    symbol_norm = normalize_symbol(symbol)
    features = forecast.get("features") or {}
    momentum = _to_float(features.get("momentum_5m"))
    edge = _to_float(forecast.get("edge"))
    notes: List[str] = []

    if symbol_norm == "BTCUSDT":
        side, strategy = _inverse_edge_strategy(
            edge,
            threshold=0.04,
            fee_bps=fee_bps,
            strategy_mode="per_asset_regime_selector",
        )
        if side is not None:
            strategy.update({
                "selected_expert": "inverse_edge_004",
                "symbol_policy": symbol_norm,
            })
            return side, strategy
        notes.append("inverse edge did not clear 4pp")

    if symbol_norm == "ETHUSDT":
        side, strategy = _fade_5m_momentum_strategy(
            momentum,
            threshold=0.0015,
            fee_bps=fee_bps,
        )
        if side is not None:
            strategy.update({
                "strategy_mode": "per_asset_regime_selector",
                "selected_expert": "eth_fade_mom_0015",
                "symbol_policy": symbol_norm,
            })
            return side, strategy
        notes.append("ETH fade momentum did not clear 0.15%")

    return None, {
        "side": "none",
        "recommendation": "hold",
        "confidence": "low",
        "strategy_mode": "per_asset_regime_selector",
        "selected_expert": "hold",
        "symbol_policy": symbol_norm,
        "momentum_threshold": round(max(0.0, float(momentum_threshold or 0.0)), 8),
        "edge_threshold": 0.04,
        "fee_bps": fee_bps,
        "reason": "; ".join(notes) if notes else "No per-asset expert is enabled for this symbol/regime.",
    }


def _apply_crypto_5m_paper_strategy(
    forecast: Dict[str, Any],
    *,
    strategy_mode: str,
    momentum_threshold: float,
    fee_bps: float,
) -> tuple[Dict[str, Any], str]:
    mode = (strategy_mode or "adaptive_model").strip().lower().replace("-", "_")
    if mode in {"adaptive", "model", "edge", "adaptive_model"}:
        applied = dict(forecast)
        applied["strategy"] = dict(forecast.get("strategy") or {})
        applied["strategy"]["strategy_mode"] = "adaptive_model"
        return applied, "adaptive_model"
    if mode == "fade_5m_momentum":
        features = forecast.get("features") or {}
        side, strategy = _fade_5m_momentum_strategy(
            features.get("momentum_5m"),
            threshold=momentum_threshold,
            fee_bps=fee_bps,
        )
        applied = dict(forecast)
        applied["predicted_outcome"] = side
        applied["recommendation"] = strategy["recommendation"]
        applied["strategy"] = strategy
        return applied, "fade_5m_momentum"
    if mode == "follow_5m_momentum":
        features = forecast.get("features") or {}
        side, strategy = _follow_5m_momentum_strategy(
            features.get("momentum_5m"),
            threshold=momentum_threshold,
            fee_bps=fee_bps,
        )
        applied = dict(forecast)
        applied["predicted_outcome"] = side
        applied["recommendation"] = strategy["recommendation"]
        applied["strategy"] = strategy
        return applied, "follow_5m_momentum"
    if mode == "per_asset_regime_selector":
        side, strategy = _per_asset_regime_strategy(
            str(forecast.get("symbol") or ""),
            forecast,
            momentum_threshold=momentum_threshold,
            fee_bps=fee_bps,
        )
        applied = dict(forecast)
        applied["predicted_outcome"] = side
        applied["recommendation"] = strategy["recommendation"]
        applied["strategy"] = strategy
        return applied, "per_asset_regime_selector"
    raise CryptoModelError(
        "strategy_mode must be adaptive_model, fade_5m_momentum, follow_5m_momentum, "
        "or per_asset_regime_selector."
    )


def forecast_crypto_5m(
    symbol: str = "BTCUSDT",
    *,
    target_price: Optional[float] = None,
    horizon_minutes: float = 5.0,
    lookback_minutes: int = 240,
    market_probability: Optional[float] = None,
    ml_mode: str = "fixed",
    training_window: int = 500,
    edge_threshold: float = 0.03,
    fee_bps: float = 0.0,
    klines: Optional[List[Any]] = None,
    include_ml_validation: bool = True,
    adaptive_training_examples: Optional[tuple[List[List[float]], List[int]]] = None,
) -> Dict[str, Any]:
    """Return a conservative probability for a short-horizon crypto UP market.

    ``market_probability`` is optional and represents the prediction-market
    probability for the UP outcome. If supplied, the response includes edge and a
    simple recommendation.
    """
    symbol_norm = normalize_symbol(symbol)
    horizon = max(1.0, min(float(horizon_minutes or 5.0), 15.0))
    horizon_int = max(1, min(int(round(horizon)), 15))
    lookback = max(30, min(int(lookback_minutes or 240), 1000))
    mode = (ml_mode or "fixed").strip().lower()
    if mode not in {"fixed", "adaptive"}:
        raise CryptoModelError("ml_mode must be 'fixed' or 'adaptive'.")
    train_window = max(40, min(int(training_window or 500), 900))
    fetch_limit = lookback
    if mode == "adaptive":
        fetch_limit = min(1000, max(lookback, lookback + horizon_int + train_window + 1))
    raw = klines if klines is not None else _fetch_klines(symbol_norm, limit=fetch_limit)
    candles = _parse_candles(raw)
    closes = [c["close"] for c in candles] if candles else _parse_closes(raw)
    if len(closes) < 20:
        raise CryptoModelError("Need at least 20 one-minute closes to price a 5-minute crypto market.")

    returns = _log_returns(closes)
    if len(returns) < 19:
        raise CryptoModelError("Not enough valid price movement in the supplied candles.")

    current = closes[-1]
    target = float(target_price) if target_price is not None else current
    if target <= 0.0:
        raise CryptoModelError("target_price must be positive.")

    vol_window = returns[-min(180, len(returns)):]
    mean_window = returns[-min(60, len(returns)):]
    sigma_1m = max(pstdev(vol_window), 1e-6)
    mean_1m = mean(mean_window)
    if len(closes) >= 6:
        momentum_5m = math.log(closes[-1] / closes[-6])
    else:
        momentum_5m = 0.0

    # Crypto 5m markets are close to efficient. Shrink drift/momentum toward
    # zero and cap them relative to realized volatility to avoid overconfident
    # directional calls from one short burst.
    raw_drift_1m = 0.20 * mean_1m + 0.10 * (momentum_5m / 5.0)
    drift_cap = 0.20 * sigma_1m
    drift_1m = max(-drift_cap, min(drift_cap, raw_drift_1m))

    ml_features = _ml_signal_features(candles, returns, sigma_1m) if candles else {}
    ml_diagnostics: Dict[str, Any] = {"ml_mode": mode, "ml_mode_effective": "none"}
    if ml_features and mode == "adaptive":
        if adaptive_training_examples is not None:
            ml_probability, ml_score, ml_diagnostics = _adaptive_ml_probability_from_examples(
                ml_features,
                adaptive_training_examples[0],
                adaptive_training_examples[1],
                include_validation=include_ml_validation,
            )
        else:
            ml_probability, ml_score, ml_diagnostics = _adaptive_ml_probability(
                candles,
                ml_features,
                horizon=horizon_int,
                lookback=lookback,
                training_window=train_window,
                include_validation=include_ml_validation,
            )
    elif ml_features:
        ml_probability, ml_score = _ml_probability(ml_features)
        ml_diagnostics.update({
            "ml_mode_effective": "fixed",
            "ml_feature_names": list(_FEATURE_NAMES),
        })
    else:
        ml_probability, ml_score = 0.5, 0.0

    log_moneyness = math.log(current / target)
    mean_move = drift_1m * horizon
    sd_move = sigma_1m * math.sqrt(horizon)
    z = (log_moneyness + mean_move) / max(sd_move, 1e-9)
    probability_vol_model = max(0.01, min(0.99, _normal_cdf(z)))
    probability_ts_model, ts_diagnostics = _ar1_ewma_probability(
        returns,
        current=current,
        target=target,
        horizon=horizon_int,
    )
    # Ensemble: volatility/moneyness gets most weight; ML overlay contributes
    # only when quant features agree, avoiding overfitting from tiny samples.
    probability_up = max(0.01, min(
        0.99,
        0.55 * probability_vol_model + 0.25 * probability_ts_model + 0.20 * ml_probability,
    ))
    probability_down = 1.0 - probability_up
    predicted_outcome = "up" if probability_up >= 0.5 else "down"

    market_p = _normalize_market_probability(market_probability)
    edge = probability_up - market_p if market_p is not None else None
    threshold = max(0.0, min(float(edge_threshold or 0.0), 0.5))
    strategy = _strategy_from_edge(edge, probability_up, sigma_1m, horizon, threshold, fee_bps=fee_bps)
    recommendation = strategy["recommendation"]

    target_return = (target / current) - 1.0
    return {
        "symbol": symbol_norm,
        "horizon_minutes": horizon,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "binance_1m_klines",
        "current_price": round(current, 8),
        "target_price": round(target, 8),
        "target_return": round(target_return, 8),
        "probability_up": round(probability_up, 4),
        "probability_down": round(probability_down, 4),
        "predicted_outcome": predicted_outcome,
        "component_probabilities": {
            "volatility_moneyness": round(probability_vol_model, 4),
            "ar1_ewma_timeseries": round(probability_ts_model, 4),
            "ml_overlay": round(ml_probability, 4),
        },
        "market_probability_up": round(market_p, 4) if market_p is not None else None,
        "edge": round(edge, 4) if edge is not None else None,
        "recommendation": recommendation,
        "strategy": strategy,
        "features": {
            "lookback_minutes": lookback,
            "candles_used": len(closes),
            "mean_return_1m": round(mean_1m, 8),
            "realized_vol_1m": round(sigma_1m, 8),
            "momentum_5m": round(momentum_5m, 8),
            "drift_1m_shrunk": round(drift_1m, 8),
            "z_score": round(z, 4),
            "ml_score": round(ml_score, 4),
            "edge_threshold": round(threshold, 4),
            "fee_bps": fee_bps,
            **ml_diagnostics,
            **ts_diagnostics,
            **{k: round(v, 6) for k, v in ml_features.items()},
        },
        "method": (
            "Quant/ML ensemble over recent 1m candles: shrunken-drift lognormal "
            "moneyness model, AR(1)+EWMA time-series forecast, and fixed or "
            "adaptive logistic overlay from momentum, reversal, volatility "
            "regime, range position, and volume imbalance. Recommendations are "
            "fee-aware and require positive net expected value. "
            "Use as a fast fair-value estimate for 5-minute UP/DOWN markets, not "
            "as trading advice."
        ),
    }


def _brier(rows: List[Dict[str, Any]]) -> Optional[float]:
    if not rows:
        return None
    return sum((r["probability_up"] - r["outcome_up"]) ** 2 for r in rows) / len(rows)


def _log_loss(rows: List[Dict[str, Any]]) -> Optional[float]:
    if not rows:
        return None
    eps = 1e-6
    total = 0.0
    for row in rows:
        p = _clip(row["probability_up"], eps, 1.0 - eps)
        y = row["outcome_up"]
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / len(rows)


def _calibration(rows: List[Dict[str, Any]], bins: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        bucket = [
            r for r in rows
            if lo <= r["probability_up"] < hi
            or (i == bins - 1 and lo <= r["probability_up"] <= hi)
        ]
        if not bucket:
            continue
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": len(bucket),
            "avg_probability": round(_safe_mean([r["probability_up"] for r in bucket]), 4),
            "empirical_up_rate": round(_safe_mean([r["outcome_up"] for r in bucket]), 4),
        })
    return out


def _max_drawdown(equity: List[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = min(max_dd, value - peak)
    return max_dd


def _strategy_metrics_for_threshold(
    rows: List[Dict[str, Any]],
    *,
    market_up: float,
    fee: float,
    edge_threshold: float,
) -> Dict[str, Any]:
    threshold = max(0.0, min(float(edge_threshold or 0.0), 0.5))
    trades = []
    equity = []
    running_pnl = 0.0
    for row in rows:
        edge = row.get("edge")
        traded = edge is not None and abs(edge) >= threshold
        side = "up" if (edge or 0.0) > 0 else "down"
        pnl = 0.0
        if traded:
            outcome_up = row["outcome_up"]
            if side == "up":
                price = market_up
                win = outcome_up == 1
            else:
                price = 1.0 - market_up
                win = outcome_up == 0
            pnl = (1.0 - price if win else -price) - fee
            trades.append({"side": side, "outcome_up": outcome_up, "pnl": pnl})
        running_pnl += pnl
        equity.append(running_pnl)

    wins = [
        trade for trade in trades
        if (trade["side"] == "up" and trade["outcome_up"] == 1)
        or (trade["side"] == "down" and trade["outcome_up"] == 0)
    ]
    trade_pnl = sum(trade["pnl"] for trade in trades)
    return {
        "edge_threshold": round(threshold, 4),
        "trades": len(trades),
        "trade_rate": round(len(trades) / len(rows), 4) if rows else 0.0,
        "hit_rate": round(len(wins) / len(trades), 4) if trades else None,
        "pnl_per_contract": round(trade_pnl, 6),
        "avg_pnl_per_trade": round(trade_pnl / len(trades), 6) if trades else None,
        "max_drawdown": round(_max_drawdown(equity), 6),
    }


def _select_threshold_from_rows(
    rows: List[Dict[str, Any]],
    *,
    thresholds: List[float],
    market_up: float,
    fee: float,
    min_trade_count: int,
    selection_fraction: float,
) -> Dict[str, Any]:
    split_at = int(len(rows) * selection_fraction)
    if len(rows) >= 4:
        split_at = max(1, min(len(rows) - 1, split_at))
    else:
        split_at = len(rows)
    selection_rows = rows[:split_at]
    holdout_rows = rows[split_at:]
    threshold_results = []
    for threshold in thresholds:
        all_metrics = _strategy_metrics_for_threshold(
            rows,
            market_up=market_up,
            fee=fee,
            edge_threshold=threshold,
        )
        selection_metrics = _strategy_metrics_for_threshold(
            selection_rows,
            market_up=market_up,
            fee=fee,
            edge_threshold=threshold,
        )
        holdout_metrics = _strategy_metrics_for_threshold(
            holdout_rows,
            market_up=market_up,
            fee=fee,
            edge_threshold=threshold,
        ) if holdout_rows else None
        threshold_results.append({
            **all_metrics,
            "selection": selection_metrics,
            "holdout": holdout_metrics,
        })

    eligible = [
        item for item in threshold_results
        if item["selection"]["trades"] >= min_trade_count
    ]
    selected = max(
        eligible or threshold_results,
        key=lambda item: (
            item["selection"]["pnl_per_contract"],
            (
                item["selection"]["avg_pnl_per_trade"]
                if item["selection"]["avg_pnl_per_trade"] is not None else -1e9
            ),
            -abs(item["selection"]["max_drawdown"]),
            item["edge_threshold"],
        ),
    ) if threshold_results else None
    return {
        "selection_rows": len(selection_rows),
        "holdout_rows": len(holdout_rows),
        "thresholds": threshold_results,
        "selected_threshold": selected,
        "holdout_selected_threshold": selected["holdout"] if selected else None,
    }


def _rolling_fold_ranges(total_rows: int, *, folds: int) -> List[tuple[int, int]]:
    fold_count = max(1, int(folds or 1))
    if fold_count <= 1 or total_rows < 8:
        return [(0, total_rows)]
    window = max(4, total_rows // fold_count)
    ranges: List[tuple[int, int]] = []
    for fold in range(fold_count):
        start = fold * window
        end = total_rows if fold == fold_count - 1 else min(total_rows, start + window)
        if end - start >= 4:
            ranges.append((start, end))
    return ranges or [(0, total_rows)]


def walk_forward_backtest(
    symbol: str = "BTCUSDT",
    *,
    klines: Optional[List[Any]] = None,
    horizon_minutes: int = 5,
    lookback_minutes: int = 240,
    market_probability: float = 0.5,
    fee_bps: float = 0.0,
    step_minutes: int = 1,
    max_rows: Optional[int] = None,
    include_rows: bool = False,
    ml_mode: str = "fixed",
    training_window: int = 500,
    edge_threshold: float = 0.03,
) -> Dict[str, Any]:
    """Walk-forward evaluation for 5-minute crypto UP/DOWN strategy.

    Each row uses only candles available at that minute, predicts whether the
    close ``horizon_minutes`` ahead will be above the current close, then scores
    the model and the strategy decision. ``market_probability`` is the synthetic
    UP price used when real contract prices are unavailable; pass the observed
    average market price when backtesting a specific venue export.
    """
    symbol_norm = normalize_symbol(symbol)
    horizon = max(1, min(int(horizon_minutes or 5), 15))
    lookback = max(30, min(int(lookback_minutes or 240), 1000))
    step = max(1, int(step_minutes or 1))
    mode = (ml_mode or "fixed").strip().lower()
    if mode not in {"fixed", "adaptive"}:
        raise CryptoModelError("ml_mode must be 'fixed' or 'adaptive'.")
    train_window = max(40, min(int(training_window or 500), 900))
    threshold = max(0.0, min(float(edge_threshold or 0.0), 0.5))
    model_history = lookback if mode == "fixed" else min(1000, lookback + horizon + train_window + 1)
    needed = model_history + horizon + 1
    live_limit = 1000 if max_rows is None else model_history + horizon + max(1, int(max_rows)) * step
    raw = klines if klines is not None else _fetch_klines(
        symbol_norm,
        limit=max(needed, min(1000, live_limit)),
    )
    candles = _parse_candles(raw)
    if len(candles) < needed:
        raise CryptoModelError(
            f"Need at least {needed} valid 1-minute candles for a walk-forward backtest."
        )

    rows: List[Dict[str, Any]] = []
    fee = max(0.0, float(fee_bps or 0.0)) / 10000.0
    market_up = _normalize_market_probability(market_probability)
    if market_up is None:
        market_up = 0.5

    stop = len(candles) - horizon
    starts = range(model_history - 1, stop, step)
    if max_rows is not None:
        starts = list(starts)[-max(1, int(max_rows)):]

    precomputed_examples: List[tuple[int, List[float], int]] = []
    if mode == "adaptive":
        precomputed_examples = _precompute_ml_training_examples(
            candles,
            horizon=horizon,
            min_history=max(30, min(lookback, 240)),
        )

    equity = []
    running_pnl = 0.0
    for i in starts:
        current = candles[i]["close"]
        future = candles[i + horizon]["close"]
        outcome_up = 1 if future > current else 0
        history = [
            [0, c["open"], c["high"], c["low"], c["close"], c["volume"]]
            for c in candles[:i + 1]
        ]
        forecast_history = history[-model_history:]
        adaptive_examples = None
        if precomputed_examples:
            cutoff = i - horizon
            selected = [
                (vector, label)
                for idx, vector, label in precomputed_examples
                if idx <= cutoff
            ][-train_window:]
            adaptive_examples = (
                [vector for vector, _ in selected],
                [label for _, label in selected],
            )
        forecast = forecast_crypto_5m(
            symbol_norm,
            target_price=current,
            horizon_minutes=horizon,
            lookback_minutes=lookback,
            market_probability=market_up,
            ml_mode=mode,
            training_window=train_window,
            edge_threshold=threshold,
            fee_bps=fee_bps,
            klines=forecast_history,
            include_ml_validation=False,
            adaptive_training_examples=adaptive_examples,
        )
        rec = forecast["recommendation"]
        side = forecast["strategy"]["side"]
        traded = rec in {"buy_up", "buy_down"}
        pnl = 0.0
        if traded:
            if side == "up":
                price = market_up
                win = outcome_up == 1
            else:
                price = 1.0 - market_up
                win = outcome_up == 0
            pnl = (1.0 - price if win else -price) - fee
        running_pnl += pnl
        equity.append(running_pnl)
        rows.append({
            "index": i,
            "symbol": symbol_norm,
            "current_price": current,
            "future_price": future,
            "outcome_up": outcome_up,
            "probability_up": forecast["probability_up"],
            "predicted_outcome": forecast["predicted_outcome"],
            "edge": forecast["edge"],
            "recommendation": rec,
            "side": side,
            "traded": traded,
            "pnl": round(pnl, 6),
        })

    brier = _brier(rows)
    log_loss = _log_loss(rows)
    accuracy = _safe_mean([
        1.0 if (r["probability_up"] >= 0.5) == (r["outcome_up"] == 1) else 0.0
        for r in rows
    ])
    threshold_metrics = _strategy_metrics_for_threshold(
        rows,
        market_up=market_up,
        fee=fee,
        edge_threshold=threshold,
    )
    result: Dict[str, Any] = {
        "symbol": symbol_norm,
        "horizon_minutes": horizon,
        "lookback_minutes": lookback,
        "ml_mode": mode,
        "training_window": train_window if mode == "adaptive" else None,
        "edge_threshold": round(threshold, 4),
        "market_probability_up": round(market_up, 4),
        "fee_bps": fee_bps,
        "n": len(rows),
        "brier_score": round(brier, 6) if brier is not None else None,
        "log_loss": round(log_loss, 6) if log_loss is not None else None,
        "accuracy": round(accuracy, 4),
        "base_up_rate": round(_safe_mean([r["outcome_up"] for r in rows]), 4),
        "avg_probability_up": round(_safe_mean([r["probability_up"] for r in rows]), 4),
        "avg_edge": round(_safe_mean([r["edge"] or 0.0 for r in rows]), 4),
        "calibration": _calibration(rows),
        **{k: v for k, v in threshold_metrics.items() if k != "edge_threshold"},
    }
    if include_rows:
        result["rows"] = rows
    return result


def _strategy_summary_from_trades(trades: List[Dict[str, Any]], total_rows: int) -> Dict[str, Any]:
    running = 0.0
    equity: List[float] = []
    wins = 0
    for trade in trades:
        pnl = float(trade.get("pnl") or 0.0)
        running += pnl
        equity.append(running)
        if trade.get("side") == trade.get("actual_outcome"):
            wins += 1
    count = len(trades)
    return {
        "trades": count,
        "trade_rate": round(count / total_rows, 4) if total_rows else 0.0,
        "hit_rate": round(wins / count, 4) if count else None,
        "pnl_per_contract": round(running, 6),
        "avg_pnl_per_trade": round(running / count, 6) if count else None,
        "max_drawdown": round(_max_drawdown(equity), 6),
    }


def _side_pnl(
    *,
    side: Optional[str],
    outcome_up: int,
    market_up: float,
    fee: float,
) -> Optional[float]:
    if side not in {"up", "down"}:
        return None
    if side == "up":
        price = market_up
        win = outcome_up == 1
    else:
        price = 1.0 - market_up
        win = outcome_up == 0
    return (1.0 - price if win else -price) - fee


def _current_selector_side(symbol: str, forecast: Dict[str, Any], fee_bps: float) -> Optional[str]:
    side, _strategy = _per_asset_regime_strategy(
        symbol,
        forecast,
        momentum_threshold=0.0025,
        fee_bps=fee_bps,
    )
    return side


def compare_ott_crypto_5m_strategies(
    symbols: Optional[List[str]] = None,
    *,
    klines_by_symbol: Optional[Dict[str, List[Any]]] = None,
    history_days: Optional[float] = None,
    max_candles: int = 10_000,
    horizon_minutes: int = 5,
    lookback_minutes: int = 60,
    market_probability: float = 0.5,
    fee_bps: float = 2.0,
    step_minutes: int = 1,
    max_rows: Optional[int] = None,
    ml_mode: str = "adaptive",
    training_window: int = 120,
    ott_length: int = 2,
    ott_percent: float = 1.4,
) -> Dict[str, Any]:
    """Compare OTT variants with the current per-asset crypto selector."""
    raw_symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    symbol_list = [normalize_symbol(symbol) for symbol in raw_symbols]
    horizon = max(1, min(int(horizon_minutes or 5), 15))
    lookback = max(30, min(int(lookback_minutes or 60), 1000))
    mode = (ml_mode or "adaptive").strip().lower()
    if mode not in {"fixed", "adaptive"}:
        raise CryptoModelError("ml_mode must be 'fixed' or 'adaptive'.")
    train_window = max(40, min(int(training_window or 120), 900))
    market_up = _normalize_market_probability(market_probability)
    if market_up is None:
        market_up = 0.5
    fee = max(0.0, float(fee_bps or 0.0)) / 10000.0
    step = max(1, int(step_minutes or 1))
    model_history = lookback if mode == "fixed" else min(1000, lookback + horizon + train_window + 1)
    needed = model_history + horizon + 1
    by_symbol_input = klines_by_symbol or {}
    strategies = {
        "ott_follow": "Follow current OTT trend each row",
        "ott_fade_after_reversal": "Fade the new OTT direction only on reversal rows",
        "current_selector": "BTC inverse edge >=4pp; ETH fade |5m momentum| >=0.15%; SOL hold",
        "selector_ott_filter": "Current selector only when its side agrees with OTT trend",
    }
    results: Dict[str, Any] = {}
    failures: List[Dict[str, str]] = []

    for symbol in symbol_list:
        try:
            klines = by_symbol_input.get(symbol) or by_symbol_input.get(symbol.replace("USDT", ""))
            if klines is None:
                if history_days is not None:
                    klines = fetch_klines_history(symbol, days=history_days, max_candles=max_candles)
                else:
                    limit = max(needed, min(1000, needed + (max_rows or 300) * step))
                    klines = _fetch_klines(symbol, limit=limit)
            candles = _parse_candles(klines)
            if len(candles) < needed:
                raise CryptoModelError(
                    f"Need at least {needed} valid 1-minute candles for OTT comparison."
                )
            ott = compute_ott(candles, length=ott_length, percent=ott_percent)
            stop = len(candles) - horizon
            starts = range(model_history - 1, stop, step)
            if max_rows is not None:
                starts = list(starts)[-max(1, int(max_rows)):]
            precomputed_examples: List[tuple[int, List[float], int]] = []
            if mode == "adaptive":
                precomputed_examples = _precompute_ml_training_examples(
                    candles,
                    horizon=horizon,
                    min_history=max(30, min(lookback, 240)),
                )
            trades_by_strategy: Dict[str, List[Dict[str, Any]]] = {key: [] for key in strategies}
            evaluated = 0
            reversals = 0
            for i in starts:
                current = candles[i]["close"]
                future = candles[i + horizon]["close"]
                outcome_up = 1 if future > current else 0
                actual = "up" if outcome_up == 1 else "down"
                history = [
                    [0, c["open"], c["high"], c["low"], c["close"], c["volume"]]
                    for c in candles[:i + 1]
                ]
                forecast_history = history[-model_history:]
                adaptive_examples = None
                if precomputed_examples:
                    cutoff = i - horizon
                    selected = [
                        (vector, label)
                        for idx, vector, label in precomputed_examples
                        if idx <= cutoff
                    ][-train_window:]
                    adaptive_examples = (
                        [vector for vector, _ in selected],
                        [label for _, label in selected],
                    )
                forecast = forecast_crypto_5m(
                    symbol,
                    target_price=current,
                    horizon_minutes=horizon,
                    lookback_minutes=lookback,
                    market_probability=market_up,
                    ml_mode=mode,
                    training_window=train_window,
                    edge_threshold=0.03,
                    fee_bps=fee_bps,
                    klines=forecast_history,
                    include_ml_validation=False,
                    adaptive_training_examples=adaptive_examples,
                )
                ott_row = ott[i]
                ott_side = str(ott_row["trend_side"])
                selector_side = _current_selector_side(symbol, forecast, fee_bps)
                side_by_strategy = {
                    "ott_follow": ott_side,
                    "ott_fade_after_reversal": (
                        "down" if ott_side == "up" else "up"
                    ) if ott_row.get("reversal") else None,
                    "current_selector": selector_side,
                    "selector_ott_filter": (
                        selector_side if selector_side == ott_side else None
                    ),
                }
                evaluated += 1
                reversals += 1 if ott_row.get("reversal") else 0
                for strategy_name, side in side_by_strategy.items():
                    pnl = _side_pnl(
                        side=side,
                        outcome_up=outcome_up,
                        market_up=market_up,
                        fee=fee,
                    )
                    if pnl is None:
                        continue
                    trades_by_strategy[strategy_name].append({
                        "index": i,
                        "side": side,
                        "actual_outcome": actual,
                        "pnl": pnl,
                    })
            results[symbol] = {
                "rows": evaluated,
                "ott_reversals": reversals,
                "strategies": {
                    key: {
                        "description": strategies[key],
                        **_strategy_summary_from_trades(trades, evaluated),
                    }
                    for key, trades in trades_by_strategy.items()
                },
            }
        except CryptoModelError as exc:
            failures.append({"symbol": symbol, "error": str(exc)})

    aggregate_trades: Dict[str, List[Dict[str, Any]]] = {key: [] for key in strategies}
    total_rows = 0
    for symbol, symbol_result in results.items():
        total_rows += int(symbol_result.get("rows", 0))
        for key in strategies:
            summary = symbol_result["strategies"][key]
            # Reconstruct aggregate only from summary would lose drawdown order,
            # so aggregate metrics intentionally sum independent symbol summaries.
            aggregate_trades[key].append({
                "symbol": symbol,
                "trades": summary["trades"],
                "hit_rate": summary["hit_rate"],
                "pnl_per_contract": summary["pnl_per_contract"],
            })
    aggregate: Dict[str, Any] = {}
    for key, items in aggregate_trades.items():
        trades = sum(int(item["trades"]) for item in items)
        pnl = sum(float(item["pnl_per_contract"]) for item in items)
        wins = 0.0
        for item in items:
            if item["hit_rate"] is not None:
                wins += float(item["hit_rate"]) * int(item["trades"])
        aggregate[key] = {
            "description": strategies[key],
            "trades": trades,
            "trade_rate": round(trades / total_rows, 4) if total_rows else 0.0,
            "hit_rate": round(wins / trades, 4) if trades else None,
            "pnl_per_contract": round(pnl, 6),
            "avg_pnl_per_trade": round(pnl / trades, 6) if trades else None,
        }

    ranked = sorted(
        aggregate.items(),
        key=lambda item: (
            item[1]["pnl_per_contract"],
            item[1]["avg_pnl_per_trade"] if item[1]["avg_pnl_per_trade"] is not None else -1e9,
        ),
        reverse=True,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols_requested": symbol_list,
        "symbols_evaluated": sorted(results),
        "history_days": history_days,
        "max_candles": max_candles if history_days is not None else None,
        "horizon_minutes": horizon,
        "lookback_minutes": lookback,
        "market_probability_up": round(market_up, 4),
        "fee_bps": fee_bps,
        "step_minutes": step,
        "max_rows": max_rows,
        "ml_mode": mode,
        "training_window": train_window if mode == "adaptive" else None,
        "ott": {"length": max(1, int(ott_length or 1)), "percent": float(ott_percent or 0.0)},
        "strategies": strategies,
        "by_symbol": results,
        "aggregate": aggregate,
        "ranked": [{"strategy": key, **metrics} for key, metrics in ranked],
        "best": {"strategy": ranked[0][0], **ranked[0][1]} if ranked else None,
        "failures": failures,
        "note": (
            "Research comparison on 1-minute candles. Contract PnL assumes the "
            "configured market_probability_up and fee_bps; no spread or slippage."
        ),
    }


def sweep_crypto_5m_thresholds(
    symbol: str = "BTCUSDT",
    *,
    klines: Optional[List[Any]] = None,
    horizon_minutes: int = 5,
    lookback_minutes: int = 240,
    market_probability: float = 0.5,
    fee_bps: float = 0.0,
    step_minutes: int = 1,
    max_rows: Optional[int] = None,
    ml_modes: Optional[List[str]] = None,
    training_window: int = 500,
    edge_thresholds: Optional[List[float]] = None,
    min_trades: int = 1,
    selection_fraction: float = 0.6,
    folds: int = 1,
) -> Dict[str, Any]:
    """Compare model modes and abstention thresholds on walk-forward rows."""
    symbol_norm = normalize_symbol(symbol)
    modes = ml_modes or ["fixed", "adaptive"]
    thresholds = edge_thresholds or [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    thresholds = sorted({round(max(0.0, min(float(t), 0.5)), 4) for t in thresholds})
    market_up = _normalize_market_probability(market_probability)
    if market_up is None:
        market_up = 0.5
    fee = max(0.0, float(fee_bps or 0.0)) / 10000.0
    min_trade_count = max(0, int(min_trades or 0))
    split_fraction = _clip(float(selection_fraction or 0.6), 0.1, 0.9)
    fold_count = max(1, int(folds or 1))

    model_results: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for mode in modes:
        mode_norm = (mode or "fixed").strip().lower()
        backtest = walk_forward_backtest(
            symbol_norm,
            klines=klines,
            horizon_minutes=horizon_minutes,
            lookback_minutes=lookback_minutes,
            market_probability=market_up,
            fee_bps=fee_bps,
            step_minutes=step_minutes,
            max_rows=max_rows,
            include_rows=True,
            ml_mode=mode_norm,
            training_window=training_window,
            edge_threshold=0.0,
        )
        rows = backtest.pop("rows")
        selection = _select_threshold_from_rows(
            rows,
            thresholds=thresholds,
            market_up=market_up,
            fee=fee,
            min_trade_count=min_trade_count,
            selection_fraction=split_fraction,
        )
        threshold_results = [
            {**item, "ml_mode": mode_norm}
            for item in selection["thresholds"]
        ]
        selected = {**selection["selected_threshold"], "ml_mode": mode_norm}
        candidate = {
            **selected,
            "holdout_score": selected["holdout"]["pnl_per_contract"] if selected["holdout"] else None,
        }
        candidates.append(candidate)
        fold_results = []
        for fold_index, (start, end) in enumerate(_rolling_fold_ranges(len(rows), folds=fold_count), start=1):
            fold_selection = _select_threshold_from_rows(
                rows[start:end],
                thresholds=thresholds,
                market_up=market_up,
                fee=fee,
                min_trade_count=min_trade_count,
                selection_fraction=split_fraction,
            )
            selected_fold = fold_selection["selected_threshold"]
            if selected_fold is None:
                continue
            fold_results.append({
                "fold": fold_index,
                "row_start": start,
                "row_end": end,
                "selection_rows": fold_selection["selection_rows"],
                "holdout_rows": fold_selection["holdout_rows"],
                "selected_threshold": {**selected_fold, "ml_mode": mode_norm},
                "holdout_selected_threshold": selected_fold["holdout"],
            })
        model_results.append({
            "ml_mode": mode_norm,
            "forecast_metrics": {
                key: backtest[key]
                for key in (
                    "n",
                    "brier_score",
                    "log_loss",
                    "accuracy",
                    "base_up_rate",
                    "avg_probability_up",
                    "avg_edge",
                )
            },
            "selection_rows": selection["selection_rows"],
            "holdout_rows": selection["holdout_rows"],
            "thresholds": threshold_results,
            "selected_threshold": selected,
            "holdout_selected_threshold": selected["holdout"],
            "folds": fold_results,
        })

    eligible_candidates = [m for m in candidates if m["selection"]["trades"] >= min_trade_count]
    best = max(
        eligible_candidates or candidates,
        key=lambda item: (
            item["holdout_score"] if item["holdout_score"] is not None else item["selection"]["pnl_per_contract"],
            (
                item["holdout"]["avg_pnl_per_trade"]
                if item["holdout"] and item["holdout"]["avg_pnl_per_trade"] is not None
                else -1e9
            ),
            -abs(item["holdout"]["max_drawdown"] if item["holdout"] else item["selection"]["max_drawdown"]),
            item["edge_threshold"],
        ),
    ) if candidates else None
    return {
        "symbol": symbol_norm,
        "horizon_minutes": max(1, min(int(horizon_minutes or 5), 15)),
        "lookback_minutes": max(30, min(int(lookback_minutes or 240), 1000)),
        "market_probability_up": round(market_up, 4),
        "fee_bps": fee_bps,
        "thresholds": thresholds,
        "min_trades": min_trade_count,
        "selection_fraction": round(split_fraction, 4),
        "folds": fold_count,
        "selection_rule": (
            "select threshold on the earlier chronological selection split, "
            "then rank model modes by selected-threshold holdout pnl_per_contract"
        ),
        "best": best,
        "models": model_results,
    }


def _aggregate_selected_holdouts(sweeps: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected = [sweep.get("best") for sweep in sweeps if sweep.get("best")]
    holdouts = [item.get("holdout") for item in selected if item and item.get("holdout")]
    active_holdouts = [h for h in holdouts if int(h.get("trades", 0)) > 0]
    total_trades = sum(int(h.get("trades", 0)) for h in holdouts)
    total_pnl = sum(float(h.get("pnl_per_contract", 0.0)) for h in holdouts)
    total_wins = 0.0
    for holdout in holdouts:
        hit_rate = holdout.get("hit_rate")
        trades = int(holdout.get("trades", 0))
        if hit_rate is not None:
            total_wins += float(hit_rate) * trades
    selected_modes = sorted({item["ml_mode"] for item in selected if item.get("ml_mode")})
    selected_thresholds = sorted({item["edge_threshold"] for item in selected if "edge_threshold" in item})
    stable_selection = len(selected_modes) == 1 and len(selected_thresholds) == 1
    if total_trades >= 100 and stable_selection and len(active_holdouts) == len(sweeps):
        evidence_quality = "strong"
    elif total_trades >= 30 and len(active_holdouts) >= max(1, len(sweeps) // 2):
        evidence_quality = "moderate"
    elif total_trades > 0:
        evidence_quality = "thin"
    else:
        evidence_quality = "none"
    return {
        "symbols": len(sweeps),
        "symbols_with_holdout": len(holdouts),
        "symbols_with_holdout_trades": len(active_holdouts),
        "total_holdout_trades": total_trades,
        "holdout_pnl_per_contract": round(total_pnl, 6),
        "holdout_hit_rate": round(total_wins / total_trades, 4) if total_trades else None,
        "avg_holdout_pnl_per_symbol": round(total_pnl / len(holdouts), 6) if holdouts else None,
        "selected_modes": selected_modes,
        "selected_edge_thresholds": selected_thresholds,
        "stable_selection": stable_selection,
        "trade_coverage": round(len(active_holdouts) / len(sweeps), 4) if sweeps else 0.0,
        "evidence_quality": evidence_quality,
    }


def _aggregate_fold_holdouts(sweeps: List[Dict[str, Any]]) -> Dict[str, Any]:
    fold_items: List[Dict[str, Any]] = []
    for sweep in sweeps:
        for model in sweep.get("models", []):
            for fold in model.get("folds", []):
                selected = fold.get("selected_threshold") or {}
                holdout = fold.get("holdout_selected_threshold")
                if holdout is None:
                    continue
                fold_items.append({
                    "symbol": sweep.get("symbol"),
                    "ml_mode": model.get("ml_mode"),
                    "edge_threshold": selected.get("edge_threshold"),
                    "holdout": holdout,
                })
    active = [item for item in fold_items if int(item["holdout"].get("trades", 0)) > 0]
    total_trades = sum(int(item["holdout"].get("trades", 0)) for item in fold_items)
    total_pnl = sum(float(item["holdout"].get("pnl_per_contract", 0.0)) for item in fold_items)
    total_wins = 0.0
    for item in fold_items:
        hit_rate = item["holdout"].get("hit_rate")
        trades = int(item["holdout"].get("trades", 0))
        if hit_rate is not None:
            total_wins += float(hit_rate) * trades
    modes = sorted({item["ml_mode"] for item in fold_items if item.get("ml_mode")})
    thresholds = sorted({
        item["edge_threshold"] for item in fold_items
        if item.get("edge_threshold") is not None
    })
    stable = len(modes) == 1 and len(thresholds) == 1
    if total_trades >= 200 and stable and len(active) == len(fold_items):
        quality = "strong"
    elif total_trades >= 60 and stable and len(active) >= max(1, len(fold_items) // 2):
        quality = "moderate"
    elif total_trades > 0:
        quality = "thin"
    else:
        quality = "none"
    return {
        "folds": len(fold_items),
        "folds_with_holdout_trades": len(active),
        "total_holdout_trades": total_trades,
        "holdout_pnl_per_contract": round(total_pnl, 6),
        "holdout_hit_rate": round(total_wins / total_trades, 4) if total_trades else None,
        "selected_modes": modes,
        "selected_edge_thresholds": thresholds,
        "stable_selection": stable,
        "trade_coverage": round(len(active) / len(fold_items), 4) if fold_items else 0.0,
        "evidence_quality": quality,
    }


def benchmark_crypto_5m_strategy(
    symbols: Optional[List[str]] = None,
    *,
    klines_by_symbol: Optional[Dict[str, List[Any]]] = None,
    history_days: Optional[float] = None,
    max_candles: int = 10_000,
    horizon_minutes: int = 5,
    lookback_minutes: int = 240,
    market_probability: float = 0.5,
    fee_bps: float = 0.0,
    step_minutes: int = 1,
    max_rows: Optional[int] = None,
    ml_modes: Optional[List[str]] = None,
    training_window: int = 500,
    edge_thresholds: Optional[List[float]] = None,
    min_trades: int = 1,
    selection_fraction: float = 0.6,
    folds: int = 1,
) -> Dict[str, Any]:
    """Run threshold/model sweeps across symbols and aggregate holdout evidence."""
    raw_symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    symbol_list = [normalize_symbol(symbol) for symbol in raw_symbols]
    sweeps: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    by_symbol = klines_by_symbol or {}

    for symbol in symbol_list:
        try:
            klines = by_symbol.get(symbol) or by_symbol.get(symbol.replace("USDT", ""))
            if klines is None and history_days is not None:
                klines = fetch_klines_history(symbol, days=history_days, max_candles=max_candles)
            sweep = sweep_crypto_5m_thresholds(
                symbol,
                klines=klines,
                horizon_minutes=horizon_minutes,
                lookback_minutes=lookback_minutes,
                market_probability=market_probability,
                fee_bps=fee_bps,
                step_minutes=step_minutes,
                max_rows=max_rows,
                ml_modes=ml_modes,
                training_window=training_window,
                edge_thresholds=edge_thresholds,
                min_trades=min_trades,
                selection_fraction=selection_fraction,
                folds=folds,
            )
        except CryptoModelError as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
            continue
        sweeps.append(sweep)

    aggregate = _aggregate_selected_holdouts(sweeps)
    fold_aggregate = _aggregate_fold_holdouts(sweeps)
    return {
        "symbols_requested": symbol_list,
        "symbols_evaluated": [sweep["symbol"] for sweep in sweeps],
        "horizon_minutes": max(1, min(int(horizon_minutes or 5), 15)),
        "lookback_minutes": max(30, min(int(lookback_minutes or 240), 1000)),
        "history_days": history_days,
        "max_candles": max_candles if history_days is not None else None,
        "market_probability_up": round(_normalize_market_probability(market_probability) or 0.5, 4),
        "fee_bps": fee_bps,
        "selection_fraction": round(_clip(float(selection_fraction or 0.6), 0.1, 0.9), 4),
        "folds": max(1, int(folds or 1)),
        "selection_rule": (
            "per-symbol threshold selected on earlier rows; aggregate is selected "
            "holdout performance across evaluated symbols"
        ),
        "aggregate": aggregate,
        "fold_aggregate": fold_aggregate,
        "sweeps": sweeps,
        "failures": failures,
    }


def summarize_crypto_5m_benchmark(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact benchmark record suitable for longitudinal JSONL logs."""
    sweeps = result.get("sweeps", [])
    best_by_symbol = []
    for sweep in sweeps:
        best = sweep.get("best") or {}
        holdout = best.get("holdout") or {}
        best_by_symbol.append({
            "symbol": sweep.get("symbol"),
            "ml_mode": best.get("ml_mode"),
            "edge_threshold": best.get("edge_threshold"),
            "holdout_trades": holdout.get("trades", 0),
            "holdout_pnl_per_contract": holdout.get("pnl_per_contract", 0),
            "holdout_hit_rate": holdout.get("hit_rate"),
        })
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "type": "crypto_5m_benchmark",
        "symbols_requested": result.get("symbols_requested", []),
        "symbols_evaluated": result.get("symbols_evaluated", []),
        "horizon_minutes": result.get("horizon_minutes"),
        "lookback_minutes": result.get("lookback_minutes"),
        "history_days": result.get("history_days"),
        "max_candles": result.get("max_candles"),
        "market_probability_up": result.get("market_probability_up"),
        "fee_bps": result.get("fee_bps"),
        "selection_fraction": result.get("selection_fraction"),
        "folds": result.get("folds"),
        "aggregate": result.get("aggregate", {}),
        "fold_aggregate": result.get("fold_aggregate", {}),
        "best_by_symbol": best_by_symbol,
        "failures": result.get("failures", []),
    }


def append_crypto_5m_benchmark_log(
    result: Dict[str, Any],
    *,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append a compact benchmark record to a JSONL evidence log."""
    log_path = path or DEFAULT_BENCHMARK_LOG_PATH
    record = summarize_crypto_5m_benchmark(result)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "path": str(log_path),
        "record": record,
    }


def _jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    if str(path).endswith(".gz"):
        import gzip
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _signal_record_id(record: Dict[str, Any]) -> str:
    symbol = str(record.get("symbol") or "UNKNOWN")
    start = int(record.get("start_time_ms") or 0)
    expiry = int(record.get("expiry_time_ms") or 0)
    horizon = int(record.get("horizon_minutes") or 5)
    target = _to_float(record.get("target_price"))
    target_key = f"{target:.8f}" if target is not None else "none"
    return f"{symbol}:{start}:{expiry}:{horizon}:{target_key}"


def init_crypto_5m_signal_db(*, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Create the SQLite signal table used as a queryable paper-trade dataset."""
    path = db_path or DEFAULT_SIGNAL_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crypto_5m_signals (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                logged_at TEXT,
                symbol TEXT NOT NULL,
                status TEXT,
                strategy_name TEXT,
                recommendation TEXT,
                predicted_outcome TEXT,
                actual_outcome TEXT,
                prediction_correct INTEGER,
                start_time_ms INTEGER,
                expiry_time_ms INTEGER,
                horizon_minutes INTEGER,
                target_price REAL,
                resolved_price REAL,
                probability_up REAL,
                market_probability_up REAL,
                edge REAL,
                fee_bps REAL,
                edge_threshold REAL,
                realized_vol_1m REAL,
                momentum_5m REAL,
                ml_mode_effective TEXT,
                pnl_per_contract REAL,
                raw_json TEXT NOT NULL
            )
        """)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(crypto_5m_signals)").fetchall()}
        for name, ddl_type in (
            ("realized_vol_1m", "REAL"),
            ("momentum_5m", "REAL"),
            ("ml_mode_effective", "TEXT"),
            ("strategy_name", "TEXT"),
        ):
            if name not in existing:
                conn.execute(f"ALTER TABLE crypto_5m_signals ADD COLUMN {name} {ddl_type}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_crypto_5m_symbol_status ON crypto_5m_signals(symbol, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_crypto_5m_expiry ON crypto_5m_signals(expiry_time_ms)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_crypto_5m_recommendation ON crypto_5m_signals(recommendation)")
        conn.commit()
    return {"db_path": str(path)}


def _signal_db_row(record: Dict[str, Any]) -> tuple[Any, ...]:
    strategy = record.get("strategy") or {}
    features = record.get("features") or {}
    prediction_correct = record.get("prediction_correct")
    if prediction_correct is None:
        correct_value = None
    else:
        correct_value = 1 if prediction_correct else 0
    return (
        _signal_record_id(record),
        str(record.get("type") or "crypto_5m_signal"),
        record.get("logged_at"),
        str(record.get("symbol") or "UNKNOWN"),
        record.get("status"),
        record.get("strategy_name", "adaptive_model"),
        record.get("recommendation"),
        record.get("predicted_outcome"),
        record.get("actual_outcome"),
        correct_value,
        int(record.get("start_time_ms") or 0),
        int(record.get("expiry_time_ms") or 0),
        int(record.get("horizon_minutes") or 5),
        _to_float(record.get("target_price")),
        _to_float(record.get("resolved_price")),
        _to_float(record.get("probability_up")),
        _to_float(record.get("market_probability_up")),
        _to_float(record.get("edge")),
        _to_float(record.get("fee_bps")),
        _to_float(strategy.get("edge_threshold")),
        _to_float(features.get("realized_vol_1m")),
        _to_float(features.get("momentum_5m")),
        features.get("ml_mode_effective"),
        _to_float(record.get("pnl_per_contract")),
        json.dumps(record, sort_keys=True),
    )


def upsert_crypto_5m_signal_records(
    records: List[Dict[str, Any]],
    *,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Upsert signal records into SQLite, keyed by symbol/start/expiry/target."""
    path = db_path or DEFAULT_SIGNAL_DB_PATH
    init_crypto_5m_signal_db(db_path=path)
    signal_records = [record for record in records if record.get("type") == "crypto_5m_signal"]
    with closing(sqlite3.connect(path)) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO crypto_5m_signals (
                id, type, logged_at, symbol, status, strategy_name, recommendation,
                predicted_outcome, actual_outcome, prediction_correct,
                start_time_ms, expiry_time_ms, horizon_minutes,
                target_price, resolved_price, probability_up, market_probability_up,
                edge, fee_bps, edge_threshold, realized_vol_1m, momentum_5m,
                ml_mode_effective, pnl_per_contract, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [_signal_db_row(record) for record in signal_records])
        conn.commit()
    return {"db_path": str(path), "upserted": len(signal_records)}


def import_crypto_5m_signal_log_to_db(
    *,
    path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Import the JSONL signal log into SQLite."""
    log_path = path or DEFAULT_SIGNAL_LOG_PATH
    records = _jsonl_records(log_path)
    result = upsert_crypto_5m_signal_records(records, db_path=db_path)
    result["path"] = str(log_path)
    result["records"] = len(records)
    return result


def _signal_db_records(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT raw_json FROM crypto_5m_signals ORDER BY start_time_ms, symbol"
        ).fetchall()
    records: List[Dict[str, Any]] = []
    for (raw_json,) in rows:
        try:
            payload = json.loads(raw_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _bucket_abs(value: Any, cuts: List[float], labels: List[str]) -> str:
    number = _to_float(value)
    if number is None:
        return "unknown"
    magnitude = abs(number)
    for cut, label in zip(cuts, labels):
        if magnitude < cut:
            return label
    return labels[-1]


def _bucket_signed(value: Any, *, small: float, large: float) -> str:
    number = _to_float(value)
    if number is None:
        return "unknown"
    if number <= -large:
        return "strong_down"
    if number <= -small:
        return "down"
    if number < small:
        return "flat"
    if number < large:
        return "up"
    return "strong_up"


def _bucket_vol(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "unknown"
    if number < 0.0004:
        return "low"
    if number < 0.0008:
        return "normal"
    if number < 0.0014:
        return "high"
    return "extreme"


def _bucket_validation_accuracy(record: Dict[str, Any]) -> str:
    features = record.get("features") or {}
    validation = features.get("ml_validation") or {}
    accuracy = _to_float(validation.get("accuracy"))
    if accuracy is None:
        return "unknown"
    if accuracy < 0.45:
        return "bad"
    if accuracy < 0.53:
        return "coinflip"
    if accuracy < 0.60:
        return "good"
    return "strong"


def _new_analysis_group() -> Dict[str, Any]:
    return {
        "signals": 0,
        "resolved": 0,
        "open": 0,
        "scored": 0,
        "correct": 0,
        "trades": 0,
        "trade_wins": 0,
        "pnl_values": [],
        "baseline": {
            "always_up": [0, 0],
            "always_down": [0, 0],
            "follow_5m_momentum": [0, 0],
            "fade_5m_momentum": [0, 0],
            "follow_1m_return": [0, 0],
            "fade_1m_return": [0, 0],
        },
    }


def _score_baseline(group: Dict[str, Any], name: str, predicted: Optional[str], actual: str) -> None:
    if predicted not in {"up", "down"} or actual not in {"up", "down"}:
        return
    group["baseline"][name][0] += 1
    if predicted == actual:
        group["baseline"][name][1] += 1


def _add_analysis_record(group: Dict[str, Any], record: Dict[str, Any]) -> None:
    if record.get("type") != "crypto_5m_signal":
        return
    group["signals"] += 1
    if record.get("status") == "resolved":
        group["resolved"] += 1
    else:
        group["open"] += 1
        return

    actual = record.get("actual_outcome")
    predicted = record.get("predicted_outcome")
    if actual in {"up", "down", "tie"} and predicted in {"up", "down", "tie"}:
        group["scored"] += 1
        if actual == predicted:
            group["correct"] += 1

    side = _trade_side(record)
    pnl = _to_float(record.get("pnl_per_contract"))
    if side in {"up", "down"} and pnl is not None:
        group["trades"] += 1
        group["pnl_values"].append(pnl)
        if actual == side:
            group["trade_wins"] += 1

    if actual not in {"up", "down"}:
        return
    features = record.get("features") or {}
    momentum_5m = _to_float(features.get("momentum_5m"))
    reversal_1m_z = _to_float(features.get("reversal_1m_z"))
    _score_baseline(group, "always_up", "up", actual)
    _score_baseline(group, "always_down", "down", actual)
    if momentum_5m is not None and momentum_5m != 0:
        follow = "up" if momentum_5m > 0 else "down"
        _score_baseline(group, "follow_5m_momentum", follow, actual)
        _score_baseline(group, "fade_5m_momentum", "down" if follow == "up" else "up", actual)
    if reversal_1m_z is not None and reversal_1m_z != 0:
        # reversal_1m_z is negative last-return z. Flip sign to recover the last return direction.
        follow = "up" if reversal_1m_z < 0 else "down"
        _score_baseline(group, "follow_1m_return", follow, actual)
        _score_baseline(group, "fade_1m_return", "down" if follow == "up" else "up", actual)


def _finalize_analysis_group(group: Dict[str, Any]) -> Dict[str, Any]:
    pnl_values = [float(value) for value in group["pnl_values"]]
    total_pnl = sum(pnl_values)
    baselines = {}
    for name, (n, correct) in group["baseline"].items():
        baselines[name] = {
            "n": n,
            "accuracy": round(correct / n, 4) if n else None,
        }
    return {
        "signals": group["signals"],
        "open": group["open"],
        "resolved": group["resolved"],
        "scored": group["scored"],
        "accuracy": round(group["correct"] / group["scored"], 4) if group["scored"] else None,
        "trades": group["trades"],
        "trade_rate": round(group["trades"] / group["resolved"], 4) if group["resolved"] else 0.0,
        "trade_hit_rate": round(group["trade_wins"] / group["trades"], 4) if group["trades"] else None,
        "pnl_per_contract": round(total_pnl, 6),
        "avg_pnl_per_trade": round(total_pnl / group["trades"], 6) if group["trades"] else None,
        "baselines": baselines,
    }


def analyze_crypto_5m_signal_db(
    *,
    db_path: Optional[Path] = None,
    min_group_signals: int = 1,
) -> Dict[str, Any]:
    """Analyze the SQLite paper-signal dataset by regime and simple baselines."""
    path = db_path or DEFAULT_SIGNAL_DB_PATH
    records = _signal_db_records(path)
    min_size = max(1, int(min_group_signals or 1))
    overall = _new_analysis_group()
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {
        "by_strategy": {},
        "by_symbol": {},
        "by_threshold": {},
        "by_recommendation": {},
        "by_predicted_outcome": {},
        "by_edge_bucket": {},
        "by_volatility_bucket": {},
        "by_momentum_5m_bucket": {},
        "by_validation_accuracy_bucket": {},
    }

    def add_group(section: str, key: Any, record: Dict[str, Any]) -> None:
        key_str = str(key if key is not None else "unknown")
        group = groups[section].setdefault(key_str, _new_analysis_group())
        _add_analysis_record(group, record)

    for record in records:
        if record.get("type") != "crypto_5m_signal":
            continue
        features = record.get("features") or {}
        _add_analysis_record(overall, record)
        add_group("by_strategy", record.get("strategy_name", "adaptive_model"), record)
        add_group("by_symbol", record.get("symbol"), record)
        add_group("by_threshold", (record.get("strategy") or {}).get("edge_threshold"), record)
        add_group("by_recommendation", record.get("recommendation"), record)
        add_group("by_predicted_outcome", record.get("predicted_outcome"), record)
        add_group(
            "by_edge_bucket",
            _bucket_abs(record.get("edge"), [0.01, 0.03, 0.05], ["<1pp", "1-3pp", "3-5pp", ">=5pp"]),
            record,
        )
        add_group("by_volatility_bucket", _bucket_vol(features.get("realized_vol_1m")), record)
        add_group(
            "by_momentum_5m_bucket",
            _bucket_signed(features.get("momentum_5m"), small=0.0003, large=0.0010),
            record,
        )
        add_group("by_validation_accuracy_bucket", _bucket_validation_accuracy(record), record)

    finalized_groups: Dict[str, Dict[str, Any]] = {}
    for section, section_groups in groups.items():
        finalized_groups[section] = {
            key: metrics
            for key, metrics in sorted(
                ((key, _finalize_analysis_group(group)) for key, group in section_groups.items()),
                key=lambda item: item[0],
            )
            if metrics["signals"] >= min_size
        }
    return {
        "db_path": str(path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_group_signals": min_size,
        "overall": _finalize_analysis_group(overall),
        **finalized_groups,
    }


def _replay_candidate_side(record: Dict[str, Any], candidate: str) -> Optional[str]:
    features = record.get("features") or {}
    momentum = _to_float(features.get("momentum_5m"))
    realized_vol = _to_float(features.get("realized_vol_1m"))
    edge = _to_float(record.get("edge"))
    if candidate == "inverse_edge_004":
        if edge is None or abs(edge) < 0.04:
            return None
        return "down" if edge > 0 else "up"
    if candidate == "follow_mom_0025_no_extreme_vol":
        if momentum is None or abs(momentum) < 0.0025:
            return None
        if realized_vol is not None and realized_vol >= 0.0014:
            return None
        return "up" if momentum > 0 else "down"
    if candidate.startswith("follow_mom_"):
        threshold = float(candidate.split("_")[-1]) / 10000.0
        if momentum is None or abs(momentum) < threshold:
            return None
        return "up" if momentum > 0 else "down"
    if candidate.startswith("fade_mom_"):
        threshold = float(candidate.split("_")[-1]) / 10000.0
        if momentum is None or abs(momentum) < threshold:
            return None
        return "down" if momentum > 0 else "up"
    return None


def _candidate_threshold_key(value: float) -> str:
    return f"{int(round(value * 10000)):04d}"


def crypto_5m_per_asset_regime_report(
    *,
    db_path: Optional[Path] = None,
    min_trades: int = 20,
) -> Dict[str, Any]:
    """Replay simple expert strategies by asset and collection slice.

    This is an evidence report for the paper-only per-asset regime selector. It
    does not fit a flexible model; it ranks explicit experts so the strategy
    remains auditable while the dataset is small.
    """
    path = db_path or DEFAULT_SIGNAL_DB_PATH
    records = [
        record for record in _signal_db_records(path)
        if record.get("type") == "crypto_5m_signal"
        and record.get("status") == "resolved"
        and record.get("actual_outcome") in {"up", "down"}
    ]
    thresholds = [0.0015, 0.0020, 0.0025, 0.0030]
    candidates = ["inverse_edge_004"]
    candidates += [f"follow_mom_{_candidate_threshold_key(threshold)}" for threshold in thresholds]
    candidates += [f"fade_mom_{_candidate_threshold_key(threshold)}" for threshold in thresholds]
    candidates += ["follow_mom_0025_no_extreme_vol"]
    slices = {
        "all": records,
        "adaptive_model": [r for r in records if r.get("strategy_name", "adaptive_model") == "adaptive_model"],
        "fade_5m_momentum": [r for r in records if r.get("strategy_name") == "fade_5m_momentum"],
        "follow_5m_momentum": [r for r in records if r.get("strategy_name") == "follow_5m_momentum"],
    }

    def score(rows: List[Dict[str, Any]], symbol: str, candidate: str) -> Dict[str, Any]:
        trades = 0
        wins = 0
        pnl_values: List[float] = []
        for record in rows:
            if normalize_symbol(str(record.get("symbol") or "")) != symbol:
                continue
            side = _replay_candidate_side(record, candidate)
            pnl = _contract_pnl(
                side=side or "none",
                actual_outcome=str(record.get("actual_outcome") or ""),
                market_probability_up=record.get("market_probability_up"),
                fee_bps=float(record.get("fee_bps") or 0.0),
            )
            if pnl is None:
                continue
            trades += 1
            pnl_values.append(float(pnl))
            if side == record.get("actual_outcome"):
                wins += 1
        total_pnl = sum(pnl_values)
        return {
            "trades": trades,
            "hit_rate": round(wins / trades, 4) if trades else None,
            "pnl_per_contract": round(total_pnl, 6),
            "avg_pnl_per_trade": round(total_pnl / trades, 6) if trades else None,
        }

    symbols = sorted({
        normalize_symbol(str(record.get("symbol") or ""))
        for record in records
        if record.get("symbol")
    })
    by_symbol: Dict[str, Any] = {}
    min_count = max(1, int(min_trades or 1))
    for symbol in symbols:
        candidate_rows: Dict[str, Any] = {}
        for candidate in candidates:
            slice_scores = {
                slice_name: score(slice_rows, symbol, candidate)
                for slice_name, slice_rows in slices.items()
            }
            candidate_rows[candidate] = slice_scores

        eligible = []
        for candidate, slice_scores in candidate_rows.items():
            all_score = slice_scores["all"]
            current_score = slice_scores["follow_5m_momentum"]
            if all_score["trades"] < min_count:
                continue
            positive_slices = sum(
                1 for name, metrics in slice_scores.items()
                if name != "all" and metrics["trades"] > 0 and metrics["pnl_per_contract"] > 0
            )
            eligible.append((
                current_score["pnl_per_contract"],
                positive_slices,
                all_score["pnl_per_contract"],
                all_score["trades"],
                candidate,
            ))
        selected = None
        if eligible:
            eligible.sort(reverse=True)
            best = eligible[0][-1]
            metrics = candidate_rows[best]
            selected = {
                "candidate": best,
                "all": metrics["all"],
                "current_oos": metrics["follow_5m_momentum"],
                "positive_nonempty_slices": sum(
                    1 for name, score_row in metrics.items()
                    if name != "all" and score_row["trades"] > 0 and score_row["pnl_per_contract"] > 0
                ),
            }
        by_symbol[symbol] = {
            "selected": selected,
            "candidates": candidate_rows,
        }

    return {
        "db_path": str(path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_trades": min_count,
        "rows": len(records),
        "slices": {name: len(rows) for name, rows in slices.items()},
        "by_symbol": by_symbol,
        "implemented_selector": {
            "BTCUSDT": ["inverse_edge_004"],
            "ETHUSDT": ["fade_mom_0015"],
            "SOLUSDT": ["hold_until_stable"],
        },
    }


def _record_ott_trend_sides(records: List[Dict[str, Any]]) -> Dict[str, str]:
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        symbol = normalize_symbol(str(record.get("symbol") or ""))
        price = _to_float(record.get("target_price") or record.get("current_price"))
        if price is None or price <= 0.0:
            continue
        by_symbol.setdefault(symbol, []).append(record)
    trend_by_id: Dict[str, str] = {}
    for symbol_records in by_symbol.values():
        symbol_records.sort(key=lambda row: (int(row.get("start_time_ms") or 0), str(row.get("id") or "")))
        candles = []
        for record in symbol_records:
            price = float(record.get("target_price") or record.get("current_price"))
            candles.append({
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
            })
        for record, ott_row in zip(symbol_records, compute_ott(candles, length=2, percent=1.4)):
            trend_by_id[_signal_record_id(record)] = str(ott_row.get("trend_side") or "")
    return trend_by_id


def _selector_btc_inverse_eth_fade_side(record: Dict[str, Any]) -> Optional[str]:
    symbol = normalize_symbol(str(record.get("symbol") or ""))
    features = record.get("features") or {}
    momentum = _to_float(features.get("momentum_5m"))
    edge = _to_float(record.get("edge"))
    if symbol == "BTCUSDT" and edge is not None and abs(edge) >= 0.04:
        return "down" if edge > 0 else "up"
    if symbol == "ETHUSDT" and momentum is not None and abs(momentum) >= 0.0015:
        return "down" if momentum > 0 else "up"
    return None


def _crypto_replay_side(
    record: Dict[str, Any],
    candidate: str,
    *,
    ott_trend_by_id: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    symbol = normalize_symbol(str(record.get("symbol") or ""))
    features = record.get("features") or {}
    momentum = _to_float(features.get("momentum_5m"))
    edge = _to_float(record.get("edge"))
    if candidate == "btc_inverse_edge_004":
        if symbol != "BTCUSDT" or edge is None or abs(edge) < 0.04:
            return None
        return "down" if edge > 0 else "up"
    if candidate == "eth_fade_mom_0015":
        if symbol != "ETHUSDT" or momentum is None or abs(momentum) < 0.0015:
            return None
        return "down" if momentum > 0 else "up"
    if candidate == "sol_fade_mom_0015":
        if symbol != "SOLUSDT" or momentum is None or abs(momentum) < 0.0015:
            return None
        return "down" if momentum > 0 else "up"
    if candidate == "btc_eth_inverse_edge_004":
        if symbol not in {"BTCUSDT", "ETHUSDT"} or edge is None or abs(edge) < 0.04:
            return None
        return "down" if edge > 0 else "up"
    if candidate == "pooled_fade_mom_0015":
        if symbol not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"} or momentum is None or abs(momentum) < 0.0015:
            return None
        return "down" if momentum > 0 else "up"
    if candidate == "selector_btc_inverse_eth_fade":
        return _selector_btc_inverse_eth_fade_side(record)
    if candidate == "selector_ott_filter":
        side = _selector_btc_inverse_eth_fade_side(record)
        if side is None:
            return None
        trend = (ott_trend_by_id or {}).get(_signal_record_id(record))
        return side if trend == side else None
    return None


def _load_crypto_5m_equity_fallback(db_path: Path) -> Optional[Dict[str, Any]]:
    # Freshest committed source first: the GitHub tick regenerates + commits the
    # equity payload every ~15 min, so the raw-GitHub copy is the current one.
    remote_url = os.environ.get("CRYPTO_5M_EQUITY_URL", DEFAULT_EQUITY_REMOTE_URL).strip()
    if remote_url:
        try:
            import requests

            headers = {
                **_HEADERS,
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
            resp = requests.get(_cache_busted_url(remote_url), headers=headers, timeout=5)
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, dict):
                    out = dict(payload)
                    out["fallback"] = True
                    out["fallback_source"] = remote_url
                    out["db_path"] = str(db_path)
                    return out
        except Exception:
            pass
    # Bundled snapshot baked into the image at deploy time.
    if DEFAULT_EQUITY_FALLBACK_PATH.exists():
        try:
            payload = json.loads(DEFAULT_EQUITY_FALLBACK_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            out = dict(payload)
            out["fallback"] = True
            out["fallback_source"] = str(DEFAULT_EQUITY_FALLBACK_PATH)
            out["db_path"] = str(db_path)
            out["note"] = (
                str(out.get("note") or "")
                + " Static deployment snapshot; live collector DB was not present in this container."
            ).strip()
            return out
    # Last resort: a gcsfuse-mounted bucket file. This was the freshest path while
    # the Cloud Run collector job was alive; that job has been removed, so the file
    # is only used if both committed sources above are unavailable.
    mounted = os.environ.get("CRYPTO_5M_EQUITY_FILE", "").strip()
    if mounted:
        mounted_path = Path(mounted)
        if mounted_path.exists():
            try:
                payload = json.loads(mounted_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                out = dict(payload)
                out["fallback"] = True
                out["fallback_source"] = str(mounted_path)
                out["db_path"] = str(db_path)
                return out
    return None


def crypto_5m_candidate_equity(
    *,
    db_path: Optional[Path] = None,
    since_hours: Optional[float] = 72.0,
) -> Dict[str, Any]:
    """Return paper equity curves for auditable 5-minute crypto strategy candidates."""
    path = db_path or DEFAULT_SIGNAL_DB_PATH
    records = [
        record for record in _signal_db_records(path)
        if record.get("type") == "crypto_5m_signal"
        and record.get("status") == "resolved"
        and record.get("actual_outcome") in {"up", "down"}
    ]
    if not records and db_path is None:
        fallback = _load_crypto_5m_equity_fallback(path)
        if fallback is not None:
            return fallback
    cutoff_ms = None
    if since_hours is not None and float(since_hours) > 0:
        latest = max((int(record.get("start_time_ms") or 0) for record in records), default=0)
        cutoff_ms = latest - int(float(since_hours) * 60 * 60 * 1000)
        records = [record for record in records if int(record.get("start_time_ms") or 0) >= cutoff_ms]
    records.sort(key=lambda record: (int(record.get("start_time_ms") or 0), str(record.get("symbol") or "")))
    ott_trend_by_id = _record_ott_trend_sides(records)

    candidates = [
        ("selector_btc_inverse_eth_fade", "Recommended: BTC inverse + ETH fade"),
        ("selector_ott_filter", "Trend-filtered selector"),
        ("btc_inverse_edge_004", "BTC inverse edge >=4pp"),
        ("eth_fade_mom_0015", "ETH fade |5m mom| >=0.15%"),
    ]
    colors = {
        "selector_btc_inverse_eth_fade": "#34d399",
        "selector_ott_filter": "#8b5cf6",
        "btc_inverse_edge_004": "#60a5fa",
        "eth_fade_mom_0015": "#f59e0b",
    }
    curves: List[Dict[str, Any]] = []
    for key, label in candidates:
        running = 0.0
        equity: List[float] = []
        times: List[int] = []
        wins = 0
        trades = 0
        by_symbol: Dict[str, Dict[str, Any]] = {}
        last_trade_at = None
        first_trade_at = None
        first_trade_ms = None
        for record in records:
            side = _crypto_replay_side(record, key, ott_trend_by_id=ott_trend_by_id)
            pnl = _contract_pnl(
                side=side or "none",
                actual_outcome=str(record.get("actual_outcome") or ""),
                market_probability_up=record.get("market_probability_up"),
                fee_bps=float(record.get("fee_bps") or 0.0),
            )
            if pnl is None:
                continue
            symbol = normalize_symbol(str(record.get("symbol") or ""))
            trades += 1
            if side == record.get("actual_outcome"):
                wins += 1
            running += float(pnl)
            equity.append(round(running, 6))
            times.append(int(record.get("start_time_ms") or 0))
            last_trade_at = record.get("resolved_at") or record.get("logged_at")
            if first_trade_at is None:
                first_trade_at = record.get("logged_at") or record.get("resolved_at")
                first_trade_ms = int(record.get("start_time_ms") or 0) or None
            sym = by_symbol.setdefault(symbol, {"trades": 0, "wins": 0, "pnl_per_contract": 0.0})
            sym["trades"] += 1
            sym["wins"] += 1 if side == record.get("actual_outcome") else 0
            sym["pnl_per_contract"] += float(pnl)
        if not trades:
            continue
        for summary in by_symbol.values():
            summary["hit_rate"] = round(summary["wins"] / summary["trades"], 4) if summary["trades"] else None
            summary["pnl_per_contract"] = round(summary["pnl_per_contract"], 6)
            summary.pop("wins", None)
        # Max drawdown of the cumulative-PnL curve (peak-to-trough), and a
        # per-day return so the long-term durability of the strategy is legible.
        peak = 0.0
        max_drawdown = 0.0
        for value in equity:
            peak = max(peak, value)
            max_drawdown = max(max_drawdown, peak - value)
        last_ms = None
        for record in reversed(records):
            if _crypto_replay_side(record, key, ott_trend_by_id=ott_trend_by_id) in {"up", "down"}:
                last_ms = int(record.get("start_time_ms") or 0) or None
                break
        span_days = None
        if first_trade_ms and last_ms and last_ms > first_trade_ms:
            span_days = round((last_ms - first_trade_ms) / (24 * 60 * 60 * 1000), 2)
        curves.append({
            "key": key,
            "label": label,
            "color": colors[key],
            "points": equity,
            "times": times,
            "trades": trades,
            "hit_rate": round(wins / trades, 4),
            "pnl_per_contract": round(running, 6),
            "avg_pnl_per_trade": round(running / trades, 6),
            "max_drawdown": round(max_drawdown, 6),
            "span_days": span_days,
            "pnl_per_day": round(running / span_days, 6) if span_days else None,
            "first_trade_at": first_trade_at,
            "last_trade_at": last_trade_at,
            "by_symbol": dict(sorted(by_symbol.items())),
        })
    recommended = next((curve for curve in curves if curve["key"] == "selector_btc_inverse_eth_fade"), None)
    earliest_ms = min((int(r.get("start_time_ms") or 0) for r in records), default=0) or None
    latest_ms = max((int(r.get("start_time_ms") or 0) for r in records), default=0) or None
    coverage_days = None
    if earliest_ms and latest_ms and latest_ms > earliest_ms:
        coverage_days = round((latest_ms - earliest_ms) / (24 * 60 * 60 * 1000), 2)
    return {
        "db_path": str(path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "since_hours": since_hours,
        "cutoff_start_time_ms": cutoff_ms,
        "resolved_rows": len(records),
        "coverage": {
            "first_row_time_ms": earliest_ms,
            "last_row_time_ms": latest_ms,
            "days": coverage_days,
        },
        "curves": curves,
        "recommendation": {
            "policy": "Main paper strategy: trade against very strong BTC model calls, fade sharp ETH 5-minute moves, and skip SOL until its rule is stable.",
            "candidate": "selector_btc_inverse_eth_fade",
            "trades": recommended.get("trades") if recommended else 0,
            "hit_rate": recommended.get("hit_rate") if recommended else None,
            "pnl_per_contract": recommended.get("pnl_per_contract") if recommended else 0.0,
            "status": (
                "paper-positive"
                if recommended and recommended.get("pnl_per_contract", 0.0) > 0
                else "collecting"
            ),
        },
        "note": "Paper replay from live-collected 5-minute crypto rows. Uses 50c contract pricing in the collector; no slippage.",
    }


def _contract_pnl(
    *,
    side: str,
    actual_outcome: str,
    market_probability_up: Optional[float],
    fee_bps: float,
) -> Optional[float]:
    market_up = _normalize_market_probability(market_probability_up)
    if market_up is None or side not in {"up", "down"} or actual_outcome not in {"up", "down"}:
        return None
    fee = max(0.0, float(fee_bps or 0.0)) / 10000.0
    price = market_up if side == "up" else 1.0 - market_up
    win = side == actual_outcome
    return round((1.0 - price if win else -price) - fee, 6)


def _is_trade_recommendation(record: Dict[str, Any]) -> bool:
    return record.get("recommendation") in {"buy_up", "buy_down"}


def _trade_side(record: Dict[str, Any]) -> str:
    if not _is_trade_recommendation(record):
        return "none"
    side = (record.get("strategy") or {}).get("side")
    return side if side in {"up", "down"} else "none"


def _new_signal_summary_group() -> Dict[str, Any]:
    return {
        "signals": 0,
        "open": 0,
        "resolved": 0,
        "scored": 0,
        "correct": 0,
        "trades": 0,
        "trade_wins": 0,
        "pnl_values": [],
        "equity": [],
    }


def _add_signal_summary_record(group: Dict[str, Any], record: Dict[str, Any]) -> None:
    if record.get("type") != "crypto_5m_signal":
        return
    group["signals"] += 1
    if record.get("status") == "resolved":
        group["resolved"] += 1
    else:
        group["open"] += 1
        return

    if record.get("prediction_correct") is not None:
        group["scored"] += 1
        group["correct"] += 1 if record.get("prediction_correct") else 0

    side = _trade_side(record)
    pnl = _to_float(record.get("pnl_per_contract"))
    if side not in {"up", "down"} or pnl is None:
        return
    group["trades"] += 1
    group["pnl_values"].append(float(pnl))
    running = (group["equity"][-1] if group["equity"] else 0.0) + float(pnl)
    group["equity"].append(running)
    if record.get("actual_outcome") == side:
        group["trade_wins"] += 1


def _finalize_signal_summary_group(group: Dict[str, Any]) -> Dict[str, Any]:
    pnl_values = [float(value) for value in group["pnl_values"]]
    total_pnl = sum(pnl_values)
    return {
        "signals": group["signals"],
        "open": group["open"],
        "resolved": group["resolved"],
        "scored": group["scored"],
        "accuracy": round(group["correct"] / group["scored"], 4) if group["scored"] else None,
        "trades": group["trades"],
        "trade_rate": round(group["trades"] / group["resolved"], 4) if group["resolved"] else 0.0,
        "trade_hit_rate": round(group["trade_wins"] / group["trades"], 4) if group["trades"] else None,
        "pnl_per_contract": round(total_pnl, 6),
        "avg_pnl_per_trade": round(total_pnl / group["trades"], 6) if group["trades"] else None,
        "max_drawdown": round(_max_drawdown(group["equity"]), 6) if group["equity"] else 0.0,
    }


def summarize_crypto_5m_signal_log(
    *,
    path: Optional[Path] = None,
    min_resolved_trades: int = 200,
    min_total_pnl: float = 0.0,
    min_hit_rate: float = 0.53,
) -> Dict[str, Any]:
    """Summarize resolved paper signals as a trading-journal PnL report."""
    log_path = path or DEFAULT_SIGNAL_LOG_PATH
    records = _jsonl_records(log_path)
    overall = _new_signal_summary_group()
    by_symbol: Dict[str, Dict[str, Any]] = {}
    by_recommendation: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if record.get("type") != "crypto_5m_signal":
            continue
        symbol = str(record.get("symbol") or "UNKNOWN")
        recommendation = str(record.get("recommendation") or "unknown")
        _add_signal_summary_record(overall, record)
        _add_signal_summary_record(by_symbol.setdefault(symbol, _new_signal_summary_group()), record)
        _add_signal_summary_record(
            by_recommendation.setdefault(recommendation, _new_signal_summary_group()),
            record,
        )

    summary = _finalize_signal_summary_group(overall)
    trade_count_required = max(1, int(min_resolved_trades or 1))
    pnl_required = float(min_total_pnl or 0.0)
    hit_rate_required = max(0.0, min(float(min_hit_rate or 0.0), 1.0))
    avg_pnl = summary["avg_pnl_per_trade"]
    hit_rate = summary["trade_hit_rate"]
    trade_ready = (
        summary["trades"] >= trade_count_required
        and summary["pnl_per_contract"] > pnl_required
        and avg_pnl is not None and avg_pnl > 0.0
        and hit_rate is not None and hit_rate >= hit_rate_required
    )
    if summary["trades"] < trade_count_required:
        reason = "not enough resolved trades"
    elif summary["pnl_per_contract"] <= pnl_required or avg_pnl is None or avg_pnl <= 0.0:
        reason = "paper-trade PnL is not positive enough"
    elif hit_rate is None or hit_rate < hit_rate_required:
        reason = "trade hit rate is below the readiness threshold"
    else:
        reason = "paper-trade evidence clears the configured thresholds"
    return {
        "path": str(log_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria": {
            "min_resolved_trades": trade_count_required,
            "min_total_pnl_per_contract": round(pnl_required, 6),
            "min_trade_hit_rate": round(hit_rate_required, 4),
        },
        "trade_ready": trade_ready,
        "readiness_reason": reason,
        **summary,
        "by_symbol": {
            symbol: _finalize_signal_summary_group(group)
            for symbol, group in sorted(by_symbol.items())
        },
        "by_recommendation": {
            recommendation: _finalize_signal_summary_group(group)
            for recommendation, group in sorted(by_recommendation.items())
        },
    }


def record_crypto_5m_signal(
    symbol: str = "BTCUSDT",
    *,
    target_price: Optional[float] = None,
    horizon_minutes: int = 5,
    lookback_minutes: int = 240,
    market_probability: Optional[float] = 0.5,
    fee_bps: float = 0.0,
    ml_mode: str = "fixed",
    training_window: int = 500,
    edge_threshold: float = 0.03,
    strategy_mode: str = "adaptive_model",
    momentum_threshold: float = 0.002,
    path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    klines: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Record a live/paper signal for later 5-minute outcome resolution."""
    symbol_norm = normalize_symbol(symbol)
    mode = (ml_mode or "fixed").strip().lower()
    lookback = max(30, min(int(lookback_minutes or 240), 1000))
    horizon = max(1, min(int(horizon_minutes or 5), 15))
    train_window = max(40, min(int(training_window or 500), 900))
    fetch_limit = lookback if mode == "fixed" else min(1000, max(lookback, lookback + horizon + train_window + 1))
    raw = klines if klines is not None else _fetch_klines(symbol_norm, limit=fetch_limit)
    timed = _parse_timed_candles(raw)
    if not timed:
        raise CryptoModelError("Need timed candles to record a 5-minute signal.")
    start_ms = int(timed[-1]["open_time_ms"])
    expiry_ms = start_ms + horizon * _ONE_MINUTE_MS
    forecast = forecast_crypto_5m(
        symbol_norm,
        target_price=target_price,
        horizon_minutes=horizon,
        lookback_minutes=lookback,
        market_probability=market_probability,
        fee_bps=fee_bps,
        ml_mode=mode,
        training_window=train_window,
        edge_threshold=edge_threshold,
        klines=raw,
    )
    applied, strategy_name = _apply_crypto_5m_paper_strategy(
        forecast,
        strategy_mode=strategy_mode,
        momentum_threshold=momentum_threshold,
        fee_bps=fee_bps,
    )
    record = {
        "type": "crypto_5m_signal",
        "status": "open",
        "strategy_name": strategy_name,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol_norm,
        "start_time_ms": start_ms,
        "expiry_time_ms": expiry_ms,
        "horizon_minutes": horizon,
        "target_price": applied["target_price"],
        "market_probability_up": applied["market_probability_up"],
        "fee_bps": fee_bps,
        "predicted_outcome": applied["predicted_outcome"],
        "probability_up": applied["probability_up"],
        "edge": applied["edge"],
        "recommendation": applied["recommendation"],
        "strategy": applied["strategy"],
        "component_probabilities": applied["component_probabilities"],
        "features": applied["features"],
    }
    log_path = path or DEFAULT_SIGNAL_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    db_result = upsert_crypto_5m_signal_records([record], db_path=db_path) if db_path else None
    return {"path": str(log_path), "db": db_result, "record": record}


DEFAULT_BACKFILL_LOG_PATH = Path("data/crypto_5m_backfill_log.jsonl")


def backfill_crypto_5m_signals(
    symbols: Optional[List[str]] = None,
    *,
    days: float = 30.0,
    horizon_minutes: int = 5,
    lookback_minutes: int = 60,
    market_probability: Optional[float] = 0.50,
    fee_bps: float = 2.0,
    ml_mode: str = "adaptive",
    training_window: int = 120,
    strategy_mode: str = "per_asset_regime_selector",
    momentum_threshold: float = 0.0025,
    step_minutes: Optional[int] = None,
    max_candles: int = 50_000,
    end_time_ms: Optional[int] = None,
    path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reconstruct a long paper track record by replaying the live signal logic
    over historical Binance klines.

    For each historical 5-minute bar we hand the *same* trailing window the live
    collector would have fetched to :func:`record_crypto_5m_signal`, so the
    backfilled rows are schema-identical to live ones and land in the same signal
    DB. Outcomes are then resolved from the known future candles. This lets the
    equity curve show months of behaviour instead of only the live-collected
    window, so a strategy's long-term character is visible.
    """
    symbols = symbols or ["BTC", "ETH", "SOL"]
    horizon = max(1, min(int(horizon_minutes or 5), 15))
    lookback = max(30, min(int(lookback_minutes or 60), 1000))
    mode = (ml_mode or "fixed").strip().lower()
    train_window = max(40, min(int(training_window or 120), 900))
    step = max(1, int(step_minutes or horizon))
    # The live collector fetches this many trailing 1-minute candles per tick;
    # replay each historical bar with the same window so features/training match.
    window = lookback if mode == "fixed" else min(1000, max(lookback, lookback + horizon + train_window + 1))
    log_path = path or DEFAULT_BACKFILL_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Start each backfill run from a clean log so re-runs are idempotent (the DB
    # upsert is keyed by signal id, so re-importing the same bars is harmless).
    _write_jsonl(log_path, [])

    per_symbol: Dict[str, Any] = {}
    total_recorded = 0
    klines_by_symbol: Dict[str, List[Any]] = {}
    for symbol in symbols:
        symbol_norm = normalize_symbol(symbol)
        raw = fetch_klines_history(
            symbol_norm, days=days, end_time_ms=end_time_ms, max_candles=max_candles
        )
        klines_by_symbol[symbol_norm] = raw
        if len(raw) < window + horizon + 1:
            per_symbol[symbol_norm] = {"candles": len(raw), "recorded": 0, "skipped": "insufficient history"}
            continue
        recorded = 0
        # Need `window` candles of context and `horizon` future candles to resolve.
        first = window - 1
        last = len(raw) - horizon - 1
        for i in range(first, last + 1, step):
            chunk = raw[i - window + 1: i + 1]
            try:
                record_crypto_5m_signal(
                    symbol_norm,
                    horizon_minutes=horizon,
                    lookback_minutes=lookback,
                    market_probability=market_probability,
                    fee_bps=fee_bps,
                    ml_mode=mode,
                    training_window=train_window,
                    strategy_mode=strategy_mode,
                    momentum_threshold=momentum_threshold,
                    path=log_path,
                    db_path=None,  # resolve step does the single DB upsert
                    klines=chunk,
                )
                recorded += 1
            except CryptoModelError:
                continue
        per_symbol[symbol_norm] = {"candles": len(raw), "recorded": recorded}
        total_recorded += recorded

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    resolved = resolve_crypto_5m_signal_log(
        path=log_path,
        db_path=db_path,
        now_ms=now_ms,
        klines_by_symbol=klines_by_symbol,
    )
    return {
        "path": str(log_path),
        "days": days,
        "symbols": per_symbol,
        "recorded": total_recorded,
        "resolved": resolved,
    }


def _kline_close_time_ms(row: Any) -> Optional[int]:
    """Close time of a raw Binance kline row, matching `_parse_timed_candles`."""
    if not isinstance(row, (list, tuple)) or len(row) < 6:
        return None
    if len(row) > 6:
        close_time = _to_float(row[6])
        if close_time is not None:
            return int(close_time)
    open_time = _to_float(row[0])
    return int(open_time) + _ONE_MINUTE_MS - 1 if open_time is not None else None


def _index_klines_by_close_time(rows: Optional[List[Any]]) -> tuple[List[int], List[Any]]:
    """Sort raw klines by close time so a settlement candle can be found with a
    bisect instead of an O(N) scan + full re-parse per resolution."""
    indexed = []
    for row in rows or []:
        close_ms = _kline_close_time_ms(row)
        if close_ms is not None:
            indexed.append((close_ms, row))
    indexed.sort(key=lambda item: item[0])
    return [ct for ct, _ in indexed], [row for _, row in indexed]


def resolve_crypto_5m_signal_log(
    *,
    path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    now_ms: Optional[int] = None,
    klines_by_symbol: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    """Resolve open paper signals in a JSONL log and rewrite the log in place."""
    log_path = path or DEFAULT_SIGNAL_LOG_PATH
    records = _jsonl_records(log_path)
    by_symbol = klines_by_symbol or {}
    # Index each symbol's candles by close time once, so resolving a record is a
    # bisect + tiny re-parse rather than an O(N) scan over every candle. This
    # turns large backfills (tens of thousands of rows) from O(rows×candles) into
    # roughly O(rows·log candles).
    index_cache: Dict[str, tuple] = {}

    def _settlement_window(symbol: str, expiry_ms: int) -> Optional[List[Any]]:
        if symbol not in index_cache:
            rows = by_symbol.get(symbol) or by_symbol.get(symbol.replace("USDT", ""))
            index_cache[symbol] = _index_klines_by_close_time(rows) if rows else ([], [])
        times, sorted_rows = index_cache[symbol]
        if not times:
            return None
        idx = bisect.bisect_right(times, expiry_ms) - 1
        if idx < 0:
            return []
        # A small slice (not just one row) tolerates any close-time edge cases.
        return sorted_rows[max(0, idx - 2): idx + 1]

    resolved_now = 0
    open_count = 0
    already_resolved = 0
    pnl_values: List[float] = []
    correct = 0
    scored = 0
    for record in records:
        if record.get("type") != "crypto_5m_signal":
            continue
        if record.get("status") == "resolved":
            already_resolved += 1
            pnl = record.get("pnl_per_contract")
            if _trade_side(record) in {"up", "down"} and pnl is not None:
                pnl_values.append(float(pnl))
            if record.get("prediction_correct") is not None:
                scored += 1
                correct += 1 if record.get("prediction_correct") else 0
            continue
        symbol = str(record.get("symbol") or "")
        klines = _settlement_window(symbol, int(record["expiry_time_ms"]))
        resolution = resolve_crypto_5m_outcome(
            symbol,
            target_price=float(record["target_price"]),
            start_time_ms=int(record["start_time_ms"]),
            expiry_time_ms=int(record["expiry_time_ms"]),
            predicted_outcome=record.get("predicted_outcome"),
            klines=klines,
            now_ms=now_ms,
        )
        if resolution["status"] == "pending":
            open_count += 1
            record["resolution"] = resolution
            continue
        side = _trade_side(record)
        pnl = _contract_pnl(
            side=side,
            actual_outcome=resolution["actual_outcome"],
            market_probability_up=record.get("market_probability_up"),
            fee_bps=float(record.get("fee_bps") or 0.0),
        )
        record.update({
            "status": "resolved",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "resolution": resolution,
            "actual_outcome": resolution["actual_outcome"],
            "resolved_price": resolution["resolved_price"],
            "prediction_correct": resolution["prediction_correct"],
            "pnl_per_contract": pnl,
        })
        resolved_now += 1
        if pnl is not None:
            pnl_values.append(float(pnl))
        if resolution["prediction_correct"] is not None:
            scored += 1
            correct += 1 if resolution["prediction_correct"] else 0
    _write_jsonl(log_path, records)
    db_result = upsert_crypto_5m_signal_records(records, db_path=db_path) if db_path else None
    return {
        "path": str(log_path),
        "db": db_result,
        "records": len(records),
        "resolved_now": resolved_now,
        "already_resolved": already_resolved,
        "open": open_count,
        "scored": scored,
        "accuracy": round(correct / scored, 4) if scored else None,
        "pnl_per_contract": round(sum(pnl_values), 6),
        "avg_pnl_per_contract": round(sum(pnl_values) / len(pnl_values), 6) if pnl_values else None,
    }


def run_crypto_5m_paper_loop(
    symbols: Optional[List[str]] = None,
    *,
    iterations: int = 1,
    sleep_seconds: float = 60.0,
    path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    market_probability: float = 0.5,
    fee_bps: float = 0.0,
    horizon_minutes: int = 5,
    lookback_minutes: int = 240,
    ml_mode: str = "fixed",
    training_window: int = 500,
    edge_threshold: float = 0.03,
    strategy_mode: str = "adaptive_model",
    momentum_threshold: float = 0.002,
    record_signals: bool = True,
    resolve_signals: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a small paper-trading loop that records and resolves 5-minute signals."""
    symbol_list = [normalize_symbol(symbol) for symbol in (symbols or ["BTCUSDT"])]
    loop_count = max(1, int(iterations or 1))
    log_path = path or DEFAULT_SIGNAL_LOG_PATH
    iteration_results: List[Dict[str, Any]] = []
    for index in range(loop_count):
        iteration: Dict[str, Any] = {
            "iteration": index + 1,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "resolved": None,
            "signals": [],
            "errors": [],
        }
        if resolve_signals and not dry_run:
            try:
                iteration["resolved"] = resolve_crypto_5m_signal_log(path=log_path, db_path=db_path)
            except CryptoModelError as exc:
                iteration["errors"].append({"action": "resolve", "error": str(exc)})
        elif resolve_signals:
            iteration["resolved"] = {
                "path": str(log_path),
                "dry_run": True,
            }

        if record_signals:
            for symbol in symbol_list:
                try:
                    if dry_run:
                        forecast = forecast_crypto_5m(
                            symbol,
                            horizon_minutes=horizon_minutes,
                            lookback_minutes=lookback_minutes,
                            market_probability=market_probability,
                            fee_bps=fee_bps,
                            ml_mode=ml_mode,
                            training_window=training_window,
                            edge_threshold=edge_threshold,
                        )
                        applied, strategy_name = _apply_crypto_5m_paper_strategy(
                            forecast,
                            strategy_mode=strategy_mode,
                            momentum_threshold=momentum_threshold,
                            fee_bps=fee_bps,
                        )
                        iteration["signals"].append({
                            "path": str(log_path),
                            "dry_run": True,
                            "record": {
                                "symbol": normalize_symbol(symbol),
                                "strategy_name": strategy_name,
                                "target_price": applied["target_price"],
                                "predicted_outcome": applied["predicted_outcome"],
                                "probability_up": applied["probability_up"],
                                "edge": applied["edge"],
                                "recommendation": applied["recommendation"],
                                "strategy": applied["strategy"],
                            },
                        })
                    else:
                        iteration["signals"].append(record_crypto_5m_signal(
                            symbol,
                            horizon_minutes=horizon_minutes,
                            lookback_minutes=lookback_minutes,
                            market_probability=market_probability,
                            fee_bps=fee_bps,
                            ml_mode=ml_mode,
                            training_window=training_window,
                            edge_threshold=edge_threshold,
                            strategy_mode=strategy_mode,
                            momentum_threshold=momentum_threshold,
                            path=log_path,
                            db_path=db_path,
                        ))
                except CryptoModelError as exc:
                    iteration["errors"].append({"action": "record", "symbol": symbol, "error": str(exc)})
        iteration_results.append(iteration)
        if index < loop_count - 1 and sleep_seconds > 0:
            time.sleep(float(sleep_seconds))
    final_summary = resolve_crypto_5m_signal_log(path=log_path, db_path=db_path) if log_path.exists() and not dry_run else None
    return {
        "path": str(log_path),
        "db_path": str(db_path) if db_path else None,
        "iterations": loop_count,
        "symbols": symbol_list,
        "dry_run": dry_run,
        "final_summary": final_summary,
        "iteration_results": iteration_results,
    }
