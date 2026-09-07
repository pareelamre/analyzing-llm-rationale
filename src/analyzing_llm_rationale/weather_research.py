"""Read-only, source-aware evidence for prediction-market weather contracts.

The venue's settlement rule remains authoritative.  This module never turns a
forecast or observation into a settlement value; it retrieves official NWS
inputs only for contracts whose supplied rules name NWS, and describes other
official sources honestly when they are not integrated.
"""
from __future__ import annotations

import copy
import logging
import math
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from analyzing_llm_rationale.weather_markets import classify_weather_market

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("foresea.weather_research")
meter = metrics.get_meter("foresea.weather_research")

weather_research_requests = meter.create_counter(
    "weather_research.requests",
    unit="1",
    description="Read-only weather market research requests by source and outcome",
)
weather_research_duration = meter.create_histogram(
    "weather_research.duration",
    unit="s",
    description="Duration of source-aware weather market research",
)

_NWS_API = "https://api.weather.gov"
_REQUEST_TIMEOUT_S = 8
_CACHE_TTL_S = max(30, min(900, int(os.environ.get("FORESEA_WEATHER_RESEARCH_CACHE_TTL_S", "300"))))
_CACHE_MAX_ENTRIES = max(1, min(128, int(os.environ.get("FORESEA_WEATHER_RESEARCH_CACHE_MAX_ENTRIES", "32"))))
_NWS_FAILURES_BEFORE_CIRCUIT = max(
    1, min(10, int(os.environ.get("FORESEA_WEATHER_RESEARCH_NWS_FAILURES_BEFORE_CIRCUIT", "3")))
)
_NWS_CIRCUIT_COOLDOWN_S = max(
    15, min(900, int(os.environ.get("FORESEA_WEATHER_RESEARCH_NWS_CIRCUIT_COOLDOWN_S", "120")))
)
_HEADERS = {
    "User-Agent": "Foresea-weather-research/1.0 (support@foresea.ink)",
    "Accept": "application/geo+json",
}
_runtime_lock = threading.RLock()
_nws_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_nws_consecutive_failures = 0
_nws_circuit_open_until = 0.0


def _value(props: Mapping[str, Any], name: str) -> Optional[float]:
    raw = props.get(name)
    if isinstance(raw, Mapping):
        raw = raw.get("value")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _celsius_to_fahrenheit(value: Optional[float]) -> Optional[float]:
    return round((value * 9 / 5) + 32, 1) if value is not None else None


def _json_get(http_get: Callable[..., Any], url: str) -> Mapping[str, Any]:
    response = http_get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("weather source returned an invalid payload")
    return payload


_STATION_COORDINATES: Dict[str, Tuple[float, float]] = {
    "KNYC": (40.7829, -73.9654),  # New York Central Park
    "KMDW": (41.7868, -87.7522),  # Chicago Midway
    "KORD": (41.9742, -87.9073),  # Chicago O'Hare
    "KMIA": (25.7959, -80.2870),  # Miami International
    "KAUS": (30.1975, -97.6664),  # Austin Bergstrom
    "KDEN": (39.8561, -104.6737), # Denver International
    "KPHL": (39.8721, -75.2411), # Philadelphia International
    "KSFO": (37.6213, -122.3790), # San Francisco
    "KLAX": (33.9416, -118.4085), # Los Angeles
    "KBOS": (42.3656, -71.0096),  # Boston Logan
    "KDFW": (32.8998, -97.0403),  # Dallas/Fort Worth
    "KATL": (33.6407, -84.4277),  # Atlanta Hartsfield
    "KSEA": (47.4502, -122.3088), # Seattle Tacoma
    "KDCA": (38.8512, -77.0402),  # Washington Reagan
}
_OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"
_GOOGLE_WEATHER_API = "https://weather.googleapis.com/v1/forecast/hours:lookup"

_STATION_BIAS_PROFILES: Dict[str, Dict[str, Any]] = {
    "KNYC": {
        "name": "New York Central Park",
        "elevation_ft": 154,
        "bias_offset_f": -0.5,
        "microclimate_note": (
            "Central Park is an urban green canopy surrounded by Manhattan's concrete heat island. "
            "Radiative nocturnal cooling is higher than coastal LGA/JFK, with sea-breeze moderation."
        ),
    },
    "KMDW": {
        "name": "Chicago Midway Airport",
        "elevation_ft": 620,
        "bias_offset_f": 1.5,
        "microclimate_note": (
            "Midway sits in Chicago's South Side inland urban heat basin. Frequently 2°F to 4°F "
            "warmer than lakefront sensors during Lake Michigan northeasterly lake breeze events."
        ),
    },
    "KORD": {
        "name": "Chicago O'Hare Airport",
        "elevation_ft": 672,
        "bias_offset_f": 0.0,
        "microclimate_note": "Suburban airport northwest of Chicago; standard midwestern continental exposure.",
    },
    "KMIA": {
        "name": "Miami International Airport",
        "elevation_ft": 9,
        "bias_offset_f": 0.0,
        "microclimate_note": (
            "Subtropical coastal maritime exposure. Diurnal maximum is governed by the onset timing "
            "of convective sea-breeze afternoon cloud cover and thunderstorm downdrafts."
        ),
    },
    "KAUS": {
        "name": "Austin Bergstrom Airport",
        "elevation_ft": 542,
        "bias_offset_f": 0.8,
        "microclimate_note": (
            "Colorado River basin terrain. High diurnal amplitude with rapid nocturnal radiative inversion "
            "and strong daytime insolation."
        ),
    },
    "KDEN": {
        "name": "Denver International Airport",
        "elevation_ft": 5431,
        "bias_offset_f": -1.0,
        "microclimate_note": (
            "High-plains station 25 miles east of downtown Denver at 5,431 ft elevation. "
            "Prone to strong nocturnal drainage winds and downslope Chinook compression heating."
        ),
    },
    "KPHL": {
        "name": "Philadelphia International Airport",
        "elevation_ft": 36,
        "bias_offset_f": 0.5,
        "microclimate_note": (
            "Delaware River coastal plain microclimate. Urban corridor warmth modified by Delaware Bay breeze."
        ),
    },
    "KSFO": {
        "name": "San Francisco International Airport",
        "elevation_ft": 13,
        "bias_offset_f": -1.2,
        "microclimate_note": (
            "Pacific marine layer and San Bruno Gap wind funneling create sharp localized cooling compared to bay interior."
        ),
    },
    "KLAX": {
        "name": "Los Angeles International Airport",
        "elevation_ft": 125,
        "bias_offset_f": -1.0,
        "microclimate_note": (
            "Immediate coastal marine layer stratus and sea breeze suppress daytime high relative to the LA basin interior."
        ),
    },
    "KBOS": {
        "name": "Boston Logan Airport",
        "elevation_ft": 20,
        "bias_offset_f": -0.8,
        "microclimate_note": (
            "Harbor-side airport subject to sharp marine cold front pushes and easterly sea breezes."
        ),
    },
    "KATL": {
        "name": "Atlanta Hartsfield Airport",
        "elevation_ft": 1026,
        "bias_offset_f": 0.5,
        "microclimate_note": "Piedmont plateau airport; extensive asphalt runway urban heat footprint.",
    },
}


def _extract_coordinates(
    geometry: Mapping[str, Any], station: Optional[str] = None
) -> Optional[Tuple[float, float]]:
    coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
    if isinstance(coordinates, list) and len(coordinates) >= 2:
        try:
            lon, lat = float(coordinates[0]), float(coordinates[1])
            return lat, lon
        except (TypeError, ValueError):
            pass
    if station and station.upper() in _STATION_COORDINATES:
        return _STATION_COORDINATES[station.upper()]
    return None


def _fetch_google_weather_forecast(
    http_get: Callable[..., Any], lat: float, lon: float, api_key: str
) -> Optional[Dict[str, Any]]:
    """Fetch hourly forecast from Google Maps Platform Weather API (DeepMind WeatherNext / MetNet)."""
    try:
        url = (
            f"{_GOOGLE_WEATHER_API}?location.latitude={lat:.4f}&location.longitude={lon:.4f}&"
            f"hours=24&unitsSystem=IMPERIAL&key={api_key}"
        )
        response = http_get(url, timeout=_REQUEST_TIMEOUT_S)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        payload = response.json() if hasattr(response, "json") else None
        if not isinstance(payload, Mapping):
            return None
        forecast_hours = payload.get("forecastHours")
        if not isinstance(forecast_hours, list) or not forecast_hours:
            return None
        valid_pairs: List[Tuple[str, float]] = []
        for fh in forecast_hours:
            if not isinstance(fh, Mapping):
                continue
            interval = fh.get("interval")
            time_str = interval.get("startTime") if isinstance(interval, Mapping) else None
            temp_obj = fh.get("temperature")
            temp_f = None
            if isinstance(temp_obj, Mapping):
                deg = temp_obj.get("degrees")
                unit = str(temp_obj.get("unit") or "CELSIUS").upper()
                if deg is not None:
                    try:
                        deg_val = float(deg)
                        if "FAHRENHEIT" in unit:
                            temp_f = deg_val
                        else:
                            temp_f = (deg_val * 9.0 / 5.0) + 32.0
                    except (TypeError, ValueError):
                        pass
            elif isinstance(temp_obj, (int, float)):
                temp_f = float(temp_obj)
            if time_str and temp_f is not None:
                valid_pairs.append((time_str, temp_f))
        if not valid_pairs:
            return None
        valid_temps = [p[1] for p in valid_pairs]
        return {
            "provider": "google_maps_weather",
            "model": "google_deepmind_weathernext_metnet",
            "forecast_start": valid_pairs[0][0],
            "forecast_end": valid_pairs[-1][0],
            "projected_high_f": round(max(valid_temps), 1),
            "projected_low_f": round(min(valid_temps), 1),
            "hourly_curve": [
                {"time": t, "temperature_f": round(temp, 1)}
                for t, temp in valid_pairs[::2][:8]
            ],
            "notice": (
                "Google Maps Platform Weather API (Google DeepMind WeatherNext / MetNet neural model reference). "
                "Official settlement is governed strictly by the contract's named settlement source."
            ),
        }
    except Exception as exc:
        logger.debug("Google weather forecast fetch unavailable: %s", exc)
        return None


def _fetch_open_meteo_forecast(
    http_get: Callable[..., Any], lat: float, lon: float
) -> Optional[Dict[str, Any]]:
    """Fetch high-resolution global numerical / AI forecast via Open-Meteo."""
    try:
        url = (
            f"{_OPEN_METEO_API}?latitude={lat:.4f}&longitude={lon:.4f}&"
            f"hourly=temperature_2m&temperature_unit=fahrenheit&forecast_days=2"
        )
        response = http_get(url, timeout=_REQUEST_TIMEOUT_S)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        payload = response.json() if hasattr(response, "json") else None
        if not isinstance(payload, Mapping):
            return None
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        if not times or not temps or len(times) != len(temps):
            return None
        valid_pairs = [(t, float(temp)) for t, temp in zip(times[:24], temps[:24]) if temp is not None]
        if not valid_pairs:
            return None
        valid_temps = [p[1] for p in valid_pairs]
        return {
            "provider": "open-meteo",
            "model": "high_res_multi_model",
            "forecast_start": valid_pairs[0][0],
            "forecast_end": valid_pairs[-1][0],
            "projected_high_f": round(max(valid_temps), 1),
            "projected_low_f": round(min(valid_temps), 1),
            "hourly_curve": [
                {"time": t, "temperature_f": temp}
                for t, temp in valid_pairs[::2][:8]
            ],
            "notice": (
                "Predictive multi-model numerical/AI forecast reference. "
                "Official settlement is governed strictly by the contract's named settlement source."
            ),
        }
    except Exception as exc:
        logger.debug("Open-Meteo model forecast fetch unavailable: %s", exc)
        return None


def _fetch_model_forecast(
    http_get: Callable[..., Any],
    lat: float,
    lon: float,
    station: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch predictive numerical / neural AI forecast and calibrate with station bias.

    Checks Google Maps Platform Weather API first if GOOGLE_WEATHER_API_KEY / GMP_API_KEY
    is set, falling back to Open-Meteo multi-model. Applies empirical microclimate station bias
    if the station is recognized.
    """
    gmp_key = os.environ.get("GOOGLE_WEATHER_API_KEY") or os.environ.get("GMP_API_KEY")
    data = None
    if gmp_key:
        data = _fetch_google_weather_forecast(http_get, lat, lon, gmp_key)
    if data is None:
        data = _fetch_open_meteo_forecast(http_get, lat, lon)
    if data is None:
        return None

    profile = _STATION_BIAS_PROFILES.get(station.upper()) if station else None
    if profile:
        raw_high = data["projected_high_f"]
        raw_low = data["projected_low_f"]
        offset = profile["bias_offset_f"]
        data["raw_projected_high_f"] = raw_high
        data["raw_projected_low_f"] = raw_low
        data["projected_high_f"] = round(raw_high + offset, 1)
        data["projected_low_f"] = round(raw_low + offset, 1)
        data["station_bias_offset_f"] = offset
        data["station_microclimate_note"] = profile["microclimate_note"]
        data["station_name"] = profile["name"]
        data["station_elevation_ft"] = profile["elevation_ft"]
    else:
        data["raw_projected_high_f"] = data["projected_high_f"]
        data["raw_projected_low_f"] = data["projected_low_f"]
        data["station_bias_offset_f"] = 0.0
        data["station_microclimate_note"] = None

    return data


def parse_market_strike(quote: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract strike threshold or bracket interval from a weather contract quote."""
    text_candidates = [
        str(quote.get("subtitle") or ""),
        str(quote.get("question") or ""),
        str(quote.get("title") or ""),
    ]
    full_text = " ".join(t for t in text_candidates if t)
    if not full_text:
        return None

    # Between / interval: e.g. "74° to 77°" or "74.5 - 77.5"
    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:°|deg)?\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:°|deg)?",
        full_text,
        re.IGNORECASE,
    )
    if range_match:
        try:
            low, high = float(range_match.group(1)), float(range_match.group(2))
            if low > high:
                low, high = high, low
            return {
                "strike_type": "between",
                "strike_low_f": low,
                "strike_high_f": high,
            }
        except (ValueError, TypeError):
            pass

    # Below / less than: e.g. "Below 77.5°" or "< 77.5"
    below_match = re.search(
        r"(?:below|less than|under|<)\s*(\d+(?:\.\d+)?)\s*(?:°|deg)?",
        full_text,
        re.IGNORECASE,
    )
    if below_match:
        try:
            return {
                "strike_type": "less_than",
                "strike_f": float(below_match.group(1)),
            }
        except (ValueError, TypeError):
            pass

    # Greater than or equal: e.g. "77.5° or above", "77.5° or higher", "77.5° and above"
    above_or_equal_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:°|deg)?\s*(?:or above|or higher|or more|and above)",
        full_text,
        re.IGNORECASE,
    )
    if above_or_equal_match:
        try:
            return {
                "strike_type": "greater_than_or_equal",
                "strike_f": float(above_or_equal_match.group(1)),
            }
        except (ValueError, TypeError):
            pass

    # Above / greater than: e.g. "Above 77.5°" or "> 77.5"
    above_match = re.search(
        r"(?:above|greater than|over|>)\s*(\d+(?:\.\d+)?)\s*(?:°|deg)?",
        full_text,
        re.IGNORECASE,
    )
    if above_match:
        try:
            return {
                "strike_type": "greater_than",
                "strike_f": float(above_match.group(1)),
            }
        except (ValueError, TypeError):
            pass

    return None


def calculate_bracket_probability(
    projected_mean_f: float,
    strike_spec: Mapping[str, Any],
    uncertainty_std_f: float = 1.8,
) -> Dict[str, Any]:
    """Calculate normal cumulative probability for a strike bracket given forecast mean and uncertainty.

    Uses the Gaussian error function to integrate probability density over the contract strike.
    """
    strike_type = str(strike_spec.get("strike_type") or "less_than")
    std = max(0.5, float(uncertainty_std_f))

    def _cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf((x - projected_mean_f) / (std * math.sqrt(2.0))))

    if strike_type in ("less_than", "less_than_or_equal"):
        strike_val = float(strike_spec.get("strike_f", projected_mean_f))
        prob = _cdf(strike_val)
    elif strike_type in ("greater_than", "greater_than_or_equal"):
        strike_val = float(strike_spec.get("strike_f", projected_mean_f))
        prob = 1.0 - _cdf(strike_val)
    elif strike_type == "between":
        low = float(strike_spec.get("strike_low_f", projected_mean_f - 1.0))
        high = float(strike_spec.get("strike_high_f", projected_mean_f + 1.0))
        prob = max(0.0, _cdf(high) - _cdf(low))
    else:
        prob = 0.5

    prob = round(max(0.001, min(0.999, prob)), 3)
    conf_low = round(projected_mean_f - 1.96 * std, 1)
    conf_high = round(projected_mean_f + 1.96 * std, 1)

    return {
        "strike_spec": dict(strike_spec),
        "model_mean_high_f": round(projected_mean_f, 1),
        "uncertainty_std_f": round(std, 2),
        "model_probability": prob,
        "confidence_interval_95_f": [conf_low, conf_high],
    }


def _forecast_periods(
    http_get: Callable[..., Any], geometry: Mapping[str, Any], station: Optional[str] = None
) -> List[Dict[str, Any]]:
    coords = _extract_coordinates(geometry, station)
    if not coords:
        return []
    lat, lon = coords
    try:
        point = _json_get(http_get, f"{_NWS_API}/points/{lat:.4f},{lon:.4f}")
        hourly_url = (point.get("properties") or {}).get("forecastHourly")
        if not isinstance(hourly_url, str) or not hourly_url.startswith("https://"):
            return []
        hourly = _json_get(http_get, hourly_url)
        periods = (hourly.get("properties") or {}).get("periods") or []
        result: List[Dict[str, Any]] = []
        for period in periods[:8]:
            if not isinstance(period, Mapping):
                continue
            precip = period.get("probabilityOfPrecipitation") or {}
            result.append({
                "start_time": period.get("startTime"),
                "temperature_f": period.get("temperature"),
                "short_forecast": period.get("shortForecast"),
                "precipitation_probability": precip.get("value") if isinstance(precip, Mapping) else None,
            })
        return result
    except Exception as exc:
        logger.debug("NWS forecast periods unavailable: %s", exc)
        return []


def _cached_nws_research(station: str) -> Optional[Dict[str, Any]]:
    now = time.monotonic()
    with _runtime_lock:
        entry = _nws_cache.get(station)
        if entry is None or entry[0] <= now:
            if entry is not None:
                _nws_cache.pop(station, None)
            return None
        result = copy.deepcopy(entry[1])
    result["research_cached"] = True
    return result


def _cache_nws_research(station: str, result: Dict[str, Any]) -> None:
    with _runtime_lock:
        if len(_nws_cache) >= _CACHE_MAX_ENTRIES:
            oldest = min(_nws_cache, key=lambda key: _nws_cache[key][0])
            _nws_cache.pop(oldest, None)
        stored = copy.deepcopy(result)
        stored["research_cached"] = False
        _nws_cache[station] = (time.monotonic() + _CACHE_TTL_S, stored)


def _nws_circuit_is_open() -> bool:
    with _runtime_lock:
        return _nws_circuit_open_until > time.monotonic()


def _record_nws_success() -> None:
    global _nws_consecutive_failures, _nws_circuit_open_until
    with _runtime_lock:
        _nws_consecutive_failures = 0
        _nws_circuit_open_until = 0.0


def _record_nws_failure() -> bool:
    """Return whether this failure has opened the bounded NWS circuit."""
    global _nws_consecutive_failures, _nws_circuit_open_until
    with _runtime_lock:
        _nws_consecutive_failures += 1
        if _nws_consecutive_failures >= _NWS_FAILURES_BEFORE_CIRCUIT:
            _nws_circuit_open_until = time.monotonic() + _NWS_CIRCUIT_COOLDOWN_S
        return _nws_circuit_open_until > time.monotonic()


def _reset_runtime_state_for_test() -> None:
    """Reset only in-process cache/circuit state for deterministic unit tests."""
    global _nws_consecutive_failures, _nws_circuit_open_until
    with _runtime_lock:
        _nws_cache.clear()
        _nws_consecutive_failures = 0
        _nws_circuit_open_until = 0.0


def research_weather_market(
    quote: Mapping[str, Any],
    *,
    http_get: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Return bounded weather evidence with explicit source authority.

    For an NWS-settled contract with a named ICAO station, the output includes
    the current NWS station observation and nearby NWS hourly forecast.  The
    daily climate report remains the settlement authority and can differ from
    preliminary observations.  Weather Company contracts are not queried
    because Foresea has no licensed Weather Company data-feed integration.
    """
    started = time.perf_counter()
    with tracer.start_as_current_span("weather_research.market") as span:
        try:
            brief = classify_weather_market(quote)
            span.set_attributes({
                "weather.is_weather": brief.is_weather,
                "weather.market_type": brief.market_type,
                "weather.settlement_source": brief.settlement_source,
            })
            if not brief.is_weather:
                result = {
                    "weather_market": brief.as_dict(),
                    "source_status": "not_applicable",
                    "observations": [],
                    "forecast_periods": [],
                    "model_forecast": None,
                    "bracket_probability": None,
                    "notice": "This contract is not classified as weather.",
                }
                outcome = "not_applicable"
            elif brief.settlement_source == "weather_company":
                result = {
                    "weather_market": brief.as_dict(),
                    "source_status": "official_source_not_integrated",
                    "observations": [],
                    "forecast_periods": [],
                    "model_forecast": None,
                    "bracket_probability": None,
                    "notice": (
                        "The Weather Company is the named settlement source. Foresea has no licensed "
                        "Weather Company feed, so no proxy is represented as official settlement data."
                    ),
                }
                outcome = "not_integrated"
            elif brief.settlement_source != "nws_daily_climate_report" or not brief.station:
                reason = "missing_station" if brief.settlement_source == "nws_daily_climate_report" else "unverified_source"
                result = {
                    "weather_market": brief.as_dict(),
                    "source_status": reason,
                    "observations": [],
                    "forecast_periods": [],
                    "model_forecast": None,
                    "bracket_probability": None,
                    "notice": "No source-matched NWS lookup was performed; the contract data is incomplete for this research pass.",
                }
                outcome = reason
            else:
                cached = _cached_nws_research(brief.station)
                if cached is not None:
                    span.set_attributes({"weather.source_status": cached["source_status"], "outcome": "cache_hit"})
                    weather_research_requests.add(1, {
                        "settlement_source": brief.settlement_source,
                        "outcome": "cache_hit",
                    })
                    return cached
                if _nws_circuit_is_open():
                    result = {
                        "weather_market": brief.as_dict(),
                        "source_status": "nws_circuit_open",
                        "observations": [],
                        "forecast_periods": [],
                        "model_forecast": None,
                        "bracket_probability": None,
                        "research_cached": False,
                        "notice": "NWS research is temporarily paused after repeated upstream failures; retry later.",
                    }
                    outcome = "circuit_open"
                    span.set_attributes({"weather.source_status": result["source_status"], "outcome": outcome})
                    weather_research_requests.add(1, {
                        "settlement_source": brief.settlement_source,
                        "outcome": outcome,
                    })
                    return result
                if http_get is None:
                    import requests
                    http_get = requests.get
                try:
                    payload = _json_get(http_get, f"{_NWS_API}/stations/{brief.station}/observations/latest")
                    props = payload.get("properties") or {}
                    geometry = payload.get("geometry") or {}
                    observation = {
                        "source": "NWS station observation",
                        "station": brief.station,
                        "timestamp": props.get("timestamp"),
                        "temperature_f": _celsius_to_fahrenheit(_value(props, "temperature")),
                        "dewpoint_f": _celsius_to_fahrenheit(_value(props, "dewpoint")),
                        "wind_speed_mps": _value(props, "windSpeed"),
                        "precipitation_last_hour_mm": _value(props, "precipitationLastHour"),
                        "authority": "preliminary_observation_not_final_daily_settlement",
                    }
                    periods = _forecast_periods(http_get, geometry if isinstance(geometry, Mapping) else {}, brief.station)
                    coords = _extract_coordinates(geometry if isinstance(geometry, Mapping) else {}, brief.station)
                    model_forecast = (
                        _fetch_model_forecast(http_get, coords[0], coords[1], station=brief.station)
                        if coords
                        else None
                    )
                    bracket_prob = None
                    if model_forecast and isinstance(model_forecast.get("projected_high_f"), (int, float)):
                        strike_spec = parse_market_strike(quote)
                        if strike_spec:
                            bracket_prob = calculate_bracket_probability(
                                float(model_forecast["projected_high_f"]),
                                strike_spec,
                                uncertainty_std_f=1.8,
                            )
                            for price_key in ("price", "yes_ask", "yes_bid", "last_price"):
                                val = quote.get(price_key)
                                if isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0:
                                    bracket_prob["market_implied_probability"] = round(float(val), 3)
                                    bracket_prob["model_edge"] = round(
                                        bracket_prob["model_probability"] - float(val), 3
                                    )
                                    break
                                elif isinstance(val, (int, float)) and 1.0 < float(val) <= 100.0:
                                    prob_val = float(val) / 100.0
                                    bracket_prob["market_implied_probability"] = round(prob_val, 3)
                                    bracket_prob["model_edge"] = round(
                                        bracket_prob["model_probability"] - prob_val, 3
                                    )
                                    break
                    result = {
                        "weather_market": brief.as_dict(),
                        "source_status": "nws_observation_available",
                        "observations": [observation],
                        "forecast_periods": periods,
                        "model_forecast": model_forecast,
                        "bracket_probability": bracket_prob,
                        "research_cached": False,
                        "notice": (
                            "NWS observations and forecast are source-matched research inputs. The final NWS Daily "
                            "Climate Report named in the contract remains the settlement authority."
                        ),
                    }
                    _record_nws_success()
                    _cache_nws_research(brief.station, result)
                    outcome = "available"
                except Exception as exc:
                    circuit_opened = _record_nws_failure()
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR))
                    result = {
                        "weather_market": brief.as_dict(),
                        "source_status": "nws_temporarily_unavailable",
                        "observations": [],
                        "forecast_periods": [],
                        "model_forecast": None,
                        "bracket_probability": None,
                        "research_cached": False,
                        "notice": (
                            "NWS source-matched research is temporarily unavailable; no proxy data was used. "
                            "The contract's final NWS Daily Climate Report remains authoritative."
                        ),
                    }
                    outcome = "circuit_opened" if circuit_opened else "upstream_failure"
                    logger.warning("NWS weather market research unavailable", exc_info=True)
            span.set_attributes({"weather.source_status": result["source_status"], "outcome": outcome})
            weather_research_requests.add(1, {
                "settlement_source": brief.settlement_source,
                "outcome": outcome,
            })
            logger.info(
                "weather market research source=%s status=%s",
                brief.settlement_source,
                result["source_status"],
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            weather_research_requests.add(1, {"settlement_source": "unknown", "outcome": "failure"})
            logger.warning("weather market research failed", exc_info=True)
            raise
        finally:
            weather_research_duration.record(time.perf_counter() - started)
