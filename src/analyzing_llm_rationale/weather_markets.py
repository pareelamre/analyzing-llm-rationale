"""Settlement-aware classification for prediction-market weather contracts.

This module intentionally does *not* forecast weather or invent a settlement
feed.  It turns the venue's contract text into a bounded provenance record so
research and paper execution can distinguish a useful meteorological input
from the exact source that will determine the contract outcome.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("foresea.weather_markets")
meter = metrics.get_meter("foresea.weather_markets")

weather_market_classifications = meter.create_counter(
    "weather_markets.classifications",
    unit="1",
    description="Weather-contract provenance classifications by result",
)
weather_market_classification_duration = meter.create_histogram(
    "weather_markets.classification.duration",
    unit="s",
    description="Duration of weather-contract provenance classification",
)

_WEATHER_MARKERS = (
    "weather", "temperature", "temp", "rain", "snow", "precipitation",
    "wind", "hurricane", "tornado", "storm", "heatwave", "freeze",
)
_DAILY_TEMPERATURE_MARKERS = (
    "highest temperature", "high temperature", "lowest temperature", "low temperature",
    "daily temperature", "daily high", "daily low", "daily climate report",
)
_HOURLY_TEMPERATURE_MARKERS = (
    "hourly temperature", "temperature at ", "temperature at", " at 1 pm",
    " at 2 pm", " at 3 pm", " at 4 pm", " at 5 pm", " at 6 pm", " at 7 pm",
    " at 8 pm", " at 9 pm", " at 10 pm", " at 11 pm", " at 12 pm",
)
_PRECIPITATION_MARKERS = ("precipitation", "rainfall", "rain", "snowfall", "snow")
_NWS_MARKERS = ("nws", "national weather service", "daily climate report")
_WEATHER_COMPANY_MARKERS = ("weather company", "weather.com/kalshi", "weather.com")
_STATION_RE = re.compile(r"\b(?:station|airport|coordinates?)\s*(?:code)?\s*[:#-]?\s*([A-Z]{4})\b|\b(K[A-Z]{3})\b")


@dataclass(frozen=True)
class WeatherMarketBrief:
    """A conservative, UI/model-safe reading of a weather contract's rules."""

    is_weather: bool
    market_type: str
    settlement_source: str
    settlement_source_label: str
    source_explicit: bool
    station: Optional[str]
    trade_permitted: bool
    blocker: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_weather": self.is_weather,
            "market_type": self.market_type,
            "settlement_source": self.settlement_source,
            "settlement_source_label": self.settlement_source_label,
            "source_explicit": self.source_explicit,
            "station": self.station,
            "trade_permitted": self.trade_permitted,
            "blocker": self.blocker,
        }


def _contract_text(quote: Mapping[str, Any]) -> Tuple[str, str, str]:
    question = str(quote.get("question") or "")
    category = str(quote.get("category") or "")
    contract_text = "\n".join(
        str(quote.get(field) or "")
        for field in ("resolution_criteria", "resolution_source", "description")
    )
    # ``Climate and Weather`` is a broad venue category containing contracts
    # such as earthquakes and EV-share targets.  Those must not inherit
    # weather-trading gates merely from their category, so classification is
    # driven by the contract text; only the exact ``Weather`` category is a
    # fallback for an otherwise terse contract.
    return f"{question}\n{contract_text}".lower(), contract_text, category.lower().strip()


def _looks_like_weather_contract(text: str, category: str) -> bool:
    if category == "weather":
        return True
    return any(re.search(rf"\b{re.escape(marker)}\b", text) for marker in _WEATHER_MARKERS)


def _market_type(text: str) -> str:
    if any(marker in text for marker in _HOURLY_TEMPERATURE_MARKERS):
        return "hourly_temperature"
    if any(marker in text for marker in _DAILY_TEMPERATURE_MARKERS):
        return "daily_temperature"
    if any(marker in text for marker in _PRECIPITATION_MARKERS):
        return "precipitation"
    return "other_weather"


def _official_source(contract_text: str) -> Tuple[str, str, bool]:
    lowered = contract_text.lower()
    if any(marker in lowered for marker in _NWS_MARKERS):
        return "nws_daily_climate_report", "NWS Daily Climate Report", True
    if any(marker in lowered for marker in _WEATHER_COMPANY_MARKERS):
        return "weather_company", "The Weather Company", True
    return "unknown", "No verified contract source", False


def _station(contract_text: str) -> Optional[str]:
    match = _STATION_RE.search(contract_text)
    if not match:
        return None
    return next((value for value in match.groups() if value), None)


def classify_weather_market(quote: Mapping[str, Any]) -> WeatherMarketBrief:
    """Classify a quote without assuming anything missing from its rules.

    A non-weather quote is always permitted by this weather-only gate.  A
    weather quote may only open new paper exposure once a source is explicitly
    named in the quote's venue-provided rules/source fields.  Exits are handled
    by the normal reduce-only trading guard and are never blocked here.
    """
    started = time.perf_counter()
    with tracer.start_as_current_span("weather_markets.classify") as span:
        try:
            text, contract_text, category = _contract_text(quote)
            is_weather = _looks_like_weather_contract(text, category)
            if not is_weather:
                result = WeatherMarketBrief(
                    is_weather=False,
                    market_type="not_weather",
                    settlement_source="not_applicable",
                    settlement_source_label="Not a weather contract",
                    source_explicit=False,
                    station=None,
                    trade_permitted=True,
                    blocker=None,
                )
            else:
                market_type = _market_type(text)
                source, label, explicit = _official_source(contract_text)
                station = _station(contract_text)
                blocker: Optional[str] = None
                if not explicit:
                    blocker = "missing_contract_settlement_source"
                elif market_type == "daily_temperature" and source != "nws_daily_climate_report":
                    blocker = "daily_temperature_requires_nws_source"
                elif market_type == "hourly_temperature" and source != "weather_company":
                    blocker = "hourly_temperature_requires_weather_company_source"
                elif market_type == "hourly_temperature" and station is None:
                    blocker = "hourly_temperature_missing_station"
                result = WeatherMarketBrief(
                    is_weather=True,
                    market_type=market_type,
                    settlement_source=source,
                    settlement_source_label=label,
                    source_explicit=explicit,
                    station=station,
                    trade_permitted=blocker is None,
                    blocker=blocker,
                )
            outcome = "permitted" if result.trade_permitted else "blocked"
            span.set_attributes({
                "weather.is_weather": result.is_weather,
                "weather.market_type": result.market_type,
                "weather.settlement_source": result.settlement_source,
                "weather.source_explicit": result.source_explicit,
                "outcome": outcome,
            })
            weather_market_classifications.add(1, {
                "market_type": result.market_type,
                "outcome": outcome,
            })
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            weather_market_classifications.add(1, {"market_type": "unknown", "outcome": "failure"})
            logger.warning("weather market classification failed", exc_info=True)
            raise
        finally:
            weather_market_classification_duration.record(time.perf_counter() - started)


def format_weather_market_brief(quote: Mapping[str, Any]) -> str:
    """Return compact candidate context; weather forecasts remain advisory."""
    brief = classify_weather_market(quote)
    if not brief.is_weather:
        return ""
    parts = [
        f"Weather contract type: {brief.market_type.replace('_', ' ')}.",
        f"Official settlement source: {brief.settlement_source_label}.",
    ]
    if brief.station:
        parts.append(f"Named station: {brief.station}.")
    if brief.trade_permitted:
        parts.append(
            "Use independent forecast/observation evidence as inputs only; the named source determines settlement."
        )
    else:
        parts.append(
            f"NO NEW PAPER POSITION: contract provenance is incomplete ({brief.blocker}); research only."
        )
    return " ".join(parts)
