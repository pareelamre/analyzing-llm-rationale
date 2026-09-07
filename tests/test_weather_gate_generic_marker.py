"""The weather trading gate fired on rules that excluded weather.

_looks_like_weather_contract matched any marker anywhere in the contract
text -- question, description and resolution criteria concatenated. The
generic word "weather" appears in rules that have nothing to do with
weather, nearly always in an exclusion. A live geopolitics market,
"Israel closes its airspace by September 30?", says:

    airspace closures which occur solely due to weather conditions
    will not qualify

That sentence alone made it is_weather=true, market_type "other_weather",
and -- since no weather settlement source is named in a geopolitics
contract -- trade_permitted=false with blocker
missing_contract_settlement_source. The gate blocked trading on it.

The opposite error was live at the same time. _contract_text read only
`question`, and weather-radar rows key their subject as `title`, so three
real "Highest temperature in Austin" contracts contributed no subject at
all and classified as not-weather. For a gate that exists to withhold
trading until a settlement source is verified, that fails open, which is
the worse direction.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.weather_markets import (  # noqa: E402
    _GENERIC_WEATHER_MARKERS,
    _SPECIFIC_WEATHER_MARKERS,
    classify_weather_market,
)

_EXCLUSION = (
    "This market resolves Yes if Israel initiates a major closure of its "
    "airspace. Airspace closures which occur solely due to weather conditions "
    "will not qualify. A qualifying closure must apply generally to all flights."
)


class GenericMarkerOnlyCountsInTheTitleTests(unittest.TestCase):
    def test_an_exclusion_clause_does_not_gate_the_market(self):
        brief = classify_weather_market({
            "question": "Israel closes its airspace by September 30?",
            "resolution_criteria": _EXCLUSION,
        }).as_dict()
        self.assertFalse(brief["is_weather"])
        self.assertTrue(brief["trade_permitted"])
        self.assertIsNone(brief["blocker"])

    def test_weather_in_the_title_still_gates(self):
        brief = classify_weather_market({
            "question": "Will the weather station report a record on Sep 30?",
        }).as_dict()
        self.assertTrue(brief["is_weather"])

    def test_a_specific_marker_in_the_rules_still_gates(self):
        """Specific markers keep their meaning wherever they appear."""
        for marker in ("temperature", "snow", "hurricane", "precipitation"):
            with self.subTest(marker=marker):
                brief = classify_weather_market({
                    "question": "Some contract",
                    "resolution_criteria": f"Resolves on the recorded {marker} at the site.",
                }).as_dict()
                self.assertTrue(brief["is_weather"], marker)

    def test_the_weather_category_is_still_an_override(self):
        brief = classify_weather_market({
            "question": "Terse contract", "category": "weather",
        }).as_dict()
        self.assertTrue(brief["is_weather"])

    def test_generic_and_specific_partition_the_markers(self):
        self.assertEqual(set(_GENERIC_WEATHER_MARKERS), {"weather"})
        self.assertNotIn("weather", _SPECIFIC_WEATHER_MARKERS)
        self.assertIn("temperature", _SPECIFIC_WEATHER_MARKERS)


class TitleIsReadAsTheSubjectTests(unittest.TestCase):
    """Weather-radar rows carry `title`, not `question`."""

    def test_a_title_keyed_temperature_contract_is_gated(self):
        brief = classify_weather_market({
            "title": "Highest temperature in Austin on Sep 7, 2026? 97 to 98",
        }).as_dict()
        self.assertTrue(brief["is_weather"])
        self.assertEqual(brief["market_type"], "daily_temperature")

    def test_it_fails_closed_without_a_named_source(self):
        """The gate's purpose: no verified source, no new exposure."""
        brief = classify_weather_market({
            "title": "Highest temperature in Austin on Sep 7, 2026?",
        }).as_dict()
        self.assertFalse(brief["trade_permitted"])
        # No source named at all, so the generic blocker -- the NWS-specific
        # one fires only when a source is named and is the wrong one.
        self.assertEqual(brief["blocker"], "missing_contract_settlement_source")

    def test_question_still_wins_when_both_are_present(self):
        brief = classify_weather_market({
            "question": "Will it snow in Denver?", "title": "unrelated",
        }).as_dict()
        self.assertTrue(brief["is_weather"])

    def test_a_contract_with_neither_is_not_weather(self):
        self.assertFalse(classify_weather_market({}).as_dict()["is_weather"])


if __name__ == "__main__":
    unittest.main()
