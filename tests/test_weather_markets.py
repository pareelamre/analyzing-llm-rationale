from __future__ import annotations

import unittest

from analyzing_llm_rationale.weather_markets import (
    classify_weather_market,
    format_weather_market_brief,
)


class WeatherMarketClassificationTests(unittest.TestCase):
    def test_non_weather_quote_is_not_constrained(self):
        brief = classify_weather_market({"question": "Will CPI exceed 3%?", "category": "Economics"})
        self.assertFalse(brief.is_weather)
        self.assertTrue(brief.trade_permitted)
        self.assertEqual(brief.market_type, "not_weather")

    def test_broad_climate_weather_category_does_not_capture_non_weather_contracts(self):
        brief = classify_weather_market({
            "question": "Will EV share exceed 30% by 2030?",
            "category": "Climate and Weather",
        })
        self.assertFalse(brief.is_weather)
        self.assertTrue(brief.trade_permitted)

    def test_daily_temperature_requires_explicit_nws_source(self):
        quote = {
            "question": "What will the highest temperature in Chicago be today?",
            "category": "Climate and Weather",
            "resolution_criteria": "Resolves by the NWS Daily Climate Report for Chicago O'Hare.",
        }
        brief = classify_weather_market(quote)
        self.assertTrue(brief.is_weather)
        self.assertEqual(brief.market_type, "daily_temperature")
        self.assertEqual(brief.settlement_source, "nws_daily_climate_report")
        self.assertTrue(brief.trade_permitted)

    def test_hourly_temperature_requires_weather_company_and_station(self):
        quote = {
            "question": "Will Chicago temperature at 5 PM EDT exceed 80F?",
            "category": "Weather",
            "resolution_criteria": (
                "Resolves using The Weather Company reported temperature at station KORD."
            ),
        }
        brief = classify_weather_market(quote)
        self.assertEqual(brief.market_type, "hourly_temperature")
        self.assertEqual(brief.settlement_source, "weather_company")
        self.assertEqual(brief.station, "KORD")
        self.assertTrue(brief.trade_permitted)

    def test_hourly_temperature_without_station_is_research_only(self):
        brief = classify_weather_market({
            "question": "Will Chicago temperature at 5 PM EDT exceed 80F?",
            "category": "Weather",
            "resolution_criteria": "The Weather Company is the official source.",
        })
        self.assertFalse(brief.trade_permitted)
        self.assertEqual(brief.blocker, "hourly_temperature_missing_station")
        self.assertIn("NO NEW PAPER POSITION", format_weather_market_brief({
            "question": "Will Chicago temperature at 5 PM EDT exceed 80F?",
            "category": "Weather",
            "resolution_criteria": "The Weather Company is the official source.",
        }))

    def test_daily_temperature_with_wrong_source_is_blocked(self):
        brief = classify_weather_market({
            "question": "What will the highest temperature in Chicago be today?",
            "category": "Weather",
            "resolution_criteria": "The Weather Company will determine this market.",
        })
        self.assertFalse(brief.trade_permitted)
        self.assertEqual(brief.blocker, "daily_temperature_requires_nws_source")

    def test_weather_contract_without_venue_source_is_blocked(self):
        brief = classify_weather_market({
            "question": "Will it snow in Denver tomorrow?",
            "category": "Weather",
        })
        self.assertFalse(brief.trade_permitted)
        self.assertEqual(brief.blocker, "missing_contract_settlement_source")


if __name__ == "__main__":
    unittest.main()
