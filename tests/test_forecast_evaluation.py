from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.forecast_evaluation import (
    ResolvedForecast,
    Trade,
    build_trades,
    domain_probability_buckets,
    evaluation_report,
    market_clustered_brier_skill_interval,
    simulate_compounded_portfolio,
)

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _forecast(
    forecast_id: str,
    *,
    model_probability: float,
    market_probability: float,
    outcome: int,
    domain: str = "economics",
    market_id: str | None = None,
    forecasted_at: datetime = BASE_TIME,
    market_bid: float | None = None,
    market_ask: float | None = None,
) -> ResolvedForecast:
    return ResolvedForecast(
        forecast_id=forecast_id,
        platform="kalshi",
        market_id=market_id or forecast_id,
        model="council",
        forecasted_at=forecasted_at,
        resolved_at=forecasted_at + timedelta(days=10),
        model_probability=model_probability,
        market_probability=market_probability,
        outcome=outcome,
        domain=domain,
        market_bid=market_bid,
        market_ask=market_ask,
    )


class ForecastEvaluationTests(unittest.TestCase):
    def test_report_compares_model_and_market_without_rounding(self):
        forecasts = [
            _forecast("a", model_probability=0.8, market_probability=0.6, outcome=1),
            _forecast("b", model_probability=0.3, market_probability=0.5, outcome=0),
        ]

        report = evaluation_report(forecasts)

        self.assertAlmostEqual(report["model_brier"], 0.065)
        self.assertAlmostEqual(report["market_brier"], 0.205)
        self.assertAlmostEqual(report["skill_vs_market"], 0.14)
        self.assertEqual(report["n"], 2)

    def test_domain_buckets_shrink_small_samples_toward_global_rate(self):
        forecasts = [
            _forecast(
                "a",
                model_probability=0.75,
                market_probability=0.5,
                outcome=1,
                domain="economics",
            ),
            _forecast(
                "b",
                model_probability=0.75,
                market_probability=0.5,
                outcome=0,
                domain="politics",
            ),
            _forecast(
                "c",
                model_probability=0.75,
                market_probability=0.5,
                outcome=0,
                domain="politics",
            ),
            _forecast(
                "d",
                model_probability=0.75,
                market_probability=0.5,
                outcome=1,
                domain="politics",
            ),
        ]

        buckets = domain_probability_buckets(
            forecasts,
            prior_strength=3.0,
            min_domain_n=2,
        )
        by_domain = {row["domain"]: row for row in buckets}

        self.assertEqual(by_domain["economics"]["raw_outcome_rate"], 1.0)
        self.assertAlmostEqual(by_domain["economics"]["shrunk_outcome_rate"], 0.625)
        self.assertFalse(by_domain["economics"]["eligible_for_domain_model"])
        self.assertTrue(by_domain["politics"]["eligible_for_domain_model"])

    def test_skill_interval_clusters_revisions_by_market(self):
        forecasts = [
            _forecast(
                "a-early",
                market_id="a",
                model_probability=0.8,
                market_probability=0.5,
                outcome=1,
            ),
            _forecast(
                "a-late",
                market_id="a",
                model_probability=0.8,
                market_probability=0.5,
                outcome=1,
                forecasted_at=BASE_TIME + timedelta(hours=1),
            ),
            _forecast(
                "b",
                market_id="b",
                model_probability=0.2,
                market_probability=0.5,
                outcome=0,
            ),
        ]

        interval = market_clustered_brier_skill_interval(forecasts)

        self.assertEqual(interval["n_forecasts"], 3)
        self.assertEqual(interval["n_markets"], 2)
        self.assertAlmostEqual(interval["mean_skill"], 0.21)
        self.assertAlmostEqual(interval["lower"], 0.21)

    def test_trade_builder_uses_executable_ask_and_bid(self):
        forecasts = [
            _forecast(
                "yes",
                model_probability=0.7,
                market_probability=0.5,
                market_ask=0.55,
                outcome=1,
            ),
            _forecast(
                "no",
                model_probability=0.3,
                market_probability=0.5,
                market_bid=0.45,
                outcome=0,
            ),
        ]

        trades = build_trades(forecasts, min_edge=0.1)

        self.assertEqual([trade.side for trade in trades], ["NO", "YES"])
        self.assertTrue(all(trade.entry_price == 0.55 for trade in trades))

    def test_trade_builder_takes_only_first_qualifying_revision(self):
        forecasts = [
            _forecast(
                "first",
                model_probability=0.7,
                market_probability=0.5,
                outcome=1,
                market_id="same",
            ),
            _forecast(
                "later",
                model_probability=0.8,
                market_probability=0.5,
                outcome=1,
                market_id="same",
                forecasted_at=BASE_TIME + timedelta(hours=1),
            ),
        ]

        trades = build_trades(forecasts)

        self.assertEqual([trade.trade_id for trade in trades], ["first"])

    def test_sequential_returns_compound_on_current_equity(self):
        trades = [
            Trade(
                trade_id="win",
                opened_at=BASE_TIME,
                settled_at=BASE_TIME + timedelta(days=1),
                side="YES",
                entry_price=0.5,
                outcome=1,
                requested_fraction=0.1,
            ),
            Trade(
                trade_id="loss",
                opened_at=BASE_TIME + timedelta(days=2),
                settled_at=BASE_TIME + timedelta(days=3),
                side="YES",
                entry_price=0.5,
                outcome=0,
                requested_fraction=0.1,
            ),
        ]

        result = simulate_compounded_portfolio(trades, max_total_exposure=0.5)

        self.assertAlmostEqual(result["final_bankroll"], 99.0)
        self.assertAlmostEqual(result["compound_return"], -0.01)
        self.assertAlmostEqual(result["max_drawdown"], 0.1)
        self.assertEqual(result["open_positions"], 0)

    def test_overlapping_positions_respect_total_exposure_cap(self):
        trades = [
            Trade(
                trade_id="a-win",
                opened_at=BASE_TIME,
                settled_at=BASE_TIME + timedelta(days=1),
                side="YES",
                entry_price=0.5,
                outcome=1,
                requested_fraction=0.2,
            ),
            Trade(
                trade_id="b-loss",
                opened_at=BASE_TIME,
                settled_at=BASE_TIME + timedelta(days=1),
                side="YES",
                entry_price=0.5,
                outcome=0,
                requested_fraction=0.2,
            ),
        ]

        result = simulate_compounded_portfolio(trades, max_total_exposure=0.25)
        entry_points = [
            point for point in result["equity_curve"] if point["event"] == "entry"
        ]

        self.assertEqual(entry_points[-1]["open_exposure"], 25.0)
        self.assertAlmostEqual(result["final_bankroll"], 115.0)
        self.assertEqual(result["n_opened"], 2)


if __name__ == "__main__":
    unittest.main()
