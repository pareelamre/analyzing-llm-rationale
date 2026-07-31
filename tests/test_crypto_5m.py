from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import crypto_5m  # noqa: E402


def _klines_from_prices(prices):
    rows = []
    for i, price in enumerate(prices):
        open_time = i * 60_000
        open_price = prices[i - 1] if i else price
        high = max(open_price, price) * 1.0002
        low = min(open_price, price) * 0.9998
        close_time = open_time + 60_000 - 1
        rows.append([open_time, str(open_price), str(high), str(low), str(price), str(10 + i), close_time])
    return rows


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class Crypto5mTests(unittest.TestCase):
    def setUp(self):
        self._orig_requests = sys.modules.get("requests")

    def tearDown(self):
        if self._orig_requests is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = self._orig_requests

    def test_normalize_symbol_defaults_to_usdt(self):
        self.assertEqual(crypto_5m.normalize_symbol("btc"), "BTCUSDT")
        self.assertEqual(crypto_5m.normalize_symbol("eth/usdc"), "ETHUSDC")

    def test_forecast_prices_up_market_from_supplied_klines(self):
        prices = [100 + i * 0.1 for i in range(80)]

        result = crypto_5m.forecast_crypto_5m(
            "BTC",
            target_price=prices[-1] * 0.999,
            market_probability=45,
            klines=_klines_from_prices(prices),
        )

        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertGreater(result["probability_up"], 0.5)
        self.assertAlmostEqual(result["market_probability_up"], 0.45)
        self.assertEqual(result["recommendation"], "buy_up")
        self.assertIn("realized_vol_1m", result["features"])
        self.assertIn("ml_overlay", result["component_probabilities"])
        self.assertIn("ar1_ewma_timeseries", result["component_probabilities"])
        self.assertIn("ts_ar1_phi", result["features"])
        self.assertEqual(result["predicted_outcome"], "up")
        self.assertEqual(result["strategy"]["side"], "up")
        self.assertGreater(result["strategy"]["max_position_fraction"], 0.0)
        self.assertGreater(result["strategy"]["expected_value_per_contract"], 0.0)

    def test_forecast_uses_live_binance_klines_when_not_supplied(self):
        payload = _klines_from_prices([100 + i * 0.01 for i in range(40)])
        calls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append((url, params))
            return FakeResponse(payload)

        sys.modules["requests"] = SimpleNamespace(get=fake_get)

        result = crypto_5m.forecast_crypto_5m("ETH", lookback_minutes=40)

        self.assertEqual(result["symbol"], "ETHUSDT")
        self.assertEqual(calls[0][1]["interval"], "1m")
        self.assertEqual(calls[0][1]["limit"], 40)
        self.assertIn("ml_score", result["features"])

    def test_forecast_rejects_too_few_candles(self):
        with self.assertRaises(crypto_5m.CryptoModelError):
            crypto_5m.forecast_crypto_5m("BTC", klines=_klines_from_prices([100, 101]))

    def test_resolve_crypto_5m_outcome_scores_completed_market(self):
        rows = _klines_from_prices([100, 101, 102, 103, 104, 105])

        result = crypto_5m.resolve_crypto_5m_outcome(
            "BTC",
            target_price=102,
            start_time_ms=0,
            horizon_minutes=5,
            predicted_outcome="up",
            klines=rows,
            now_ms=6 * 60_000,
        )

        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["actual_outcome"], "up")
        self.assertTrue(result["prediction_correct"])
        self.assertEqual(result["resolved_price"], 104)

    def test_resolve_crypto_5m_outcome_returns_pending_before_expiry(self):
        result = crypto_5m.resolve_crypto_5m_outcome(
            "ETH",
            target_price=100,
            start_time_ms=0,
            horizon_minutes=5,
            now_ms=4 * 60_000,
        )

        self.assertEqual(result["status"], "pending")
        self.assertIsNone(result["actual_outcome"])
        self.assertIsNone(result["prediction_correct"])

    def test_resolve_crypto_5m_outcome_fetches_resolution_window(self):
        payload = _klines_from_prices([100, 99, 98, 97, 96, 95])
        calls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append((url, params))
            return FakeResponse(payload)

        sys.modules["requests"] = SimpleNamespace(get=fake_get)

        result = crypto_5m.resolve_crypto_5m_outcome(
            "SOL",
            target_price=100,
            start_time_ms=0,
            horizon_minutes=5,
            predicted_outcome="up",
            now_ms=10 * 60_000,
        )

        self.assertEqual(result["symbol"], "SOLUSDT")
        self.assertEqual(result["actual_outcome"], "down")
        self.assertFalse(result["prediction_correct"])
        self.assertEqual(calls[0][1]["startTime"], 0)
        self.assertEqual(calls[0][1]["endTime"], 5 * 60_000)

    def test_fetch_klines_history_paginates_recent_window(self):
        payload = _klines_from_prices([100 + i * 0.01 for i in range(1200)])
        calls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append((url, params))
            start_time = int(params.get("startTime", 0))
            end_time = int(params.get("endTime", 10**18))
            limit = int(params["limit"])
            batch = [
                row for row in payload
                if start_time <= row[0] <= end_time
            ][:limit]
            return FakeResponse(batch)

        sys.modules["requests"] = SimpleNamespace(get=fake_get)

        result = crypto_5m.fetch_klines_history(
            "BTC",
            days=1,
            end_time_ms=payload[-1][0],
            max_candles=1100,
        )

        self.assertEqual(len(result), 1100)
        self.assertEqual(calls[0][1]["limit"], 1000)
        self.assertEqual(calls[1][1]["startTime"], payload[1000][0])
        self.assertEqual(calls[1][1]["limit"], 100)

    def test_adaptive_forecast_fits_recent_logistic_overlay(self):
        prices = [
            100 + i * 0.002 + math.sin(i / 3) * 0.35
            for i in range(180)
        ]

        result = crypto_5m.forecast_crypto_5m(
            "SOL",
            klines=_klines_from_prices(prices),
            lookback_minutes=40,
            ml_mode="adaptive",
            training_window=80,
        )

        self.assertEqual(result["symbol"], "SOLUSDT")
        self.assertEqual(result["features"]["ml_mode"], "adaptive")
        self.assertEqual(result["features"]["ml_mode_effective"], "adaptive")
        self.assertGreaterEqual(result["features"]["ml_training_examples"], 40)
        self.assertIn("ml_coefficients", result["features"])
        self.assertIn("ml_validation", result["features"])

    def test_edge_threshold_controls_recommendation(self):
        prices = [100 + i * 0.08 for i in range(90)]

        loose = crypto_5m.forecast_crypto_5m(
            "BTC",
            target_price=prices[-1] * 0.999,
            market_probability=0.5,
            edge_threshold=0.0,
            klines=_klines_from_prices(prices),
        )
        strict = crypto_5m.forecast_crypto_5m(
            "BTC",
            target_price=prices[-1] * 0.999,
            market_probability=0.5,
            edge_threshold=0.5,
            klines=_klines_from_prices(prices),
        )

        self.assertEqual(loose["recommendation"], "buy_up")
        self.assertEqual(strict["recommendation"], "hold")
        self.assertEqual(strict["strategy"]["edge_threshold"], 0.5)

    def test_fee_gate_blocks_positive_edge_that_is_not_profitable(self):
        prices = [100 + i * 0.02 for i in range(90)]

        free = crypto_5m.forecast_crypto_5m(
            "BTC",
            target_price=prices[-1],
            market_probability=0.49,
            edge_threshold=0.0,
            fee_bps=0,
            klines=_klines_from_prices(prices),
        )
        expensive = crypto_5m.forecast_crypto_5m(
            "BTC",
            target_price=prices[-1],
            market_probability=0.49,
            edge_threshold=0.0,
            fee_bps=5000,
            klines=_klines_from_prices(prices),
        )

        self.assertEqual(free["recommendation"], "buy_up")
        self.assertEqual(expensive["recommendation"], "hold")
        self.assertLess(expensive["strategy"]["expected_value_per_contract"], 0.0)
        self.assertEqual(expensive["strategy"]["fee_bps"], 5000)

    def test_walk_forward_backtest_reports_metrics_and_rows(self):
        prices = [100 + i * 0.04 for i in range(120)]

        result = crypto_5m.walk_forward_backtest(
            "BTC",
            klines=_klines_from_prices(prices),
            horizon_minutes=5,
            lookback_minutes=40,
            market_probability=0.45,
            max_rows=20,
            include_rows=True,
        )

        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertEqual(result["n"], 20)
        self.assertEqual(len(result["rows"]), 20)
        self.assertIn("brier_score", result)
        self.assertIn("predicted_outcome", result["rows"][0])
        self.assertIn("calibration", result)
        self.assertEqual(sum(bucket["n"] for bucket in result["calibration"]), result["n"])
        self.assertGreaterEqual(result["trades"], 1)
        self.assertIsNotNone(result["hit_rate"])

    def test_walk_forward_backtest_no_trades_when_edge_inside_band(self):
        prices = [100 + ((i % 2) * 0.01) for i in range(80)]

        result = crypto_5m.walk_forward_backtest(
            "ETH",
            klines=_klines_from_prices(prices),
            horizon_minutes=5,
            lookback_minutes=35,
            market_probability=0.5,
            max_rows=10,
        )

        self.assertEqual(result["n"], 10)
        self.assertEqual(result["trades"], 0)
        self.assertIsNone(result["hit_rate"])
        self.assertEqual(result["pnl_per_contract"], 0)

    def test_walk_forward_live_fetch_requests_enough_klines_for_max_rows(self):
        payload = _klines_from_prices([100 + i * 0.01 for i in range(100)])
        calls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append((url, params))
            return FakeResponse(payload)

        sys.modules["requests"] = SimpleNamespace(get=fake_get)

        result = crypto_5m.walk_forward_backtest(
            "BTC",
            horizon_minutes=5,
            lookback_minutes=40,
            max_rows=20,
        )

        self.assertEqual(calls[0][1]["limit"], 65)
        self.assertEqual(result["n"], 20)

    def test_walk_forward_backtest_supports_adaptive_ml(self):
        prices = [
            100 + i * 0.001 + math.sin(i / 4) * 0.3
            for i in range(170)
        ]

        result = crypto_5m.walk_forward_backtest(
            "BTC",
            klines=_klines_from_prices(prices),
            horizon_minutes=5,
            lookback_minutes=40,
            market_probability=0.5,
            ml_mode="adaptive",
            training_window=60,
            max_rows=10,
        )

        self.assertEqual(result["ml_mode"], "adaptive")
        self.assertEqual(result["training_window"], 60)
        self.assertEqual(result["n"], 10)
        self.assertIn("brier_score", result)

    def test_sweep_compares_thresholds_and_model_modes(self):
        prices = [
            100 + i * 0.001 + math.sin(i / 4) * 0.3
            for i in range(170)
        ]

        result = crypto_5m.sweep_crypto_5m_thresholds(
            "BTC",
            klines=_klines_from_prices(prices),
            horizon_minutes=5,
            lookback_minutes=40,
            market_probability=0.5,
            ml_modes=["fixed", "adaptive"],
            training_window=60,
            edge_thresholds=[0.0, 0.03, 0.1],
            max_rows=8,
            selection_fraction=0.5,
            folds=2,
        )

        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertEqual(result["thresholds"], [0.0, 0.03, 0.1])
        self.assertEqual(result["selection_fraction"], 0.5)
        self.assertEqual(result["folds"], 2)
        self.assertEqual(len(result["models"]), 2)
        self.assertIn(result["best"]["ml_mode"], {"fixed", "adaptive"})
        for model in result["models"]:
            self.assertGreater(model["selection_rows"], 0)
            self.assertGreater(model["holdout_rows"], 0)
            self.assertEqual(len(model["thresholds"]), 3)
            self.assertIn("selected_threshold", model)
            self.assertIn("holdout_selected_threshold", model)
            self.assertIn("selection", model["selected_threshold"])
            self.assertIn("holdout", model["selected_threshold"])
            self.assertEqual(len(model["folds"]), 2)
            self.assertIn("holdout_selected_threshold", model["folds"][0])

    def test_benchmark_aggregates_multi_symbol_holdouts(self):
        klines_by_symbol = {
            "BTCUSDT": _klines_from_prices([
                100 + i * 0.001 + math.sin(i / 4) * 0.3
                for i in range(170)
            ]),
            "ETHUSDT": _klines_from_prices([
                80 + i * 0.002 + math.sin(i / 5) * 0.2
                for i in range(170)
            ]),
            "SOLUSDT": _klines_from_prices([
                20 + i * 0.0015 + math.sin(i / 6) * 0.12
                for i in range(170)
            ]),
        }

        result = crypto_5m.benchmark_crypto_5m_strategy(
            ["BTC", "ETH", "SOL"],
            klines_by_symbol=klines_by_symbol,
            horizon_minutes=5,
            lookback_minutes=40,
            market_probability=0.5,
            ml_modes=["fixed", "adaptive"],
            training_window=60,
            edge_thresholds=[0.0, 0.03, 0.08],
            max_rows=8,
            selection_fraction=0.5,
            folds=2,
        )

        self.assertEqual(result["symbols_requested"], ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        self.assertEqual(result["symbols_evaluated"], ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        self.assertEqual(len(result["sweeps"]), 3)
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["aggregate"]["symbols"], 3)
        self.assertIn("holdout_pnl_per_contract", result["aggregate"])
        self.assertIn("stable_selection", result["aggregate"])
        self.assertIn("symbols_with_holdout_trades", result["aggregate"])
        self.assertIn("trade_coverage", result["aggregate"])
        self.assertIn(result["aggregate"]["evidence_quality"], {"none", "thin", "moderate", "strong"})
        self.assertEqual(result["folds"], 2)
        self.assertIn("fold_aggregate", result)
        self.assertIn("folds_with_holdout_trades", result["fold_aggregate"])
        self.assertIn(result["fold_aggregate"]["evidence_quality"], {"none", "thin", "moderate", "strong"})

    def test_benchmark_log_appends_compact_evidence_record(self):
        result = {
            "symbols_requested": ["BTCUSDT", "ETHUSDT"],
            "symbols_evaluated": ["BTCUSDT"],
            "horizon_minutes": 5,
            "lookback_minutes": 40,
            "history_days": 1,
            "max_candles": 1600,
            "market_probability_up": 0.5,
            "fee_bps": 2,
            "selection_fraction": 0.6,
            "folds": 2,
            "aggregate": {"total_holdout_trades": 4, "holdout_pnl_per_contract": 0.12},
            "fold_aggregate": {"folds_with_holdout_trades": 2, "evidence_quality": "thin"},
            "failures": [{"symbol": "ETHUSDT", "error": "sample"}],
            "sweeps": [{
                "symbol": "BTCUSDT",
                "best": {
                    "ml_mode": "fixed",
                    "edge_threshold": 0.03,
                    "holdout": {
                        "trades": 4,
                        "pnl_per_contract": 0.12,
                        "hit_rate": 0.75,
                    },
                },
            }],
        }

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "benchmarks.jsonl"
            first = crypto_5m.append_crypto_5m_benchmark_log(result, path=path)
            second = crypto_5m.append_crypto_5m_benchmark_log(result, path=path)

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(first["path"], str(path))
            self.assertEqual(second["record"]["type"], "crypto_5m_benchmark")
            self.assertEqual(records[0]["best_by_symbol"][0]["symbol"], "BTCUSDT")
            self.assertEqual(records[0]["failures"][0]["symbol"], "ETHUSDT")

    def test_unstable_fold_selection_caps_evidence_quality_at_thin(self):
        sweeps = [{
            "symbol": "BTCUSDT",
            "models": [{
                "ml_mode": "fixed",
                "folds": [{
                    "selected_threshold": {"edge_threshold": 0.01},
                    "holdout_selected_threshold": {
                        "trades": 40,
                        "pnl_per_contract": 2.0,
                        "hit_rate": 0.55,
                    },
                }],
            }, {
                "ml_mode": "adaptive",
                "folds": [{
                    "selected_threshold": {"edge_threshold": 0.08},
                    "holdout_selected_threshold": {
                        "trades": 40,
                        "pnl_per_contract": 2.0,
                        "hit_rate": 0.55,
                    },
                }],
            }],
        }]

        result = crypto_5m._aggregate_fold_holdouts(sweeps)

        self.assertEqual(result["total_holdout_trades"], 80)
        self.assertFalse(result["stable_selection"])
        self.assertEqual(result["evidence_quality"], "thin")

    def test_signal_log_records_and_resolves_paper_trade(self):
        history = _klines_from_prices([100 + i * 0.05 for i in range(80)])
        future = _klines_from_prices([100 + i * 0.05 for i in range(90)])

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "signals.jsonl"
            opened = crypto_5m.record_crypto_5m_signal(
                "BTC",
                market_probability=0.45,
                fee_bps=2,
                edge_threshold=0.0,
                klines=history,
                path=path,
            )
            record = opened["record"]
            result = crypto_5m.resolve_crypto_5m_signal_log(
                path=path,
                now_ms=record["expiry_time_ms"] + 1,
                klines_by_symbol={"BTCUSDT": future},
            )

            self.assertEqual(result["records"], 1)
            self.assertEqual(result["resolved_now"], 1)
            self.assertEqual(result["open"], 0)
            saved = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(saved[0]["status"], "resolved")
            self.assertIn(saved[0]["actual_outcome"], {"up", "down", "tie"})
            self.assertIn("pnl_per_contract", saved[0])

    def test_signal_log_can_record_fade_5m_momentum_strategy(self):
        history = _klines_from_prices([100 + i * 0.12 for i in range(80)])

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "signals.jsonl"
            db_path = Path(tmp) / "signals.sqlite"
            opened = crypto_5m.record_crypto_5m_signal(
                "BTC",
                market_probability=0.5,
                fee_bps=2,
                edge_threshold=0.5,
                strategy_mode="fade_5m_momentum",
                momentum_threshold=0.001,
                klines=history,
                path=path,
                db_path=db_path,
            )
            record = opened["record"]

            self.assertEqual(record["strategy_name"], "fade_5m_momentum")
            self.assertEqual(record["recommendation"], "buy_down")
            self.assertEqual(record["predicted_outcome"], "down")
            self.assertEqual(record["strategy"]["side"], "down")
            self.assertEqual(record["strategy"]["strategy_mode"], "fade_5m_momentum")

            with closing(sqlite3.connect(db_path)) as conn:
                row = conn.execute(
                    "SELECT strategy_name, recommendation FROM crypto_5m_signals"
                ).fetchone()
            self.assertEqual(row, ("fade_5m_momentum", "buy_down"))

    def test_signal_log_can_record_follow_5m_momentum_strategy(self):
        history = _klines_from_prices([100 + i * 0.12 for i in range(80)])

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "signals.jsonl"
            opened = crypto_5m.record_crypto_5m_signal(
                "BTC",
                market_probability=0.5,
                fee_bps=2,
                edge_threshold=0.5,
                strategy_mode="follow_5m_momentum",
                momentum_threshold=0.001,
                klines=history,
                path=path,
            )
            record = opened["record"]

            self.assertEqual(record["strategy_name"], "follow_5m_momentum")
            self.assertEqual(record["recommendation"], "buy_up")
            self.assertEqual(record["predicted_outcome"], "up")
            self.assertEqual(record["strategy"]["side"], "up")
            self.assertEqual(record["strategy"]["strategy_mode"], "follow_5m_momentum")

    def test_backfill_seeds_resolved_history_into_signal_db(self):
        # A long synthetic 1-minute series stands in for fetched Binance history,
        # so the backfill is exercised without any network access.
        series = _klines_from_prices([100 + math.sin(i / 9.0) * 4 + i * 0.01 for i in range(600)])
        orig = crypto_5m.fetch_klines_history
        crypto_5m.fetch_klines_history = lambda symbol, **_: series
        try:
            with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
                log_path = Path(tmp) / "backfill.jsonl"
                db_path = Path(tmp) / "signals.sqlite"
                result = crypto_5m.backfill_crypto_5m_signals(
                    ["BTC"],
                    days=1,
                    lookback_minutes=60,
                    training_window=120,
                    ml_mode="adaptive",
                    step_minutes=5,
                    path=log_path,
                    db_path=db_path,
                )

                self.assertGreater(result["recorded"], 0)
                self.assertEqual(result["recorded"], result["resolved"]["records"])
                self.assertEqual(result["resolved"]["open"], 0)

                equity = crypto_5m.crypto_5m_candidate_equity(db_path=db_path, since_hours=0)
                self.assertEqual(equity["resolved_rows"], result["recorded"])
                self.assertIsNotNone(equity["coverage"]["days"])
                for curve in equity["curves"]:
                    self.assertIn("max_drawdown", curve)
                    self.assertIn("pnl_per_day", curve)
        finally:
            crypto_5m.fetch_klines_history = orig

    def test_candidate_equity_all_time_window_keeps_every_row(self):
        history = _klines_from_prices([100 + i * 0.05 for i in range(80)])
        future = _klines_from_prices([100 + i * 0.05 for i in range(90)])
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "signals.jsonl"
            db_path = Path(tmp) / "signals.sqlite"
            opened = crypto_5m.record_crypto_5m_signal(
                "BTC", market_probability=0.45, fee_bps=2, edge_threshold=0.0,
                klines=history, path=path, db_path=db_path,
            )
            crypto_5m.resolve_crypto_5m_signal_log(
                path=path, db_path=db_path,
                now_ms=opened["record"]["expiry_time_ms"] + 1,
                klines_by_symbol={"BTCUSDT": future},
            )
            full = crypto_5m.crypto_5m_candidate_equity(db_path=db_path, since_hours=0)
            self.assertEqual(full["since_hours"], 0)
            self.assertIsNone(full["cutoff_start_time_ms"])

    def test_gzipped_seed_imports_into_signal_db(self):
        # The committed backfill seed is gzipped; the tick imports it straight
        # into the signal DB, so the equity curve keeps its long-term history.
        import gzip
        history = _klines_from_prices([100 + i * 0.05 for i in range(80)])
        future = _klines_from_prices([100 + i * 0.05 for i in range(90)])
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            log = Path(tmp) / "signals.jsonl"
            db = Path(tmp) / "signals.sqlite"
            opened = crypto_5m.record_crypto_5m_signal(
                "BTC", market_probability=0.45, fee_bps=2, edge_threshold=0.0,
                klines=history, path=log,
            )
            crypto_5m.resolve_crypto_5m_signal_log(
                path=log, now_ms=opened["record"]["expiry_time_ms"] + 1,
                klines_by_symbol={"BTCUSDT": future},
            )
            seed = Path(tmp) / "seed.jsonl.gz"
            seed.write_bytes(gzip.compress(log.read_bytes()))

            result = crypto_5m.import_crypto_5m_signal_log_to_db(path=seed, db_path=db)
            self.assertGreater(result["records"], 0)
            equity = crypto_5m.crypto_5m_candidate_equity(db_path=db, since_hours=0)
            self.assertEqual(equity["resolved_rows"], result["records"])

    def test_remote_equity_fallback_bypasses_raw_github_cache(self):
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append((url, headers, timeout))
            return FakeResponse({"generated_at": "fresh", "curves": []})

        sys.modules["requests"] = SimpleNamespace(get=fake_get)
        original = os.environ.get("CRYPTO_5M_EQUITY_URL")
        os.environ["CRYPTO_5M_EQUITY_URL"] = "https://example.test/equity.json?ref=main"
        original_time = crypto_5m.time.time
        crypto_5m.time.time = lambda: 1234.567
        try:
            payload = crypto_5m._load_crypto_5m_equity_fallback(Path("missing.sqlite"))
        finally:
            crypto_5m.time.time = original_time
            if original is None:
                os.environ.pop("CRYPTO_5M_EQUITY_URL", None)
            else:
                os.environ["CRYPTO_5M_EQUITY_URL"] = original

        self.assertEqual(payload["generated_at"], "fresh")
        self.assertEqual(payload["fallback_source"], "https://example.test/equity.json?ref=main")
        self.assertEqual(calls[0][0], "https://example.test/equity.json?ref=main&_ts=1234567")
        self.assertEqual(calls[0][1]["Cache-Control"], "no-cache")
        self.assertEqual(calls[0][1]["Pragma"], "no-cache")

    def test_per_asset_regime_selector_uses_inverse_edge_for_btc(self):
        forecast = {
            "symbol": "BTCUSDT",
            "edge": 0.05,
            "features": {"momentum_5m": 0.003, "realized_vol_1m": 0.0007},
        }

        side, strategy = crypto_5m._per_asset_regime_strategy(
            "BTC",
            forecast,
            momentum_threshold=0.0025,
            fee_bps=2,
        )

        self.assertEqual(side, "down")
        self.assertEqual(strategy["recommendation"], "buy_down")
        self.assertEqual(strategy["strategy_mode"], "per_asset_regime_selector")
        self.assertEqual(strategy["selected_expert"], "inverse_edge_004")

    def test_per_asset_regime_report_ranks_candidates_by_symbol(self):
        records = []
        for idx in range(24):
            actual = "down" if idx < 18 else "up"
            records.append({
                "type": "crypto_5m_signal",
                "status": "resolved",
                "strategy_name": "follow_5m_momentum",
                "symbol": "BTCUSDT",
                "recommendation": "hold",
                "predicted_outcome": None,
                "actual_outcome": actual,
                "prediction_correct": None,
                "start_time_ms": idx,
                "expiry_time_ms": idx + 100,
                "horizon_minutes": 5,
                "target_price": 100.0,
                "resolved_price": 99.0 if actual == "down" else 101.0,
                "probability_up": 0.55,
                "market_probability_up": 0.5,
                "edge": 0.05,
                "fee_bps": 2,
                "strategy": {"side": "none"},
                "features": {"momentum_5m": 0.001, "realized_vol_1m": 0.0005},
            })

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            db_path = Path(tmp) / "signals.sqlite"
            crypto_5m.upsert_crypto_5m_signal_records(records, db_path=db_path)

            result = crypto_5m.crypto_5m_per_asset_regime_report(
                db_path=db_path,
                min_trades=10,
            )

            selected = result["by_symbol"]["BTCUSDT"]["selected"]
            self.assertEqual(selected["candidate"], "inverse_edge_004")
            self.assertEqual(selected["all"]["trades"], 24)
            self.assertEqual(selected["all"]["hit_rate"], 0.75)

    def test_candidate_equity_reports_curves_for_crypto_strategy_lab(self):
        records = []
        for idx in range(24):
            actual = "down" if idx < 18 else "up"
            records.append({
                "type": "crypto_5m_signal",
                "status": "resolved",
                "strategy_name": "per_asset_regime_selector",
                "symbol": "BTCUSDT",
                "recommendation": "hold",
                "predicted_outcome": None,
                "actual_outcome": actual,
                "prediction_correct": None,
                "start_time_ms": idx * 60_000,
                "expiry_time_ms": idx * 60_000 + 300_000,
                "horizon_minutes": 5,
                "target_price": 100.0,
                "resolved_price": 99.0 if actual == "down" else 101.0,
                "probability_up": 0.55,
                "market_probability_up": 0.5,
                "edge": 0.05,
                "fee_bps": 2,
                "strategy": {"side": "none"},
                "features": {"momentum_5m": 0.001, "realized_vol_1m": 0.0005},
            })

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            db_path = Path(tmp) / "signals.sqlite"
            crypto_5m.upsert_crypto_5m_signal_records(records, db_path=db_path)

            result = crypto_5m.crypto_5m_candidate_equity(
                db_path=db_path,
                since_hours=None,
            )

            curve = next(c for c in result["curves"] if c["key"] == "btc_inverse_edge_004")
            self.assertEqual(curve["trades"], 24)
            self.assertEqual(curve["hit_rate"], 0.75)
            self.assertEqual(len(curve["points"]), 24)
            self.assertGreater(curve["pnl_per_contract"], 0)
            self.assertEqual(result["recommendation"]["candidate"], "selector_btc_inverse_eth_fade")
            self.assertIn("strong BTC model calls", result["recommendation"]["policy"])

    def test_ott_filtered_selector_requires_trend_agreement(self):
        record = {
            "symbol": "BTCUSDT",
            "start_time_ms": 0,
            "expiry_time_ms": 300_000,
            "horizon_minutes": 5,
            "target_price": 100.0,
            "edge": 0.05,
            "features": {"momentum_5m": 0.0},
        }
        record_key = crypto_5m._signal_record_id(record)

        self.assertEqual(
            crypto_5m._crypto_replay_side(
                record,
                "selector_ott_filter",
                ott_trend_by_id={record_key: "down"},
            ),
            "down",
        )
        self.assertIsNone(
            crypto_5m._crypto_replay_side(
                record,
                "selector_ott_filter",
                ott_trend_by_id={record_key: "up"},
            )
        )

    def test_signal_db_upserts_recorded_and_resolved_signals(self):
        history = _klines_from_prices([100 + i * 0.05 for i in range(80)])
        future = _klines_from_prices([100 + i * 0.05 for i in range(90)])

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "signals.jsonl"
            db_path = Path(tmp) / "signals.sqlite"
            opened = crypto_5m.record_crypto_5m_signal(
                "BTC",
                market_probability=0.45,
                fee_bps=2,
                edge_threshold=0.0,
                klines=history,
                path=path,
                db_path=db_path,
            )
            record = opened["record"]
            self.assertEqual(opened["db"]["upserted"], 1)

            crypto_5m.resolve_crypto_5m_signal_log(
                path=path,
                db_path=db_path,
                now_ms=record["expiry_time_ms"] + 1,
                klines_by_symbol={"BTCUSDT": future},
            )

            with closing(sqlite3.connect(db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT symbol, status, strategy_name, recommendation, actual_outcome,
                           pnl_per_contract, realized_vol_1m, momentum_5m,
                           ml_mode_effective
                    FROM crypto_5m_signals
                    """
                ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "BTCUSDT")
            self.assertEqual(rows[0][1], "resolved")
            self.assertEqual(rows[0][2], "adaptive_model")
            self.assertIn(rows[0][3], {"buy_up", "buy_down", "hold"})
            self.assertIn(rows[0][4], {"up", "down", "tie"})
            self.assertIsNotNone(rows[0][5])
            self.assertIsNotNone(rows[0][6])
            self.assertIsNotNone(rows[0][7])
            self.assertIn(rows[0][8], {"fixed", "adaptive", "fixed_fallback"})

    def test_signal_db_analysis_reports_regimes_and_baselines(self):
        records = [{
            "type": "crypto_5m_signal",
            "status": "resolved",
            "strategy_name": "adaptive_model",
            "symbol": "BTCUSDT",
            "recommendation": "buy_up",
            "predicted_outcome": "up",
            "actual_outcome": "up",
            "prediction_correct": True,
            "start_time_ms": 1,
            "expiry_time_ms": 2,
            "horizon_minutes": 5,
            "target_price": 100.0,
            "resolved_price": 101.0,
            "probability_up": 0.56,
            "market_probability_up": 0.5,
            "edge": 0.06,
            "fee_bps": 2,
            "pnl_per_contract": 0.4998,
            "strategy": {"side": "up", "edge_threshold": 0.03},
            "features": {
                "realized_vol_1m": 0.0005,
                "momentum_5m": 0.0012,
                "reversal_1m_z": -0.5,
                "ml_mode_effective": "adaptive",
                "ml_validation": {"accuracy": 0.56},
            },
        }, {
            "type": "crypto_5m_signal",
            "status": "resolved",
            "strategy_name": "fade_5m_momentum",
            "symbol": "ETHUSDT",
            "recommendation": "buy_down",
            "predicted_outcome": "down",
            "actual_outcome": "up",
            "prediction_correct": False,
            "start_time_ms": 3,
            "expiry_time_ms": 4,
            "horizon_minutes": 5,
            "target_price": 100.0,
            "resolved_price": 101.0,
            "probability_up": 0.44,
            "market_probability_up": 0.5,
            "edge": -0.06,
            "fee_bps": 2,
            "pnl_per_contract": -0.5002,
            "strategy": {"side": "down", "edge_threshold": 0.03},
            "features": {
                "realized_vol_1m": 0.001,
                "momentum_5m": -0.0012,
                "reversal_1m_z": 0.5,
                "ml_mode_effective": "adaptive",
                "ml_validation": {"accuracy": 0.40},
            },
        }]

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            db_path = Path(tmp) / "signals.sqlite"
            crypto_5m.upsert_crypto_5m_signal_records(records, db_path=db_path)
            result = crypto_5m.analyze_crypto_5m_signal_db(db_path=db_path)

            self.assertEqual(result["overall"]["signals"], 2)
            self.assertEqual(result["overall"]["trades"], 2)
            self.assertEqual(result["overall"]["trade_hit_rate"], 0.5)
            self.assertEqual(result["overall"]["baselines"]["always_up"]["accuracy"], 1.0)
            self.assertIn("adaptive_model", result["by_strategy"])
            self.assertIn("fade_5m_momentum", result["by_strategy"])
            self.assertIn("0.03", result["by_threshold"])
            self.assertIn("BTCUSDT", result["by_symbol"])
            self.assertIn(">=5pp", result["by_edge_bucket"])
            self.assertIn("normal", result["by_volatility_bucket"])
            self.assertIn("strong_up", result["by_momentum_5m_bucket"])
            self.assertIn("bad", result["by_validation_accuracy_bucket"])

    def test_signal_log_does_not_assign_pnl_to_hold(self):
        history = _klines_from_prices([100 + i * 0.01 for i in range(80)])
        future = _klines_from_prices([100 + i * 0.01 for i in range(90)])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.jsonl"
            opened = crypto_5m.record_crypto_5m_signal(
                "BTC",
                market_probability=0.5,
                edge_threshold=0.5,
                klines=history,
                path=path,
            )
            record = opened["record"]
            self.assertEqual(record["recommendation"], "hold")

            crypto_5m.resolve_crypto_5m_signal_log(
                path=path,
                now_ms=record["expiry_time_ms"] + 1,
                klines_by_symbol={"BTCUSDT": future},
            )

            saved = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(saved[0]["status"], "resolved")
            self.assertIsNone(saved[0]["pnl_per_contract"])

    def test_signal_summary_reports_trade_readiness_and_ignores_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.jsonl"
            records = [
                {
                    "type": "crypto_5m_signal",
                    "status": "resolved",
                    "symbol": "BTCUSDT",
                    "recommendation": "buy_up",
                    "strategy": {"side": "up"},
                    "actual_outcome": "up",
                    "prediction_correct": True,
                    "pnl_per_contract": 0.498,
                },
                {
                    "type": "crypto_5m_signal",
                    "status": "resolved",
                    "symbol": "ETHUSDT",
                    "recommendation": "buy_down",
                    "strategy": {"side": "down"},
                    "actual_outcome": "down",
                    "prediction_correct": True,
                    "pnl_per_contract": 0.398,
                },
                {
                    "type": "crypto_5m_signal",
                    "status": "resolved",
                    "symbol": "BTCUSDT",
                    "recommendation": "hold",
                    "strategy": {"side": "up"},
                    "actual_outcome": "up",
                    "prediction_correct": False,
                    "pnl_per_contract": 0.498,
                },
                {
                    "type": "crypto_5m_signal",
                    "status": "open",
                    "symbol": "SOLUSDT",
                    "recommendation": "buy_up",
                    "strategy": {"side": "up"},
                },
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records))

            result = crypto_5m.summarize_crypto_5m_signal_log(
                path=path,
                min_resolved_trades=2,
                min_total_pnl=0,
                min_hit_rate=0.5,
            )

            self.assertTrue(result["trade_ready"])
            self.assertEqual(result["signals"], 4)
            self.assertEqual(result["open"], 1)
            self.assertEqual(result["resolved"], 3)
            self.assertEqual(result["scored"], 3)
            self.assertEqual(result["accuracy"], 0.6667)
            self.assertEqual(result["trades"], 2)
            self.assertEqual(result["trade_hit_rate"], 1.0)
            self.assertAlmostEqual(result["pnl_per_contract"], 0.896)
            self.assertEqual(result["by_recommendation"]["hold"]["trades"], 0)
            self.assertEqual(result["by_symbol"]["BTCUSDT"]["trades"], 1)

    def test_paper_loop_dry_run_does_not_write_log(self):
        payload = _klines_from_prices([100 + i * 0.01 for i in range(80)])

        def fake_get(url, params=None, headers=None, timeout=None):
            return FakeResponse(payload)

        sys.modules["requests"] = SimpleNamespace(get=fake_get)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.jsonl"
            result = crypto_5m.run_crypto_5m_paper_loop(
                ["BTC"],
                iterations=1,
                path=path,
                market_probability=0.5,
                dry_run=True,
            )

            self.assertTrue(result["dry_run"])
            self.assertFalse(path.exists())
            self.assertEqual(len(result["iteration_results"]), 1)
            self.assertEqual(len(result["iteration_results"][0]["signals"]), 1)

    def test_paper_loop_records_signals_when_not_dry_run(self):
        payload = _klines_from_prices([100 + i * 0.01 for i in range(80)])

        def fake_get(url, params=None, headers=None, timeout=None):
            return FakeResponse(payload)

        sys.modules["requests"] = SimpleNamespace(get=fake_get)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.jsonl"
            result = crypto_5m.run_crypto_5m_paper_loop(
                ["BTC"],
                iterations=1,
                path=path,
                market_probability=0.5,
                sleep_seconds=0,
            )

            self.assertFalse(result["dry_run"])
            self.assertTrue(path.exists())
            saved = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["type"], "crypto_5m_signal")
            self.assertIn(saved[0]["status"], {"open", "resolved"})

    def test_compute_ott_returns_trend_state(self):
        prices = [100, 105, 110, 115, 90, 80, 70, 60, 65, 75, 85]
        candles = crypto_5m._parse_candles(_klines_from_prices(prices))

        result = crypto_5m.compute_ott(candles, length=2, percent=1.4)

        self.assertEqual(len(result), len(prices))
        self.assertIn(result[-1]["trend_side"], {"up", "down"})
        self.assertIn("support", result[-1])
        self.assertIn("ott", result[-1])
        self.assertTrue(any(row["reversal"] for row in result))

    def test_compare_ott_crypto_5m_strategies_reports_requested_candidates(self):
        base_prices = [100 + math.sin(i / 3) * 0.3 + i * 0.01 for i in range(90)]
        klines_by_symbol = {
            "BTCUSDT": _klines_from_prices(base_prices),
            "ETHUSDT": _klines_from_prices([price * 1.1 for price in base_prices]),
            "SOLUSDT": _klines_from_prices([price * 0.7 for price in base_prices]),
        }

        result = crypto_5m.compare_ott_crypto_5m_strategies(
            ["BTC", "ETH", "SOL"],
            klines_by_symbol=klines_by_symbol,
            horizon_minutes=5,
            lookback_minutes=30,
            market_probability=0.5,
            fee_bps=2,
            max_rows=20,
            ml_mode="fixed",
        )

        self.assertEqual(result["symbols_evaluated"], ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        self.assertEqual(
            set(result["aggregate"]),
            {"ott_follow", "ott_fade_after_reversal", "current_selector", "selector_ott_filter"},
        )
        self.assertEqual(result["by_symbol"]["BTCUSDT"]["rows"], 20)
        self.assertEqual(result["best"]["strategy"], result["ranked"][0]["strategy"])


if __name__ == "__main__":
    unittest.main()
