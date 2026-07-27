from __future__ import annotations

import asyncio
import importlib.util
import tempfile
import threading
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest import mock

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "track_record_tick.py"
_SPEC = importlib.util.spec_from_file_location("track_record_tick_test_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
track_record_tick = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(track_record_tick)


class TrackRecordTickTests(unittest.TestCase):
    def test_forecast_fn_rejects_response_from_a_different_model(self):
        quote = {
            "question": "Will the test event happen?",
            "platform": "Polymarket",
            "market_url": "https://polymarket.com/event/test",
            "probability": 0.4,
        }
        response = {
            "model_key": "gpt-oss-120b",
            "market_analysis": {
                "model_probability": 0.7,
                "market_probability": 0.4,
            },
        }
        with (
            mock.patch.object(track_record_tick, "_post_predict", return_value=response),
            mock.patch.object(track_record_tick, "_predict_stats", Counter()) as stats,
        ):
            result = asyncio.run(
                track_record_tick.forecast_fn(quote, 3, model="council")
            )

        self.assertIsNone(result)
        self.assertEqual(stats["model_mismatches"], 1)

    def test_predict_circuit_skips_queued_calls_after_repeated_failures(self):
        circuit = threading.Event()
        with (
            mock.patch.object(track_record_tick, "_predict_stats", Counter()),
            mock.patch.object(track_record_tick, "_predict_circuit_open", circuit),
            mock.patch.object(track_record_tick, "_predict_consecutive_failures", 0),
            mock.patch.object(track_record_tick, "_PREDICT_FAILURE_CIRCUIT_THRESHOLD", 2),
            mock.patch.object(track_record_tick, "_PREDICT_RETRIES", 1),
            mock.patch.object(track_record_tick, "_PREDICT_MIN_INTERVAL_S", 0.0),
            mock.patch.object(track_record_tick, "_last_predict_ts", 0.0),
            mock.patch.object(
                track_record_tick.urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("upstream unavailable"),
            ) as urlopen_mock,
        ):
            self.assertIsNone(track_record_tick._post_predict({"question": "q1"}))
            self.assertIsNone(track_record_tick._post_predict({"question": "q2"}))
            self.assertTrue(circuit.is_set())
            self.assertIsNone(track_record_tick._post_predict({"question": "queued"}))

            self.assertEqual(urlopen_mock.call_count, 2)
            self.assertEqual(track_record_tick._predict_stats["attempts"], 2)
            self.assertEqual(track_record_tick._predict_stats["failures"], 2)
            self.assertEqual(track_record_tick._predict_stats["circuit_opened"], 1)
            self.assertEqual(track_record_tick._predict_stats["circuit_skipped"], 1)

    def test_main_uses_configured_primary_independent_of_model_order(self):
        progress = {
            "snapshots": 10,
            "resolved_snapshots": 4,
            "latest_snapshot_ts": "2026-07-18T21:08:23.295099+00:00",
            "latest_resolved_ts": "2026-07-18T20:00:00+00:00",
        }
        aggregate = {
            "n_markets_resolved": 210,
            "n_markets_open": 39,
            "n_snapshots_resolved": 1310,
        }
        store = mock.Mock()
        with tempfile.TemporaryDirectory() as td:
            public_path = Path(td) / "track_record_live.json"
            evaluation_path = Path(td) / "forecast_evaluation.json"
            with (
                mock.patch.object(track_record_tick, "DuckDBStore", return_value=store),
                mock.patch.object(track_record_tick, "PUBLIC_PATH", public_path),
                mock.patch.object(
                    track_record_tick,
                    "EVALUATION_PATH",
                    evaluation_path,
                ),
                mock.patch.object(track_record_tick, "PRICE_ONLY", True),
                mock.patch.object(track_record_tick, "MODEL", "council"),
                mock.patch.object(track_record_tick, "TRACK_MODELS", ["gpt-oss-120b", "council"]),
                mock.patch.object(
                    track_record_tick,
                    "_model_progress",
                    return_value=progress,
                ) as progress_mock,
                mock.patch.object(
                    track_record_tick.trl,
                    "resolve_open_snapshots",
                    return_value=0,
                ),
                mock.patch.object(track_record_tick.trl, "record_price_points", return_value=0),
                mock.patch.object(
                    track_record_tick.trl,
                    "record_convergence_trades",
                    return_value=0,
                ),
                mock.patch.object(
                    track_record_tick,
                    "sync_snapshot_ledger",
                    return_value={
                        "snapshots_scanned": 10,
                        "forecast_events_appended": 0,
                        "resolution_events_appended": 0,
                    },
                ),
                mock.patch.object(
                    track_record_tick,
                    "_evaluate_ledger",
                    return_value={},
                ) as evaluate_mock,
                mock.patch.object(
                    track_record_tick.trl,
                    "aggregate",
                    return_value=aggregate,
                ) as aggregate_mock,
            ):
                rc = asyncio.run(track_record_tick.main())
            self.assertTrue(evaluation_path.exists())

        self.assertEqual(rc, 0)
        self.assertEqual(
            [call.args[1] for call in progress_mock.call_args_list],
            ["council", "council"],
        )
        evaluate_mock.assert_called_once_with(store, model="council")
        aggregate_mock.assert_called_once_with(
            store,
            model="council",
            variant=track_record_tick.VARIANT,
            temperature=track_record_tick.TEMPERATURE,
        )

    def test_main_retries_snapshot_pass_when_first_pass_has_only_failures(self):
        progress_before = {
            "snapshots": 10,
            "resolved_snapshots": 4,
            "latest_snapshot_ts": "2026-07-18T21:08:23.295099+00:00",
            "latest_resolved_ts": "2026-07-18T20:00:00+00:00",
        }
        progress_after = {
            "snapshots": 11,
            "resolved_snapshots": 4,
            "latest_snapshot_ts": "2026-07-18T22:08:23.295099+00:00",
            "latest_resolved_ts": "2026-07-18T20:00:00+00:00",
        }
        aggregate = {
            "n_markets_resolved": 210,
            "n_markets_open": 39,
            "n_snapshots_resolved": 1310,
        }

        async def fake_record_snapshots(*args, **kwargs):
            call_index = fake_record_snapshots.calls
            fake_record_snapshots.calls += 1
            if call_index == 0:
                track_record_tick._predict_stats["attempts"] += 5
                track_record_tick._predict_stats["failures"] += 5
                return 0
            track_record_tick._predict_stats["attempts"] += 4
            track_record_tick._predict_stats["successes"] += 4
            return 2

        fake_record_snapshots.calls = 0

        with tempfile.TemporaryDirectory() as td:
            public_path = Path(td) / "track_record_live.json"
            evaluation_path = Path(td) / "forecast_evaluation.json"
            with (
                mock.patch.object(track_record_tick, "DuckDBStore", return_value=mock.Mock()),
                mock.patch.object(track_record_tick, "PUBLIC_PATH", public_path),
                mock.patch.object(
                    track_record_tick,
                    "EVALUATION_PATH",
                    evaluation_path,
                ),
                mock.patch.object(track_record_tick, "PRICE_ONLY", False),
                mock.patch.object(track_record_tick, "_predict_stats", Counter()),
                mock.patch.object(track_record_tick, "_SNAPSHOT_PASS_RETRIES", 2),
                mock.patch.object(track_record_tick, "_SNAPSHOT_PASS_RETRY_SLEEP_S", 0.0),
                mock.patch.object(track_record_tick, "_model_progress", side_effect=[progress_before, progress_after]),
                mock.patch.object(track_record_tick, "_get_pending_markets", return_value=[]),
                mock.patch.object(track_record_tick, "_mark_enrolled"),
                mock.patch.object(track_record_tick.trl, "resolve_open_snapshots", return_value=0),
                mock.patch.object(track_record_tick.trl, "record_price_points", return_value=0),
                mock.patch.object(track_record_tick.trl, "record_convergence_trades", return_value=0),
                mock.patch.object(
                    track_record_tick,
                    "sync_snapshot_ledger",
                    return_value={
                        "snapshots_scanned": 10,
                        "forecast_events_appended": 0,
                        "resolution_events_appended": 0,
                    },
                ),
                mock.patch.object(track_record_tick, "_evaluate_ledger", return_value={}),
                mock.patch.object(track_record_tick.trl, "aggregate", return_value=aggregate),
                mock.patch.object(track_record_tick.trl, "record_snapshots", new=fake_record_snapshots),
            ):
                rc = asyncio.run(track_record_tick.main())
        self.assertEqual(rc, 0)
        self.assertEqual(fake_record_snapshots.calls, 2)

    def test_main_warns_instead_of_failing_when_primary_model_does_not_progress(self):
        progress = {
            "snapshots": 10,
            "resolved_snapshots": 4,
            "latest_snapshot_ts": "2026-07-18T21:08:23.295099+00:00",
            "latest_resolved_ts": "2026-07-18T20:00:00+00:00",
        }
        aggregate = {
            "n_markets_resolved": 210,
            "n_markets_open": 39,
            "n_snapshots_resolved": 1310,
        }
        with tempfile.TemporaryDirectory() as td:
            public_path = Path(td) / "track_record_live.json"
            evaluation_path = Path(td) / "forecast_evaluation.json"
            with (
                mock.patch.object(track_record_tick, "DuckDBStore", return_value=mock.Mock()),
                mock.patch.object(track_record_tick, "PUBLIC_PATH", public_path),
                mock.patch.object(
                    track_record_tick,
                    "EVALUATION_PATH",
                    evaluation_path,
                ),
                mock.patch.object(track_record_tick, "PRICE_ONLY", False),
                mock.patch.object(track_record_tick, "_predict_stats", Counter({"attempts": 3, "successes": 3})),
                mock.patch.object(track_record_tick, "_model_progress", side_effect=[progress, progress]),
                mock.patch.object(track_record_tick, "_get_pending_markets", return_value=[]),
                mock.patch.object(track_record_tick, "_mark_enrolled"),
                mock.patch.object(track_record_tick.trl, "resolve_open_snapshots", return_value=0),
                mock.patch.object(track_record_tick.trl, "record_price_points", return_value=0),
                mock.patch.object(track_record_tick.trl, "record_convergence_trades", return_value=0),
                mock.patch.object(
                    track_record_tick,
                    "sync_snapshot_ledger",
                    return_value={
                        "snapshots_scanned": 10,
                        "forecast_events_appended": 0,
                        "resolution_events_appended": 0,
                    },
                ),
                mock.patch.object(track_record_tick, "_evaluate_ledger", return_value={}),
                mock.patch.object(track_record_tick.trl, "aggregate", return_value=aggregate),
                mock.patch.object(track_record_tick.trl, "record_snapshots", new=mock.AsyncMock(return_value=0)),
            ):
                rc = asyncio.run(track_record_tick.main())
        self.assertEqual(rc, 0)

    def test_main_still_fails_on_http_401(self):
        progress = {
            "snapshots": 10,
            "resolved_snapshots": 4,
            "latest_snapshot_ts": "2026-07-18T21:08:23.295099+00:00",
            "latest_resolved_ts": "2026-07-18T20:00:00+00:00",
        }
        aggregate = {
            "n_markets_resolved": 210,
            "n_markets_open": 39,
            "n_snapshots_resolved": 1310,
        }
        with tempfile.TemporaryDirectory() as td:
            public_path = Path(td) / "track_record_live.json"
            evaluation_path = Path(td) / "forecast_evaluation.json"
            with (
                mock.patch.object(track_record_tick, "DuckDBStore", return_value=mock.Mock()),
                mock.patch.object(track_record_tick, "PUBLIC_PATH", public_path),
                mock.patch.object(
                    track_record_tick,
                    "EVALUATION_PATH",
                    evaluation_path,
                ),
                mock.patch.object(track_record_tick, "PRICE_ONLY", False),
                mock.patch.object(track_record_tick, "_predict_stats", Counter({"attempts": 1, "http_401": 1})),
                mock.patch.object(track_record_tick, "_model_progress", side_effect=[progress, progress]),
                mock.patch.object(track_record_tick, "_get_pending_markets", return_value=[]),
                mock.patch.object(track_record_tick, "_mark_enrolled"),
                mock.patch.object(track_record_tick.trl, "resolve_open_snapshots", return_value=0),
                mock.patch.object(track_record_tick.trl, "record_price_points", return_value=0),
                mock.patch.object(track_record_tick.trl, "record_convergence_trades", return_value=0),
                mock.patch.object(
                    track_record_tick,
                    "sync_snapshot_ledger",
                    return_value={
                        "snapshots_scanned": 10,
                        "forecast_events_appended": 0,
                        "resolution_events_appended": 0,
                    },
                ),
                mock.patch.object(track_record_tick, "_evaluate_ledger", return_value={}),
                mock.patch.object(track_record_tick.trl, "aggregate", return_value=aggregate),
                mock.patch.object(track_record_tick.trl, "record_snapshots", new=mock.AsyncMock(return_value=0)),
            ):
                rc = asyncio.run(track_record_tick.main())
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
