from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "track_record_tick.py"
_SPEC = importlib.util.spec_from_file_location("track_record_tick_test_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
track_record_tick = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(track_record_tick)


class TrackRecordTickTests(unittest.TestCase):
    def test_forecast_workflow_runs_15_minute_cycles_and_uses_configured_scads_models(self):
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github" / "workflows" / "track-record-forecast.yml"
        ).read_text()
        self.assertIn('cron: "*/15 * * * *"', workflow)
        self.assertIn('CYCLE_INTERVAL_MINUTES: "15"', workflow)
        self.assertIn('SNAPSHOT_SLOT_MINUTES: "15"', workflow)
        self.assertIn('TRACK_RECORD_PREDICT_MODE: "local"', workflow)
        self.assertIn("strategy:", workflow)
        self.assertIn("continue-on-error: true", workflow)
        self.assertIn("fail-fast: false", workflow)
        self.assertIn("group: track-record-forecast", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("max-parallel: 2", workflow)
        self.assertIn("forecast (${{ matrix.model }})", workflow)
        self.assertIn("target_shard_count", workflow)
        self.assertIn("target_shard_index", workflow)
        self.assertIn("max_targets", workflow)
        self.assertIn('TRACK_MODELS: ${{ matrix.model }}', workflow)
        self.assertIn('TRACK_MODEL_SHARD_COUNT: "1"', workflow)
        self.assertIn('TRACK_MODEL_SHARD_INDEX: "0"', workflow)
        self.assertIn('TRACK_MODEL_SHARD_ALWAYS: ""', workflow)
        self.assertIn('TRACK_MODEL_SHARD_SLOT_MINUTES: "15"', workflow)
        self.assertIn("TRACK_TARGET_SHARD_COUNT:", workflow)
        self.assertIn("TRACK_TARGET_SHARD_INDEX:", workflow)
        self.assertIn('TRACK_TARGET_SHARD_SLOT_MINUTES: "15"', workflow)
        self.assertIn("TRACK_FORECAST_MAX_TARGETS:", workflow)
        self.assertIn("SCADS_AI_API_KEY: ${{ secrets.SCADS_AI_API_KEY }}", workflow)
        self.assertIn("Verify forecast secrets", workflow)
        self.assertIn("SCADS_AI_API_KEY must be configured", workflow)
        self.assertIn('pip install --quiet -e ".[serve,pipeline]"', workflow)
        self.assertIn('MAX_TOKENS: "4096"', workflow)
        self.assertIn('TRACK_RECORD_EVIDENCE_DETAIL: "summary"', workflow)
        self.assertIn('PREDICT_CONCURRENCY: "1"', workflow)
        self.assertIn('PREDICT_MIN_INTERVAL_S: "8.0"', workflow)
        self.assertIn('PREDICT_RETRIES: "2"', workflow)
        self.assertIn('PREDICT_TIMEOUT_S: "300"', workflow)
        self.assertIn('PROVIDER_MAX_RETRIES: "1"', workflow)
        self.assertIn('PROVIDER_TIMEOUT_S: "300"', workflow)
        self.assertIn('COUNCIL_MEMBER_TIMEOUT_S: "180"', workflow)
        self.assertIn('PREDICT_FAILURE_CIRCUIT_THRESHOLD: "2"', workflow)
        self.assertIn('TRACK_RECORD_SNAPSHOT_PASS_RETRIES: "1"', workflow)
        self.assertIn("python scripts/track_record_tick.py --snapshot-only", workflow)
        self.assertIn("actions/upload-artifact@v4", workflow)
        self.assertIn("track-record-store-${{ matrix.artifact }}", workflow)
        self.assertIn("publish:", workflow)
        self.assertIn("needs: forecast", workflow)
        self.assertIn("actions/download-artifact@v4", workflow)
        self.assertIn("pattern: track-record-store-*", workflow)
        self.assertIn("forecast-publish-batch", workflow)
        self.assertIn("track record: forecast batch [skip ci]", workflow)
        self.assertIn("No model forecast artifacts were available to publish.", workflow)
        self.assertIn("python scripts/merge_track_record_store.py", workflow)
        self.assertIn('"${forecast_stores[@]}"', workflow)
        self.assertIn("python scripts/track_record_tick.py --mtm-only", workflow)
        self.assertIn("git add static/mark_to_market_live.json", workflow)
        self.assertIn("pull-requests: write", workflow)
        self.assertNotIn("forecast-publish-${{ matrix.artifact }}", workflow)
        self.assertNotIn("track record: forecast ${{ matrix.model }} [skip ci]", workflow)
        self.assertNotIn("static/track_record_live.json static/forecast_evaluation.json", workflow)
        # Store persistence moved off git into GCS (the file outgrew GitHub's
        # 100MB push limit) -- these lock in the new plumbing.
        self.assertIn("google-github-actions/auth@v2", workflow)
        self.assertIn("gcloud storage cp", workflow)
        self.assertIn("track-record-store-gcs-write", workflow)
        self.assertNotIn("git add data/track_record_store.duckdb", workflow)
        for model in (
            "council",
            "scads-alias-code",
            "scads-alias-ha",
            "scads-alias-reasoning",
            "scads-alias-huge",
            "scads-alias-huge-no-thinking",
            "llama-3.1-8b-instruct",
            "llama-3.3-70b-instruct",
            "gpt-oss-120b",
            "gemma-4-31b-it",
            "minimax-m3",
            "kimi-k2.7-code",
            "kimi-k2.5",
            "kimi-k2.6",
            "llama-4-scout-17b-16e-instruct",
            "qwen3-vl-8b-instruct",
            "qwen3-coder-30b-a3b-instruct",
            "glm-5.2-fp8",
            "deepseek-v3",
            "crowd-follow",
        ):
            self.assertIn(model, track_record_tick.TRACK_MODELS)

    def test_model_sharding_rotates_models_and_keeps_always_models(self):
        models = [
            "council",
            "scads-alias-code",
            "scads-alias-ha",
            "scads-alias-reasoning",
            "gpt-oss-120b",
            "crowd-follow",
        ]

        self.assertEqual(
            track_record_tick._select_model_shard(
                models,
                count=3,
                index=0,
                always=["crowd-follow"],
            ),
            ["council", "scads-alias-reasoning", "crowd-follow"],
        )
        self.assertEqual(
            track_record_tick._select_model_shard(
                models,
                count=3,
                index=1,
                always=["crowd-follow"],
            ),
            ["scads-alias-code", "gpt-oss-120b", "crowd-follow"],
        )
        self.assertEqual(
            track_record_tick._select_model_shard(
                models,
                count=1,
                index=0,
                always=["crowd-follow"],
            ),
            models,
        )

    def test_auto_model_shard_index_uses_forecast_slot(self):
        self.assertEqual(
            track_record_tick._resolve_model_shard_index(
                count=4,
                raw_index="",
                slot_minutes=15,
                now_s=0,
            ),
            0,
        )
        self.assertEqual(
            track_record_tick._resolve_model_shard_index(
                count=4,
                raw_index=None,
                slot_minutes=15,
                now_s=45 * 60,
            ),
            3,
        )
        self.assertEqual(
            track_record_tick._resolve_model_shard_index(
                count=4,
                raw_index="2",
                slot_minutes=15,
                now_s=0,
            ),
            2,
        )

    def test_mtm_and_resolved_workflows_publish_independent_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        mtm_workflow = (
            root / ".github" / "workflows" / "track-record-tick.yml"
        ).read_text()
        resolved_workflow = (
            root / ".github" / "workflows" / "track-record-resolved.yml"
        ).read_text()

        self.assertIn('cron: "*/15 * * * *"', mtm_workflow)
        self.assertIn("track-record-mtm", mtm_workflow)
        self.assertIn("python scripts/track_record_tick.py --mtm-only", mtm_workflow)
        self.assertIn("static/mark_to_market_live.json", mtm_workflow)
        self.assertIn("pull-requests: write", mtm_workflow)
        self.assertNotIn("static/forecast_evaluation.json", mtm_workflow)

        self.assertIn('cron: "7 * * * *"', resolved_workflow)
        self.assertIn("track-record-resolved", resolved_workflow)
        self.assertIn("python scripts/track_record_tick.py --resolved-only", resolved_workflow)
        self.assertIn("static/track_record_live.json static/forecast_evaluation.json", resolved_workflow)
        self.assertIn("pull-requests: write", resolved_workflow)
        self.assertNotIn("static/mark_to_market_live.json", resolved_workflow)

    def test_artifact_publish_helper_falls_back_to_pull_request(self):
        helper = (
            Path(__file__).resolve().parents[1]
            / ".github" / "scripts" / "push_with_rebase_retry.sh"
        ).read_text()

        self.assertIn("publish_through_pr", helper)
        self.assertIn("gh pr create", helper)
        self.assertIn("gh pr merge", helper)
        self.assertIn("git push --force-with-lease", helper)

    def test_public_payloads_split_mtm_from_resolved_quality(self):
        long_curve = [
            {
                "ts": f"2026-07-29T{i:02d}:00:00+00:00",
                "account_value": 10000 + i,
                "event_type": "trade",
                "open_positions": [{"ident": "heavy", "quantity": i}],
            }
            for i in range(120)
        ]
        aggregate = {
            "generated_at": "2026-07-29T04:00:00+00:00",
            "overall": {"accuracy": 0.7},
            "models_comparison": [{"model": "council"}],
            "edge_board": [{"question": "Live edge?"}],
            "n_markets_open": 3,
            "n_markets_tracked": 8,
            "mark_to_market_account": {"account_value": 9999.5},
            "mark_to_market_by_model": [
                {"model": "council", "account": {"value_curve": long_curve}},
            ],
            "half_kelly_20pct_by_model": [
                {"model": "council", "account": {"value_curve": long_curve}},
            ],
            "growth_1pct_by_model": [{"model": "council"}],
            "growth_2pct_by_model": [{"model": "council"}],
            "mark_to_market_cycle_minutes": 15,
            "arbitrage_signals": [],
        }

        resolved = track_record_tick._resolved_public_payload(aggregate)
        mtm = track_record_tick._mark_to_market_payload(aggregate)

        self.assertEqual(resolved["overall"]["accuracy"], 0.7)
        self.assertEqual(resolved["models_comparison"][0]["model"], "council")
        self.assertNotIn("mark_to_market_account", resolved)
        self.assertNotIn("half_kelly_20pct_by_model", resolved)
        self.assertNotIn("growth_1pct_by_model", resolved)
        self.assertNotIn("edge_board", resolved)
        self.assertEqual(mtm["source"], "mark_to_market_live")
        self.assertEqual(mtm["mark_to_market_account"]["account_value"], 9999.5)
        self.assertEqual(mtm["half_kelly_20pct_by_model"][0]["model"], "council")
        half_curve = mtm["half_kelly_20pct_by_model"][0]["account"]["value_curve"]
        self.assertEqual(len(half_curve), 80)
        self.assertEqual(half_curve[0]["ts"], long_curve[0]["ts"])
        self.assertEqual(half_curve[-1]["ts"], long_curve[-1]["ts"])
        self.assertNotIn("open_positions", half_curve[0])
        self.assertEqual(mtm["growth_1pct_by_model"][0]["model"], "council")
        self.assertEqual(mtm["growth_2pct_by_model"][0]["model"], "council")
        self.assertEqual(mtm["edge_board"][0]["question"], "Live edge?")

    def test_forecast_fn_delivers_rules_and_venue_news_context(self):
        quote = {
            "question": "Will the test event happen?",
            "platform": "Kalshi",
            "ident": "TEST-26",
            "market_url": "https://kalshi.com/markets/TEST-26",
            "probability": 0.4,
            "resolution_criteria": "Resolves Yes only if the official filing is published.",
            "resolution_source": "https://example.gov/filings",
            "venue_news_articles": [
                {
                    "title": "Venue-linked filing update",
                    "url": "https://example.com/update",
                    "source": "Kalshi",
                }
            ],
        }
        response = {
            "model_key": "council",
            "market_analysis": {
                "model_probability": 0.55,
                "market_probability": 0.4,
            },
            "evidence_sources": [
                {"title": "Venue-linked filing update", "url": "https://example.com/update"},
                {"title": "Fresh external report", "url": "https://example.org/report"},
            ],
        }
        with mock.patch.object(
            track_record_tick,
            "_post_predict",
            return_value=response,
        ) as post_predict:
            result = asyncio.run(
                track_record_tick.forecast_fn(quote, 3, model="council")
            )

        payload = post_predict.call_args.args[0]
        self.assertEqual(payload["resolution_criteria"], quote["resolution_criteria"])
        self.assertEqual(payload["market_resolution_source"], quote["resolution_source"])
        self.assertEqual(payload["news_articles"], quote["venue_news_articles"])
        self.assertTrue(payload["attach_evidence"])
        self.assertEqual(payload["evidence_detail"], "summary")
        self.assertEqual(result["evidence_count"], 2)
        self.assertEqual(result["venue_news_count"], 1)
        self.assertTrue(result["rules_present"])
        self.assertTrue(result["context_complete"])

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

    def test_predict_circuit_isolates_repeated_failures_to_one_model(self):
        with (
            mock.patch.object(track_record_tick, "_predict_stats", Counter()),
            mock.patch.object(
                track_record_tick,
                "_predict_circuit_open_models",
                set(),
            ) as open_models,
            mock.patch.object(
                track_record_tick,
                "_predict_consecutive_failures",
                Counter(),
            ),
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
            kimi = {"question": "q1", "model": "kimi-k2.6"}
            council = {"question": "q2", "model": "council"}
            self.assertIsNone(track_record_tick._post_predict(kimi))
            self.assertIsNone(track_record_tick._post_predict(kimi))
            self.assertEqual(open_models, {"kimi-k2.6"})
            self.assertIsNone(track_record_tick._post_predict(kimi))
            self.assertIsNone(track_record_tick._post_predict(council))

            self.assertEqual(urlopen_mock.call_count, 3)
            self.assertEqual(track_record_tick._predict_stats["attempts"], 3)
            self.assertEqual(track_record_tick._predict_stats["failures"], 3)
            self.assertEqual(track_record_tick._predict_stats["circuit_opened"], 1)
            self.assertEqual(track_record_tick._predict_stats["circuit_skipped"], 1)
            self.assertEqual(
                track_record_tick._predict_stats["failure_model:kimi-k2.6"],
                2,
            )
            self.assertEqual(
                track_record_tick._predict_stats["failure_model:council"],
                1,
            )

    def test_local_predict_mode_uses_in_process_prediction_without_cloud_run_http(self):
        response = {
            "model_key": "council",
            "market_analysis": {
                "model_probability": 0.58,
                "market_probability": 0.5,
            },
        }
        with (
            mock.patch.object(track_record_tick, "_predict_stats", Counter()) as stats,
            mock.patch.object(track_record_tick, "PREDICT_MODE", "local"),
            mock.patch.object(track_record_tick, "_PREDICT_RETRIES", 1),
            mock.patch.object(track_record_tick, "_PREDICT_MIN_INTERVAL_S", 0.0),
            mock.patch.object(track_record_tick, "_last_predict_ts", 0.0),
            mock.patch.object(track_record_tick, "_local_predict", return_value=response) as local_mock,
            mock.patch.object(track_record_tick.urllib.request, "urlopen") as urlopen_mock,
        ):
            result = track_record_tick._post_predict(
                {"question": "Will the test event happen?", "model": "council"}
            )

        self.assertEqual(result, response)
        local_mock.assert_called_once()
        urlopen_mock.assert_not_called()
        self.assertEqual(stats["attempts"], 1)
        self.assertEqual(stats["successes"], 1)
        self.assertEqual(stats["success_model:council"], 1)

    def test_local_predict_retries_transient_http_exception_then_succeeds(self):
        class LocalHTTPError(Exception):
            status_code = 503
            detail = "temporarily unavailable"
            headers = {"Retry-After": "7"}

        response = {
            "model_key": "council",
            "market_analysis": {"model_probability": 0.58},
        }
        with (
            mock.patch.object(track_record_tick, "_predict_stats", Counter()) as stats,
            mock.patch.object(track_record_tick, "PREDICT_MODE", "local"),
            mock.patch.object(track_record_tick, "_PREDICT_RETRIES", 2),
            mock.patch.object(track_record_tick, "_PREDICT_MIN_INTERVAL_S", 0.0),
            mock.patch.object(track_record_tick, "_PREDICT_RETRY_JITTER_FRACTION", 0.0),
            mock.patch.object(track_record_tick, "_last_predict_ts", 0.0),
            mock.patch.object(
                track_record_tick,
                "_local_predict",
                side_effect=[LocalHTTPError(), response],
            ) as local_mock,
            mock.patch.object(track_record_tick.time, "sleep") as sleep_mock,
        ):
            result = track_record_tick._post_predict(
                {"question": "Will the test event happen?", "model": "council"}
            )

        self.assertEqual(result, response)
        self.assertEqual(local_mock.call_count, 2)
        sleep_mock.assert_called_once_with(7.0)
        self.assertEqual(stats["http_503"], 1)
        self.assertEqual(stats["retry_sleeps"], 1)
        self.assertEqual(stats["successes"], 1)

    def test_local_predict_http_exception_is_counted_by_status_code(self):
        class LocalHTTPError(Exception):
            status_code = 503
            detail = "Alternate models are not configured on this runner."

        with (
            mock.patch.object(track_record_tick, "_predict_stats", Counter()) as stats,
            mock.patch.object(track_record_tick, "PREDICT_MODE", "local"),
            mock.patch.object(track_record_tick, "_PREDICT_RETRIES", 1),
            mock.patch.object(track_record_tick, "_PREDICT_MIN_INTERVAL_S", 0.0),
            mock.patch.object(track_record_tick, "_last_predict_ts", 0.0),
            mock.patch.object(track_record_tick, "_local_predict", side_effect=LocalHTTPError()),
        ):
            result = track_record_tick._post_predict(
                {"question": "Will the test event happen?", "model": "council"}
            )

        self.assertIsNone(result)
        self.assertEqual(stats["http_503"], 1)
        self.assertEqual(stats["failures"], 1)
        self.assertEqual(stats["failure_model:council"], 1)

    def test_main_fails_fast_when_local_predict_secret_is_missing(self):
        with (
            mock.patch.object(track_record_tick, "PRICE_ONLY", False),
            mock.patch.object(track_record_tick, "MARK_TO_MARKET_ONLY", False),
            mock.patch.object(track_record_tick, "RESOLVED_ONLY", False),
            mock.patch.object(track_record_tick, "SNAPSHOT_ONLY", False),
            mock.patch.object(track_record_tick, "PREDICT_MODE", "local"),
            mock.patch.object(
                track_record_tick,
                "_local_predict_credentials_available",
                return_value=False,
            ),
            mock.patch.object(track_record_tick, "DuckDBStore") as store_mock,
        ):
            rc = asyncio.run(track_record_tick.main())

        self.assertEqual(rc, 1)
        store_mock.assert_not_called()

    def test_http_error_detail_is_logged_and_attributed_to_model(self):
        error = urllib.error.HTTPError(
            url="https://foresea.ink/predict",
            code=502,
            msg="Bad Gateway",
            hdrs=None,
            fp=mock.Mock(
                read=mock.Mock(
                    return_value=b'{"detail":"council quorum unavailable"}'
                )
            ),
        )
        with (
            mock.patch.object(track_record_tick, "_predict_stats", Counter()) as stats,
            mock.patch.object(
                track_record_tick,
                "_predict_circuit_open_models",
                set(),
            ),
            mock.patch.object(
                track_record_tick,
                "_predict_consecutive_failures",
                Counter(),
            ),
            mock.patch.object(track_record_tick, "_PREDICT_RETRIES", 1),
            mock.patch.object(track_record_tick, "_PREDICT_MIN_INTERVAL_S", 0.0),
            mock.patch.object(track_record_tick, "_last_predict_ts", 0.0),
            mock.patch.object(
                track_record_tick.urllib.request,
                "urlopen",
                side_effect=error,
            ),
            mock.patch("builtins.print") as print_mock,
        ):
            result = track_record_tick._post_predict(
                {"question": "q", "model": "council"}
            )

        self.assertIsNone(result)
        self.assertEqual(stats["failure_model:council"], 1)
        self.assertTrue(
            any(
                "council quorum unavailable" in str(call)
                and "model=council" in str(call)
                for call in print_mock.call_args_list
            )
        )

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
            mtm_path = Path(td) / "mark_to_market_live.json"
            evaluation_path = Path(td) / "forecast_evaluation.json"
            with (
                mock.patch.object(track_record_tick, "DuckDBStore", return_value=store),
                mock.patch.multiple(
                    track_record_tick,
                    PUBLIC_PATH=public_path,
                    MARK_TO_MARKET_PATH=mtm_path,
                    EVALUATION_PATH=evaluation_path,
                    PRICE_ONLY=True,
                    MARK_TO_MARKET_ONLY=False,
                    RESOLVED_ONLY=False,
                    SNAPSHOT_ONLY=False,
                    MODEL="council",
                    TRACK_MODELS=["gpt-oss-120b", "council"],
                ),
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
            cycle_minutes=track_record_tick.CYCLE_INTERVAL_MINUTES,
            tracked_models=track_record_tick.ALL_TRACK_MODELS,
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
            mtm_path = Path(td) / "mark_to_market_live.json"
            evaluation_path = Path(td) / "forecast_evaluation.json"
            with (
                mock.patch.object(track_record_tick, "DuckDBStore", return_value=mock.Mock()),
                mock.patch.multiple(
                    track_record_tick,
                    PUBLIC_PATH=public_path,
                    MARK_TO_MARKET_PATH=mtm_path,
                    EVALUATION_PATH=evaluation_path,
                    PRICE_ONLY=False,
                    MARK_TO_MARKET_ONLY=False,
                    RESOLVED_ONLY=False,
                    SNAPSHOT_ONLY=False,
                ),
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

    def test_snapshot_pass_uses_configured_target_shard_bounds(self):
        async_mock = mock.AsyncMock(return_value=1)
        with (
            mock.patch.object(track_record_tick.trl, "record_snapshots", async_mock),
            mock.patch.object(track_record_tick, "_SNAPSHOT_PASS_RETRIES", 1),
            mock.patch.object(track_record_tick, "_predict_stats", Counter({"attempts": 1, "successes": 1})),
            mock.patch.multiple(
                track_record_tick,
                TRACK_TARGET_SHARD_COUNT=4,
                TRACK_TARGET_SHARD_INDEX=2,
                TRACK_FORECAST_MAX_TARGETS=15,
            ),
        ):
            recorded = asyncio.run(
                track_record_tick._record_snapshots_with_retries(
                    mock.Mock(),
                    seeds=[],
                )
            )

        self.assertEqual(recorded, 1)
        self.assertEqual(async_mock.call_args.kwargs["target_shard_count"], 4)
        self.assertEqual(async_mock.call_args.kwargs["target_shard_index"], 2)
        self.assertEqual(async_mock.call_args.kwargs["max_targets"], 15)

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
            mtm_path = Path(td) / "mark_to_market_live.json"
            evaluation_path = Path(td) / "forecast_evaluation.json"
            with (
                mock.patch.object(track_record_tick, "DuckDBStore", return_value=mock.Mock()),
                mock.patch.multiple(
                    track_record_tick,
                    PUBLIC_PATH=public_path,
                    MARK_TO_MARKET_PATH=mtm_path,
                    EVALUATION_PATH=evaluation_path,
                    PRICE_ONLY=False,
                    MARK_TO_MARKET_ONLY=False,
                    RESOLVED_ONLY=False,
                    SNAPSHOT_ONLY=False,
                ),
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
            mtm_path = Path(td) / "mark_to_market_live.json"
            evaluation_path = Path(td) / "forecast_evaluation.json"
            with (
                mock.patch.object(track_record_tick, "DuckDBStore", return_value=mock.Mock()),
                mock.patch.multiple(
                    track_record_tick,
                    PUBLIC_PATH=public_path,
                    MARK_TO_MARKET_PATH=mtm_path,
                    EVALUATION_PATH=evaluation_path,
                    PRICE_ONLY=False,
                    MARK_TO_MARKET_ONLY=False,
                    RESOLVED_ONLY=False,
                    SNAPSHOT_ONLY=False,
                ),
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
