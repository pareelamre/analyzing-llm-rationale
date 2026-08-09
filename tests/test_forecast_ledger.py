from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.forecast_ledger import (
    ForecastLedger,
    ImmutableEventConflict,
    LedgerValidationError,
    sync_snapshot_ledger,
)
from analyzing_llm_rationale.trackrec_store import DuckDBStore, Entity


class ForecastLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = DuckDBStore(Path(self.temp_dir.name) / "ledger.duckdb")
        self.forecasted_at = datetime(2026, 1, 27, 10, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.store.close()
        self.temp_dir.cleanup()

    def _snapshot(
        self,
        *,
        probability: float = 0.7,
        snapshot_ts: datetime | None = None,
        resolved: bool = True,
        outcome: int = 1,
    ) -> Entity:
        snapshot = Entity(
            self.store.key("ForecastSnapshot", "kalshi:RATE-26:council:2026-07-27")
        )
        snapshot.update(
            platform="Kalshi",
            ident="RATE-26",
            model="council",
            question="Will the rate be cut?",
            snapshot_ts=snapshot_ts or self.forecasted_at,
            close_time=self.forecasted_at + timedelta(days=30),
            model_probability=probability,
            market_probability=0.5,
            market_bid=0.49,
            market_ask=0.51,
            domain="economics",
            horizon="14-30d",
            resolved=resolved,
            outcome=outcome if resolved else None,
            resolved_ts=(
                self.forecasted_at + timedelta(days=31) if resolved else None
            ),
        )
        return snapshot

    def test_sync_is_idempotent_and_produces_resolved_pairs(self):
        self.store.put(self._snapshot())

        first = sync_snapshot_ledger(self.store)
        second = sync_snapshot_ledger(self.store)
        resolved = ForecastLedger(self.store).resolved_forecasts()

        self.assertEqual(first["forecast_events_appended"], 1)
        self.assertEqual(first["resolution_events_appended"], 1)
        self.assertEqual(second["forecast_events_appended"], 0)
        self.assertEqual(second["resolution_events_appended"], 0)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["outcome"], 1)
        self.assertEqual(resolved[0]["domain"], "economics")
        self.assertIsNotNone(resolved[0]["ledger_ingested_at"])
        self.assertFalse(resolved[0]["ledger_audit_grade"])

    def test_sync_hydrates_question_domain_close_time_for_normalized_rows(self):
        # A snapshot written after market-level fields stopped being
        # duplicated per row -- question/domain/close_time are NULL on the
        # row itself, only present in `markets`. The ledger event is
        # immutable once written, so getting this right matters: a naive
        # blanket join would risk crystallizing values back onto the row on
        # a later put(), but the ledger only ever reads snapshots, never
        # writes them back -- confirm the sync still produces a real event,
        # not "" / "other" / "unknown" / a broken post_close signal.
        market = Entity(self.store.key("Market", "Kalshi:RATE-26"))
        market.update(platform="Kalshi", ident="RATE-26", question="Will the rate be cut?",
                       domain="economics", close_time=self.forecasted_at + timedelta(days=30))
        self.store.put(market)

        snapshot = Entity(self.store.key("ForecastSnapshot", "kalshi:RATE-26:council:2026-07-27"))
        snapshot.update(
            platform="Kalshi", ident="RATE-26", model="council",
            snapshot_ts=self.forecasted_at,
            model_probability=0.7, market_probability=0.5,
            market_bid=0.49, market_ask=0.51,
            horizon="14-30d",
            resolved=True, outcome=1,
            resolved_ts=self.forecasted_at + timedelta(days=31),
        )
        self.store.put(snapshot)

        result = sync_snapshot_ledger(self.store)
        resolved = ForecastLedger(self.store).resolved_forecasts()

        self.assertEqual(result["forecast_events_appended"], 1)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["question"], "Will the rate be cut?")
        self.assertEqual(resolved[0]["domain"], "economics")
        self.assertIsNotNone(resolved[0]["close_time"])
        # forecasted well before close_time -> post_close detection intact
        # (this is the exact signal that silently breaks -- always False --
        # if close_time isn't hydrated and comes through as None).
        self.assertTrue(resolved[0]["ledger_forecast_before_close"])

    def test_replaced_snapshot_preserves_each_forecast_revision(self):
        first = self._snapshot(resolved=False)
        self.store.put(first)
        sync_snapshot_ledger(self.store)

        revised = self._snapshot(
            probability=0.62,
            snapshot_ts=self.forecasted_at + timedelta(hours=1),
            resolved=False,
        )
        self.store.put(revised)
        sync_snapshot_ledger(self.store)

        revised["resolved"] = True
        revised["outcome"] = 0
        revised["resolved_ts"] = self.forecasted_at + timedelta(days=31)
        self.store.put(revised)
        result = sync_snapshot_ledger(self.store)
        resolved = ForecastLedger(self.store).resolved_forecasts()

        self.assertEqual(result["forecast_events_appended"], 0)
        self.assertEqual(result["resolution_events_appended"], 1)
        self.assertEqual(len(resolved), 2)
        self.assertEqual(
            {row["model_probability"] for row in resolved},
            {0.7, 0.62},
        )
        self.assertTrue(all(row["outcome"] == 0 for row in resolved))

    def test_resolution_cannot_be_rewritten(self):
        ledger = ForecastLedger(self.store)
        resolved_at = self.forecasted_at + timedelta(days=1)
        self.assertTrue(
            ledger.record_resolution(
                platform="Kalshi",
                ident="RATE-26",
                outcome=1,
                resolved_at=resolved_at,
            )
        )
        self.assertFalse(
            ledger.record_resolution(
                platform="kalshi",
                ident="RATE-26",
                outcome=1,
                resolved_at=resolved_at + timedelta(hours=1),
            )
        )
        with self.assertRaises(ImmutableEventConflict):
            ledger.record_resolution(
                platform="Kalshi",
                ident="RATE-26",
                outcome=0,
                resolved_at=resolved_at,
            )

    def test_promptly_ingested_forecast_is_in_prospective_audit_cohort(self):
        snapshot = self._snapshot(resolved=False)
        self.store.put(snapshot)
        with mock.patch(
            "analyzing_llm_rationale.forecast_ledger._now_utc",
            return_value=self.forecasted_at + timedelta(hours=1),
        ):
            sync_snapshot_ledger(self.store)

        snapshot["resolved"] = True
        snapshot["outcome"] = 1
        snapshot["resolved_ts"] = self.forecasted_at + timedelta(days=31)
        self.store.put(snapshot)
        with mock.patch(
            "analyzing_llm_rationale.forecast_ledger._now_utc",
            return_value=self.forecasted_at + timedelta(days=31, hours=1),
        ):
            sync_snapshot_ledger(self.store)

        resolved = ForecastLedger(self.store).resolved_forecasts()
        self.assertEqual(len(resolved), 1)
        self.assertTrue(resolved[0]["ledger_audit_grade"])
        self.assertTrue(resolved[0]["ledger_forecast_before_close"])
        self.assertEqual(resolved[0]["ledger_audit_delay_seconds"], 3600.0)

    def test_post_close_forecast_is_not_in_prospective_audit_cohort(self):
        snapshot = self._snapshot(
            snapshot_ts=self.forecasted_at + timedelta(days=30, hours=1),
            resolved=False,
        )
        self.store.put(snapshot)
        with mock.patch(
            "analyzing_llm_rationale.forecast_ledger._now_utc",
            return_value=self.forecasted_at + timedelta(days=30, hours=2),
        ):
            sync_snapshot_ledger(self.store)
        event = next(self.store.query(kind="ForecastLedgerEvent").fetch())
        self.assertEqual(event["ingest_state"], "post_close")

        snapshot["resolved"] = True
        snapshot["outcome"] = 1
        snapshot["resolved_ts"] = self.forecasted_at + timedelta(days=31)
        self.store.put(snapshot)
        with mock.patch(
            "analyzing_llm_rationale.forecast_ledger._now_utc",
            return_value=self.forecasted_at + timedelta(days=31, hours=1),
        ):
            sync_snapshot_ledger(self.store)

        resolved = ForecastLedger(self.store).resolved_forecasts()
        self.assertEqual(len(resolved), 1)
        self.assertFalse(resolved[0]["ledger_forecast_before_close"])
        self.assertFalse(resolved[0]["ledger_audit_grade"])

    def test_forecast_rejects_evidence_from_the_future(self):
        snapshot = self._snapshot(resolved=False)
        snapshot["evidence_as_of"] = self.forecasted_at + timedelta(seconds=1)

        with self.assertRaises(LedgerValidationError):
            ForecastLedger(self.store).record_forecast(
                snapshot,
                snapshot_key=str(snapshot.key.id),
            )
        self.assertEqual(self.store.count("ForecastLedgerEvent"), 0)


if __name__ == "__main__":
    unittest.main()
