from __future__ import annotations

import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from analyzing_llm_rationale.twin.replay import (
    ReplayValidationError,
    causal_events,
    split_replay_dataset,
)

NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
SPLIT = NOW - timedelta(days=10)


def row(
    suffix: str, *, forecast_at: datetime, outcome_at: datetime,
    cluster: str | None = None, outcome: int = 1,
) -> dict:
    return {
        "forecast_id": f"forecast-{suffix}", "event_cluster_id": cluster or f"cluster-{suffix}",
        "instrument_id": f"kalshi:demo:{suffix}", "platform": "kalshi", "model": "council",
        "forecast_occurred_at": forecast_at.isoformat(),
        "forecast_observed_at": (forecast_at + timedelta(minutes=1)).isoformat(),
        "resolved_at": (outcome_at - timedelta(minutes=1)).isoformat(),
        "outcome_observed_at": outcome_at.isoformat(),
        "model_probability": .75 if outcome else .25, "market_probability": .5,
        "market_bid": .49, "market_ask": .51, "outcome": outcome, "domain": "politics",
    }


def dataset(records: list[dict]) -> dict:
    return {"schema_version": 1, "captured_at": NOW.isoformat(), "records": records}


class TwinReplayTests(unittest.TestCase):
    def test_late_evidence_never_changes_earlier_replay(self):
        rows = [{"id": "first", "observed_at": NOW}, {"id": "late", "observed_at": NOW + timedelta(seconds=1)}, {"id": "first", "observed_at": NOW}]
        self.assertEqual([item["id"] for item in causal_events(rows, as_of=NOW)], ["first"])

    def test_future_occurrence_and_conflicting_duplicate_are_not_replayed(self):
        self.assertEqual(causal_events([{"id": "future", "observed_at": NOW, "occurred_at": NOW + timedelta(seconds=1)}], as_of=NOW), [])
        with self.assertRaises(ReplayValidationError):
            causal_events([
                {"id": "same", "observed_at": NOW, "value": 1},
                {"id": "same", "observed_at": NOW, "value": 2},
            ], as_of=NOW)

    def test_split_is_time_and_event_disjoint_and_deduplicated(self):
        training_at = SPLIT - timedelta(days=3)
        records = [
            row("train", forecast_at=training_at, outcome_at=SPLIT - timedelta(days=1), cluster="shared"),
            row("train-duplicate", forecast_at=training_at + timedelta(hours=1), outcome_at=SPLIT - timedelta(hours=12), cluster="shared"),
            row("test-shared", forecast_at=SPLIT + timedelta(days=1), outcome_at=NOW - timedelta(days=1), cluster="shared"),
            row("test", forecast_at=SPLIT + timedelta(days=2), outcome_at=NOW - timedelta(days=1)),
        ]
        frozen = split_replay_dataset(dataset(records), split_at=SPLIT, evaluation_as_of=NOW)
        self.assertEqual([item.forecast_id for item in frozen.calibration], ["forecast-train"])
        self.assertEqual([item.forecast_id for item in frozen.test], ["forecast-test"])
        self.assertEqual(frozen.excluded["duplicate_cluster"], 1)
        self.assertEqual(frozen.excluded["training_cluster"], 1)

    def test_future_and_leaked_outcomes_are_excluded(self):
        leaked = row("leaked", forecast_at=SPLIT + timedelta(days=1), outcome_at=NOW - timedelta(days=1))
        leaked["resolved_at"] = leaked["forecast_occurred_at"]
        leaked["outcome_observed_at"] = leaked["forecast_observed_at"]
        future = row("future", forecast_at=SPLIT + timedelta(days=1), outcome_at=NOW + timedelta(days=1))
        frozen = split_replay_dataset(dataset([leaked, future]), split_at=SPLIT, evaluation_as_of=NOW)
        self.assertEqual(frozen.test, ())
        self.assertEqual(frozen.excluded["leaked_outcome"], 1)
        self.assertEqual(frozen.excluded["future_outcome"], 1)

    def test_dataset_capture_time_is_the_availability_boundary(self):
        captured = NOW - timedelta(days=2)
        unavailable = row(
            "after-capture", forecast_at=SPLIT + timedelta(days=1),
            outcome_at=NOW - timedelta(days=1),
        )
        document = dataset([unavailable])
        document["captured_at"] = captured.isoformat()
        frozen = split_replay_dataset(document, split_at=SPLIT, evaluation_as_of=NOW)
        self.assertEqual(frozen.test, ())
        self.assertEqual(frozen.excluded["future_outcome"], 1)

    def test_conflicting_capture_and_future_dataset_fail_closed(self):
        first = row("same", forecast_at=SPLIT + timedelta(days=1), outcome_at=NOW - timedelta(days=1))
        conflict = deepcopy(first)
        conflict["model_probability"] = .1
        with self.assertRaisesRegex(ReplayValidationError, "conflicting replay forecast ID"):
            split_replay_dataset(dataset([first, conflict]), split_at=SPLIT, evaluation_as_of=NOW)
        future_dataset = dataset([])
        future_dataset["captured_at"] = (NOW + timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ReplayValidationError, "captured after"):
            split_replay_dataset(future_dataset, split_at=SPLIT, evaluation_as_of=NOW)


if __name__ == "__main__":
    unittest.main()
