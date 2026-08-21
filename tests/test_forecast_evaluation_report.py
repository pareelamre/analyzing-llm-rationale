from __future__ import annotations

import math
import sys
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.forecast_evaluation import ResolvedForecast
from analyzing_llm_rationale.forecast_evaluation_report import (
    EvaluationArtifactValidationError,
    EvaluationPolicy,
    build_evaluation_artifact,
    compact_evaluation_summary,
    validate_evaluation_artifact,
)

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _forecasts(*, n: int, skilled: bool) -> list[ResolvedForecast]:
    rows = []
    for index in range(n):
        outcome = index % 2
        probability = (0.9 if outcome else 0.1) if skilled else (0.1 if outcome else 0.9)
        forecasted_at = BASE_TIME + timedelta(days=index * 2)
        rows.append(
            ResolvedForecast(
                forecast_id=f"forecast-{index}",
                platform="kalshi",
                market_id=f"market-{index}",
                model="council",
                forecasted_at=forecasted_at,
                resolved_at=forecasted_at + timedelta(days=1),
                model_probability=probability,
                market_probability=0.5,
                market_bid=0.5,
                market_ask=0.5,
                outcome=outcome,
                domain="economics" if index % 2 else "politics",
            )
        )
    return rows


class ForecastEvaluationReportTests(unittest.TestCase):
    def test_historical_mirror_can_never_satisfy_promotion(self):
        artifact = build_evaluation_artifact(
            model="council",
            snapshot_mirror=_forecasts(n=100, skilled=True),
            prospective_audit=[],
            generated_at=BASE_TIME,
        )

        self.assertFalse(
            artifact["cohorts"]["snapshot_mirror"]["promotion_eligible_source"]
        )
        self.assertEqual(artifact["promotion"]["status"], "collecting")
        self.assertFalse(artifact["promotion"]["eligible"])
        self.assertEqual(
            artifact["promotion"]["checks"]["minimum_resolved_markets"]["actual"],
            0,
        )

    def test_skilled_prospective_cohort_satisfies_every_gate(self):
        policy = EvaluationPolicy(
            min_resolved_markets=100,
            min_paper_trades=30,
        )
        prospective = _forecasts(n=100, skilled=True)

        artifact = build_evaluation_artifact(
            model="council",
            snapshot_mirror=[],
            prospective_audit=prospective,
            generated_at=BASE_TIME,
            policy=policy,
        )

        self.assertTrue(artifact["promotion"]["eligible"])
        self.assertEqual(
            artifact["promotion"]["status"],
            "eligible_for_shadow_promotion",
        )
        self.assertTrue(
            all(
                check["passed"]
                for check in artifact["promotion"]["checks"].values()
            )
        )
        self.assertGreater(
            artifact["cohorts"]["prospective_audit"]["portfolio"][
                "compound_return"
            ],
            0.0,
        )

    def test_bad_prospective_cohort_is_not_qualified_after_collection(self):
        artifact = build_evaluation_artifact(
            model="council",
            snapshot_mirror=[],
            prospective_audit=_forecasts(n=100, skilled=False),
            generated_at=BASE_TIME,
        )

        self.assertFalse(artifact["promotion"]["eligible"])
        self.assertEqual(artifact["promotion"]["status"], "not_qualified")
        self.assertFalse(
            artifact["promotion"]["checks"]["positive_skill_lower_bound"][
                "passed"
            ]
        )
        self.assertFalse(
            artifact["promotion"]["checks"][
                "positive_compound_return_after_fees"
            ]["passed"]
        )

    def test_compact_summary_omits_curves_and_buckets(self):
        artifact = build_evaluation_artifact(
            model="council",
            snapshot_mirror=_forecasts(n=2, skilled=True),
            prospective_audit=[],
            generated_at=BASE_TIME,
        )

        summary = compact_evaluation_summary(artifact)

        self.assertIn("snapshot_mirror", summary)
        self.assertNotIn("portfolio", summary["snapshot_mirror"])
        self.assertNotIn("calibration", summary["snapshot_mirror"])

    def test_validator_accepts_a_generated_artifact(self):
        artifact = build_evaluation_artifact(
            model="council",
            snapshot_mirror=_forecasts(n=2, skilled=True),
            prospective_audit=[],
            generated_at=BASE_TIME,
        )

        validate_evaluation_artifact(artifact)

    def test_validator_rejects_tampered_promotion_claim(self):
        artifact = build_evaluation_artifact(
            model="council",
            snapshot_mirror=_forecasts(n=2, skilled=True),
            prospective_audit=[],
            generated_at=BASE_TIME,
        )
        tampered = deepcopy(artifact)
        tampered["promotion"]["status"] = "eligible_for_shadow_promotion"

        with self.assertRaisesRegex(
            EvaluationArtifactValidationError,
            "promotion must exactly match prospective_audit and policy",
        ):
            validate_evaluation_artifact(tampered)

    def test_validator_reports_a_malformed_cohort_without_crashing(self):
        artifact = build_evaluation_artifact(
            model="council",
            snapshot_mirror=[],
            prospective_audit=[],
            generated_at=BASE_TIME,
        )
        del artifact["cohorts"]["prospective_audit"]["portfolio"]

        with self.assertRaisesRegex(
            EvaluationArtifactValidationError,
            "prospective_audit cannot be used to recompute promotion",
        ):
            validate_evaluation_artifact(artifact)

    def test_policy_rejects_non_finite_skill_threshold(self):
        with self.assertRaisesRegex(ValueError, "min_skill_lower_bound must be finite"):
            EvaluationPolicy(min_skill_lower_bound=math.nan)


if __name__ == "__main__":
    unittest.main()
