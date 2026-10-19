"""Historical forecast skill may prioritize research, never authorize sizing."""

import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from analyzing_llm_rationale.twin.forecast_prior import load_forecast_research_prior
from analyzing_llm_rationale.twin.market_capture import (
    MarketCaptureBatch,
    MarketCaptureError,
    MarketCapturePolicy,
    capture_markets,
)
from analyzing_llm_rationale.twin.runtime_app import _market_capture_policy
from tests.test_twin_market_capture import kalshi_market

NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)


class ForecastPriorTests(unittest.TestCase):
    def _artifact(self, root: Path, *, generated_at: datetime, significant=True):
        path = root / "track_record_live.json"
        path.write_text(json.dumps({
            "source": "live", "generated_at": generated_at.isoformat(),
            "by_domain": [{
                "domain": "sports", "n": 120,
                "skill_significant": significant,
                "skill_ci_low": 0.004,
                "skill_ci_high": 0.016, "skill_ci_mean": 0.01,
                "n_markets": 60, "skill_ci_comparisons": 10,
                "skill_ci_method": "market_means_normal_bonferroni_v1",
                "skill_vs_market": 0.01,
            }],
        }), encoding="utf-8")
        return path

    def test_legacy_or_underpowered_evidence_cannot_prioritize_research(self):
        with TemporaryDirectory() as directory:
            for field, value in (
                ("skill_ci_method", None), ("skill_ci_method", "snapshot_normal"),
                ("n_markets", 29), ("n_markets", True), ("n_markets", 121),
                ("skill_ci_comparisons", 1), ("skill_ci_comparisons", "10"),
                ("skill_ci_mean", -0.01), ("skill_ci_high", float("nan")),
            ):
                path = self._artifact(Path(directory), generated_at=NOW - timedelta(hours=1))
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["by_domain"][0][field] = value
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(field=field, value=value):
                    self.assertIsNone(load_forecast_research_prior(path, as_of=NOW))

    def test_accepts_only_prior_positive_resolved_skill(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._artifact(root, generated_at=NOW - timedelta(hours=1))
            prior = load_forecast_research_prior(path, as_of=NOW)
            self.assertEqual(prior.validated_domains, frozenset({"sports"}))
            self.assertIsNone(load_forecast_research_prior(path, as_of=NOW - timedelta(days=1)))
            self.assertIsNone(load_forecast_research_prior(
                path, as_of=NOW - timedelta(hours=1),
            ))
            self.assertIsNone(load_forecast_research_prior(path, as_of=NOW + timedelta(days=8)))
            self._artifact(root, generated_at=NOW - timedelta(hours=1), significant=False)
            self.assertIsNone(load_forecast_research_prior(path, as_of=NOW))

    def test_malformed_rows_do_not_mask_valid_domain(self):
        with TemporaryDirectory() as directory:
            path = self._artifact(Path(directory), generated_at=NOW - timedelta(hours=1))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["by_domain"].insert(0, {
                "domain": "economics", "n": 120, "skill_significant": True,
                "skill_ci_low": None, "skill_vs_market": 0.01,
            })
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_forecast_research_prior(path, as_of=NOW).validated_domains,
                frozenset({"sports"}),
            )
            for count, low, skill in ((29, 0.01, 0.01), (30, 0, 0.01), (30, 0.01, 0)):
                payload["by_domain"][1].update(n=count, skill_ci_low=low, skill_ci_mean=skill)
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertIsNone(load_forecast_research_prior(path, as_of=NOW))

    def test_market_interval_is_not_vetoed_by_snapshot_weighting(self):
        with TemporaryDirectory() as directory:
            path = self._artifact(Path(directory), generated_at=NOW - timedelta(hours=1))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["by_domain"][0]["skill_vs_market"] = -0.2
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_forecast_research_prior(path, as_of=NOW).validated_domains,
                             frozenset({"sports"}))
            path.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
            self.assertIsNone(load_forecast_research_prior(path, as_of=NOW))
            for key, value in (
                ("source", "test"),
                ("generated_at", "2026-10-18T23:00:00"),
                ("by_domain", {}),
            ):
                valid = json.loads(self._artifact(
                    Path(directory), generated_at=NOW - timedelta(hours=1),
                ).read_text(encoding="utf-8"))
                valid[key] = value
                path.write_text(json.dumps(valid), encoding="utf-8")
                self.assertIsNone(load_forecast_research_prior(path, as_of=NOW))

    def test_runtime_policy_reads_current_bundled_report_at_decision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            static = root / "static"
            static.mkdir()
            self._artifact(static, generated_at=NOW - timedelta(hours=1))
            strategy = SimpleNamespace(max_research_candidates=2)
            self.assertIsNone(_market_capture_policy(strategy, root, as_of=NOW).research_prior)
            policy = _market_capture_policy(strategy, root, as_of=NOW, use_history=True)
            self.assertEqual(policy.research_prior.validated_domains, frozenset({"sports"}))
            self.assertEqual(policy.max_candidates, 2)
            self.assertIsNone(_market_capture_policy(
                strategy, root, as_of=NOW + timedelta(days=8), use_history=True,
            ).research_prior)

    def test_prior_reorders_only_discovered_markets_and_records_provenance(self):
        class Gateway:
            def discover(self, venue, **kwargs):
                del kwargs
                if venue == "kalshi":
                    return [
                        {"market_id": "KXPOL", "category": "politics", "question": "Will a bill pass?"},
                        {"market_id": "KXSPORT", "category": "sports", "question": "Will the NBA team win?"},
                    ]
                return []

            def fetch(self, venue, identifier):
                if venue != "kalshi":
                    raise AssertionError("unexpected venue")
                market = kalshi_market()
                market["ticker"] = identifier
                market["category"] = "sports" if identifier == "KXSPORT" else "politics"
                market["close_time"] = (NOW + timedelta(days=10)).isoformat()
                return market, {}

        with TemporaryDirectory() as directory:
            prior = load_forecast_research_prior(
                self._artifact(Path(directory), generated_at=NOW - timedelta(hours=1)),
                as_of=NOW,
            )
            policy = MarketCapturePolicy(
                max_candidates=1, candidates_per_venue=2, research_prior=prior,
            )
            batch = capture_markets(Gateway(), now=NOW, policy=policy)
            self.assertEqual(batch.markets[0].instrument.venue_instrument_id, "KXSPORT")
            self.assertEqual(batch.selection_prior_hash, prior.artifact_hash)
            self.assertEqual(MarketCaptureBatch.from_storage(batch.to_storage()), batch)
            legacy = batch.to_storage()
            legacy["schema_version"] = 2
            del legacy["selection_prior_hash"]
            self.assertIsNone(MarketCaptureBatch.from_storage(legacy).selection_prior_hash)
            legacy["schema_version"] = 1
            self.assertIsNone(MarketCaptureBatch.from_storage(legacy).selection_prior_hash)
            malformed = batch.to_storage()
            del malformed["selection_prior_hash"]
            with self.assertRaises(MarketCaptureError):
                MarketCaptureBatch.from_storage(malformed)
            for bad_hash in (["a"] * 64, 123, True, "A" * 64):
                malformed = batch.to_storage()
                malformed["selection_prior_hash"] = bad_hash
                with self.assertRaises(MarketCaptureError):
                    MarketCaptureBatch.from_storage(malformed)
            baseline = capture_markets(Gateway(), now=NOW, policy=MarketCapturePolicy(
                max_candidates=1, candidates_per_venue=2,
            ))
            self.assertEqual(baseline.markets[0].instrument.venue_instrument_id, "KXPOL")
            self.assertIsNone(baseline.selection_prior_hash)
            self.assertEqual(baseline.to_storage()["schema_version"], 2)
            for inactive in (
                replace(prior, validated_domains=frozenset({"science"})),
                replace(prior, generated_at=NOW - timedelta(days=8)),
            ):
                unchanged = capture_markets(Gateway(), now=NOW, policy=MarketCapturePolicy(
                    max_candidates=1, candidates_per_venue=2, research_prior=inactive,
                ))
                self.assertEqual(unchanged.markets[0].instrument.venue_instrument_id, "KXPOL")
                self.assertIsNone(unchanged.selection_prior_hash)


if __name__ == "__main__":
    unittest.main()
