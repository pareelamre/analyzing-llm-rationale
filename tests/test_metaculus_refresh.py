"""Behavior tests for periodic, evidence-backed revisions (synthetic data only)."""
import json
import os
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from analyzing_llm_rationale.metaculus_audit_storage import restore_audit
from analyzing_llm_rationale.metaculus_bot import (
    ForecastCycleConfig,
    MetaculusError,
    SubmissionOutcomeUnknownError,
    _question_prompt,
    _revision_delta,
    _submission_payload_matches,
    run_forecast_cycle,
)
from test_metaculus_audit_storage import _Blob
from test_metaculus_bot import FakeProvider


class Client:
    def __init__(self, *, recent=False, unanswered=False):
        start = datetime.now(timezone.utc).timestamp() if recent else 1.0
        self.posts = [{"id": 12, "status": "open", "question": {
            "id": 44, "type": "binary", "my_forecasts": {"latest": {
                "author_id": 99, "probability_yes": 0.7, "start_time": start,
            }},
        }}]
        if unanswered:
            self.posts.append({"id": 13, "status": "open", "question": {
                "id": 45, "type": "binary", "my_forecasts": {"latest": None},
            }})
        self.submissions = []
        self.comments = []
        self.verifications = []

    def list_open_posts(self, *args, **kwargs):
        return [{"id": p["id"]} for p in self.posts] if kwargs.get("offset", 0) == 0 else []

    def get_post(self, post_id):
        return deepcopy(next(p for p in self.posts if p["id"] == post_id))

    def submit_forecast(self, question_id, payload):
        self.submissions.append((question_id, payload))

    def verify_submission(self, *args, **kwargs):
        self.verifications.append(kwargs)

    def submit_private_comment(self, post_id, text, **kwargs):
        self.comments.append((post_id, text))


class RefreshTests(unittest.TestCase):
    def test_default_hourly_refresh_preserves_cooldown_and_reassesses_due_forecast(self):
        self.assertEqual(ForecastCycleConfig().refresh_interval_s, 3600)
        for minutes, expected in ((30, 0), (59, 0), (61, 1), (90, 1)):
            with self.subTest(age_minutes=minutes), TemporaryDirectory() as directory:
                client, provider = Client(), FakeProvider('{"probability_yes":0.8}')
                client.posts[0]["question"]["my_forecasts"]["latest"]["start_time"] = (
                    datetime.now(timezone.utc).timestamp() - minutes * 60)
                summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl")
                self.assertEqual(summary.submitted, expected)
                self.assertEqual(bool(provider.calls), bool(expected))
                self.assertEqual(summary.skipped, 1 - expected)

    def run_cycle(self, client, provider, path, **kwargs):
        return run_forecast_cycle(
            client, provider,
            ForecastCycleConfig(max_questions=1, submit=True,
                                refresh_forecasted=True, audit_log_path=path),
            expected_author_id=99, bot_username="synthetic-bot", **kwargs,
        )

    def test_recent_forecast_is_not_researched_or_reposted(self):
        client, provider = Client(recent=True), FakeProvider('{"probability_yes":0.8}')
        research_calls = []
        with TemporaryDirectory() as directory:
            summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl",
                                     research_provider=lambda p: research_calls.append(p) or [])
        self.assertEqual(provider.calls, 0)
        self.assertEqual(research_calls, [])
        self.assertEqual(client.submissions, [])
        self.assertEqual(summary.failed, 0)

    def test_changed_revision_researches_and_verifies_new_forecast_and_comment(self):
        client, provider = Client(), FakeProvider('{"probability_yes":0.8}')
        evidence = [{"title": "New verified observation", "summary": "A qualifying event was reported."}]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            summary = self.run_cycle(client, provider, path, research_provider=lambda p: evidence)
            events = [json.loads(s) for s in path.read_text().splitlines()]
        self.assertEqual(summary.submitted, 1)
        self.assertEqual(client.submissions[0][1]["probability_yes"], 0.8)
        self.assertEqual(len(client.comments), 1)
        self.assertEqual(client.verifications[0]["previous_forecast_start_time"], 1.0)
        self.assertEqual([e["outcome"] for e in events],
                         ["prepared", "forecast_verified_comment_pending", "submission_verified"])
        self.assertEqual(events[-1]["evidence_count"], 1)
        self.assertAlmostEqual(events[-1]["forecast_provenance"]["revision_probability_delta"], 0.1)
        self.assertIn("New verified observation", provider.messages[0][1]["content"])
        self.assertIn("previous_forecast_start_time", provider.messages[0][1]["content"])

    def test_closed_or_expired_question_is_not_revised(self):
        for modification in ({"status": "closed"}, {"scheduled_close_time": "2020-01-01T00:00:00Z"}):
            with self.subTest(modification=modification), TemporaryDirectory() as directory:
                client, provider = Client(), FakeProvider('{"probability_yes":0.8}')
                client.posts[0].update(modification)
                self.run_cycle(client, provider, Path(directory) / "audit.jsonl")
                self.assertEqual(provider.calls, 0)
                self.assertEqual(client.submissions, [])

    def test_ambiguous_prior_submission_still_halts_revisions(self):
        client, provider = Client(), FakeProvider('{"probability_yes":0.8}')
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            with path.open("w") as handle:
                handle.write(json.dumps({"question_id": 44, "outcome": "submission_unknown"}) + "\n")
            with self.assertLogs("analyzing_llm_rationale.metaculus_bot", "ERROR"):
                with self.assertRaises(SubmissionOutcomeUnknownError):
                    self.run_cycle(client, provider, path)
        self.assertEqual(provider.calls, 0)

    def test_probability_distance_handles_accepted_forecast_shapes(self):
        cases = [
            ({"type": "binary"}, {"probability_yes": 0.8}, {"forecast_values": [0.7]}, 0.1),
            ({"type": "binary"}, {"probability_yes": 0.8}, {"forecast_values": [0.3, 0.7]}, 0.1),
            ({"type": "multiple_choice", "options": ["A", "B"]},
             {"probability_yes_per_category": {"A": 0.4, "B": 0.6}}, {"forecast_values": [0.3, 0.7]}, 0.1),
            ({"type": "multiple_choice", "options": ["A", "B"]},
             {"probability_yes_per_category": {"A": 0.4, "B": 0.6}},
             {"probability_yes_per_category": {"B": 0.7, "A": 0.3}}, 0.1),
            ({"type": "numeric"}, {"continuous_cdf": [0, 0.5, 1]}, {"forecast_values": [0, 0.3, 1]}, 0.2),
            ({"type": "discrete"}, {"continuous_cdf": [0, 0.5, 1]}, {"continuous_cdf": [0, 0.3, 1]}, 0.2),
        ]
        for question, payload, latest, expected in cases:
            with self.subTest(question=question):
                self.assertAlmostEqual(_revision_delta(question, payload, latest), expected)
        for prior in ([float("nan")], [True], [], [0.1, 0.2]):
            with self.subTest(prior=prior), self.assertRaises(MetaculusError):
                _revision_delta({"type": "binary"}, {"probability_yes": 0.8}, {"forecast_values": prior})

    def test_prior_forecast_is_context_not_evidence_of_increasing_certainty(self):
        prompt = _question_prompt(Client().posts[0], Client().posts[0]["question"])
        self.assertIn("Elapsed time alone does not justify increasing confidence", prompt)

    def test_missing_or_wrong_account_prior_fails_without_publication(self):
        for change in ({"start_time": None}, {"author_id": 1}):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                client, provider = Client(), FakeProvider('{"probability_yes":0.8}')
                client.posts[0]["question"]["my_forecasts"]["latest"].update(change)
                with self.assertLogs("analyzing_llm_rationale.metaculus_bot", "WARNING"):
                    summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl")
                self.assertEqual(summary.failed, 1)
                self.assertEqual(client.submissions, [])
                self.assertEqual(provider.calls, 0)
    def test_unanswered_question_gets_batch_slot_before_revision(self):
        client, provider = Client(unanswered=True), FakeProvider('{"probability_yes":0.8}')
        with TemporaryDirectory() as directory:
            summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl")
        self.assertEqual([q for q, p in client.submissions], [45])
        self.assertEqual(summary.submitted, 1)

    def test_short_remaining_window_is_reassessed_before_closing(self):
        client, provider = Client(), FakeProvider('{"probability_yes":0.8}')
        now = datetime.now(timezone.utc)
        client.posts[0]["scheduled_close_time"] = (now + timedelta(minutes=30)).isoformat()
        client.posts[0]["question"]["my_forecasts"]["latest"]["start_time"] = now.timestamp() - 1800
        with TemporaryDirectory() as directory:
            summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl")
        self.assertEqual(summary.submitted, 1)

    def test_question_closed_during_research_is_not_posted(self):
        client, provider = Client(), FakeProvider('{"probability_yes":0.8}')

        def research(post):
            client.posts[0]["status"] = "closed"
            return []

        with TemporaryDirectory() as directory:
            self.run_cycle(client, provider, Path(directory) / "audit.jsonl", research_provider=research)
        self.assertEqual(client.submissions, [])

    def test_unchanged_reassessment_is_audited_and_cooled_down(self):
        client, provider = Client(), FakeProvider('{"probability_yes":0.705}')
        research_calls = []
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            summary = self.run_cycle(client, provider, path,
                                     research_provider=lambda p: research_calls.append(p) or [])
            self.assertEqual(summary.forecasted, 1)
            self.assertEqual(summary.submitted, 0)
            self.assertEqual(summary.skipped, 0)
            events = [json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual(events[-1]["outcome"], "reviewed_unchanged")
            self.run_cycle(client, provider, path)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(research_calls), 1)
        self.assertEqual(client.comments, [])
        self.assertEqual(client.submissions, [])

    def test_failed_research_consumes_the_batch_attempt_budget(self):
        client, provider = Client(unanswered=True), FakeProvider('{"probability_yes":0.8}')
        calls = []

        def research(post):
            calls.append(post["id"])
            raise RuntimeError("Synthetic research failure")

        with TemporaryDirectory() as directory:
            summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl", research_provider=research)
        self.assertEqual(calls, [13])
        self.assertEqual(summary.failed, 1)
        self.assertEqual(provider.calls, 0)

    def test_criteria_scale_or_prior_changed_during_research_blocks_publication(self):
        changes = [{"resolution_criteria": "Different outcome"}, {"scaling": {"range_max": 200}},
                   {"type": "numeric"}, {"my_forecasts": {"latest": {
                       "author_id": 99, "probability_yes": 0.7, "start_time": 2}}}]
        for change in changes:
            with self.subTest(change=change), TemporaryDirectory() as directory:
                client, provider = Client(), FakeProvider('{"probability_yes":0.8}')

                def research(post, client=client, change=change):
                    client.posts[0]["question"].update(change)
                    return []

                summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl", research_provider=research)
                self.assertEqual(summary.failed, 1)
                self.assertEqual(client.submissions, [])

    def test_exact_one_percentage_point_change_is_published(self):
        client, provider = Client(), FakeProvider('{"probability_yes":0.71}')
        with TemporaryDirectory() as directory:
            summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl")
        self.assertEqual(summary.submitted, 1)

    def test_unanswered_on_second_page_precedes_revision_without_duplicate_listing(self):
        client, provider = Client(), FakeProvider('{"probability_yes":0.8}')
        template = deepcopy(client.posts[0])
        client.posts = [{**deepcopy(template), "id": number} for number in range(1, 101)]
        client.posts.append({"id": 101, "status": "open", "question": {
            "id": 45, "type": "binary", "my_forecasts": {"latest": None}}})
        offsets = []

        def listing(*args, **kwargs):
            offset = kwargs.get("offset", 0)
            offsets.append(offset)
            return [{"id": p["id"]} for p in client.posts[offset:offset + 100]]

        client.list_open_posts = listing
        with TemporaryDirectory() as directory:
            summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl")
        self.assertEqual(offsets, [0, 100])
        self.assertEqual([q for q, p in client.submissions], [45])
        self.assertEqual(summary.examined, 101)

    def test_changed_staff_clarification_blocks_publication(self):
        client, provider = Client(), FakeProvider('{"probability_yes":0.8}')
        comments = iter(([{"text": "Initial criterion"}], [{"text": "Revised criterion"}]))
        with TemporaryDirectory() as directory:
            summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl",
                                     staff_comment_provider=lambda post_id: next(comments))
        self.assertEqual(summary.failed, 1)
        self.assertEqual(client.submissions, [])

    def test_unchanged_cooldown_survives_restore_on_fresh_runner(self):
        client, provider = Client(), FakeProvider('{"probability_yes":0.705}')
        blob = _Blob(content=b"", generation=1)
        uri = "gs://synthetic-bucket/temporal.jsonl"
        with TemporaryDirectory() as first, TemporaryDirectory() as second, mock.patch.dict(
            os.environ, {"METACULUS_AUDIT_GCS_URI": uri},
        ), mock.patch("analyzing_llm_rationale.metaculus_audit_storage._blob", return_value=blob):
            path = Path(first) / "audit.jsonl"
            restore_audit(path, uri)
            self.run_cycle(client, provider, path)
            self.assertIn(b"reviewed_unchanged", blob.content)
            fresh_path = Path(second) / "audit.jsonl"
            restore_audit(fresh_path, uri)
            second_provider = FakeProvider('{"probability_yes":0.8}')
            summary = self.run_cycle(client, second_provider, fresh_path)
        self.assertEqual(second_provider.calls, 0)
        self.assertEqual(summary.skipped, 1)
        self.assertEqual(client.submissions, [])

    def test_multiple_choice_and_full_numeric_revision_publish(self):
        cases = [
            ({"type": "multiple_choice", "options": ["A", "B"]}, [0.3, 0.7],
             {"probability_yes_per_category": {"A": 0.4, "B": 0.6}}),
            ({"type": "numeric", "scaling": {"continuous_range": list(range(201))}},
             [index / 200 for index in range(201)],
             {"continuous_cdf": [(index / 200) ** 2 for index in range(201)]}),
        ]
        for shape, prior, candidate in cases:
            with self.subTest(kind=shape["type"]), TemporaryDirectory() as directory:
                client, provider = Client(), FakeProvider(json.dumps(candidate))
                client.posts[0]["question"].update(shape)
                client.posts[0]["question"]["my_forecasts"]["latest"] = {
                    "author_id": 99, "start_time": 1, "forecast_values": prior}
                summary = self.run_cycle(client, provider, Path(directory) / "audit.jsonl")
                self.assertEqual(summary.submitted, 1)
                self.assertEqual(len(client.comments), 1)
                self.assertEqual(client.verifications[0]["previous_forecast_start_time"], 1)

    def test_official_binary_no_yes_vector_is_verified_without_reversing_options(self):
        question, payload = {"type": "binary"}, {"probability_yes": 0.8}
        self.assertTrue(_submission_payload_matches(question, payload, {"forecast_values": [0.2, 0.8]}))
        self.assertFalse(_submission_payload_matches(question, payload, {"forecast_values": [0.8, 0.2]}))
        self.assertFalse(_submission_payload_matches(question, payload, {"forecast_values": [0.8, 0.8]}))
        self.assertFalse(_submission_payload_matches(question, payload, {"forecast_values": [float("nan"), 0.8]}))
