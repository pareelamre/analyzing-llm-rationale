from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ["ANALYTICS_DB"] = str(Path(tempfile.gettempdir()) / "foresea_test_analytics.duckdb")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale.server import _ANALYTICS_DB, _state, app  # noqa: E402


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.response = {
            "predicted_answer": "Yes",
            "confidence": 0.7,
            "rationale": "Evidence supports a yes forecast.",
        }

    def chat_completion(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        return json.dumps(self.response)


class FakeEvidencePipeline:
    def __init__(self):
        self.calls = []

    def fetch_summarize_rank(self, question, top_k=5):
        self.calls.append((question, top_k))
        return [
            {
                "title": "Central bank signals policy shift",
                "source": "Example News",
                "url": "https://example.com/rates",
                "publish_date": "2026-05-01",
                "summary": "Officials discussed conditions for a possible rate cut.",
                "relevance_score": 0.91,
                "search_query": "Federal Reserve rate cut July 2026",
            }
        ]


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.evidence_pipeline = FakeEvidencePipeline()
        _state.clear()
        _state.update({
            "provider": self.provider,
            "evidence_pipeline": self.evidence_pipeline,
            "variants": {
                "variant0_neutral_baseline": SimpleNamespace(
                    output_fields=("predicted_answer", "confidence", "rationale")
                )
            },
            "system_prompt": "System",
            "prompt_templates": {
                "variant0_neutral_baseline": "[question]\nReturn JSON.",
            },
            "temperature": 0.0,
            "max_tokens": 256,
            "model_key": "test-model",
        })
        self.client = TestClient(app)

    def tearDown(self):
        _state.clear()
        if _ANALYTICS_DB.exists():
            _ANALYTICS_DB.unlink()

    def test_predict_fetches_and_returns_evidence(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "Will the Fed cut rates before July 31, 2026?",
                "description": "A forecasting question.",
                "variant": "variant0_neutral_baseline",
                "evidence_top_k": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_answer"], "Yes")
        self.assertEqual(payload["rationale"], "Evidence supports a yes forecast.")
        self.assertEqual(payload["model_rationale"], "Evidence supports a yes forecast.")
        self.assertEqual(payload["evidence_error"], None)
        self.assertEqual(payload["evidence_sources"][0]["source"], "Example News")
        self.assertEqual(
            payload["evidence_sources"][0]["url"],
            "https://example.com/rates",
        )
        self.assertEqual(len(payload["evidence_articles"]), 1)
        self.assertEqual(payload["evidence_articles"][0]["relevance_score"], 0.91)
        self.assertEqual(
            self.evidence_pipeline.calls,
            [("Will the Fed cut rates before July 31, 2026?", 3)],
        )
        self.assertIn("Central bank signals", self.provider.calls[0][1]["content"])

    def test_predict_uses_supplied_articles_without_fetching(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "Will event X happen?",
                "news_articles": [
                    {
                        "title": "Supplied evidence",
                        "summary": "The caller already provided this article.",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["evidence_articles"][0]["title"], "Supplied evidence")
        self.assertEqual(self.evidence_pipeline.calls, [])

    def test_predict_strips_html_from_returned_evidence(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "Will event X happen?",
                "news_articles": [
                    {
                        "title": "Supplied evidence",
                        "source": "Example News",
                        "summary": '<a href="https://example.com">Evidence</a>&nbsp;details',
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["evidence_articles"][0]["summary"], "Evidence details")

    def test_records_anonymous_page_visit(self):
        response = self.client.post(
            "/analytics/visit",
            json={
                "path": "/",
                "referrer": "",
                "timezone": "Europe/Berlin",
            },
            headers={"user-agent": "test-client"},
        )
        self.assertEqual(response.status_code, 200)

        summary = self.client.get("/analytics/summary")
        self.assertEqual(summary.status_code, 200)
        payload = summary.json()
        self.assertGreaterEqual(payload["total_visits"], 1)
        self.assertGreaterEqual(payload["unique_visitors"], 1)
        self.assertEqual(payload["by_day"][0]["visits"], 1)

    def test_predict_multiple_choice_returns_options(self):
        self.provider.response = {
            "type": "multiple_choice",
            "options": [
                {"label": "Alice", "probability": 0.2},
                {"label": "Bob", "probability": 0.7},
                {"label": "Carol", "probability": 0.1},
            ],
            "rationale": "Bob has the strongest polling and fundraising evidence.",
        }

        response = self.client.post(
            "/predict",
            json={
                "question": "Who will win the Example City mayoral election?",
                "question_type": "multiple_choice",
                "options": ["Alice", "Bob", "Carol"],
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question_type"], "multiple_choice")
        self.assertEqual(payload["predicted_answer"], "Bob")
        self.assertEqual(payload["confidence"], 0.7)
        self.assertEqual(len(payload["options"]), 3)
        self.assertIn("multiple_choice", self.provider.calls[0][-1]["content"])
        self.assertIn("Alice, Bob, Carol", self.provider.calls[0][-1]["content"])
        self.assertIn("overrides any earlier variant template", self.provider.calls[0][-1]["content"])
        self.assertIn("Only binary questions should use a Yes/No", self.provider.calls[0][-1]["content"])

    def test_predict_numeric_returns_range_forecast(self):
        self.provider.response = {
            "type": "numeric",
            "p10": 42,
            "p50": 55,
            "p90": 73,
            "unit": "USD",
            "rationale": "Recent guidance supports a mid-range estimate.",
        }

        response = self.client.post(
            "/predict",
            json={
                "question": "What will Example Corp revenue be in Q4 2026?",
                "question_type": "numeric",
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question_type"], "numeric")
        self.assertEqual(payload["predicted_answer"], "55")
        self.assertEqual(payload["confidence"], None)
        self.assertEqual(
            payload["range_forecast"],
            {"p10": "42", "p50": "55", "p90": "73", "unit": "USD"},
        )
        self.assertIn('"type":"numeric"', self.provider.calls[0][-1]["content"])
        self.assertIn("Only binary questions should use a Yes/No", self.provider.calls[0][-1]["content"])

    def test_predict_without_question_type_asks_model_to_infer_schema(self):
        self.provider.response = {
            "type": "numeric",
            "p10": 2.1,
            "p50": 2.6,
            "p90": 3.4,
            "unit": "%",
            "rationale": "Inflation is likely to stay near recent targets.",
        }

        response = self.client.post(
            "/predict",
            json={
                "question": "What will US CPI inflation be in December 2026?",
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question_type"], "numeric")
        self.assertEqual(payload["predicted_answer"], "2.6")
        prompt = self.provider.calls[0][-1]["content"]
        self.assertIn("First infer the question type", prompt)
        self.assertIn("- numeric:", prompt)

    def test_predict_date_returns_range_forecast(self):
        self.provider.response = {
            "type": "date",
            "p10": "2026-07-01",
            "p50": "2026-09-15",
            "p90": "2026-12-31",
            "rationale": "The event is most likely in the second half of 2026.",
        }

        response = self.client.post(
            "/predict",
            json={
                "question": "When will the Example spacecraft launch?",
                "question_type": "date",
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question_type"], "date")
        self.assertEqual(payload["predicted_answer"], "2026-09-15")
        self.assertEqual(payload["range_forecast"]["p50"], "2026-09-15")

    def test_predict_rejects_unknown_question_type(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "What will Example Corp revenue be in Q4 2026?",
                "question_type": "essay",
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
