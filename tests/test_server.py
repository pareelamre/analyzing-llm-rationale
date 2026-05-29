from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale.server import _state, app  # noqa: E402


class FakeProvider:
    def __init__(self):
        self.calls = []

    def chat_completion(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        return json.dumps({
            "predicted_answer": "Yes",
            "confidence": 0.7,
            "rationale": "Evidence supports a yes forecast.",
        })


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


if __name__ == "__main__":
    unittest.main()
