import unittest

from starlette.testclient import TestClient

from analyzing_llm_rationale.server import app


class TestServerNewCapabilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_llms_full_txt(self):
        resp = self.client.get("/llms-full.txt")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Foresea Full Agent & Developer Reference", resp.text)
        self.assertIn("/predict/ensemble", resp.text)
        self.assertIn("/analytics/export.parquet", resp.text)

    def test_analytics_export_parquet_edge_board(self):
        resp = self.client.get("/analytics/export.parquet?dataset=edge_board")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("content-type"), "application/vnd.apache.parquet")
        self.assertTrue(resp.content.startswith(b"PAR1"))
        etag = resp.headers.get("etag")
        self.assertIsNotNone(etag)

        # Test If-None-Match ETag 304 response
        resp_304 = self.client.get("/analytics/export.parquet?dataset=edge_board", headers={"If-None-Match": etag})
        self.assertEqual(resp_304.status_code, 304)

    def test_analytics_export_parquet_models_comparison(self):
        resp = self.client.get("/analytics/export.parquet?dataset=models_comparison")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("content-type"), "application/vnd.apache.parquet")
        self.assertTrue(resp.content.startswith(b"PAR1"))

    def test_analytics_export_parquet_invalid_dataset(self):
        resp = self.client.get("/analytics/export.parquet?dataset=unknown_dataset")
        self.assertEqual(resp.status_code, 400)

    def test_predict_ensemble_with_member_forecasts(self):
        payload = {
            "question": "Will SpaceX launch Starship Flight 6 before November 2026?",
            "market_probability": 0.45,
            "category": "science",
            "member_forecasts": [
                {"model": "gemma-4-26b-a4b-it", "probability": 0.65, "rationale": "Strong launch cadence."},
                {"model": "gpt-oss-120b", "probability": 0.60, "rationale": "Regulatory approval likely."},
                {"model": "crowd-follow", "probability": 0.45, "rationale": "Market average."},
            ],
        }
        resp = self.client.post("/predict/ensemble", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("ensemble_probability", data)
        self.assertGreater(data["ensemble_probability"], 0.45)
        self.assertEqual(len(data["confidence_interval"]), 2)
        self.assertIn("consensus_level", data)
        self.assertIn("member_contributions", data)
        self.assertEqual(len(data["member_contributions"]), 3)
        self.assertIsNotNone(data["edge"])

    def test_agent_manifest_includes_new_endpoints(self):
        resp = self.client.get("/.well-known/agent.json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("llms_full_txt_url", data)
        http_endpoints = data.get("http", {})
        self.assertIn("ensemble_forecast", http_endpoints)
        self.assertIn("radar_stream", http_endpoints)
        self.assertIn("feed_stream", http_endpoints)
        self.assertIn("analytics_export_parquet", http_endpoints)


if __name__ == "__main__":
    unittest.main()
