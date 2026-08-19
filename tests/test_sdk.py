"""Unit tests for Foresea Official Python SDK."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.sdk import AsyncForesea, Foresea  # noqa: E402


class ForeseaSDKTests(unittest.TestCase):
    def setUp(self):
        self.client = Foresea(base_url="https://test.foresea.ink", api_key="test-key")

    @patch("requests.Session.post")
    def test_forecast_sdk(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "predicted_answer": "NO",
            "confidence": 0.75,
            "rationale": "High base rate resistance.",
        }
        mock_post.return_value = mock_resp

        res = self.client.forecast("Will inflation exceed 4%?")
        self.assertEqual(res["predicted_answer"], "NO")
        self.assertEqual(res["confidence"], 0.75)
        self.assertTrue(mock_post.called)

    @patch("requests.Session.get")
    def test_radar_and_health_sdk(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "healthy", "markets": []}
        mock_get.return_value = mock_resp

        res = self.client.system_health()
        self.assertEqual(res["status"], "healthy")

        rad = self.client.radar(limit=5)
        self.assertEqual(rad["status"], "healthy")

    def test_async_sdk_init(self):
        async_client = AsyncForesea(base_url="https://test.foresea.ink")
        self.assertEqual(async_client.base_url, "https://test.foresea.ink")


if __name__ == "__main__":
    unittest.main()
