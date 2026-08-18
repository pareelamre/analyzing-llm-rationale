"""Unit tests for Foresea Widget and Embed endpoints."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.server import _STATIC_DIR, app  # noqa: E402


class WidgetEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_widget_js_served_with_cors_and_content_type(self):
        resp = self.client.get("/widget.js")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("application/javascript", resp.headers["content-type"])
        self.assertEqual(resp.headers.get("access-control-allow-origin"), "*")
        self.assertIn("Foresea", resp.text)

    def test_static_widget_file_exists(self):
        widget_file = _STATIC_DIR / "widget.js"
        self.assertTrue(widget_file.exists())
        content = widget_file.read_text(encoding="utf-8")
        self.assertIn("foresea-widget-card", content)
        self.assertIn("ForeseaWidget", content)

    def test_embed_forecast_invalid_id_404(self):
        resp = self.client.get("/embed/forecast/invalid_id!@#")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
