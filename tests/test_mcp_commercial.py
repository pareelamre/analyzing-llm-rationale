import unittest
from unittest.mock import patch

from analyzing_llm_rationale.mcp_server import (
    ForeseaClient,
    create_mcp_server,
    main_cli,
)


class TestMCPCommercial(unittest.TestCase):
    def test_client_authorization_headers(self):
        client = ForeseaClient(base_url="https://foresea.ink", api_key="test_key_123")
        headers = client._headers()
        self.assertEqual(headers["X-API-Key"], "test_key_123")
        self.assertEqual(headers["Authorization"], "Bearer test_key_123")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_client_headers_without_api_key(self):
        client = ForeseaClient(base_url="https://foresea.ink", api_key="")
        headers = client._headers()
        self.assertNotIn("X-API-Key", headers)
        self.assertNotIn("Authorization", headers)

    def test_feed_latest_uses_configured_base_url_and_headers(self):
        response = type(
            "Response",
            (),
            {"status_code": 200, "json": staticmethod(lambda: {"market_edge_signals": []})},
        )()
        session = type("Session", (), {"get": unittest.mock.Mock(return_value=response)})()
        client = ForeseaClient(
            base_url="https://api.example.test/foresea",
            api_key="test_key_123",
            timeout_s=42,
            session=session,
        )

        self.assertEqual(client.feed_latest(limit=3, min_edge=0.2), {"market_edge_signals": []})
        session.get.assert_called_once_with(
            "https://api.example.test/foresea/feed/latest",
            headers=client._headers(),
            params={"limit": 3, "min_edge": 0.2},
            timeout=42,
        )

    def test_feed_latest_fallback_extracts_edge_board_items(self):
        response = type("Response", (), {"status_code": 503})()
        session = type("Session", (), {"get": unittest.mock.Mock(return_value=response)})()
        client = ForeseaClient(base_url="https://foresea.ink", session=session)
        with patch.object(client, "edge_board", return_value={"edge_board": [{"id": "one"}, {"id": "two"}]}) as edge_board:
            payload = client.feed_latest(limit=1)

        self.assertEqual(payload["market_edge_signals"], [{"id": "one"}])
        edge_board.assert_called_once_with()

    def test_server_creation_and_resources(self):
        mcp = create_mcp_server(base_url="https://foresea.ink", api_key="test_key")
        self.assertIsNotNone(mcp)

    @patch("analyzing_llm_rationale.mcp_server.run_mcp_server")
    def test_main_cli_arg_parsing(self, mock_run):
        exit_code = main_cli([
            "--transport", "stdio",
            "--base-url", "https://custom.foresea.ink",
            "--api-key", "secret_live_key",
            "--timeout", "60",
        ])
        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once_with(
            transport="stdio",
            base_url="https://custom.foresea.ink",
            api_key="secret_live_key",
            timeout_s=60.0,
            host="127.0.0.1",
            port=8000,
        )


if __name__ == "__main__":
    unittest.main()
