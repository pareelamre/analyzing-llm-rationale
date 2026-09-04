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
