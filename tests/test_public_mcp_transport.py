"""Exercise the public mount with the real MCP transport, without upstream calls."""

import asyncio
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from analyzing_llm_rationale import server


class PublicMCPTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = FastAPI()
        with patch.object(server, "app", self.app), patch.object(
            server, "_PUBLIC_MCP"
        ), patch.object(server, "_PUBLIC_MCP_APP"), patch.dict(
            "os.environ", {"DISABLE_PUBLIC_MCP": "false"}
        ):
            server._mount_public_mcp_endpoint()
            self.mcp = server._PUBLIC_MCP
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="https://foresea.test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_get_and_head_finish_without_opening_an_event_stream(self):
        async with self.mcp.session_manager.run():
            for path in ("/mcp", "/mcp/"):
                for method in ("GET", "HEAD"):
                    with self.subTest(path=path, method=method):
                        response = await asyncio.wait_for(
                            self.client.request(
                                method, path, headers={"Accept": "text/event-stream"}
                            ), timeout=1,
                        )
                        self.assertEqual(response.status_code, 405)
                        self.assertEqual(response.headers["allow"], "POST")
                        self.assertNotIn("text/event-stream", response.headers.get("content-type", ""))

    async def test_post_initialization_discovery_and_tool_execution_still_work(self):
        headers = {"Accept": "application/json, text/event-stream"}
        async def rpc(method, params=None):
            response = await self.client.post(
                self.rpc_path, headers=headers, follow_redirects=True,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("application/json", response.headers["content-type"])
            body = response.json()
            self.assertNotIn("error", body)
            return body["result"]

        async with self.mcp.session_manager.run():
            for self.rpc_path in ("/mcp", "/mcp/"):
                await self._check_post(rpc, headers)

    async def _check_post(self, rpc, headers):
        initialized = await rpc("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "transport-test", "version": "1"},
        })
        headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
        notification = await self.client.post(
            self.rpc_path, headers=headers, follow_redirects=True,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        self.assertEqual(notification.status_code, 202)
        tools = await rpc("tools/list")
        self.assertTrue(tools["tools"])
        await rpc("ping")
        resources = await rpc("resources/list")
        self.assertTrue(resources["resources"])
        with patch(
            "analyzing_llm_rationale.mcp_server.ForeseaClient.ascan_markets",
            return_value={"markets": [], "source": "synthetic-test"},
        ):
            result = await rpc("tools/call", {"name": "foresea_scan_markets", "arguments": {}})
        self.assertFalse(result.get("isError", False))
        self.assertIn("synthetic-test", str(result))

    async def test_slashless_post_preserves_307_redirect(self):
        response = await self.client.post("/mcp", json={"jsonrpc": "2.0"})
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/mcp/")

    async def test_official_sdk_client_initializes_and_calls_tools(self):
        self.client.follow_redirects = True
        async with self.mcp.session_manager.run():
            async with streamable_http_client(
                "https://foresea.test/mcp", http_client=self.client,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self.assertTrue(listed.tools)
                    with patch(
                        "analyzing_llm_rationale.mcp_server.ForeseaClient.ascan_markets",
                        return_value={"markets": [], "source": "sdk-test"},
                    ):
                        result = await session.call_tool("foresea_scan_markets", {})
                    self.assertFalse(result.isError)
                    self.assertIn("sdk-test", str(result))

    async def test_other_methods_still_reach_the_mounted_transport(self):
        async with self.mcp.session_manager.run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.mcp.streamable_http_app()),
                base_url="https://foresea.test",
            ) as sdk_client:
                for method in ("DELETE", "OPTIONS"):
                    expected = await sdk_client.request(method, "/")
                    actual = await self.client.request(method, "/mcp/")
                    self.assertEqual(actual.status_code, expected.status_code)
                    self.assertEqual(actual.headers.get("allow"), expected.headers.get("allow"))
                    self.assertEqual(actual.content, expected.content)

    async def test_disabled_public_mcp_registers_no_endpoint(self):
        disabled_app = FastAPI()
        with patch.object(server, "app", disabled_app), patch.dict(
            "os.environ", {"DISABLE_PUBLIC_MCP": "true"}
        ):
            server._mount_public_mcp_endpoint()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=disabled_app), base_url="https://foresea.test"
        ) as client:
            for path in ("/mcp", "/mcp/"):
                for method in ("GET", "HEAD", "POST"):
                    response = await client.request(method, path)
                    self.assertEqual(response.status_code, 404)

    async def test_outer_app_preserves_cors_and_request_id(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="https://foresea.ink"
        ) as client:
            response = await client.get(
                "/mcp/", headers={"Origin": "https://foresea.ink", "Accept": "text/event-stream"}
            )
            self.assertEqual(response.status_code, 405)
            self.assertEqual(response.headers["access-control-allow-origin"], "https://foresea.ink")
            self.assertIn("x-request-id", response.headers)
            response = await client.options("/mcp/", headers={
                "Origin": "https://foresea.ink", "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "mcp-protocol-version,content-type",
            })
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
