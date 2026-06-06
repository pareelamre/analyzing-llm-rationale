"""Venue MCP client: Foresea consuming Polymarket/Kalshi MCP servers as tools.
SDK-touching calls are mocked so the suite runs without the mcp SDK / network."""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import venue_mcp  # noqa: E402


class VenueMcpTests(unittest.TestCase):
    def test_configured_venues_reads_env(self):
        with mock.patch.dict("os.environ", {"POLYMARKET_MCP_URL": "https://poly/mcp/",
                                            "KALSHI_MCP_URL": ""}, clear=False):
            venues = dict(venue_mcp.configured_venues())
        self.assertEqual(venues.get("polymarket"), "https://poly/mcp")  # trailing / stripped
        self.assertNotIn("kalshi", venues)  # empty url skipped

    def test_no_venues_means_no_tools(self):
        # Regression: unconfigured → discover yields nothing → agent loop unchanged.
        with mock.patch.object(venue_mcp, "configured_venues", return_value=[]):
            out = asyncio.run(venue_mcp.discover_tools())
        self.assertEqual(out, {})

    def test_discover_namespaces_and_caps(self):
        async def fake_list(url):
            return [(f"t{i}", f"desc{i}") for i in range(20)]
        with mock.patch.object(venue_mcp, "configured_venues",
                               return_value=[("polymarket", "https://poly")]), \
                mock.patch.object(venue_mcp, "_list_tools", side_effect=fake_list):
            out = asyncio.run(venue_mcp.discover_tools())
        self.assertLessEqual(len(out), venue_mcp._MAX_TOOLS_PER_VENUE)  # capped
        self.assertTrue(all(k.startswith("polymarket_") for k in out))
        self.assertEqual(out["polymarket_t0"]["name"], "t0")

    def test_call_tool_safe_truncates_and_never_raises(self):
        async def fake_call(url, name, args):
            return "x" * 99999
        with mock.patch.object(venue_mcp, "_call_tool", side_effect=fake_call):
            out = asyncio.run(venue_mcp.call_tool_safe("u", "t", {}))
        self.assertLessEqual(len(out), venue_mcp._MAX_OUTPUT)

        async def boom(url, name, args):
            raise RuntimeError("down")
        with mock.patch.object(venue_mcp, "_call_tool", side_effect=boom):
            out = asyncio.run(venue_mcp.call_tool_safe("u", "t", {}))
        self.assertIn("unavailable", out)

    def test_make_tool_fn_binds_and_calls(self):
        async def fake_call(url, name, args):
            return f"{name}:{args.get('q')}"
        with mock.patch.object(venue_mcp, "_call_tool", side_effect=fake_call):
            fn = venue_mcp.make_tool_fn("https://poly", "orderbook")
            out = asyncio.run(fn({"q": "abc"}))
        self.assertEqual(out, "orderbook:abc")


if __name__ == "__main__":
    unittest.main()
