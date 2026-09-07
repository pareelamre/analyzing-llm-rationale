"""Every MCP tool must log its public name, not its internal method name.

``_tool_name`` maps the client coroutine to the public tool name and falls
back to the coroutine's own ``__name__``. The fallback is silent, so four
entries were simply missing and those tools wrote their internal name --
``arecent_trades``, ``aoptimize_portfolio``, ``adebate_market``,
``aopenapi`` -- into ``jsonPayload.tool``, which Cloud Run indexes. Usage
grouped by tool name misfiled them while every other tool was correct.

Nothing failed, so only reading the map against the call sites found it.
This test does that reading.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from analyzing_llm_rationale import mcp_server as mcp  # noqa: E402

_SOURCE = (_SRC / "analyzing_llm_rationale" / "mcp_server.py").read_text(encoding="utf-8")

#: Client methods handed to _call_tool / _call_tool_async by a tool or resource.
_DISPATCHED = frozenset(
    re.findall(r"_call_tool(?:_async)?\(\s*client\.(\w+)", _SOURCE)
)


class ToolNamesCoverTheDispatchTests(unittest.TestCase):
    def test_the_scan_found_the_call_sites(self):
        """Guard the regex: if it stops matching, the test must not go quiet."""
        self.assertGreaterEqual(len(_DISPATCHED), 20)
        self.assertIn("aforecast", _DISPATCHED)

    def test_every_dispatched_method_has_a_public_name(self):
        missing = sorted(m for m in _DISPATCHED if m not in mcp._TOOL_NAMES)
        self.assertEqual(
            missing, [],
            "these log their internal method name into jsonPayload.tool: "
            + ", ".join(missing),
        )

    def test_no_public_name_is_an_internal_method_name(self):
        """The fallback's output is exactly what a correct entry must not be."""
        for method, public in mcp._TOOL_NAMES.items():
            with self.subTest(method=method):
                self.assertNotEqual(public, method)
                self.assertTrue(
                    public.startswith("foresea_"),
                    f"{method} logs as {public!r}, which is not a public tool name",
                )

    def test_sync_and_async_spellings_agree(self):
        """A tool must log the same name whichever path served it."""
        for method, public in mcp._TOOL_NAMES.items():
            if not method.startswith("a"):
                continue
            sync = method[1:]
            if sync in mcp._TOOL_NAMES:
                with self.subTest(method=method):
                    self.assertEqual(mcp._TOOL_NAMES[sync], public)

    def test_the_four_that_were_missing_are_named(self):
        for method, expected in (
            ("arecent_trades", "foresea_recent_trades"),
            ("aoptimize_portfolio", "foresea_optimize_portfolio"),
            ("adebate_market", "foresea_debate_market"),
            ("aopenapi", "foresea_openapi"),
        ):
            with self.subTest(method=method):
                self.assertEqual(mcp._TOOL_NAMES.get(method), expected)


if __name__ == "__main__":
    unittest.main()
