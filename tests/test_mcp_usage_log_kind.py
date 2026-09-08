"""A resource read and a tool call were the same line in the usage log.

Four of the five MCP resources are backed by the same client method as a
tool of the same name. Reading foresea://track-record and calling
foresea_track_record both wrote

    {"event": "mcp_tool_call", "tool": "foresea_track_record", ...}

so grouping Cloud Run's jsonPayload by `tool` could not say whether
resources were being used at all, and every resource read was counted as
a tool call. Only foresea_openapi was distinguishable, and only by
accident: it is the one resource with no tool of the same name.

`kind` is additive -- `tool` still says the same thing it always did, so
any existing query keeps working and simply stops conflating the two.
"""

from __future__ import annotations

import io
import json
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from analyzing_llm_rationale import mcp_server as mcp  # noqa: E402

_SOURCE = (_SRC / "analyzing_llm_rationale" / "mcp_server.py").read_text(
    encoding="utf-8"
)


def logged(**kwargs):
    """The record _log_mcp_tool_call prints, parsed back."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        mcp._log_mcp_tool_call(**kwargs)
    return json.loads(buffer.getvalue())


class LogRecordTests(unittest.TestCase):
    def test_a_call_is_a_tool_unless_it_says_otherwise(self):
        record = logged(tool="foresea_track_record")
        self.assertEqual(record["kind"], "tool")

    def test_a_resource_read_says_so(self):
        record = logged(tool="foresea_track_record", kind="resource")
        self.assertEqual(record["kind"], "resource")

    def test_the_two_are_now_distinguishable(self):
        as_tool = logged(tool="foresea_track_record")
        as_resource = logged(tool="foresea_track_record", kind="resource")
        self.assertEqual(as_tool["tool"], as_resource["tool"])
        self.assertNotEqual(as_tool["kind"], as_resource["kind"])

    def test_the_existing_fields_are_untouched(self):
        """Additive: a query grouping by `tool` keeps working."""
        record = logged(tool="foresea_edge_board", context={"limit": 5})
        self.assertEqual(record["event"], "mcp_tool_call")
        self.assertEqual(record["tool"], "foresea_edge_board")
        self.assertEqual(record["limit"], 5)
        self.assertIn("ts", record)

    def test_context_cannot_quietly_overwrite_the_kind(self):
        """context is spread last, so this records the real precedence."""
        record = logged(tool="t", context={"kind": "sneaky"}, kind="resource")
        self.assertEqual(record["kind"], "sneaky")


class EveryResourceIsTaggedTests(unittest.TestCase):
    """Read against the source, because a missed one is silent.

    The same shape of gap put four tools in the log under their internal
    method names: the fallback is quiet, so only comparing the call sites
    finds it.
    """

    @staticmethod
    def _resource_bodies():
        # each @mcp.resource block up to the next decorator or def at that indent
        blocks = re.split(r"\n    @mcp\.resource\(", _SOURCE)[1:]
        return [block.split("\n    @mcp.")[0] for block in blocks]

    def test_every_resource_dispatches_as_a_resource(self):
        bodies = self._resource_bodies()
        self.assertEqual(len(bodies), 5, "expected five MCP resources")
        untagged = [
            body.splitlines()[0]
            for body in bodies
            if "_call_tool_async(" in body and '_kind="resource"' not in body
        ]
        self.assertEqual(
            untagged, [], f"these resources log as tool calls: {untagged}"
        )

    def test_no_tool_claims_to_be_a_resource(self):
        tool_blocks = re.split(r"\n    @mcp\.tool\(\)", _SOURCE)[1:]
        mislabelled = [
            block.splitlines()[1].strip()
            for block in tool_blocks
            if '_kind="resource"' in block.split("\n    @mcp.")[0]
        ]
        self.assertEqual(mislabelled, [])


if __name__ == "__main__":
    unittest.main()
