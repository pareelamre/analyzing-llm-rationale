"""/radar carries 2.12 MB, and 96% of it is track-record ledgers.

Measured against the live endpoint:

    models_comparison        1,115,704   52.5%   (99.2% of it nested paper_pnl)
    paper_pnl                  438,834   20.7%
    primary_paper_pnl          281,120   13.2%
    mark_to_market_by_model    206,790    9.7%
    markets                     31,418    1.5%   <- what the endpoint is for

``limit`` does not touch any of that. It bounds ``markets``, so /radar?limit=1
still returns 2.06 MB. The only frontend caller renders three market rows and
reads nothing else, so the landing page downloaded 2.1 MB to draw a table.

``include_track_record=false`` drops the ledgers. It defaults true, so nothing
that reads them today changes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale import server as server_module  # noqa: E402


def _radar_payload():
    """A radar response carrying every block, shaped like the live one."""
    return server_module.RadarResponse(
        updated_at="2026-09-07T12:00:00Z",
        markets=[],
        models_comparison=[{"model": "m", "paper_pnl": {"flat": {"pnl": 1.0}}}],
        paper_pnl={"flat": {"pnl": 2.0}},
        primary_paper_pnl={"flat": {"pnl": 3.0}},
        mark_to_market_by_model=[{"model": "m", "value": 4.0}],
        quarter_kelly_by_model=[{"model": "m"}],
        growth_1pct_by_model=[{"model": "m"}],
        growth_2pct_by_model=[{"model": "m"}],
        calibration=[{"bin": "0-10"}],
        n_markets_open=7,
    )


class RadarTrackRecordToggleTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server_module.app)
        patcher = mock.patch.object(
            server_module, "_radar_from_track_record",
            lambda limit: _radar_payload(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_default_still_carries_everything(self):
        """Existing callers must not notice this parameter exists."""
        body = self.client.get("/radar").json()
        for key in server_module._RADAR_TRACK_RECORD_BLOCKS:
            with self.subTest(key=key):
                self.assertTrue(body[key], f"{key} should be populated by default")

    def test_opting_out_drops_the_ledgers(self):
        body = self.client.get("/radar?include_track_record=false").json()
        for key in server_module._RADAR_TRACK_RECORD_BLOCKS:
            with self.subTest(key=key):
                self.assertFalse(body[key], f"{key} should be empty")

    def test_opting_out_keeps_the_radar_itself(self):
        """Dropping the ledgers must not drop what the endpoint is named for."""
        body = self.client.get("/radar?include_track_record=false").json()
        self.assertIn("markets", body)
        self.assertEqual(body["n_markets_open"], 7)
        self.assertEqual(body["updated_at"], "2026-09-07T12:00:00Z")
        self.assertTrue(body["calibration"])

    def test_the_keys_are_emptied_not_removed(self):
        """The response model declares them; a client reading one still gets
        the type it expects rather than a KeyError."""
        body = self.client.get("/radar?include_track_record=false").json()
        self.assertEqual(body["models_comparison"], [])
        self.assertEqual(body["mark_to_market_by_model"], [])
        self.assertIsNone(body["paper_pnl"])
        self.assertIsNone(body["primary_paper_pnl"])

    def test_it_is_materially_smaller(self):
        import json

        full = self.client.get("/radar").content
        lean = self.client.get("/radar?include_track_record=false").content
        self.assertLess(len(lean), len(full))
        json.loads(lean)  # still valid JSON


class TheTwoBulkKeyListsAgreeTests(unittest.TestCase):
    """server and mcp_server drop the same blocks for the same reason.

    They cannot import each other, so the lists are separate literals. If one
    grows a block the other does not, the same payload is bulky on one
    surface and lean on the other, and nobody finds out from a failure.
    """

    def test_the_lists_match(self):
        from analyzing_llm_rationale import mcp_server

        self.assertEqual(
            tuple(server_module._RADAR_TRACK_RECORD_BLOCKS),
            tuple(mcp_server._TRACK_RECORD_BULK_KEYS),
        )


if __name__ == "__main__":
    unittest.main()
