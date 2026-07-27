from __future__ import annotations

import unittest
from pathlib import Path


class LandingTrackRecordPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_html = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

    def test_transparency_panel_uses_live_track_record(self) -> None:
        self.assertIn('id="transparencyPerformancePanel"', self.index_html)
        self.assertIn("refreshTransparencyPanel()", self.index_html)
        self.assertIn("_liveJsonFetch('/track-record')", self.index_html)
        self.assertIn("TRANSPARENCY_REFRESH_MS = 60_000", self.index_html)

    def test_transparency_panel_no_longer_ships_fixed_metrics(self) -> None:
        self.assertIn('id="trustAccuracy"', self.index_html)
        self.assertIn('id="trustResolved"', self.index_html)
        self.assertIn('id="trustBrier"', self.index_html)
        self.assertIn("return String(bin.bin).replace(/-/g, ' - ')", self.index_html)
        self.assertIn("return `${low}% - ${low + 10}%`", self.index_html)
        self.assertNotIn("`${low}%?${low + 10}%`", self.index_html)
        self.assertNotIn(">76.9%</span>", self.index_html)
        self.assertNotIn(">234</span>", self.index_html)
        self.assertNotIn(">0.197</span>", self.index_html)


if __name__ == "__main__":
    unittest.main()
