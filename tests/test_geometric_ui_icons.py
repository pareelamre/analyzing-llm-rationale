from __future__ import annotations

import unittest
from pathlib import Path


class GeometricUiIconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        static_dir = Path(__file__).resolve().parents[1] / "static"
        cls.index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        cls.foresea_html = "\n".join(
            (static_dir / name).read_text(encoding="utf-8")
            for name in ("index.html", "agents.html", "trade.html")
        )

    def test_stop_icon_is_geometric_and_centered(self) -> None:
        self.assertIn("width: 40px !important;", self.index_html)
        self.assertIn("height: 40px !important;", self.index_html)
        self.assertIn("place-items: center;", self.index_html)
        self.assertIn(
            ".send-btn-stop-icon {\n"
            "      background: currentColor;\n"
            "      border-radius: 1px;\n"
            "      display: block;\n"
            "      height: 8px;\n"
            "      width: 8px;",
            self.index_html,
        )
        self.assertIn(
            "'<span class=\"send-btn-stop-icon\" aria-hidden=\"true\"></span>'",
            self.index_html,
        )

    def test_ui_icons_do_not_depend_on_text_glyphs(self) -> None:
        for glyph in (
            "✕",
            "✎",
            "⧉",
            "■",
            "↻",
            "↑",
            "⌄",
            "☆",
            "★",
            "✓",
            "⚠",
            "←",
            "→",
            "↗",
            "＋",
            "▲",
            "▼",
            "📈",
            "⚗",
        ):
            with self.subTest(glyph=glyph):
                self.assertNotIn(glyph, self.foresea_html)

    def test_shared_svg_icon_system_is_present(self) -> None:
        self.assertIn('<symbol id="ui-x"', self.index_html)
        self.assertIn('<symbol id="ui-star"', self.index_html)
        self.assertIn("function uiIcon(name, className = '')", self.index_html)


if __name__ == "__main__":
    unittest.main()
