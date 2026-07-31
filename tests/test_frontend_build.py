from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class FrontendBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.package = json.loads((cls.root / "package.json").read_text(encoding="utf-8"))

    def test_frontend_build_script_is_declared(self):
        self.assertEqual(self.package["scripts"]["frontend:build"], "vite build")
        self.assertIn("vite", self.package["devDependencies"])

    def test_frontend_source_entries_exist(self):
        frontend = self.root / "frontend"
        for name in ("index.html", "trade.html", "agents.html"):
            with self.subTest(name=name):
                source = (frontend / name).read_text(encoding="utf-8")
                self.assertIn('/src/page-context.ts', source)

    def test_built_pages_load_vite_context_asset(self):
        asset_pattern = re.compile(r'<script type="module" crossorigin src="/static/assets/page-context-[^"]+\.js"></script>')
        static = self.root / "static"
        for name in ("index.html", "trade.html", "agents.html"):
            with self.subTest(name=name):
                built = (static / name).read_text(encoding="utf-8")
                self.assertRegex(built, asset_pattern)


if __name__ == "__main__":
    unittest.main()
