"""static/ is built from frontend/; edits to static/ do not survive.

vite.config.mjs sets root=frontend, outDir=static, and the Dockerfile runs
`npm run frontend:build` and copies /app/static/ into the image. So every
deploy regenerates static/ from frontend/. An edit made only to static/ is
overwritten by the next image build.

That happened once: #476 added a weather badge to static/index.html and
never touched frontend/index.html. The commit merged, CI passed, and the
badge was gone from production -- the served page had `venuePill` and no
`weatherPill` at all. Nothing failed, because nothing was looking.

This looks. Any identifier declared in a built page must exist in the page
it is built from; if it does not, it was written into the artifact and is
already lost.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: Entry points from vite.config.mjs rollupOptions.input.
_PAGES = ("index", "agents", "trade")

_DECLARATION = re.compile(r"\b(?:const|let|function)\s+([A-Za-z_$][\w$]*)")


def _identifiers(text: str) -> set[str]:
    return set(_DECLARATION.findall(text))


class BuiltPagesComeFromSourceTests(unittest.TestCase):
    def test_every_entry_point_exists_in_both_trees(self):
        for page in _PAGES:
            with self.subTest(page=page):
                self.assertTrue((_ROOT / "frontend" / f"{page}.html").exists())
                self.assertTrue((_ROOT / "static" / f"{page}.html").exists())

    def test_the_scan_is_not_vacuous(self):
        """A guard that stops matching must fail, not fall silent."""
        built = (_ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertGreater(len(_identifiers(built)), 500)

    def test_no_identifier_exists_only_in_the_built_page(self):
        for page in _PAGES:
            built = (_ROOT / "static" / f"{page}.html").read_text(encoding="utf-8")
            source = (_ROOT / "frontend" / f"{page}.html").read_text(encoding="utf-8")
            orphans = sorted(
                name for name in _identifiers(built)
                if not re.search(rf"\b{re.escape(name)}\b", source)
            )
            with self.subTest(page=page):
                self.assertEqual(
                    orphans, [],
                    f"declared in static/{page}.html but absent from "
                    f"frontend/{page}.html, so the next `npm run frontend:build` "
                    f"deletes it: {orphans}",
                )

    def test_the_weather_badge_is_in_the_source(self):
        """The specific regression: #476 wrote it only into the artifact."""
        source = (_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        # assertTrue, not assertIn: the haystack is 600KB and assertIn prints it.
        self.assertTrue("weatherPill" in source, "weatherPill missing from frontend/index.html")
        self.assertTrue("${weatherPill}" in source, "weatherPill is declared but never rendered")


if __name__ == "__main__":
    unittest.main()
