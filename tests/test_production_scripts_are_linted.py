"""Scripts a workflow actually runs get the same lint as src/.

CI checks `scripts/` with seven rules -- F821, F811, F632, F502, F522,
F701, B002 -- because the directory also holds one-off research scripts
whose style nobody wants to litigate. Those seven catch the errors that
break a script outright.

But a dozen of these run in production: they publish the board, tick the
agents, build the track record. They are not research code and they are
already clean under the full ruleset, so the narrow gate is not buying
them anything -- it is only withholding cover.

The list is derived from the workflows rather than pinned here, so a
script becomes covered the moment a workflow starts running it. That is
the point: the thing that makes a script production is being invoked by
one.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / ".github" / "workflows"

_INVOCATION = re.compile(r"scripts/([A-Za-z0-9_]+\.py)")


def _production_scripts() -> list[str]:
    """Every scripts/*.py named by a workflow file."""
    named: set[str] = set()
    for wf in sorted(_WORKFLOWS.glob("*.yml")):
        named |= set(_INVOCATION.findall(wf.read_text(encoding="utf-8", errors="replace")))
    return sorted(f"scripts/{name}" for name in named if (_ROOT / "scripts" / name).exists())


def _ruff_available() -> bool:
    try:
        return subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True, cwd=_ROOT,
        ).returncode == 0
    except OSError:
        return False


class ProductionScriptsAreFullyLintedTests(unittest.TestCase):
    def test_the_workflows_name_some_scripts(self):
        """If the scan breaks, fail loudly rather than covering nothing."""
        found = _production_scripts()
        self.assertGreaterEqual(len(found), 8, f"only found {found}")
        self.assertIn("scripts/build_agent_trading_board.py", found)

    def test_every_named_script_exists(self):
        """A workflow pointing at a deleted script fails only when it runs."""
        missing = []
        for wf in sorted(_WORKFLOWS.glob("*.yml")):
            text = wf.read_text(encoding="utf-8", errors="replace")
            for name in _INVOCATION.findall(text):
                if not (_ROOT / "scripts" / name).exists():
                    missing.append(f"{wf.name} -> scripts/{name}")
        self.assertEqual(sorted(set(missing)), [], "workflow references a missing script")

    def test_they_pass_the_full_ruleset(self):
        if not _ruff_available():
            raise unittest.SkipTest("ruff is not installed")
        targets = _production_scripts()
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", *targets,
             "--no-cache", "--output-format=concise"],
            capture_output=True, text=True, cwd=_ROOT,
        )
        self.assertEqual(
            proc.returncode, 0,
            "scripts run by a workflow must pass the same lint as src/:\n"
            + (proc.stdout or proc.stderr),
        )


if __name__ == "__main__":
    unittest.main()
