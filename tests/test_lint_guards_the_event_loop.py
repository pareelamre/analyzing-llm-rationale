"""The ASYNC lint rules must stay selected.

This service is FastAPI in front of market adapters that speak sync
``requests``. A blocking call reaching an ``async def`` does not fail or
log -- it stalls every other request on the instance for the length of an
upstream round trip, which on Cloud Run means one slow Kalshi call degrades
every concurrent caller.

The codebase routes those through ``run_in_executor`` and has zero
violations. Nothing enforces that except the lint select list, and dropping
a rule from that list is silent. This notices.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"

#: A handler doing each thing the rules exist to stop.
_BLOCKING_ASYNC_HANDLER = '''
import time
import requests


async def handler(url: str):
    resp = requests.get(url, timeout=5)
    time.sleep(1)
    with open("/tmp/x") as fh:
        fh.read()
    return resp
'''


def _lint_config() -> dict:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        raise unittest.SkipTest("tomllib needs 3.11+") from None
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["ruff"]["lint"]


class AsyncRulesStaySelectedTests(unittest.TestCase):
    def test_async_is_selected(self):
        self.assertIn("ASYNC", _lint_config()["select"])

    def test_async_is_not_then_ignored(self):
        """Selecting a family and ignoring it back reads as enforcement."""
        ignored = _lint_config().get("ignore", [])
        self.assertEqual([r for r in ignored if r.startswith("ASYNC")], [])

    def test_the_configured_ruleset_actually_rejects_a_blocking_handler(self):
        """Assert the behaviour, not just the string in the config file."""
        try:
            import ruff  # noqa: F401
        except ImportError:
            proc = subprocess.run(
                [sys.executable, "-m", "ruff", "--version"],
                capture_output=True, cwd=_ROOT,
            )
            if proc.returncode != 0:
                raise unittest.SkipTest("ruff is not installed") from None

        with tempfile.TemporaryDirectory(dir=_ROOT) as tmp:
            probe = Path(tmp) / "blocking_handler.py"
            probe.write_text(textwrap.dedent(_BLOCKING_ASYNC_HANDLER), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "ruff", "check", str(probe),
                 "--no-cache", "--output-format=concise"],
                capture_output=True, text=True, cwd=_ROOT,
            )

        out = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, f"blocking handler passed lint:\n{out}")
        for code in ("ASYNC210", "ASYNC251", "ASYNC230"):
            self.assertIn(code, out, f"{code} not raised; got:\n{out}")


if __name__ == "__main__":
    unittest.main()
