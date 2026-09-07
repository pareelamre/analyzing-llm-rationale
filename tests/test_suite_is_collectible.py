"""Guard: every test must be visible to the runner CI actually uses.

CI runs `python -m unittest discover -s tests`, which collects only
unittest.TestCase subclasses. A module-level `def test_...` runs under
pytest locally and is silently ignored there -- the file looks tested, the
suite looks green, and nothing covers the code.

That was not hypothetical. tests/test_auth_dedup.py and
tests/test_crypto_kalshi.py were both in that state; the latter reported
"Ran 0 tests" while holding the only coverage of the Kalshi BTC
snapshot -> resolve -> equity path.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent


def _module_level_test_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]


class SuiteIsCollectibleTests(unittest.TestCase):
    def test_no_test_file_hides_tests_from_unittest(self):
        offenders = {}
        for path in sorted(_TESTS_DIR.glob("test_*.py")):
            found = _module_level_test_functions(path)
            if found:
                offenders[path.name] = found
        self.assertEqual(
            offenders,
            {},
            "These define module-level test functions, which "
            "`unittest discover` ignores. Move them into a "
            "unittest.TestCase subclass so CI runs them:\n"
            + "\n".join(f"  {f}: {', '.join(n)}" for f, n in offenders.items()),
        )

    def test_every_test_file_defines_at_least_one_test_case(self):
        """A file with no TestCase contributes nothing to CI at all."""
        empty = []
        for path in sorted(_TESTS_DIR.glob("test_*.py")):
            if path.name == Path(__file__).name:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            has_case = any(
                isinstance(node, ast.ClassDef)
                and any(
                    "TestCase" in ast.unparse(base) for base in node.bases
                )
                for node in ast.walk(tree)
            )
            if not has_case:
                empty.append(path.name)
        self.assertEqual(
            empty, [], f"No unittest.TestCase in: {', '.join(empty)}"
        )
