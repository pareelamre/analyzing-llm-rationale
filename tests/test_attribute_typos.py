"""Guard: an attribute read that no assignment matches, next to one that does.

`feed_latest` read `self._base_url` while the attribute is `self.base_url`.
Ruff cannot see it -- attribute names are not names it resolves -- and the
test suite could not, because a bare `except Exception: pass` caught the
AttributeError and fell through to a fallback. The tool raised a confusing
error from a different line and had, as far as anyone could tell, never
worked.

Only flags a read whose near-identical twin IS assigned (differing solely by
leading underscores). A plain unknown attribute is usually inherited,
injected by a decorator, or set on an instance elsewhere; a name that shadows
an assigned one is a typo.
"""
from __future__ import annotations

import ast
import collections
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# scripts/ too: it holds build_agent_trading_board.py and
# agent_trading_tick.py, which publish the board and run the trades. Both are
# production code by any measure, and CI did not lint them at all until #502.
_SCANNED = (_ROOT / "src" / "analyzing_llm_rationale", _ROOT / "scripts")


def _typo_reads(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError:  # pragma: no cover - a file ruff would already reject
        return []
    out: list[str] = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        assigned: set[str] = set()
        read: collections.Counter = collections.Counter()
        for node in ast.walk(cls):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    assigned.add(node.attr)
                else:
                    read[node.attr] += 1
        methods = {
            m.name
            for m in ast.walk(cls)
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for attr in read:
            if attr in assigned or attr in methods:
                continue
            twin = [a for a in assigned if a.strip("_") == attr.strip("_") and a != attr]
            if twin:
                out.append(
                    f"{path.name}::{cls.name}: reads self.{attr} but assigns self.{twin[0]}"
                )
    return out


class AttributeTypoTests(unittest.TestCase):
    def test_no_class_reads_a_near_miss_of_an_attribute_it_assigns(self):
        found: list[str] = []
        for directory in _SCANNED:
            for path in sorted(directory.glob("*.py")):
                found.extend(_typo_reads(path))
        self.assertEqual(
            found,
            [],
            "These read an attribute that is never assigned, while a name "
            "differing only in underscores is:\n  " + "\n  ".join(found),
        )
