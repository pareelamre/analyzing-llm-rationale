"""The visitor hashes pseudonymise nothing while the salt is published.

_visitor_id and _visitor_hash store sha256("{ip}:{ua}:{salt}") rather than
the address itself -- the docstring says "no raw IP stored". That holds
only while the salt is secret, and ANALYTICS_SALT is unset on the service,
so the salt is the one written in the source.

With a known salt, asking "did this address visit?" costs a single hash.
Recovering an unknown address costs a scan of the IPv4 space: measured at
503,791 hashes/sec on one CPU core, so 2.4 hours there and about half a
second on a commodity GPU at ~10 GH/s.

This warns. It does not change the salt: doing so changes every hash and
restarts cumulative unique-visitor counts, which is a decision about a
metric rather than something to change underneath one.
"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import server as server_module  # noqa: E402


class WarningTests(unittest.TestCase):
    def test_it_warns_when_the_salt_is_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "ANALYTICS_SALT"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertLogs(server_module.logger, level="WARNING") as caught:
                server_module._warn_if_the_analytics_salt_is_the_published_one()
        message = "\n".join(caught.output)
        self.assertIn("ANALYTICS_SALT", message)
        self.assertIn("pseudonymise", message)

    def test_the_warning_says_the_counts_will_reset(self):
        """Someone acting on this should know the cost before they act."""
        env = {k: v for k, v in os.environ.items() if k != "ANALYTICS_SALT"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertLogs(server_module.logger, level="WARNING") as caught:
                server_module._warn_if_the_analytics_salt_is_the_published_one()
        self.assertIn("reset", "\n".join(caught.output))

    def test_it_says_nothing_once_a_salt_is_configured(self):
        with mock.patch.dict(os.environ, {"ANALYTICS_SALT": "something-private"}, clear=False):
            with mock.patch.object(server_module.logger, "warning") as warn:
                server_module._warn_if_the_analytics_salt_is_the_published_one()
        warn.assert_not_called()

    def test_it_is_wired_into_startup(self):
        import ast

        source = Path(server_module.__file__).read_text(encoding="utf-8", errors="replace")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
                called = {
                    n.func.id for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
                self.assertIn("_warn_if_the_analytics_salt_is_the_published_one", called)
                return
        self.fail("lifespan not found")


class WhyItMattersTests(unittest.TestCase):
    def test_a_known_salt_confirms_a_visitor_in_one_hash(self):
        """The hash is only as private as the salt."""
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        salt = server_module._DEFAULT_ANALYTICS_SALT
        stored = hashlib.sha256(f"198.51.100.7:{ua}:{salt}".encode()).hexdigest()

        guess = hashlib.sha256(f"198.51.100.7:{ua}:{salt}".encode()).hexdigest()
        self.assertEqual(guess, stored)

        other = hashlib.sha256(f"198.51.100.8:{ua}:{salt}".encode()).hexdigest()
        self.assertNotEqual(other, stored)

    def test_a_private_salt_defeats_the_same_check(self):
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        stored = hashlib.sha256(f"198.51.100.7:{ua}:a-private-salt".encode()).hexdigest()
        guess = hashlib.sha256(
            f"198.51.100.7:{ua}:{server_module._DEFAULT_ANALYTICS_SALT}".encode()
        ).hexdigest()
        self.assertNotEqual(guess, stored)


class TheDefaultIsNamedOnceTests(unittest.TestCase):
    def test_no_call_site_repeats_the_literal(self):
        """Two copies drift; the warning would then describe one of them."""
        source = Path(server_module.__file__).read_text(encoding="utf-8", errors="replace")
        self.assertEqual(source.count('"foresea-analytics"'), 1)

    def test_the_visitor_helpers_use_the_named_default(self):
        source = Path(server_module.__file__).read_text(encoding="utf-8", errors="replace")
        self.assertEqual(
            source.count('os.environ.get("ANALYTICS_SALT", _DEFAULT_ANALYTICS_SALT)'), 2,
        )


if __name__ == "__main__":
    unittest.main()
