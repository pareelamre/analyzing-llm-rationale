"""Say at startup when the API key guard is checking nothing.

``check_api_key`` returns without checking when no key is configured:

    if not required_api_key:
        return

That is a fine default for a service with nothing to protect. This service
has /analytics/users, which returns every registered account with its email
address and last login, and on 2026-09-07 that endpoint answered an
unauthenticated request with all three of them, because API_KEY was unset on
the Cloud Run service.

Nothing said so. The endpoint calls _check_api_key and reads as protected;
the guard was simply inert. This makes the state visible at startup rather
than leaving it to be discovered by request.

It does not change who can reach anything -- closing the endpoints is a
deployment decision, not one to take by merging.
"""

from __future__ import annotations

import ast
import logging
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analyzing_llm_rationale import server as server_module  # noqa: E402

_SOURCE = (_ROOT / "src" / "analyzing_llm_rationale" / "server.py").read_text(
    encoding="utf-8", errors="replace",
)


class WarningTests(unittest.TestCase):
    def test_it_warns_when_no_key_is_configured(self):
        with mock.patch.object(server_module, "_REQUIRED_API_KEY", None):
            with self.assertLogs(server_module.logger, level=logging.WARNING) as caught:
                server_module._warn_if_api_key_guard_is_inert()
        message = "\n".join(caught.output)
        self.assertIn("API_KEY is not set", message)
        self.assertIn("/analytics/users", message)

    def test_the_warning_names_the_personal_data_endpoints_separately(self):
        """An open traffic counter and an open list of accounts differ."""
        with mock.patch.object(server_module, "_REQUIRED_API_KEY", None):
            with self.assertLogs(server_module.logger, level=logging.WARNING) as caught:
                server_module._warn_if_api_key_guard_is_inert()
        self.assertIn("personal data", "\n".join(caught.output))

    def test_it_says_nothing_when_a_key_is_configured(self):
        with mock.patch.object(server_module, "_REQUIRED_API_KEY", "a-key"):
            with mock.patch.object(server_module.logger, "warning") as warn:
                server_module._warn_if_api_key_guard_is_inert()
        warn.assert_not_called()

    def test_the_warning_does_not_contain_the_key(self):
        """A warning about a secret must not print the secret."""
        with mock.patch.object(server_module, "_REQUIRED_API_KEY", None):
            with self.assertLogs(server_module.logger, level=logging.WARNING) as caught:
                server_module._warn_if_api_key_guard_is_inert()
        self.assertNotIn("a-key", "\n".join(caught.output))


class TheListedEndpointsAreTheGuardedOnesTests(unittest.TestCase):
    """The list is written by hand; the truth is in the call sites.

    If an endpoint starts or stops relying on _check_api_key and the list is
    not updated, the warning names the wrong set -- which is worse than no
    warning, because it reads as a survey.
    """

    def _endpoints_calling_the_guard(self):
        routes = [
            (m.start(), m.group(1))
            for m in re.finditer(r'@app\.(?:get|post|put|delete)\(\s*\n?\s*"([^"]+)"', _SOURCE)
        ]
        found = set()
        for call in re.finditer(r"_check_api_key\(request\)", _SOURCE):
            before = [r for r in routes if r[0] < call.start()]
            if before:
                found.add(before[-1][1])
        return found

    def test_the_scan_found_call_sites(self):
        """Guard the regex: an empty scan must not pass silently."""
        self.assertGreaterEqual(len(self._endpoints_calling_the_guard()), 5)

    def test_the_list_matches_the_call_sites(self):
        self.assertEqual(
            set(server_module._API_KEY_GUARDED_ENDPOINTS),
            self._endpoints_calling_the_guard(),
        )

    def test_the_personal_data_subset_is_a_subset(self):
        self.assertTrue(
            set(server_module._API_KEY_GUARDED_PERSONAL_DATA)
            <= set(server_module._API_KEY_GUARDED_ENDPOINTS)
        )


class TheGuardStillFailsOpenTests(unittest.TestCase):
    """State the behaviour plainly, so changing it is deliberate.

    This is not an endorsement. It records what the code does today, so that
    making it fail closed shows up as a test that has to change rather than
    as a silent shift in who can read what.
    """

    def test_no_key_configured_admits_everyone(self):
        from analyzing_llm_rationale import server_security

        request = mock.Mock()
        request.headers = {}
        server_security.check_api_key(request, None)  # must not raise

    def test_a_configured_key_is_enforced(self):
        from fastapi import HTTPException

        from analyzing_llm_rationale import server_security

        request = mock.Mock()
        request.headers = {}
        with self.assertRaises(HTTPException) as ctx:
            server_security.check_api_key(request, "expected")
        self.assertEqual(ctx.exception.status_code, 401)

        request.headers = {"X-API-Key": "expected"}
        server_security.check_api_key(request, "expected")  # must not raise


class TheWarningIsWiredIntoStartupTests(unittest.TestCase):
    def test_lifespan_calls_it(self):
        """A warning nothing invokes is not a warning."""
        tree = ast.parse(_SOURCE)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
                called = {
                    n.func.id for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
                self.assertIn("_warn_if_api_key_guard_is_inert", called)
                return
        self.fail("lifespan not found")


if __name__ == "__main__":
    unittest.main()
