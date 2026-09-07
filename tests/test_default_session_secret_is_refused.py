"""A deployment must not sign sessions with the published placeholder.

_SESSION_SECRET falls back to "change-me-in-production". decode_session
verifies an HS256 signature against it, so with the placeholder in force
anyone who has read the source can sign a token carrying any ``sub`` and
be that user. It is not a weak secret, it is a published one.

The deployed service does set SESSION_SECRET -- checked against the Cloud
Run configuration on 2026-09-07 -- so this changes nothing about how it
runs today. It exists so a deployment that forgets fails at startup rather
than serving forgeable sessions quietly.

The demonstration below signs a token with the placeholder and decodes it
through the real code path, against a throwaway secret in this process
only. It touches nothing deployed.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import server as server_module  # noqa: E402
from analyzing_llm_rationale import server_security  # noqa: E402


class RefusalTests(unittest.TestCase):
    def test_it_refuses_on_cloud_run(self):
        with (
            mock.patch.object(server_module, "_SESSION_SECRET",
                              server_module._DEFAULT_SESSION_SECRET),
            mock.patch.dict(os.environ, {"K_SERVICE": "analyzing-llm-rationale"}, clear=False),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                server_module._refuse_to_serve_with_the_default_session_secret()
        self.assertIn("SESSION_SECRET", str(ctx.exception))

    def test_it_only_warns_off_cloud_run(self):
        env = {k: v for k, v in os.environ.items() if k != "K_SERVICE"}
        with (
            mock.patch.object(server_module, "_SESSION_SECRET",
                              server_module._DEFAULT_SESSION_SECRET),
            mock.patch.dict(os.environ, env, clear=True),
        ):
            with self.assertLogs(server_module.logger, level="WARNING") as caught:
                server_module._refuse_to_serve_with_the_default_session_secret()
        self.assertIn("placeholder", "\n".join(caught.output))

    def test_a_real_secret_passes_anywhere(self):
        for env in ({"K_SERVICE": "svc"}, {}):
            with self.subTest(env=env):
                with (
                    mock.patch.object(server_module, "_SESSION_SECRET", "a-real-secret"),
                    mock.patch.dict(os.environ, env, clear=False),
                ):
                    server_module._refuse_to_serve_with_the_default_session_secret()

    def test_it_is_wired_into_startup(self):
        import ast

        source = (Path(server_module.__file__)).read_text(encoding="utf-8", errors="replace")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
                called = {
                    n.func.id for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
                self.assertIn(
                    "_refuse_to_serve_with_the_default_session_secret", called,
                )
                return
        self.fail("lifespan not found")


class WhyItMattersTests(unittest.TestCase):
    """The placeholder is a signing key, not a password."""

    def test_a_token_signed_with_the_placeholder_is_accepted_by_the_real_decoder(self):
        import jwt

        forged = jwt.encode(
            {"sub": "somebody-elses-account", "email": "them@example.com"},
            server_module._DEFAULT_SESSION_SECRET,
            algorithm="HS256",
        )
        claims = server_security.decode_session(
            forged, server_module._DEFAULT_SESSION_SECRET,
        )
        self.assertEqual(claims["sub"], "somebody-elses-account")

    def test_the_same_token_is_rejected_under_a_real_secret(self):
        import jwt
        from fastapi import HTTPException

        forged = jwt.encode(
            {"sub": "somebody-elses-account"},
            server_module._DEFAULT_SESSION_SECRET,
            algorithm="HS256",
        )
        with self.assertRaises(HTTPException) as ctx:
            server_security.decode_session(forged, "a-real-secret")
        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
