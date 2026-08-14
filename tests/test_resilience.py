"""Tests for the production-hardening layer: provider retry/backoff, scrubbed
error mapping, security headers, body-size guard, request IDs, and readiness."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ["ANALYTICS_DB"] = str(Path(tempfile.gettempdir()) / "foresea_test_resilience.duckdb")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale import server  # noqa: E402
from analyzing_llm_rationale.providers import (  # noqa: E402
    ContextLimitError,
    ProviderResponseError,
    RetryableProviderError,
)
from analyzing_llm_rationale.server import (  # noqa: E402
    _ANALYTICS_DB,
    _local_cache,
    _provider_chat,
    _provider_http_error,
    _rate_limiter,
    _state,
    app,
)


class FlakyProvider:
    """Fails with ``error`` for the first ``fail_times`` calls, then succeeds."""

    def __init__(self, fail_times, error, ok='{"predicted_answer": "Yes", "confidence": 0.6, "rationale": "ok"}'):
        self.fail_times = fail_times
        self.error = error
        self.ok = ok
        self.calls = 0

    def chat_completion(self, messages, temperature, max_tokens):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return self.ok


class ProviderRetryUnitTests(unittest.TestCase):
    def setUp(self):
        # No sleeping in tests.
        self._patches = [
            mock.patch.object(server, "_PROVIDER_BACKOFF_BASE_S", 0.0),
            mock.patch.object(server, "_PROVIDER_MAX_RETRIES", 2),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_retries_then_succeeds(self):
        provider = FlakyProvider(fail_times=2, error=RetryableProviderError("503"))
        out = asyncio.run(_provider_chat(provider, [], 0.0, 64))
        self.assertIn("predicted_answer", out)
        self.assertEqual(provider.calls, 3)  # 2 failures + 1 success

    def test_retries_exhausted_raises_retryable(self):
        provider = FlakyProvider(fail_times=99, error=RetryableProviderError("503"))
        with self.assertRaises(RetryableProviderError):
            asyncio.run(_provider_chat(provider, [], 0.0, 64))
        self.assertEqual(provider.calls, 3)  # retries + 1, no more

    def test_context_limit_not_retried(self):
        provider = FlakyProvider(fail_times=99, error=ContextLimitError("too long"))
        with self.assertRaises(ContextLimitError):
            asyncio.run(_provider_chat(provider, [], 0.0, 64))
        self.assertEqual(provider.calls, 1)  # raised immediately, never retried

    def test_nonretryable_propagates_immediately(self):
        provider = FlakyProvider(fail_times=99, error=ProviderResponseError("bad"))
        with self.assertRaises(ProviderResponseError):
            asyncio.run(_provider_chat(provider, [], 0.0, 64))
        self.assertEqual(provider.calls, 1)

    def test_call_can_disable_retries(self):
        provider = FlakyProvider(fail_times=99, error=RetryableProviderError("503"))
        with self.assertRaises(RetryableProviderError):
            asyncio.run(_provider_chat(provider, [], 0.0, 64, max_retries=0))
        self.assertEqual(provider.calls, 1)

    def test_call_can_override_timeout(self):
        class SlowProvider:
            model_name = "slow"

            def chat_completion(self, messages, temperature, max_tokens):
                time.sleep(0.05)
                return "{}"

        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(
                _provider_chat(
                    SlowProvider(),
                    [],
                    0.0,
                    64,
                    timeout_s=0.01,
                    max_retries=0,
                )
            )


class ProviderErrorMappingTests(unittest.TestCase):
    def test_context_limit_maps_to_422(self):
        self.assertEqual(_provider_http_error(ContextLimitError("x")).status_code, 422)

    def test_retryable_maps_to_503_with_retry_after(self):
        exc = _provider_http_error(RetryableProviderError("x"))
        self.assertEqual(exc.status_code, 503)
        self.assertEqual(exc.headers.get("Retry-After"), "10")

    def test_timeout_maps_to_503(self):
        self.assertEqual(_provider_http_error(asyncio.TimeoutError()).status_code, 503)

    def test_other_maps_to_502(self):
        self.assertEqual(_provider_http_error(ProviderResponseError("x")).status_code, 502)

    def test_detail_never_leaks_raw_exception_text(self):
        secret = "secret-upstream-url-and-key"
        for exc in (RetryableProviderError(secret), ProviderResponseError(secret), ContextLimitError(secret)):
            self.assertNotIn(secret, _provider_http_error(exc).detail)


def _configure_state(provider):
    _state.clear()
    _state.update({
        "provider": provider,
        "evidence_pipeline": None,
        "variants": {
            "variant0_neutral_baseline": SimpleNamespace(
                output_fields=("predicted_answer", "confidence", "rationale")
            )
        },
        "system_prompt": "System",
        "prompt_templates": {"variant0_neutral_baseline": "[question]\nReturn JSON."},
        "temperature": 0.0,
        "max_tokens": 256,
        "model_key": "test-model",
    })


class MiddlewareAndProbeTests(unittest.TestCase):
    def setUp(self):
        _local_cache.clear()
        _rate_limiter._log.clear()
        self._require_auth_patch = mock.patch.object(
            server,
            "_require_auth",
            return_value={"sub": "test-user", "email": "test@example.com", "name": "Test User"}
        )
        self._require_auth_patch.start()
        _configure_state(FlakyProvider(fail_times=0, error=RetryableProviderError("x")))
        self.client = TestClient(app)

    def tearDown(self):
        self._require_auth_patch.stop()
        _state.clear()
        _local_cache.clear()
        if _ANALYTICS_DB.exists():
            _ANALYTICS_DB.unlink()

    def test_security_headers_present(self):
        r = self.client.get("/health")
        self.assertEqual(r.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(r.headers.get("X-Frame-Options"), "DENY")
        self.assertIn("max-age=", r.headers.get("Strict-Transport-Security", ""))
        self.assertIn("default-src", r.headers.get("Content-Security-Policy", ""))

    def test_request_id_generated_and_echoed(self):
        r = self.client.get("/health")
        self.assertTrue(r.headers.get("X-Request-ID"))

        r2 = self.client.get("/health", headers={"X-Request-ID": "abc123"})
        self.assertEqual(r2.headers.get("X-Request-ID"), "abc123")

    def test_oversized_body_rejected(self):
        with mock.patch.object(server, "_MAX_BODY_BYTES", 10):
            r = self.client.post("/predict", json={"question": "Will this very long body be rejected?"})
        self.assertEqual(r.status_code, 413)

    def test_oversized_chunked_body_rejected(self):
        with mock.patch.object(server, "_MAX_BODY_BYTES", 10):
            r = self.client.post(
                "/predict",
                content=b"x" * 11,
                headers={"Transfer-Encoding": "chunked"},
            )
        self.assertEqual(r.status_code, 413)

    def test_ready_ok_when_provider_configured(self):
        r = self.client.get("/ready")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ready"])

    def test_ready_503_when_provider_missing(self):
        _state.pop("provider", None)
        r = self.client.get("/ready")
        self.assertEqual(r.status_code, 503)
        self.assertFalse(r.json()["ready"])

    def test_predict_retries_transient_failure(self):
        # First call 503s, retry succeeds → user sees a clean 200.
        with mock.patch.object(server, "_PROVIDER_BACKOFF_BASE_S", 0.0):
            _configure_state(FlakyProvider(fail_times=1, error=RetryableProviderError("503")))
            r = self.client.post(
                "/predict",
                json={
                    "question": "Will the retry path recover?",
                    "variant": "variant0_neutral_baseline",
                    "attach_evidence": False,
                    "chat_mode": False,
                },
            )
        self.assertEqual(r.status_code, 200)

    def test_predict_exhausted_returns_clean_503(self):
        with mock.patch.object(server, "_PROVIDER_BACKOFF_BASE_S", 0.0):
            _configure_state(FlakyProvider(fail_times=99, error=RetryableProviderError("503")))
            r = self.client.post(
                "/predict",
                json={
                    "question": "Will the model stay down forever?",
                    "variant": "variant0_neutral_baseline",
                    "attach_evidence": False,
                    "chat_mode": False,
                },
            )
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("503", r.json()["detail"])  # raw upstream text not leaked


if __name__ == "__main__":
    unittest.main()
