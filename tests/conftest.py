"""Pytest session-level configuration.

Set ENABLE_OTEL=0 before any test file imports ``server`` (or any module that
imports it), so the OTel SDK never installs OTLP exporters during the test run.
Without this guard, ``init_observability(app)`` is called at module-import time
and every TestClient HTTP span is shipped to production Superlog, creating
false-positive incidents.
"""
import os

os.environ.setdefault("ENABLE_OTEL", "0")
