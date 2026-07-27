"""pytest / unittest conftest — runs before any test module is imported.

Setting OTEL_SDK_DISABLED=true prevents test suites that import server.py from
accidentally initialising the real OTel exporters and shipping spans/logs/metrics
to the production Superlog backend.
"""
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
