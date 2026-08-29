from __future__ import annotations

import os
import sys
from pathlib import Path

# Prevent the OTel exporter from sending test-generated telemetry to the
# production Superlog endpoint.  observability.init_observability() checks
# this variable before installing any provider (see observability.py:39).
os.environ.setdefault("ENABLE_OTEL", "0")

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    _src = str(_SRC)
    if _src in sys.path:
        sys.path.remove(_src)
    sys.path.insert(0, _src)
