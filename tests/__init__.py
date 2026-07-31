from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists():
    _src = str(_SRC)
    if _src in sys.path:
        sys.path.remove(_src)
    sys.path.insert(0, _src)
