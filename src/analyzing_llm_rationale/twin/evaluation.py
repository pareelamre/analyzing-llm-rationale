"""Canonical, reproducible readiness summary for shadow outcomes."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence


def readiness_artifact(outcomes: Sequence[Mapping], *, code_hash: str, config_hash: str) -> Mapping[str, object]:
    payload = {"code_hash": code_hash, "config_hash": config_hash, "outcomes": list(outcomes)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return {**payload, "artifact_hash": hashlib.sha256(canonical.encode()).hexdigest(), "complete": bool(outcomes)}
