"""Rewrap legacy exchange connections with KMS-backed per-record data keys.

Run with Cloud Run's service-account credentials (or equivalent ADC) after setting
both ``FORESEA_TRADING_KMS_KEY_NAME`` and the retiring
``FORESEA_CREDENTIALS_ENCRYPTION_KEY``. The default is dry-run and never prints
connection metadata beyond counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from analyzing_llm_rationale import server  # noqa: E402


def _user_id_for(entity: Any) -> str | None:
    parent = getattr(getattr(entity, "key", None), "parent", None)
    if parent is None or getattr(parent, "kind", None) != "User":
        return None
    return getattr(parent, "name", None)


def run(*, apply: bool, limit: int | None) -> int:
    mode = "apply" if apply else "dry_run"
    with server._tracer.start_as_current_span("trading.connection.migrate") as span:
        span.set_attribute("trading.migration.mode", mode)
        if limit is not None:
            span.set_attribute("trading.migration.limit", limit)
        server.logger.info("starting trading connection encryption migration mode=%s", mode)
        client = server._get_datastore()
        if client is None:
            span.set_attribute("outcome", "failure")
            print("Datastore is unavailable; migration did not run.", file=sys.stderr)
            return 2
        query = client.query(kind=server._TRADING_CONNECTION_KIND)
        records = list(query.fetch(limit=limit))
        legacy_records = [
            entity
            for entity in records
            if int(dict(entity).get("credential_version") or 1) == 1
        ]
        if apply and legacy_records:
            try:
                server._trading_kms_key_name()
                server._get_trading_kms_client()
                server._legacy_trading_fernet()
            except server.SecureTradingConnectionError as exc:
                span.record_exception(exc)
                span.set_attribute("outcome", "failure")
                print(f"Migration configuration error: {exc}", file=sys.stderr)
                return 2

        candidates = migrated = skipped = invalid = 0
        for entity in records:
            record = dict(entity)
            version = int(record.get("credential_version") or 1)
            if version == server._TRADING_CONNECTION_ENVELOPE_VERSION:
                skipped += 1
                continue
            if version != 1:
                invalid += 1
                continue
            candidates += 1
            if not apply:
                continue
            user_id = _user_id_for(entity)
            platform = str(record.get("platform") or "").strip().lower()
            if not user_id or platform not in {"kalshi", "polymarket"}:
                invalid += 1
                continue
            try:
                server._stored_trading_credentials(user_id, platform)
                migrated += 1
                server._trading_connection_actions.add(
                    1, {"venue": platform, "action": "migrate", "outcome": "success"}
                )
            except server.SecureTradingConnectionError as exc:
                span.record_exception(exc)
                invalid += 1
                server._trading_connection_actions.add(
                    1, {"venue": platform, "action": "migrate", "outcome": "error"}
                )

        span.set_attribute("trading.migration.candidates", candidates)
        span.set_attribute("trading.migration.migrated", migrated)
        span.set_attribute("trading.migration.invalid", invalid)
        span.set_attribute("outcome", "success" if invalid == 0 else "failure")
        result_mode = "Migrated" if apply else "Would migrate"
        print(
            f"{result_mode}: {migrated if apply else candidates}; "
            f"already KMS: {skipped}; invalid: {invalid}."
        )
        server.logger.info(
            "completed trading connection encryption migration mode=%s candidates=%s migrated=%s invalid=%s",
            mode,
            candidates,
            migrated,
            invalid,
        )
        return 0 if invalid == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform the rewrap; default is dry-run.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum connection records to inspect.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return run(apply=args.apply, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
