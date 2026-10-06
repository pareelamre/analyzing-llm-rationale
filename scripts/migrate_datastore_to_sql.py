"""Copy every entity out of Cloud Datastore into the SQL shim.

Run this once, from a machine with GCP credentials, before cutting traffic over
to the new VPS:

    FORESEA_DATASTORE_BACKEND=gcp python scripts/migrate_datastore_to_sql.py \
        --project brave-drive-471109-d9 \
        --output foresea.sqlite3

Then verify the copy is complete:

    python scripts/migrate_datastore_to_sql.py --verify \
        --project brave-drive-471109-d9 --output foresea.sqlite3

The script reads through the *real* Datastore client and writes through the
*shim*, so it exercises the same code path the app will use in production. It
is read-only against Datastore: nothing is deleted or modified there, so it is
safe to re-run and safe to run while the old deployment is still serving.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import datastore_sql  # noqa: E402


def _gcp_client(project: str):
    from google.cloud import datastore

    credentials, _ = _default_credentials()
    return datastore.Client(project=project, credentials=credentials)


def _default_credentials():
    """Application Default Credentials, else the active gcloud user's token.

    The fallback exists so the migration can run from a workstation that is
    authenticated with ``gcloud auth login`` but has not run
    ``gcloud auth application-default login``. It reuses that same
    already-authorized session instead of requiring a new credential file.

    The token is short-lived (~1 hour). The copy is an upsert, so if it
    expires mid-run the fix is simply to re-run the command.
    """
    import google.auth

    try:
        return google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        pass

    import shutil
    import subprocess

    from google.oauth2.credentials import Credentials

    # On Windows gcloud is a .cmd/.ps1 shim, so a bare "gcloud" is not
    # executable by subprocess without a shell.
    executable = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if executable is None:
        raise RuntimeError(
            "No credentials found and the gcloud CLI is not on PATH. Run "
            "`gcloud auth application-default login` to set up ADC."
        )
    result = subprocess.run(
        [executable, "auth", "print-access-token"],
        capture_output=True, text=True, check=False,
    )
    token = result.stdout.strip()
    if result.returncode != 0 or not token:
        raise RuntimeError(
            "No credentials found. Run `gcloud auth application-default login`, "
            "or `gcloud auth login` to use the access-token fallback."
        )
    return Credentials(token), None


def _kinds(client) -> list[str]:
    """Discover every application kind present.

    Datastore exposes its own metadata as the ``__kind__`` kind; querying it
    keys-only is the documented way to enumerate kinds without scanning data.
    Names beginning with ``__`` are Datastore internals (``__Stat_*`` index
    statistics, ``__namespace__``), not application data, so they are excluded
    -- copying them would create junk tables and inflate the verify counts.
    """
    query = client.query(kind="__kind__")
    query.keys_only()
    return sorted({
        entity.key.name for entity in query.fetch()
        if not entity.key.name.startswith("__")
    })


def _namespaces(client) -> list[str | None]:
    query = client.query(kind="__namespace__")
    query.keys_only()
    names = {entity.key.name for entity in query.fetch()}
    # Datastore reports the default namespace as an empty name.
    return sorted(names, key=lambda n: (n is not None, n or ""))

def migrate(project: str, output: str, *, batch: int = 500) -> int:
    source = _gcp_client(project)
    target = datastore_sql.Client(path=output)
    total = 0

    for namespace in _namespaces(source):
        ns = namespace or None
        for kind in _kinds(source):
            query = source.query(kind=kind, namespace=ns)
            copied = 0
            for entity in query.fetch():
                # Rebuild the key through the shim from the source key's path.
                # Datastore's key.path is a list of {"kind", "name"|"id"} dicts
                # (named keys carry "name", auto-allocated keys carry "id"),
                # so both have to be read to avoid losing the identifier.
                flat: list[str] = []
                for element in entity.key.path:
                    ident = element.get("name")
                    if ident is None:
                        ident = element.get("id")
                    flat.extend([element["kind"], str(ident)])
                key = target.key(*flat, namespace=ns)
                copy = datastore_sql.Entity(key)
                copy.update(dict(entity))
                target.put(copy)
                copied += 1
                total += 1
                if copied % batch == 0:
                    print(f"  {kind}: {copied}...", flush=True)
            if copied:
                print(f"{kind} (namespace={ns or 'default'}): {copied} entities")

    target.close()
    print(f"\nCopied {total} entities into {output}")
    return total


def verify(project: str, output: str) -> int:
    """Compare per-kind counts between Datastore and the SQL copy."""
    source = _gcp_client(project)
    target = datastore_sql.Client(path=output)
    mismatches = 0

    for namespace in _namespaces(source):
        ns = namespace or None
        for kind in _kinds(source):
            expected = len(list(source.query(kind=kind, namespace=ns).fetch()))
            actual = len(list(target.query(kind=kind, namespace=ns).fetch()))
            status = "ok" if expected == actual else "MISMATCH"
            if expected != actual:
                mismatches += 1
            if expected or actual:
                print(f"{status:9} {kind} (namespace={ns or 'default'}): "
                      f"datastore={expected} sql={actual}")

    target.close()
    if mismatches:
        print(f"\n{mismatches} kind(s) do not match -- do not cut over yet.")
        return 1
    print("\nAll kinds match.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="GCP project id")
    parser.add_argument("--output", required=True, help="SQLite file to write")
    parser.add_argument("--verify", action="store_true",
                        help="compare counts instead of copying")
    args = parser.parse_args()

    if args.verify:
        return verify(args.project, args.output)
    migrate(args.project, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
