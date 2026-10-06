"""Chooses the Datastore implementation: Google Cloud or SQL.

``FORESEA_DATASTORE_BACKEND`` selects the backend:

* ``gcp`` (default) -- ``google.cloud.datastore``, unchanged behaviour.
* ``sql`` / ``sqlite`` -- the local SQLite shim in ``datastore_sql``.

The shim is SQLite-only today. Postgres is a documented follow-up (see
``deploy/vps/README.md``, "Known gaps"): the query shapes are identical, but the
dialect differences (``INSERT OR REPLACE`` vs ``ON CONFLICT``, ``?`` vs
``%s``, ``PRAGMA``) need a dialect layer that cannot be verified without a
live server, so it is deliberately not claimed here.

This module mirrors the ``google.cloud.datastore`` module surface (``Client``,
``Entity``, ``Key``, ``PropertyFilter``) so call sites can swap

    from google.cloud import datastore as _ds

for

    from analyzing_llm_rationale import datastore_backend as _ds

and keep every ``_ds.Entity(...)`` / ``_ds.Client()`` reference working
unchanged. That keeps the migration a one-line change per import rather than a
rewrite of ~40 call sites.
"""
from __future__ import annotations

import os
from typing import Any

__all__ = ["Client", "Entity", "Key", "PropertyFilter", "backend", "is_sql"]

_SQL_BACKENDS = {"sql", "sqlite"}


def backend() -> str:
    return os.environ.get("FORESEA_DATASTORE_BACKEND", "gcp").strip().lower() or "gcp"


def is_sql() -> bool:
    return backend() in _SQL_BACKENDS


def _sql_module():
    from analyzing_llm_rationale import datastore_sql

    return datastore_sql


def _gcp_module():
    from google.cloud import datastore

    return datastore


def Client(**kwargs: Any) -> Any:  # noqa: N802 - mirrors datastore.Client
    """Return a Datastore client for the configured backend."""
    if is_sql():
        return _sql_module().Client(**kwargs)
    return _gcp_module().Client(**kwargs)


def Entity(key: Any = None, exclude_from_indexes: Any = ()) -> Any:  # noqa: N802
    """Return an entity for the configured backend."""
    if is_sql():
        return _sql_module().Entity(key, exclude_from_indexes=exclude_from_indexes)
    return _gcp_module().Entity(key=key, exclude_from_indexes=exclude_from_indexes)


def Key(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    if is_sql():
        return _sql_module().Key(*args, **kwargs)
    return _gcp_module().Key(*args, **kwargs)


def PropertyFilter(name: str, op: str, value: Any) -> Any:  # noqa: N802
    if is_sql():
        return _sql_module().PropertyFilter(name, op, value)
    from google.cloud.datastore.query import PropertyFilter as _GcpPropertyFilter

    return _GcpPropertyFilter(name, op, value)
