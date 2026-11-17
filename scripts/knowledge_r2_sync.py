#!/usr/bin/env python3
"""Copy the validated knowledge base to private Cloudflare R2 for the live /predict server.

    python scripts/knowledge_r2_sync.py STAGED_DIR

Packs ``STAGED_DIR`` (``entities.jsonl`` + ``sources/``) into one deterministic archive, uploads
it to ``$R2_STATE_BUCKET/knowledge/knowledge.tar.gz`` with the workflow's read/write key only when
its content changed, then reads it back with the server's read-only key, so a key the server
cannot use fails here and not silently on foresea.ink. Prints sizes and counts only: the base
holds licensed reports and the workflow log is public.

Environment: R2_ENDPOINT_URL, R2_STATE_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (writer),
R2_RUNTIME_ACCESS_KEY_ID, R2_RUNTIME_SECRET_ACCESS_KEY (the server's read-only key).
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402

_ENDPOINT = re.compile(r"https://[a-f0-9]{32}\.r2\.cloudflarestorage\.com")
_MISSING = {"404", "NoSuchKey", "NotFound"}


def pack(root: Path) -> bytes:
    """A gzipped tar of every file under ``root``; the same files always give the same bytes."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                info = tarfile.TarInfo(path.relative_to(root).as_posix())
                info.size, info.mtime, info.mode = path.stat().st_size, 0, 0o644
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
    return buf.getvalue()


def _remote_md5(client, bucket: str, key: str):
    try:
        return str(client.head_object(Bucket=bucket, Key=key).get("ETag") or "").strip('"')
    except Exception as exc:  # botocore ClientError; checked by code so tests need no botocore
        if str(getattr(exc, "response", {}).get("Error", {}).get("Code")) in _MISSING:
            return None
        raise


def sync(data: bytes, *, writer, reader, bucket: str, key: str = kb.REMOTE_KEY) -> str:
    """Upload ``data`` when it differs from the stored copy; prove the reader gets it intact."""
    if not key.startswith("knowledge/"):
        raise ValueError(f"refusing to write outside knowledge/: {key}")
    digest = hashlib.md5(data).digest()
    status = "unchanged"
    if _remote_md5(writer, bucket, key) != digest.hex():
        writer.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/gzip",
                          ContentMD5=base64.b64encode(digest).decode())
        status = "uploaded"
    got = reader.get_object(Bucket=bucket, Key=key)["Body"].read()
    if hashlib.md5(got).digest() != digest:
        raise RuntimeError("the server's read-only key read back a different archive")
    return status


def _client(access: str, secret: str):
    import boto3
    endpoint = os.environ.get("R2_ENDPOINT_URL", "")
    if not _ENDPOINT.fullmatch(endpoint):
        raise ValueError("R2_ENDPOINT_URL must be the account's HTTPS R2 endpoint")
    if not access or not secret:
        raise ValueError("R2 credentials are missing")
    return boto3.client("s3", endpoint_url=endpoint, region_name="auto",
                        aws_access_key_id=access, aws_secret_access_key=secret)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    staged = Path(argv[0])
    data = pack(staged)
    with tempfile.TemporaryDirectory() as tmp:  # the archive must open as the server opens it
        kb.unpack(data, Path(tmp))
        corpus = kb.load(Path(tmp))
    if corpus.errors:
        print(f"::error title=Knowledge archive invalid::{len(corpus.errors)} problem(s) after packing. Not synced.")
        return 1
    bucket = os.environ.get("R2_STATE_BUCKET", "").strip()
    if not bucket:
        raise ValueError("R2_STATE_BUCKET is required")
    writer = _client(os.environ.get("R2_ACCESS_KEY_ID", ""), os.environ.get("R2_SECRET_ACCESS_KEY", ""))
    reader = _client(os.environ.get("R2_RUNTIME_ACCESS_KEY_ID", ""),
                     os.environ.get("R2_RUNTIME_SECRET_ACCESS_KEY", ""))
    status = sync(data, writer=writer, reader=reader, bucket=bucket)
    print(f"Knowledge archive {status}: {len(data)} bytes, {len(corpus.sources)} sources, "
          f"{len(corpus.claims)} claims; the server's read-only key reads it back intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
