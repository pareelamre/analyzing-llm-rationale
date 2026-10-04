"""Private R2 state storage; GCS remains the default until explicit cutover.

Workflow entry point: PYTHONPATH=src python -m analyzing_llm_rationale.r2_store cp SOURCE DEST
Only the state bucket is remapped. Dashboard payloads and backups have separate migrations.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

from botocore.exceptions import ClientError
from opentelemetry import metrics, trace

logger = logging.getLogger("foresea")
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
transferred = meter.create_counter("storage.transfer.bytes", unit="By")


def get_client():
    endpoint = os.environ.get("R2_ENDPOINT_URL", "")
    if not re.fullmatch(r"https://[a-f0-9]{32}\.r2\.cloudflarestorage\.com", endpoint):
        raise ValueError("R2_ENDPOINT_URL must be the account's HTTPS R2 endpoint")
    access = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    if not access or not secret:
        raise ValueError("Both R2 credential environment variables are required")
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3", endpoint_url=endpoint, region_name="auto",
        aws_access_key_id=access, aws_secret_access_key=secret,
        config=Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 3}),
    )


class R2Blob:
    """Small adapter preserving the existing generation-aware runtime reader."""

    def __init__(self, client, bucket: str, key: str):
        self.client, self.bucket, self.key = client, bucket, key
        self.generation = None
        self.size = None

    def reload(self):
        metadata = self.client.head_object(Bucket=self.bucket, Key=self.key)
        self.generation = metadata["ETag"]
        self.size = metadata["ContentLength"]

    @tracer.start_as_current_span("storage.r2.download")
    def download_to_filename(self, filename: str):
        if self.generation is None:
            self.reload()
        response = self.client.get_object(
            Bucket=self.bucket, Key=self.key, IfMatch=self.generation,
        )
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with closing(response["Body"]) as body:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                    temporary = Path(output.name)
                    while chunk := body.read(1024 * 1024):
                        output.write(chunk)
            size = temporary.stat().st_size
            if size != self.size or size != response["ContentLength"]:
                raise ValueError("R2 download length does not match metadata")
            temporary.replace(target)
            transferred.add(size, {"backend": "r2", "direction": "download"})
            trace.get_current_span().set_attribute("payload.bytes", size)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class R2Bucket:
    def __init__(self, client, name: str):
        self.client, self.name = client, name

    def blob(self, key: str):
        return R2Blob(self.client, self.name, key)


class R2Client:
    def __init__(self):
        self.client = get_client()

    def bucket(self, name: str):
        return R2Bucket(self.client, name)


@tracer.start_as_current_span("storage.state.copy")
def copy(source: str, destination: str, *, allow_missing: bool = False, missing_json: bool = False):
    backend = os.environ.get("FORESEA_STORAGE_BACKEND", "gcs").strip()
    trace.get_current_span().set_attribute("storage.backend", backend)
    downloading = source.startswith("gs://")
    if downloading == destination.startswith("gs://"):
        raise ValueError("Exactly one endpoint must be a gs:// state URL")
    if not downloading:
        path = Path(source)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Refusing to upload an absent or empty state file")
    if backend == "gcs":
        if allow_missing or missing_json:
            from google.cloud import storage
            parsed_source = urlsplit(source)
            if not downloading:
                raise ValueError("Missing-object options apply only to downloads")
            blob = storage.Client().bucket(parsed_source.netloc).blob(parsed_source.path.lstrip("/"))
            if not blob.exists():
                if missing_json:
                    Path(destination).write_text("{}", encoding="utf-8")
                return
        subprocess.run(["gcloud", "storage", "cp", source, destination], check=True)
        return
    if backend != "r2":
        raise ValueError("FORESEA_STORAGE_BACKEND must be gcs or r2")
    bucket = os.environ.get("R2_STATE_BUCKET", "").strip()
    if not bucket:
        raise ValueError("R2_STATE_BUCKET is required for R2")
    remote, local = (source, destination) if downloading else (destination, source)
    parsed = urlsplit(remote)
    expected = os.environ.get("TRACK_STORE_BUCKET_NAME") or urlsplit(
        os.environ.get("TRACK_STORE_BUCKET") or os.environ.get("AGENT_TRADING_STORE_BUCKET")
        or "gs://brave-drive-471109-d9-track-record-store"
    ).netloc
    if parsed.scheme != "gs" or parsed.netloc != expected or not parsed.path.strip("/"):
        raise ValueError("Only the configured GCS state bucket can be mapped to R2")
    key = parsed.path.lstrip("/")
    client = get_client()
    if downloading:
        blob = R2Blob(client, bucket, key)
        try:
            blob.reload()
        except ClientError as exc:
            if (allow_missing or missing_json) and exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                if missing_json:
                    Path(local).write_text("{}", encoding="utf-8")
                return
            raise
        blob.download_to_filename(local)
    else:
        path = Path(local)
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".sqlite":
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
                if database.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise ValueError("SQLite integrity check failed")
        elif path.suffix == ".duckdb":
            import duckdb
            with duckdb.connect(str(path), read_only=True) as database:
                database.execute("SHOW TABLES").fetchall()
        try:
            previous = client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            client.copy_object(
                Bucket=bucket, Key=key + ".previous",
                CopySource={"Bucket": bucket, "Key": key},
                CopySourceIfMatch=previous["ETag"],
            )
        client.upload_file(str(path), bucket, key)
        transferred.add(path.stat().st_size, {"backend": "r2", "direction": "upload"})
    logger.info("State copy completed (backend=r2, direction=%s)", "download" if downloading else "upload")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["cp"])
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--missing-json", action="store_true")
    args = parser.parse_args()
    copy(args.source, args.destination, allow_missing=args.allow_missing, missing_json=args.missing_json)


if __name__ == "__main__":
    main()
