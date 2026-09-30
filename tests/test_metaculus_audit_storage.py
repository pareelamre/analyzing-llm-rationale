"""Write-ahead GCS audit behavior for short-lived GitHub Actions runners."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from google.api_core.exceptions import Forbidden, NotFound, PreconditionFailed

from analyzing_llm_rationale.metaculus_audit_storage import (
    AuditStorageError,
    restore_audit,
    upload_audit,
)


class _Blob:
    def __init__(self, content: bytes | None = None, generation: int = 0):
        self.content = content
        self.generation = generation
        self.fail_upload = False

    @property
    def size(self):
        return len(self.content) if self.content is not None else None

    def reload(self, **kwargs):
        if self.content is None:
            raise NotFound("missing")

    def download_to_filename(self, filename: str, **kwargs):
        Path(filename).write_bytes(self.content)

    def upload_from_filename(self, filename: str, *, if_generation_match: int, **kwargs):
        if self.fail_upload or if_generation_match != self.generation:
            raise PreconditionFailed("stale or unavailable")
        self.content = Path(filename).read_bytes()
        self.generation += 1


class AuditStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "metaculus_audit.jsonl"
        self.uri = "gs://foresea-bucket/metaculus/qwen.jsonl"
        self.blob = _Blob()
        patcher = mock.patch("analyzing_llm_rationale.metaculus_audit_storage._client")
        self.client = patcher.start()
        self.addCleanup(patcher.stop)
        self.client.return_value.bucket.return_value.blob.return_value = self.blob

    def test_missing_audit_never_silently_restarts_history(self):
        with self.assertRaises(AuditStorageError):
            restore_audit(self.path, self.uri)
        self.assertFalse(self.path.exists())

    def test_provisioned_empty_audit_restores_and_uploads(self):
        self.blob.content = b""
        self.blob.generation = 1
        restore_audit(self.path, self.uri)
        self.assertEqual(self.path.read_bytes(), b"")
        self.path.write_bytes(b'{"outcome":"prepared"}\n')
        upload_audit(self.path, self.uri)
        self.assertEqual(self.blob.content, self.path.read_bytes())
        self.assertEqual(self.blob.generation, 2)

    def test_existing_remote_audit_is_restored_before_next_run(self):
        self.blob.content = b'{"outcome":"submission_unknown"}\n'
        self.blob.generation = 9
        restore_audit(self.path, self.uri)
        self.assertEqual(self.path.read_bytes(), self.blob.content)
        self.path.write_bytes(self.path.read_bytes() + b'{"outcome":"submission_verified"}\n')
        upload_audit(self.path, self.uri)
        self.assertEqual(self.blob.generation, 10)

    def test_stale_generation_fails_closed(self):
        self.blob.content = b""
        self.blob.generation = 2
        restore_audit(self.path, self.uri)
        self.blob.generation = 3
        with self.assertRaises(AuditStorageError):
            upload_audit(self.path, self.uri)

    def test_rotation_preserves_latest_quarantine_state_and_archives_history(self):
        self.blob.content = b""
        self.blob.generation = 1
        restore_audit(self.path, self.uri)
        data = (
            b'{"question_id":1,"outcome":"prepared"}\n'
            b'{"question_id":1,"outcome":"submission_verified"}\n'
            b'{"question_id":2,"outcome":"submission_unknown"}\n'
            b'{"question_id":2,"outcome":"previewed"}\n'
        )
        self.path.write_bytes(data)
        archive = _Blob()
        with mock.patch("analyzing_llm_rationale.metaculus_audit_storage._ROTATE_AUDIT_BYTES", 1), mock.patch(
            "analyzing_llm_rationale.metaculus_audit_storage._blob", side_effect=[self.blob, archive]
        ):
            upload_audit(self.path, self.uri)
        self.assertEqual(archive.content, data)
        self.assertNotIn(b'"prepared"', self.path.read_bytes())
        self.assertIn(b'"submission_unknown"', self.path.read_bytes())
        self.assertIn(b'"submission_verified"', self.path.read_bytes())

    def test_permission_error_is_not_mistaken_for_first_run(self):
        with mock.patch.object(self.blob, "reload", side_effect=Forbidden("denied")):
            with self.assertRaises(AuditStorageError):
                restore_audit(self.path, self.uri)
        self.assertFalse(self.path.exists())

    def test_invalid_uri_and_missing_restore_marker_fail_closed(self):
        with self.assertRaises(AuditStorageError):
            restore_audit(self.path, "https://example.com/audit")
        self.path.write_bytes(b"event\n")
        with self.assertRaises(AuditStorageError):
            upload_audit(self.path, self.uri)


if __name__ == "__main__":
    unittest.main()
