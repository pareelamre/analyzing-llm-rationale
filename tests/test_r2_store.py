from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from botocore.exceptions import ClientError

from analyzing_llm_rationale import gcs_store, r2_store


class R2StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "store.db"
        self.client = mock.Mock()
        self.client.head_object.return_value = {"ETag": '"v1"', "ContentLength": 4}
        self.client.get_object.return_value = {
            "Body": io.BytesIO(b"data"), "ContentLength": 4,
        }

    def test_download_is_pinned_to_metadata_version(self):
        blob = r2_store.R2Blob(self.client, "private", "store.db")
        blob.reload()
        blob.download_to_filename(str(self.path))
        self.assertEqual(self.path.read_bytes(), b"data")
        self.client.get_object.assert_called_once_with(
            Bucket="private", Key="store.db", IfMatch='"v1"',
        )

    def test_partial_download_preserves_existing_file(self):
        self.path.write_bytes(b"previous")
        self.client.get_object.return_value["Body"] = io.BytesIO(b"bad")
        blob = r2_store.R2Blob(self.client, "private", "store.db")
        blob.reload()
        with self.assertRaises(ValueError):
            blob.download_to_filename(str(self.path))
        self.assertEqual(self.path.read_bytes(), b"previous")

    def test_credentials_are_not_sent_to_arbitrary_endpoint(self):
        with mock.patch.dict(os.environ, {
            "R2_ENDPOINT_URL": "https://example.com", "R2_ACCESS_KEY_ID": "fixture",
            "R2_SECRET_ACCESS_KEY": "fixture",
        }, clear=True), self.assertRaises(ValueError):
            r2_store.get_client()

    def test_explicit_r2_configuration_does_not_fall_back_to_gcs(self):
        with mock.patch.dict(os.environ, {"FORESEA_STORAGE_BACKEND": "r2"}, clear=True):
            with self.assertRaises(ValueError):
                r2_store.copy("gs://source/store.db", str(self.path))

    def test_gcs_remains_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(r2_store.subprocess, "run") as run:
                r2_store.copy("gs://source/store.db", str(self.path))
        self.assertTrue(run.call_args.kwargs["check"])

    def test_runtime_uses_private_r2_bucket_and_caches_unchanged_etag(self):
        from test_gcs_store import _expire_debounce, _reset_module_state

        _reset_module_state()
        self.addCleanup(_reset_module_state)
        bucket = r2_store.R2Bucket(self.client, "private")
        adapter = mock.Mock()
        adapter.bucket.return_value = bucket
        with mock.patch.dict(os.environ, {
            "FORESEA_STORAGE_BACKEND": "r2", "R2_STATE_BUCKET": "private",
        }), mock.patch.object(r2_store, "R2Client", return_value=adapter):
            self.assertTrue(gcs_store.ensure_local_copy(self.path))
            _expire_debounce()
            self.assertTrue(gcs_store.ensure_local_copy(self.path))
        self.assertEqual(self.path.read_bytes(), b"data")
        self.assertEqual(self.client.get_object.call_count, 1)
        adapter.bucket.assert_called_with("private")

    def test_upload_refuses_empty_file(self):
        self.path.write_bytes(b"")
        with mock.patch.dict(os.environ, {
            "FORESEA_STORAGE_BACKEND": "r2", "R2_STATE_BUCKET": "private",
        }), mock.patch.object(r2_store, "get_client", return_value=self.client):
            with self.assertRaises(ValueError):
                r2_store.copy(str(self.path), "gs://brave-drive-471109-d9-track-record-store/store.db")
        self.client.upload_file.assert_not_called()

    def test_unrelated_gcs_bucket_cannot_be_remapped(self):
        with mock.patch.dict(os.environ, {
            "FORESEA_STORAGE_BACKEND": "r2", "R2_STATE_BUCKET": "private",
        }), mock.patch.object(r2_store, "get_client") as client:
            with self.assertRaises(ValueError):
                r2_store.copy("gs://other-bucket/store.db", str(self.path))
        client.assert_not_called()

    def test_upload_keeps_bounded_previous_copy(self):
        self.path.write_bytes(b"new state")
        with mock.patch.dict(os.environ, {
            "FORESEA_STORAGE_BACKEND": "r2", "R2_STATE_BUCKET": "private",
        }), mock.patch.object(r2_store, "get_client", return_value=self.client):
            r2_store.copy(str(self.path), "gs://brave-drive-471109-d9-track-record-store/store.db")
        self.client.copy_object.assert_called_once_with(
            Bucket="private", Key="store.db.previous",
            CopySource={"Bucket": "private", "Key": "store.db"}, CopySourceIfMatch='"v1"',
        )
        self.client.upload_file.assert_called_once_with(str(self.path), "private", "store.db")

    def test_failed_recovery_copy_prevents_overwrite(self):
        self.path.write_bytes(b"new state")
        self.client.copy_object.side_effect = RuntimeError("backup failed")
        with mock.patch.dict(os.environ, {
            "FORESEA_STORAGE_BACKEND": "r2", "R2_STATE_BUCKET": "private",
        }), mock.patch.object(r2_store, "get_client", return_value=self.client):
            with self.assertRaises(RuntimeError):
                r2_store.copy(str(self.path), "gs://brave-drive-471109-d9-track-record-store/store.db")
        self.client.upload_file.assert_not_called()

    def test_optional_missing_is_distinct_from_access_denied(self):
        with mock.patch.dict(os.environ, {
            "FORESEA_STORAGE_BACKEND": "r2", "R2_STATE_BUCKET": "private",
        }), mock.patch.object(r2_store, "get_client", return_value=self.client):
            self.client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
            r2_store.copy("gs://brave-drive-471109-d9-track-record-store/store.db", str(self.path), allow_missing=True)
            self.assertFalse(self.path.exists())
            self.client.head_object.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadObject")
            with self.assertRaises(ClientError):
                r2_store.copy("gs://brave-drive-471109-d9-track-record-store/store.db", str(self.path), allow_missing=True)

    def test_gcs_also_refuses_empty_upload(self):
        self.path.write_bytes(b"")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(r2_store.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                r2_store.copy(str(self.path), "gs://source/store.db")
        run.assert_not_called()

    def test_runtime_configuration_failure_preserves_local_state(self):
        from test_gcs_store import _reset_module_state

        _reset_module_state()
        self.addCleanup(_reset_module_state)
        self.path.write_bytes(b"previous")
        with mock.patch.dict(os.environ, {"FORESEA_STORAGE_BACKEND": "r2"}), mock.patch.object(
            r2_store, "R2Client", side_effect=ValueError("missing configuration"),
        ), mock.patch.object(gcs_store, "_get_gcs_client") as gcs:
            self.assertTrue(gcs_store.ensure_local_copy(self.path))
        self.assertEqual(self.path.read_bytes(), b"previous")
        gcs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
