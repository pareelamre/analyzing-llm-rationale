from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import gcs_store


class _FakeBlob:
    def __init__(self, generation, content=b"data"):
        self.generation = generation
        self._content = content
        self.download_calls = 0

    def reload(self):
        pass

    def download_to_filename(self, path):
        self.download_calls += 1
        Path(path).write_bytes(self._content)


class _FakeBucket:
    def __init__(self, blob):
        self._blob = blob

    def blob(self, _name):
        return self._blob


class _FakeClient:
    def __init__(self, blob):
        self._bucket = _FakeBucket(blob)
        self.reload_calls = 0

    def bucket(self, _name):
        self._bucket._blob.reload_wrapper = self
        return self._bucket


def _reset_module_state():
    gcs_store._last_synced_generation = None
    gcs_store._last_check_monotonic = None
    gcs_store._last_success_monotonic = None


def _expire_debounce():
    """Force the next ensure_local_copy call to treat GCS as due for a check,
    simulating that _CHECK_INTERVAL_S has elapsed since the last one."""
    gcs_store._last_check_monotonic = None


class GcsStoreTests(unittest.TestCase):
    def setUp(self):
        _reset_module_state()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.local_path = Path(self.tmpdir.name) / "store.duckdb"

    def test_no_client_falls_back_to_local_existence(self):
        with mock.patch.object(gcs_store, "_get_gcs_client", return_value=None):
            self.assertFalse(gcs_store.ensure_local_copy(self.local_path))
            self.local_path.write_bytes(b"already here")
            self.assertTrue(gcs_store.ensure_local_copy(self.local_path))

    def test_downloads_on_first_call(self):
        blob = _FakeBlob(generation=1, content=b"seed data")
        with mock.patch.object(gcs_store, "_get_gcs_client", return_value=_FakeClient(blob)):
            ok = gcs_store.ensure_local_copy(self.local_path)
        self.assertTrue(ok)
        self.assertEqual(blob.download_calls, 1)
        self.assertEqual(self.local_path.read_bytes(), b"seed data")

    def test_skips_redownload_when_generation_unchanged(self):
        blob = _FakeBlob(generation=1, content=b"seed data")
        with mock.patch.object(gcs_store, "_get_gcs_client", return_value=_FakeClient(blob)):
            gcs_store.ensure_local_copy(self.local_path)
            _expire_debounce()
            gcs_store.ensure_local_copy(self.local_path)
        self.assertEqual(blob.download_calls, 1)

    def test_redownloads_when_generation_changes(self):
        blob = _FakeBlob(generation=1, content=b"v1")
        with mock.patch.object(gcs_store, "_get_gcs_client", return_value=_FakeClient(blob)):
            gcs_store.ensure_local_copy(self.local_path)
            _expire_debounce()
            blob.generation = 2
            blob._content = b"v2"
            gcs_store.ensure_local_copy(self.local_path)
        self.assertEqual(blob.download_calls, 2)
        self.assertEqual(self.local_path.read_bytes(), b"v2")

    def test_metadata_check_failure_falls_back_to_local_existence(self):
        class _BrokenBlob(_FakeBlob):
            def reload(self):
                raise RuntimeError("network down")

        blob = _BrokenBlob(generation=1)
        with mock.patch.object(gcs_store, "_get_gcs_client", return_value=_FakeClient(blob)):
            self.assertFalse(gcs_store.ensure_local_copy(self.local_path))
            self.local_path.write_bytes(b"stale but present")
            _expire_debounce()
            self.assertTrue(gcs_store.ensure_local_copy(self.local_path))

    def test_debounce_skips_gcs_entirely_within_interval(self):
        """The whole point of the fix: a burst of requests within the debounce
        window must not each pay a live GCS round trip -- reload() itself
        should only be called once, not once per ensure_local_copy call."""
        blob = _FakeBlob(generation=1)
        reload_calls = []
        real_reload = blob.reload
        blob.reload = lambda: (reload_calls.append(1), real_reload())[-1]

        with mock.patch.object(gcs_store, "_get_gcs_client", return_value=_FakeClient(blob)):
            for _ in range(5):
                gcs_store.ensure_local_copy(self.local_path)
        self.assertEqual(len(reload_calls), 1)

    def test_cleanup_oserror_during_failed_download_does_not_raise(self):
        """A failure during the download's own cleanup (e.g. read-only
        filesystem) must not escape as an unhandled exception -- there's no
        try/except around this call at either server.py call site."""
        class _FailingDownloadBlob(_FakeBlob):
            def download_to_filename(self, path):
                raise RuntimeError("disk full")

        blob = _FailingDownloadBlob(generation=1)
        with (
            mock.patch.object(gcs_store, "_get_gcs_client", return_value=_FakeClient(blob)),
            mock.patch.object(Path, "unlink", side_effect=OSError("permission denied")),
        ):
            # Must not raise, even though both the download AND its own
            # cleanup attempt fail.
            result = gcs_store.ensure_local_copy(self.local_path)
        self.assertFalse(result)

    def test_never_raises_on_totally_unexpected_client_error(self):
        def _boom():
            raise RuntimeError("something nobody anticipated")

        with mock.patch.object(gcs_store, "_get_gcs_client", side_effect=_boom):
            result = gcs_store.ensure_local_copy(self.local_path)
        self.assertFalse(result)

    def test_stale_failure_escalates_to_error_log(self):
        blob = _FakeBlob(generation=1)
        with mock.patch.object(gcs_store, "_get_gcs_client", return_value=_FakeClient(blob)):
            gcs_store.ensure_local_copy(self.local_path)  # establish a success baseline

        def _broken_reload():
            raise RuntimeError("gcs down")

        blob.reload = _broken_reload
        # Simulate an hour with no successful sync.
        gcs_store._last_success_monotonic -= gcs_store._STALE_ALERT_S + 1
        _expire_debounce()
        with (
            mock.patch.object(gcs_store, "_get_gcs_client", return_value=_FakeClient(blob)),
            mock.patch.object(gcs_store.logger, "error") as mock_error,
            mock.patch.object(gcs_store.logger, "warning") as mock_warning,
        ):
            gcs_store.ensure_local_copy(self.local_path)
        mock_error.assert_called_once()
        mock_warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
