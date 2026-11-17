"""The live server's copy of the private knowledge base: knowledge-sync.yml, its R2 script and docker.yml.

The base holds licensed reports, so it must reach the server through the private R2 state bucket,
never the Docker image (also pushed to GHCR), Google Cloud storage, or the public workflow log.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_knowledge import _sync_script, make_base  # noqa: E402

from analyzing_llm_rationale import knowledge as kb  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_SYNC = _ROOT / ".github" / "workflows" / "knowledge-sync.yml"
_DOCKER = _ROOT / ".github" / "workflows" / "docker.yml"


def _steps() -> dict[str, dict]:
    parsed = yaml.safe_load(_SYNC.read_text(encoding="utf-8"))
    return {s.get("name") or s.get("uses"): s for s in parsed["jobs"]["sync"]["steps"]}


class KnowledgeSyncWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.parsed = yaml.safe_load(_SYNC.read_text(encoding="utf-8"))
        self.steps = _steps()
        self.names = list(self.steps)

    def test_runs_on_a_schedule_and_by_hand(self):
        triggers = self.parsed[True]  # PyYAML reads the bare `on:` key as True
        self.assertIn("schedule", triggers)
        self.assertIn("workflow_dispatch", triggers)
        self.assertEqual(self.parsed["permissions"], {"contents": "read"})

    def test_copies_to_r2_with_the_writer_key_and_checks_the_server_key(self):
        step = self.steps["Copy it to R2"]
        self.assertEqual(step["run"].strip(), "python scripts/knowledge_r2_sync.py staged")
        self.assertEqual(step["env"]["R2_ACCESS_KEY_ID"], "${{ secrets.R2_ACCESS_KEY_ID }}")
        self.assertEqual(step["env"]["R2_RUNTIME_ACCESS_KEY_ID"], "${{ secrets.R2_RUNTIME_ACCESS_KEY_ID }}")
        self.assertEqual(self.parsed["env"]["R2_STATE_BUCKET"], "${{ vars.R2_STATE_BUCKET }}")

    def test_uses_no_google_cloud_storage(self):
        text = _SYNC.read_text(encoding="utf-8")
        for word in ("gcloud", "gs://", "google-github-actions", "GCP_SA_KEY"):
            self.assertNotIn(word, text)

    def test_validates_before_it_copies(self):
        self.assertLess(self.names.index("Stage and validate what the server reads"),
                        self.names.index("Copy it to R2"))

    def test_staging_copies_only_what_the_server_reads_and_logs_only_counts(self):
        run = self.steps["Stage and validate what the server reads"]["run"]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            make_base(work / "knowledge-base")
            (work / "knowledge-base" / ".kb").mkdir()
            (work / "knowledge-base" / ".kb" / "cache.sqlite").write_text("x")
            (work / "knowledge-base" / "README.md").write_text("notes")
            (work / "scripts").symlink_to(_ROOT / "scripts")
            script = run.replace("python3 ", f"{sys.executable} ")
            ok = subprocess.run(["bash", "-e", "-c", script], cwd=work, capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            staged = sorted(str(p.relative_to(work / "staged")) for p in (work / "staged").rglob("*") if p.is_file())
            self.assertIn("entities.jsonl", staged)
            self.assertTrue(all(p == "entities.jsonl" or p.startswith("sources/") for p in staged), staged)
            stats = json.loads(ok.stdout.strip().removeprefix("Valid: "))
            self.assertEqual(stats["sources"], 2)

            # A broken base is not synced, and its problems (which can quote data) stay out of the log.
            bad = work / "knowledge-base" / "sources" / "acme-2026-06-widgets" / "claims.jsonl"
            bad.write_text('{"id": "x", "text": "SECRET PASSAGE", "entities": ["nobody"]}\n')
            subprocess.run(["rm", "-rf", "staged"], cwd=work, check=True)
            failed = subprocess.run(["bash", "-e", "-c", script], cwd=work, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("::error title=Knowledge base invalid::", failed.stdout)
            self.assertNotIn("SECRET PASSAGE", failed.stdout + failed.stderr)

    def test_a_missing_checkout_fails_loudly(self):
        run = self.steps["Stage and validate what the server reads"]["run"]
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run(["bash", "-e", "-c", run], cwd=tmp, capture_output=True, text=True)
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("::error title=Knowledge base missing::", out.stdout)


class FakeBucket:
    def __init__(self):
        self.objects, self.puts = {}, 0

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            error = Exception("missing")
            error.response = {"Error": {"Code": "404"}}
            raise error
        return {"ETag": '"%s"' % hashlib.md5(self.objects[Key]).hexdigest()}

    def put_object(self, Bucket, Key, Body, **_):
        self.puts += 1
        self.objects[Key] = Body

    def get_object(self, Bucket, Key, IfMatch=None):
        return {"Body": io.BytesIO(self.objects[Key])}


class KnowledgeR2SyncScriptTests(unittest.TestCase):
    def setUp(self):
        self.script = _sync_script()
        self.tmp = tempfile.TemporaryDirectory()
        self.base = make_base(Path(self.tmp.name) / "staged")

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_archive_is_deterministic_and_opens_as_the_server_opens_it(self):
        data = self.script.pack(self.base)
        self.assertEqual(data, self.script.pack(self.base))
        out = Path(self.tmp.name) / "out"
        kb.unpack(data, out)
        corpus = kb.load(out)
        self.assertEqual(corpus.errors, [])
        self.assertEqual(len(corpus.sources), 2)

    def test_uploads_only_when_the_content_changed(self):
        bucket = FakeBucket()
        data = self.script.pack(self.base)
        self.assertEqual(self.script.sync(data, writer=bucket, reader=bucket, bucket="b"), "uploaded")
        self.assertEqual(self.script.sync(data, writer=bucket, reader=bucket, bucket="b"), "unchanged")
        self.assertEqual(bucket.puts, 1)
        self.assertIn(kb.REMOTE_KEY, bucket.objects)

    def test_a_server_key_that_reads_something_else_fails(self):
        writer, other = FakeBucket(), FakeBucket()
        other.objects[kb.REMOTE_KEY] = b"stale"
        with self.assertRaises(RuntimeError):
            self.script.sync(self.script.pack(self.base), writer=writer, reader=other, bucket="b")

    def test_never_writes_outside_the_knowledge_prefix(self):
        with self.assertRaises(ValueError):
            self.script.sync(b"x", writer=FakeBucket(), reader=FakeBucket(), bucket="b",
                             key="track_record_store.duckdb")

    def test_the_log_shows_counts_not_content(self):
        bucket = FakeBucket()
        env = {"R2_STATE_BUCKET": "b"}
        with unittest.mock.patch.dict(os.environ, env), \
                unittest.mock.patch.object(self.script, "_client", return_value=bucket), \
                unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(self.script.main([str(self.base)]), 0)
        text = out.getvalue()
        self.assertIn("Knowledge archive uploaded:", text)
        self.assertIn("2 sources", text)
        for claim in kb.load(self.base).claims:
            self.assertNotIn(claim["text"], text)


class LiveServerReadsR2Tests(unittest.TestCase):
    def test_deploy_gives_the_server_the_read_only_r2_key(self):
        text = _DOCKER.read_text(encoding="utf-8")
        self.assertIn("KNOWLEDGE_R2_ENDPOINT_URL=${{ vars.R2_ENDPOINT_URL }},", text)
        self.assertIn("KNOWLEDGE_R2_BUCKET=${{ vars.R2_STATE_BUCKET }},", text)
        self.assertIn("KNOWLEDGE_R2_ACCESS_KEY_ID=${{ secrets.R2_RUNTIME_ACCESS_KEY_ID }},", text)
        self.assertIn("KNOWLEDGE_R2_SECRET_ACCESS_KEY=${{ secrets.R2_RUNTIME_SECRET_ACCESS_KEY }},", text)
        self.assertNotIn("${{ secrets.R2_ACCESS_KEY_ID }}", text)  # never the writer key
        self.assertIn("--remove-env-vars KNOWLEDGE_DIR", text)
        self.assertNotIn("KNOWLEDGE_DIR=", text)
        # The image build never sees the base.
        self.assertNotIn("foresea-knowledge", text)
        self.assertNotIn("KNOWLEDGE_REPO_TOKEN", text)

    def test_the_base_never_enters_the_docker_build_context(self):
        ignored = (_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
        self.assertIn("knowledge-base/", ignored)
        self.assertIn("staged/", ignored)


if __name__ == "__main__":
    unittest.main()
