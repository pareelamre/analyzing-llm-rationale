"""The live server's copy of the private knowledge base: knowledge-sync.yml and docker.yml.

The base holds licensed reports, so it must reach the server through the private models bucket,
never the Docker image (also pushed to GHCR) or the public workflow log.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_knowledge import make_base  # noqa: E402

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

    def test_syncs_only_the_knowledge_prefix_of_the_private_bucket(self):
        self.assertEqual(self.parsed["env"]["KNOWLEDGE_BUCKET_PATH"],
                         "gs://brave-drive-471109-d9-models/knowledge")
        run = self.steps["Copy it to the models bucket"]["run"]
        self.assertIn('gs://*/knowledge) ;;', run)  # refuses any other target
        self.assertIn("--delete-unmatched-destination-objects", run)
        self.assertIn("--checksums-only", run)  # unchanged files keep their mtime on the server
        self.assertIn("--no-user-output-enabled", run)  # no per-file lines in the public log

    def test_validates_before_it_copies(self):
        self.assertLess(self.names.index("Stage and validate what the server reads"),
                        self.names.index("Copy it to the models bucket"))

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


class LiveServerReadsTheBucketTests(unittest.TestCase):
    def test_deploy_points_knowledge_dir_at_the_bucket_mount(self):
        text = _DOCKER.read_text(encoding="utf-8")
        self.assertIn("KNOWLEDGE_DIR=/app/models/knowledge,", text)
        # The image build never sees the base.
        self.assertNotIn("foresea-knowledge", text)
        self.assertNotIn("KNOWLEDGE_REPO_TOKEN", text)

    def test_the_base_never_enters_the_docker_build_context(self):
        ignored = (_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
        self.assertIn("knowledge-base/", ignored)
        self.assertIn("staged/", ignored)


if __name__ == "__main__":
    unittest.main()
