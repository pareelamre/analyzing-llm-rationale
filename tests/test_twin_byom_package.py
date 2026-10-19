"""Real loopback worker lifecycle with synthetic SQL state and denied egress."""
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import requests
import yaml

from analyzing_llm_rationale import datastore_sql
from analyzing_llm_rationale.twin.scheduler import ShadowCycleSchedule, ensure_shadow_cycle_job
from analyzing_llm_rationale.twin.worker import (
    DatastoreWorkerJobs,
    WorkerJob,
    WorkerJobError,
    WorkerJobKind,
    WorkerJobStatus,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "deploy" / "byom"

# Runs before app imports. Even an accidental collector call must fail offline.
BOOT = """
import sys
def guard(event, args):
    if event == 'socket.connect' and args[1][0] not in ('127.0.0.1', '::1'):
        print('EGRESS_BLOCKED', flush=True)
        raise RuntimeError('offline test denies non-loopback connections')
    if event == 'socket.getaddrinfo' and args[0] not in ('127.0.0.1', '::1', 'localhost'):
        print('EGRESS_BLOCKED', flush=True)
        raise RuntimeError('offline test denies external DNS')
sys.addaudithook(guard)
import uvicorn
uvicorn.run('analyzing_llm_rationale.twin.runtime_app:create_environment_app',
            factory=True, host='127.0.0.1', port=int(sys.argv[1]),
            access_log=False, log_level='warning')
"""


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class ByomPackageTests(unittest.TestCase):
    def test_package_is_opt_in_loopback_only_and_role_isolated(self):
        package = yaml.safe_load((PACKAGE / "compose.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(package["services"]), {"maintenance", "research"})
        maintenance, research = (package["services"][name] for name in ("maintenance", "research"))
        for role, service in package["services"].items():
            self.assertEqual(service["profiles"], ["twin"])
            self.assertEqual(service["network_mode"], "host")
            self.assertNotIn("ports", service)
            self.assertIn("127.0.0.1", service["command"])
            self.assertEqual(service["environment"]["FORESEA_TWIN_WORKER_ROLE"], role)
            self.assertEqual(service["environment"]["FORESEA_DATASTORE_BACKEND"], "sql")
            self.assertEqual(service["environment"]["FORESEA_TWIN_MODE"], "shadow")
            self.assertEqual(service["environment"]["FORESEA_TWIN_LIVE_CAPITAL"], "0")
            self.assertEqual(service["environment"]["FORESEA_TWIN_LIVE_MANDATE"], "")
            self.assertTrue(service["read_only"])
            self.assertEqual(service["cap_drop"], ["ALL"])
        self.assertNotEqual(maintenance["env_file"], research["env_file"])
        self.assertEqual(maintenance["volumes"][0], research["volumes"][0])
        self.assertTrue(maintenance["volumes"][0]["read_only"])
        self.assertEqual(len(research["volumes"]), 1)
        self.assertIn("state:/data", maintenance["volumes"])
        self.assertEqual(research["environment"]["FORESEA_DATASTORE_PATH"], "/tmp/research.sqlite3")

    def test_real_sql_workers_dispatch_and_survive_restart_without_collection(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        with mock.patch.dict(
            os.environ, {"FORESEA_DATASTORE_BACKEND": "sql"},
        ), requests.Session() as session:
            session.trust_env = False
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            for name in ("twin.yaml", "models.yaml"):
                shutil.copyfile(PACKAGE / name, config / name)
            policy = yaml.safe_load((config / "twin.yaml").read_text(encoding="utf-8"))
            policy["research"]["price_valid_until"] = (
                datetime.now(timezone.utc) + timedelta(days=1)
            ).isoformat()
            (config / "twin.yaml").write_text(yaml.safe_dump(policy), encoding="utf-8")
            state = root / "twin.sqlite3"
            db = datastore_sql.Client(path=str(state))
            self.addCleanup(db.close)
            jobs = DatastoreWorkerJobs(db)
            now = datetime.now(timezone.utc)
            # Complete the possible strategy buckets so dispatch cannot collect.
            schedule = ShadowCycleSchedule("shadow-scope:foresea-edge-v1", "foresea-edge-shadow-trial-v1")
            for offset in range(0, 1800, 300):
                at = now + timedelta(seconds=offset)
                job = ensure_shadow_cycle_job(jobs, schedule, now=at)
                claim = jobs.claim(job.id, worker_id="fixture", now=at)
                jobs.complete(job.id, worker_id="fixture", fence=claim.fence,
                              result={"status": "fixture_no_collection"}, now=at)
            job = jobs.add(WorkerJob(
                "offline-recovery", "shadow-offline", WorkerJobKind.RECOVERY, {},
                now + timedelta(minutes=30), created_at=now,
            ))
            maintenance_port, research_port = free_port(), free_port()
            while maintenance_port == research_port:
                research_port = free_port()
            maintenance_url = f"http://127.0.0.1:{maintenance_port}"
            research_url = f"http://127.0.0.1:{research_port}"
            maintenance_secret, dispatch_secret, gateway_secret = (secrets.token_hex(32) for _ in range(3))
            env = {key: value for key, value in os.environ.items() if key.upper() in {
                "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG",
                "APPDATA", "LOCALAPPDATA", "USERPROFILE",
            }}
            shipped = yaml.safe_load((PACKAGE / "compose.yaml").read_text(encoding="utf-8"))
            env.update(shipped["x-environment"])
            env.update({
                "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1",
                "FORESEA_DATASTORE_BACKEND": "sql", "FORESEA_TWIN_MODE": "shadow",
                "FORESEA_TWIN_LIVE_CAPITAL": "0", "FORESEA_TWIN_LIVE_MANDATE": "",
                "FORESEA_TWIN_CONFIG_DIR": str(config),
                "FORESEA_TWIN_RESEARCH_DISPATCH_SECRET": dispatch_secret,
                "FORESEA_TWIN_RESEARCH_GATEWAY_SECRET": gateway_secret,
                "FORESEA_TWIN_MAINTENANCE_URL": maintenance_url,
                "FORESEA_TWIN_DISPATCHER_ACCOUNTS": "local-dispatcher",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            })
            finished_research = jobs.add(WorkerJob(
                "offline-research", "shadow-offline", WorkerJobKind.RESEARCH,
                {field: "offline-fixture" for field in (
                    "research_assignment_id", "budget_reservation_id", "market_snapshot_id",
                    "evidence_set_id", "model_config_id", "budget_key_id", "strategy_cycle_id",
                )},
                now + timedelta(minutes=30),
                completed_result={"status": "completed", "research_result_id": "offline-result",
                                  "usage_record_id": "offline-usage"},
                status=WorkerJobStatus.COMPLETED, created_at=now, completed_at=now,
            ))
            role_envs = {
                "maintenance": {**env, "FORESEA_TWIN_WORKER_ROLE": "maintenance",
                    "FORESEA_DATASTORE_PATH": str(state),
                    "FORESEA_TWIN_MAINTENANCE_DISPATCH_SECRET": maintenance_secret,
                    "FORESEA_TWIN_RESEARCH_URL": research_url,
                    "FORESEA_TWIN_SCHEDULER_ACCOUNTS": "local-scheduler",
                    "FORESEA_TWIN_RESEARCH_ACCOUNTS": "local-research"},
                "research": {**env, "FORESEA_TWIN_WORKER_ROLE": "research",
                    "FORESEA_DATASTORE_PATH": str(root / "unused-research.sqlite3"),
                    "FORESEA_TWIN_BYOM_API_KEY": "offline-test-placeholder"},
            }
            for generation in range(2):
                processes = []
                logs = []
                if generation == 1:
                    before_restart = datetime.now(timezone.utc) - timedelta(seconds=5)
                    abandoned = jobs.add(WorkerJob(
                        "offline-abandoned", "shadow-offline", WorkerJobKind.RECOVERY, {},
                        now + timedelta(minutes=30), created_at=before_restart,
                    ))
                    jobs.claim(abandoned.id, worker_id="crashed-worker",
                               now=before_restart, lease_seconds=1)
                try:
                    for role, port in (("maintenance", maintenance_port), ("research", research_port)):
                        log = (root / f"{role}-{generation}.log").open("w+b")
                        logs.append(log)
                        process = subprocess.Popen(
                            [sys.executable, "-c", BOOT, str(port)], cwd=ROOT,
                            env=role_envs[role], stdout=log, stderr=subprocess.STDOUT,
                            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
                        )
                        processes.append(process)
                        deadline = time.monotonic() + 45
                        while True:
                            if process.poll() is not None:
                                log.seek(0)
                                self.fail(f"{role} failed startup: {log.read().decode(errors='replace')}")
                            try:
                                response = session.get(f"http://127.0.0.1:{port}/ready", timeout=1)
                                if response.status_code == 200:
                                    break
                            except requests.ConnectionError:
                                pass
                            self.assertLess(time.monotonic(), deadline, f"{role} did not become ready")
                            time.sleep(0.05)
                    for secret in (gateway_secret, dispatch_secret):
                        for path in ("dispatch", "maintain"):
                            denied = session.post(maintenance_url + "/internal/twin/" + path,
                                json={"job_id": job.id},
                                headers={"Authorization": f"Bearer {secret}"}, timeout=3)
                            self.assertEqual(denied.status_code, 401)
                    for secret in (maintenance_secret, dispatch_secret, gateway_secret):
                        status = session.get(
                            maintenance_url + "/internal/twin/research-jobs/" + finished_research.id,
                            headers={"Authorization": f"Bearer {secret}"}, timeout=3,
                        )
                        self.assertEqual(status.status_code, 200 if secret == gateway_secret else 401)
                        result = session.post(research_url + "/internal/twin/research",
                            json={"job_id": finished_research.id},
                            headers={"Authorization": f"Bearer {secret}"}, timeout=3)
                        self.assertEqual(result.status_code, 200 if secret == dispatch_secret else 401)
                        if result.status_code == 200:
                            self.assertEqual(result.json(), finished_research.completed_result)
                    self.assertEqual(session.post(research_url + "/internal/twin/maintain",
                        json={"job_id": job.id}, timeout=3).status_code, 404)
                    if generation == 1:
                        # Startup has fenced the abandoned owner before a new dispatch.
                        pending = jobs.get(abandoned.id)
                        self.assertIsNone(pending.worker_id)
                        with self.assertRaises(WorkerJobError):
                            jobs.complete(abandoned.id, worker_id="crashed-worker", fence=1,
                                result={"status": "stale"}, now=datetime.now(timezone.utc))
                    dispatched = session.post(maintenance_url + "/internal/twin/dispatch",
                        headers={"Authorization": f"Bearer {maintenance_secret}"}, timeout=5)
                    self.assertEqual(dispatched.status_code, 200)
                    self.assertEqual(dispatched.json()["tasks_enqueued"], 1)
                    if generation == 1:
                        recovered = jobs.get(abandoned.id)
                        self.assertEqual(recovered.attempts, 2)
                        self.assertEqual(recovered.fence, 2)
                        self.assertEqual(recovered.completed_result["status"], "complete")
                    duplicate = session.post(maintenance_url + "/internal/twin/maintain",
                        json={"job_id": job.id},
                        headers={"Authorization": f"Bearer {maintenance_secret}"}, timeout=5)
                    self.assertEqual(duplicate.status_code, 200)
                    self.assertEqual(duplicate.json()["status"], "complete")
                    stored = jobs.get(job.id)
                    self.assertEqual(stored.attempts, 1)
                    self.assertEqual(stored.completed_result["stale_jobs_detected"], 0)
                    if generation == 0:
                        concurrent = jobs.add(WorkerJob(
                            "offline-concurrent", "shadow-offline", WorkerJobKind.RECOVERY, {},
                            now + timedelta(minutes=30), created_at=now,
                        ))

                        def dispatch_once(_):
                            with requests.Session() as caller:
                                caller.trust_env = False
                                return caller.post(maintenance_url + "/internal/twin/dispatch",
                                    headers={"Authorization": f"Bearer {maintenance_secret}"},
                                    timeout=5).status_code

                        with ThreadPoolExecutor(max_workers=48) as callers:
                            statuses = list(callers.map(dispatch_once, range(48)))
                        self.assertTrue(all(code in {200, 503} for code in statuses), statuses)
                        self.assertIn(200, statuses)
                        self.assertEqual(jobs.get(concurrent.id).attempts, 1)
                        self.assertEqual(jobs.get(concurrent.id).completed_result["status"], "complete")
                finally:
                    # Abrupt restart, not just an in-memory TestClient restart.
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=10)
                    for log in logs:
                        log.seek(0)
                        output = log.read().decode("utf-8", errors="replace")
                        log.close()
                        self.assertNotIn("EGRESS_BLOCKED", output)
            self.assertTrue(state.is_file())


if __name__ == "__main__":
    unittest.main()
