"""A new process reuses accepted SQL research receipts without another call."""
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin.worker import (
    DatastoreResearchCompletionStore, DatastoreWorkerJobs,
    MaintenanceResearchJobGateway, ResearchCompletion, TwinResearchWorker,
    WorkerJob, WorkerJobKind, WorkerJobStatus,
)

def no_network(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('SQL recovery probe must not use network')
sys.addaudithook(no_network)
phase, root, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
counter_path = root / 'counts.json'
counts = json.loads(counter_path.read_text()) if counter_path.exists() else {
    'research': 0, 'authorize': 0, 'finalize': 0,
}
def persist():
    counter_path.write_text(json.dumps(counts))
now = datetime(2026, 10, 19, tzinfo=timezone.utc)
client = Client(path=str(root / 'jobs.sqlite3'))
jobs = DatastoreWorkerJobs(client)
receipts = DatastoreResearchCompletionStore(client)
if phase == 'first':
    jobs.add(WorkerJob('restart-research', 'shadow-probe', WorkerJobKind.RESEARCH,
        {k: 'fixture-id' for k in (
            'research_assignment_id', 'budget_reservation_id', 'market_snapshot_id',
            'evidence_set_id', 'model_config_id', 'budget_key_id', 'strategy_cycle_id',
        )}, now + timedelta(minutes=10)))

def authorize(_assignment):
    assert phase == 'first', 'replay must not claim another budget'
    counts['authorize'] += 1
    persist()

def research(_assignment):
    assert phase == 'first', 'saved-result retry must not call research'
    counts['research'] += 1
    persist()
    return ResearchCompletion('completed', result_payload={'synthetic': 'result'},
                              actual_usd='0', actual_tokens=12)

def finalize(_assignment, completion):
    counts['finalize'] += 1
    persist()
    assert completion.result_payload == {'synthetic': 'result'}
    assert completion.actual_tokens == 12
    if phase == 'first':
        raise TimeoutError('injected finalization interruption')
    assert phase == 'replay', 'terminal duplicate must not finalize again'
    return ResearchCompletion('completed', research_result_id='synthetic-result',
                              usage_record_id='synthetic-usage')

gateway = MaintenanceResearchJobGateway(
    jobs, authorize_assignment=authorize, finalize_result=finalize,
    completion_store=receipts,
)
if mode == 'factory':
    import os
    from unittest import mock
    from analyzing_llm_rationale.twin import runtime_app
    os.environ.update({
        'FORESEA_TWIN_WORKER_ROLE': 'maintenance',
        'FORESEA_DATASTORE_PATH': str(root / 'jobs.sqlite3'),
        'FORESEA_TWIN_MODE': 'shadow', 'FORESEA_TWIN_LIVE_CAPITAL': '0',
        'FORESEA_TWIN_CONFIG_DIR': str(root / 'config'),
        'FORESEA_TWIN_MAINTENANCE_DISPATCH_SECRET': 'a'*64,
        'FORESEA_TWIN_RESEARCH_DISPATCH_SECRET': 'b'*64,
        'FORESEA_TWIN_RESEARCH_GATEWAY_SECRET': 'c'*64,
        'FORESEA_TWIN_MAINTENANCE_URL': 'http://127.0.0.1:8081',
        'FORESEA_TWIN_RESEARCH_URL': 'http://127.0.0.1:8082',
        'FORESEA_TWIN_SCHEDULER_ACCOUNTS': 'local-scheduler',
        'FORESEA_TWIN_DISPATCHER_ACCOUNTS': 'local-dispatcher',
        'FORESEA_TWIN_RESEARCH_ACCOUNTS': 'local-research',
    })
    # Retain real factory persistence; inject only synthetic business operations
    # and the finalization fault. No strategy collection or dispatch is allowed.
    with mock.patch.object(runtime_app, 'create_private_worker_app', lambda runtime: runtime), \
         mock.patch.object(runtime_app, 'init_observability'):
        runtime = runtime_app.create_environment_app()
    gateway = runtime.research_gateway
    gateway._authorize_assignment = authorize
    gateway._finalize_result = finalize
    gateway._after_complete = None
worker = TwinResearchWorker(gateway, worker_id='worker-'+phase)
try:
    if phase == 'first':
        try:
            worker.handle('restart-research', now=now, research=research)
        except TimeoutError as error:
            assert str(error) == 'injected finalization interruption'
        else:
            raise AssertionError('fault must be observed')
        assert jobs.get('restart-research').status == WorkerJobStatus.RUNNING
        assert receipts.get('restart-research').actual_tokens == 12
    else:
        result = worker.handle('restart-research', now=now+timedelta(seconds=126), research=research)
        assert result == {'status': 'completed', 'research_result_id': 'synthetic-result',
                          'usage_record_id': 'synthetic-usage'}
        assert jobs.get('restart-research').status == WorkerJobStatus.COMPLETED
        assert receipts.get('restart-research') is None
    print(json.dumps(counts, sort_keys=True))
finally:
    if mode == 'factory':
        runtime.jobs._client.close()
    client.close()
"""


class SqlResearchRecoveryTests(unittest.TestCase):
    def test_saved_result_replays_across_three_fresh_processes(self):
        self._assert_recovery("direct")

    def test_byom_factory_persists_results_in_maintenance_sql(self):
        self._assert_recovery("factory")

    def _assert_recovery(self, mode):
        with TemporaryDirectory() as directory:
            if mode == "factory":
                from datetime import datetime, timedelta, timezone

                config = Path(directory) / "config"
                config.mkdir()
                for name in ("twin.yaml", "models.yaml"):
                    shutil.copyfile(ROOT / "deploy" / "byom" / name, config / name)
                policy = yaml.safe_load((config / "twin.yaml").read_text(encoding="utf-8"))
                policy["research"]["price_valid_until"] = (
                    datetime.now(timezone.utc) + timedelta(days=1)
                ).isoformat()
                (config / "twin.yaml").write_text(yaml.safe_dump(policy), encoding="utf-8")
            env = {key: value for key, value in os.environ.items() if key.upper() in {
                "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG",
                "APPDATA", "LOCALAPPDATA", "USERPROFILE",
            }}
            env.update(PYTHONPATH=str(ROOT / "src"), FORESEA_DATASTORE_BACKEND="sql")
            observed = []
            for phase in ("first", "replay", "duplicate"):
                result = subprocess.run(
                    [sys.executable, "-c", PROBE, phase, directory, mode], cwd=ROOT,
                    env=env, capture_output=True, text=True, timeout=20,
                    creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                observed.append(json.loads(result.stdout))
            self.assertEqual(observed, [
                {"research": 1, "authorize": 1, "finalize": 1},
                {"research": 1, "authorize": 1, "finalize": 2},
                {"research": 1, "authorize": 1, "finalize": 2},
            ])


if __name__ == "__main__":
    unittest.main()
