# BYOM delivery review — 2026-10-26

Single policy-required deep audit: `aud_GpuamJPUE3`, terminal **partial**:
four usable reviewers, one failed reviewer. This is not a clean full-panel
verdict. Artifact SHA-256:
`9a006de426112c38f781a045e319dd6065b01a0e850a7b78353f195e2e209031`.
Result payload SHA-256:
`90ef9bbfe4f7411ddc8c71f44120013cbc486fcb0c912098a46b4027c9c2b14d`.
Ten source findings consolidate into seven rows. No second audit was started.

| finding | decision | action | verification |
|---|---|---|---|
| B01: OpenAI f1; Claude f1/f6, major/one blocking: Compose/full-suite results were pending and delivery status overstated evidence, BYOM_DELIVERY.md and progress.json | accepted | Wait for actual results and distinguish image, Compose and PR evidence. Record the actual software scope; green required PR checks and merge remain delivery conditions. | Initial readonly image7 passed; actual Compose services healthy before/after restart, SQLite marker retained and empty queue confirmed. Local full3413 passed/268 emulator skips. Final reviewed image/Compose and required PR results are recorded on the delivery PR before merge. |
| B02: Claude f2, minor: complete-cycle test uses maintenance factory and in-process research worker, tests/test_twin_byom_cycle.py | accepted | Explicitly describe that composition. Separate it from the package test's research-role factory, authenticated HTTP completed-result replay and process restart coverage. Do not claim one full strategy cycle crosses inter-worker HTTP. | BYOM_DELIVERY.md acceptance rows name the exact test boundaries; test_twin_byom_package starts both roles and checks HTTP replay and role-secret authorization. |
| B03: Claude f3, minor: recovery beyond the first continuation deadline is unverified, runtime_app.py:_stage_strategy_continuation | accepted | Preserve the first deadline, test delivery after eleven minutes, and document the expired/no-action result and operator-reviewed fresh work requirement. Do not advertise completion of expired strategy runs. | test_continuation_retry_preserves_first_deadline_and_rejects_scope_collision calls the actual TwinWorker after expiry and asserts no maintenance operation, status expired and unchanged deadline. |
| B04: Claude f4, minor: unconstrained thread scheduling weakens the SQL race test, tests/test_twin_byom_cycle.py | accepted | Synchronize both clients before transaction entry. A barrier between absent read and put is invalid for this backend: datastore_sql.py uses BEGIN IMMEDIATE to serialize writers. Retain two independently constructed SQL clients and assert identical surviving jobs and one row. | test_concurrent_sql_continuations_keep_one_original_schedule; datastore_sql.py transaction BEGIN IMMEDIATE. This proves the SQL distribution path, not production Datastore concurrency. |
| B05: Claude f5; DeepSeek f1, minor: fallback omits schema and not_before from job identity, runtime_app.py duplicate continuation guard | accepted | Require schema_version and not_before equality as well as scope, kind and payload. Retain the first deadline and creation time. A matching strongly read stored job is the idempotency proof; missing, corrupt or conflicting records still raise. | Expanded conflict tests include not_before; WorkerJob rejects unsupported schema versions before storage; runtime guard explicitly compares both fields. |
| B06: DeepSeek f2, minor: conflict tests cover scope alone, tests/test_twin_byom_cycle.py | accepted | Add valid typed conflicts for kind, payload and not_before; assert the original stored record remains unchanged. Retain the unsupported-schema boundary check. | Expanded table-driven continuation conflict test and reviewed focused suite. |
| B07: Claude f7, minor: cache mounts hide potential assets and have no explicit memory cap, compose.yaml | accepted | Confirm Dockerfile copies no baked model cache, document the empty-cache assumption, and cap each cache tmpfs at64MiB. Keep HF/Transformers offline and the overall worker1GiB memory limit. | Existing Dockerfile does not download model weights; clean image and Compose checks exercise provider imports with the bounded cache mounts and no model inference. |

All accepted changes must pass the final focused/Linux acceptance and required
PR checks before merge. The delivery PR records those executable results; this
table records dispositions rather than substituting for them. No live release,
production IAM, venue receipt or profitability evidence is claimed.
