"""Read-only native account capture for claimed private maintenance jobs.

The server supplies a scope-bound read plan. Task payloads cannot supply
credentials, provider URLs, wallet addresses or replacement account authority.
This operation persists observations; it never increases spendable capacity or
uses absence from an open-order endpoint to release an order reservation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from .account_journal import DatastoreAccountJournal, compare_journal_snapshot
from .account_store import DatastoreAccountSnapshotStore
from .models import AccountScope
from .reconcile import AccountReadAborted, synchronize_venue_account
from .store import DatastoreTwinStore
from .venue_account import VenueAccountReadPlan
from .worker import DatastoreWorkerJobs, WorkerDegraded, WorkerJob, WorkerJobKind, WorkerJobStatus

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
operations = metrics.get_meter(__name__).create_counter('twin.account.maintenance', unit='1')
_DEGRADED_REASONS = frozenset({
    'native_account_job_invalid', 'native_account_scope_changed', 'native_account_command_history_truncated',
    'native_account_job_authority_changed', 'native_account_authority_changed',
    'native_account_reader_binding_changed', 'native_account_read_incomplete',
    'native_account_generation_superseded', 'native_account_capture_unconfirmed',
})


@dataclass(frozen=True)
class AccountReadBinding:
    """Trusted server resolver's exact connection/account binding, never task data."""

    scope: AccountScope
    plan: VenueAccountReadPlan


class NativeAccountMaintenance:
    """Capture all native collections, then fence the durable snapshot commit."""

    def __init__(self, store: DatastoreTwinStore, snapshots: DatastoreAccountSnapshotStore,
                 jobs: DatastoreWorkerJobs, *, resolve_plan: Callable[[AccountScope], AccountReadBinding],
                 clock: Callable[[], datetime], max_capture_seconds: int = 90):
        if (not isinstance(store, DatastoreTwinStore)
                or not isinstance(snapshots, DatastoreAccountSnapshotStore)
                or not isinstance(jobs, DatastoreWorkerJobs)
                or store._client is not snapshots._client or store._client is not jobs._client):
            raise ValueError('native account persistence requires one coherent durable client')
        if type(max_capture_seconds) is not int or not 1 <= max_capture_seconds <= 90:
            raise ValueError('native account capture deadline must be within 1..90 seconds')
        self.store, self.snapshots, self.jobs = store, snapshots, jobs
        self.resolve_plan, self.clock = resolve_plan, clock
        self.max_capture_seconds = max_capture_seconds
        self.journals = DatastoreAccountJournal(store, clock=clock)

    @tracer.start_as_current_span('twin.account.maintenance', record_exception=False, set_status_on_exception=False)
    def __call__(self, job: WorkerJob) -> dict[str, object]:
        span = trace.get_current_span()
        try:
            started = self.clock()
            if (started.tzinfo is None or started.utcoffset() is None
                    or job.kind is not WorkerJobKind.RECONCILE
                    or set(job.payload) != {'account_epoch_id'}):
                raise WorkerDegraded('native_account_job_invalid')
            scope = self.store.account_scope(job.account_scope_id)
            projection = self.store.projection(scope.id)
            if (scope.environment != 'live' or scope.venue not in {'kalshi', 'polymarket'}
                    or type(scope.account_epoch) is not int
                    or job.payload['account_epoch_id'] != str(scope.account_epoch)
                    or scope.account_epoch != projection.account_epoch):
                raise WorkerDegraded('native_account_scope_changed')
            span.set_attribute('venue', scope.venue)
            commands = self.store.commands(scope.id, limit=200)
            if len(commands) >= 200:
                raise WorkerDegraded('native_account_command_history_truncated')
            prior = self.snapshots.load(scope.id)
            journal = self.journals.load(scope)
            generation = 1 if prior is None else prior.generation+1
            deadline = min(job.deadline, started+timedelta(seconds=self.max_capture_seconds))

            def check_processing() -> None:
                checked = self.clock()
                current_job = self.jobs.get(job.id)
                if (checked.tzinfo is None or checked.utcoffset() is None or checked < started
                        or checked >= deadline or job.status is not WorkerJobStatus.RUNNING
                        or job.lease_expires_at is None or checked >= job.lease_expires_at
                        or current_job != job):
                    raise WorkerDegraded('native_account_job_authority_changed')

            def validate() -> None:
                check_processing()
                if (self.store.account_scope(scope.id) != scope
                        or self.store.projection(scope.id) != projection
                        or self.journals.load(scope) != journal
                        or self.store.commands(scope.id, limit=200) != commands):
                    raise WorkerDegraded('native_account_authority_changed')

            validate()
            binding = self.resolve_plan(scope)
            if (not isinstance(binding, AccountReadBinding) or binding.scope != scope
                    or not isinstance(binding.plan, VenueAccountReadPlan) or binding.plan.venue != scope.venue
                    or binding.plan.account_binding != (scope.connection_ref, scope.venue_account_ref.lower())):
                raise WorkerDegraded('native_account_reader_binding_changed')

            def check_read() -> None:
                try:
                    check_processing()
                except WorkerDegraded as exc:
                    raise AccountReadAborted(exc.reason) from None

            guarded_plan = binding.plan.guarded(check_read)
            if (guarded_plan.venue != scope.venue
                    or guarded_plan.account_binding != (scope.connection_ref, scope.venue_account_ref.lower())):
                raise WorkerDegraded('native_account_reader_binding_changed')
            result = synchronize_venue_account(
                scope.id, generation=generation, account_epoch=scope.account_epoch,
                received_at=started, read_plan=guarded_plan,
                local_command_ids={command.client_order_id for command in commands}, previous=prior,
            )
            if result.issues or result.retained_previous or result.snapshot is None:
                raise WorkerDegraded('native_account_read_incomplete')
            # This callback runs before and after the write inside the snapshot
            # transaction. Its reads participate in Google commit conflict checks.
            compared = compare_journal_snapshot(result.snapshot, journal)
            persisted = self.snapshots.save(compared, validate=validate)
            if persisted != compared:
                raise WorkerDegraded('native_account_generation_superseded')
            response = {'status':'complete', 'account_generation':persisted.generation,
                        'account_epoch':persisted.account_epoch,
                        'blocks_new_exposure':persisted.blocks_new_exposure,
                        'ledger_reconciled':journal is not None and not persisted.blocks_new_exposure,
                        'orders_observed':len(persisted.orders), 'fills_observed':len(persisted.fills)}
            outcome = 'drift' if persisted.blocks_new_exposure else 'complete'
            operations.add(1, {'venue':scope.venue, 'outcome':outcome})
            span.set_attribute('outcome', outcome)
            logger.info('Native account capture committed generation=%s outcome=%s', persisted.generation, outcome)
            return response
        except (WorkerDegraded, AccountReadAborted) as exc:
            reason = str(exc) if str(exc) in _DEGRADED_REASONS else 'native_account_capture_unconfirmed'
            operations.add(1, {'outcome':'degraded', 'reason':reason})
            span.set_status(Status(StatusCode.ERROR, reason))
            logger.warning('Native account capture requires reconciliation reason=%s', reason)
            raise WorkerDegraded(reason) from None
        except Exception:  # aqg: top-level boundary -- credentials and provider payloads must not escape
            operations.add(1, {'outcome':'degraded', 'reason':'native_account_capture_unconfirmed'})
            span.set_status(Status(StatusCode.ERROR, 'native_account_capture_unconfirmed'))
            logger.warning('Native account capture requires reconciliation reason=native_account_capture_unconfirmed')
            raise WorkerDegraded('native_account_capture_unconfirmed') from None
