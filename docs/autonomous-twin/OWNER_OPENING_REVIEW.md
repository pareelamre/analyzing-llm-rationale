# Owner-reviewed opening: review closeout

One deep AQG audit: `aud_WMIHLEfdAB`, completed 2026-10-21. Five usable voices
across four vendors; one model fallback. The initial artifact produced 19 findings,
including five marked blocking. The 15 consolidated dispositions below cover
every finding. This is remediation/adjudication of that audit, not a second panel
or a claim that the panel independently executed the tests or reviewed the fixes.

| finding | decision | action | verification |
|---|---|---|---|
| Voice0/f1 major: owner cannot inspect the hashed scope/snapshot | accepted | Owner-only review now returns full scope and normalized snapshot, with reproducible canonical JSON hashes; private review data is retained in the approval | `test_review_exposes_the_identity_and_snapshot_behind_its_hashes` on SQL and emulator |
| Voice0/f2 and Voice4/f1: foreign versus missing scope oracle | accepted | Use existing `find_account_scope`; missing and foreign scope produce identical403 responses, while backend failures remain503 | `test_anonymous_and_other_owner_cannot_read_or_confirm` |
| Voice0/f3: competing approvals not tested | accepted | Add distinct-approval and identical-approval races across independent emulator clients; one durable opening, stable retries and conflicting evidence rejected | `test_competing_distinct_confirmations_create_one_opening`, `test_concurrent_duplicate_confirmation_has_safe_retry` |
| Voice1/f1,f3 major: emulator imports missing and reported runs impossible | rejected | Supplied audit excerpt omitted module imports; the actual module imports `opening`, `mock`, `uuid4`, `Event`, `ThreadPoolExecutor` and all bases. The reviewer inferred absent imports from an excerpt | Existing module imports plus actual focused70,74,224,228 test runs and full3363 run before final remediation; fresh final checks recorded in progress/PR |
| Voice1/f2 and Voice3/f4 major: synchronous RPCs on async event loop | accepted | New handlers are plain synchronous FastAPI functions and use its worker threadpool | Owner HTTP/SQL/emulator tests and actual application route wiring |
| Voice1/f4 and Voice3/f3: session401 masked as503 | accepted | Preserve framework HTTPException. This correction was made while the same audit ran | `test_session_resolver_http_rejection_is_preserved_before_storage`; actual anonymous server GET401; authenticated real resolver/factory test |
| Voice1/f5: zero holdings differ between review and journal | accepted | Reject nonzero basis on zero quantity; exclude zero holdings consistently from the attested/journal inventory while retaining full source snapshot | `test_zero_quantity_basis_must_be_zero_and_review_matches_opening` |
| Voice2/f1 major: changed scope can replay registered approval as valid | rejected | Existing journal load checks full opening.scope equality and register_account rejects same-epoch metadata replacement, so claimed200 replay was already blocked. Also added explicit proposal scope_ref comparison as defense in depth | `test_registered_approval_cannot_be_replayed_after_scope_corruption`; store scope checks and `DatastoreAccountJournal.load` |
| Voice3/f1 major: adopted historical activity remains permanently unexplained | accepted | Persist immutable identity-to-source hashes in JournalOpening. Reconciliation explains exactly matching order/fill groups, and blocks changed/new rows including additional fills under the same client identity | `test_adopted_history_reconciles_but_changed_or_added_rows_block`; journal adoption bounds and legacy fingerprint regressions |
| Voice3/f2: approval does not verify opening economics | accepted | Reconstruct attested opening from stored proposal and compare exact typed economics, source refs, scope, time and adopted groups; Decimal scale remains semantic | `test_registered_approval_checks_the_actual_opening_economics`, `test_decimal_scale_does_not_invalidate_a_registered_opening` |
| Voice3/f5: refresh procedure/window unclear | accepted | Document one existing private RECONCILE job and dispatcher procedure, prepare evidence first, preserve strict120-second source freshness and reject concurrent refreshes; no scheduling/activation added | Native maintenance runbook and existing source/root replacement/deadline tests |
| Voice3/f6: terminal FAILED Polymarket history cannot be adopted | accepted | Permit terminal FAILED rows only with zero fee, retaining their source evidence. Pending/unknown statuses still block and later row changes become drift | `test_terminal_failed_polymarket_history_can_be_adopted_without_a_fee`, `test_unconfirmed_polymarket_fill_blocks_opening` |
| Voice3/f7: command query spans all epochs | accepted | Clarify current deliberate restriction: first opening requires no local commands anywhere in the scope; reconnect/traded scope requires trusted transition/reconciliation. No history deletion or automatic epoch creation | Runbook restriction and existing local-command/refence tests; broader native recovery remains open under T06/T15 |
| Voice3/f8: emulator race is not production pessimistic-lock evidence | accepted | Narrow stale-race assertion to Aborted/Conflict; add duplicate/conflicting approval races. Read-only production check returned PESSIMISTIC; retain production/IAM release gate | Actual local emulator races; read-only database mode check2026-10-21; no production fault drill claimed |
| Voice3/f9: raw factory may bypass manual store guard | rejected | Existing helper only caches the durable store from _get_datastore; it has no independent enablement/approval gate. For consistency and reuse the new factory now uses that exact helper and its client | `_confirmed_manual_twin_store`; `test_actual_server_owner_resolver_and_coherent_store_wiring` |

The audit resolves this owner-baseline API only. It does not verify native receipt
authenticity, settlement allocations, collateral/netting, automatic capacity
refresh, operator desk UI, production IAM or a live trading mandate. Empty adoption
is omitted from encoded journals to preserve fingerprints of existing persisted
journals. Approval changes only the account revision; existing money and mandates
remain unchanged. Private evidence/source bodies are never public repo artifacts.
