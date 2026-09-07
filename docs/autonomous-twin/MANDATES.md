# Autonomous mandate API

Autonomous mandates are owner-scoped, expiring grants for Foresea's strategy worker. They are separate from manual trade approvals. Creating a draft never activates it.

1. `POST /twin/mandates` creates a draft. The authenticated session supplies the owner; the server supplies the account identity, account epoch, venue, model, configuration, release and readiness hashes. Authority fields in the request are rejected.
2. The owner reviews the returned authority fields and their SHA-256 digest.
3. `POST /twin/mandates/{id}/approve` must submit that exact digest as `expected_hash` and an idempotency key. Approval fails if the account epoch or any server-derived hash changed, or if the mandate expired.
4. `POST /twin/mandates/{id}/revisions` creates an immutable new version and clears approval. It never expands or renews an existing approval.
5. `POST /twin/mandates/{id}/revoke` is idempotent. Revocation blocks unsent autonomous commands immediately; read-only reconciliation remains available.

Mandates allow only bounded prediction-market order actions. They cannot authorize withdrawals, credential changes, arbitrary HTTP requests, public MCP execution or self-renewal. Global, account, strategy and venue pause state is checked again at command submission.

Shadow mandates may be approved for registered shadow accounts. Live approval additionally requires a fresh, integrity-checked readiness artifact matching the reviewed release and configuration. The current T12 artifact is deliberately ineligible for live trading, so these endpoints do not enable live execution.
