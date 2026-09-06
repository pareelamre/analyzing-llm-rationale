# Copy-paste prompt for the cheaper implementation model

Use this prompt in the Foresea repository. The coding model can be whichever lower-cost model the user chooses; no expensive model is required by this plan.

```text
Implement Foresea's autonomous trading twin according to
docs/autonomous-twin/PLAN.md and docs/autonomous-twin/TASKS.md.
The target is an autonomous Foresea agent, not a personal imitation of a user.

Begin by reading AGENTS.md, this executor file and progress.json. On the first
session read PLAN.md once, then T00. On later sessions read only the relevant
PLAN sections, the earliest ready task card and its evidence/dependencies.

Do one bounded task per session. Split a card marked L into its named substeps
if necessary; do not label the card complete until all of its acceptance criteria
pass. Use a codex/ branch/worktree and preserve unrelated local changes.
Revalidate current source against the planning baseline before editing. Follow
the repo's Graphify instructions; use scoped queries and targeted source reads.

Treat the architecture and contracts in PLAN.md as decisions already made.
Reuse the existing venue, authentication, KMS, trading and evaluation code.
Do not rewrite server.py or build a second execution stack. New capability must
go through the shared deterministic policy, reservations and durable commands.
Keep public/manual confirmation requirements and benchmark shadow-only tests.
Do not generate a human confirmation phrase for the autonomous agent. Implement
the separate server-verified mandate authority described in the plan.

For this task, first name its intended behavior, touched files, risks and test
cases in a brief progress message. Then implement it and run the stated checks.
Use fake venue/model clients and injected clocks for deterministic tests.
Never send live orders from tests or ordinary smoke checks, even if credentials
exist in the environment. Do not print or persist secrets.

When adding a new business operation, read the applicable local OTel skill and
follow native instrumentation conventions without introducing a paid Superlog
dependency. Apply user instructions over conflicting optional skill suggestions.

Run the task's focused tests and repository-required full tests/lint before
committing; run Graphify update after source changes. For financial/concurrency
tasks, prove the invariant with behavior tests and required emulator runs.
Keep tests honest: no silent skips, changed assertions just to obtain green,
weakened risk limits, or marking an HTTP acknowledgement as a fill.

For committing/publishing, follow the available GitHub publish skill and repo
workflow. Stage only task files and authorized generated outputs. Protected
main changes go through a PR. Implementation authorization does not authorize
enabling live trading, adding capital or changing account access.

Update progress.json with the task/substep, commit or PR, exact checks and
exit codes, evidence paths, remaining blockers and next ready task. If work is
uncommitted, say so. End with a concise handoff: completed behavior, verification,
limitations and the next task. Do not repeat the entire plan in the handoff.

For a genuine blocker, inspect the relevant source/tests and official venue
contract, try at most two focused hypotheses, then document the failing fixture
and exact unresolved question. Do not automatically switch to a costly model,
expand the scope, or keep retrying the same failure. Continue independent work
where dependencies permit. Ask only for information actually required by a gate.

All runtime configuration stays shadow with live disabled until the exact G0-G2
evidence and owner-approved mandate in T22 exist. A completed implementation can
legitimately have a collecting/ineligible strategy gate. Never promise profit or
lower a gate to call the project finished.
```

## Small context packet for resumed sessions

Read `progress.json`, the current task card, only the linked PLAN sections, and the files/test fixtures named by the card. Full repository rescans and re-reading every transcript are unnecessary. Each task's context should include the current failing case and its acceptance criteria; keep long logs in evidence files with summaries.

Recommended end-of-session record:

```text
Task/substep:
Branch and commit/PR (or uncommitted files):
Behavior delivered:
Tests run and exit codes:
Tests not run and why:
Evidence paths:
Remaining invariant or external gate:
Next ready task:
```

Tasks T01, T04, T14-T15 and T20 require a separate focused review pass. Use the cheaper model with the diff and adversarial cases, not a request to reread the whole repo. Human review at live activation concerns real authority and capital; expensive-model review is not a replacement for that approval.

## Stop conditions

Stop a dependent step when a financial invariant fails, durable storage is absent for live mode, account identity is uncertain, venue eligibility is unverified, a submission cannot be reconciled, or live authority is missing. Record the exact blocker and continue only work that cannot weaken that boundary. These conditions do not prevent completing offline code, test fixtures, documentation or a shadow deployment.

No recurring automation is created by this plan. If the owner later authorizes running the forward trial, use a bounded runtime report rather than keeping a coding model continuously active to poll it.


