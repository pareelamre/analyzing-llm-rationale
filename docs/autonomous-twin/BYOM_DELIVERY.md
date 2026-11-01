# BYOM twin delivery

The current deliverable is a tested, distributable **shadow** twin using an
operator-supplied OpenAI-compatible model. SCADS access and a developer-owned
model subscription are not required to install or test it. The original live
trading plan remains separate; its unfinished task cards are not relabeled done.

## Acceptance

| Requirement | Evidence command or artifact |
|---|---|
| Account accounting, execution and recovery work together | `python -m unittest tests.test_twin_byom_cycle -v` uses the actual maintenance-role environment factory, research parser, budget finalizer, continuation, risk evaluation, SQL stores and shadow execution for both venue fixture formats. The research worker runs in-process against the maintenance gateway; the actual provider connects to a localhost synthetic HTTP model using operator-configured URL and credential. This test does not exercise inter-worker HTTP. Interrupting finalization and opening new SQL clients must recover the accepted result without another model request. Cash plus fee-inclusive position basis equals the starting simulation cash; fees agree with the fill history. |
| Clean installation runs a synthetic shadow cycle | Build the existing CPU image, then run the network-disabled Linux acceptance commands below. The application is installed in the image; only tests and BYOM templates are mounted, so host `src/` cannot mask the installed code. |
| Operators configure their own endpoint and credentials | [BYOM installation](../../deploy/byom/README.md) and its private YAML/environment examples. The acceptance uses the configured URL and bearer credential through the actual OpenAI-compatible provider, rather than replacing the provider's HTTP client. |
| Durable restart and duplicate handling | `tests.test_twin_byom_cycle`, `tests.test_twin_byom_package` and `tests.test_twin_sql_recovery`: saved-result replay, unchanged account on duplicate delivery and fresh-process receipt recovery. The separate package test builds both role factories, checks authenticated HTTP completed-result replay and restarts actual loopback workers; it does not run the complete strategy cycle over inter-worker HTTP. |
| Installation, costs and limits are explicit | The installation guide documents model pricing, role credentials, resource limits, persistent SQLite backup and the boundaries below. |

Synthetic calibration, market books, rules and forecasts exist only in test
processes and temporary databases. They are **not evidence of forecasting skill,
venue compatibility or profitability**. No production synthetic-data switch is
introduced. Tests do not alter the production calibration or risk gates.

## Clean Linux image acceptance

From a fresh checkout with Docker Engine available, build locally:

```sh
docker build --target cpu -t foresea-twin:byom-acceptance .
docker run --rm --network none --read-only \
  --tmpfs /tmp --tmpfs /app/.cache:size=64m --tmpfs /app/.hf_cache:size=64m --cap-drop ALL \
  --security-opt no-new-privileges \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/verification \
  --mount "type=bind,src=$(pwd)/tests,dst=/verification/tests,readonly" \
  --mount "type=bind,src=$(pwd)/deploy/byom,dst=/verification/deploy/byom,readonly" \
  foresea-twin:byom-acceptance \
  python -m unittest tests.test_twin_byom_cycle tests.test_twin_byom_package tests.test_twin_sql_recovery -v
```

The container has no external network, model weights, venue credentials,
SCADS key, production state mount or cloud credentials. The local HTTP fixture
uses loopback, which remains available with Docker's `none` network. Its
simulation cash and synthetic histories are disposable. A nonzero test exit is
a failed installation check, not permission to start collection. Local image and
Compose verification ran on Linux/amd64; other architectures need their own
target-host smoke before use.

For a Python source installation instead, use an isolated Python 3.11 virtual
environment, run `pip install -e ".[serve]"`, and run the same unittest modules from the
checkout. Keep the source checkout: this distribution currently relies on its
baseline configs and templates. A standalone wheel without those files is not
a supported deployment. The Docker path is the verified Linux distribution.

## Cost and capability boundaries

- Building and testing locally requires CPU, disk and dependency downloads. It
  provisions no paid infrastructure and performs no paid model inference.
- The actual package is opt-in: it starts two private workers, with no public
  app or schedule. It uses an existing Linux host; each worker has a one-CPU,
  one-GiB ceiling. The model needs its own capacity or API account.
- An empty installation has no model work to recover. Do not dispatch or enable
  a schedule without a configured endpoint, reviewed model prices and budgets.
  There is no automatic paid fallback. Zero API prices do not mean hosting,
  hardware or electricity is free.
- State is an operator-owned SQLite volume. Existing private R2 backup storage
  is unchanged; this package neither migrates production state nor creates a
  bucket. Configure and verify a separate private backup before keeping state.
- The runtime remains shadow-only, with zero live capital and no live mandate.
  Real venue receipt/accounting imports, native settlement/collateral semantics,
  worker live selection and live release evidence remain unfinished. Historical
  loss, selling or settlement that cannot be fully reconciled blocks new risk.
- Ambiguous order delivery can pause work for operator reconciliation; the
  package must not infer an absent live order or automatically send another one.
  Saved research-completion recovery is separately tested and does not repeat
  accepted inference.
- A repeated research callback preserves the first continuation's ten-minute
  deadline. Delivery after that deadline returns `expired` and executes no
  strategy action. It does not extend the deadline or complete the old strategy
  run. Resume with fresh work after operator review; accepted research receipts
  remain durable, and expired work must not be treated as a successful cycle.

Real-provider compatibility, production IAM, forward performance and a
capital/loss/expiry-bounded live pilot are subsequent milestones. Installing or
passing this acceptance does not authorize them.

## Verification record — 2026-10-26

- Existing CPU image built locally; network-disabled, read-only Linux acceptance:
  **7 tests passed**, including both venue fixtures and three-process research
  receipt recovery. No host application source was mounted.
- Actual Compose services became healthy before and after restart. An isolated
  SQLite marker survived; the work queue stayed empty. The test containers and
  their specifically labeled disposable volume were removed.
- Full local unit suite before the final identity checks: **3,413 tests passed,
  268 emulator-dependent skips**. The additional final SQL concurrency test passed separately. Ruff, source
  release contracts and Graphify passed. Semgrep ran 151 Python rules on both
  changed Python files with no findings. Required PR CI remains separate evidence.
- The final reviewed source and bounded-cache configuration require fresh Linux
  acceptance and required PR CI before merge; those results are recorded with
  the delivery PR rather than assumed from the earlier image run.
