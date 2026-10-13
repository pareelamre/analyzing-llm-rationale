# Independent tournament trigger

Cloudflare runs this timer every ten minutes (UTC minutes 3/13/23/33/43/53).
It triggers only the existing four-bot `metaculus-futureeval.yml` workflow on main.
Existing workflow concurrency, submission safety, private comments and
`METACULUS_BOTS_ENABLED` remain authoritative. No public HTTP trigger is enabled.

Active-run checks are best-effort, not an atomic dispatch lock. A simultaneous
native cron or repeated timer delivery can queue another workflow run. The
workflow concurrency group serializes runs; each fresh runner restores the durable
write-ahead audit and checks the account's latest forecast and revision cooldown.
This deliberately accepts occasional redundant scans, not unguarded duplicate
submissions. Unknown previous submissions remain quarantined.

Use the free Workers plan; 144 invocations/day are below its 100,000/day request
limit. The free CPU limit still applies. No paid plan upgrade is necessary.
Structured logs distinguish `dispatched`, `skipped`, and `failure`. A dispatch
receipt means GitHub accepted the request, not that a forecast was submitted.
Inspect GitHub's four job results and verified bot audit histories separately.

The native workerd contract test validates request construction and the timer
handler without making network writes. Workers currently accepts `manual` or
`follow` redirect modes, not Node's `error` mode. We use `manual` and reject any
status other than the expected 200/204; credentials never follow redirects.
Failure stages are bounded labels, not upstream error messages or bodies.

Run `npm ci` then `npm test` here. CLI is pinned in the lockfile.
Use Node 22.18+ (Node 24 recommended) and
`npx cf workers versions create --dry-run` to validate packaging.
Create a private temporary JSON secrets file containing the
`METACULUS_GITHUB_DISPATCH_TOKEN` binding from your credential store, never from
chat. Restrict that file to the deploying user. Run
`npx cf deploy --secrets-file <private-path>` to upload and activate the code and
secret together and apply the configured schedule; remove the temporary file in
a finally block. Never pass the secret itself in command arguments.

Verify the active version with `npx cf workers deployments list` and the cron
through Cloudflare's `GET /accounts/{account_id}/workers/scripts/foresea-metaculus-dispatch/schedules`.
Check settings for the secret binding (name only), compatibility date and
observability. Confirm the workers.dev and preview URLs are disabled using
the script's `/subdomain` API. Watch `wrangler tail foresea-metaculus-dispatch`
for a real scheduled event, then correlate it with a new GitHub
`workflow_dispatch` run and inspect all four jobs. A deployment receipt or an
inactive uploaded version alone is not acceptance evidence.
Never commit secrets, private forecasts, reasoning or audit histories.

Rollback/stop: remove the scheduled trigger in Cloudflare, or set GitHub's
`METACULUS_BOTS_ENABLED=false` to stop live submission from every trigger.
Keep the native GitHub schedule as a secondary trigger. Do not recreate the
retired paid GCP dispatcher.
