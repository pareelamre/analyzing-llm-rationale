# Operator-owned model configuration for the shadow twin

This bundle replaces the twin's SCADS research model with an operator-supplied
OpenAI-compatible endpoint. It changes no public chat or forecasting model and
contains no credential. The twin remains in shadow mode with zero live capital.

1. Copy `twin.yaml` and `models.yaml` to a private directory outside the repo.
   Set `models.yaml`'s `local_model_name`, `router_model_name`, and
   `api_base_url` to the exact model identity and endpoint you operate. Both
   private worker roles must see the same operator-owned, read-only files at
   the same absolute path. The strategy, shadow-trial, and replay sections
   must match Foresea's bundled `configs/twin.yaml`; startup rejects changes
   to them. Refresh the copied bundle after a Foresea upgrade. Do not edit
   the shared `configs/models.yaml` to distribute a twin.
2. Set `FORESEA_TWIN_CONFIG_DIR` to that absolute directory on both workers.
   Set `FORESEA_TWIN_BYOM_API_KEY` only on the research worker. This exact name
   is required so a copied config cannot select another worker secret. A local server
   that ignores bearer tokens can use a non-secret placeholder. Do not commit
   a real key, put it in either YAML file, or expose it through the operator UI.
3. The example has a zero USD budget because it assumes a locally operated
   model with no metered API charge. Keep the token, request, timeout, and
   fanout ceilings. For a metered provider, enter its current input/output
   prices and an owner-approved positive daily USD limit before use. Unknown
   prices fail closed. A remote endpoint with zero listed prices also fails
   closed unless you explicitly set `zero_cost_remote_attested: true` after
   verifying that no API usage is billed. `price_valid_until` must be renewed
   with price review.
4. Keep the model endpoint private. `127.0.0.1` in the example means the
   worker's own network namespace; change it if the model runs elsewhere.
   The twin accepts plaintext HTTP only for loopback URLs; use HTTPS for any
   other host. Test model compatibility and usage accounting with synthetic requests in
   an isolated environment before any shadow collection.

The existing private worker can load this bundle, but this directory is **not
yet a standalone distributable twin**. Local worker transport uses separate
maintenance, research-dispatch, and research-gateway secrets over loopback or
HTTPS; see the VPS runbook. The VPS Compose stack does not start the private
twin roles. A self-hosted release still needs worker packaging, durable SQL
restart tests, and a bounded shadow trial. No deployment or model request is
part of this change. Live order submission remains gated by T21/T22 and an
explicit owner mandate; installing this config does not grant trading authority.

The operator-configured shadow worker reads Foresea's bundled, dated
`static/track_record_live.json`
to prioritize already discovered markets from domains whose published skill
interval has a positive lower bound. It uses the same question-domain tagger as
the published track record. The report's declared generation time must predate
the decision and be at most seven days old; otherwise discovery order is unchanged. A changed order
stores the artifact hash with the market capture. The hash identifies the
bytes used; it is not a producer signature or an archive. Updating this
bundled report requires a new build. This is a research-allocation prior,
not proof that the BYOM model is calibrated: live sizing still requires the
twin's own prospective, audit-grade resolved forecasts. The selected domain
mix can affect later evaluation, so compare calibration within domain and
selection cohort before promoting any live strategy. The publisher's domain
intervals are not corrected for comparing multiple domains; treat this as a
cheap search heuristic, not evidence of transferable edge. The bundled file
is trusted as part of the operator-owned code image; a signed or otherwise
attested report is still needed before using the prior in a live release.

If the two worker roles load different model or budget settings, the stored
research configuration ID differs and the research operation degrades before
calling the model. An existing queued assignment from the old ID likewise
fails closed after a configuration change; clear or reconcile stale work
before collecting new shadow evidence.
