# Codex task backlog

Well-scoped, outsourceable tasks for the Foresea app. Each lists the goal, the
files to touch, and acceptance criteria. See `AGENTS.md` for setup/run/test.

Always run before pushing: `ruff check src tests` and `python -m unittest discover -s tests`.

---

## 1. Browser back-button navigation (History API)
**Goal:** Back button moves between views instead of leaving the site.
- `static/index.html`: views are toggled in JS (`launchApp`, `openTrackRecord`,
  `closeTrackRecord`) with no history entries, so the browser back button exits.
- Use `history.pushState`/`popstate`: pushing `app` on launch and `track` when the
  overlay opens; a `popstate` handler restores the prior view (landing ← app, and
  closes the track overlay). `replaceState` the initial `landing` state on load.
**Acceptance:** From the app, Back returns to the landing hero; with the track
overlay open, Back closes it; no double-handling or history loops.

## 2. Live forecast resolution → real user track record
**Goal:** Let *live* user forecasts eventually count in the track record.
- New: a resolution mechanism. Either (a) an authenticated admin route to mark a
  stored forecast's real outcome, or (b) a worker that re-checks Metaculus when a
  question's `resolve_time` passes.
- Persist live forecasts (currently only in browser localStorage) to a store the
  server owns — `src/analyzing_llm_rationale/db.py` already has a DuckDB
  `predictions` schema to build on.
- Feed resolved live forecasts into `scripts/build_track_record.py`.
**Acceptance:** A resolved live forecast appears in `/track-record` with the
correct hit/miss, scored by `metrics.py`. **Integrity rule: never show an
unresolved forecast as right/wrong.**

## 3. Multi-type calibration metrics + non-binary track record
**Goal:** Extend scoring beyond binary now that the app forecasts MC/numeric/date.
- `src/analyzing_llm_rationale/metrics.py`: add multi-class Brier / log-loss for
  multiple_choice and CRPS (or interval coverage) for numeric/date.
- Only surface these publicly once resolved non-binary data exists (depends on #2).
**Acceptance:** Unit tests for each new metric; track record stays binary-only
until real non-binary outcomes are available.

## 4. Multi-type support in the batch pipeline
**Goal:** The research pipeline (`run-batch`) is still binary-only; bring it to
parity with the server's typed forecasting.
- `src/analyzing_llm_rationale/pipeline.py` (`parse_model_response`,
  `build_user_prompt`), `configs/variants.yaml`, `prompts/`.
- Mirror the typed-JSON approach in `server.py` (`_typing_instruction`,
  `_build_typed_response`).
**Acceptance:** `run-batch` can produce MC/numeric forecasts; tests cover parsing.

## 5. Tests for the typed `/predict` paths
**Goal:** `tests/test_server.py` only covers the binary path.
- Add cases for `multiple_choice` (options + probabilities) and `numeric`/`date`
  (range_forecast), plus the `question_type`/`options` request fields.
**Acceptance:** New tests pass; coverage for `_build_typed_response`.

## 6. Accessibility pass
**Goal:** Keyboard + screen-reader support.
- `static/index.html`: focus trap + `Esc` to close the track overlay and mobile
  sidebar; ARIA roles/labels on nav, dialog, buttons; visible focus rings;
  `aria-live` on the message thread for new answers.
**Acceptance:** Overlay is keyboard-navigable and `Esc`-closable; Lighthouse a11y ≥ 95.

## 7. Auto-regenerate the track record in CI
**Goal:** Keep `static/track_record.json` fresh.
- `.github/workflows/`: run `python scripts/build_track_record.py` when results
  change (or on a schedule) and commit the JSON.
**Acceptance:** Track record updates without a manual run.

## 8. Shared rate limiting (only needed if scaling past 1 instance)
**Goal:** `server.py` `_RateLimiter` is per-process; Cloud Run is capped at
`--max-instances 1` today, so it's authoritative. If that cap is raised, move the
limiter to a shared store (e.g., Cloud Memorystore/Redis) or Cloud Armor.
**Acceptance:** Rate limit holds across instances.

## 9. Housekeeping
- Add `email-validator` to deps (silences a FastAPI startup warning from the
  contact email metadata), or drop the email from `app` contact.
- Decommission the unused Vertex AI endpoint if it's still running (~$48/mo idle).
- Wire the custom domain (`foresea.ai`) to Cloud Run once purchased
  (`gcloud beta run domain-mappings create ...`).
