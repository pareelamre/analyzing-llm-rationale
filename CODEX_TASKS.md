# Codex task backlog

Well-scoped, outsourceable tasks for the Foresea app. Each lists the goal, the
files to touch, and acceptance criteria. See `AGENTS.md` for setup/run/test.

Always run before pushing: `ruff check src tests` and `python -m unittest discover -s tests`.

> **Shipped recently (do not redo):** typed `/predict` (binary/MC/numeric/date) +
> tests (#5); Redis-ready shared cache + rate limiter via `REDIS_URL` (#8);
> Cloud Run scaled to max-instances 20 / concurrency 40 / 1Gi; Google + GitHub +
> email/password auth; live market fetch (`/markets/*`) + edge scan; agent layer
> (`/agent/analyze`, `/agent/scan`, custom skills, multi-turn follow-ups);
> web-search evidence (SearXNG→Tavily→Serper→Brave, keyed) with Stooq removed;
> conversational agent UI with cited sources; **RAG knowledge base store +
> `/rag/*` endpoints (see #10).**

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

---

# RAG follow-ups (knowledge base shipped; integration pending)

The vector store + endpoints exist: `src/analyzing_llm_rationale/rag.py`
(embeddings/chunk/cosine/top_k, lazy-loaded MiniLM) and `server.py`
`_rag_add/_rag_search/_rag_documents/_rag_delete` (per-user `VectorChunk` on
Datastore, in-memory fallback) behind `POST /rag/ingest`, `GET /rag/search`,
`GET /rag/documents`, `DELETE /rag/documents`. Tests in `tests/test_rag.py` use a
fake embedder. Namespaces: `kb` (user docs), `evidence`, `forecasts`.

## 10. Deploy + verify RAG embeddings on Cloud Run  ⚠ do this first
**Goal:** Confirm the local MiniLM model loads in the CPU serve image within
memory limits. The serve image installs the `pipeline` extra (sentence-
transformers + CPU torch), but the model is never loaded today (`use_embeddings=
False`); RAG is the first code path that loads it.
- After deploy, sign in and `POST /rag/ingest {"text": "..."}` then `GET
  /rag/search?q=...`; watch Cloud Run logs/memory.
- If it OOMs at 1Gi: either bump memory (`gcloud run services update ... --memory
  2Gi`) or swap `rag.embed()` to an OpenAI-compatible `/embeddings` API
  (e.g. SCADS or BYOK) to avoid loading torch in-process.
**Acceptance:** ingest+search work live without OOM; cold-start latency noted.

## 11. Use the knowledge base as forecast evidence
**Goal:** For signed-in users, retrieve top `kb` chunks for the question and merge
them into evidence in `/predict` and `/agent/analyze`.
- `server.py`: add `_optional_session(request)` (decode bearer if present, else
  None). In `predict()`, after evidence is gathered, if signed in, prepend
  `_rag_search(user_id, "kb", req.question, k=3)` hits as articles
  (`source="Knowledge base"`, set `relevance_score=score`). The agent calls
  `predict()` without `request` — thread the user id/claims through so agent
  forecasts also see the KB (avoid double rate-limiting).
**Acceptance:** A user who ingested a doc sees it cited in a forecast; anonymous
requests are unchanged; tests cover the merge.

## 12. Auto-index forecasts + fetched evidence (use cases 2 & 3)
**Goal:** Populate the `forecasts` and `evidence` namespaces automatically.
- After a forecast for a signed-in user, `_rag_add(user_id, "forecasts", [...])`
  with the question + rationale; optionally `_rag_add(user_id, "evidence", ...)`
  for fetched articles (dedupe by URL). Do this off the request path / fire-and-
  forget so it doesn't add latency (embedding cost).
**Acceptance:** Past forecasts are semantically searchable via
`/rag/search?namespace=forecasts`; ingest doesn't slow `/predict`.

## 13. Knowledge-base UI (Config panel)
**Goal:** Let users manage their KB in the app, not just the API.
- `static/index.html`: in the **Config** panel add a "Knowledge" section —
  attach text/PDF/URL (reuse the 📎 extract flow → `POST /rag/ingest`), list docs
  (`GET /rag/documents`), delete (`DELETE /rag/documents`). Show a small "used N
  KB sources" note when a forecast cites the KB.
**Acceptance:** A signed-in user can add/list/remove KB docs from the sidebar and
see them used in answers.

## 14. Scale the vector store when corpora grow
**Goal:** Datastore + in-Python cosine is brute-force (fetches all user chunks).
Fine for small KBs; if a user exceeds a few thousand chunks, move to a real ANN
index (pgvector on Cloud SQL, or Vertex AI Vector Search). Keep `rag.py`'s
interface so only the persistence layer changes.
**Acceptance:** Search stays fast (<300ms) at 10k+ chunks.

## 15. Rotate leaked credentials
The Tavily, Serper, and GitHub OAuth secrets were pasted in chat during setup.
Rotate them (new keys/secret → `gcloud secrets versions add ...`) when convenient.
