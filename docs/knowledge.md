# Knowledge base

Foresea's long-term memory: reports turned into dated, cited claims and a knowledge graph, so a
forecast on a long-horizon question can draw on what serious sources project (owner, 2026-11-17:
"as more documents get added foresea should become knowledgeable in long term horizon tasks").

## Where things live

- **Data:** the private repo `pareelamre/foresea-knowledge`. Reports are usually licensed, so
  their claims, passages and graph stay private. Nothing from it is committed here; the tests use a
  made-up fixture.
- **Engine:** `src/analyzing_llm_rationale/knowledge.py` (load, validate, index, search, ingest)
  and `scripts/knowledge.py` (CLI). Tests: `tests/test_knowledge.py`.
- **Forecasts:** `/predict` adds up to two knowledge-base articles (a source's best claims with
  page numbers) ahead of the news when `KNOWLEDGE_DIR` points at a clone of the data repo.
  `KNOWLEDGE_EVIDENCE=0` turns it off. Without `KNOWLEDGE_DIR` nothing changes. The articles carry
  only our own-words claims and page numbers, never report passages.
- **Live server:** `knowledge-sync.yml` (every 3 hours, or run it by hand after adding reports)
  validates the base, packs `entities.jsonl` + `sources/` into one archive and uploads it to the
  private Cloudflare R2 state bucket (`knowledge/knowledge.tar.gz`, via
  `scripts/knowledge_r2_sync.py`), then reads it back with the server's read-only key. `docker.yml`
  gives Cloud Run that read-only key as `KNOWLEDGE_R2_*`; the server downloads the archive into a
  temp folder and checks its ETag at most every `KNOWLEDGE_RESTAT_S` seconds (default 300), so new
  reports go live without a redeploy. If a fetch fails or a new copy has errors, the last good
  index keeps serving. The data never goes into the Docker image (it is also pushed to GHCR) or
  into Google Cloud storage (owner, 2026-11-17: no paid Google services for this). A local
  `KNOWLEDGE_DIR` that exists takes precedence over R2.
- **Edge board agent:** each market's research packet carries up to four current, cited claims
  (`knowledge_base`) that at least two retrievers agree on (`agent_research._market_packet`).
- **Fund:** every commodity pair trade carries the newest report's forecast-implied direction for
  the pair and whether it agrees with the trade (`fund_knowledge`; evidence only, sizes unchanged).
  `fund_live.json` lists the view for every pair under `knowledge`.
- **CI access (owner's one-time step):** the agent tick and the board/fund publish jobs check the
  data repo out into `knowledge-base/` when the repo secret `KNOWLEDGE_REPO_TOKEN` exists: a
  fine-grained GitHub token with read-only Contents access to `pareelamre/foresea-knowledge`.
  Without it the checkout is skipped and both jobs run as before. A missing secret or a failed
  checkout (expired token, no access to the repo) puts a "Knowledge base missing" warning on the
  run page; the job itself still goes on without the base.

## What a claim is

One atomic statement in our own words, with its source and page, a `kind` (projection, estimate,
fact, policy, finding, risk), the graph entities it is about, and for numbers its `metric`,
`value`, `unit`, `geo`, `horizon` (the year) and `scenario`. The data repo's README has the full
layout.

## Retrieval

Four ranked lists fused by reciprocal rank fusion:

1. **Lexical:** BM25 over claim text and context (SQLite FTS5, Porter stemming).
2. **Dense:** MiniLM similarity, when an embedder is loaded (the app's sentence-transformer, or
   the same model through transformers). Vectors are cached by text hash, so a rebuild only embeds
   new claims. The request path never loads a model.
3. **Graph:** claims about the entities a question names (matched by name and alias), then claims
   about their neighbours.
4. **Horizon:** projections and estimates for the years a question asks about (within two years).

Then:

- **No look-ahead:** every query takes `as_of`, and sources published later are invisible, so
  backtests stay honest. `/predict` uses today.
- **Knowledge moves on:** when a newer source projects the same metric, place, year and scenario,
  the older projection is demoted and shown as "Earlier: ..." beside the new number. The
  `timeline` command lists every revision.
- **Agreement:** forecasts only get claims that at least two retrievers agree on, so a question
  that merely shares a word with a report gets nothing.

## Adding a report

```bash
python scripts/knowledge.py add report.pdf --id <publisher>-<yyyy>-<mm>-<topic> \
  --title "..." --publisher "..." --published YYYY-MM-DD --llm
python scripts/knowledge.py validate
```

`--llm` extracts claims passage by passage with the self-hosted model and drops any claim whose
numbers are not in its passage. Review, then commit to the data repo, one report per commit.
