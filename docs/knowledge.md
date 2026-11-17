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
  `KNOWLEDGE_EVIDENCE=0` turns it off. Without `KNOWLEDGE_DIR` nothing changes.
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
- **On topic:** an agreed claim must also share the question's subject: a topical entity (a
  commodity, technology, policy or event, not a country or region) or a topic word. Years, bare
  numbers and question words ("reach", "by December 31") never count as a lexical match, since the
  horizon list already covers years. Before this, "Will Bitcoin reach $130,000 by December 31,
  2026?" got four World Bank metal forecasts.

## Adding a report

```bash
python scripts/knowledge.py add report.pdf --id <publisher>-<yyyy>-<mm>-<topic> \
  --title "..." --publisher "..." --published YYYY-MM-DD --llm
python scripts/knowledge.py validate
```

`--llm` extracts claims passage by passage with the self-hosted model and drops any claim whose
numbers are not in its passage. Review, then commit to the data repo, one report per commit.

## Growing the base (owner, 2026-11-17)

Two jobs in the data repo keep the base growing toward what Foresea is asked:

- **Gap scan** (`scripts/knowledge_gaps.py`, weekly): asks the base about every open
  long-horizon question (track-record store and Edge board, closing in 30+ days, sports and
  entertainment left out) and every fund commodity pair, and writes `gaps/latest.md` with what
  it covers, what it misses, and where plain agreement would have passed off-topic claims.
  `gaps/history.jsonl` tracks coverage over time.
- **Watch list** (`scripts/knowledge_watch.py`, daily): `watchlist.json` in the data repo names
  the publishers (EIA, USDA, World Bank, IMF, Fed, ECB, IEA, RBI, Stanford AI Index, Epoch AI) and
  how to spot a new edition on each listing page. New editions go to `inbox.jsonl` and their files
  are downloaded for reading. The first check of a publisher is a baseline, so adding one never
  floods the inbox.
- **Crawler rules:** honest user agent (`ForeseaKnowledgeWatch/1.0`), robots.txt obeyed (RFC 9309:
  an unavailable robots.txt means no rules, a 401/403 means the site refuses us), 1.5 s between
  requests to a host, at most five new editions per source per run. A site that blocks the
  crawler is reported, never worked around (USDA's main site, CBO, OPEC and OECD did on
  2026-11-17; USDA's report archive at esmis.nal.usda.gov allows it).
