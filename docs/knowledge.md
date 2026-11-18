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
- **On topic:** an agreed claim must also share the question's subject: a topical entity (a
  commodity, technology, policy or event, not a country or region) or a topic word. Years, bare
  numbers and question words ("reach", "by December 31") never count as a lexical match, since the
  horizon list already covers years. Before this, "Will Bitcoin reach $130,000 by December 31,
  2026?" got four World Bank metal forecasts.
- **Rate decisions:** "Will the Fed cut rates in December 2026?" names the Fed, not the federal
  funds rate. When a question names a bank and asks about its rates ("cut rates", "raise interest
  rates", "rate hike", "rates unchanged", "25 bps"), the bank's own policy rate joins its entities,
  and that rate's current claims lead, the asked year first. The base says which rate a bank sets
  through a `sets` relation (`rbi sets repo_rate`) or a rate alias built on the bank's name
  ("Fed rates" on the federal funds rate). In such a question "rate", "cut" and the like are not
  topic words, so one bank's rate path is never evidence for another's. Before this, the FOMC's
  unemployment projections led and the 2026 federal funds rate came eighth.

## Adding a report

```bash
python scripts/knowledge.py add report.pdf --id <publisher>-<yyyy>-<mm>-<topic> \
  --title "..." --publisher "..." --published YYYY-MM-DD --llm
python scripts/knowledge.py validate
```

`--llm` extracts claims passage by passage with the self-hosted model and drops any claim whose
numbers are not in its passage or that repeats eight words in a row from it (reports are often
licensed; the base keeps our own words). Review, then commit to the data repo, one report per
commit. For anything longer than a few pages, queue it for the night reader instead (below).

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
- **Table readers** (`scripts/knowledge_tables.py`, daily after the watch): for publishers whose
  forecasts are a table, a reader turns the current edition into claims with no model involved:
  EIA's Short-Term Energy Outlook (Brent, WTI, Henry Hub, US gasoline and diesel, power prices,
  US and world oil and gas output; yearly averages plus December values), the Fed's Summary of
  Economic Projections (FOMC medians, with the previous projection beside each), ECB staff
  projections, and the IMF World Economic Outlook for 13 economies (DataMapper API). Each reader
  uses one scenario name across editions, so a new edition supersedes the last on the same metric
  and year while other publishers stay beside it. Nothing is written unless enough values were
  found, each is in a plausible range, and none moved more than 60% (8 points for rates) from the
  publisher's previous edition; values the publisher already stated in the base are skipped. A
  failed reader is a run warning and leaves the base untouched. The fund's crude/natural-gas pair
  view now comes from the monthly EIA forecast.
- **Exchanges** (`knowledge_markets.py`, run by the same daily step, owner 2026-11-17: "add
  comex, and use exchanges as well"): the CFTC's weekly Commitments of Traders for COMEX gold,
  silver and copper and NYMEX WTI, Henry Hub, RBOB, ULSD, platinum and palladium (money managers'
  net position, its share of open interest, and the one- and four-week change; only the latest
  week is kept), and Shanghai Futures Exchange settlements for copper, aluminium, zinc, nickel,
  gold and silver (most traded, December and farthest contracts, as `estimate` claims that say they
  are futures prices, not forecasts; the last four weeks are kept, so the newest price shows with
  the previous ones as "earlier"). CME's own site (COMEX, NYMEX) does not answer the crawler, and
  ICE data, MCX, NSE and LBMA prices refuse it, so COMEX prices are not in the base; LME allows
  crawling but its prices are licensed, so it is left out.
- **Reading reports** (`scripts/knowledge_reports.py`, phase 4): on the Mac, launchd
  (`com.foresea.knowledge-reports`, every 30 minutes) queues new `kind: report` editions from the
  inbox, downloads the PDF (finding it on the edition page when the watch list has no file
  pattern) and reads it passage by passage with the on-device model, 23:00 to 07:00 local time,
  on AC power. The model belongs to the live site first: a passage starts only when no live
  request is in flight (connections into Caddy's port 8090) and none finished in the last two
  minutes (Caddy's access log), and a passage being read is dropped mid-answer the moment a live
  request arrives (the server stops work for a client that hangs up), to be read again later.
  Contents, reference lists and figure tables are skipped. Three gates: every number in a claim
  must be in its passage; a claim that repeats eight words in a row from the passage gets one
  chance to be restated by the model and must then pass both gates again; and a reviewer marks
  every candidate keep or drop (`sheet`, `decide`) before `publish` writes the source and marks
  the inbox row read. Only entities the base already knows, or that the reviewer's entity map
  (`entities`, then `publish --entities map.json`) names with a type, reach the graph: the model's
  vague names ("governments", "costs", "2026") would otherwise make unrelated claims look on topic.
  First report (2026-11-17): IEA *Sheltering from Oil Shocks*, 52 passages in about 20 minutes,
  220 candidates, 131 kept; the reviewer caught a reversed share, a wrong year, an invented
  horizon and many duplicates, so the review gate is not optional. Work in progress lives in the clone's gitignored
  `.kb/reports/<id>/`, so a stopped night resumes where it left off. Reports from publishers the
  watch list does not cover can be queued by hand (`queue file.pdf|URL --id ...`).
- **Crawler rules:** honest user agent (`ForeseaKnowledgeWatch/1.0`), robots.txt obeyed (RFC 9309:
  an unavailable robots.txt means no rules, a 401/403 means the site refuses us), 1.5 s between
  requests to a host, at most five new editions per source per run. A site that blocks the
  crawler is reported, never worked around (USDA's main site, CBO, OPEC and OECD did on
  2026-11-17; USDA's report archive at esmis.nal.usda.gov allows it).
