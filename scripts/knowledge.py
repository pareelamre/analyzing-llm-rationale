"""Foresea knowledge base CLI: add reports, check the base, and query it.

The base is a private repo (pareelamre/foresea-knowledge) cloned at KNOWLEDGE_DIR, or beside
this repo as ../foresea-knowledge. See src/analyzing_llm_rationale/knowledge.py for the layout.

    python scripts/knowledge.py validate
    python scripts/knowledge.py stats
    python scripts/knowledge.py search "data center power demand by 2035" [--as-of 2026-12-01] [--dense]
    python scripts/knowledge.py timeline "data center electricity demand"
    python scripts/knowledge.py passages "Ercot large load queue"
    python scripts/knowledge.py evidence "Will data centers use 7% of world power by 2035?"
    python scripts/knowledge.py add report.pdf --id bnef-2026-11-x --title "..." --publisher BNEF \
        --published 2026-11-11 [--llm]

``add`` stores the source and its page-tagged passages; with ``--llm`` the self-hosted model
(KNOWLEDGE_LLM_URL or FUND_LLM_URL, key SELF_HOSTED_LLM_API_KEY) also extracts claims, keeping
only those whose numbers appear in the passage they came from.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402


def _root(args) -> Path:
    root = Path(args.dir).expanduser() if args.dir else kb.default_dir(allow_sibling=True)
    if root is None or not root.is_dir():
        sys.exit("No knowledge base: clone pareelamre/foresea-knowledge beside this repo or set KNOWLEDGE_DIR.")
    return root


def _index(args, corpus: kb.Corpus) -> kb.Index:
    embed = kb.default_embedder() if getattr(args, "dense", False) else None
    if getattr(args, "dense", False) and embed is None:
        print("(no embedder available: lexical, graph and horizon only)", file=sys.stderr)
    return kb.Index(corpus, embed=embed, vector_cache=corpus.root / ".kb" / "vectors.sqlite")


def _as_of(text: str | None) -> date | None:
    if not text:
        return None
    when = kb.parse_when(text)
    if when is None:
        sys.exit(f"--as-of must be YYYY-MM-DD, got {text!r}")
    return when


def cmd_validate(args) -> int:
    corpus = kb.load(_root(args))
    for error in corpus.errors:
        print(error)
    print(json.dumps(kb.stats(corpus)))
    return 1 if corpus.errors else 0


def cmd_stats(args) -> int:
    print(json.dumps(kb.stats(kb.load(_root(args))), indent=2))
    return 0


def cmd_search(args) -> int:
    corpus = kb.load(_root(args))
    index = _index(args, corpus)
    for hit in index.search(args.question, as_of=_as_of(args.as_of), k=args.k):
        print(f"{hit.score:.4f} [{'+'.join(hit.lists)}] {hit.claim['text']}  ({hit.cite()})")
    return 0


def cmd_passages(args) -> int:
    corpus = kb.load(_root(args))
    index = _index(args, corpus)
    for p in index.passages(args.question, as_of=_as_of(args.as_of), k=args.k):
        print(f"--- {p['id']} ({p.get('section') or ''})\n{p['text'][:900]}\n")
    return 0


def cmd_timeline(args) -> int:
    corpus = kb.load(_root(args))
    for row in _index(args, corpus).timeline(args.topic, as_of=_as_of(args.as_of)):
        where = f" {row['geo']}" if row.get("geo") else ""
        print(f"{row['metric']}{where} by {row['horizon']}: {row['value']} {row.get('unit') or ''} ({row['cite']})")
        for old in row["history"]:
            print(f"    earlier: {old['value']} ({old['cite']})")
    return 0


def cmd_evidence(args) -> int:
    corpus = kb.load(_root(args))
    articles = kb.Index(corpus).evidence_articles(args.question, as_of=_as_of(args.as_of) or date.today())
    print(json.dumps(articles, indent=2, ensure_ascii=False))
    return 0


def cmd_add(args) -> int:
    root = _root(args)
    path = Path(args.path).expanduser()
    pages = kb.pdf_pages(path) if path.suffix.lower() == ".pdf" else kb.split_pages(path.read_text(encoding="utf-8"))
    folder = root / "sources" / args.id
    folder.mkdir(parents=True, exist_ok=True)
    source_path = folder / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {"id": args.id}
    for key in ("title", "publisher", "published", "url", "drive_id", "summary"):
        if getattr(args, key, None):
            source[key] = getattr(args, key)
    if args.topics:
        source["topics"] = [t.strip() for t in args.topics.split(",") if t.strip()]
    source.setdefault("kind", "report")
    source.setdefault("added", date.today().isoformat())
    source_path.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    chunks = kb.chunk_pages(args.id, pages)
    kb.write_jsonl(folder / "chunks.jsonl", chunks)
    print(f"{args.id}: {len(pages)} pages, {len(chunks)} passages")

    if args.llm:
        call = kb.llm_from_env()
        if call is None:
            sys.exit("--llm needs KNOWLEDGE_LLM_URL or FUND_LLM_URL (and SELF_HOSTED_LLM_API_KEY).")
        corpus = kb.load(root)
        entities = dict(corpus.entities)
        claims = [c for c in corpus.claims if c.get("source") == args.id]
        relations = [r for r in corpus.relations if r.get("source") == args.id]
        seen = {kb.normalize(c["text"]) for c in claims}
        for n, chunk in enumerate(chunks, 1):
            try:
                found, rels = kb.extract(chunk, source, call)
            except Exception as exc:  # one failed call must not lose the rest of the report
                print(f"  {chunk['id']}: extraction failed ({type(exc).__name__})", file=sys.stderr)
                continue
            ids = []
            for claim in found:
                if kb.normalize(claim["text"]) in seen:
                    ids.append(None)
                    continue
                seen.add(kb.normalize(claim["text"]))
                claim["entities"] = sorted(set(kb.merge_entities(entities, claim["entities"]).values()))
                claim = {"id": f"{args.id}:c{len(claims) + 1:03d}", "source": args.id, **claim,
                         "extracted_by": source.get("extracted_by") or "self-hosted LLM"}
                claims.append(claim)
                ids.append(claim["id"])
            for rel in rels:
                names = kb.merge_entities(entities, [rel["subject"], rel["object"]])
                row = {"subject": names[rel["subject"]], "predicate": rel["predicate"], "object": names[rel["object"]]}
                cid = ids[rel["claim_index"]] if rel.get("claim_index") is not None else None
                row["claims"] = [cid] if cid else []
                relations.append(row)
            print(f"  {n}/{len(chunks)} {chunk['id']}: {len(found)} claims")
        kb.write_jsonl(folder / "claims.jsonl", claims)
        kb.write_jsonl(folder / "relations.jsonl", relations)
        kb.write_jsonl(root / "entities.jsonl", [entities[k] for k in sorted(entities)])
    return cmd_validate(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", help="knowledge directory (default: KNOWLEDGE_DIR or ../foresea-knowledge)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    for name, fn in (("search", cmd_search), ("passages", cmd_passages), ("evidence", cmd_evidence)):
        p = sub.add_parser(name)
        p.add_argument("question")
        p.add_argument("--as-of")
        p.add_argument("--k", type=int, default=8)
        p.add_argument("--dense", action="store_true", help="add MiniLM similarity (loads a small model)")
        p.set_defaults(fn=fn)
    p = sub.add_parser("timeline")
    p.add_argument("topic")
    p.add_argument("--as-of")
    p.add_argument("--dense", action="store_true")
    p.set_defaults(fn=cmd_timeline)
    p = sub.add_parser("add")
    p.add_argument("path", help="a PDF, or a text export with form feeds between pages")
    p.add_argument("--id", required=True)
    for key in ("title", "publisher", "published", "url", "drive-id", "summary", "topics"):
        p.add_argument(f"--{key}")
    p.add_argument("--llm", action="store_true", help="extract claims with the self-hosted model")
    p.set_defaults(fn=cmd_add)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
