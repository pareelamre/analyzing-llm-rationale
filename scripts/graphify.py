#!/usr/bin/env python3
"""Turn this repo into an Obsidian vault — a knowledge graph of the codebase.

Parses every Python module (stdlib ``ast`` only, no deps), emits one markdown
note per module with ``[[wikilinks]]`` to the intra-repo modules it imports.
Open the output folder as an Obsidian vault and the Graph View renders the
dependency graph natively. Also writes a JSON node/edge file and a Graphviz
``.dot`` (render with ``dot -Tsvg`` if you have Graphviz) for non-Obsidian use.

Usage:
    python scripts/graphify.py [--out repo-graph] [--roots src scripts tests]
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]


def _module_id(path: Path) -> str:
    """Dotted module id for a file, treating ``src/`` as the package root."""
    rel = path.relative_to(ROOT).with_suffix("")
    parts = rel.parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _collect(roots: List[str]) -> Dict[str, Path]:
    modules: Dict[str, Path] = {}
    for root in roots:
        base = ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts or path.name == "__init__.py" and path.stat().st_size == 0:
                continue
            modules[_module_id(path)] = path
    return modules


def _resolve_import(node: ast.AST, current: str, known: Set[str]) -> Set[str]:
    """Map an import node to known intra-repo module ids it refers to."""
    out: Set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.name
            for cand in (name, name.rsplit(".", 1)[0]):
                if cand in known:
                    out.add(cand)
    elif isinstance(node, ast.ImportFrom):
        if node.level:  # relative import — resolve against current package
            base = ".".join(current.split(".")[: -node.level])
            mod = f"{base}.{node.module}" if node.module else base
        else:
            mod = node.module or ""
        candidates = {mod}
        for alias in node.names:  # `from pkg import submodule`
            candidates.add(f"{mod}.{alias.name}" if mod else alias.name)
        out |= {c for c in candidates if c in known}
    return out


def _analyze(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None
    doc = (ast.get_docstring(tree) or "").strip().split("\n")[0]
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    funcs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    lines = path.read_text(encoding="utf-8").count("\n") + 1
    return {"doc": doc, "classes": classes, "funcs": funcs, "imports": imports, "lines": lines}


def main() -> int:
    ap = argparse.ArgumentParser(description="Graphify the repo into an Obsidian vault.")
    ap.add_argument("--out", default="repo-graph", help="Output vault directory.")
    ap.add_argument("--roots", nargs="*", default=["src", "scripts", "tests"], help="Dirs to scan.")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    modules = _collect(args.roots)
    known = set(modules)

    edges: List[tuple] = []
    info: Dict[str, dict] = {}
    importers: Dict[str, Set[str]] = {m: set() for m in modules}
    for mid, path in modules.items():
        data = _analyze(path)
        if data is None:
            continue
        deps: Set[str] = set()
        for node in data["imports"]:
            deps |= _resolve_import(node, mid, known)
        deps.discard(mid)
        info[mid] = {**data, "deps": sorted(deps), "path": str(path.relative_to(ROOT))}
        for d in deps:
            edges.append((mid, d))
            importers.setdefault(d, set()).add(mid)

    # ---- per-module Obsidian notes ----
    for mid, d in info.items():
        body = [f"# `{mid}`", "", f"> {d['doc']}" if d["doc"] else "", "",
                f"- **Path:** `{d['path']}`",
                f"- **Lines:** {d['lines']} · **Classes:** {len(d['classes'])} · **Functions:** {len(d['funcs'])}",
                ""]
        if d["deps"]:
            body += ["## Imports", ""] + [f"- [[{dep}]]" for dep in d["deps"]] + [""]
        imp_by = sorted(importers.get(mid, set()))
        if imp_by:
            body += ["## Imported by", ""] + [f"- [[{m}]]" for m in imp_by] + [""]
        if d["classes"]:
            body += ["## Classes", ""] + [f"- `{c}`" for c in d["classes"]] + [""]
        if d["funcs"]:
            body += ["## Functions", ""] + [f"- `{f}()`" for f in d["funcs"]] + [""]
        (out / f"{mid}.md").write_text("\n".join(body), encoding="utf-8")

    # ---- index (Map of Content) ----
    by_pkg: Dict[str, List[str]] = {}
    for mid in sorted(info):
        by_pkg.setdefault(mid.rsplit(".", 1)[0] if "." in mid else mid, []).append(mid)
    idx = ["# Repo knowledge graph", "",
           f"{len(info)} modules · {len(edges)} import edges. Open this folder as an "
           "Obsidian vault and use **Graph View** (Ctrl/Cmd+G) to explore.", ""]
    for pkg, mods in sorted(by_pkg.items()):
        idx.append(f"## {pkg}")
        idx += [f"- [[{m}]] — {info[m]['doc']}" if info[m]["doc"] else f"- [[{m}]]" for m in mods]
        idx.append("")
    (out / "_index.md").write_text("\n".join(idx), encoding="utf-8")

    # ---- machine-readable + graphviz ----
    (out / "graph.json").write_text(json.dumps(
        {"nodes": [{"id": m, **{k: info[m][k] for k in ("doc", "lines", "path")}} for m in info],
         "edges": [{"from": a, "to": b} for a, b in edges]}, indent=2), encoding="utf-8")
    dot = ["digraph repo {", '  rankdir=LR; node [shape=box, style=rounded, fontsize=9];']
    for a, b in edges:
        dot.append(f'  "{a}" -> "{b}";')
    dot.append("}")
    (out / "graph.dot").write_text("\n".join(dot), encoding="utf-8")

    print(f"Graphified {len(info)} modules, {len(edges)} edges -> {out}")
    print(f"  Obsidian: open '{args.out}/' as a vault, then Graph View.")
    print(f"  Static:   dot -Tsvg {args.out}/graph.dot -o {args.out}/graph.svg  (needs Graphviz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
