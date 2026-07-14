#!/usr/bin/env python3
"""
Build a *blinded* human-annotation workbook for the LLM-judge reliability study.

The reviewer asked for a human validation sample: 100-200 rationales annotated
by a person for criterion alignment, evidence consistency, hallucination,
usefulness, and whether the rationale supports the numeric probability — then
compared against the two LLM judges (and the GPT/Gemma third judge).

This script does NOT produce any labels. It only assembles the material a human
needs and emits:

  analysis/manual_validation/human_annotation.html   self-contained annotator
                                                     (one card per rationale,
                                                     autosaves, exports CSV)
  analysis/manual_validation/human_annotation_blank.csv   spreadsheet template
  analysis/manual_validation/RUBRIC.md                    annotation guide

The workbook is *blinded*: it shows the question, resolution criteria, the
evidence the model saw, the model's answer + probability, and the rationale —
but NOT the ground-truth outcome and NOT any LLM-judge score, so the human
labels stay independent of the systems being validated.

Annotate, click "Download CSV", save it as
``analysis/manual_validation/human_annotations.csv``, then run
``scripts/score_human_annotation.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
SAMPLE_PATH = ROOT / "analysis" / "manual_validation" / "sample.jsonl"
OUT_DIR = ROOT / "analysis" / "manual_validation"

# The five human criteria. Each is scored No (0) / Partial (0.5) / Yes (1).
# `hallucination` is phrased positively ("free of hallucination") so that, like
# the LLM judges' non_hallucination, a high score means a *clean* rationale.
CRITERIA = [
    ("criterion_alignment", "Criterion alignment",
     "Does the rationale address the question's actual resolution criteria "
     "(the specific event/threshold/date), rather than a vaguely related topic?"),
    ("evidence_consistency", "Evidence consistency",
     "Are the rationale's factual claims consistent with the provided evidence "
     "(no contradiction of what the sources say)?"),
    ("hallucination", "Free of hallucination",
     "Does the rationale avoid asserting facts, numbers, or events that are NOT "
     "in the provided evidence? Yes = clean, No = it invents unsupported facts."),
    ("usefulness", "Usefulness",
     "Would this rationale genuinely help a human forecaster understand and "
     "sanity-check the forecast?"),
    ("probability_support", "Supports the probability",
     "Does the rationale justify the *specific* probability shown (e.g. 85% vs "
     "60%), not merely the direction (Yes/No)?"),
]


def load_sample() -> list[dict]:
    return [json.loads(line) for line in SAMPLE_PATH.read_text().splitlines() if line.strip()]


def index_dataset() -> dict[str, dict]:
    data = json.loads(DATASET_PATH.read_text())
    return {str(r.get("id")): r for r in data}


def evidence_items(record: dict, max_articles: int = 5, max_chars: int = 500) -> list[dict]:
    out = []
    for art in (record.get("news_articles") or [])[:max_articles]:
        snippet = (art.get("summary_llm") or art.get("summary") or art.get("text") or "").strip()
        snippet = " ".join(snippet.split())
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "…"
        out.append({
            "source": art.get("source") or art.get("authors") or _domain(art.get("url", "")),
            "title": (art.get("title") or "").strip(),
            "date": (art.get("publish_date") or "")[:10],
            "snippet": snippet,
        })
    return out


def _domain(url: str) -> str:
    try:
        return url.split("//", 1)[-1].split("/", 1)[0].replace("www.", "")
    except Exception:
        return ""


def build_items(sample: list[dict], ds: dict[str, dict]) -> list[dict]:
    items = []
    for rec in sample:
        ds_rec = ds.get(str(rec["id"]), {})
        crit = (ds_rec.get("resolution_criteria") or ds_rec.get("description") or "").strip()
        crit = " ".join(crit.split())
        if len(crit) > 700:
            crit = crit[:700].rstrip() + "…"
        items.append({
            "id": rec["id"],
            "question": rec.get("question", ""),
            "resolution_criteria": crit,
            "predicted_answer": rec.get("predicted_answer", ""),
            "confidence": rec.get("confidence"),
            "rationale": (rec.get("rationale") or "").strip(),
            "evidence": evidence_items(ds_rec),
            # NB: ground_truth / correct / *_scores are intentionally omitted (blinding).
        })
    return items


def write_blank_csv(items: list[dict]) -> Path:
    cols = ["id"] + [c[0] for c in CRITERIA] + ["notes"]
    lines = [",".join(cols)]
    for it in items:
        lines.append(str(it["id"]) + "," * (len(cols) - 1))
    path = OUT_DIR / "human_annotation_blank.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_rubric() -> Path:
    lines = [
        "# Human annotation rubric — LLM-judge validation",
        "",
        "Annotate each rationale on five criteria. Use the same three-point scale "
        "for every criterion:",
        "",
        "- **Yes = 1.0** — clearly satisfies the criterion",
        "- **Partial = 0.5** — partially / arguably",
        "- **No = 0.0** — does not satisfy it",
        "",
        "You are scoring the **rationale**, given the question, its resolution "
        "criteria, the evidence the model saw, and the model's answer + "
        "probability. You are *not* scoring whether the final answer was right — "
        "the true outcome is deliberately hidden so your judgement of the "
        "reasoning stays independent.",
        "",
        "## Criteria",
        "",
    ]
    for key, label, desc in CRITERIA:
        lines.append(f"- **{label}** (`{key}`): {desc}")
    lines += [
        "",
        "## Workflow",
        "",
        "1. Open `human_annotation.html` in a browser.",
        "2. Score all items (progress autosaves to the browser).",
        "3. Click **Download CSV** and save as `human_annotations.csv` in this folder.",
        "4. Run `python scripts/score_human_annotation.py`.",
        "",
        "Prefer a spreadsheet? Fill in `human_annotation_blank.csv` instead "
        "(same columns, values 0 / 0.5 / 1).",
    ]
    path = OUT_DIR / "RUBRIC.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_html(items: list[dict]) -> Path:
    data_json = json.dumps(items, ensure_ascii=False)
    crit_json = json.dumps([{"key": k, "label": lab, "desc": d} for k, lab, d in CRITERIA], ensure_ascii=False)
    tmpl = _HTML_TEMPLATE
    tmpl = tmpl.replace("/*__DATA__*/", data_json)
    tmpl = tmpl.replace("/*__CRITERIA__*/", crit_json)
    path = OUT_DIR / "human_annotation.html"
    path.write_text(tmpl)
    return path


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Human annotation — rationale validation</title>
<style>
  :root { --bg:#fff; --ink:#111; --dim:#666; --line:#e3e3e8; --paper:#f6f6f8; --acc:#0f766e; }
  * { box-sizing: border-box; font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
  body { margin:0; background:var(--paper); color:var(--ink); }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
           padding:12px 20px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; z-index:5; }
  header h1 { font-size:15px; margin:0; }
  header .sp { flex:1; }
  .bar { height:6px; background:var(--line); border-radius:999px; width:220px; overflow:hidden; }
  .bar > span { display:block; height:100%; background:var(--acc); width:0; }
  button { font-size:13px; padding:7px 14px; border-radius:8px; border:1px solid var(--line);
           background:#fff; cursor:pointer; }
  button.primary { background:var(--ink); color:#fff; border-color:var(--ink); }
  .wrap { max-width:860px; margin:0 auto; padding:20px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-bottom:18px; }
  .card.done { border-color:var(--acc); }
  .qid { font-size:11px; color:var(--dim); letter-spacing:.04em; }
  .q { font-size:17px; font-weight:600; margin:4px 0 10px; }
  .meta { font-size:12px; color:var(--dim); margin-bottom:12px; }
  .lab { font-size:10px; text-transform:uppercase; letter-spacing:.12em; color:var(--dim); margin:14px 0 6px; }
  .crit-box, .evi, .rat { font-size:14px; line-height:1.55; }
  .evi-item { border-left:3px solid var(--line); padding:4px 0 4px 12px; margin:8px 0; }
  .evi-item .src { font-weight:600; }
  .evi-item .snip { color:#333; }
  .rat { background:var(--paper); border-radius:8px; padding:12px 14px; white-space:pre-wrap; }
  .crit { display:flex; align-items:flex-start; gap:12px; padding:10px 0; border-top:1px solid var(--line); }
  .crit:first-of-type { border-top:0; }
  .crit .info { flex:1; }
  .crit .info b { font-size:13px; }
  .crit .info span { display:block; font-size:12px; color:var(--dim); margin-top:2px; }
  .opts { display:flex; gap:6px; flex-shrink:0; }
  .opts button { min-width:64px; }
  .opts button.sel[data-v="1"] { background:#0f766e; color:#fff; border-color:#0f766e; }
  .opts button.sel[data-v="0.5"] { background:#a16207; color:#fff; border-color:#a16207; }
  .opts button.sel[data-v="0"] { background:#991b1b; color:#fff; border-color:#991b1b; }
  textarea { width:100%; margin-top:10px; border:1px solid var(--line); border-radius:8px;
             padding:8px; font-size:13px; resize:vertical; }
  .hint { font-size:12px; color:var(--dim); }
</style>
</head>
<body>
<header>
  <h1>Human annotation · rationale validation</h1>
  <span class="hint" id="count">0 / 0</span>
  <div class="bar"><span id="prog"></span></div>
  <span class="sp"></span>
  <input id="annotator" placeholder="Your name/initials" style="padding:6px 10px;border:1px solid var(--line);border-radius:8px;font-size:13px" />
  <button onclick="resetAll()">Reset</button>
  <button class="primary" onclick="downloadCsv()">Download CSV</button>
</header>
<div class="wrap">
  <p class="hint">Score each rationale on the five criteria. <b>Yes</b>=1, <b>Partial</b>=0.5, <b>No</b>=0.
  The true outcome and the LLM-judge scores are intentionally hidden. Progress autosaves in this browser.</p>
  <div id="cards"></div>
  <div style="text-align:center;padding:24px 0">
    <button class="primary" onclick="downloadCsv()">Download CSV</button>
  </div>
</div>
<script>
const DATA = /*__DATA__*/;
const CRITERIA = /*__CRITERIA__*/;
const LSKEY = "human_annot_v1";
let store = JSON.parse(localStorage.getItem(LSKEY) || "{}");

function save() { localStorage.setItem(LSKEY, JSON.stringify(store)); refreshProgress(); }
function rec(id) { return (store[id] = store[id] || {}); }
function esc(s) { const d=document.createElement("div"); d.textContent=s==null?"":String(s); return d.innerHTML; }

function setVal(id, key, v, btn) {
  rec(id)[key] = v;
  const group = btn.parentElement;
  [...group.children].forEach(b => b.classList.toggle("sel", b === btn));
  const card = document.getElementById("card-" + id);
  card.classList.toggle("done", CRITERIA.every(c => rec(id)[c.key] !== undefined));
  save();
}
function setNote(id, v) { rec(id).notes = v; save(); }

function refreshProgress() {
  const done = DATA.filter(it => CRITERIA.every(c => (store[it.id]||{})[c.key] !== undefined)).length;
  document.getElementById("count").textContent = done + " / " + DATA.length;
  document.getElementById("prog").style.width = (100 * done / DATA.length) + "%";
}

function render() {
  const root = document.getElementById("cards");
  root.innerHTML = DATA.map((it, i) => {
    const conf = it.confidence == null ? "—" : Math.round(it.confidence * 100) + "%";
    const evi = (it.evidence || []).map(e =>
      `<div class="evi-item"><div class="src">${esc(e.source)}${e.date ? " · " + esc(e.date) : ""}</div>`
      + (e.title ? `<div>${esc(e.title)}</div>` : "")
      + (e.snippet ? `<div class="snip">${esc(e.snippet)}</div>` : "") + `</div>`).join("") || `<div class="hint">No evidence recorded for this item.</div>`;
    const crit = CRITERIA.map(c => {
      const cur = (store[it.id]||{})[c.key];
      const opt = v => `<button data-v="${v}" class="${cur===v?'sel':''}" onclick="setVal(${it.id},'${c.key}',${v},this)">${v===1?'Yes':v===0.5?'Partial':'No'}</button>`;
      return `<div class="crit"><div class="info"><b>${esc(c.label)}</b><span>${esc(c.desc)}</span></div>`
        + `<div class="opts">${opt(1)}${opt(0.5)}${opt(0)}</div></div>`;
    }).join("");
    return `<div class="card ${CRITERIA.every(c=>(store[it.id]||{})[c.key]!==undefined)?'done':''}" id="card-${it.id}">
      <div class="qid">#${i+1} · id ${it.id}</div>
      <div class="q">${esc(it.question)}</div>
      <div class="meta">Model answer: <b>${esc(it.predicted_answer)}</b> · probability <b>${conf}</b></div>
      <div class="lab">Resolution criteria</div><div class="crit-box">${esc(it.resolution_criteria) || '<span class="hint">n/a</span>'}</div>
      <div class="lab">Evidence the model saw</div><div class="evi">${evi}</div>
      <div class="lab">Rationale to score</div><div class="rat">${esc(it.rationale)}</div>
      <div class="lab">Your ratings</div>${crit}
      <textarea rows="2" placeholder="Notes (optional)" oninput="setNote(${it.id}, this.value)">${esc((store[it.id]||{}).notes||"")}</textarea>
    </div>`;
  }).join("");
}

function downloadCsv() {
  const cols = ["id", ...CRITERIA.map(c => c.key), "annotator", "notes"];
  const who = document.getElementById("annotator").value || "";
  const esc = s => { s = s==null?"":String(s); return /[",\\n]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s; };
  const rows = [cols.join(",")];
  for (const it of DATA) {
    const r = store[it.id] || {};
    rows.push([it.id, ...CRITERIA.map(c => r[c.key] ?? ""), who, r.notes ?? ""].map(esc).join(","));
  }
  const blob = new Blob([rows.join("\\n") + "\\n"], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "human_annotations.csv"; a.click();
}
function resetAll() { if (confirm("Clear all your ratings?")) { store = {}; save(); render(); } }

render(); refreshProgress();
</script>
</body>
</html>
"""


def main() -> None:
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"missing {SAMPLE_PATH} — run scripts/manual_validation_sample.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_sample()
    ds = index_dataset()
    items = build_items(sample, ds)
    html_path = write_html(items)
    csv_path = write_blank_csv(items)
    rubric_path = write_rubric()
    print(f"{len(items)} items → blinded annotation workbook")
    print(f"  {html_path.relative_to(ROOT)}   (open in a browser)")
    print(f"  {csv_path.relative_to(ROOT)}")
    print(f"  {rubric_path.relative_to(ROOT)}")
    n_evi = sum(1 for it in items if it["evidence"])
    print(f"  evidence attached for {n_evi}/{len(items)} items")


if __name__ == "__main__":
    main()
