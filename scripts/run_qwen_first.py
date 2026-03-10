import os
import json
import pathlib
import requests
from datetime import datetime

base = pathlib.Path(r"C:\Users\paree\Documents\Analyzing Rationale of LLMs")
input_path = base / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
output_dir = base / "results" / "Qwen2-7b-instruct" / "temperature_00"
output_dir.mkdir(parents=True, exist_ok=True)

# load prompts
system_prompt = (base / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
neutral_prompt = (base / "prompts" / "variant0_neutral_baseline.txt").read_text(encoding="utf-8").strip()

# load first record
records = json.load(input_path.open("r", encoding="utf-8"))
rec = records[0]

question = (rec.get("question") or "").strip()
description = (rec.get("description") or "").strip()
resolution = (rec.get("resolution_criteria") or "").strip()

# build evidence summaries
articles = rec.get("news_articles") or []
summary_items = []
for art in articles:
    if not isinstance(art, dict):
        continue
    summ = art.get("summary_llm") or art.get("summary") or art.get("text")
    if not summ:
        continue
    title = art.get("title") or ""
    url = art.get("url") or ""
    publish_date = art.get("publish_date") or ""
    summary_items.append({
        "title": title,
        "publish_date": publish_date,
        "url": url,
        "summary": summ.strip()
    })

# Keep evidence concise: limit to top 5 summaries
summary_items = summary_items[:5]

# assemble user prompt
user_prompt = neutral_prompt.replace("[question]", "").strip()
parts = []
parts.append(f"Question: {question}")
if description:
    parts.append(f"Description: {description}")
if resolution:
    parts.append(f"Resolution Criteria: {resolution}")
parts.append("Evidence Summaries:")
if summary_items:
    for i, s in enumerate(summary_items, 1):
        meta = " | ".join([p for p in [s.get("title"), s.get("publish_date"), s.get("url")] if p])
        parts.append(f"{i}. {meta}\n{s['summary']}")
else:
    parts.append("(none)")
parts.append("")
parts.append(user_prompt)
full_user = "\n".join(parts).strip()

# call HF router (OpenAI-compatible)
api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
payload = {
    "model": "Qwen/Qwen2.5-7B-Instruct:featherless-ai",
    "temperature": 0.0,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_user}
    ]
}
resp = requests.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=payload, timeout=60)
resp.raise_for_status()
content = resp.json()["choices"][0]["message"]["content"]

# try parse JSON output
try:
    model_json = json.loads(content)
except Exception:
    model_json = None

out = {
    "id": rec.get("id"),
    "question": question,
    "description": description,
    "resolution_criteria": resolution,
    "evidence_summaries": summary_items,
    "model": "Qwen/Qwen2.5-7B-Instruct:featherless-ai",
    "temperature": 0.0,
    "response_text": content,
    "response_json": model_json,
    "generated_at": datetime.utcnow().isoformat() + "Z"
}

out_path = output_dir / "record_0001.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(str(out_path))
