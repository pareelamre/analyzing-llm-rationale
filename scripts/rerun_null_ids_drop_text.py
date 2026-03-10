import os
import json
import pathlib
import requests
import time

base = pathlib.Path(r"C:\Users\paree\Documents\Analyzing Rationale of LLMs")
input_path = base / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
output_path = base / "results" / "Qwen2.5-7b-instruct" / "temperature_00" / "results_variant1_predicted_event.json"

system_prompt = (base / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
neutral_prompt = (base / "prompts" / "variant1_predicted_event.txt").read_text(encoding="utf-8").strip()

records = json.load(input_path.open("r", encoding="utf-8"))
rec_by_id = {r.get("id"): r for r in records}

results = json.load(output_path.open("r", encoding="utf-8"))
null_ids = [r.get("id") for r in results if r.get("predicted_answer") is None]

api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
session = requests.Session()


def build_user_prompt(rec, include_text: bool):
    question = (rec.get("question") or "").strip()
    description = (rec.get("description") or "").strip()
    resolution = (rec.get("resolution_criteria") or "").strip()
    categories = rec.get("categories") or []
    created_time = rec.get("created_time")
    publish_time = rec.get("publish_time")
    resolve_time = rec.get("resolve_time")
    days_open = rec.get("days_open")

    articles = rec.get("news_articles") or []
    summary_items = []
    for art in articles:
        if not isinstance(art, dict):
            continue
        summary_items.append({
            "url": art.get("url"),
            "title": art.get("title"),
            "authors": art.get("authors"),
            "publish_date": art.get("publish_date"),
            "summary": art.get("summary"),
            "summary_llm": art.get("summary_llm"),
            "keywords": art.get("keywords"),
            "frs": art.get("frs"),
            "credibility": art.get("credibility"),
            "text": art.get("text") if include_text else None,
        })

    user_prompt = neutral_prompt.replace("[question]", "").strip()
    parts = []
    parts.append(f"Question: {question}")
    if description:
        parts.append(f"Description: {description}")
    if resolution:
        parts.append(f"Resolution Criteria: {resolution}")
    if categories:
        parts.append(f"Categories: {categories}")
    if created_time:
        parts.append(f"Created Time: {created_time}")
    if publish_time:
        parts.append(f"Publish Time: {publish_time}")
    if resolve_time:
        parts.append(f"Resolve Time: {resolve_time}")
    if days_open is not None:
        parts.append(f"Days Open: {days_open}")

    parts.append("Evidence Summaries (full article fields):")
    if summary_items:
        for i, s in enumerate(summary_items, 1):
            parts.append("Article {}: {}".format(i, json.dumps(s, ensure_ascii=False)))
    else:
        parts.append("(none)")

    parts.append("")
    parts.append(user_prompt)
    return "\n".join(parts).strip()


for rec_id in null_ids:
    rec = rec_by_id.get(rec_id)
    if not rec:
        continue

    # Drop full text for null ids
    full_user = build_user_prompt(rec, include_text=False)

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct:featherless-ai",
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user},
        ],
    }

    content = None
    for attempt in range(3):
        try:
            resp = session.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=payload, timeout=120)
            if resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            break
        except Exception:
            if attempt == 2:
                content = None
            else:
                time.sleep(2 ** attempt)
                continue

    if content is None:
        continue

    try:
        model_json = json.loads(content)
    except Exception:
        model_json = None

    if isinstance(model_json, dict):
        for obj in results:
            if obj.get("id") == rec_id:
                obj["predicted_answer"] = model_json.get("predicted_answer")
                obj["confidence"] = model_json.get("confidence")
                obj["rationale"] = model_json.get("rationale")
                obj["predicted_event"] = model_json.get("predicted_event")
                break

    # incremental write
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"Updated {len(null_ids)} null ids")
