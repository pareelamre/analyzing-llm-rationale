import os
import json
import pathlib
import requests
import time

base = pathlib.Path(r"C:\Users\paree\Documents\Analyzing Rationale of LLMs")
input_path = base / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
output_dir = base / "results" / "Qwen2.5-7b-instruct" / "temperature_00"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "results_variant5_key_conditions.json"

system_prompt = (base / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
neutral_prompt = (base / "prompts" / "variant5_key_conditions.txt").read_text(encoding="utf-8").strip()

records = json.load(input_path.open("r", encoding="utf-8"))

api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

session = requests.Session()

# resume support
existing_ids = set()
results = []
if output_path.exists():
    try:
        existing = json.load(output_path.open("r", encoding="utf-8"))
        if isinstance(existing, list):
            results = existing
            for obj in existing:
                if isinstance(obj, dict) and "id" in obj:
                    existing_ids.add(obj["id"])
    except Exception:
        pass

max_records = int(os.environ.get("MAX_RECORDS", "0"))
processed = 0

for idx, rec in enumerate(records, 1):
    rec_id = rec.get("id")
    if rec_id in existing_ids:
        continue

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
            "text": art.get("text"),
        })

    def build_user_prompt(items):
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
        if items:
            for i, s in enumerate(items, 1):
                parts.append("Article {}: {}".format(i, json.dumps(s, ensure_ascii=False)))
        else:
            parts.append("(none)")

        parts.append("")
        parts.append(user_prompt)
        return "\n".join(parts).strip()

    user_prompt = neutral_prompt.replace("[question]", "").strip()
    full_user = build_user_prompt(summary_items)

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct:featherless-ai",
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user}
        ]
    }

    # basic retry with context-limit fallback (drop full text)
    content = None
    for attempt in range(3):
        try:
            resp = session.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=payload, timeout=120)
            if resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 400 and "maximum context length" in resp.text:
                # drop full extracted text field and retry once
                trimmed_items = []
                for s in summary_items:
                    if isinstance(s, dict):
                        s2 = dict(s)
                        s2["text"] = None
                        trimmed_items.append(s2)
                full_user = build_user_prompt(trimmed_items)
                payload["messages"][1]["content"] = full_user
                resp = session.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=payload, timeout=120)
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
        out_obj = {"id": rec_id, "predicted_answer": None, "confidence": None, "rationale": None, "key_conditions": None}
    else:
        try:
            model_json = json.loads(content)
        except Exception:
            model_json = None
        if isinstance(model_json, dict):
            out_obj = {
                "id": rec_id,
                "predicted_answer": model_json.get("predicted_answer"),
                "confidence": model_json.get("confidence"),
                "rationale": model_json.get("rationale"),
                "key_conditions": model_json.get("key_conditions"),
            }
        else:
            out_obj = {"id": rec_id, "predicted_answer": None, "confidence": None, "rationale": content, "key_conditions": None}

    results.append(out_obj)
    processed += 1
    # incremental write after each record
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if max_records and processed >= max_records:
        break

print(str(output_path))
