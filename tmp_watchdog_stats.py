import json, os

main_path = r"forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
with open(main_path, "r", encoding="utf-8") as f:
    main = json.load(f)

if isinstance(main, dict):
    for k in ["data", "items", "questions", "records"]:
        if k in main and isinstance(main[k], list):
            main = main[k]
            break

if not isinstance(main, list):
    raise SystemExit(f"Unexpected main dataset type: {type(main)}")

ids = []
for row in main:
    if isinstance(row, dict):
        _id = row.get("id", row.get("question_id", row.get("uuid")))
        if _id is not None:
            ids.append(_id)

goal = len(set(ids)) if ids else len(main)


def stats(path: str):
    if not os.path.exists(path):
        return {"total": 0, "nulls": 0, "status": "MISSING"}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ["data", "items", "results", "records"]:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    if not isinstance(data, list):
        return {"total": 0, "nulls": 0, "status": f"UNEXPECTED:{type(data)}"}

    nulls = 0
    for r in data:
        if isinstance(r, dict):
            if r.get("predicted_answer", "__MISSING__") is None:
                nulls += 1
        else:
            nulls += 1

    return {"total": len(data), "nulls": nulls, "status": "OK"}


v1_path = r"results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant1_predicted_event.json"
v2_path = r"results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant2_key_attribute.json"

out = {
    "goal": goal,
    "v1": stats(v1_path),
    "v2": stats(v2_path),
}
print(json.dumps(out, ensure_ascii=False))
