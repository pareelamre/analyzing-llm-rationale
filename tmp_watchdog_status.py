import json
from pathlib import Path

def load(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def count_records(obj):
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        for k in ("data", "examples", "items"):
            v = obj.get(k)
            if isinstance(v, list):
                return len(v)
        return len(obj)
    return 0

def iter_records(obj):
    if isinstance(obj, list):
        yield from obj
        return
    if isinstance(obj, dict):
        for k in ("data", "examples", "items"):
            v = obj.get(k)
            if isinstance(v, list):
                yield from v
                return
        for v in obj.values():
            if isinstance(v, dict):
                yield v

root = Path(__file__).resolve().parent

goal_path = root / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
goal_obj = load(goal_path)
goal = count_records(goal_obj)

outs = [
    ("variant1", root / "results" / "Qwen2.5-7b-instruct" / "temperature_00" / "results_variant1_predicted_event.json"),
    ("variant2", root / "results" / "Qwen2.5-7b-instruct" / "temperature_00" / "results_variant2_key_attribute.json"),
]

print(f"GOAL {goal}")
for name, path in outs:
    if not path.exists():
        print(f"{name} MISSING")
        continue
    obj = load(path)
    total = count_records(obj)
    nulls = sum(1 for r in iter_records(obj) if isinstance(r, dict) and r.get("predicted_answer", None) is None)
    print(f"{name} TOTAL {total} NULL {nulls}")
