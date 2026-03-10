import json
from pathlib import Path

def load_list(path: str):
    p = Path(path)
    if not p.exists():
        return []
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)

def status(path: str):
    data = load_list(path)
    written = len(data)
    nulls = sum(1 for r in data if (r.get('predicted_answer', None) is None))
    return written, nulls

goal = len(load_list('forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json'))

v5w, v5n = status('results/Qwen2.5-7b-instruct/temperature_00/results_variant5_key_conditions.json')
v6w, v6n = status('results/Qwen2.5-7b-instruct/temperature_00/results_variant6_chain_of_thought.json')

print(goal)
print(v5w, v5n)
print(v6w, v6n)
