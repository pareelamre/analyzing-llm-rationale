import json
from pathlib import Path


def goal_total(path: str) -> int:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for k in ('data', 'items', 'examples', 'records'):
            if k in data and isinstance(data[k], list):
                return len(data[k])
        return len(data)
    return 0


def file_stats(path: str):
    p = Path(path)
    if not p.exists():
        return 0, 0

    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            items = data['data']
        elif all(isinstance(v, dict) for v in data.values()):
            items = list(data.values())
        else:
            items = []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    total = len(items)
    nulls = 0
    for r in items:
        if not isinstance(r, dict):
            continue
        v = r.get('predicted_answer', '__MISSING__')
        if v is None or v == '' or v == [] or v == {}:
            nulls += 1

    return total, nulls


if __name__ == '__main__':
    goal_path = 'forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json'
    v1_path = r'results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant1_predicted_event.json'
    v2_path = r'results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant2_key_attribute.json'

    goal = goal_total(goal_path)
    v1_total, v1_null = file_stats(v1_path)
    v2_total, v2_null = file_stats(v2_path)

    print(f'GOAL {goal}')
    print(f'V1 {v1_total} null {v1_null}')
    print(f'V2 {v2_total} null {v2_null}')
