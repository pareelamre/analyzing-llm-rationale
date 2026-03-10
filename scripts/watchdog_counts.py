import json
from pathlib import Path


def count_file(p: str):
    p = Path(p)
    if not p.exists():
        return 0, 0
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    items = list(data.values()) if isinstance(data, dict) else data
    total = len(items)
    nulls = 0
    for r in items:
        if isinstance(r, dict) and r.get('predicted_answer', None) is None:
            nulls += 1
    return total, nulls


def goal_total(p: str):
    p = Path(p)
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            return len(data['data'])
        return len(data)
    return len(data)


if __name__ == '__main__':
    goal = goal_total('forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json')
    v1_total, v1_null = count_file('results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant1_predicted_event.json')
    v2_total, v2_null = count_file('results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant2_key_attribute.json')

    print(goal)
    print(v1_total, v1_null)
    print(v2_total, v2_null)
