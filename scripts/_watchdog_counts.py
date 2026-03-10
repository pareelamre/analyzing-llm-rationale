import json, os

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def goal_total(p):
    d = load_json(p)
    if isinstance(d, dict):
        for k in ('data', 'items', 'examples', 'rows'):
            if k in d and isinstance(d[k], list):
                return len(d[k])
        return len(d)
    return len(d)

def count_out(p):
    if not os.path.exists(p):
        return 0, 0
    d = load_json(p)
    if isinstance(d, dict):
        items = list(d.values())
    elif isinstance(d, list):
        items = d
    else:
        items = []
    total = len(items)
    nulls = sum(1 for r in items if isinstance(r, dict) and r.get('predicted_answer') is None)
    return total, nulls

main = 'forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json'
v1 = r'results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant1_predicted_event.json'
v2 = r'results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant2_key_attribute.json'

goal = goal_total(main)
v1t, v1n = count_out(v1)
v2t, v2n = count_out(v2)

print(goal)
print(v1t, v1n)
print(v2t, v2n)
