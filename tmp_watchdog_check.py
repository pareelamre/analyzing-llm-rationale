import json, os

def count_goal(path: str) -> int:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for k in ('data', 'items', 'examples', 'records'):
            v = data.get(k)
            if isinstance(v, list):
                return len(v)
        return len(data)
    return 0

def count_out(path: str):
    if not os.path.exists(path):
        return 0, 0
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vals = list(data.values()) if isinstance(data, dict) else data
    total = len(vals)
    nulls = 0
    for r in vals:
        if isinstance(r, dict):
            if r.get('predicted_answer', None) is None:
                nulls += 1
        else:
            nulls += 1
    return total, nulls

def main():
    base = 'forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json'
    var1 = r'results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant1_predicted_event.json'
    var2 = r'results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant2_key_attribute.json'

    goal = count_goal(base)
    t1, n1 = count_out(var1)
    t2, n2 = count_out(var2)
    print(goal)
    print(t1, n1)
    print(t2, n2)

if __name__ == '__main__':
    main()
