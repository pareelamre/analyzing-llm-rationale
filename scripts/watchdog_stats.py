import json, sys, pathlib

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def stats(path):
    data = load_json(path)
    total = len(data)
    nulls = 0
    for row in data:
        if row.get('predicted_answer', None) is None:
            nulls += 1
    return total, nulls

if __name__ == '__main__':
    goal_path = pathlib.Path('forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json')
    v1_path = pathlib.Path(r'results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant1_predicted_event.json')
    v2_path = pathlib.Path(r'results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant2_key_attribute.json')

    goal = len(load_json(goal_path))
    v1 = stats(v1_path)
    v2 = stats(v2_path)

    print(goal)
    print(f"{v1[0]},{v1[1]}")
    print(f"{v2[0]},{v2[1]}")
