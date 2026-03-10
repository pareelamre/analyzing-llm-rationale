import json, os, sys

def count_goal(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return len(data)

def count_results(path):
    if not os.path.exists(path):
        return 0, 0
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # If file is mid-write/corrupt, treat as incomplete.
            return 0, 0
    total = len(data) if isinstance(data, list) else len(data.keys())
    nulls = 0
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and row.get('predicted_answer') is None:
                nulls += 1
    else:
        for _, row in data.items():
            if isinstance(row, dict) and row.get('predicted_answer') is None:
                nulls += 1
    return total, nulls

if __name__ == '__main__':
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    goal_path = os.path.join(base, 'forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json')
    v1_path = os.path.join(base, 'results', 'Qwen2.5-7b-instruct', 'temperature_00', 'results_variant1_predicted_event.json')
    v2_path = os.path.join(base, 'results', 'Qwen2.5-7b-instruct', 'temperature_00', 'results_variant2_key_attribute.json')

    goal = count_goal(goal_path)
    v1_total, v1_null = count_results(v1_path)
    v2_total, v2_null = count_results(v2_path)

    print(goal)
    print(v1_total, v1_null)
    print(v2_total, v2_null)
