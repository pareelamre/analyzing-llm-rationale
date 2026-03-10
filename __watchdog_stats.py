import json, os

def count_goal(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return len(data)

def count_results(path):
    if not os.path.exists(path):
        return 0, 0
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total = len(data)
    nulls = 0
    for row in data:
        # row may be dict; predicted_answer key expected
        if isinstance(row, dict) and row.get('predicted_answer') is None:
            nulls += 1
    return total, nulls

def main():
    goal_path = 'forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json'
    v1_path = os.path.join('results','Qwen2.5-7b-instruct','temperature_00','results_variant1_predicted_event.json')
    v2_path = os.path.join('results','Qwen2.5-7b-instruct','temperature_00','results_variant2_key_attribute.json')

    goal = count_goal(goal_path)
    v1_total, v1_null = count_results(v1_path)
    v2_total, v2_null = count_results(v2_path)

    print(f'goal={goal}')
    print(f'variant1 total_written={v1_total} null_predicted_answer={v1_null}')
    print(f'variant2 total_written={v2_total} null_predicted_answer={v2_null}')

if __name__ == '__main__':
    main()
