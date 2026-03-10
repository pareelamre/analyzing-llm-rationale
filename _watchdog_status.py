import json
from pathlib import Path

goal_path = Path('forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json')
data = json.loads(goal_path.read_text(encoding='utf-8'))
data_list = data['data'] if isinstance(data, dict) and 'data' in data else data
goal_total = len(data_list)

def status(path: str):
    p = Path(path)
    if not p.exists():
        return 0, None
    try:
        obj = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None, None
    lst = obj['data'] if isinstance(obj, dict) and 'data' in obj else obj
    if not isinstance(lst, list):
        return None, None
    written = len(lst)
    nulls = 0
    for rec in lst:
        if (not isinstance(rec, dict)) or rec.get('predicted_answer', None) is None:
            nulls += 1
    return written, nulls

v5_written, v5_nulls = status(r'results\Qwen2.5-7b-instruct\temperature_00\results_variant5_key_conditions.json')
v6_written, v6_nulls = status(r'results\Qwen2.5-7b-instruct\temperature_00\results_variant6_chain_of_thought.json')

print(json.dumps({
    'goal_total': goal_total,
    'v5_written': v5_written,
    'v5_nulls': v5_nulls,
    'v6_written': v6_written,
    'v6_nulls': v6_nulls,
}))
