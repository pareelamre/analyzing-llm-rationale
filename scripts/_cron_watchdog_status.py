import json
from pathlib import Path

def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def goal_total(main_path: Path) -> int:
    data = load_json(main_path)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for k in ("data", "items", "questions", "records"):
            if k in data and isinstance(data[k], list):
                return len(data[k])
        # fallback: treat dict keys as records (unlikely)
        return len(data)
    return 0

def results_status(path: Path):
    if not path.exists():
        return 0, 0
    data = load_json(path)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        # sometimes stored by id
        if all(isinstance(v, dict) for v in data.values()):
            rows = list(data.values())
        else:
            # unknown dict structure
            rows = []
    else:
        rows = []

    total_written = len(rows)
    null_pred = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get('predicted_answer', None) is None:
            null_pred += 1
    return total_written, null_pred

if __name__ == '__main__':
    main = Path('forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json')
    v1 = Path('results/Qwen2.5-7b-instruct/temperature_00/results_variant1_predicted_event.json')
    v2 = Path('results/Qwen2.5-7b-instruct/temperature_00/results_variant2_key_attribute.json')

    goal = goal_total(main)
    v1_total, v1_null = results_status(v1)
    v2_total, v2_null = results_status(v2)

    print(json.dumps({
        'goal': goal,
        'variant1': {'total_written': v1_total, 'null_predicted_answer': v1_null},
        'variant2': {'total_written': v2_total, 'null_predicted_answer': v2_null},
    }))
