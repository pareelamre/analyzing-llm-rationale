import json, os

goal_file=r"forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
v5_file=r"results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant5_key_conditions.json"
v6_file=r"results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant6_chain_of_thought.json"

def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)

def stat(path):
    if not os.path.exists(path):
        return 0, 0
    data=load_json(path)
    if isinstance(data, dict):
        it=data.values(); n=len(data)
    else:
        it=data; n=len(data)
    nulls=0
    for r in it:
        if not isinstance(r, dict) or r.get('predicted_answer', None) is None:
            nulls += 1
    return n, nulls

goal=load_json(goal_file)
goal_total=len(goal) if not isinstance(goal, dict) else len(goal)

v5_n,v5_null=stat(v5_file)
v6_n,v6_null=stat(v6_file)

print(goal_total)
print(v5_n, v5_null)
print(v6_n, v6_null)
