import json, os

def goal_count(path):
    with open(path,'r',encoding='utf-8') as f:
        data=json.load(f)
    if isinstance(data,list):
        return len(data)
    if isinstance(data,dict):
        for k in ['data','items','examples','records']:
            if k in data and isinstance(data[k],list):
                return len(data[k])
        return len(data)
    return 0

def file_stats(path):
    if not os.path.exists(path):
        return 0, 0
    with open(path,'r',encoding='utf-8') as f:
        data=json.load(f)
    items=list(data.values()) if isinstance(data,dict) else data
    total=len(items)
    nulls=0
    for it in items:
        if (not isinstance(it,dict)) or it.get('predicted_answer',None) is None:
            nulls += 1
    return total, nulls

if __name__=='__main__':
    base='forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json'
    var1='results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant1_predicted_event.json'
    var2='results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant2_key_attribute.json'
    goal=goal_count(base)
    v1t,v1n=file_stats(var1)
    v2t,v2n=file_stats(var2)
    print(goal)
    print(v1t, v1n)
    print(v2t, v2n)
