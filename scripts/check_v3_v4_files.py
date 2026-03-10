import os, time, json

base = r"C:\Users\paree\Documents\Analyzing Rationale of LLMs\results\Qwen2.5-7b-instruct\temperature_00"
for fn in ["results_variant3_reasoning_type.json", "results_variant4_credibility.json"]:
    p = os.path.join(base, fn)
    st = os.stat(p)
    print(fn, "bytes", st.st_size, "mtime", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)))
    try:
        d = json.load(open(p, encoding="utf-8"))
        nulls = sum(1 for r in d if isinstance(r, dict) and r.get("predicted_answer") is None)
        print("  records", len(d), "null_predicted_answer", nulls)
    except Exception as e:
        print("  json_error", repr(e))
