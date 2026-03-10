import json
from pathlib import Path

base = Path(r"C:\Users\paree\Documents\Analyzing Rationale of LLMs")
inp = base / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
goal = len(json.load(inp.open("r", encoding="utf-8")))

out_dir = base / "results" / "Qwen2.5-7b-instruct" / "temperature_00"
for fn in ["results_variant1_predicted_event.json", "results_variant2_key_attribute.json"]:
    p = out_dir / fn
    d = json.load(p.open("r", encoding="utf-8"))
    nulls = sum(1 for r in d if isinstance(r, dict) and r.get("predicted_answer") is None)
    print(f"{fn}: written={len(d)} goal={goal} null_predicted_answer={nulls}")
