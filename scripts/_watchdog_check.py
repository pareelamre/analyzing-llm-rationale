import json
from pathlib import Path


def goal_total(path: str) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for k in ("data", "items", "records", "examples"):
            if k in data and isinstance(data[k], list):
                return len(data[k])
        return len(data)
    return 0


def status(path: str):
    p = Path(path)
    if not p.exists():
        return 0, 0, "missing"
    data = json.loads(p.read_text(encoding="utf-8"))
    items = list(data.values()) if isinstance(data, dict) else data
    total = len(items)
    nulls = 0
    for it in items:
        if isinstance(it, dict) and it.get("predicted_answer", None) is None:
            nulls += 1
    return total, nulls, "ok"


def main():
    goal = goal_total(
        "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
    )

    v1_path = r"results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant1_predicted_event.json"
    v2_path = r"results\\Qwen2.5-7b-instruct\\temperature_00\\results_variant2_key_attribute.json"

    v1_total, v1_nulls, v1_state = status(v1_path)
    v2_total, v2_nulls, v2_state = status(v2_path)

    print(f"GOAL {goal}")
    print(f"V1 {v1_total}/{goal} nulls={v1_nulls} ({v1_state})")
    print(f"V2 {v2_total}/{goal} nulls={v2_nulls} ({v2_state})")

    need_run = (v1_total < goal) or (v1_nulls > 0) or (v2_total < goal) or (v2_nulls > 0)
    Path(".watchdog_status.json").write_text(
        json.dumps(
            {
                "goal": goal,
                "v1_total": v1_total,
                "v1_nulls": v1_nulls,
                "v2_total": v2_total,
                "v2_nulls": v2_nulls,
                "need_run": need_run,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
