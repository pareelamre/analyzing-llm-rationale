# Paired Forecasting Examples: When Structure Helps and When It Hurts

These paired examples are intended to show prompt effects rather than showcase only successes. The first case shows a structured rationale helping one model identify the decisive evidence. The second shows a structured rationale pushing a different model toward overconfident, irrelevant reasoning. Together, they illustrate that prompt effects are model-dependent and not uniformly beneficial.

## Example A: Structure Helps Identify the Decisive Evidence

**Model:** `GPT-OSS-120B`, `temperature_025`

**Metaculus ID:** `39523`

**Question:** Will China be able to mass produce humanoid robots by the end of 2025?

**Resolution criteria, shortened:** Resolves **Yes** if credible reports indicate that a China-headquartered robotics company manufactured more than 1,000 humanoid robots of an identical model before the end of 2025.

**Resolved label:** `yes`

### Short Evidence Digest

- **CNBC, 2025-11-05; credibility 0.665.** Reports Xpeng robotaxi and humanoid-robot plans, but the timing is mostly forward-looking.
- **The Robot Report, 2025-09-10; credibility 0.735.** Reports that several Chinese humanoid-robot manufacturers had begun mass-production programs, with targets exceeding 1,000 units each by the end of 2025.
- **Business Insider, 2025-11-17; credibility 0.630.** Reports Xpeng’s humanoid robot plans, but with mass production framed mostly around later targets.

The decisive evidence is the Robot Report item: it connects a China-headquartered robotics sector to the 1,000-unit threshold and the 2025 deadline. The baseline output is cautious because it treats the evidence as plans rather than achieved-or-credible mass-production reports. The structured outputs identify the threshold-relevant evidence more directly.

### JSON Output Excerpts

```json
{
  "V0_neutral_baseline": {
    "predicted_answer": "No",
    "confidence": 0.35,
    "rationale_excerpt": "The available articles only describe plans or projected production targets for Chinese humanoid robots, with no confirmed reports of any company having actually manufactured over 1,000 identical units before the end of 2025."
  },
  "V4_credibility": {
    "predicted_answer": "Yes",
    "confidence": 0.68,
    "rationale_excerpt": "The Robot Report (Sept 2025) reports that six Chinese humanoid robot makers launched mass-production programs in 2024 with targets of over 1,000 units each by the end of 2025, projecting more than 10,000 units total. This directly satisfies the resolution criteria."
  },
  "V8_temporal_anchors": {
    "predicted_answer": "Yes",
    "confidence": 0.66,
    "rationale_excerpt": "Recent credible reports state that several Chinese humanoid-robot manufacturers began mass-production in 2024 with explicit goals of producing over 1,000 identical units each by the end of 2025..."
  }
}
```

**Interpretation:** Here, the structured prompts help by making the model attend to the evidence item that matches both the numeric threshold and the deadline. The probability changes meaningfully, from `No` at 0.35 under V0 to `Yes` at 0.68 under V4.

## Example B: Structure Induces Irrelevant or Overconfident Reasoning

**Model:** `Qwen2.5-7b-instruct`, `temperature_000`

**Metaculus ID:** `31316`

**Question:** Will SpaceX have a Falcon 9 launch failure in 2025?

**Resolution criteria, shortened:** Resolves **Yes** if, during calendar year 2025, any Falcon 9-family vehicle had a failed or partially failed launch according to SpaceX reports to the FAA or other credible sources.

**Resolved label:** `no`

### Short Evidence Digest

- **Space, 2026-01-11; credibility 0.700.** Reports an Indian PSLV launch failure in January 2026, not a Falcon 9 failure in 2025.
- **Ars Technica, 2026-01-09; credibility 0.613.** 2025 year-in-review context; notes SpaceX launch activity and does not report a Falcon 9 failure.
- **Spaceflight Now, 2026-01-12; credibility 0.700.** Reports another Indian PSLV failure, again outside the target vehicle and mostly outside the 2025 target window.

This evidence contains credible spaceflight reporting, but much of it is about the wrong rocket or the wrong year. The failure mode is that credibility framing can overweight reliable-but-misaligned evidence.

### JSON Output Excerpts

```json
{
  "V0_neutral_baseline": {
    "predicted_answer": "No",
    "confidence": 0.95,
    "rationale_excerpt": "While the evidence summaries indicate a significant launch failure for India's PSLV in 2026, no information suggests a Falcon 9 failure for SpaceX in 2025."
  },
  "V4_credibility": {
    "predicted_answer": "Yes",
    "confidence": 0.85,
    "rationale_excerpt": "The evidence strongly supports a 'Yes' answer based on the second consecutive launch failure of India's PSLV rocket in 2026..."
  },
  "V8_temporal_anchors": {
    "predicted_answer": "No",
    "confidence": 0.85,
    "rationale_excerpt": "The evidence summaries indicate that the Falcon 9 family did not experience any launch failures in 2025, and the relevant time range for the question is the entire year of 2025."
  }
}
```

**Interpretation:** In this case, the credibility-framed prompt harms the answer by treating credible PSLV failure reports as if they supported the Falcon 9 criterion. The temporal-anchor prompt recovers the relevant frame: Falcon 9, calendar year 2025. This is the opposite pattern from Example A, which is why the pair is more informative than a single success case.

## Takeaway

Structured rationales can sharpen evidence use, but they can also sharpen the wrong aspect of the evidence. Credibility framing helps when the decisive source is both credible and criterion-aligned; it can hurt when credible evidence is about the wrong event. Temporal anchors can mitigate this by forcing the model to check event identity and deadline alignment, but the effect varies by model and question.
