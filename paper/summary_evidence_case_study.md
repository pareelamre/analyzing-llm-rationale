# Case Study: Forecasting-Summary Evidence Can Help or Hurt

This pair compares standard runs that include forecasting-summary fields with matched `without_frs` runs. The examples use different models to avoid implying a universal prompt effect.

## Case 1: Summary Evidence Helps Qwen3 Avoid an Incorrect Forecast

**Model:** `Qwen3-32B`, `temperature_0`  
**Metaculus ID:** `7434`  
**Question:** Will PsiQuantum have a commercial quantum computer by 2025?  
**Resolved label:** `no`

**Short evidence snippets**

- `2025-11-03`, Aviation Week: PsiQuantum signed a partnership with Lockheed Martin to develop quantum algorithms.
- `2026-01-22`, MarketWatch: PsiQuantum received new investment for utility-scale, fault-tolerant quantum systems.
- Forecasting-summary frame: the target event is not general progress; it is a company report that a commercial quantum computer was for sale by 2025.

**Without forecasting-summary fields**

```json
{
  "predicted_answer": "Yes",
  "confidence": 0.85,
  "key_rationale_sentence": "Multiple recent articles indicate that PsiQuantum is making significant progress toward its goal of building a commercial quantum computer by 2025."
}
```

**With forecasting-summary fields**

```json
{
  "predicted_answer": "No",
  "confidence": 0.35,
  "key_rationale_sentence": "There is no direct confirmation that they have a commercial quantum computer for sale by 2025."
}
```

**Why this is useful:** the structured summary helps separate adjacent evidence about funding and partnerships from the actual resolution criterion: commercial availability by the deadline.

## Case 2: Removing Summary Fields Helps GPT-OSS Avoid Redundant Framing

**Model:** `GPT-OSS-120B`, `temperature_0`  
**Metaculus ID:** `37036`  
**Question:** Will Dan Osborn announce his candidacy for US Senator from Nebraska before June 1, 2025?  
**Resolved label:** `no`

**Short evidence snippets**

- `2025-04-03`, KLIN: Osborn was exploring a 2026 run against Senator Pete Ricketts.
- `2024-06-01`, Nebraska Examiner: Osborn was already running in the 2024 Senate race against Deb Fischer.
- `2024-08-20`, Nebraska Examiner: Osborn submitted signatures for the 2024 Senate ballot.

**With forecasting-summary fields**

```json
{
  "predicted_answer": "Yes",
  "confidence": 0.92,
  "key_rationale_sentence": "Multiple news articles from June 2024 and August 2024 report that Dan Osborn publicly announced a nonpartisan bid for the U.S. Senate and submitted signatures to appear on the ballot."
}
```

**Without forecasting-summary fields**

```json
{
  "predicted_answer": "No",
  "confidence": 0.73,
  "key_rationale_sentence": "The latest available report (April 3, 2025) indicates Osborn is only exploring a run against Senator Pete Ricketts and has not made a formal candidacy announcement."
}
```

**Why this is useful:** removing the summary fields improves calibration and avoids redundant event framing from older Senate-race evidence. The model focuses on the relevant 2025 question: whether Osborn formally announced the later Nebraska Senate candidacy before June 1, 2025.

## Takeaway

Forecasting-summary evidence is not simply beneficial or harmful. For Qwen3, it corrected an overgeneralized inference from progress signals. For GPT-OSS, removing it avoided a misleading reuse of superficially similar older campaign evidence.
