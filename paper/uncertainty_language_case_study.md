# V0/V7 Example: Linguistic Uncertainty Is Not Calibration

This matched example illustrates why cautious wording should not be treated as calibrated uncertainty. V7 adds lexical uncertainty markers such as "likely" and "could", but the numeric forecast becomes much less aligned with the resolved outcome.

## Example

**Model:** `Qwen2.5-7b-instruct`, `temperature_000`  
**Metaculus ID:** `34676`  
**Question:** Will François Bayrou step down or be removed from his position as Prime Minister of France before March 31, 2025?  
**Resolved label:** `no`

**Resolution criterion, shortened:** resolves **Yes** only if Bayrou resigned, ceased to be prime minister, or was removed by a no-confidence motion before March 31, 2025.

**Short evidence snippet:** the retrieved news centered on Marine Le Pen's March 31, 2025 conviction and political fallout. It did not report Bayrou's resignation or removal.

## Matched Outputs

```json
{
  "V0_neutral_baseline": {
    "predicted_answer": "No",
    "confidence": 0.85,
    "key_rationale_sentence": "There is no evidence suggesting that François Bayrou will be removed from his position as Prime Minister of France before March 31, 2025."
  },
  "V7_uncertainty_language": {
    "predicted_answer": "Yes",
    "confidence": 0.95,
    "key_rationale_sentence": "Given the conviction of Marine Le Pen and the ban on her running for public office, it is highly likely that she will appeal the ruling, which could lead to her stepping down or being removed from her position as Prime Minister of France before the specified date."
  }
}
```

## Why This Demonstrates the Disconnect

V7 contains more cautious language: "highly likely" and "could". But the uncertainty language is attached to an irrelevant chain of reasoning about Marine Le Pen rather than the target event involving François Bayrou. The output assigns `0.95` confidence to `Yes`, while the resolved label is `no`. In probability terms, the forecast is badly calibrated despite sounding more uncertainty-aware.

The contrast with V0 is the key point: V0 uses less explicit uncertainty language, but its probability is better aligned with the final outcome because it keeps the actor and event criterion fixed.
