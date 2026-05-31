# Matched Probability, Contrasting Rationale Quality

These examples have similar forecast probabilities: both outputs use `0.80` confidence. Their judge scores diverge sharply, showing that forecast probability and rationale quality measure different things. Scores below are the mean of the Gemma-4-31B-it and Kimi-K2.5 judge ratings.

## High-Quality Rationale

**Model:** `GPT-OSS-120B`, `temperature_025`, `V0_neutral_baseline`  
**Metaculus ID:** `35724`  
**Question:** Will the Artemis Accords have 54 signatories on April 1, 2025?  
**Forecast:** `No`, confidence `0.80`  
**Resolved outcome:** `no`

**Short evidence snippet:** Space.com reported that Finland became the 53rd Artemis Accords signatory on January 21, 2025. The evidence did not report a 54th signatory before April 1, 2025.

**Rationale**

> The latest evidence shows Finland became the 53rd signatory on Jan. 21, 2025, and no additional countries are reported to have joined before April 1, 2025. Without any indication of a 54th nation signing in the interim, the likelihood of reaching 54 signatories by the target date is low.

**Judge attributes**

| Attribute | Score |
| --- | ---: |
| Plausibility | 0.98 |
| Completeness | 0.95 |
| Source consistency | 1.00 |
| Non-hallucination | 1.00 |
| Informativeness | 0.95 |
| Conciseness | 0.95 |
| Mean | 0.97 |

**Why it scores well:** the rationale identifies the decisive count, anchors it to the target date, and avoids inventing an additional signatory. It is complete enough for the resolution criterion while remaining concise.

## Low-Quality Rationale

**Model:** `Qwen2.5-7b-instruct`, `temperature_000`, `V7_uncertainty_language`  
**Metaculus ID:** `39392`  
**Question:** On December 31, 2025, will Google, Meta, Amazon, Tesla, or X accept crypto as a payment?  
**Forecast:** `Yes`, confidence `0.80`  
**Resolved outcome:** `no`

**Short evidence snippet:** the retrieved article was about the concentration of the "Magnificent Seven" in the S&P 500. It did not report that Google, Meta, Amazon, Tesla, X, or their subsidiaries accepted cryptocurrency for a core product or service on December 31, 2025.

**Rationale**

> Given the increasing mainstream acceptance of cryptocurrencies and the financial interests of large tech companies, it is likely that one of Google, Meta, Amazon, Tesla, or X will accept crypto as a payment by December 31, 2025.

**Judge attributes**

| Attribute | Score |
| --- | ---: |
| Plausibility | 0.25 |
| Completeness | 0.10 |
| Source consistency | 0.05 |
| Non-hallucination | 0.45 |
| Informativeness | 0.10 |
| Conciseness | 0.90 |
| Mean | 0.31 |

**Why it scores poorly:** the rationale is concise, but it omits the decisive evidence requirement: actual payment acceptance by one of the named companies on the target date. It instead introduces a broad, unsupported claim about mainstream crypto acceptance and infers the outcome from general market framing rather than source-consistent evidence.

## Takeaway

Both forecasts have confidence `0.80`, but their rationale quality is very different. The high-quality example links the probability to the resolution criterion and evidence. The low-quality example sounds plausible at a high level but is not grounded in the supplied source material, so its probability is not backed by a reliable explanation.
