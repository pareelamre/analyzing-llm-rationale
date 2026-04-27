# Rationale Evaluation Summary

- Results: `/data/horse/ws/paam844f-codabench/analyzing-llm-rationale/results/Qwen2.5-7b-instruct/temperature_000/results_variant0_neutral_baseline.json`
- Dataset: `/data/horse/ws/paam844f-codabench/analyzing-llm-rationale/forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json`
- Scored examples: `1580`
- Correctness uses the dataset ground-truth answer.

## Overall

| Metric | Mean | Median | P10 | P90 |
| --- | ---: | ---: | ---: | ---: |
| forecast_correct | 0.754 | 1.000 | 0.000 | 1.000 |
| confidence | 0.852 | 0.850 | 0.750 | 0.950 |
| context_grounding | 0.618 | 0.625 | 0.450 | 0.783 |
| question_focus | 0.703 | 0.750 | 0.364 | 1.000 |
| article_support | 0.308 | 0.313 | 0.101 | 0.511 |
| temporal_specificity | 0.899 | 1.000 | 0.000 | 1.000 |
| conciseness | 0.958 | 1.000 | 0.750 | 1.000 |
| hedging_alignment | 0.679 | 0.800 | 0.300 | 1.000 |
| quality_proxy | 0.694 | 0.708 | 0.544 | 0.819 |

## Correct vs Incorrect Forecasts

| Metric | Correct | Incorrect | Delta |
| --- | ---: | ---: | ---: |
| confidence | 0.859 | 0.830 | +0.029 |
| context_grounding | 0.624 | 0.600 | +0.024 |
| question_focus | 0.715 | 0.665 | +0.051 |
| article_support | 0.319 | 0.274 | +0.045 |
| temporal_specificity | 0.908 | 0.874 | +0.034 |
| conciseness | 0.962 | 0.947 | +0.014 |
| hedging_alignment | 0.685 | 0.659 | +0.026 |
| quality_proxy | 0.702 | 0.670 | +0.032 |

## Strongest Correct Examples

- ID `41271` | conf `0.85` | quality `0.938` | grounding `0.952` | article `0.674`
  Question: Will ABBV's market close price on 2025-12-31 be higher than its market close price on 2025-12-24?
  Rationale: The latest available trading day for the target date (2025-12-31) is 2025-12-30, with a closing price of 229.74. This is lower than the baseline closing price of 229.89 on 2025-12-24, indicating that ABBV’s market close price on 2025-12-31 will not be higher than its market close price on 2025-12-24.
- ID `31124` | conf `0.95` | quality `0.930` | grounding `0.967` | article `0.779`
  Question: Will car crash deaths in the US be below 40,000 for 2024?
  Rationale: Preliminary data from the National Highway Traffic Safety Administration (NHTSA) shows that 39,345 people died in traffic crashes in 2024, which is a decrease of about 3.8% compared to 2023. This marks the first time since 2020 that the number of fatalities fell below 40,000.
- ID `41434` | conf `0.85` | quality `0.925` | grounding `0.933` | article `0.617`
  Question: Will WY's market close price on 2026-01-16 be higher than its market close price on 2026-01-09?
  Rationale: The Stooq-derived daily close prices show that the market close price of WY on 2026-01-16 ($27.06) is higher than its market close price on 2026-01-09 ($25.73).
- ID `41287` | conf `0.85` | quality `0.924` | grounding `0.905` | article `0.639`
  Question: Will MTB's market close price on 2026-01-01 be higher than its market close price on 2025-12-27?
  Rationale: The latest available trading day for the baseline date (2025-12-27) had a closing price of 207.28, while the latest available trading day for the target date (2026-01-01) had a closing price of 201.48. Since the target closing price is lower than the baseline closing price, the market close price on 2026-01-01 will not be higher than its market close price on 2025-12-27.
- ID `31151` | conf `0.95` | quality `0.923` | grounding `0.964` | article `0.573`
  Question: Will the International Longshoremen's Association (ILA) go on strike before February 1, 2025?
  Rationale: Credible news sources, including PBS and The Guardian, report that the International Longshoremen's Association and the U.S. Maritime Alliance reached a tentative six-year contract agreement a week before the January 15, 2025 deadline, effectively averting any potential strike before February 1, 2025.

## High-Confidence Failures

- ID `36877` | conf `0.95` | quality `0.556` | grounding `0.037` | article `0.000`
  Question: Чи зустрінеться Зеленський з Трампом до липня 2025?
  Rationale: Since the last meeting in February 2025, there has been no indication of any upcoming personal meetings between President Volodymyr Zelensky of Ukraine and President Donald Trump of the United States. Given the lack of concrete plans or announcements, it is unlikely they will meet before July 1, 2025.
- ID `36173` | conf `0.95` | quality `0.565` | grounding `0.370` | article `0.027`
  Question: Will the Guatemala City, Guatemala fail to reach reach 80 degrees F on any day after the launch of this question and before April 1, 2025?
  Rationale: The evidence provided does not contain any relevant weather data for Guatemala City, Guatemala, and there is no indication of an unusual weather pattern that would prevent temperatures from reaching 80 degrees F. Given the typical climate in Guatemala City, it is highly unlikely that the temperature will not reach 80 degrees F before April 1, 2025.
- ID `37436` | conf `0.95` | quality `0.601` | grounding `0.548` | article `0.188`
  Question: Will 300 or more UN staff be reported to be killed in Gaza according to UN's OCHA in its final impact snapshot before July 1, 2025?
  Rationale: The latest evidence provided is a security verification notice from the UNRWA website, which does not contain any information about UN staff fatalities in Gaza. Given the lack of relevant data and the time proximity to the resolution date, it is unlikely that the number of reported UN staff killed will reach 300 by July 1, 2025.
- ID `31217` | conf `0.95` | quality `0.596` | grounding `0.567` | article `0.085`
  Question: Will Brazil officially launch its digital currency (Drex) by March 2025?
  Rationale: The evidence indicates that by March 2025, the Central Bank of Brazil had already made significant progress in the Drex project, with an 18-month pilot completed and the currency ready for public use. Given the timeline and the lack of any official postponements mentioned, it is highly likely that Drex was launched as planned.
- ID `35474` | conf `0.95` | quality `0.703` | grounding `0.577` | article `0.342`
  Question: Will India's Gaganyaan-1 program launch a flight test before April 1, 2025?
  Rationale: The article mentions that India's Gaganyaan mission is planned for 2025 and explicitly states that the first space hotel is set to begin construction in 2025. Given the timeline and the explicit mention, it is highly likely that the Gaganyaan-1 program will launch a flight test before April 1, 2025.

## Grounded But Wrong

- ID `39489` | conf `0.85` | quality `0.910` | grounding `0.926` | article `0.636`
  Question: In 2025, will the domestic content requirements become mandatory for obtaining the IRA's Production Tax Credits (45Y)?
  Rationale: The One Big Beautiful Bill Act (OBBBA), signed into law on July 4, 2025, harmonizes the domestic content requirements under Section 48E with those under Section 45Y for projects beginning construction on or after June 16, 2025. This indicates that the domestic content requirements become mandatory for the Production Tax Credits (PTC) described in Section 45Y.
- ID `35472` | conf `0.95` | quality `0.903` | grounding `0.895` | article `0.632`
  Question: Will SpaceX's Fram2 mission launch before April 1, 2025?
  Rationale: The SpaceX Fram2 mission launched on March 31, 2025, and the crew began uploading views of Earth's poles from their polar orbit. This confirms the mission launched before April 1, 2025.
- ID `39357` | conf `0.85` | quality `0.756` | grounding `0.870` | article `0.355`
  Question: Which of these changes to the 45X clean energy manufacturing tax credit of the 2022 Inflation Reduction Act will occur before January 1, 2026? (Capped credits)
  Rationale: The One Big Beautiful Bill Act, signed into law on July 4, 2025, by President Trump, includes provisions to cap 45X tax credits, which directly addresses the question of whether capped credits will occur before January 1, 2026.
- ID `35480` | conf `0.95` | quality `0.875` | grounding `0.864` | article `0.588`
  Question: Will 900,000 Russian personnel losses be reported before March 21, 2025?
  Rationale: On March 21, 2025, the Commander-in-Chief of the Armed Forces of Ukraine, Oleksandr Syrskyi, announced that Russian personnel losses in Ukraine had exceeded 900,000. This announcement meets the criteria for resolving the question as 'Yes'.
- ID `39475` | conf `0.85` | quality `0.513` | grounding `0.862` | article `0.459`
  Question: Will a dense machine learning model with at least 100 trillion parameters be trained before 2026?
  Rationale: The NVIDIA Rubin platform, as described in the article, is designed to support large-scale AI inference and training, potentially accommodating models with over 100 trillion parameters. The platform's architecture, including the Vera CPU and Rubin GPU, is optimized for sustained intelligence production and high throughput, suggesting it could support such a dense model.
