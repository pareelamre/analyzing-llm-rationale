# Forecasting SQL Analytics

DuckDB analytics over the Metaculus-style forecasting questions, model predictions, and news logs.

## 1. Accuracy per model (all variants combined)

CSV: `1_accuracy_per_model_all_variants_combined.csv`

```text
              model     n  accuracy
       GPT-OSS-120B 96368    0.8103
Qwen2.5-7b-instruct 96361    0.7293
          Qwen3-32B 96333    0.7083
```

## 2. Best-performing variant per model

CSV: `2_best_performing_variant_per_model.csv`

```text
              model                            variant  accuracy    n
       GPT-OSS-120B variant12_random_structure_control    0.8288 1577
Qwen2.5-7b-instruct     variant15_neutral_no_rationale    0.7873 1580
          Qwen3-32B          variant13_rationale_first    0.7840 1579
```

## 3. Confidence calibration (10 bins): stated confidence vs actual accuracy

CSV: `3_confidence_calibration_10_bins_stated_confidence_vs_actual_accuracy.csv`

```text
 conf_bin     n  avg_confidence  accuracy  calibration_gap
      0.0   179           0.026    0.8492          -0.8228
      0.1  1585           0.147    0.8164          -0.6698
      0.2  7023           0.239    0.8246          -0.5854
      0.3  7233           0.339    0.7769          -0.4381
      0.4  4452           0.435    0.6644          -0.2290
      0.5  6969           0.548    0.5661          -0.0184
      0.6 43958           0.647    0.6632          -0.0160
      0.7 76015           0.747    0.6892           0.0576
      0.8 90127           0.845    0.7693           0.0754
      0.9 50741           0.937    0.8916           0.0451
      1.0   780           1.000    0.9308           0.0692
```

## 4. Brier score per model (lower is better)

CSV: `4_brier_score_per_model_lower_is_better.csv`

```text
              model     n  brier_score
       GPT-OSS-120B 96368      0.16983
Qwen2.5-7b-instruct 96361      0.20335
          Qwen3-32B 96333      0.22800
```

## 5. Consensus questions: all models predict the same answer

CSV: `5_consensus_questions_all_models_predict_the_same_answer.csv`

```text
 question_id                                                                                                           question ground_truth  model_count consensus_answer  avg_confidence
       38057                                                            Will Merab Dvalishvili defeat Sean O'Malley at UFC 316?          yes            3              Yes           0.964
       40927                                                      Will André Ventura win Portugal's 2026 presidential election?           no            3               No           0.959
       38055                                                                          Will Joe Pyfer defeat someone at UFC 316?          yes            3              Yes           0.957
       38557                                         Will Andrew Cuomo win the 2025 Democratic primary for New York City mayor?           no            3               No           0.954
       38529                                         Will Andrew Cuomo win the 2025 Democratic primary for New York City mayor?           no            3               No           0.953
       35476                                              Will Nasa's SphereX space telescope be launched before April 1, 2025?          yes            3              Yes           0.947
       34507                                       Will Kash Patel be confirmed by the Senate as FBI Director by June 30, 2025?          yes            3              Yes           0.946
       39955 Is it the case that Sunderland AFC and Aston Villa will finish their September 21 EPL match with identical scores?          yes            3              Yes           0.946
       37238                                                                                  Will Donald Trump attend UFC 316?          yes            3              Yes           0.946
       37087                                     Will Chris Stapleton win an award at the 60th Academy of Country Music Awards?          yes            3              Yes           0.944
```

## 6. Disagreement questions: highest confidence variance across models

CSV: `6_disagreement_questions_highest_confidence_variance_across_models.csv`

```text
 question_id                                                                                                                                                                                                                                   question ground_truth  model_count  confidence_stddev  answer_variety
       40595                                                                                                                          Will OpenAI file an initial registration statement (S-1) with the SEC to launch an IPO, before December 15, 2025?           no            3             0.3255               1
       37249                                                                                                                    Before July 1, 2025, will Reform UK be the highest polling party in the UK by at least 4 points, according to Politico?          yes            3             0.3173               2
       39970                                                                                                                                                                Who will be the next non-caretaker prime minister of Nepal? (Balendra Shah)           no            3             0.3106               1
       10176                                                                                                                                      Will more than 34 countries have committed to a stringent anti-solar-geoengineering pact before 2026?           no            3             0.3063               1
       40667 Will an official international supervisory board explicitly tasked with overseeing transitional governance of the Gaza Strip be constituted, with at least two publicly named individual members, between 2025-10-15 and 2025-12-31 (UTC)?          yes            3             0.2943               1
       37008                                                                                                                                                                         Will a state of emergency be in effect in Samoa on April 30, 2025?           no            3             0.2919               2
       34955                                                                                                                  Before March 15, 2025, will Reform UK be the highest polling party in the UK by at least 2 points, according to Politico?          yes            3             0.2885               1
       35726                                                                                                                                                                   Will a pilot report of UFOs come from a country in Africa in March 2025?           no            3             0.2884               2
       37296                                                                                                                    Before July 1, 2025, will Reform UK be the highest polling party in the UK by at least 4 points, according to Politico?          yes            3             0.2881               1
       40440                                                                                                                                                                                  Will Beyond Meat hit $12 a share before November 1, 2025?           no            3             0.2870               1
```

## 7. Variant lift over baseline (variant0): accuracy delta per model

CSV: `7_variant_lift_over_baseline_variant0_accuracy_delta_per_model.csv`

```text
              model                                 variant  base_acc  var_acc  delta
          Qwen3-32B               variant13_rationale_first    0.7146   0.7840 0.0694
          Qwen3-32B          variant15_neutral_no_rationale    0.7146   0.7829 0.0683
          Qwen3-32B        variant10_length_matched_neutral    0.7146   0.7608 0.0462
          Qwen3-32B variant14_temporal_credibility_combined    0.7146   0.7595 0.0449
Qwen2.5-7b-instruct          variant15_neutral_no_rationale    0.7434   0.7873 0.0439
Qwen2.5-7b-instruct      variant12_random_structure_control    0.7434   0.7816 0.0382
          Qwen3-32B    variant11_generic_detailed_rationale    0.7146   0.7519 0.0373
Qwen2.5-7b-instruct variant14_temporal_credibility_combined    0.7434   0.7797 0.0363
Qwen2.5-7b-instruct        variant10_length_matched_neutral    0.7434   0.7797 0.0363
          Qwen3-32B      variant12_random_structure_control    0.7146   0.7481 0.0335
Qwen2.5-7b-instruct    variant11_generic_detailed_rationale    0.7434   0.7722 0.0288
          Qwen3-32B                    variant4_credibility    0.7146   0.7301 0.0155
       GPT-OSS-120B      variant12_random_structure_control    0.8148   0.8288 0.0140
       GPT-OSS-120B          variant15_neutral_no_rationale    0.8148   0.8241 0.0093
       GPT-OSS-120B        variant10_length_matched_neutral    0.8148   0.8205 0.0057
```

## 8. Temperature sensitivity: accuracy by temperature per model

CSV: `8_temperature_sensitivity_accuracy_by_temperature_per_model.csv`

```text
              model  temperature    n  accuracy
       GPT-OSS-120B         0.00 1580    0.8247
       GPT-OSS-120B         0.25 1580    0.8215
       GPT-OSS-120B         0.75 1580    0.8259
       GPT-OSS-120B         1.25 1580    0.8095
       GPT-OSS-120B         1.75 1580    0.8184
       GPT-OSS-120B         2.00 1580    0.7886
Qwen2.5-7b-instruct         0.00 1580    0.7544
Qwen2.5-7b-instruct         0.25 1580    0.7449
Qwen2.5-7b-instruct         0.75 1580    0.7411
Qwen2.5-7b-instruct         1.25 1580    0.7418
Qwen2.5-7b-instruct         1.75 1580    0.7399
Qwen2.5-7b-instruct         2.00 1580    0.7380
          Qwen3-32B         0.00 1580    0.7228
          Qwen3-32B         0.25 1580    0.7133
          Qwen3-32B         0.75 1580    0.7222
          Qwen3-32B         1.25 1580    0.7038
          Qwen3-32B         1.75 1580    0.7101
          Qwen3-32B         2.00 1577    0.7153
```

## 9. Overconfident errors: wrong predictions with confidence > 0.8

CSV: `9_overconfident_errors_wrong_predictions_with_confidence_0_8.csv`

```text
              model                        variant  question_id predicted_answer ground_truth  confidence
Qwen2.5-7b-instruct      variant13_rationale_first        34976              Yes           no         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        39773              Yes           no         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        35286              Yes           no         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        40185              Yes           no         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        40128               No          yes         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        35472              Yes           no         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        39191              Yes           no         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        40306               No          yes         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        40994              Yes           no         1.0
       GPT-OSS-120B      variant13_rationale_first        39028               No          yes         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        41634              Yes           no         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        35480              Yes           no         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        39028               No          yes         1.0
Qwen2.5-7b-instruct      variant13_rationale_first        42008              Yes           no         1.0
Qwen2.5-7b-instruct variant15_neutral_no_rationale        40128               No          yes         1.0
```

## 10. Category difficulty: accuracy per question category (hardest first)

CSV: `10_category_difficulty_accuracy_per_question_category_hardest_first.csv`

```text
               category    n  accuracy
          entertainment   36    0.5000
artificial-intelligence 2610    0.6613
     computing-and-math  576    0.6632
    environment-climate  791    0.6662
       health-pandemics 2016    0.6880
                  space 1116    0.6980
       economy-business 5651    0.7206
       cryptocurrencies  648    0.7361
              elections 3150    0.7416
            geopolitics 4806    0.7511
                    law 1979    0.7575
               politics 3960    0.7816
             technology 1368    0.7917
   sports-entertainment 1296    0.8002
                nuclear  612    0.8072
```
