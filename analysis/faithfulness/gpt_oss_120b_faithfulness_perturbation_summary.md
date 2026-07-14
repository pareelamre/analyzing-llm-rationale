# Rationale Faithfulness Perturbation Summary

Paired perturbation forecasts analyzed: 397

| Perturbation | n | Mean dP(Yes) | Mean abs dP(Yes) | Median abs dP(Yes) | >=5pp | >=10pp | Answer flips |
|---|---:|---:|---:|---:|---:|---:|---:|
| actor_date_swap | 100 | -0.046 | 0.222 | 0.160 | 0.830 | 0.650 | 0.210 |
| contradiction | 100 | -0.111 | 0.153 | 0.120 | 0.790 | 0.590 | 0.020 |
| criterion_swap | 100 | -0.189 | 0.256 | 0.165 | 0.880 | 0.710 | 0.190 |
| evidence_masking | 97 | -0.080 | 0.189 | 0.120 | 0.763 | 0.577 | 0.052 |

Interpretation: low movement under order perturbation alone is not sufficient evidence of direct rationale faithfulness. Evidence masking, contradiction injection, actor/date swaps, and criterion swaps test whether forecasts move when the support, stated rationale, target entity/time, or resolution rule is perturbed.
