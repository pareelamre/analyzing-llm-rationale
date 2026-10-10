# Structured-prompt comparison under a 30-day pre-event cutoff

All variants (V0-V8) evaluated with evidence restricted to articles published
at least 30 days before each question's knowable time (`--cutoff-reference event_end
--forecast-lead-days 30`), at temperature 0.0. Compared against the retrospective
oracle condition (full evidence up to resolution).

Dataset base rate (P(yes)): 0.3519

## GPT-OSS-120B  (n=1521)

| Variant | Acc (30d) | Brier (30d) | ECE (30d) | Brier (oracle) | Brier delta |
|---|---|---|---|---|---|
| V0 | 0.7048 | 0.2029 | 0.0451 | 0.1583 | +0.0446 |
| V1 | 0.6930 | 0.2311 | 0.0917 | 0.1639 | +0.0672 |
| V2 | 0.6923 | 0.2303 | 0.0839 | 0.1729 | +0.0574 |
| V3 | 0.6936 | 0.2351 | 0.0904 | 0.1827 | +0.0524 |
| V4 | 0.7035 | 0.2140 | 0.0619 | 0.1703 | +0.0437 |
| V5 | 0.6857 | 0.2487 | 0.1095 | 0.1505 | +0.0982 |
| V6 | 0.6936 | 0.2439 | 0.1097 | 0.1631 | +0.0808 |
| V7 | 0.6923 | 0.2114 | 0.0515 | 0.1759 | +0.0355 |
| V8 | 0.6897 | 0.2273 | 0.0928 | 0.1535 | +0.0738 |

Evidence filtering: 0.54 of 2.14 articles kept on average (23.3%); 72.8% of questions had **zero** evidence kept.

vs V0 under the cutoff: 7 variant(s) significantly worse, 0 significantly better (p<0.05, paired bootstrap).
  - worse: V1, V2, V3, V4, V5, V6, V8

## Qwen2.5-7b-instruct  (n=1580)

| Variant | Acc (30d) | Brier (30d) | ECE (30d) | Brier (oracle) | Brier delta |
|---|---|---|---|---|---|
| V0 | 0.6652 | 0.2750 | 0.2386 | 0.1874 | +0.0876 |
| V1 | 0.6544 | 0.2697 | 0.2215 | 0.1976 | +0.0721 |
| V2 | 0.6627 | 0.2761 | 0.2385 | 0.1981 | +0.0780 |
| V3 | 0.6608 | 0.2666 | 0.2178 | 0.1978 | +0.0688 |
| V4 | 0.6658 | 0.2735 | 0.2335 | 0.1994 | +0.0741 |
| V5 | 0.6538 | 0.2802 | 0.2409 | 0.2193 | +0.0609 |
| V6 | 0.6582 | 0.2801 | 0.2445 | 0.2005 | +0.0796 |
| V7 | 0.6589 | 0.2779 | 0.2394 | 0.1987 | +0.0792 |
| V8 | 0.6620 | 0.2742 | 0.2353 | 0.1974 | +0.0768 |

Evidence filtering: 0.54 of 2.14 articles kept on average (23.3%); 72.8% of questions had **zero** evidence kept.

vs V0 under the cutoff: 0 variant(s) significantly worse, 0 significantly better (p<0.05, paired bootstrap).

## Qwen3-32B  (n=1580)

| Variant | Acc (30d) | Brier (30d) | ECE (30d) | Brier (oracle) | Brier delta |
|---|---|---|---|---|---|
| V0 | 0.6335 | 0.3262 | 0.2566 | 0.1997 | +0.1265 |
| V1 | 0.6089 | 0.3159 | 0.2375 | 0.2149 | +0.1010 |
| V2 | 0.6247 | 0.3333 | 0.2722 | 0.2143 | +0.1190 |
| V3 | 0.6127 | 0.3306 | 0.2652 | 0.2146 | +0.1160 |
| V4 | 0.6380 | 0.3330 | 0.2677 | 0.2031 | +0.1299 |
| V5 | 0.6120 | 0.3287 | 0.2614 | 0.2173 | +0.1114 |
| V6 | 0.6228 | 0.3553 | 0.3049 | 0.2239 | +0.1314 |
| V7 | 0.5987 | 0.3252 | 0.2550 | 0.2233 | +0.1019 |
| V8 | 0.6222 | 0.3454 | 0.2912 | 0.2108 | +0.1346 |

Evidence filtering: 0.54 of 2.14 articles kept on average (23.3%); 72.8% of questions had **zero** evidence kept.

vs V0 under the cutoff: 2 variant(s) significantly worse, 0 significantly better (p<0.05, paired bootstrap).
  - worse: V6, V8
