# Temporal placement and criterion-alignment diagnostics

The automated lexical census is complete. It retains all 1,580 questions, 4,327 article-question records and 5,651 saved statements. Human review is **skipped by the user**, not a pending gate. These results establish reproducible string-placement counts, not a semantic-validity rate or forecasting gain.

## Exact census counts

| Fixed lexical indicator | Statements / 5,651 | Article records / 4,327 | Questions / 1,580 |
|---|---:|---:|---:|
| Unspecified horizon with a fixed timing cue elsewhere in claim fields | 121 | 119 | 107 |
| Bare numeric duration horizon with a fixed timing cue elsewhere | 4 | 4 | 4 |
| Fixed support-quote timing cue absent literally from the claim fields and horizon | 1,241 | 1,188 | 611 |
| At least one of the three indicators, deduplicated | 1,339 | 1,276 | 626 |

Indicators overlap, so their marginal counts must not be added. The denominator includes 572 questions with no retained articles and 76 article records with no saved statements. Neither an empty FRS nor an unflagged record is a semantic pass. Records can repeat a news story across questions; these are article-question counts, not globally unique articles. All six saved window/status cells, zero-hit rows, empty rows and per-field evidence are retained in report.json.

The vocabulary covers full English weekdays and eight fixed relative phrases. It omits dates, month names, abbreviations, numerical years and synonyms. Matching literal cues does not determine whether they refer to the extracted event, a past report, irrelevant background or an equivalent time expressed differently. The rules were fixed after the earlier eight-case screen and before this census: this is exploratory, hypothesis-driven work, not a new test on untouched data.

## Three fixed examples read in full

Root read the first selected record per indicator, deduplicated: D001, D003 and D005. This cap was fixed before the counts were inspected. D005 was already known; D001 and D003 are newly read. The three exported second examples were not reviewed. No prevalence or false-positive-rate estimate follows from this diagnostic set.

**D001 — a different meaning of “appeal.”** The saved question title asks whether Elon Musk will lose his appeal, but the criteria define appeal through Google Trends interest in 2025 relative to 2020. The article and FRS instead discuss his Delaware legal appeal over a pay package. This is a concrete criterion mismatch in the supplied materials; it does not identify the retrieval/ranking mechanism responsible or prove that a later forecasting model was misled. The flagged “this week” occurs in “Earlier this week,” dating an already-filed appeal. Its absence is not proof of an omitted future-decision deadline.

**D003 — timing survives outside its dedicated field.** The Digg FRS has an unspecified horizon, while its rationale explicitly preserves Wednesday. The complete representation therefore retains the cue. Separately, the question's criteria enumerate eight Reddit alternatives and Digg is not among them. A report about launching a Reddit rival does not directly address the specified candidates' popularity comparison. This supports a target-contract concern beyond timing-field placement, without establishing exact historical availability or a target outcome.

**D005 — a reporting day is not an event deadline.** In the previously inspected TikTok case, Monday survives in the rationale alongside “most likely”; 90 days describes the proposed reprieve's duration. The quote-only cue Saturday dates Trump's statement. Both lexical flags are reproducible, but neither demonstrates that the future Monday was lost. This is a direct reason not to describe the 1,188 support-only cue hits as a timing-loss count.

Exact source/criterion quotations and 11 checked character-offset spans are in assistant-interpretation.json. The labels and reasoning are AI-assisted diagnostics, not independent human validation. No gold outcomes or stored forecast scores were used to trigger or select cases.

## Implications for the research code and later paper

1. **Use the full claim representation when measuring retained information.** A missing dedicated horizon can coexist with a precise cue in rationale. Evaluating only the dedicated field can overstate information loss.
2. **Separate event timing, duration and reporting time in a future schema.** The present untyped horizon field can hold a duration while the rationale supplies an event anchor. A reporting weekday in a support quote need not be an event deadline. This is a design recommendation, not a tested improvement.
3. **Make the question's target contract explicit in a future retrieval/extraction comparison.** The popularity/legal-appeal ambiguity and the closed list of Reddit alternatives show why title/topic similarity and source-faithful FRS alone cannot establish criterion relevance. A later change should test criteria-aware retrieval/extraction against the current behavior on a frozen set, retaining adverse and unknown cases. That comparison has not been run here.

No source, extraction policy or frozen output was repaired, reclassified or retried. The submitted paper and shared code remain unchanged. Historical body/question versions and genuinely unknown outcomes at a defensible forecast origin remain unverified where documentary evidence is absent. This slice supports a more precise failure taxonomy and concrete code hypotheses; it does not validate a forecasting cohort or improve any measured forecast score.
