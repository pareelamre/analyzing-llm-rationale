# Summary Correctness Audit

Dataset: `forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json`.

This audit is deterministic and uses the stored artifact only. It does not verify the current live web pages, so it cannot determine whether pages were edited after publication or after collection.

## Evidence Representation Provenance

- Article retrieval: the repository retrieval script uses the question text to construct a compact semantic/lexical query, then tries DuckDuckGo News, Google News/GNews, and GDELT, with full-text enrichment through `trafilatura` when article text is short. The final released artifact does not preserve `source_channel`, `search_query`, fetch timestamp, HTTP headers, content hash, or page revision metadata, so exact per-article retrieval provenance is only partially reconstructable.
- Pre/post resolution: all parsed article publish dates in the artifact are before formal `resolve_time`; this audit found zero articles on or after formal resolution. However, a formal-resolution filter is weaker than a true ex-ante forecast-time filter: 177 articles are after the approximate event-window end inferred from the question text and resolution time.
- Page updates: the artifact stores extracted text, not a fetch timestamp, content hash, Last-Modified header, or archived URL. Therefore this audit cannot determine whether a page was updated after its displayed publication date or after it was collected.
- Forecasting-summary generation: `scripts/summarize_articles.py` first asks an LLM to rate article relevance on a 1-6 scale and summarizes articles rated at least 4. The summary prompt contains only the forecasting question, article title, and the first 1,500 characters of article text; it does not include the resolved answer. The default summarizer is `gpt-oss-120b` through the SCADS OpenAI-compatible endpoint, with temperature 0.2 for summary generation.
- Outcome access: the summary-generation prompt does not pass `answer` or `resolve_time`. The generator can still indirectly see post-outcome information when the retrieved article itself is post-event but pre-formal-resolution, which is why the `post_event_window` flag matters.
- Reuse across models and variants: batch prompting reads `summary_llm` and `frs` from the shared dataset. Standard model/variant runs therefore use the same stored summaries unless the run explicitly uses a no-evidence, without-FRS, full-text, or forecast-cutoff configuration.
- Correctness checking: before this audit, the repository had downstream rationale-quality and human-annotation checks, but no dedicated stored summary-correctness audit. This report provides a first deterministic screen and identifies cases requiring manual review.

## Aggregate Findings

- Records: 1580
- Records with at least one article: 1484
- Articles: 3387
- Articles per record: {0: 96, 1: 428, 2: 209, 3: 847}
- Articles with `summary_llm`: 3382 (5 missing)
- Articles with `frs`: 3387
- Unparsed article publish dates: 8
- Articles published on/after formal `resolve_time`: 0
- Articles published after approximate event-window end: 177
- Articles published before the question publish/create time: 659

## Flag Counts

- `before_question_publish`: 659
- `frs_no_conditions_meta`: 594
- `frs_rationale_na`: 1915
- `many_summary_terms_absent_from_source`: 2522
- `missing_summary_llm`: 5
- `post_event_window`: 177
- `summary_numbers_not_verbatim_in_source`: 533
- `unparsed_publish_date`: 8

## Manual Follow-Up Examples

### Metaculus 124 article 1

- Flags: `many_summary_terms_absent_from_source;frs_rationale_na;frs_no_conditions_meta`
- Question: Will the world population grow every year from 2016 to 2025 (inclusive)?
- Article: What Will the World Look Like at the End of Trump’s Second Term? - inkstickmedia.com (2025-12-31T08:00:00)
- Event-window end: 2025-12-31T23:59:59+00:00
- Summary start: In his second term, President Trump has aggressively dismantled the geopolitical framework that has supported U.S. global hegemony for eight decades. Beginning in January 2025, he pursued a strategy to overturn the established world order, employing a varied and seemingly random set of policies. The article notes that his approach toward Latin America is clearly hostile, while his policy toward the Asia‑Pacific region appears confused and ambiguo

### Metaculus 124 article 2

- Flags: `post_event_window;many_summary_terms_absent_from_source`
- Question: Will the world population grow every year from 2016 to 2025 (inclusive)?
- Article: Japan's Debt Eased by Dankai Generation Inheritance Taxes - 조선일보 (2026-01-01T08:00:00)
- Event-window end: 2025-12-31T23:59:59+00:00
- Summary start: Japan's national debt-to-GDP ratio, which had been rising, has begun to stabilize after 2020, partly due to higher inheritance tax revenues as the 8‑million‑person "Dankai" baby‑boomer cohort (born 1947‑1949) passes away. Inheritance tax receipts grew from about 1.5 trillion yen in 2005 to 3.4 trillion yen in 2024, coinciding with an increase in annual deaths from 1.33 million to 1.54 million. Researchers note that the elderly hold 68% of househo

### Metaculus 229 article 0

- Flags: `many_summary_terms_absent_from_source;frs_no_conditions_meta`
- Question: Contact lenses for augmented reality in use by innovators before 2026?
- Article: poLight ASA Confirms Design Win with Vuzix Shield Industrial AR Smart Glasses - Intellectia AI (2026-02-16T01:12:57)
- Event-window end: 2026-02-16T23:44:00+00:00
- Summary start: poLight ASA announced that it has secured a design win to supply components for Vuzix's Shield industrial augmented reality smart glasses. The Shield glasses are part of Vuzix's broader portfolio of AI‑powered wearable devices aimed at enterprise, medical, defense, and consumer markets, which includes the M series, Blade, and LX1 models. Vuziz has recently launched integrated enterprise solutions such as Remote Assist, featuring native Microsoft 

### Metaculus 229 article 1

- Flags: `missing_summary_llm;frs_no_conditions_meta`
- Question: Contact lenses for augmented reality in use by innovators before 2026?
- Article: Augmented Reality Contact Lenses: Will They Ever Be a Reality? - UC Today (2025-11-10T00:00:00Z)
- Event-window end: 2026-02-16T23:44:00+00:00
- Summary start: 

### Metaculus 229 article 2

- Flags: `missing_summary_llm;frs_no_conditions_meta`
- Question: Contact lenses for augmented reality in use by innovators before 2026?
- Article: These Contact Lenses Zoom and Project Augmented Reality &#8211; Nerdy Digest (2025-08-16T00:00:00Z)
- Event-window end: 2026-02-16T23:44:00+00:00
- Summary start: 

### Metaculus 273 article 0

- Flags: `many_summary_terms_absent_from_source;frs_rationale_na;frs_no_conditions_meta`
- Question: 50% Carbon-neutral electricity by 2025?
- Article: Why Germany has Banned Apple's Carbon Neutrality Claim - Sustainability Magazine (2025-09-01T07:00:00)
- Event-window end: 2025-09-01T16:22:00+00:00
- Summary start: A Frankfurt court ruled that Apple cannot market the Apple Watch Series 9 as carbon neutral in Germany because the company's carbon offset plan relies on eucalyptus plantations in Paraguay that lack guaranteed long‑term contracts, making the claim unsubstantiated. Apple had advertised the watch as carbon neutral based on recycled materials, clean‑energy manufacturing, and offsets through its Apple Restore Fund. The court found the lack of future 

### Metaculus 273 article 1

- Flags: `many_summary_terms_absent_from_source;frs_rationale_na;frs_no_conditions_meta`
- Question: 50% Carbon-neutral electricity by 2025?
- Article: BMW iX3 – Groundbreaking holistic product sustainability. - BMW Group (2025-08-28T07:00:00)
- Event-window end: 2025-09-01T16:22:00+00:00
- Summary start: The new BMW iX3 illustrates the BMW Group’s holistic strategy for product sustainability throughout its entire life cycle. Comprehensive measures were applied in the supply chain, production, and vehicle‑use phases to conserve resources and lower environmental impact. Thanks to extensive decarbonisation, the iX3 50 xDrive’s CO2e emissions fall below those of a comparable internal‑combustion model after roughly 21,500 kilometres when charged with 

### Metaculus 477 article 0

- Flags: `many_summary_terms_absent_from_source;summary_numbers_not_verbatim_in_source`
- Question: Efficacy confirmation of a new Alzheimer's treatment protocol?
- Article: Lecanemab in Early Alzheimer’s Disease | New England Journal of Medicine - New England Journal of Medicine (2022-11-29T08:00:00)
- Event-window end: 2025-02-18T07:59:00+00:00
- Summary start: The phase 3 Clarity AD trial evaluated lecanemab, a monoclonal antibody targeting soluble amyloid-beta protofibrils, in 1,795 participants with early Alzheimer's disease over 18 months. Compared with placebo, lecanemab produced a statistically significant reduction in clinical decline, with a 0.45‑point smaller increase in the Clinical Dementia Rating–Sum of Boxes score and modest improvements on secondary cognitive and functional measures. Amylo

### Metaculus 512 article 0

- Flags: `many_summary_terms_absent_from_source;frs_rationale_na;frs_no_conditions_meta`
- Question: Will an AI system do credibly well on a full math SAT exam by 2025?
- Article: Opinion | An Interview With the Herald of the Apocalypse - The New York Times (2025-05-15T07:00:00)
- Event-window end: 2025-12-31T23:59:59+00:00
- Summary start: The New York Times opinion piece presents an edited transcript of a podcast episode titled “Interesting Times,” featuring an interview with AI researcher Daniel Kokotajlo. Kokotajlo predicts that by 2027 a form of machine intelligence, described as a “machine god,” could emerge, potentially creating a post‑scarcity utopia or posing an existential threat. Host Ross Douthat frames the discussion around the speed of the AI revolution, the implicatio

### Metaculus 1079 article 0

- Flags: `many_summary_terms_absent_from_source;frs_rationale_na;frs_no_conditions_meta`
- Question: Will Elon Musk (eventually) lose his appeal?
- Article: Senior Trump Aide Finally Deems Elon Musk Too Annoying - The New Republic (2025-01-27T08:00:00)
- Event-window end: 2025-02-01T01:00:00+00:00
- Summary start: Chief of staff Susie Wiles, who also co-manages Donald Trump’s 2024 campaign, blocked Elon Musk from obtaining a permanent West Wing office, assigning his team instead to the Eisenhower Executive Office Building. Wiles said she will not tolerate solo‑star behavior, backbiting, or drama, aiming to streamline the administration and limit access to the president. Trump confirmed Musk will not lead a new "Department of Government Efficiency" (DOGE) i

### Metaculus 1079 article 1

- Flags: `many_summary_terms_absent_from_source;frs_rationale_na`
- Question: Will Elon Musk (eventually) lose his appeal?
- Article: In Elon Musk, Libertarianism and Authoritarianism Combine - Jacobin (2025-01-29T08:00:00)
- Event-window end: 2025-02-01T01:00:00+00:00
- Summary start: The Jacobin article argues that Elon Musk has merged libertarian economic ideas with authoritarian political tactics, using his ownership of X (formerly Twitter) to amplify right‑wing and anti‑democratic messages worldwide. It details his public support for Donald Trump, appearances at Germany’s AfD campaign launch, and praise for leaders such as Italy’s Giorgia Meloni, describing these actions as part of a broader strategy to undermine democrati

### Metaculus 1079 article 2

- Flags: `many_summary_terms_absent_from_source;frs_rationale_na;frs_no_conditions_meta`
- Question: Will Elon Musk (eventually) lose his appeal?
- Article: Bogus Ads Use Elon Musk's Image for Scams - AARP (2025-01-31T08:00:00)
- Event-window end: 2025-02-01T01:00:00+00:00
- Summary start: Criminals are exploiting Elon Musk's likeness in a range of scams, including a fraudulent "energy‑saving" handheld device, a deepfake‑driven investment scheme called Quantum AI, and fake giveaways that demand fees or personal data. The scams use fabricated videos and images—sometimes featuring other celebrities—to convince victims to part with money, with reported losses ranging from a few hundred dollars to several thousand. The article highligh

