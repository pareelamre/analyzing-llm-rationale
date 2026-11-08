# Edge Board research review disposition

2026-10-31. External deep review `aud_MIHGd4PT6h`: five completed reviewers,
26 original findings (including seven marked blocking), consolidated into the
18 root-cause rows below. The initial panel verdict was `has-serious-issues`;
this document records subsequent verification and remediation, not a new clean
panel verdict. No second audit was run for this logical change.

V1–V5 identify reviewers in the returned result order. All accepted findings
were addressed; no unresolved finding requires a user decision. Tests use
synthetic market/news/provider inputs and do not establish live forecasting
quality, deployed latency or monthly API savings.

| finding | decision | action | verification |
|---|---|---|---|
| V1/f1, V4/f2, V5/f9 (major): prefetch can delay inference | accepted | Use a 20-second shared deadline, at most two daemon workers across retries, direct model-free fetch/rank without nested executors, and explicit unavailable packets. | Deadline regression passes; a separate Python process exited in 1.88s while its daemon adapter remained stalled for 300s. |
| V1/f2, V3/f3, V4/f1 (major/minor): failed packets bypass research retry | accepted | Only a successful book/evidence packet marks initial research available; retain the existing retry for failed/malformed packets. | test_missing_prefetch_data_still_requires_the_research_retry; successful packets still permit one-turn finalization. |
| V2/f1 (major/blocking): consecutive initial user messages | accepted | Combine bounded initial observations into the first user message, preserving system/user roles. Accept compatibility hardening without asserting an untested provider HTTP failure. | First-turn test asserts exactly system/user roles; no live vendor call was made. |
| V2/f2, V4/f4, V5/f3 (minor/major/blocking): None truncation fallback | accepted | Use explicit None handling and logical model calls for fallback; persist model_calls in cycle transcript JSON. The actual AgentReport already copied res.truncated, so the claimed universal inversion was overstated. | Worker persistence regression checks None fallback, truncated=1, tool steps excluding prefetch and durable logical calls. |
| V2/f3, V4/f3 (minor): one malformed article drops all evidence | accepted | Skip malformed URLs and nonnumeric/nonfinite relevance per article, retaining valid sources. | Malformed URL, None and NaN relevance test keeps the valid source. |
| V3/f1, V5/f7 (major/minor): only mocked integration shapes | accepted | Exercise agent_analyze to real tool loop, real NewsPipeline lexical ranking, actual Kalshi envelope adapter, reversed Polymarket outcome normalization, and malformed input persistence checks. | Integration/contract regressions and real SDK telemetry smoke use synthetic HTTP/inference inputs only. |
| V3/f2 (major): ambiguous Kalshi price units and bid sides | accepted | Label bid ladders and units; derive YES/NO buy asks in dollars and instruct rechecking before execution. | Cent and dollar ladder fixtures verify derived 0.46 YES buy ask and 0.59 NO buy ask. |
| V3/f4 (minor): missing search query sends portfolio prompt | accepted | Return an error observation for missing/blank query; never fall back to the worker prompt. | test_missing_followup_query_does_not_send_the_worker_prompt_to_search asserts no search call. |
| V3/f5 (minor): trace context lost across worker threads | accepted | Copy context into each daemon worker and create a market span; mark caught book/news errors. | Real SDK OTLP smoke exports linked research.news, research.market and research.prefetch spans plus counters. |
| V3/f6 (minor): Polymarket token may be NO | rejected | The trusted worker quote comes from market_data._polymarket_quote, which selects the token at the Yes outcome's index. Public callers cannot supply private candidates. | Real normalizer fixture with outcomes No/Yes and tokens 54321/12345 sends token 12345; malformed numeric tokens are rejected. |
| V3/f7 (minor): deployed environment could override default | rejected | The change explicitly promises a default with MAX_TOOL_STEPS override, not a deployed turn-count reduction. No repository workflow sets MAX_TOOL_STEPS; deployments remain unmodified. | Repository workflow search found no override; operator documentation preserves the environment override and requires measured cycles. |
| V4/f5 (minor): first turn can exceed token budget | rejected | Retain the existing soft research budget semantics: at least one model turn and terminal finalization remain permitted. This change bounds added initial context and does not introduce a hard billing/token cap. | run_tool_loop retains existing and-step guard and mandatory terminal call; default research budget is 200000 estimated tokens, packet context is capped at 12000 characters. |
| V5/f1 (major/blocking): non-mapping candidate crashes | accepted | Skip non-mappings before reading fields; isolate packet failures. | Invalid identity test includes None and string candidates and verifies no network calls. |
| V5/f2 (major/blocking): malformed observations crash/repr injection | accepted | Skip items missing string action/observation. Existing bounded_observation already bounded repr text, so no bound or public privilege bypass was demonstrated. | Malformed initial item is skipped while the research retry remains active; typed string observations reach the first prompt. |
| V5/f4 (major): incorrect book sorting | rejected | The finding's own evidence confirms the implementation sorts bids descending and asks ascending with a three-level slice; equal-price stable sorting does not violate the contract. | Kalshi bids and Polymarket asks regressions assert correct best levels and the three-level bound. |
| V5/f5 (major/blocking): candidate-order contradiction | rejected | The finding explicitly retracts its own claim. The worker passes new quotes then held quotes, exactly matching documentation. | run_cycle research_candidates=[*new_quotes, *held_quotes]; duplicate truncation issue is handled separately. |
| V5/f6 (minor): hidden inference/publication-date enforcement | rejected | Direct NewsPipeline fetch/rank/select disables query planning, summarization and embeddings. Every source includes published_at; the documented policy asks the model to check dates rather than claiming automatic freshness enforcement. | Real ranking contract raises if summarize is called; NewsPipeline only creates an LLM when planning or summaries are enabled; system rule states empty news is not negative evidence. |
| V5/f8 (major/blocking): require model tool even after successful prefetch | rejected | Successful system-fetched data is research and intentionally enables a one-turn decision. Requiring a redundant model tool would undo the approved optimization. Failed packets retain the guard. | One-turn valid seed test and missing-data retry test distinguish successful preparation from failed retrieval. |

The extended Semgrep scan reported nine legacy findings; an identical scan of
the base server source returned the same nine rule/source fingerprints. No
introduced finding was identified. Production model credentials, schedules,
roster and execution permissions were not changed.
