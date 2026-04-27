#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@dataclass
class Block:
    page: int
    x: float
    y: float
    w: float
    h: float
    content: str
    wrap: int
    font_size: float = 9.0
    leading: float = 10.8
    title_lines: list[str] = field(default_factory=list)
    title_size: float = 12.0
    fill: bool = True


def wrap_markdownish(content: str, width: int) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for raw in content.strip().splitlines():
        line = raw.rstrip()
        if not line:
            lines.append(("blank", ""))
            continue
        if line.startswith("## "):
            lines.append(("header", line[3:]))
            continue
        if line.startswith("- "):
            wrapped = textwrap.wrap(
                line[2:],
                width=width - 2,
                initial_indent="- ",
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            for item in wrapped:
                lines.append(("body", item))
            continue
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for item in wrapped:
            lines.append(("body", item))
    return lines


def render_block(block: Block) -> bytes:
    commands: list[str] = ["q"]
    if block.fill:
        commands.extend(
            [
                "1 1 1 rg",
                f"{block.x:.1f} {block.y:.1f} {block.w:.1f} {block.h:.1f} re",
                "f",
            ]
        )

    cursor = block.y + block.h - 18
    for title in block.title_lines:
        commands.extend(
            [
                "BT",
                "/FBOLD "
                + f"{block.title_size:.1f} Tf",
                "0 g",
                f"{block.x + 10:.1f} {cursor:.1f} Td",
                f"({pdf_escape(title)}) Tj",
                "ET",
            ]
        )
        cursor -= block.title_size + 6

    for kind, text in wrap_markdownish(block.content, block.wrap):
        if kind == "blank":
            cursor -= block.leading * 0.55
            continue
        if kind == "header":
            commands.extend(
                [
                    "BT",
                    "/FBOLD "
                    + f"{block.font_size + 1.0:.1f} Tf",
                    "0 g",
                    f"{block.x + 10:.1f} {cursor:.1f} Td",
                    f"({pdf_escape(text)}) Tj",
                    "ET",
                ]
            )
            cursor -= block.leading * 1.2
            continue
        commands.extend(
            [
                "BT",
                "/FREG "
                + f"{block.font_size:.1f} Tf",
                "0 g",
                f"{block.x + 10:.1f} {cursor:.1f} Td",
                f"({pdf_escape(text)}) Tj",
                "ET",
            ]
        )
        cursor -= block.leading

    commands.append("Q")
    return "\n".join(commands).encode("utf-8")


TITLE_BLOCK = Block(
    page=1,
    x=60,
    y=718,
    w=480,
    h=92,
    content="",
    wrap=80,
    fill=True,
    title_lines=[
        "Prompt Structure and Forecast Reliability",
        "in LLM Forecasting",
        "A Large-Scale Prompt Ablation Study",
    ],
    title_size=15.5,
)

ABSTRACT_BLOCK = Block(
    page=1,
    x=60,
    y=190,
    w=235,
    h=430,
    wrap=29,
    font_size=10.0,
    leading=11.7,
    content="""
## Abstract
Large language models can produce plausible forecasting rationales, but it is
unclear whether adding structure improves forecast quality. We evaluate 1,580
resolved binary questions from Metaculus with three models
(Qwen2.5-7B-Instruct, Qwen3-32B, and GPT-OSS-120B), nine prompt variants, and
six temperatures per model. We score 162 runs with strict accuracy, Brier
score, and Expected Calibration Error. Across the full sweep, the neutral
baseline is the strongest overall setting on average, showing that extra
rationale constraints often hurt both accuracy and calibration. Among
structured prompts, temporal anchors and credibility are the most robust:
temporal anchors produce the best single run, reaching 83.0% accuracy with
GPT-OSS-120B at T=0.125, while credibility prompting gives Qwen3-32B its best
calibration. By contrast, predicted-event restatement, key attributes, key
conditions, step-by-step reasoning, and uncertainty language usually
underperform the baseline. Temperature effects are model-specific rather than
monotonic, so rationale design should be tuned jointly with model family and
decoding settings.
""",
)

PAGE1_CONTEXT = Block(
    page=1,
    x=300,
    y=180,
    w=240,
    h=245,
    wrap=40,
    content="""
## Benchmark Context
Dedicated forecasting benchmarks such as Autocast, ForecastBench, and FutureX
already evaluate forward-looking prediction with verifiable resolutions. These
studies show that retrieved evidence helps, but LLM forecasts remain sensitive
to temporal grounding, calibration, and prompt design.

This paper therefore studies a narrower question: when question and evidence
context are held fixed, which prompt-imposed rationale fields help or hurt
forecast quality?
""",
)

PAGE2 = Block(
    page=2,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## Our Work
We run a prompt-format ablation on 1,580 resolved binary Metaculus questions.
The released runs cover three models (Qwen2.5-7B-Instruct, Qwen3-32B, and
GPT-OSS-120B), nine prompt variants (V0-V8), and six temperature settings per
model, yielding 162 model-variant-temperature evaluations.

We report strict accuracy, Brier score (Brier, 1950), and ECE-10 (Guo et al.,
2017). A small human study is retained only as exploratory appendix evidence
because agreement is low and the protocol is underpowered for strong
preference claims.

## Main Findings
- The neutral baseline is the strongest average setting across the full sweep.
- Temporal anchors and credibility prompting are the most competitive
  structured variants.
- Predicted-event restatement, key conditions, step-by-step prompting, and
  forced uncertainty usually underperform the baseline.
- Temperature effects are model-specific rather than monotonic; this revision
  does not claim a universal optimal T.

## 1.1 Research Questions
- Which prompt-imposed rationale fields help or hurt forecast metrics when the
  evidence context is held fixed?
- When a structured prompt helps, does it appear to add grounding or mostly
  add output burden?
- Do human preferences for rationales align with quantitative forecast quality?

## 1.2 Prompt Dimensions and Design Rationale
These dimensions are treated as exploratory design axes rather than
pre-registered hypotheses. V1 tests outcome restatement for label alignment;
V2 tests specificity via key attributes; V3 tests explicit epistemic framing;
V4 tests evidence-credibility grounding; V5 tests causal conditions from
superforecasting practice; V6 tests step decomposition; V7 tests verbal
uncertainty; and V8 tests temporal anchoring. The set is motivated by
crowd-forecasting practice, calibration work, and prompting literature.

## 1.3 Contributions
- A large-scale prompt ablation over 162 runs on 1,580 resolved questions.
- Evidence that richer rationales do not reliably improve forecasting metrics.
- A cautious account of human preference: low-agreement judgments are not
  treated as evidence of rationale faithfulness.
""",
)

PAGE3 = Block(
    page=3,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## Calibration and Rationale Faithfulness
Calibration is used here as a proxy for probabilistic reliability, not as a
direct measure of explanation faithfulness. A well-calibrated forecast can
still be justified by a post-hoc rationale, and a faithful rationale can
accompany a poorly calibrated probability. The revised paper therefore
separates claims about forecast quality from claims about explanation quality.

## 2 Related Work
Three strands of prior work are most relevant. First, live forecasting
benchmarks such as Autocast, ForecastBench, and FutureX (Zeng et al., 2025,
arXiv:2508.11987), as well as retrieval-based forecasting studies, show that
LLMs can use time-bounded evidence but still lag strong human forecasters and
remain brittle under real-world uncertainty. Our work complements these
benchmarks by holding the question/evidence context fixed and varying prompt
structure.

Second, temporal reasoning and calibration work explains why this setting is
difficult. Benchmarks such as TimeBench, TRAM, TORQUE, and time-sensitive QA
datasets document weaknesses in temporal grounding, event ordering, and horizon
sensitivity. Calibration work likewise shows that verbal confidence and numeric
reliability often diverge. This motivates evaluating both accuracy and
calibration rather than raw answer correctness alone.

Third, explanation and rationalization research warns that plausible
justifications are not necessarily faithful. ERASER, ROSCOE, and related
faithfulness work evaluate evidence alignment or reasoning quality directly.
Our paper does not do that. Instead, it studies prompt-format ablations: does
asking for predicted events, key conditions, steps, credibility, uncertainty,
or temporal anchors change forecast metrics?

This positioning matters for the paper's claims. The contribution is not a
comprehensive evaluation of rationale quality in the abstract. It is a
controlled study of how different structured rationale fields interact with
forecasting accuracy and calibration. Human judgments are included only as
noisy, exploratory signals about perceived helpfulness.
""",
)

PAGE4 = Block(
    page=4,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## 3 Approach
Figure 1 should be read as a high-level pipeline sketch rather than an exact
count summary. The revised experiments operate on the current released corpus
and run logs.

## 3.1 Prompt Variants
We evaluate nine prompt variants. V0 is a neutral baseline. V1 asks the model
to restate the predicted event. V2 requests a key attribute (for example
actor, quantity, or time). V3 asks for an explicit reasoning type. V4 asks the
model to ground its rationale in evidence credibility. V5 requests key
conditions. V6 requests short step-by-step reasoning. V7 asks for uncertainty
language. V8 requests temporal anchors. Each variant changes one prompt
dimension while preserving the same forecasting task and JSON schema.

## 3.2 Experimental Setup
The current paper reports three models from the released config set:
Qwen2.5-7B-Instruct, Qwen3-32B, and GPT-OSS-120B. Exact model identifiers are
the repo configuration values Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-32B, and
openai/gpt-oss-120b. All runs return predicted_answer, confidence, and
rationale, plus the variant-specific structured field when applicable.

We sweep six temperature settings per model. Because provider interfaces
differ, the exact final tags are not identical across models; the paper
therefore compares temperatures descriptively rather than as a perfectly
factorial control.

## 3.3 Dataset Construction
The dataset is
forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json,
which contains 1,580 resolved binary Metaculus questions. Each record includes
the question text, resolution criteria, answer, timestamps, categories, and
associated news articles. Articles store raw text, summaries, keywords,
structured future-related snippets, and heuristic credibility metadata.
""",
)

PAGE5_CAPTION = Block(
    page=5,
    x=58,
    y=430,
    w=482,
    h=72,
    wrap=92,
    font_size=8.7,
    leading=10.0,
    content="""
Figure 1: Updated study scope. The released evaluation uses 1,580 resolved
Metaculus questions, nine prompt variants, three models, and six temperatures
per model. The figure should be read as a pipeline sketch rather than as a
count-accurate summary of the current runs.
""",
)

PAGE5 = Block(
    page=5,
    x=58,
    y=118,
    w=482,
    h=300,
    wrap=88,
    content="""
Evidence articles include URL, title, publish date, full text, summary,
summary_llm, keyword lists, structured FRS fields, and a heuristic credibility
score. The credibility score is stored as released article metadata and is
derived from source_reliability, track_record, claim_support, and
timeliness_proximity components. The revised paper therefore refers to
heuristic credibility rather than to a separate "extraction confidence"
variable.

To reduce leakage, article timestamps are checked against question resolution
dates and post-resolution evidence is excluded. Appendix A now serves as the
concrete example pointer for a representative record.

We evaluate on the full 1,580-question corpus rather than on the earlier
210-question analysis split used in the draft. The present paper is
descriptive and comparative: it reports model/variant/temperature outcomes
over the full released dataset instead of presenting a tuned held-out
benchmark.

## 3.4 Evaluation Metrics
We use three complementary metrics from first mention onward: accuracy, Brier
score (Brier, 1950), and ECE-10 (Guo et al., 2017). Formal definitions are
kept in Appendix B. This metric suite measures forecast correctness and
calibration, but not direct rationale faithfulness or logical validity.
""",
)

PAGE6 = Block(
    page=6,
    x=58,
    y=82,
    w=482,
    h=670,
    wrap=88,
    content="""
## 3.5 Human Evaluation (Exploratory)
We keep the human study only as an exploratory appendix analysis of perceived
preference and concreteness. Participants compared selected structured
rationales against V0 in pairwise form. Krippendorff's alpha is used as a
chance-corrected inter-rater agreement coefficient.

Because agreement is low in the final retained analysis (alpha about 0.02 for
preference and 0.20 for concreteness), the revised paper does not use these
judgments to validate prompt superiority or explanation faithfulness. They are
reported only to document how noisy subjective preference is in this setting.

## 3.6 Implementation Details
The experiments are run from a shared config-driven pipeline. Each run records
model identifiers, temperature tags, output schema, prompt hashes, and result
metadata next to the result files. Outputs are stored as structured JSON plus
run_metadata JSON, not as anonymous free text.

## Aggregate Prompt-Level Summary
Across all model-temperature cells, V0 is the strongest average setting (mean
accuracy 0.758, mean Brier 0.189, mean ECE 0.105). Among structured prompts,
V4 credibility (0.759 / 0.195 / 0.109) and V8 temporal anchors
(0.756 / 0.193 / 0.111) are the most competitive. The largest average
degradations relative to V0 come from V5 key conditions, V6 step-by-step
reasoning, and V7 uncertainty language.

## Best Stable Runs
- GPT-OSS-120B: V8 at T=0.125 reaches 0.830 accuracy.
- Qwen2.5-7B-Instruct: V0 at T=0.000 reaches 0.754 accuracy.
- Qwen3-32B: V4 at T=0.000 reaches 0.740 accuracy; V4 at T=0.075 gives the
  best stable ECE (0.075).

This replaces the cluttered draft figures and tables with the current
count-accurate summary.
""",
)

PAGE7 = Block(
    page=7,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## 4 Results and Analysis
## 4.1 Model-Level Performance
GPT-OSS-120B is the strongest model overall in the released sweep, with mean
accuracies around 0.81-0.82 across its strongest settings. Qwen2.5-7B-Instruct
is consistently mid-tier, typically around 0.72-0.74 depending on variant and
temperature. Qwen3-32B is weaker on average but benefits disproportionately
from credibility prompting.

## 4.2 Prompt Variant Effects
The neutral baseline is the best reference point for most model-temperature
cells. V4 credibility and V8 temporal anchors are the only structured prompts
that remain consistently competitive. V1 predicted-event restatement and V5
key conditions are the most harmful on average. V6 step-by-step reasoning is
mildly positive only for GPT-OSS-120B and negative on the two Qwen models.

## 4.3 Calibration Analysis
Calibration gains are model-specific. Qwen3-32B shows its clearest improvement
under V4 credibility, while V8 temporal anchors stays competitive without
being universally best. Extremely low ECE values at incomplete high-temperature
runs should not be over-interpreted; missing outputs make those settings less
trustworthy as deployment recommendations.

## 4.4 Human Evaluation
The human study does not support strong qualitative claims. Low agreement and
roughly 21 judgments per variant comparison make the preference signals too
unstable for confirmatory conclusions. We therefore use them only to note that
readers may prefer richer causal rationales even when those rationales do not
improve forecast metrics.

## 4.5 Uncertainty Prompting
Forcing uncertainty language does not reliably improve calibration. The main
effect of V7 is lexical: rationales contain more hedging, but forecast quality
usually worsens relative to V0.

## 4.6 Temperature Sensitivity
The revised multi-model sweep does not support the earlier claim that moderate
stochasticity is broadly optimal. Qwen2.5 and Qwen3 achieve their best average
accuracy at or near deterministic decoding. GPT-OSS-120B is more stable across
low temperatures and reaches its best single run at T=0.125, but this is a
model-specific result rather than a general rule.
""",
)

PAGE8 = Block(
    page=8,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## 5 Discussion
This revision reframes the paper as a study of prompt structure and forecast
reliability, not as a direct evaluation of intrinsic rationale quality. Most
added structure acts as an output constraint. For weaker models, that extra
structure often behaves like cognitive load: the model spends capacity
formatting a rationale instead of allocating it to the forecast. The
exceptions, V4 and V8, are lightweight grounding cues rather than verbose
reasoning formats.

Temporal anchors likely help because they tie the rationale to deadlines and
time windows that are already central to Metaculus-style questions. Credibility
prompting helps when the model can exploit source-quality cues without
generating long multi-part justifications. By contrast, V5 and V6 frequently
ask the model to verbalize causal structure or steps that may sound compelling
without improving the probability estimate.

## 6 Limitations
This paper still has important limits. It uses one forecasting platform, three
currently released models, and some temperature settings with incomplete
outputs. The human study is exploratory, underpowered, and low-agreement; the
earlier draft also underreported recruitment, compensation, and ethics details,
so the revised paper avoids using that study for strong claims. Most
importantly, the paper does not measure rationale faithfulness, factual
alignment, or logical consistency directly. It measures forecast metrics under
different prompt formats.

## 7 Conclusion and Future Work
The current evidence supports a narrower conclusion than the draft: richer
rationale formats do not reliably improve LLM forecasting, while temporal
anchoring and credibility grounding are the most robust structured
interventions in the present sweep. Future work should evaluate rationale
content directly, add stronger statistical testing, preregister human studies,
and test whether these prompt effects transfer beyond Metaculus.
""",
)

PAGE12 = Block(
    page=12,
    x=58,
    y=96,
    w=482,
    h=290,
    wrap=88,
    content="""
## D Human Evaluation Protocol
The human study is retained for transparency but is not central to the paper's
claims. It uses 21 volunteer participants and 126 pairwise judgments, roughly
21 per structured variant versus V0. Because the sample is small and agreement
is low, the revised paper treats the study as exploratory.

Participants saw the question, resolution criteria, and two anonymized
rationales in randomized order, then selected a preferred rationale and rated
concreteness on a 1-5 scale. Ground truth outcomes were withheld. The protocol
description in the draft was incomplete with respect to recruitment,
compensation, and ethics reporting; this revision therefore avoids presenting
the study as strong human-subject evidence.
""",
)

PAGE13 = Block(
    page=13,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## D.4 Inter-Annotator Reliability
The final retained agreement estimates are low: Krippendorff alpha is
approximately 0.02 for preference and 0.20 for concreteness. These values
indicate that subjective judgments of rationale quality are unstable in this
task. The earlier contradictory moderate-agreement numbers in the draft should
be disregarded.

## E Human Evaluation Details
With about 21 pairwise judgments per variant comparison, the human results
should be interpreted only as noisy descriptive signals. V5 received the
highest preference share despite weak forecast quality, which is consistent
with a reader bias toward explicit causal wording. That pattern is interesting,
but it is not strong enough to support claims about explanation quality,
faithfulness, or user trust.

The revised manuscript therefore keeps the appendix table only as an
exploratory summary and removes any strong conclusion that humans "prefer the
best rationale" or that preference implies better reasoning.
""",
)

PAGE14 = Block(
    page=14,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## F Uncertainty Prompting Revisited
The uncertainty appendix should be read as a lexical analysis, not as evidence
that verbal hedging produces better forecasts. V7 reliably increases hedge
terms, but its quantitative effects are weak and often negative. This mirrors
the main-text result: linguistic uncertainty and numeric calibration are
related only loosely.

## G Extended Temperature Analysis
Temperature is now analyzed across three models rather than only one. The
central pattern is variation, not a universal optimum. Qwen2.5-7B-Instruct and
Qwen3-32B perform best on average at or near deterministic settings. GPT-OSS-
120B remains stable across low temperatures and reaches its best single run
under V8 at T=0.125, but that does not generalize to the other models.

When ECE improves at extreme settings, the revision checks completeness before
drawing conclusions. Some of the lowest high-temperature ECE values coincide
with missing outputs, so they are not treated as reliable deployment defaults.
""",
)

PAGE15 = Block(
    page=15,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## G.1 Aggregate Temperature Summary
- GPT-OSS-120B: best average accuracy at T=0.00; best single run at V8,
  T=0.125; high-temperature ECE gains are confounded by missing outputs.
- Qwen2.5-7B-Instruct: best average accuracy and Brier score at T=0.00; best
  stable ECE at T=0.125.
- Qwen3-32B: best average accuracy and Brier score at T=0.00; best stable ECE
  at T=0.025.

## G.2 Interpretation
The revised result is therefore conservative. Temperature should be tuned per
model, and often per deployment objective. If accuracy is primary,
deterministic or near-deterministic decoding is a strong default for the
present models. If calibration is primary, modest retuning may help, but no
single temperature dominates across architectures.

## G.3 Practical Guidance
- Start from V0 or V8 rather than from complex multi-field prompts.
- Sweep temperature on held-out validation data for the exact model you plan to
  use.
- Treat incomplete high-temperature runs as warnings about robustness, not as
  evidence of superior calibration.
""",
)

PAGE16 = Block(
    page=16,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## J Extended Discussion
The updated evidence suggests that prompt structure helps when it adds compact
grounding constraints and hurts when it demands verbose self-explanation. V4
and V8 fall into the first class. They ask the model to attend to source
reliability or timing without forcing long chains of prose. V5 and V6 fall
into the second class: they often encourage plausible but not necessarily
useful elaboration.

This distinction matters for explainable AI claims. A prompt can make outputs
look more structured without making the underlying prediction process more
faithful or more reliable. The revised paper therefore avoids language that
equates better calibration with a "more reliable reasoning process".
Calibration is about the honesty of the probability estimate, not about
whether the verbal rationale faithfully mirrors the model's internal
computation.

Human preference results fit the same story. Readers may reward concrete
conditional language even when it does not improve the forecast. That gap is
important for interface design: systems should not use rationale likability as
a substitute for forecast quality.
""",
)

PAGE17 = Block(
    page=17,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## K Limitations
- Scope: one platform, one released Metaculus-style corpus, and three evaluated
  models.
- Missingness: some temperature/model combinations have incomplete outputs,
  especially at extreme settings.
- Human study: small convenience sample, low agreement, and incomplete protocol
  reporting in the original draft.
- Rationale quality: no direct annotation of factual alignment, citation
  support, logical consistency, or faithfulness.
- Inference strength: most comparisons are descriptive; many differences are
  small enough that they should be read cautiously rather than as universal
  laws.

These limitations motivate a narrower interpretation of the paper. The work
provides evidence about prompt-format sensitivity in forecasting pipelines. It
does not yet establish a general theory of high-quality rationales.
""",
)

PAGE18 = Block(
    page=18,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## L Prompt-Dimension Summary
- V0 neutral baseline: strongest average reference point across the sweep.
- V1 predicted event: usually harmful, suggesting that explicit restatement
  adds burden more often than alignment.
- V2 key attribute: mixed, but rarely better than V0.
- V3 reasoning type: small and model-dependent effects.
- V4 credibility: strongest structured variant for Qwen3 and one of the two
  best structured options overall.
- V5 key conditions: often preferred by readers, but quantitatively weak.
- V6 step-by-step: mildly helpful only for GPT-OSS-120B; otherwise negative.
- V7 uncertainty language: increases hedging more than it improves calibration.
- V8 temporal anchors: best single run and the most robust structured scaffold.

## M Future Directions
Next steps should directly score rationale content: factual support from
retrieved evidence, internal logical consistency, citation grounding, and
faithfulness-oriented tests. A preregistered human evaluation with clearer
recruitment and ethics reporting would also be necessary before making stronger
user-facing claims.
""",
)

PAGE19 = Block(
    page=19,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## N Practical Use
For the current models, the safest default is to start with V0. If a
structured rationale is required, V8 is the most portable option and V4 is
promising when evidence credibility metadata are available. More verbose
rationale formats should be treated as optional interface layers, not as
default forecasting aids.

## O Reproducibility
The revised claims are backed by the released repo artifacts: prompt templates
in Appendix A and prompts/, model identifiers in configs/models.yaml, run
outputs in results/, and aggregated metrics in
analysis/metrics_by_model_temperature_variant.csv. Each run also records
metadata such as temperature tags and prompt hashes.
""",
)

PAGE20 = Block(
    page=20,
    x=58,
    y=118,
    w=482,
    h=655,
    wrap=88,
    content="""
## End of Appendix
This revised PDF supersedes the earlier draft's model-selection FAQ and the old
five-model appendix analysis. The paper's current claims are the ones stated
in Sections 1-7 and Appendices D-G and J-O above.
""",
)


BLOCKS = [
    TITLE_BLOCK,
    ABSTRACT_BLOCK,
    PAGE1_CONTEXT,
    PAGE2,
    PAGE3,
    PAGE4,
    PAGE5_CAPTION,
    PAGE5,
    PAGE6,
    PAGE7,
    PAGE8,
    PAGE12,
    PAGE13,
    PAGE14,
    PAGE15,
    PAGE16,
    PAGE17,
    PAGE18,
    PAGE19,
    PAGE20,
]


def ensure_fonts(writer: PdfWriter, page) -> None:
    resources = page.get("/Resources")
    if resources is None:
        resources = DictionaryObject()
    else:
        resources = resources.get_object()

    fonts = resources.get("/Font")
    if fonts is None:
        fonts = DictionaryObject()
    else:
        fonts = fonts.get_object()

    regular_font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    bold_font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
        }
    )

    fonts[NameObject("/FREG")] = writer._add_object(regular_font)
    fonts[NameObject("/FBOLD")] = writer._add_object(bold_font)
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources


def append_stream(writer: PdfWriter, page, stream_data: bytes) -> None:
    overlay_stream = DecodedStreamObject()
    overlay_stream.set_data(stream_data)
    overlay_ref = writer._add_object(overlay_stream)

    contents = page.get("/Contents")
    if contents is None:
        page[NameObject("/Contents")] = overlay_ref
        return
    if isinstance(contents, ArrayObject):
        contents.append(overlay_ref)
        return
    page[NameObject("/Contents")] = ArrayObject([contents, overlay_ref])


def update_pdf(input_path: Path, output_path: Path, backup_path: Path | None) -> None:
    if backup_path is not None and not backup_path.exists():
        shutil.copy2(input_path, backup_path)

    reader = PdfReader(str(input_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    for block in BLOCKS:
        page = writer.pages[block.page - 1]
        ensure_fonts(writer, page)
        append_stream(writer, page, render_block(block))

    with output_path.open("wb") as handle:
        writer.write(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay a revised ACL manuscript onto the PDF.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ACL_Evaluating Rationale of LLMs.original.pdf"),
        help="Input PDF path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ACL_Evaluating Rationale of LLMs.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=Path("ACL_Evaluating Rationale of LLMs.revision_backup.pdf"),
        help="Optional backup path for the current output. Pass an empty string to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backup_path = None if str(args.backup) == "" else args.backup
    if args.output.exists() and backup_path is not None and not backup_path.exists():
        shutil.copy2(args.output, backup_path)
    update_pdf(args.input, args.output, None)


if __name__ == "__main__":
    main()
