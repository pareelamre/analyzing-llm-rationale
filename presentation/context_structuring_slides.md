# Context Structuring for LLM Reasoning

---

## Slide 1. Title
**Structuring Context for Better LLM Reasoning**

*How the way you frame information changes what an LLM can do with it*

---

## Slide 2. The Core Idea
**Not how much context — but which kind, and how it's shaped**

- Giving an LLM a raw question leaves it with nothing to anchor on
- Dumping too much unstructured information overwhelms it
- The real lever is **deciding what goes in, and in what form**

This project tests that idea directly — by building a dataset where each question is paired with carefully structured context, then measuring how different structures affect LLM accuracy and calibration

---

## Slide 3. Experimental Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│               Metaculus (1,580 questions)                    │
│     binary · resolved · resolution criteria + news articles  │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     8 Prompt Variants       │
          │  each targets a different   │
          │   reasoning component       │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │        5 LLMs               │
          │  proprietary + open-weight  │
          │                             │
          │  Output: { answer,          │
          │            confidence,      │
          │            rationale }      │
          └──────────────┬──────────────┘
                         │
     ┌───────────────────▼──────────────────────┐
     │              Evaluation                   │
     │  Accuracy  ·  Brier Score  ·  ECE         │
     │  Human evaluation of rationale quality    │
     └───────────────────────────────────────────┘
```

---

## Slide 4. How Context Is Structured

Each question is augmented with layers of context — question, background, news evidence, and resolution criteria.

The key design choice: **curated and concise over raw and complete.**
Each layer adds a specific type of signal, not just more text.

---

## Slide 5. The 8 Prompt Variants

Beyond what context is included, the project tests *how the reasoning is framed*.

Each variant asks the model to structure its rationale around a different component — the predicted outcome, the key conditions, the reasoning type, the uncertainty, the temporal deadline, and more.

The goal is to isolate which dimensions of structured reasoning actually improve forecasting.

---

## Slide 6. What the Results Show

- Some structures improve accuracy; others improve how humans perceive the reasoning — these are not the same thing
- Temporal grounding is the most consistent and transferable improvement across all models
- Forcing explicit uncertainty language can hurt calibration rather than help it
- The right structure depends on the model and what you are optimizing for

---

## Slide 7. Why This Generalizes

The same structuring choices appear in any domain where an LLM must reason under uncertainty — medicine, law, finance, policy.

The components tested here are not forecasting-specific. They are the building blocks of structured reasoning in any high-stakes context.

---

## Slide 8. The Takeaway

- Context structure is a design decision, not an afterthought
- Accuracy, calibration, and interpretability do not always point to the same structure
- This research gives a principled basis for deciding **what to tell an LLM** — not just which LLM to use

---
