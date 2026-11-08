# Foresea Fund

One paper portfolio, run by a team of models the owner controls, with a
deterministic orchestrator making the decisions. This file is the owner's brief
(2026-11-08) and the design that follows from it; change it only with the owner.

## The brief

- **One portfolio.** All models work together on a single account. The Agentic
  dashboard may include any model, not just one.
- **Owned models.** As much as possible runs on open weights or Foresea's own
  code, on the owner's Mac or free CI runners: Qwen3.5-9B on the Mac, FinBERT,
  classifiers trained on Foresea's own graded forecasts, weather models, and the
  index and pairs models already on the site. No paid model APIs.
- **All instruments are open.** Prediction markets (Kalshi, Polymarket), Kalshi
  perpetuals and futures, NSE/BSE indices and MCX commodities.
- **Start small and liquid.** Indices and commodity pairs start with the 10 of
  each that trade the most volume.
- **Offset risk across venues.** The same event on Kalshi and Polymarket; an MCX
  position against a Kalshi perp or a prediction-market price contract on the
  same underlying.
- **A strong orchestrator.** Models advise; plain code decides, sizes, hedges and
  records.
- **Solid attribution.** Every trade records which models drove it and by how
  much; profit and loss is split back by model, asset class and hedge.
- **Grounded reasons.** Each position's "why" is built from the models' numbers
  and cited sources. The LLM only puts them into words, and a note that states
  a figure not in its inputs is rejected for a plain fact sheet.

## Design

```
analysts ──► signals ──► orchestrator ──► paper ledger ──► Fund page
 (models)    (one shape)   combine          one USD account   holdings, trades,
                           size             positions in      notes, attribution
                           hedge            every venue
                           attribute
```

- **Signal**: instrument, direction, horizon, edge (probability gap or expected
  return), uncertainty, evidence (source, figure) and the analyst that made it.
- **Combine**: analysts on the same instrument are weighted by measured skill;
  equal weights until each has a graded record.
- **Size**: fractional Kelly with hard limits: 10% of the account per position,
  40% per asset class, 20% kept in cash.
- **Hedge**: net exposure per underlying (gold, crude, US equities, Indian
  equities, each event) across venues, and offset it with the cheapest liquid
  leg: a Kalshi perp, a prediction-market price contract, or the same event on
  another venue. Perp funding (every 8 hours) counts as a cost.
- **Execute**: paper fills at the quoted bid/ask; INR instruments are held in
  USD at the day's USD/INR.
- **Attribute**: each fill stores the signal weights that produced it; daily
  P&L is split by analyst, asset class and hedge from those weights.

## Sources

| Instruments | Prices | Notes |
|---|---|---|
| Kalshi, Polymarket events | existing venue clients | |
| Kalshi perps and futures | `GET /trade-api/v2/margin/markets` (public) | a missing ask comes back as 9223372036854775807 |
| NSE/BSE indices, MCX | Upstox open candles (`index_forecast`, `commodity_pairs`) | |
| USD/INR, gold, platinum | fawazahmed0 currency API | |
| Brent, WTI, Henry Hub | FRED | a few days behind |

## Build order

1. **Ledger and orchestrator.** The current $10,000 on-device account becomes
   the fund. Analysts: the Mac model's forecasts (prediction markets), the index
   model and the pairs model. Instruments: the top 10 indices and top 10
   commodity pairs by volume, plus their Kalshi hedge legs. Grounded notes,
   attribution, and the Fund view on the Agentic tab. Twice a day, after the
   agent cycle.
2. FinBERT news sentiment.
3. Weather models for weather markets.
4. A classifier trained on Foresea's graded forecasts: which gaps pay.
5. Skill-based weights once each analyst has a record.
