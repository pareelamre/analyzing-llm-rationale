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

## Universe (measured 2026-11-08, `fund_universe.build_universe`)

Ranked by average daily notional over two weeks; re-ranked each month.

- **Indices (10):** Nifty 50 (~$1.3B/day), Nifty Bank, US Tech 100 (Kalshi perp),
  Nifty Midcap Select, US 500 (Kalshi perp), Sensex, Nifty Financial Services,
  Nifty Next 50, Bankex, BSE Focused IT. Sector indices have no futures, so
  they stay on the Indices page only.
- **Commodity pairs (10):** Gold/Silver, Crude/Natural gas, Copper/Gold,
  Copper/Zinc, Copper/Aluminium (all MCX); Crude MCX/Kalshi WTI, Gold MCX/Kalshi
  perp, Zinc/Lead (MCX), Silver MCX/Kalshi perp, Gold/Silver (Kalshi perps).
- MCX legs use the smallest liquid lot (Gold Petal, Silver Micro, Crude Mini,
  Natural Gas Mini, Zinc Mini, Aluminium Mini) so a $10,000 fund holds real lots.

## Models

Open-weight models come first; the owner's own fine-tunes follow.

- **Text:** FinBERT, CrudeBERT (FinBERT fine-tuned on crude-oil headlines with
  supply/demand events, arXiv 2305.06140), and the Mac's Qwen3.5-9B.
- **Time series:** pretrained forecasters such as Chronos, TimesFM and Moirai,
  next to the index model and pairs model already on the site.
- **Weather:** NWS and open-weight global models (GraphCast, Aurora, GenCast)
  for weather markets.
- **Fine-tuning:** BERT variants for the fund's own underlyings (no gold or
  Indian-index version of CrudeBERT exists), trained on Kaggle or Colab once the
  Wunderfund competition is over. Ask the owner before using Kaggle hours.

## Shadow mode

A new analyst does not trade on arrival. It records its probability for each
market every day (`analyst_predictions` in the fund store), the trade it would
have made, and is graded as markets resolve; it earns trust, and with it a
share of the fund's decisions, only from that record.

- **blend_model** (2026-11-09, `scripts/train_fund_blend.py`): a logistic
  regression on logit(market), logit(model) and their gap, trained on 80,988
  graded snapshots from 698 markets, each market weighted once. Time-split test
  on 116 later markets: Brier 0.1179 against the market's 0.1190 (difference
  -0.0011, 95% CI -0.0060 to +0.0042, so not yet distinguishable from the
  market). Raw model forecasts were worse than the market (+0.0076, CI excludes
  0); boosted trees overfit and were dropped. The weights lean on the market
  about nine times more than on the model.

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
