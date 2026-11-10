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
| NSE index options | Upstox order book with `UPSTOX_ANALYTICS_TOKEN`, else open last trades (`fund_options`) | last trades are widened 2% each side and need 100 lots traded |
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

## Options (added 2026-11-10, owner's request)

The fund trades all four instrument types on indices and commodities: futures
(NSE, BSE, MCX), Kalshi perps, and now options. Futures and perps were already
in the universe; they trade when the pairs model fires and as hedges. Nothing
yet gives a direction for an index future, because the index model has no skill
on direction.

The index model does forecast ranges well (2016-2026 walk-forward: its 50%, 80%
and 90% ranges held 49-51%, 78-80% and 88-90% of closes), and a range is exactly
what an option prices. So the **options analyst** (`fund_options`) values NSE
index options (Nifty 50, Nifty Bank, Nifty Financial Services, Nifty Midcap
Select, Nifty Next 50) with the model's own distribution:

- **Distribution:** the model's quantiles for 1, 5 and 21 sessions (and its 1%
  and 99% ones), interpolated to the time left by the square root of time, never
  given fatter tails than its own, then moved so its mean is the market's forward
  from put-call parity. No drift is traded.
- **Structures, defined risk only:** a straddle (MOVE) when the model expects a
  wider range than the options price in; an iron condor (RANGE), put credit
  spread (BULL) or call credit spread (BEAR) when it expects a narrower one.
  Short strikes sit near the model's quartiles and wings near its 10% and 90%
  levels. Each structure is one position that costs at most what it can lose;
  the fund never writes an uncovered option.
- **Entry:** the nearest expiry 2 to 25 sessions away; the best structure per
  index if the model values it at least 5% above its cost after 2% for charges;
  every leg liquid. An index whose model straddle is more than 50% from the
  market's is skipped as a likely model or data error.
- **Size:** at most 2% of the fund at risk per structure at the starting trust,
  10% across all options, one structure per index and expiry.
- **Exit:** at expiry, at the index's close; earlier when the model values a
  structure 5% or more below its bid after costs, the same bar as entries.
- **Commodity options** (MCX gold, silver, crude, natural gas, copper, zinc) wait
  for a commodity range model with its own backtest: the MCX price history the
  fund has is six months long, too short to forecast a range from.

The first live check, against the 2026-11-09 close: Nifty options priced about
12% above the model's range (the usual premium for selling options); the best
structure, an iron condor, was 2.6% above its cost after charges, under the 5%
bar, so nothing traded.

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
- **rationale_check** (2026-11-09, `scripts/fund_rationale_check.py`,
  `.github/workflows/fund-rationale-check.yml`): Open-Jev 2B (Apache-2.0 LoRA
  and decision head on Qwen3.5-2B, run on GitHub's CPUs) reads each on-device
  forecast's rationale and says whether it concludes the event is likely; a
  number pointing the other way is flagged. It is a guard against the
  2026-11-08 inversions, not a forecaster, and it blocks nothing in shadow.

## News sentiment evaluation (pre-registered 2026-11-09, before any result)

Question: does headline sentiment predict the next day's move well enough to
trade? Fixed before the data is seen:

- **Models:** ProsusAI/finbert (CC-BY-NC-SA-4.0, evaluated for research only)
  and mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
  (Apache-2.0, the one the fund could deploy). CrudeBERT has no license and
  is not used.
- **Headlines:** GDELT DOC 2.0 (free with attribution), English, the last
  three months, four groups: Indian equities, US equities, gold, crude oil.
- **Signal:** for each trading day t and instrument, the mean of
  P(positive) - P(negative) over headlines seen in the 24 hours before that
  instrument's close; no signal with fewer than 5 headlines. Nothing after
  the close is used.
- **Target:** the next close-to-close return. Instruments: Nifty 50 and Nifty
  Bank (Indian equities, close 10:00 UTC), S&P 500 (US, 20:00 UTC), MCX gold
  (18:00 UTC), WTI (18:30 UTC).
- **Split:** the first half of the dates calibrates only the threshold tau
  (the median |signal|); the sign is fixed (positive sentiment means long).
  Everything below is measured on the second half.
- **Metrics:** pooled Spearman correlation of signal and next-day return;
  a rule that goes long above +tau, short below -tau, holds one day, and pays
  0.05% per round trip on index futures and 0.10% on gold and oil; its mean
  net return per trade against always-long on the same days. Confidence
  intervals bootstrap over dates (keeping same-day instruments together), at
  97.5% because two models are tested.
- **GO**, for a model, only if all four hold: at least 150 test
  instrument-days with a signal; the correlation's interval above 0; the
  rule's net mean interval above 0; the rule beats always-long. Otherwise
  **NO-GO**: the model stays a shadow analyst, recording daily, and the test
  is re-run each month on the growing forward record.

### Amendment 1 (2026-11-09, before any data from the new source was read)

GDELT's rate limit let a 150-minute collection gather about 9 days of
headlines (909 in all), far short of the sample the rule needs, so the
headline source changes. **The signal, window, 5-headline minimum, split,
threshold, costs, metrics, 97.5% intervals and the four GO conditions are
unchanged.**

- **Headlines:** the `idleengine/financial-news-multisource` corpus, used
  under its non-commercial research terms (accepted by the owner), subsets
  `sp500_daily_headlines`, `cnbc_headlines`, `finsen_us_2007_2023`,
  `yahoo_finance_felixdrinkall`, `bloomberg_reuters` and
  `us_financial_news_2018_jeet` (FNSPID, 11 GB, and Reddit are left out).
  Only each item's first line (its headline) is scored.
- **Groups:** a headline joins a group when it matches that group's terms
  (US equities: S&P, Wall Street, Dow, Nasdaq, U.S. stocks; crude: crude,
  oil price(s), Brent, OPEC, WTI; Indian equities: Nifty, Sensex, Indian
  shares/stocks); every `sp500_daily_headlines` item is US equities.
- **Timing:** minute-level timestamps are used as given; a day-level item
  (stamped midnight UTC) is treated as seen at the end of its UTC day, so it
  can never inform that day's close.
- **Instruments and period:** S&P 500 (FRED, November 2016 on), WTI (FRED),
  Nifty 50 and Nifty Bank (Upstox, 2008 on), each over the dates where its
  headlines and prices overlap. **Gold is dropped from this test:** FRED no
  longer serves the LBMA fix and the alternative tried sits behind a bot
  check, which we do not get around; gold is judged on the forward record
  with MCX prices instead.

### Result (2026-11-09): NO-GO for both models

Run as pre-registered (`reports/fund_finbert_eval.json`). About 186k headlines
were built; 43k fell inside an instrument's window and were scored. The test
half had 1,513 instrument-days (S&P 500 1,109, WTI 396, Nifty 4 each), well
above the 150 required.

| | FinBERT | DistilRoBERTa |
|---|---|---|
| Rank correlation with next-day return | 0.005 (CI −0.056 to 0.068) | −0.008 (CI −0.070 to 0.054) |
| Rule's mean net return per trade | −0.05% (CI −0.32% to 0.22%) | −0.07% (CI −0.28% to 0.16%) |
| Beats always-long | yes | no |

Headline sentiment did not predict the next day's move, so neither model
trades in the fund. Re-running on the same corpus would change nothing; a new
test needs new data, such as sentiment recorded forward in shadow mode.

## Weather evaluation (pre-registered 2026-11-10, before any result)

**Question.** Does a calibrated forecast from eight weather models price Kalshi's
daily high-temperature markets better than the market does, by enough to pay
Kalshi's fees? If it does, the weather analyst trades; if not, it records its
predictions in shadow mode until a second test.

**Markets.** Every settled daily `KXHIGH*` series that settles on The Weather
Company (24 US cities, from New York and Los Angeles to Trenton). Kalshi moved
these contracts to The Weather Company on 2026-09-02, so the test uses events
from 2026-09-03 to the last settled day. The outcome is each market's result;
the observed high is its event's `expiration_value`.

**Prices.** Kalshi's hourly event candlesticks. At each decision time, the last
hourly candle that closed at or before it (no more than 3 hours earlier),
using its yes bid and yes ask closes.

**Forecasts.** Open-Meteo's Previous Runs API: hourly 2 m temperature at the
station's NWS coordinates from GFS, ECMWF IFS, ECMWF AIFS, ICON, GEM, JMA, UKMO
and Meteo-France, in degrees Fahrenheit. Each model's high is its largest hourly
value over the local calendar day.

**Decision points** (the fund's two cycles), with forecasts no newer than a
real cycle would have had:

- **A:** 08:00 UTC on day D, that day's markets, `previous_day1` forecasts.
- **B:** 20:00 UTC on day D-1, the next day's markets, `previous_day2` forecasts.

**Model (fixed here, not tuned on the results).**

- Centre: the mean of the eight models' highs plus the city's bias. The bias is
  the mean of (observed minus model mean) over the city's last 30 settled days
  known at the decision (days up to D-1 for A, D-2 for B). It is zero with fewer
  than 7 such days.
- Spread: sigma squared = c + d x s squared, where s is the spread between the
  models. c and d come from least squares of squared de-biased errors on s
  squared, over every city's last 45 known days. Sigma is at least 1 degree F.
- The high is Normal(centre, sigma). A bracket's chance uses whole degrees: "61
  to 62" is 60.5 up to 62.5, "less than 61" is below 60.5, "greater than 68" is
  68.5 and above.
- Decisions count from the first day with at least 21 days of prior
  observations across cities for the spread fit.

**Trading rule.** The fund's event rule unchanged, except that the fee is
Kalshi's own (0.07 x price x (1 - price) per contract) instead of the flat 2%:

- buy Yes at the ask, or No at 1 minus the bid;
- only when the model's chance beats the price by at least 5 points, and by at
  least 5% per dollar after the fee;
- only at prices from 3 to 97 cents;
- only when the forecast is within 30 points of the mid.

Each trade risks one dollar and is held to settlement.

**Metrics.**

1. Brier skill against the market's mid, on every bracket with both quotes:
   1 - Brier(model) / Brier(market).
2. The rule's mean net return per dollar.

Confidence intervals: 95%, from 2,000 bootstrap draws that resample settlement
dates, so all of a day's brackets move together. Results are reported by
decision point and by city, but the verdict is on everything pooled.

**Verdict.**

- **GO:** the rule's mean net return per dollar has a 95% interval entirely
  above zero. The weather analyst then trades at the starting trust (0.25),
  under the fund's limits.
- **SHADOW:** anything else. The analyst records its predictions every cycle,
  and the test is re-run unchanged once 60 more days have settled.

No setting above may be changed after the results are seen. A variant can only
be tested as a dated amendment written before it is run.

### Result (2026-11-10): SHADOW

Run on every settled day from 2026-09-24 to 2026-11-08: 24 cities, 13,104
brackets priced, 10,894 with both quotes. The first run lost Oklahoma City to a
Kalshi rate limit (HTTP 429). Requests now retry, and the run was repeated on
the full 24 cities, with nothing else changed.

| | Trades | Mean return per dollar, after Kalshi's fee (95% CI) | Brier skill against the market (95% CI) |
|---|---|---|---|
| Pooled | 4,960 | -25.2% (-32.5% to -17.5%) | -22.1% (-25.7% to -18.7%) |
| A: 08:00 UTC, that day's markets | 2,324 | -23.8% (-32.4% to -14.8%) | -24.1% |
| B: 20:00 UTC, the next day's markets | 2,636 | -26.4% (-34.0% to -18.2%) | -20.3% |

The model's own chances were roughly right: brackets it gave 15-30% happened
20% of the time, and those it gave 30-50% happened 40% of the time. The market
was sharper. Its favourite bracket for a day was priced at a median of 53 cents
(half the days between 45 and 61). The model's favourite got a median of 40%,
because it works from runs a day or more old. When the two disagreed, the market
was usually right. Only New
York City came out ahead (+3.3% on 183 trades), well within chance.

So the weather analyst records in shadow mode. Live, it uses the newest model
runs, which the test could not use without looking ahead, so its shadow record
will show whether fresher forecasts close the gap. The test is re-run unchanged
once 60 more days have settled. The report is `reports/fund_weather_eval.json`.

## Decided-ranges evaluation (pre-registered 2026-11-10, before any result)

**Question.** A day's high can only rise. Once the airport thermometer has
passed a range, that range has lost, and an "or above" range below the reading
has won. Does buying the side that has already won, at the fund's afternoon
cycle, make money after Kalshi's fee? The risk is that settlement uses The
Weather Company's reading, not the airport's.

**Markets and prices.** The same 24 cities and days as the weather evaluation
(settled 2026-09-03 to 2026-11-09). Kalshi's hourly event candlesticks: the last
candle that closed at or before the decision, no more than 3 hours earlier, and
its yes bid and yes ask closes.

**Readings.** Every routine and special METAR at the settlement station, from the
Iowa Environmental Mesonet archive (`tmpf`, degrees F). The running high is the
largest reading since local midnight, up to the decision.

**Decision.** 20:00 UTC on day D, the fund's 01:30 IST cycle, on that day's
markets. Every hour from 18:00 to 23:00 UTC is also reported, for description
only.

**Decided ranges.** A range pays Yes on a settled high in [low, high) on half
degrees, as in the weather evaluation. To cover the gap between airport readings
and the settlement value, a range counts as decided only with a 1 degree F
margin:

- **Decided No:** the running high is at least the range's top plus 1 degree.
  This covers "or below" ranges and in-between ranges.
- **Decided Yes:** for an "or above" range, the running high is at least its
  bottom plus 1 degree.

**Trade.** Buy the decided side: No at 1 minus the yes bid, Yes at the yes ask.
Only at a price of 99 cents or less, and only when a dollar pays more than the
price plus Kalshi's fee (0.07 x price x (1 - price)). Each trade risks one dollar
and is held to settlement.

**Metrics.**

- Trades, wins and losses.
- The mean net return per dollar, with a 95% interval from 2,000 bootstrap
  draws over settlement dates.
- The worst day.
- How often the settlement contradicted a decided call.

**Verdict.**

- **GO:** at least 100 trades, and the mean net return's 95% interval is
  entirely above zero.
- **NO-GO:** anything else. This rule is mechanical, so a NO-GO drops it rather
  than shadowing it.

Kalshi's history carries no order sizes, so a GO says nothing about how much
can be bought. Live fills would be limited to the size the order book shows at
the price.

### Result (2026-11-10): NO-GO

By 20:00 UTC on the 46 settled days, the airport readings had decided 1,818
ranges across the 24 cities:

- 1,808 were already priced at a full dollar for the side that had won, which
  leaves nothing to gain.
- 8 had no quote.
- 2 were cheaper. One won, buying a decided Yes at 91 cents. The other lost: on
  2026-10-24 Denver's airport read 60 degrees F, The Weather Company settled at
  57, and the market had priced exactly that.

The settlement contradicted the airport call once in 1,818 times. The market is
faster than a twice-a-day cycle on outcomes that are already known, and a
2-trade sample fails the 100-trade minimum anyway. The rule is dropped. The
report is `reports/fund_weather_lock_eval.json`.

## Build order

1. **Ledger and orchestrator.** The current $10,000 on-device account becomes
   the fund. Analysts: the Mac model's forecasts (prediction markets), the index
   model and the pairs model. Instruments: the top 10 indices and top 10
   commodity pairs by volume, plus their Kalshi hedge legs. Grounded notes,
   attribution, and the Fund view on the Agentic tab. Twice a day, after the
   agent cycle.
2. FinBERT news sentiment: tested 2026-11-09, NO-GO (see Result above).
2a. Index options from the index model's ranges (see Options above), live
   2026-11-10. Commodity options follow a commodity range model.
3. Weather models for weather markets: tested 2026-11-10, SHADOW (see Weather
   evaluation above). Buying ranges the airport readings had already decided:
   NO-GO, because the market prices them at a full dollar (see Decided-ranges
   evaluation). Next: Polymarket's daily temperature markets (about 80 cities,
   settled on airport METAR readings), as a dated amendment.
4. A classifier trained on Foresea's graded forecasts: which gaps pay.
5. Skill-based weights once each analyst has a record.
