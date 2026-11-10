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

### Nifty option premium backtest (pre-registered 2026-11-10, before any result)

**Question.** Nifty options usually cost more than the moves that follow. On
2026-11-09 they were about 12% above the index model's range. Has selling
defined-risk structures made money after costs since India VIX began?

**Data.** Nifty 50 and India VIX daily closes from Upstox's open candles,
2009-03 to the last close. Option prices from that era cannot be had for free,
so each option is priced with Black-76 at that day's India VIX (VIX measures
30-day implied volatility from Nifty's own options). The price is taken flat
across strikes and the forward equals the index.

- Strikes sit on a 50-point grid.
- Each leg costs its price plus a half-spread of the larger of 0.5 points and 1%
  of its price, and sells for its price less the same.
- Charges are the options analyst's 2% of the money at risk.

The index model has no direction skill, so neither side gets a drift.

**Decisions.** Every 21 sessions from the first session with enough history,
with one structure held to the close 21 sessions later. Holds never overlap.
The index model (`index_forecast.fit_horizon`, 21 sessions) is refit on the
data up to each decision, and its quantiles (1% to 99%) feed
`fund_options.Distribution`. Strikes come from `fund_options.candidate_structures`,
as they do live.

- **Test A, the live rule:** the options analyst's own rule. Take the best
  structure if the model values it at least 5% above its cost after the 2%
  charges, and skip it when the model's straddle is more than 50% from the
  market's.
- **Test B, the systematic sale:** sell the iron condor every time, whatever the
  model says.

**Metrics.** For each test:

- the mean net return per rupee at risk, with a 95% interval from 2,000 bootstrap
  draws over decisions;
- the share of winning trades and the worst trade;
- the largest fall in cumulative return;
- the annual return to the fund if 2% or 10% of it is at risk per decision.

Put-side and call-side spreads are reported separately, because the flat-VIX
price undercharges puts and overcharges calls.

**Verdicts.**

- **Test A:** GO keeps the options analyst trading. NO-GO moves it to shadow mode.
- **Test B:** GO is the case for adding a systematic premium sleeve, which the
  owner decides.

Either needs at least 60 trades and a 95% interval entirely above zero.

#### Result (2026-11-10): NO-GO for both tests

Covers 2010-05 to 2026-10, with one decision every 21 sessions.

| | Trades | Mean net return per rupee at risk (95% CI) | Wins | Fund, 10% at risk per decision |
|---|---|---|---|---|
| A: live rule | 145 | +4.0% (-5.7% to +13.6%) | 62% | +2.7% a year, worst fall -48% |
| B: condor every time | 194 | +3.4% (-4.7% to +11.2%) | 61% | +2.1% a year, worst fall -56% |

Neither interval clears zero, so the options analyst moves to shadow mode. It
records what it would trade, and the cycle reads its verdict from
`reports/fund_nifty_premium_eval.json`.

The two halves of the condor tell different stories. This is description only,
and not a verdict:

- **Put credit spreads:** +6.7% per rupee at risk (95% CI +1.0% to +11.7%),
  82% wins, positive in 14 of 17 years. At 10% of the fund at risk per decision,
  +7.4% a year with a worst fall of -23%.
- **Call credit spreads:** -7.7% (-14.5% to -1.7%).

Flat-VIX pricing undercharges puts and overcharges calls, so the true put result
is probably better and the call result worse. Selling puts also collects
Nifty's long rise. A put-only sleeve needs its own test on data it has not seen.

#### Out-of-sample check: put credit spreads on the Nasdaq-100 (pre-registered 2026-11-10, before any result)

**Why.** Selling Nifty put spreads made money in the backtest above (+6.7% per
rupee at risk, 95% CI +1.0% to +11.7%). That came from a breakdown of a test that
failed, and it was seen before it was chosen. So it must hold somewhere it has
not been seen before the fund trades it.

**Data.** FRED's Nasdaq-100 index (`NASDAQ100`) and its volatility index
(`VXNCLS`, from 2001-02), daily closes to the last close. They are independent of
Nifty and India VIX.

**Rule.** The Nifty backtest's put side, unchanged except for scale.

- Every 21 sessions, refit the index model (`index_forecast.fit_horizon`, 21
  sessions) on the Nasdaq-100 with VXN in India VIX's place, on data up to that
  day.
- Take the put credit spread from `fund_options.candidate_structures`: short
  near the model's 25% level, long near its 10% level.
- Price it with Black-76 at that day's VXN, flat across strikes, with the
  forward equal to the index.
- Strikes sit on a grid of 0.2% of the index, rounded to 1, 2.5 or 5 times a
  power of ten. That matches Nifty's 50 points at 25,000.
- Each leg pays a half-spread of the larger of 0.002% of the index and 1% of its
  price.
- Charges are 2% of the money at risk.
- Hold to the close 21 sessions later.

**Verdict.** GO if the put spread's mean net return per dollar at risk has a 95%
interval (2,000 bootstrap draws over decisions) entirely above zero, over at
least 60 trades.

- **GO** is the case for a put-spread-only Nifty sleeve in the options analyst,
  which the owner decides.
- **NO-GO** leaves the Nifty put result unconfirmed, and nothing trades.

The condor and the call spread are reported alongside, for description only.

##### Result (2026-11-10): GO

Covers 2003-03 to 2026-10.

| | Trades | Mean net return per $ at risk (95% CI) | Wins | Years positive | Fund, 10% at risk per decision |
|---|---|---|---|---|---|
| Put credit spread | 283 | +10.3% (+5.8% to +14.6%) | 84% | 20 of 24 | +12.0% a year, worst fall -24% |
| Condor (description) | 283 | +7.5% (+1.0% to +13.8%) | 59% | | |
| Call credit spread (description) | 283 | -9.6% (-15.8% to -3.4%) | 69% | | |

The losing years were 2004 (-17%), 2011, 2015 and 2022 (all within -4%). 2008
(+3%) and 2020 (+12%) made money, because spreads sold when volatility was high
collected rich premiums. As on Nifty, the call side lost, so the edge is in
selling downside insurance.

Flat-volatility pricing leaves out the skew that makes real index puts dearer.
A put spread is worth more under that skew, so real sales would collect more
than this test credits. The bias runs against the result.

The put side is now confirmed on a market it was not chosen on. That is the
case for a put-spread-only Nifty sleeve in the options analyst. The owner
decides; until then the analyst stays in shadow mode. The report is
`reports/fund_ndx_put_eval.json`.

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

## Commodity correlation scan (pre-registered 2026-11-10, before any result)

**Question.** Do commodity prices, or the commodity pairs' spreads, say anything
about what Indian index futures or the pairs themselves do next, enough to trade
after costs? The pairs tracker's own backtest says its mean-reversion rule
mostly broke even, so this looks for any lead the fund could act on.

**Data.** Daily, free, 2010-01 to the last close:

- FRED: Brent, WTI and Henry Hub spot, the broad dollar index, the US 10-year
  yield, VIX, the Nasdaq-100 and the rupee.
- Upstox's open candles: Nifty 50 and Nifty Bank (open and close) and India VIX.

**Predictors.** Twelve, fixed here, each known before the next NSE session
opens (the latest US date before it, and the previous NSE session):

1-3. The one-day log returns of Brent, WTI and Henry Hub.
4-5. The Brent-WTI and WTI-gas spread z-scores, by the tracker's rule (250-day
     hedge ratio, 60-day z).
6. Brent's 5-day return.
7. The Nasdaq-100's one-day return.
8. VIX's one-day change.
9. The dollar index's one-day return.
10. The US 10-year yield's one-day change.
11. The rupee's one-day change.
12. India VIX's one-day change.

**Targets.**

- Nifty 50 and Nifty Bank from open to close on the next NSE session. That much
  is tradable: buy or sell the future at the open, close it at the close.
- For the two energy spreads, each spread's change over the next 5 sessions,
  tested against its own z-score.

That makes 26 tests.

**Method.**

- **Discovery, 2010 to 2018:** a test is a candidate when its correlation has
  |t| of at least 3.0.
- **Holdout, 2019 to the end:** a candidate is confirmed when its correlation
  has the same sign and t of at least 2.0.
- **Trading:** each confirmed test becomes a rule over the holdout. Trade the
  target in the direction the discovery correlation points, on every session
  where the predictor is at least one discovery standard deviation from its
  discovery mean. Costs are 0.05% per round trip for index futures and 0.2% for
  a spread (two legs).

**Verdict.** GO for a rule whose mean holdout net return per trade has a 95%
interval (2,000 bootstrap draws) entirely above zero. A GO is the case for a new
analyst, which the owner decides. Anything else is NO-GO. Nothing here changes
once a result is seen.

### Result (2026-11-10): commodities NO-GO; one GO from US VIX, and it is fragile

Discovery ran from 2010 to 2018 and the holdout from 2019 to 2026-11-09. Of the
26 tests, 4 were discovery candidates and 1 was confirmed.

- **Commodities and the pairs did not help.** Brent's and WTI's one-day returns
  looked like they led Bank Nifty from open to close in discovery (t -3.4 and
  -3.2), but nothing was left in the holdout (t -0.8 and 0.0). No commodity or
  spread predictor held up for either index.
- **The pairs against their own next 5 sessions** were not discovery candidates,
  so they were not traded. The Brent-WTI spread did mean-revert strongly in the
  holdout (t -9.2, inflated by overlapping 5-day windows), but in discovery it
  showed nothing (t +0.2). That is a change of regime, not a rule.
- **Confirmed: a change in US VIX predicts Bank Nifty from open to close the next
  session** (discovery t 3.0, holdout t 5.2). The rule made 393 holdout trades
  averaging +0.17% each after 0.05% costs (95% CI +0.02% to +0.33%), with 56%
  wins. That is GO.

The GO is fragile. These checks were run after the result, for description:

- 2020 alone averaged +0.54% a trade. Without 2020 the mean is +0.07% (95% CI
  -0.05% to +0.19%).
- It was positive in 6 of the 8 holdout years.
- All of the effect is on one side. Buying Bank Nifty at the open after a large
  rise in VIX earned +0.01%. Selling it at the open after a large fall in VIX
  earned +0.34% (+0.12% to +0.57%): when US fear eases sharply, India opens high
  and Bank Nifty gives some of that back by the close.

Whether to give this its own analyst is the owner's call. A shadow record of the
sell side, graded forward, would show whether the effect survives outside 2020.
The report is `reports/fund_correlation_scan.json`.

### US VIX fade: forward shadow test (pre-registered 2026-11-10, before any forward session)

**Rule, fixed from the scan.** The discovery mean of US VIX's one-day change is
+0.006 points, with a standard deviation of 1.651.

- **Short:** before each NSE session, take the latest US VIX one-day change dated
  before it. At -1.645 points or below, the record shorts Bank Nifty at the open
  and closes at the close.
- **Long:** at +1.657 or above, it records a long, so the scan's two-sided rule
  keeps a record too.
- Each trade pays 0.05%.

**Data.** FRED's VIXCLS and Upstox's open Bank Nifty daily candles, for sessions
from 2026-11-10 onward.

FRED posts each VIX close about a day late. The record is therefore written once
the data is in, a day or two after each session. It uses only values dated
before the session, so it is still a test on days the scan never saw. Live
trading would need a source that has VIX's close the same night. Cboe's site
and Stooq both refuse automated clients, and the fund does not get around that.

**Verdict.** Due once 50 short trades are recorded. GO if the short side's mean
net return has a 95% interval (2,000 bootstrap draws) entirely above zero.
Otherwise NO-GO. A GO is the case for a live analyst, which the owner decides.

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

## Live Polymarket decided-ranges test (pre-registered 2026-11-10, before any live record)

**Why.** Price history could not show whether a range the airport readings had
already decided could still be bought: most of its prices predated the deciding
reading (Polymarket weather evaluation, below). This test records in real time
what Polymarket's order book actually offered.

**When.** At every fund run (the board publish job, about every 30 minutes) from
2026-11-10. It records and does not trade.

**Markets.** Open Polymarket "Highest temperature in ..." events that settle on
the METAR rule and end within 3 days.

**Readings.** aviationweather.gov METARs for each station over the last 30 hours,
read as the market's own rules read them:

- routine reports only (a SPECI does not count);
- the temperature from the T-group in tenths when it is present, otherwise the
  whole degrees in the report's body.

The running high is the largest reading since local midnight at the station,
rounded as the market rounds.

**Decided.** As in that evaluation's Test 2. A range is decided No once the
rounded high reaches its top whole degree plus one. An "or higher" range is
decided Yes once the high reaches its bottom plus one.

**Fill.**

- Walk the asks of the decided side's token from the best price up.
- Take only levels at or below 99 cents where a dollar pays more than the price
  plus Polymarket's weather taker fee (0.05 x price x (1 - price)).
- Spend up to $100 per range.
- Each range fills once, at the first run that finds it decided with something
  to buy. Decided ranges with nothing to buy are counted.

**Grading.** At settlement (Gamma's outcome prices): net return per dollar =
(payout - cost - fee) / (cost + fee).

**Verdict.** Due once 100 fills have settled over at least 7 settlement days.

- **GO:** the mean net return per dollar has a 95% interval, from 2,000 bootstrap
  draws over settlement days, entirely above zero. The fund then buys decided
  ranges at the asks within its limits.
- **NO-GO:** anything else. The rule is dropped.

Dollars that could be filled per day are reported alongside.

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

## Polymarket weather evaluation (pre-registered 2026-11-10, before any Polymarket price was read)

**Why.** Kalshi's weather markets priced forecasts and already-decided ranges
better than the fund could (both results above). Polymarket's daily
highest-temperature markets cover about 57 cities. Most are outside the US, and
most have far less money behind them. They also settle on the airport's own
routine METAR readings. This tests the same two rules there.

**Markets.** Every settled Polymarket "Highest temperature in ..." event whose
rules settle on the hourly temperature remark of routine METAR observations, in
whole degrees after the stated rounding (nearest degree, halves up; Fahrenheit
markets convert from Celsius first). From all such events ending
2026-04-01 to 2026-11-08, a random sample of 2,000 events is drawn with seed
20261110 before any price is read. Ranges: "N°C", "N°C or below", "N°C or
higher", "N-N°F", "N°F or below" and "N°F or higher", read as half-degree bounds
in the market's unit. The outcome is each market's settled price, 1 or 0.

**Prices.** Polymarket's hourly price history (CLOB `prices-history`): the last
point at or before the decision, no more than 3 hours earlier. The history
gives one price, not a bid and ask, so a buy costs that price plus 1 cent, and a
No costs 1 minus the price plus 1 cent. The fee is Polymarket's weather taker
fee, 0.05 x price x (1 - price) per share. Half-spreads of 0.5 and 2 cents are
also reported, for description only.

**Stations and readings.** The ICAO code in each event's rules, with its
coordinates from OurAirports, and every routine METAR from the Iowa Environmental
Mesonet archive. The settlement value is rebuilt from those readings with the
market's own rounding.

**Decisions.** At the fund's two cycles, 08:00 and 20:00 UTC, while the event is
open.

**Test 1: forecast model.**

- The weather evaluation's model unchanged (bias over 30 days, spread fit over
  45, at least 21 days of history, Normal), in the market's unit. The minimum
  sigma is 0.5 degrees C or 1 degree F.
- The bias and spread come from the rebuilt settlement values, known through the
  day before the decision.
- Only decisions before 09:00 local time on the day, within 36 hours of it.
- Forecasts use Open-Meteo's `previous_dayK` runs, with K the smallest number of
  days that keeps every run at least 6 hours older than the decision, across the
  whole local day.
- Rule: the fund's event rule with the 1-cent half-spread and Polymarket's fee.
- GO if the mean net return per dollar has a 95% interval (bootstrap over
  settlement dates) entirely above zero. Otherwise SHADOW.

**Test 2: decided ranges.**

- At either cycle while the local day is under way: the running high is the
  largest routine reading since local midnight, rounded as the market rounds.
- Decided No when that rounded high is at least the range's top plus one whole
  degree. Decided Yes for an "or higher" range when it is at least the range's
  bottom plus one whole degree.
- Buy the decided side at the price plus 1 cent, if that is 99 cents or less and
  a dollar pays more than the price plus the fee.
- GO with at least 100 trades and a 95% interval of the mean net return entirely
  above zero. Otherwise NO-GO.

Nothing above may change once a Polymarket price has been read.

### Result (2026-11-10): Test 1 SHADOW; Test 2 GO on paper but invalid

Only 816 settled events used the METAR rule, all from the last 17 days (earlier
events settled on another source), so the whole set was used instead of 2,000.
48 stations. The settlement rebuilt from the archive's readings matched the
winning range for 779 of the 816 events.

- **Test 1 (forecast model): SHADOW.** 26,380 ranges were priced, giving 6,331
  trades. The mean net return per dollar was -14.0% (95% CI -21.6% to -4.6%), and
  Brier skill against the market -17.1%. As on Kalshi, the market beat the
  models.
- **Test 2 (decided ranges): GO under the written rule.** There were 165 trades
  and none lost. The mean net return was +55.6% (95% CI +43.7% to +65.9%), at a
  mean price of 76 cents.

**Test 2's verdict is not acted on, because the test was flawed.** A check made
after the run found that 130 of the 165 trades used a price point set before the
airport reading that decided the range. Price history is hourly and may be up to
3 hours old at a decision, so those prices were no longer on offer once the
outcome was known. The other 35 trades used later points, but the history holds
one quoted price, not the ask. On a thin book that number can stay unchanged
long after the book has moved.

Price history cannot settle the question. A live test can: at each cycle, take
the decided ranges from the latest METARs, the best ask and its size from
Polymarket's public order book, and record what could really have been bought,
graded at settlement. That test needs its own dated pre-registration before it
runs. The report is `reports/fund_polymarket_weather_eval.json`.

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
