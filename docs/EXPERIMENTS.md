# What we tested, what happened, and what shipped

Ripple is an attention assistant. These experiments informed which methods it uses; they do **not** establish that users make better decisions or that forecasts produce trading profits. No user study, trading backtest with execution costs, or prospective deployment study has established those outcomes.

This document summarizes saved reports. Preparing this release did not retrain a model, regenerate forecasts, or inspect untouched holdout data. Historical exploratory results, observational follow-ups, and sealed fine-tuning evaluations are identified separately below. Failure and `null` results are part of the record.

## Reading the numbers

- **B line:** original Kronos-base, ten assets, 256 one-minute input bars and a 30-minute forecast. Except for three explicit multipath replay cases, each asset has **one saved sampled path**. Subsequent analyses reuse it.
- **Major group:** BTC, ETH, SOL, BNB. **Smaller group:** XRP, DOGE, ADA, AVAX, LINK, LTC. These are project group labels, not a general classification of market capitalization.
- **Drawdown:** maximum decline from the running peak of the close-price path, including the price at the forecast origin.
- **Volatility:** population standard deviation of minute log returns, multiplied by the square root of the number of forecast minutes; not annualized.
- **MAE:** mean absolute error. **1 bp = 0.01 percentage points.** Smaller error is better.
- **Top3 expected recall:** fraction of three risk-ranking slots recovered, with fractional credit for ties. It is not an exact-three-assets hit rate.
- B-line metrics average assets within an event, then events equally. Systemic events score all ten assets; non-systemic events exclude all tied origin assets. Exceptions are stated explicitly.
- Asset-event pairs, paths, overlapping forecast windows, and repeated events within a day are not independent trials. A `null` correlation or metric is not replaced with zero.

## 1. Shock detection and the four-method comparison

**Question.** After a large observed move, can the model identify the assets with the most subsequent volatility or drawdown more accurately than simple historical rules?

**Method.** A candidate occurs when the absolute five-minute log return exceeds `k` times the population standard deviation of the preceding 60 overlapping five-minute returns. That baseline excludes the current five-minute return. A zero baseline is unavailable, not infinite. The implemented frozen rule keeps the first trigger and suppresses further triggers for that asset in `[t, t+30 minutes)`: it does not look ahead to choose the largest later trigger.

The earliest unassigned trigger anchors an event; triggers through `t0+10` are merged without extending the merge window. Tied earliest assets remain tied. A systemic label requires at least three assets to trigger within `t0 ± 10 minutes`; this label is only knowable afterward and is never a contemporaneous model input.

**Sample.** January 2026 was used for calibration and exploration. The grid `k = 3, 4, 5, 6` produced 1,053 / 435 / 179 / 84 eligible events. None met the original 20–60-event target. The project owner explicitly accepted **all 84 events at k=6**, rather than selecting successful cases. There were 20 systemic and 64 non-systemic events. January contained three pairs of overlapping 30-minute outcome windows.

Four methods used the same available history:

| Method | What it predicts |
| --- | --- |
| Original Kronos | One cached 30-minute generated path per asset |
| No propagation | Origin assets replay their previous 30 minute returns; other prices remain constant |
| BTC beta | Estimate each asset's beta from the preceding 255 returns, then replay the last 30 historical BTC returns multiplied by that beta |
| Historical 30 | Each asset replays its own last 30 minute returns from its current price |

Neither baseline uses future BTC returns or future labels. The BTC-beta baseline is unavailable if historical BTC variance is zero.

**January results: 84/84 events completed for every method.**

| Method | Volatility Top3 recall ↑ | Volatility MAE, bp ↓ | Drawdown MAE, bp ↓ | Drawdown Top3 recall ↑ |
| --- | ---: | ---: | ---: | ---: |
| Original Kronos | 61.51% | 29.926 | **37.639** | 49.21% |
| No propagation | 36.37% | 53.591 | 62.970 | 34.25% |
| BTC beta | 63.89% | 27.894 | 42.615 | **54.37%** |
| Historical 30 | **73.41%** | **20.351** | 46.116 | 49.21% |

**Limits and product decision.** January is an exploratory sample, not a sealed model test. Lower drawdown magnitude error does not establish superior drawdown ranking. Ripple uses observed historical volatility to prioritize attention and shows the original model's drawdown estimate separately. It does not use the model to decide the volatility Top3.

Sources: [frozen definition](../research/shock_radar/DEFINITION.md), [January report](../research/shock_radar/REPORT.md).

## 2. Does the result survive a second month?

**Method and sample.** Apply the same `k=6` detector and original model to February 2026. All **56 eligible events** received complete paired predictions. February is an **observational follow-up**, not an A-line selection input or a new sealed acceptance gate.

| Method | February drawdown MAE, bp ↓ | February volatility Top3 recall ↑ |
| --- | ---: | ---: |
| Original Kronos | **49.494** | 53.57% |
| No propagation | 74.470 | 32.75% |
| BTC beta | 57.100 | 55.36% |
| Historical 30 | 59.333 | **64.29%** |

**Result and limit.** The drawdown magnitude advantage over Historical 30 appeared in both months. Its source was not stable: down-shock events contributed about **95%** of January's aggregate improvement but only **52.43%** of February's. This is a decomposition of error differences, not directional accuracy or a prospective success rate. Two months and one sampled path per asset do not establish reliable future superiority.

Source: [February report](../research/shock_radar/closure/artifacts/february-v1/FEBRUARY.md).

## 3. Direction, downside volatility, and forecast horizon

All analyses in this section reuse the **84 frozen January forecasts**, without new inference.

### Up shocks versus down shocks

**Question.** Is the overall result driven by one trigger direction? Each cell reports volatility Top3 recall / volatility MAE / drawdown MAE; errors are bp.

| Method | Up shocks, n=41 | Down shocks, n=43 |
| --- | ---: | ---: |
| Original Kronos | 64.23% / 23.70 / 24.97 | 58.91% / 35.86 / 49.72 |
| No propagation | 34.95% / 49.25 / 47.40 | 37.73% / 57.73 / 77.82 |
| BTC beta | 68.29% / 22.39 / 31.20 | 59.69% / 33.14 / 53.50 |
| Historical 30 | 78.05% / 15.21 / 25.84 | 68.99% / 25.25 / 65.45 |

Mixed/unknown direction: n=0, metrics `null`. The model's January drawdown advantage over Historical 30 came mainly from down shocks; versus BTC beta, about 61% of improvement came from up shocks. The explanation depends on the comparator.

### Downside volatility

**Method.** Use zero-centered downside semivariance: `sum(min(r, 0)^2) / 30`, then take its square root and multiply by `sqrt(30)`. Do not divide only by the number of negative returns or subtract their mean.

| Method | Downside Top3 recall ↑ | Downside MAE, bp ↓ |
| --- | ---: | ---: |
| Original Kronos | 62.70% | 24.05 |
| No propagation | 35.23% | 43.72 |
| BTC beta | 59.92% | 25.76 |
| Historical 30 | **63.10%** | **23.57** |

All methods have n=84, with no nulls. **Decision:** no claim of model superiority at identifying downside-volatility leaders.

### Terminal return direction

**Method and sample.** Compare the sign of predicted and realized 30-minute terminal return for all ten assets, including origins. Of 840 asset-event pairs, 15 had exactly zero realized return; the common binary denominator is 825.

| Method | Correct / nonzero outcomes | Accuracy |
| --- | ---: | ---: |
| Original Kronos | 422/825 | 51.15% |
| Random sign | Theoretical expectation, not a sampled run | 50.00% |
| Always continue the shock direction | 388/825 | 47.03% |
| Always reverse the shock direction | 437/825 | 52.97% |

**Decision:** no demonstrated direction-prediction ability; no up/down recommendation feature. The pairs are correlated, not 825 independent tests.

### Effective forecast radius

**Method.** Divide each saved path into three ten-minute segments. Each segment resets its starting price and drawdown peak; volatility uses its ten returns. Historical 30's saved replay path is segmented identically.

| Segment | Drawdown MAE: Kronos / Historical 30, bp | Volatility MAE: Kronos / Historical 30, bp |
| --- | ---: | ---: |
| 0–10 minutes | 24.62 / 24.16 | 21.98 / 27.56 |
| 10–20 minutes | 18.19 / 17.30 | 15.07 / 12.12 |
| 20–30 minutes | 17.34 / 39.71 | 14.98 / 21.73 |

Each segment has n=84. A continuous advantage from minute zero lasts **10 minutes for volatility error only** under this comparator. Drawdown radius = `null`; joint drawdown-and-volatility radius = `null`. The good last segment cannot retroactively establish 30 continuous minutes of superiority. **Decision:** no dependable drawdown warning radius is advertised.

Sources: [direction/downside report](../research/shock_radar/supplement/SUPPLEMENT_1.md), [radius/group report](../research/shock_radar/supplement/SUPPLEMENT_2.md).

## 4. Tier, shock strength, and systemic grouping

**Method.** Keep all January events and the original scoring convention. Each method cell is drawdown MAE in bp / volatility Top3 recall.

| Group | n: MAE / Top3 | Kronos | No propagation | BTC beta | Historical 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Major | 84 / 82 | 30.49 / 91.87% | 50.25 / 86.75% | 35.00 / 92.68% | 36.24 / 93.09% |
| Smaller | 84 / 84 | 42.17 / 69.84% | 71.31 / 55.58% | 47.85 / 72.22% | 52.95 / 77.78% |
| 6–8 sigma, excluding 8 | 60 / 60 | 32.55 / 60.56% | 50.01 / 34.53% | 31.94 / 64.44% | 32.89 / 73.89% |
| At least 8 sigma | 24 / 24 | 50.35 / 63.89% | 95.38 / 40.97% | 69.31 / 62.50% | 79.18 / 72.22% |
| Systemic | 20 / 20 | 69.83 / 43.33% | 116.36 / 44.63% | 85.54 / 58.33% | 93.70 / 60.00% |
| Non-systemic | 64 / 64 | 27.58 / 67.19% | 46.29 / 33.79% | 29.20 / 65.62% | 31.25 / 77.60% |

**Limits.** Two major-group events have fewer than three eligible assets after origin exclusion, so Top3 is `null` while their MAE remains. Selecting three of four assets is much easier than selecting three of six; tier recalls cannot be compared as equivalent tasks. These grouping dimensions overlap. Systemic labels are retrospective.

**Decision.** None of these groups justifies replacing historical volatility ranking with the model. Source: [complete grouping report](../research/shock_radar/supplement/SUPPLEMENT_2.md).

## 5. Temporal graph, fingerprints, and uncertainty

### Temporal graph and its randomized comparison

**Question and method.** Does cross-asset trigger timing contain more clustering than a specified random timing model? January's graph records origin A followed by asset B's first retained trigger within `(0, 30]` minutes. It contains 100 ordered asset pairs and **68 nonzero edges**, totaling **120 edge observations**. Simultaneous origins do not produce zero-delay edges.

For 200 randomized trials, independently circular-shift each asset's trigger sequence, then rebuild events and edges using the same rules. Compare edge count and mean delay using one-sided tests and Holm correction.

| Statistic | Observed | Null-distribution median | Adjusted p |
| --- | ---: | ---: | ---: |
| Nonzero edge count | 68 | 23 | 0.00995 |
| Observation-weighted mean delay | 3.01 min | 15.63 min | 0.00995 |

**Limit and decision.** Timing alignment exceeds this null model, but this does not establish causal contagion, stable individual edges, or predictive value. Randomized trials contain 190–212 eligible events versus 84 observed; the null does not preserve common market timing. The visualization is a historical timing map.

The broadest observed sequences were ETH followed by nine other assets within three minutes on January 13; DOGE followed by seven within two minutes on January 30; and BTC/SOL followed by seven within four minutes on January 6. These are observations, not forecasts.

### Fingerprints and the singleton cluster

**Method.** Encode ten assets' observed direction, trigger strength, and delay; apply the frozen distance and average linkage with four clusters. Historical lookup compares equal observed prefixes and excludes the query event itself.

**Result.** Cluster sizes are **70 / 5 / 8 / 1**; overall silhouette is **0.4807**. Under leave-one-event-out reclustering, the singleton remains isolated in **82/83** trials where it is present. Membership is unchanged in 76/84 removals; adjusted Rand index has median 1.00 and minimum 0.7604.

**Limit and decision.** A stable singleton remains one observation, not a repeatable event category. Complete fingerprints use post-event information. Full-month neighbor search is exploratory retrieval, not an as-of historical backtest or a probability forecast. The generic nearest-event interface was explored but was not retained as a core attention feature.

### Multipath interval coverage

**Method and sample.** Three replay events were chosen before inspecting prediction success: strongest systemic, strongest non-systemic, and a median-strength case. These alone received ten generated paths per asset. Calculate drawdown separately on each path before taking p05/p50/p95.

**Result.** Actual drawdown falls inside `[p05, p95]` for **10/30 = 33.3%** of asset-event pairs: major 2/12, smaller 8/18. Fourteen actual outcomes exceed the upper bound and six are below the lower bound. Event-level coverage is 2/10, 4/10, and 4/10.

**Limit and decision.** The sample is three selected events, not thirty independent events, and each interval uses only ten paths. The bands are overconfident on this sample and are displayed as **uncalibrated sampled intervals**, never as validated 90% confidence intervals.

Sources: [graph and fingerprints](../research/shock_radar/closure/FINGERPRINTS.md), [four diagnostic checks](../research/shock_radar/deep_checks/REPORT.md).

## 6. Drawdown failures and threshold calibration

**Failure question.** What do the worst errors look like? Across all 140 January/February events, the ten largest event-level drawdown errors average **205.62 bp**. Nine are down shocks and eight are systemic. Every one of the **98 scored asset-event pairs** in those ten events underestimates drawdown.

The full sample is 47.9% down shocks and 26.4% systemic, with average model error 42.38 bp. The worst asset comes from the smaller tier in 70% of worst events versus 70.7% overall: no additional concentration in smaller assets was demonstrated by that statistic.

**Limit and decision.** This is an outcome-selected failure description. It does not yield a validated rule for deciding beforehand when to distrust the model.

**Threshold question and method.** On 84 January events × all ten assets, compare predicted and actual future drawdown against the same strict threshold, before rounding.

| Threshold | TP / FP / FN / TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 1% | 13 / 36 / 109 / 682 | 26.53% | 10.66% | 15.20% |
| 2% | 0 / 6 / 27 / 807 | 0.00% | 0.00% | 0.00% |
| 3% | 0 / 1 / 9 / 830 | 0.00% | 0.00% | 0.00% |
| 5% | 0 / 0 / 0 / 840 | null | null | null |

An exploratory recommendation rule first required at least ten actual and ten predicted positives, then selected the highest F1. It selected **1%**, but performance was weak. The existing **3% rule was not changed** by this analysis. This is neither independent threshold validation nor probability calibration; the 840 pairs are correlated.

Sources: [failure report](../research/shock_radar/deep_checks/REPORT.md), [threshold report](../research/shock_radar/closure/THRESHOLDS.md).

## 7. What does “check three first” miss?

**Method.** At each frozen event, select exactly three of ten assets using observed past-30-minute volatility, including origin assets. For volatility and drawdown separately, count an omission when one of the other seven assets' realized next-30-minute values strictly exceeds the weakest selected asset on that metric. Ties are not omissions. No new forecast is generated.

**Sample and result.** All 140 events and all ten assets are retained.

| Event-level omission | January | February | Combined |
| --- | ---: | ---: | ---: |
| Volatility | 67/84 | 45/56 | 112/140 = 80.00% |
| Drawdown | 75/84 | 50/56 | 125/140 = 89.29% |
| Either measure | 79/84 | 52/56 | 131/140 = 93.57% |

There are 282 volatility and 456 drawdown omitted asset-event records. Conditional on being omitted, relative excess `(omitted − weakest selected) / weakest selected` is:

| Measure | Median | p75 | p90 |
| --- | ---: | ---: | ---: |
| Volatility | 14.01% | 30.35% | 48.07% |
| Drawdown | 32.93% | 65.18% | 105.39% |

The corresponding median absolute excess is 9.70 bp for volatility and 13.89 bp for drawdown. A 105.39% relative excess means about 2.05 times the comparator's drawdown, **not a 105% investment loss**. Small denominators can inflate relative differences.

**Limit and decision.** “At least one omission” is a broad event-level measure, not the proportion of assets missed or the rate of serious losses. The amplitude distribution describes already-omitted records, not all cases. Nevertheless, omissions are not uniformly marginal: all ten assets stay accessible, and prioritization never means that the other seven are safe to ignore. Ranking and drawdown remain separate views. These tests do not establish the net human benefit of prioritization.

Sources: [omission rate](../research/shock_radar/omission/REPORT.md), [amplitude distribution](../research/shock_radar/omission/relative-amplitude/REPORT.md).

## 8. Recovery time and additional descriptive checks

### Historical recovery reference

**Question and method.** How does volatility subside after these historical shocks? Divide each asset's ten-minute rolling volatility by a pre-shock 120-return baseline ending at `t0−5`. Take the tier median within each event, then the median curve across events. Recovery is confirmed after ten consecutive minute points at or below 1.2× baseline. Later shocks remain in the observations.

| Sample | Valid events | Major median-curve confirmation | Smaller median-curve confirmation |
| --- | ---: | ---: | ---: |
| January | 83/84 | 46 min | 37 min |
| February | 56/56 | 30 min | 25 min |
| Combined | 139/140 | **46 min** | **25 min** |

The median of individual-event recovery times is a **different statistic**: 36 minutes for major and 30 for smaller assets in the combined sample. Nineteen major-group and eleven smaller-group events had no confirmed recovery within 120 minutes; censored observations were not replaced by 120.

**Product decision.** The fixed 46/25-minute values are a historical reference, not a countdown or prediction for the selected asset. The current-volatility indicator is a separate computation: population standard deviation of the latest ten one-minute returns divided by that of the latest 120 returns, sharing the same endpoint and including the ten recent returns in the baseline. At least 1.2× is “above usual”; invalid or zero-baseline inputs are unavailable. This live indicator is not a trained recovery-time estimator.

### Other checks from the same 140 frozen events

| Question | Result | Interpretation and decision |
| --- | --- | --- |
| Are volatility and drawdown leaders identical? | Smaller-group realized Top3 sets exactly match in 46/140 events | Related measures are not interchangeable; keep separate panels. Selecting three of four major assets mechanically creates high overlap. |
| Does the model specialize in path shapes? | 9 V reversals, 9 upward false breakouts, 0 qualifying one-way declines; 122 “other” | Stable shape-specialist ability = **null**. Labels use 60-minute future truth and are unavailable at prediction time. |
| Does the graph establish useful lead time? | 120 edge observations: mean 3.01 min, p90 4 min, p99 25 min | The graph admits only `(0,30]` delays by construction. It cannot establish the omitted tail or net lead time after detection/inference. |
| Should trust vary by UTC time block? | All 12 prespecified adjusted comparison intervals include zero | Time-of-day trust rule = **null**; no product adaptation. |
| Is there a reliable prospective failure profile? | All 10 prespecified adjusted feature contrasts include zero; each monthly tail has fewer than ten events | Reliable “do not trust now” profile = **null**. The largest-error description is not a validated predictor. |
| Is volume elevated at detection? | 97/140 events meet matched-control requirements; combined ratios 1.34×/1.62× for major up/down and 1.29×/1.60× for smaller up/down | Candidate feature only. The observed window includes the triggering move, so this is not evidence of a pre-shock leading signal. |

Volume-control exclusions were explicit: 43 January events lacked at least three eligible historical controls. Time-block, failure-feature and volume comparisons used 20,000 month-stratified UTC-day block bootstrap draws and Bonferroni intervals across 26 planned contrasts. Other dependence and out-of-month generalization remain untested.

Source: [seven supplementary analyses](../research/shock_radar/seven_checks/REPORT.md).

## 9. LoRA A: can fine-tuning improve volatility ranking?

**Data split and method.** Train on January 1–25: 298 origins × ten assets = **2,980 samples**. January 26–31 supplies 48 observation-only origins. February's 300 fixed origins select a checkpoint; March's 300 fixed origins form the sealed final test. The dynamic comparator uses the 255 returns from a 256-bar input, not the B-line 30-minute ranking baseline.

Kronos-base uses a frozen tokenizer and the original two-level next-token objective. LoRA rank 4, alpha 8, dropout 0.1 targets attention projections; learning rate is 4e-5, batch size 8, with at most two epochs. Normalization statistics use only the 256 input bars. The selected candidate is epoch 2. Original and candidate share inputs and path-derived seeds; failed outputs are not resampled until successful.

**Acceptance gates.** Smaller-group exact Top2-set accuracy must strictly beat the original model, dynamic baseline, and January-frozen AVAX+LINK pair. Major-group accuracy must be no worse than dynamic baseline minus two percentage points. Confidence intervals are reported but do not replace these frozen gates.

| March exact Top2-set accuracy | LoRA | Original | Dynamic baseline | Fixed AVAX+LINK |
| --- | ---: | ---: | ---: | ---: |
| Smaller, 300/300 paired origins | 42.00% | 37.33% | 57.67% | 61.33% |
| Major, 300/300 paired origins | 46.00% | 53.00% | 86.33% | Not applicable |

**Result: failed.** Smaller-group improvement over the original is +4.67 percentage points, with a descriptive day-block 95% interval of −0.35 to +9.57 pp. Major-group change is −7.00 pp. Both groups lose to the dynamic baseline; the smaller group also loses to the fixed pair. Observational drawdown MAE worsens by about 1.37% for smaller and 1.36% for major assets, with intervals on the error differences crossing zero.

**Post-hoc behavior probe.** All ten assets' mean volatility forecasts moved upward: smaller group +5.06%, major group +4.66%. This is not evidence that the model learned a new financial mechanism. Some smaller-group improvements corrected selections toward a frequent AVAX+LINK combination, while major-group changes often broke correct ETH+SOL selections. A special high-volatility advantage was not established. The probe reuses the test predictions and is descriptive, not another acceptance test.

**Product decision.** This adapter did not ship. Exact Top2-set accuracy within four/six assets is not comparable to B-line Top3 expected recall across its eligible assets.

Sources: [A protocol](../lora_a/PROTOCOL.md), [March sealed report](../lora_a/runs/20260927-v21-recovery01/march/sealed-report.md), [final conclusion](../lora_a/runs/20260927-v21-recovery01/march/conclusion.md).

## 10. LoRA C: can selecting for drawdown help?

**Data split and method.** Fresh initialization, same January 1–25 training set and training objective. April's 300 fixed origins select by smaller-group drawdown MAE; May's 300 fixed origins are a separate sealed acceptance set. This line does not use A's February/March data or checkpoints. June–August remain reserved under its protocol. Changing the checkpoint-selection metric does not turn the training loss into a direct drawdown loss.

**Gates.** April's best smaller-group drawdown MAE must improve on the original, with a major-group degradation guard during selection. The selected epoch 1 must then strictly beat both the original and Historical 30 on smaller-group May drawdown MAE. Final major-group results are reported without adding a new post-hoc gate.

| May drawdown MAE, bp ↓ | LoRA epoch 1 | Original | Historical 30 |
| --- | ---: | ---: | ---: |
| Smaller, 300/300 paired origins | 19.600 | **19.189** | 20.361 |
| Major, 300/300 paired origins | **14.898** | 15.206 | 15.654 |

**Result: failed.** The smaller-group candidate beats Historical 30 but is worse than the original. Its error difference against the original is +0.411 bp, with a descriptive 95% day-block interval of approximately −0.133 to +0.973 bp. The major-group difference is −0.309 bp, with an interval of approximately −0.696 to +0.119 bp. Neither establishes a reliable improvement. The dynamic volatility-only comparator has drawdown MAE `null`; it is not assigned an invented drawdown forecast.

**Product decision.** This adapter did not ship. Both LoRA lines preserve the negative result instead of presenting checkpoint availability as evidence of improvement.

Sources: [C protocol](../lora_c/PROTOCOL.md), [May sealed report](../lora_c/runs/20260927-c01/may/sealed-report.md).

## 11. Output quality and remaining claims

The January original-model batch generated 25,200 forecast bars. **2,270** needed mechanical OHLC containment repair and **716** had finite negative volume/amount fields. High and low are expanded to contain original open/close; open and close are never changed. Original values and correction flags remain recorded. Negative unused volume fields are marked unavailable rather than modifying price metrics. Structural price failures remain failures. These quality counts do not establish predictive accuracy.

The shipped choices are therefore narrow: observed volatility for attention ranking; original-model price paths for a separately identified drawdown estimate; historical replay and temporal association for explanation; and explicit limitations on direction, uncertainty, and recovery references. Whether that combination improves a person's monitoring workflow remains an open evaluation question.
