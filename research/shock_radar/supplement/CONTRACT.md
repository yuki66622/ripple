# B line deterministic supplements (frozen before outcome calculations)

Both user supplements remain in scope. Deliver two concise one-page Markdown reports plus machine-readable evidence. NO model imports/calls, training, parameter tuning, new samples or modification of old evidence. Read only the 84 saved inference event JSON files, January CSV truth bound to the original manifest, and existing metadata. No other months or weight files. Existing model output, forecast IDs and four baseline paths are immutable. Stop after delivery.

## Ownership

- root: this contract, common.py (read-only frozen evidence), grouped_direction.py (supplement 1 cuts 1/3), generation/integration/tests, two report files and new output evidence.
- demo_model: downside_radius.py only; pure in-memory cut 2 and three-horizon calculations, no I/O/model; functions defined below.
- demo_data: groups.py only; pure in-memory tier/strength/systemic groups, no I/O/model.
- scenario_review: read-only independent implementation of all requested statistics from Jan CSV + frozen JSON; no source writes, no model. Review methods and compare final evidence.

Root creates common input. All branches receive rows = list of 84 dictionaries:
{event_id, timestamp, direction: +/-1|'mixed', magnitude_sigma, systemic_flag, origin_assets, assets:[10 ordered names], scored_assets:[original scoring universe], spots:{asset:float}, actual:{asset:[30 future close]}, paths:{method:{asset:[30 future close]}}, saved_score:original event score}.
METHODS = kronos_base, no_propagation, btc_beta, historical_30. Methods must all exist; reject structural invalidity explicitly, not drop rows. All risk arithmetic includes boundary close. Return pure JSON-serializable dicts. Root saves new output once.

## Common scoring

Cuts 1/2, radius and groups retain original scored_assets (systemic all10, otherwise exclude every simultaneous origin). Risk errors first average assets within an event, then equal-weight events. A group tier intersects scored_assets. Compute Top3 expected recall with original uniform cutoff-tie weights. Do not shrink k: fewer than 3 eligible assets => Top3 null with count/reason retained; risk MAE can still use nonempty groups. Empty group means null, never zero. Each table lists event count and Top3 valid count; four methods use identical event/asset sets.

Use existing pure risk_metrics and rank_scores (metrics.py) without modifying them. No inference runner/report loader that imports a model adapter. For semivariance use exact zero target (not sample mean). No future prices become model inputs; truth is only posthoc scoring.

## Supplement 1

1. Frozen trigger direction +1 = up; -1 = down; mixed/unknown gets separate explicitly reported row if present, with n=0 disclosed if absent. Reaggregate original volatility Top3 / volatility MAE / MDD MAE by event direction. Decompose total MDD advantage relative to EACH fixed baseline as n_group/84 * (baseline MAE - model MAE), so contribution is not confused with per-group average and no best-comparator switching.
2. downside_radius.downside(rows) returns per_event plus aggregate for all84, four methods. For 30 one-minute log returns r, downside semivariance = mean(min(r,0)^2) over ALL30 observations; downside volatility = sqrt(semivariance)*sqrt30 = sqrt(sum(min(r,0)^2)). Do not center, do not divide by only negative observations. Compute Top3 and MAE of this nonannualized downside volatility. Include all84, not only falling shocks; missing/empty stays visible.
3. Terminal direction uses ALL10 assets per event (840 asset-event pairs), comparing sign(C30/C0-1) without arbitrary epsilon. Binary-comparable truth nonzero only, and disclose true zero and predicted flat separately; predicted flat counts as incorrect against nonzero truth, not silently dropped. Also record exact three-sign accuracy on all840. Random 50% is analytic expectation on nonzero truth, not a simulated result. Always continue = event trigger sign for every asset; reverse = opposite. Mixed trigger has null continuation/reversal, report separately; for common comparison use only known trigger directions and truth nonzero, same denominator for all. Report model, original four frozen paths as secondary, and requested 50%/continue/reverse controls; main direction table can show only Kronos + three requested controls. Accuracy = pooled correct/eligible pairs with counts; these correlated pairs are NOT independent samples. Freeze interpretation: 45–55% is near chance and warrants “无方向预测能力（本样本未证实），不进 pitch”; outside this range is descriptive only, not proof from one month. No significance claim.

## Supplement 2

1. downside_radius.radius(rows) returns per_event plus aggregate for offsets 0/10/20, each 10 future returns. Segment boundary is each path's own C0/C10/C20, never re-anchor prediction to actual future price. MDD resets peak to that boundary; vol = population std of the 10 log returns *sqrt10. Compare only Kronos vs frozen historical_30. Retain original event scoring universe and same paired84. Report each segment MDD MAE and vol MAE in bps. Define descriptive advantage radius as contiguous prefix of segments starting at0 where mean errors are STRICTLY lower than historical_30: record separate MDD, vol, and joint radii. A missing first-segment advantage yields joint radius null (not a fabricated positive X); report “未测得共同优势半径”. Later isolated wins do not extend the prefix; this is an exploratory definition, not guaranteed future warning capability.
2. groups.grouped(rows) returns per_event and summaries for each axis independently (no cross-product): tier big={BTC,ETH,SOL,BNB}, small={XRP,DOGE,ADA,AVAX,LINK,LTC}; strength [6,8) and [8,infinity); systemic true/false (label false described as non-systemic, may include tied origins, not fabricated unique source). Each group computes MDD MAE and volatility Top3 expected recall for all four methods. Tier intersects scored_assets; retain null Top3 when fewer than3. Summaries include MAE event_count, Top3 defined/null n and mean chance3/N among valid events. Label overlapping grouping axes; do not combine them as extra independent samples.

## Acceptance

All84 saved input file hashes before/after unchanged. Bound January CSV SHA and each forecast/window identity to original evidence; reconstruct actual close paths, original full30 risk metrics and stored scores for consistency. Pure tests include negative-only vs full-denominator downside; prefix segment indexing/anchor; tier <3; zero/mixed direction. Independent reviewer recomputes aggregate results with separate formulas. No fallback or result-based filtering. Save result JSON and two brief Markdown pages; don't expand scope or start new model work.
