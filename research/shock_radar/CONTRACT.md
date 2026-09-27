# B line: shock radar v1 (approved amendments, implementation contract)

Newer user authorization: closure/CONTRACT.md governs the four-part drawdown closure. Only THREE existing demo events may get a new ten-path batch; February gets a NEW observational batch. The original January84 predictions remain immutable; graph/fingerprint/threshold work is deterministic. March remains forbidden; A-line and live rules unchanged.

Latest scope: the original inference batch is complete. The user's two new supplements authorize deterministic calculations on those frozen84 predictions ONLY; no new model execution. See supplement/CONTRACT.md. The historical inference permission below is not permission to rerun for supplements.

## Explicit count-range amendment (after original calibration)

User accepted “用 k=6 的全部 84 次，接受样本数超出原范围”. The original four-k calibration and its selected_k=null remain immutable evidence. Proceed with ALL 84 eligible k=6 January events via a separately recorded user-authorized selection, despite exceeding 20–60. No cap, no new grid, no outcome-based filtering. This amendment supersedes only the count-range stop gate; all causal/data/model boundaries remain unchanged.

User accepted all four review amendments. Only research/shock_radar/ and project scope notes may change; data/model/metric/demo modules and LoRA work are read-only. January 2026 only for now; never open sealed March or retired 2025 data. February remains unused unless January is insufficient and a defined next stage needs it. No training, cloud spend or downloads. Original local Kronos-base/tokenizer immutable; root alone may run model inference after event calibration succeeds. Output artifacts are exclusive/new, never replace prior evidence.

## Ownership and interfaces

- Root: this contract, io.py (January hash-checked data access), baselines.py, run.py, reports/replays/integration and real execution.
- demo_data: detection.py, test_detection.py handed off/frozen. It now owns report.py only, building the human-readable report and preselected replay JSON from root's saved artifacts after inference. No model, no new event selection, no source modifications outside report.py. Root executes it and validates output.
- demo_model: metrics.py, test_metrics.py only. Pure risk/ranking/event aggregation, no model or real data access.
- scenario_review: read-only review and independent checks, no source writes, no data beyond January, no model.

Dataset interface: io.load_january() returns {assets: ordered BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC, times: N UTC candle-end strings, closes: NumPy (N,10), histories: {asset: N validated OHLCVA rows}, provenance: hash metadata}. io.market_window(data, index) contains ONLY histories[index-255:index+1], contract v1 and content-derived window_id; no event/systemic labels or future truth. Root creates these interfaces first. Agents must not edit them.

## Detector frozen definitions BEFORE counting

At completed minute t: r5[t]=log(C[t]/C[t-5]); sigma[t]=population std of r5[t-60:t] (60 overlapping five-minute returns, excludes r5[t]). First computable t=65. Zero sigma is undefined and counted, no epsilon/infinite significance. Threshold is strict abs(r5)>k*sigma, k in {3,4,5,6} only.

Per asset keep first threshold crossing; subsequent candidates are suppressed until t >= last_kept+30 (cooldown [t,t+30)). Keep chronological order and record suppressed counts. No retrospective maximum selection or resampling. Run state from t=65 to month end; do not restart at an evaluation origin.

Merge retained triggers across assets chronologically into fixed non-chaining episodes: first unassigned trigger anchors t0; consume all remaining triggers through t0+10 inclusive. Start next episode after that boundary. t0 and earliest origin asset(s) never change. Simultaneous earliest origins all retained; origin_asset is a string only if unique, else null with origin_assets array (CSV joins tied names). direction is +1/-1 if tied origins agree, else 'mixed'; magnitude_sigma is max z among tied origin triggers at t0, not a future maximum.

systemic_flag uses >=3 DISTINCT assets with retained triggers in [t0-10,t0+10] inclusive across all triggers; posthoc classification, confirmed at t0+10, never used to trigger/select t0 or as model input. Raw candidates suppressed by cooldown are not additional triggers. Record systemic_asset_count and confirmed time; incomplete end-of-month confirmation flagged.

Eligible event has t0>=255 and t0+30<N, so 256 input closes and 30 future returns fully exist. This timestamp-only availability rule is applied before k choice; excluded edge events and all candidate/retained/merged counts stay visible. No outcome-quality filtering. Calibrate count = eligible merged episodes. Choose feasible k with 20<=n<=60 closest to 40, tie larger k. If none qualifies: selected_k=null; write all four counts/catalog diagnostics, report mismatch and do NOT automatically expand k/grid, cap/subsample events or run model. This is a calibration gate, not task success.

Detection interface: detect_events(closes,times,assets,k) returns {k,events:[all event objects],triggers:[kept trigger objects],stats:{raw_candidate_count,retained_trigger_count,suppressed_candidate_count,zero_sigma_count,merged_event_count,eligible_event_count,excluded_event_count}}. Every trigger has index,timestamp,asset,direction,magnitude_sigma. Every event has event_id,index,timestamp,origin_asset,origin_assets,direction,magnitude_sigma,systemic_flag,systemic_asset_count,systemic_confirmed_at,eligible,exclusion_reasons,member_triggers. calibrate(closes,times,assets) returns {selected_k,grid:[stats plus k],catalogs:{str(k):detect_result},selection_rule}.

## Prediction and deterministic comparators

For every eligible event at selected k: original Kronos, lookback256/horizon30, path_count=1, seed20260926, existing T/top_p/output corrections and volume audits unchanged. Retain raw output/errors, no repeated sampling to rescue failures. Cached immutable forecast_id. This is a single-sample exploratory benchmark; no calibrated uncertainty claim. Each asset gets only its own past OHLCVA, so product claim is post-shock risk ranking, not causal transmission.

Ten-asset metrics are future close-path max_drawdown (includes current spot) and volatility (population std of 30 log returns *sqrt30), separate rankings and MAE. Compute on each model path then scalar mean if expanded in later version; current path count is one.

Baselines, all use only <=t0:
1. no_propagation: replay source(s)' own last30 one-minute returns from their current spot; all other close paths flat. 'No propagation' means that explicit path convention, not an unobserved absence of abnormality.
2. btc_beta: beta_i = demeaned covariance of each asset/BTC over 255 past one-minute returns divided by BTC population variance; no intercept. Replay last30 BTC one-minute returns, multiplied by beta_i, from each asset current spot. If BTC variance=0: explicit unavailable, no silent substitute. No future BTC truth or extra model.
3. historical_30: replay each asset's own last30 returns from its current spot; equals persistence of its trailing30 MDD/vol. Strong control retained alongside user's two baselines.

Metric interface: risk_metrics(spot, closes)->{max_drawdown,volatility}; rank_scores(predicted:list,actual:list,k=3)->{spearman:float|null,top3_recall_expected:float,spearman_reason:str|null}; score_event(event,predictions,actual,assets)->serializable event result. predictions={method:{asset:{max_drawdown,volatility}}}, actual={asset:{...}}. Event scoring universe: if systemic all10; else exclude all simultaneous origin_assets (usually one). Compute per-event risk MAE then equal-weight event aggregate, units decimal plus bps for reports. Spearman average ties; undefined constant vectors ->null with reason and coverage, not 0. top3 expected recall handles cutoff ties uniformly: select top3 inclusion weights (1 above cutoff, remaining_slots/tied_count at cutoff), dot(pred_weights,truth_weights)/3. All-flat prediction gives chance 3/N, labeled tie/chance behavior, never perfect score. No lexicographic tie-break or include-all-ties inflated hits.

Reporting compares four methods on identical complete paired events plus per-method failures and all-catalog denominator, separated systemic/non_systemic/all; n means events (overlapping horizons remain dependent), never n*10. No claim of real-time causal detection from posthoc labels. n too small stays reported rather than hidden.

## Replays / completion

Preselect up to3 UNIQUE IDs using ONLY catalog trigger metadata: largest magnitude systemic, largest magnitude non-systemic, and the remaining event closest to the median initial magnitude of ALL eligible events (all84), not a recomputed median of the remaining82; ties earliest timestamp. This median-reference clarification is recorded before inference in replay-selection-policy.json; the previously saved three selections are unchanged. Missing categories disclosed. Selection occurs before inference; failed chosen sample yields explicit failed replay, not replaced with a successful prediction. Include all10 price closes for t0-30..t0+30 with observed/forecast split, all four predicted risks, real risks, forecast_id, audit flags and source/provenance. No selecting successes or deleting failures.

Acceptance: causality/prefix and cooldown/merge/tie tests; metrics vs existing engine; no future access in baselines; independent Jan counts; zero March reads; actual original-model outputs when gate passes; truthful failure/paired coverage; traceable data/model/code hashes. Once authorized deliverables are complete, stop.
