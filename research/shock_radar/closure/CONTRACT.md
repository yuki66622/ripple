# B-line drawdown closure: four deliverables

User explicitly authorizes: January-only frozen graph/fingerprints/threshold calibration; NEW February k6 same-definition detection and ORIGINAL Kronos evaluation, observational B only; NEW 10 paths per asset for exactly the3 already-selected January demo events. No existing predictions overwritten/retried, no March access, no A-line changes, no live alarm/demo changes. User time estimates are not measured guarantees. Local only, no paid resources.

## Parallel ownership

- root: contract, multipath.py and running both GPU workloads, integration/validation/summary docs.
- demo_data: graph_fingerprints.py, test_graph_fingerprints.py, FINGERPRINTS.md; may run January deterministic graph/fingerprint generation to new artifacts/january-graph-v1. No model/FEB/A writes.
- demo_model: february.py and FEBRUARY.md renderer plus tests; owns Feb fixed-reader/detect/eval code only. May run February detection-only preparation; ROOT ALONE runs new model execution after review. No January re-inference, no March/A reads/writes.
- scenario_review: thresholds.py, test_thresholds.py, THRESHOLDS.md; may run deterministic January threshold calibration to new artifacts/thresholds-v1, then independent read-only review of graph/Feb/multipath outputs. No model.

All files above under research/shock_radar/closure/. Existing research/data-probe/binance-2026-01/02 read-only. Original models scenario-lab/models/Kronos-base and Kronos-Tokenizer-base read-only. Old shock_radar sources/artifacts read-only. Root records frozen Jan hashes before/after. Communicate schema before root integration. Root runs code/tests and independently checks real results, not just agent assertions.

## 1. Directed temporal graph and fingerprints (highest priority)

Use only selected84 January event metadata + all retained k6 triggers from frozen catalog-k6.json (not suppressed raw candidates). Each event earliest origin set at t0: for each origin A, each other asset B's FIRST retained trigger with 0<delay<=30 yields edge observation A->B. Same-minute simultaneous origins are co-origins, NOT zero-delay causal edges; preserve metadata. Targets may belong to later episodes. Overlapping event windows can share triggers; don't pretend independent episodes. Exclude origin assets as subsequent targets. Counts per directed pair mean number of eligible source-events with B follow-up; report source-event opportunities and conditional rate alongside count. Median magnitude means B's absolute magnitude_sigma at follow-up. Direction/delay/event identity kept per observation.

Output 100-row 10x10 ordered edge CSV: source,target,count,mean_delay_minutes,median_target_magnitude_sigma,source_event_count,followup_rate. Diagonal count0 with null delay/magnitude/rate, no invented self edges. Offdiagonal zero count has null delay/magnitude; rate0 if opportunities>0, else null. Graph JSON nodes/all84 event timelines and positive edges {source,target,weight:count,delay/magnitude/rate,observations}. Labels explicitly temporal association, not causal contagion.

Strongest3 distinct event sequences ranked solely by number of distinct subsequent target assets desc, then initial origin magnitude desc, then time. Chain is an observed ordered sequence, not proof of hop-by-hop causation. Tied timestamp assets stay a set; don't manufacture ordering. If <3 nonempty sequences, report available count/null.

Fingerprint keeps ten ordered asset slots: trigger direction (+/-1), magnitude_sigma, delay_minutes and present boolean. Origins at delay0 (keep each origin's individual actual magnitude/direction from triggers); targets first follow-up<=30; missing absent/null, not zero strength pretending a measurement. Store raw fingerprints.

Freeze unsupervised method BEFORE reading resulting clusters: average-linkage agglomerative, exactly4 clusters (or min(4,N)); distance is mean over10 assets: both absent0, presence mismatch1, both present mean(direction mismatch 0/1, abs(log1p(z1)-log1p(z2))/(1+abs(log1p(z1)-log1p(z2))), abs(delay1-delay2)/30). Deterministic stable event ordering; no tuning for narrative. Medoid minimizes withincluster distance, tie earlier event. Names derive transparently from most frequent earliest origin (all ties retained), majority trigger sign (tie mixed), and median follow-up count; say “先触发/下跌/广度”, not causal “主导”. Report all4 clusters incl sizes, medoid, silhouette if straightforward/valid else null (quality not assumed).

Demo callable + CLI: nearest_events(index, vector, observed_minutes, exclude_event_id=None, k=3). Validate exact10 assets, finite positive observed magnitudes, sign ±1 and0<=delay<=observed_minutes<=30. Compare new query to HISTORICAL FINGERPRINTS CENSORED AT SAME observed_minutes (prevents using future fingerprints to match an incomplete event). No cluster filter. Return3 {distance,event_id,timestamp,origin_assets,classification_available_at,observed_prefix_fingerprint,full_historical_event(future-labelled),saved predicted/actual risk or reference}. Full fingerprint/cluster label unavailable until t0+30; retrieval similarities NOT probabilities/forecasts. CLI reads supplied JSON only, no model. Add one real query demo packet with self excluded.

## 2. February observational replication

Load ONLY fixed February directory verified manifest/SHA, 40320 rows/asset UTC minute grid. New market windows must say 2026-02, not reuse January profile name unmodified. Reuse frozen detect_events k6 first crossing/cooldown/fixedmerge/systemic definition; no recalibration to count target. Include ALL eligible events, retain excluded edge events and count diagnostics. Archive exact catalog/selection/config/code hashes BEFORE predictions; refuse overwrite/resampling.

Original model config lookback256,horizon30,path_count1,seed20260926, same T/top_p and existing OHLC/volume audit. Four frozen methods same as January. Each snapshot retains raw/corrected paths/errors/runtime/forecast_id, no retry to fix output. Same original scoring universe (systemic all10/non excludes all origins), complete paired events + failure denominators. Save full metrics though one-page primary report only MDD MAE and volatility Top3. Summarize all/up/down/mixed with explicit counts, all four methods. For the “95% comes from down” replication use FIXED historical_30 comparator: contribution_g = n_g/N*(baseline MDD MAE_g - Kronos MDD MAE_g), ratio contribution_down / total only when total>0; else null with reason. This is observational description, no invented significance threshold or A-line decision. Jan benchmark downshare ~95% and all MDD gap8.477bps; do not conflate strongest BTC-beta comparator. No 03 inspection or A-line output access.

## 3. January alarm calibration

Only frozen84 predictions and validated saved actual MDD; primary model Kronos one cached path. For EACH threshold .01/.02/.03/.05 define predicted positive = predicted close-path MDD STRICTLY > threshold, true positive label = actual close-path MDD > SAME threshold. Include ALL84x10=840 pairs (alarm per asset, no origin exclusion) and disclose correlated cases. Output TP/FP/FN/TN,predicted positives,true positives,precision,recall,F1, event count and asset-pair count, with denominator0=>null and reason. F1=0 if errors present and TP0; allzero truth/pred means undefined. No rounding before comparison.

Recommendation rule fixed now: restrict to thresholds with >=10 true positives AND >=10 predicted positives to avoid tiny-cell recommendation; pick highest F1, ties lower threshold. If none qualify recommend null. Describe only within-Jan exploratory calibration, no deployed rule change; existing3% rule unchanged. Retain allfour rows, no result-based pruning. Store source hashes, no models.

## 4. Demo sampling intervals

Exactly the existing3 preselected demo IDs, original model, ten independent paths PER ASSET,256->30,seed20260926; generate a separate new run never replace original one-path snapshot. This is300 asset-path calls (actual measured count/time reported); path0 may repeat deterministic seed of old output by explicit exception, don't relabel the old original snapshot. No resampling invalid paths; disclose failures; if a whole adapter event fails preserve raw evidence and outputnull rather than rescue sampling.

Compute each sampled asset path's MDD with its current spot using frozen risk_metrics, THEN numpy.quantile at .05/.50/.95 method='linear' over10 MDD scalars. No averaging price paths before MDD. Output per event+asset {p05,p50,p95,n_expected:10,n_valid,per_path_mdd,forecast_id}, audit/source identities/timing. Publish interval values only with10 valid samples; otherwise null. These are model-sampling empirical percentiles from10 draws, NOT calibrated95% coverage or independent joint-portfolio scenarios. Per-asset only, no fabricated crossasset path correlation. Keep raw/corrected full outputs separately; page JSON numerical audit fields may stay hidden.

## Verification / stop

Prefix-censored retrieval vs self/partial/missing, exact100 edges+raw observation reproduction, cluster counts84; threshold confusion totals840 and zeros; Feb no-lookahead/same formulas/file hashes; multipath actualcalls300, quantile independent recompute and original Jan snapshots unchanged. Report all nulls/failures. Once files+reports+Demo callable exist and are verified, stop.
