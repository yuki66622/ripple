# B-line frozen-data honesty checks — 2026-09-27

Only delivered frozen January84 / February56 event artifacts, January catalog/graph/fingerprints, and completed demo-multipath-v2 artifacts may be read. No inference/model imports, raw market-data reads, downloads, March access, A-line changes, or edits to frozen sources. New code/results live only in this directory. Hash inputs before/after. Deliver one compact report plus auditable detail; stop.

## Ownership

- root: coverage.py, coverage.json, integration, report and verification.
- graph_null: permutation.py, permutation.json and tests; frozen January catalog/graph only.
- failure_modes: failures.py, failures.json, top10.csv and tests; frozen January84/February56 outputs only.
- demo_data (cluster-stability role): clustering.py, clustering.json and tests; frozen January fingerprints only.

## Locked analysis before results

1. Coverage: inclusive p05<=actual_MDD<=p95 for exactly3x10=30 asset-events. Use the matching original event's saved actual MDD; verify directly from frozen replay truth if available. No calibrated coverage interpretation. Report pooled, each event, major BTC/ETH/SOL/BNB vs remaining6, below/above interval, invalid/missing. <70% uses user's mandatory 'overconfident; demo must label uncalibrated'. Otherwise observed ratio is still only3 selected, correlated events, not evidence of population calibration. Report tier counts regardless of result, so no conditional cherry-picking.

2. Temporal null:200 replicates, numpy default_rng seed20260927. Independently circularly shift each asset's entire retained-trigger timeline by a uniform offset in [0,44639] minutes modulo44640. This is a cyclic permutation of minute positions preserving per-asset counts, directions, magnitudes and cyclic gaps, while breaking cross-asset alignment. Reconstruct fixed10-minute episodes and original256-history/30-future eligibility from shifted triggers; do not rerun price-based detection. Then original graph rule (first other-asset trigger in (t0,t0+30], exclude simultaneous origins). Zero shift reconstruction must exactly reproduce frozen84-event graph before simulation. Timestamp origin is Jan1 00:01 UTC. All shifted metadata clearly synthetic and kept out of real graphs.

   Primary statistics: number of nonzero directed pairs (upper tail) and mean delay over all source-event-target observations (lower tail). Also keep weighted observation count, eligible/source-event counts and edge-unweighted mean delay for diagnostics. Monte Carlo p=(1+null as/extremer)/(201), ties included. Empty-delay replicates count as +infinity for short-delay testing and have JSON null diagnostics. Holm correction over the two prespecified statistics; alpha .05. Report both p values and quantile position/null range. If neither significant use 'graph may be noise; ripple animation is illustrative'. If significant, only temporal alignment exceeds this null, not causal contagion, stable edges or predictive value. Month shifts do not preserve common daily regimes; explicitly state this null limitation, and variable eligible event counts. Do not change null or seed after seeing results.

3. Failures: rank140 complete paired events by original event-level Kronos MDD MAE, mean over original scored assets (systemic all10, otherwise excluding all origins), descending, ties timestamp+ID. Keep top10 and compare their categorical fractions against all140: month, direction, systemic, strength [6,8)/[8,infinity). Per-event small vs major mean absolute error uses scored assets; don't compare summed errors across tiers of6/4. Report tier available counts and worst asset. Commonality descriptive, postselected, no causal/generalization claim; if no clean concentration say so. Preserve missing/failure denominators, never silently drop. No new risk thresholds or model decisions.

4. Cluster stability: frozen custom distance/average-linkage at k4. First full rerun must reproduce original84 labels as sets. Leave each of84 out and recluster83 with same algorithm/k4; compare common-item partitions via adjusted Rand index (label invariant), matching Jaccard per original class. For the original singleton, report isolated-singleton persistence in the83 deletions that retain it; deleting itself is an absent class, not a failed retention trial. Descriptive stability flag: median ARI>=.9, p10 ARI>=.8, singleton persistence>=.9. These are engineering thresholds, not significance. If fails, keep original frozen four classes but label singleton unstable in supplement. If passes, still label it one isolated observed case with no replicated class evidence; deleting it cannot validate/recover its class. No retune k3 or new taxonomy based on result.

## Verification

Independent math review, source hashes unchanged, exact denominators, no imports/calls to model adapters; targeted tests for tail ties, empty/null denominators, ranking, boundary and label-invariance. Report all four results including unfavorable ones. No UI/demo source edits: required labels are deliverable guidance, not permission to touch unrelated code.
