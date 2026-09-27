# Frozen-window baseline comparison

`comparison.py` answers whether the frozen Kronos configuration beats two simple
baselines on the exact same realized windows. It does not train or invoke a
model, fetch data, choose favorable windows, or alter the product's alert rules.

```sh
scenario-lab/.venv/bin/python -m evaluation.comparison \
  --manifest evaluation/runs/expanded96-20260926/manifest.json \
  --batch-dir evaluation/runs/expanded96-local-20260926/batch_c5e66f2c515e47faa9d8f7b871cac3c8 \
  --out evaluation/evidence/a-new-comparison.json
```

`--out` is a new JSON file; existing results are never overwritten. Optional
`--bootstrap-replicates` is 2,000 by default, or 0 to disable intervals;
`--bootstrap-seed` defaults to 20260926. Exit code 0 means every manifest task
passed evidence checks, **not** that Kronos beat a baseline. Exit code 2 records
partial/no comparable results while still saving complete accounting.

Python: `compare_batch(manifest_dict, batch_dir, bootstrap_replicates=2000)`.

## Evidence and denominators

- The complete manifest fixes the denominator. Every task remains in `tasks`.
  All JSON artifacts under `batch_dir/tasks/` are inspected. Missing, failed,
  pending, incomplete, duplicate and invalid records remain explicit; no retry
  or favorable-attempt selection occurs. Unknown files are separately counted.
- Scored artifacts must match the manifest's input/config/model specification
  and the batch's full `expected_identity`, including sampling, tokenizer,
  adapter, policy, backend and framework versions. This intentionally rejects
  mixing execution identities in one baseline report.
- For each scored artifact, the tool checks untouched raw output against both
  audits, verifies the forecast content ID, compares the input against local
  Binance January data, and calls the existing scorer again against independent
  local truth. Original rows, pathwise metrics, truth metrics and baseline
  artifacts must match. Only `binance_jan2025` is allowed through this gate;
  it cannot silently call a live-data API.
- All 27 scoring rows are required. Model and baseline errors always share
  an asset, origin and realized target. Metrics are computed per path before
  their means are used; no mean price curve replaces pathwise arithmetic.

## Results

`overall.errors` and each fixed `calendar_segments` entry contain:

| Output | Definition |
|---|---|
| Terminal-return MAE | Kronos versus `naive-last`; units are decimal returns |
| Volatility MAE | Kronos versus `historical-volatility`; shared horizon formula |
| Skill | `1 - model_MAE / baseline_MAE`; null for zero baseline MAE |
| Win/tie/loss | Paired per-window absolute-error comparisons |
| Equal-weight assets | Average three asset errors within each window, then windows; not portfolio error |

Calendar partitions are January 1–7, 8–14, 15–21 and 22–31 **UTC origins**.
An empty partition retains its requested/paired denominator and null metrics.

`top2_ranking` compares the exact set of realized highest-volatility assets
against Kronos, past-volatility ranking, and constant ETH+SOL. Competition rank
includes ties: a set can contain all three assets. Reports include exact-set hit
rate, precision, recall, Jaccard, set size, asset-selection counts, and paired
exact-set gains versus both baselines. At-least-one-overlap is not presented as
skill: any two size-two sets among three assets already overlap.

`drawdown_3pct` independently reports strict `mean path MDD > .03` confusion
counts and actual positives by asset. It retains the core P0-plus-closes formula.
Recall is null with no realized positives, and precision is null with no
predicted positives. Sparse positives do not establish useful alert performance.

Optional skill intervals resample whole UTC-day blocks, preserving all windows
within the sampled day and the model/baseline pairing. They are exploratory
percentile intervals, not IID confidence claims or evidence of future-regime
performance. A single represented day has no interval. Cross-day dependence and
this already-inspected January dataset limit interpretation.

## Verified run

The saved `evaluation/evidence/expanded96-comparison-20260926.json` revalidated
all 96 preselected windows, with 96 paired and zero exclusions. Its normalized
cross-asset MAEs agree with root's independent calculation:

| Metric | Kronos MAE | Baseline MAE | Skill |
|---|---:|---:|---:|
| Terminal return | 0.005182523 | 0.004344056 | -19.3015% |
| Volatility | 0.001800942 | 0.001386623 | -29.8797% |

These results do not support a baseline-superiority claim for the current
single-path configuration. All three assets and all four fixed calendar segments
have negative skill for both metrics. Exact-set ranking hits are 75/96 for
Kronos, 86/96 for historical volatility, and 84/96 for constant ETH+SOL. The 3%
drawdown rule has only three realized positives, all SOL: one TP and two FN.

## Checks

```sh
scenario-lab/.venv/bin/python -m unittest evaluation.test_comparison -q
```

17 checks cover arithmetic, equal weighting, denominators, failures and
duplicates, zero baselines, tied ranks, strict 3% thresholds, UTC segmentation,
block-bootstrap repeatability, identity mixing and malformed JSON. Synthetic
ledger tests explicitly isolate the raw-evidence gate. Separate full-audit
synthetic forecasts against real local truth verify rejection of modified
prices, volume records, baseline scores and coherently modified stored truth.
No check performs real model inference.
