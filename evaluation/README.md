# Frozen rolling evaluation

Use the existing virtual environment. This module keeps the architecture's
3% drawdown/top-2 volatility rules unchanged. It does not calibrate regimes,
train LoRA, repair invalid predictions, or claim quality from a small sample.

## Immediate scoring

```python
from evaluation.scoring import score_forecast

evaluation = score_forecast(window, frozen_forecast.data, frozen_forecast.identity)
```

The prediction is validated/frozen, each path is processed by `forecast_metrics`,
and scalar metrics are aggregated only afterward. Separately loaded future
candles pass through the same engine as one observed path. They are not supplied
to the predictor. Every numerical row identifies the forecast, origin, asset,
model/baseline and metric. Full pathwise metrics remain in the result.

Two baselines have distinct honest output types:

- `historical-volatility`: scalar H-step volatility from the last H observed log
  returns, using only H+1 input closes. It reuses the existing engine formula.
- `naive-last`: sktime `NaiveForecaster(strategy="last", sp=1)` predicts closes;
  its terminal return is compared separately. No fake OHLC/risk path is made.

Statuses: `scored`, `pending_truth`, `incomplete_truth`, `failed`.
`sample_count` counts scored origins, not paths, assets or metric rows.
`path_count` is reported separately. Errors and incomplete/pending counts remain
visible. Regime is `unclassified` because past-only calibration thresholds are
not frozen; `quality_status` remains `insufficient_evidence`.

## Correction quality is separate evidence

The user-approved `containment-expand-v1` policy is implemented by the model
adapter. M6 does not reapply it. Each score retains `correction_quality` with
`policy`, `total_candles`, `corrected_candles`, `correction_rate`,
`correction_rate_pct`, and `max_adjustment_bps`. A separate
`correction_quality_status` records `validated`, `invalid`, `missing` or
`unvalidated`. Pending truth or a later scoring failure does not erase a valid
structural signal.

New runner output requires untouched `raw_paths` and passes
`model_adapter.validate_forecast_output(forecast, raw_paths)` before publication.
The scorer also requires both audits using the same validator. There is no
no-audit bypass for new scoring. Existing historical artifacts are preserved;
they are not silently migrated or treated as valid new-policy evidence.

Summary `correction_quality.by_model` separates model label, actual weight
revision and correction policy. It adds candle counts across tasks and computes
`sum(corrected_candles) / sum(total_candles)`, rather than averaging per-task
percentages. Invalid/missing/unvalidated audits have their own counts and are
excluded from that denominator. A group with no validated candles has `null`
rate, not 0%. Maximum adjustment is the maximum validated adjustment in bps.

`correction_quality.matched_comparisons` additionally restricts each model pair
to identical `window_id`, complete prediction configuration (including path
count, horizon and seed), and complete runtime controls: tokenizer/adapter
revision, the entire sampling object, output/volume policies, backend,
PyTorch/sktime versions and model/output timestamp semantics. Model label and
weight revision may differ; the runtime weight revision must agree with the
forecast. Missing legacy identity remains usable for descriptive single-model
counts but cannot establish fair comparability. `unmatched_a_count` and
`unmatched_b_count` expose valid ledgers excluded from each paired denominator.
Different path counts can contribute to each model's
weighted descriptive summary, but cannot be silently paired for a fair model
comparison. Equivalent same-model/window/config scalar audits count once.
Conflicting repeats, including changed runtime controls at the same workload,
are excluded from the weighted summary with all attempts
retained; they do not throw or silently select the more favorable attempt.

Fewer corrections indicate better structural validity under equal settings.
They do **not** establish improved forecast accuracy. Financial formulas,
3%/top-2 alert definitions and regime classification are unchanged.

## Unused volume evidence and price errors

`unused-volume-audit-v1` permits finite negative predicted volume/amount while
preserving their raw values. Price metrics do not use those fields. Missing,
nonfinite or otherwise structurally invalid outputs still fail the model-owned
validator. A negative volume and amount in the same candle count only once.

Each score retains the scalar `volume_quality` fields `policy`, `status`,
`total_candles`, `volume_invalid_count`, and `volume_invalid_rate`, plus separate
`volume_quality_status` validation state. Pending/failed results keep this
evidence when available. Neither missing evidence nor invalid evidence counts
as zero anomalies. This metadata is for artifacts/reports, not the product UI.

`evaluation.volume_quality.summarize_volume_quality(artifacts,
high_rate_threshold=.25, min_group_windows=5)` returns per-window observations,
coverage, zero/nonzero groups and low/high groups. It groups the same profile,
source, quote currency, model/revisions, and exact prediction configuration.
Rates are weighted by candle counts. Comparisons reuse existing terminal-return
and volatility absolute errors, equally averaged over the three assets and then
distinct windows; raw-dollar errors are never combined across assets.

Equivalent repeats count once per origin in each group. Conflicting repeats
remain visible and are excluded from the association rather than cherry-picked.
Pending, failed, incomplete and missing-context records remain in coverage.
The default high-rate threshold of 25% is only an exploratory grouping setting,
configurable with `--volume-high-rate`. It creates no alarm. Fewer than five
eligible windows in either group yields `insufficient_evidence`; meeting that
minimum permits descriptive comparison only, not statistical or causal claims.

## Freeze tasks without inference

```sh
scenario-lab/.venv/bin/python -m evaluation.runner \
  --limit 4 --stride 30 --paths 1 --out evaluation/runs/four-windows --plan-only
```

`sktime.SlidingWindowSplitter` freezes a 256-bar input and a full 30-bar future
horizon. A 30-minute stride yields 1,479 available January origins; the explicit
limit controls how many are selected. A 10-minute stride yields overlapping
labels that must not be counted as independent samples. Default limit is four,
so no command silently starts a whole-month job.

Generated `manifest.json` contains complete past-only MarketWindow tasks. Each
task covers all three assets and all requested paths. Stable task IDs include
input identity, model specification and sampling configuration. Identical folds
must be reused for original versus separately trained compatible local weights.
An optional `--model-path`, `--tokenizer-path`, and `--model-label` select those
weights; an absent/failing fine-tuned checkpoint generates no fictional score.

## Execute only when the model device is available

```sh
scenario-lab/.venv/bin/python -m evaluation.runner \
  --manifest evaluation/runs/four-windows/manifest.json \
  --out evaluation/runs/four-windows
```

For a single task on another computer:

```sh
scenario-lab/.venv/bin/python -m evaluation.runner \
  --manifest /path/to/manifest.json --task-id task_HASH \
  --out /path/to/output --device auto --no-score
```

Default model paths resolve relative to each checkout. Explicit model paths in
a manifest must also exist on that machine. A remote worker can skip scoring
when it lacks the full historical archive; the coordinator runs `score_forecast`
on the returned result using the original input and locally held truth. The
worker does not need Binance data when its complete input arrives in the task.

Python integration: `build_manifest(...)`, `run_task(task, out_dir,
adapter=resident_adapter, score=False)`, and `summarize(selected_attempts)`.
Raw successful output and rejected model paths are saved per attempt. Nonfinite
rejected values are explicitly encoded as `NaN`/`Infinity` strings, never silently
replaced with numerical values. A retry creates a new file. Aggregation rejects
duplicate task IDs until the coordinator explicitly selects one attempt.

Every run retains a summary with scored, failed, incomplete, pending and not-run
denominators. Summary completion describes execution accounting, not model
quality. No distributed acceleration has been claimed by this module.

## Checks without model inference

```sh
scenario-lab/.venv/bin/python -m unittest \
  evaluation.test_evaluation evaluation.test_volume_quality -v
```

Tests use marked numerical fixtures only for prediction, actual local January
data, real sktime splitting/persistence, and the shared deterministic engine.
