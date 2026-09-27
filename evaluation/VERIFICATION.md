# Evaluation module checks — 2026-09-27 UTC

Latest dual-audit integration:
`scenario-lab/.venv/bin/python -m unittest evaluation.test_evaluation evaluation.test_volume_quality -q`
passed **30 tests in 2.797 seconds**. Prediction and ledger fixtures are explicitly
synthetic; the validators, local January data, sktime and deterministic metric
engine are real. No new model inference or SSH execution was performed.

Coverage now includes mandatory OHLC and volume audits, finite negative
volume/amount preserved while prices score, a both-negative candle counted once,
pending evidence retention, normalized price-error grouping, configurable
exploratory thresholds, separate model/config groups, distinct-window dedup,
conflicting-repeat exclusion, and visible failed/incomplete/missing coverage.
The additional force-run regression verifies OHLC summary duplicates cannot
throw before volume summarization: equivalent audits count once; conflicting
audits retain all attempts without selecting a more favorable result.

The volume association is descriptive and remains `insufficient_evidence` for
small groups. These checks establish accounting and grouping behavior, not that
anomalous volume predicts worse prices. No product alarm was added.

The following correction-only verification is historical, preceding the new
mandatory volume audit. Old files are preserved, not silently promoted to the
new policy.

Latest correction-quality integration:
`scenario-lab/.venv/bin/python -m unittest evaluation.test_evaluation -v`
passed all **19 tests in 2.378 seconds**, using the actual model-owned correction
validator with explicit numerical fixtures and no real model inference.

New checks cover retained metadata for pending/failed scores, invalid summaries
and records, mandatory raw/audit publication validation, candle-weighted rates,
per-model separation, exact-workload matched comparisons, and missing audit
denominators. The weighted counterexample is 6 corrected of 6 candles plus
0 corrected of 12 candles: result 6/18 = 33.33%, never the 50% mean of task rates.
At that historical stage, legacy no-audit arithmetic fixtures explicitly reported
missing correction quality. New scoring now requires both audits.

Separately, consumed P2's already-generated real Kronos artifact
`model_adapter/evidence/containment-expand-v1-mps-256-30.json` without rerunning
inference. Both its 1-path and 2-path forecasts passed the authoritative audit
validator and scored against independent local truth, producing 27 metric rows
each. Correction counts were 1/90 and 7/180, so the descriptive weighted rate
was 8/270 = 2.96296%, with maximum adjustment 5.2885360422 bps. Saved results:
`evaluation/evidence/containment-scoring-v1.json`. These are the same origin with
two configurations, not two independent model-quality samples. No superiority
over a baseline or calibrated forecast correctness is claimed.

The earlier verification below remains historical evidence for the initial module.

Observed: `scenario-lab/.venv/bin/python -m unittest evaluation.test_evaluation -v`
passed all 12 tests in 2.500 seconds. No Kronos model was invoked by this work.

- Actual local January data and actual sktime `SlidingWindowSplitter`/
  `NaiveForecaster` were used.
- An explicitly marked test fixture that copies truth into the prediction has
  zero forecast error under the shared engine. This verifies scoring arithmetic;
  it is not evidence of model quality.
- Two opposing paths retain positive pathwise volatility even when their average
  price path is flat, verifying metrics-before-path-aggregation.
- Changing future truth does not change historical-volatility baseline output.
- Pending truth, corrupt truth, mismatched IDs, rejected samples and not-run tasks
  remain distinct and in accounting.
- Rejected raw nonfinite values and retry attempts remain in separate artifacts.
- Remote-mode results must contain every requested path and the requested horizon.
- Duplicate task results cannot be silently counted twice.

Actual command executed without inference:

```sh
scenario-lab/.venv/bin/python -m evaluation.runner \
  --limit 2 --stride 30 --paths 1 --out evaluation/runs/smoke-plan --plan-only
```

Result: 2 planned tasks, 1,479 available complete input/truth windows, generated
`evaluation/runs/smoke-plan/manifest.json`. Each task contains only 256 historical
input candles per asset, not future truth. No full-month run or remote execution
was started. Real model quality, speedup and fine-tuning outcomes are unmeasured
by these module tests.
