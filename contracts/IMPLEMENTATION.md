# Frozen demo implementation contract — v1

Implements ARCHITECTURE_BASELINE.md without changing its architecture or financial definitions. Python package imports run from project root, using scenario-lab/.venv/bin/python. No financial formula copies in UI, data or model modules.

## Data / P1

Export `data_pipeline.load_window(profile_id='binance_jan2025', *, lookback=256, as_of=None) -> dict` and `load_truth(window, *, horizon=30) -> dict | None`. `as_of` is optional ISO UTC candle END; default replay window must leave 30 known future bars for immediate replay scoring. Profiles `binance_jan2025` (Binance spot USDT) and `kraken_live` (Kraken USD). Source/quote changes are explicit. Network timeouts bounded; no fake candles.

MarketWindow keys: `schema_version:1`, `window_id`, `profile_id`, `source`, `mode:'replay'|'live'`, `quote_currency`, `as_of`, `interval_seconds:60`, `assets:['BTC','ETH','SOL']`, `histories:{symbol:[{time:ISO_UTC_END,open,high,low,close,volume,amount}]}`, `quality:{amount_source, ...}`. Only complete aligned closed candles; no silent filling. ID covers immutable numerical input and provenance, not wall-clock fetch time. Truth is separate: `{window_id,as_of,times:[future_end],assets:{symbol:{open:[],high:[],low:[],close:[],volume:[],amount:[]}}}`; never attach to inference window.

## Model / P2

Export `model_adapter.KronosAdapter` with `identity() -> dict` (model_revision, tokenizer_revision, adapter_revision, backend) and `predict(window, config, prediction_run_id) -> {'forecast':dict,'runtime':dict}`. Config: `{lookback:256,horizon:30,path_count:1,seed:20260926}`; support 1..16 paths with preserved individual output. Forecast exactly matches forecast_metrics.freeze_forecast: model_revision, prediction_run_id, source, quote_currency, as_of, interval_seconds, times, spots, paths[{path_id,assets:{symbol:{close:[],high:[],low:[]}}}], pairing=`paired_scenarios_not_calibrated_joint_distribution`. Raw OHLCVA optional separate result key `raw_paths`, not discarded. Runtime includes load_ms, inference_ms, backend and path_count. Validate actual prices, no silent OHLC repair.

Use selected sktime KronosForecaster/local weights. Upstream averages sample_count: generate individual sample_count=1 calls if required to preserve real paths, reuse weights safely; do not claim multiple paths from one averaged result. fixed per-window/asset/path seeds independent of task assignment. Underlying global RNG access serialized. Actual inference is P2-owned until root receives handoff.

### User-approved output policy: containment-expand-v1

User explicitly approved this limited amendment after actual native OHLC failures. For each original candle use `high'=max(high,low,open,close)` and `low'=min(low,high,open,close)`; never alter open/close, never interpolate missing bars, never retry samples until valid. Missing fields/bars, nonfinite values and nonpositive prices still reject the whole batch. All other architecture and financial formulas remain frozen.

ForecastSnapshot adds `ohlc_corrections`: `{policy:'containment-expand-v1',bps_denominator:'original_close',total_candles,corrected_candles,correction_rate:decimal,correction_rate_pct:number,max_adjustment_bps:number,records:[{path_id,asset,time,original:{open,high,low,close},corrected:{open,high,low,close},high_adjustment_bps,low_adjustment_bps,max_adjustment_bps,was_corrected:bool}]}`. Keep one record for EVERY output candle; zero adjustment and false for unchanged bars. Bps = absolute change divided by original positive close times 10000; max_adjustment_bps is max of high/low adjustments. correction_rate denominator is paths×assets×horizon; correction_rate_pct=rate×100. raw_paths retain untouched complete OHLCVA. forecast.paths uses corrected bounds and unchanged closes. The added audit field participates in existing content hash.

M6 preserves per-forecast correction_quality `{policy,total_candles,corrected_candles,correction_rate,correction_rate_pct,max_adjustment_bps}` separately from forecast accuracy. Fewer corrections show improved structural validity under equal data/paths/settings, not proof of price accuracy. UI displays rate, max bps and raw/adjusted record access. Application/worker boundary verifies correction metadata against original and corrected values before publication. No financial formulas are changed.

## App / root

### Subsequent user-approved amendment: unused-volume-audit-v1

Finite negative predicted volume/amount no longer reject otherwise valid prices: these fields are unused by v1 financial metrics. Do not clip, drop, fill or resample them. All six OHLCVA fields remain required, finite, correctly shaped; nonpositive OHLC or missing candles still reject the entire batch. Input-market validation is unchanged. Existing containment-expand-v1 applies only to high/low.

ForecastSnapshot additionally has `volume_quality:{policy:'unused-volume-audit-v1',status:'valid'|'volume_forecast_unavailable',total_candles:int,volume_invalid_count:int,volume_invalid_rate:decimal,records:[{path_id,asset,time,volume:number,amount:number,volume_valid:bool}]}`. Every candle has one record bound to untouched raw output; volume_valid is true iff both volume and amount are >=0. Count/total expresses e.g. 30/90; one candle negative in both fields counts once. Snapshot hash includes this audit. Adapter revision changes so old caches cannot supply new-policy predictions.

Export `model_adapter.validate_forecast_output(forecast, raw_paths=None)` to require both exact OHLC correction and complete volume audit for new publication/scoring. Existing validate_corrections remains usable for preserved old artifacts; old results are not silently migrated. New app, evaluation runner and distributed-worker publication must call the full validator with raw_paths. Restore may inspect old legacy snapshots under their existing stricter rules, without claiming new volume evidence.

M6 stores scalar volume_quality separately from OHLC correction rate. Compare abnormal-volume rate against normalized price errors across distinct windows under the same profile/model/configuration. Emit per-window observations and explicitly exploratory zero/nonzero and configurable high-rate groups (default threshold 0.25, evaluator parameter only, not a product alarm). Preserve pending/failed samples and coverage; no automatic suspicious-batch alarm until evidence and a rule are separately approved. Price accuracy evidence uses existing terminal_return and volatility absolute errors, never mixed-asset raw price errors. The UI must not display predicted volume, volume flags or volume-quality signals. Data artifacts and machine-readable reports preserve them for review. No financial formula changes.

ViewState:
```
{view_revision:int,status:'idle'|'loading'|'predicting'|'ready'|'error',
 job:null|{job_id,status,request_id},error:null|{code,message,retryable},stale:bool,
 profiles:[{profile_id,label,mode,quote_currency}],profile_id:null|string,
 window:null|MarketWindow,forecast:null|{forecast_id,...ForecastSnapshot},
 metrics:null|existing_metrics,alerts:null|existing_alerts,
 counters:{model_runs,cache_hits},runtime:null|dict,
 evaluation:{status,rows:[],...},policy:{drawdown_threshold:0.03,top_k:2}}
```
GET /api/state is read-only. POST /api/analyze `{request_id,profile_id,predict_config:{lookback,horizon,path_count,seed},force_run,expected_view_revision,as_of?}` -> 202 `{job_id,...ViewState}`; same request_id is idempotent. GET state polls progress. POST /api/holdings `{capital,weights:{BTC:.5,ETH:.3,SOL:.2},expected_forecast_id,expected_view_revision}` -> ViewState. POST /api/alerts `{drawdown_threshold,top_k,expected_view_revision}` -> ViewState. GET /api/report -> engine.report_payload; GET /api/evaluation -> Evaluation. Errors `{error:{code,message,retryable},view_revision}` with 400/409/503. Percentages decimal over API. All versions server-authoritative. Request ID conflict must be explicit, not run twice.

## UI / P4

Single page vanilla HTML/CSS/JS in demo_web/, no build/CDN. Same-origin API only. Display replay/live, source/quote/time/horizon, manual analysis, asset metrics and ranking/reasons, portfolio representative path (label path ID, never label it mean risk), fixed-quantity holdings editor, threshold editor, model vs baseline quality/error curve, provenance and JSON download. Every active result comes from server. Initial empty, busy, error/stale, conflict, ready states. Chinese, restrained ivory/ink visual style, keyboard accessible, 390px and desktop. 3% + top2 remain unchanged. Always visible concise disclosure: paired scenarios do not model cross-asset correlation; sample quantiles are not calibrated probabilities. UI must explain sparse one-path quantiles without inventing intervals. No news/event or report LLM scope.

## Evaluation / P5 (next available slot)

Use sktime SlidingWindowSplitter/NaiveForecaster and financial metrics from forecast_metrics. Equal input windows, truth only scorer, explicit failure denominator. Baseline close persistence and historical volatility scalar are separate honest outputs. Module API negotiated with root before writing. Process isolated evaluation should reuse same KronosAdapter and MarketWindow. No automatic whole-month run. Multi-device dispatch leaves a full window/three assets/all paths in one task.
