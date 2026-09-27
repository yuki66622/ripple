# Public snapshot validation

This repository is a clean release snapshot of the working HackUMBC project, not its private development directory or a claim that every experiment succeeded.

The release copy passed 62 frontend tests and 70 Python tests covering live ingestion, per-asset volatility state, forecast identity, deterministic financial arithmetic, OHLC containment correction, volume-quality handling, model-input validation, cache behavior, historical replay, and refresh scheduling.

The original working demo was rendered and exercised in English and Chinese against real Binance closed-minute data and local Kronos-base forecasts. The last UI update keeps historical recovery references separate from live per-asset volatility. Desktop was tested; mobile was intentionally outside the requested demo scope.

These software tests check implementation behavior. They do not establish trading profitability, forecast accuracy in a new market regime, or calibrated uncertainty. Research conclusions and failed fine-tuning experiments are documented separately in [EXPERIMENTS.md](EXPERIMENTS.md).

Source code, frozen summary tables, event catalogs, the frontend's saved replay data, and selected research artifacts are included. Model weights, raw monthly archives, per-run large prediction dumps, credentials, process logs and local environments are excluded. Some research reproduction tests require those omitted original artifacts; only the commands in the root README are the portable release test suite. Historical HTML replay works without model weights. Live inference requires the checkpoint download step.

No training, new historical inference, checkpoint selection or raw holdout access was performed to prepare this release. Existing numerical artifacts were retained; local absolute paths in documentation were converted to repository-relative references. The cropped recording preserves its original audio and duration.
