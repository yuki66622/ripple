"""Score frozen predictions against separately loaded observed candles.

Prediction quality is not inferred from engineering success. Regimes stay
unclassified until a past-only calibration rule is explicitly frozen.
"""

from __future__ import annotations

import math
import warnings

from data_pipeline import load_truth
from forecast_metrics.engine import ASSET_METRICS, PAIRING, asset_metrics
from forecast_metrics.engine import freeze_forecast, make_holdings, recalculate
from .corrections import DISCLOSURE, extract_quality, validate_for_publication
from .volume_quality import extract_volume_quality
from .universe import asset_list, portfolio_policy


def _base(status="pending_truth"):
    return {
        "status": status, "rows": [], "sample_count": 0, "failed_count": 0,
        "pending_count": 0, "incomplete_count": 0,
        "quality_status": "insufficient_evidence", "regime": "unclassified",
        "regime_note": "Past-only regime calibration thresholds have not been frozen.",
        "sample_unit": "forecast origin/window; not paths, assets or metric rows",
        "disclosure": "Single-window errors do not establish forecasting superiority; overlapping origins are not independent.",
        "correction_quality": None, "correction_quality_status": "missing",
        "correction_quality_disclosure": DISCLOSURE,
        "volume_quality": None, "volume_quality_status": "missing",
    }


def _row(as_of, symbol, forecast_id, model, metric, predicted=None, actual=None, status="scored"):
    return {
        "as_of": as_of, "symbol": symbol, "forecast_id": forecast_id,
        "model": model, "metric": metric, "predicted": predicted, "actual": actual,
        "absolute_error": abs(predicted - actual) if predicted is not None and actual is not None else None,
        "status": status, "regime": "unclassified",
    }


def _naive_last(history, future_times):
    """sktime supplies a close-only persistence baseline, not fake OHLC bars."""
    import pandas as pd
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.naive import NaiveForecaster

    dates = pd.DatetimeIndex(pd.to_datetime([r["time"] for r in history], utc=True), freq="1min")
    closes = pd.Series([r["close"] for r in history], index=dates, name="close")
    fh = ForecastingHorizon(pd.DatetimeIndex(pd.to_datetime(future_times, utc=True), freq="1min"), is_relative=False)
    # The 1.2 constructor (also called by fit/reset) warns before instance config
    # can be set. Suppress only that default-change warning; no update is used.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The default of config ``remember_data`` will change", category=FutureWarning)
        forecaster = NaiveForecaster(strategy="last", sp=1)
        forecaster.set_config(remember_data=False)
        result = forecaster.fit(closes).predict(fh)
    values = [float(value) for value in result]
    if len(values) != len(future_times) or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("naive-last returned an invalid close path")
    return values


def _historical_volatility(history, horizon):
    # Same deterministic formula, using only H log returns before the origin.
    if len(history) < horizon + 1:
        raise ValueError("historical-volatility baseline needs horizon+1 input candles")
    previous = history[-horizon - 1:]
    return asset_metrics(previous[0]["close"], [r["close"] for r in previous[1:]],
                         [r["high"] for r in previous[1:]], [r["low"] for r in previous[1:]], 1)["volatility"]


def _validate_pair(window, frozen):
    prediction = frozen.data
    assets = asset_list(window["assets"])
    for key in ("as_of", "source", "quote_currency", "interval_seconds"):
        if prediction.get(key) != window.get(key):
            raise ValueError(f"forecast/window {key} mismatch")
    if set(prediction["spots"]) != set(assets):
        raise ValueError("forecast/window assets mismatch")
    for symbol in window["assets"]:
        if prediction["spots"][symbol] != window["histories"][symbol][-1]["close"]:
            raise ValueError(f"forecast/window {symbol} spot mismatch")


def score_forecast(window, forecast_dict, forecast_id, *, model_label="Kronos-base"):
    """Return one-window Evaluation, preserving pending/incomplete/failed state.

    sample_count counts scored origins, not paths. Asset rows compare mean of
    per-path metrics against the single realized path. Full pathwise arithmetic
    is retained in prediction_metrics; no averaging of prices precedes metrics.
    """
    result = _base()
    result.update({"forecast_id": forecast_id, "as_of": window.get("as_of"),
                   "window_id": window.get("window_id"), "model": model_label,
                   "profile_id": window.get("profile_id"), "source": window.get("source"),
                   "quote_currency": window.get("quote_currency")})
    stage = "correction_metadata"
    try:
        result["correction_quality"] = extract_quality(forecast_dict)
        if result["correction_quality"] is not None:
            result["correction_quality_status"] = "unvalidated"
        stage = "volume_metadata"
        result["volume_quality"] = extract_volume_quality(forecast_dict)
        if result["volume_quality"] is not None:
            result["volume_quality_status"] = "unvalidated"
        stage = "validate_forecast_output"
        validate_for_publication(forecast_dict)
        result["correction_quality_status"] = "validated"
        result["volume_quality_status"] = "validated"
        stage = "validate_forecast"
        frozen = freeze_forecast(forecast_dict)
        if frozen.identity != forecast_id:
            raise ValueError("forecast content does not match forecast_id")
        _validate_pair(window, frozen)
        prediction = frozen.data
        horizon = len(prediction["times"])
        policy = portfolio_policy(window)
        holdings = (make_holdings(frozen, policy["initial_capital"], policy["weights"])
                    if policy is not None else make_holdings(frozen))
        if policy is not None:
            result.update(assets=list(window["assets"]), portfolio_policy=policy)
        predicted_metrics = recalculate(frozen, holdings)
        result["path_count"] = predicted_metrics["sample_count"]
        result["prediction_metrics"] = predicted_metrics
        stage = "load_truth"
        truth = load_truth(window, horizon=horizon)
        if truth is None:
            result.update(status="pending_truth", pending_count=1,
                          message="完整预测跨度的真实行情尚不可用；未填补或提前评分。")
            result["rows"] = [
                _row(window["as_of"], symbol, forecast_id, model_label, metric,
                     predicted_metrics["summary"]["assets"][symbol][metric]["mean"], status="pending_truth")
                for symbol in window["assets"] for metric in ASSET_METRICS
            ]
            return result
        stage = "validate_truth"
        if (truth.get("window_id") != window["window_id"] or truth.get("as_of") != window["as_of"]
                or truth.get("times") != prediction["times"] or set(truth.get("assets", {})) != set(window["assets"])):
            raise ValueError("truth identity, assets or future grid mismatch")
        truth_forecast = freeze_forecast({
            "model_revision": "observed-market-candles-v1", "source": prediction["source"],
            "quote_currency": prediction["quote_currency"], "prediction_run_id": "truth:" + window["window_id"],
            "pairing": PAIRING, "as_of": prediction["as_of"], "interval_seconds": prediction["interval_seconds"],
            "times": truth["times"], "spots": prediction["spots"],
            "paths": [{"path_id": "observed", "assets": {
                symbol: {field: truth["assets"][symbol][field] for field in ("close", "high", "low")}
                for symbol in window["assets"]}}],
        })
        stage = "calculate_truth"
        actual_metrics = recalculate(truth_forecast, holdings)
        result["truth_metrics"] = actual_metrics
        result["truth_kind"] = "single observed joint market path; not a model sample"
        stage = "baselines"
        baseline_artifacts = {}
        for symbol in window["assets"]:
            pred = predicted_metrics["summary"]["assets"][symbol]
            actual = actual_metrics["summary"]["assets"][symbol]
            for metric in ASSET_METRICS:
                result["rows"].append(_row(window["as_of"], symbol, forecast_id, model_label,
                                           metric, pred[metric]["mean"], actual[metric]["mean"]))
            past_vol = _historical_volatility(window["histories"][symbol], horizon)
            naive_close = _naive_last(window["histories"][symbol], prediction["times"])
            spot = prediction["spots"][symbol]
            naive_return = (naive_close[-1] - spot) / spot
            result["rows"].append(_row(window["as_of"], symbol, forecast_id, "historical-volatility",
                                       "volatility", past_vol, actual["volatility"]["mean"]))
            result["rows"].append(_row(window["as_of"], symbol, forecast_id, "naive-last",
                                       "terminal_return", naive_return, actual["terminal_return"]["mean"]))
            baseline_artifacts[symbol] = {
                "historical-volatility": {"kind": "scalar_metric", "metric": "volatility", "value": past_vol,
                    "input_first_end": window["histories"][symbol][-horizon-1]["time"],
                    "input_last_end": window["as_of"], "returns_count": horizon,
                    "definition": "population std of last H observed log returns times sqrt(H)"},
                "naive-last": {"kind": "close_path", "times": prediction["times"], "close": naive_close,
                    "metric": "terminal_return", "value": naive_return},
            }
        result.update(status="scored", sample_count=1, baseline_artifacts=baseline_artifacts,
                      message="本窗口已评分；当前证据不足以声称模型优于基线。")
    except Exception as exc:
        if stage == "correction_metadata":
            result["correction_quality_status"] = "invalid"
        if stage == "volume_metadata":
            result["volume_quality_status"] = "invalid"
        # Invalid or unavailable truth must be distinguishable from model/engine failures.
        incomplete = stage in {"load_truth", "validate_truth"}
        result.update(status="incomplete_truth" if incomplete else "failed",
                      failed_count=0 if incomplete else 1, incomplete_count=1 if incomplete else 0,
                      error={"stage": stage, "type": type(exc).__name__, "message": str(exc)})
        # Keep partial numerical rows, but none counts as a completed evaluation.
        for row in result["rows"]:
            row["status"] = result["status"]
    return result
