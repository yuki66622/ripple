"""Exploratory unused-volume anomaly evidence, separate from price accuracy."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from statistics import mean
from .universe import PROFILE_ASSETS, asset_list

FIELDS = ("policy", "status", "total_candles", "volume_invalid_count", "volume_invalid_rate")
METRICS = ("terminal_return", "volatility")
DISCLOSURE = (
    "Exploratory association only: finite negative predicted volume/amount are unused by price metrics. "
    "Volume abnormality is not OHLC correction rate, does not prove worse price forecasts, "
    "and does not trigger a product alarm. Overlapping forecast windows may be dependent."
)


def extract_volume_quality(forecast):
    """Copy checked scalar counts; full raw/record verification lives in P2."""
    audit = forecast.get("volume_quality")
    if audit is None:
        return None
    if not isinstance(audit, dict):
        raise ValueError("volume_quality must be an object")
    value = {key: audit[key] for key in FIELDS}
    if value["policy"] != "unused-volume-audit-v1":
        raise ValueError("unsupported volume audit policy")
    total, invalid, rate = value["total_candles"], value["volume_invalid_count"], value["volume_invalid_rate"]
    if any(isinstance(n, bool) or not isinstance(n, int) for n in (total, invalid)) or total <= 0 or not 0 <= invalid <= total:
        raise ValueError("invalid volume audit counts")
    if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate)
            or not math.isclose(rate, invalid / total, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("volume anomaly rate must equal invalid_count/total_candles")
    expected = "volume_forecast_unavailable" if invalid else "valid"
    if value["status"] != expected:
        raise ValueError("volume audit status does not match invalid count")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _assets(artifact, evaluation, forecast, profile, model):
    candidates = []
    for value in (artifact.get("assets"), (artifact.get("window") or {}).get("assets"),
                  evaluation.get("assets"), PROFILE_ASSETS.get(profile)):
        if value is not None:
            candidates.append(asset_list(value))
    for value in (forecast.get("spots"),
                  (evaluation.get("prediction_metrics") or {}).get("summary", {}).get("assets"),
                  (evaluation.get("truth_metrics") or {}).get("summary", {}).get("assets")):
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("asset metric/spot coverage must be an object")
            candidates.append(asset_list(tuple(value)))
    if not candidates:
        rows = evaluation.get("rows") or []
        symbols = list(dict.fromkeys(row.get("symbol") for row in rows
                       if row.get("model") == model and row.get("metric") in METRICS))
        candidates.append(asset_list(symbols))
    if any(set(value) != set(candidates[0]) for value in candidates[1:]):
        raise ValueError("declared asset universe disagrees with forecast or metric coverage")
    return candidates[0]


def _errors(evaluation, model, assets):
    errors = {metric: {} for metric in METRICS}
    invalid = False
    for row in evaluation.get("rows", []):
        if row.get("model") != model or row.get("metric") not in METRICS or row.get("status") != "scored":
            continue
        symbol, metric, value = row.get("symbol"), row["metric"], row.get("absolute_error")
        if (symbol not in assets or symbol in errors[metric] or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            invalid = True
            continue
        errors[metric][symbol] = float(value)
    complete = bool(assets) and not invalid and all(set(errors[metric]) == set(assets) for metric in METRICS)
    return ({metric: {"by_asset": values,
                      "mean_absolute_error": mean(values.values()) if complete else None}
             for metric, values in errors.items()}, complete)


def _observe(artifact):
    evaluation = artifact.get("evaluation") or {}
    forecast = (artifact.get("result") or {}).get("forecast") or artifact.get("forecast") or {}
    runtime = artifact.get("runtime") or (artifact.get("result") or {}).get("runtime") or {}
    identity = runtime.get("identity") or {}
    model = (artifact.get("model_spec") or {}).get("model_label") or evaluation.get("model") or "unavailable"
    config = artifact.get("predict_config") or evaluation.get("predict_config")
    profile = artifact.get("profile_id") or evaluation.get("profile_id") or "unavailable"
    source = artifact.get("source") or forecast.get("source") or evaluation.get("source") or "unavailable"
    quote = artifact.get("quote_currency") or forecast.get("quote_currency") or evaluation.get("quote_currency") or "unavailable"
    revision = forecast.get("model_revision") or (evaluation.get("prediction_metrics") or {}).get("model_revision") or identity.get("model_revision") or "unavailable"
    universe_error = None
    try:
        assets = _assets(artifact, evaluation, forecast, profile, evaluation.get("model") or model)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        assets, universe_error = (), str(exc)
    context = {"profile_id": profile, "source": source, "quote_currency": quote, "model": model,
               "model_revision": revision, "adapter_revision": identity.get("adapter_revision"),
               "tokenizer_revision": identity.get("tokenizer_revision"), "sampling": identity.get("sampling"),
               "predict_config": config, "assets": list(assets), "volume_policy": "unused-volume-audit-v1"}
    quality = artifact.get("volume_quality")
    if quality is None:
        quality = evaluation.get("volume_quality")
    quality_status = artifact.get("volume_quality_status") or evaluation.get("volume_quality_status") or "missing"
    if quality is not None:
        try:
            quality = extract_volume_quality({"volume_quality": quality})
        except (KeyError, TypeError, ValueError):
            quality, quality_status = None, "invalid"
    status = evaluation.get("status") or artifact.get("status") or "not_evaluated"
    errors, complete_errors = _errors(evaluation, evaluation.get("model") or model, assets)
    context_complete = (all(value != "unavailable" for value in (profile, source, quote, revision, model))
                        and isinstance(config, dict) and set(config) == {"lookback", "horizon", "path_count", "seed"})
    valid_quality = quality_status == "validated" and quality is not None
    reasons = []
    if universe_error is not None:
        reasons.append("invalid_or_missing_asset_universe: " + universe_error)
    if not context_complete:
        reasons.append("missing_comparable_context")
    if not valid_quality:
        reasons.append("volume_audit_not_validated")
    if status != "scored":
        reasons.append("price_truth_not_scored")
    elif not complete_errors:
        reasons.append("incomplete_or_invalid_price_error_rows")
    observation = {
        "window_id": artifact.get("window_id") or evaluation.get("window_id"),
        "as_of": artifact.get("as_of") or evaluation.get("as_of") or forecast.get("as_of"),
        "forecast_id": artifact.get("forecast_id") or evaluation.get("forecast_id"),
        "task_id": artifact.get("task_id"), "artifact_path": artifact.get("artifact_path"),
        "status": status, "volume_quality": quality, "volume_quality_status": quality_status,
        "volume_invalid_rate": quality["volume_invalid_rate"] if valid_quality else None,
        "price_errors": errors, "eligible": not reasons, "exclusion_reasons": reasons,
        "duplicate_count": 0,
    }
    if not observation["as_of"] and not observation["window_id"]:
        observation["eligible"] = False
        observation["exclusion_reasons"].append("missing_window_identity")
    return context, observation


def _deduplicate(observations):
    grouped = {}
    for index, observation in enumerate(observations):
        # Same source/config origin cannot gain statistical weight by rerunning
        # or reloading a revised input hash at the same timestamp.
        key = observation["as_of"] or observation["window_id"] or f"missing-origin-record-{index}"
        grouped.setdefault(key, []).append(observation)
    output = []
    for _, copies in sorted(grouped.items()):
        chosen = dict(copies[0])
        chosen["duplicate_count"] = len(copies) - 1
        if len(copies) > 1:
            signature_fields = ("status", "volume_quality", "volume_quality_status", "price_errors", "eligible", "exclusion_reasons")
            signatures = {_canonical({key: row[key] for key in signature_fields}) for row in copies}
            chosen["attempts"] = [{key: row[key] for key in ("forecast_id", "task_id", "artifact_path", "status", "volume_invalid_rate", "price_errors")} for row in copies]
            if len(signatures) > 1:
                chosen.update(status="conflicting_duplicates", eligible=False, volume_invalid_rate=None,
                              volume_quality=None, volume_quality_status="ambiguous",
                              price_errors={metric: {"by_asset": {}, "mean_absolute_error": None} for metric in METRICS},
                              exclusion_reasons=["conflicting_repeated_window_outputs; no best-run selection"])
        output.append(chosen)
    return output


def _bucket(observations):
    eligible = [row for row in observations if row["eligible"]]
    audits = [row["volume_quality"] for row in observations if row["volume_invalid_rate"] is not None]
    total = sum(q["total_candles"] for q in audits)
    invalid = sum(q["volume_invalid_count"] for q in audits)
    return {
        "window_count": len(observations), "eligible_price_error_windows": len(eligible),
        "total_candles": total, "volume_invalid_count": invalid,
        "volume_invalid_rate": invalid / total if total else None,
        "mean_absolute_errors": {metric: mean(row["price_errors"][metric]["mean_absolute_error"] for row in eligible) if eligible else None
                                 for metric in METRICS},
    }


def _comparison(left, right, minimum):
    enough = min(left["eligible_price_error_windows"], right["eligible_price_error_windows"]) >= minimum
    delta = {metric: (right["mean_absolute_errors"][metric] - left["mean_absolute_errors"][metric]
                       if left["mean_absolute_errors"][metric] is not None and right["mean_absolute_errors"][metric] is not None else None)
             for metric in METRICS}
    return {"status": "descriptive_only" if enough else "insufficient_evidence",
            "right_minus_left_mean_absolute_error": delta,
            "minimum_windows_per_group": minimum,
            "interpretation": "An observed difference is exploratory association, not causal or predictive evidence; no automatic alarm."}


def summarize_volume_quality(artifacts, high_rate_threshold=.25, min_group_windows=5):
    """Compare anomaly rate with existing normalized errors by distinct origin.

    Scalar evidence is counted only after publication validation. Exactly
    equivalent reruns coalesce; conflicting reruns are shown and excluded, never
    selected for the best outcome. The minimum is a descriptive coverage gate,
    not a claim of statistical power or independence.
    """
    if (isinstance(high_rate_threshold, bool) or not isinstance(high_rate_threshold, (int, float))
            or not math.isfinite(high_rate_threshold) or not 0 < high_rate_threshold <= 1):
        raise ValueError("high_rate_threshold must be in (0,1]")
    if isinstance(min_group_windows, bool) or not isinstance(min_group_windows, int) or min_group_windows < 1:
        raise ValueError("min_group_windows must be a positive integer")
    grouped = {}
    for artifact in artifacts:
        context, observation = _observe(artifact)
        key = _canonical(context)
        grouped.setdefault(key, (context, []))[1].append(observation)
    reports = []
    for key, (context, records) in sorted(grouped.items()):
        observations = _deduplicate(records)
        known = [row for row in observations if row["volume_invalid_rate"] is not None]
        zero = _bucket([row for row in known if row["volume_invalid_rate"] == 0])
        nonzero = _bucket([row for row in known if row["volume_invalid_rate"] > 0])
        low = _bucket([row for row in known if row["volume_invalid_rate"] < high_rate_threshold])
        high = _bucket([row for row in known if row["volume_invalid_rate"] >= high_rate_threshold])
        coverage = {
            "input_records": len(records), "distinct_windows": len(observations),
            "duplicate_records": len(records) - len(observations),
            "record_status_counts": dict(Counter(row["status"] for row in records)),
            "distinct_window_status_counts": dict(Counter(row["status"] for row in observations)),
            "volume_audit_status_counts": dict(Counter(row["volume_quality_status"] for row in observations)),
            "eligible_price_error_windows": sum(row["eligible"] for row in observations),
            "conflicting_duplicate_windows": sum(row["status"] == "conflicting_duplicates" for row in observations),
        }
        reports.append({
            "group_id": hashlib.sha256(key.encode()).hexdigest(), **context,
            "coverage": coverage, "observations": observations, "volume_summary": _bucket(known),
            "zero_vs_nonzero": {"zero": zero, "nonzero": nonzero, "comparison": _comparison(zero, nonzero, min_group_windows)},
            "low_vs_high": {"threshold": high_rate_threshold, "high_inclusive": True,
                            "low": low, "high": high, "comparison": _comparison(low, high, min_group_windows)},
            "status": "insufficient_evidence" if (min(zero["eligible_price_error_windows"], nonzero["eligible_price_error_windows"],
                                                        low["eligible_price_error_windows"], high["eligible_price_error_windows"]) < min_group_windows) else "descriptive_only",
        })
    return {
        "policy": "unused-volume-audit-v1", "status": "exploratory", "by_group": reports,
        "high_rate_threshold": high_rate_threshold, "minimum_windows_per_group": min_group_windows,
        "minimum_is_statistical_power_claim": False,
        "anomaly_definition": "volume_invalid_count/total_candles; a candle negative in both fields counts once",
        "price_error_definition": "existing absolute terminal_return and volatility errors; equal-weight mean over every declared asset, then distinct windows",
        "duplicate_policy": "coalesce equivalent repeated origins; expose and exclude conflicting repeats; never choose best run",
        "automatic_alarm": False, "disclosure": DISCLOSURE,
    }
