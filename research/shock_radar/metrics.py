"""Pure close-risk arithmetic and event-level comparisons for shock radar v1.

No data, model, checkpoint or network access. All ranks sort descending risk;
cutoff ties use uniform inclusion probabilities, never asset-name ordering.
"""

from __future__ import annotations

from collections import Counter
from math import isfinite, log, sqrt
from numbers import Real
from statistics import mean, pstdev

RISK_METRICS = ("max_drawdown", "volatility")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def risk_metrics(spot, closes):
    """Current spot plus future closes; population horizon vol, not annualized."""
    spot = _number(spot, "spot")
    if isinstance(closes, (str, bytes, dict)):
        raise ValueError("closes must be a numerical sequence")
    values = [_number(value, "close") for value in closes]
    if len(values) < 2 or spot <= 0 or any(value <= 0 for value in values):
        raise ValueError("positive spot and at least two positive future closes required")
    prices = [spot, *values]
    returns = [log(b) - log(a) for a, b in zip(prices, prices[1:])]
    peak, drawdown = spot, 0.
    for price in values:
        peak = max(peak, price)
        drawdown = max(drawdown, (peak - price) / peak)
    return {"max_drawdown": drawdown, "volatility": pstdev(returns) * sqrt(len(returns))}


def _average_ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for index in ordered[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _top_weights(values, k):
    cutoff = sorted(values, reverse=True)[k - 1]
    above = sum(value > cutoff for value in values)
    tied = sum(value == cutoff for value in values)
    fraction = (k - above) / tied
    return [1. if value > cutoff else fraction if value == cutoff else 0. for value in values], tied


def rank_scores(predicted, actual, k=3):
    """Average-tie Spearman and expected top-k recall under independent tie draws."""
    p = [_number(value, "predicted risk") for value in predicted]
    a = [_number(value, "actual risk") for value in actual]
    if len(p) != len(a) or isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(p):
        raise ValueError("same-length nonempty risk vectors and 1 <= k <= asset count required")
    pred_constant, actual_constant = len(set(p)) == 1, len(set(a)) == 1
    reason = None
    if pred_constant or actual_constant:
        reason = ("constant_prediction_and_actual" if pred_constant and actual_constant
                  else "constant_prediction" if pred_constant else "constant_actual")
        spearman = None
    else:
        pr, ar = _average_ranks(p), _average_ranks(a)
        pm, am = mean(pr), mean(ar)
        numerator = sum((x - pm) * (y - am) for x, y in zip(pr, ar))
        denominator = sqrt(sum((x - pm)**2 for x in pr) * sum((y - am)**2 for y in ar))
        spearman = max(-1., min(1., numerator / denominator))
    pw, pt = _top_weights(p, k)
    aw, at = _top_weights(a, k)
    return {
        "spearman": spearman, "spearman_reason": reason,
        "top3_recall_expected": sum(x * y for x, y in zip(pw, aw)) / k,
        "k": k, "asset_count": len(p), "predicted_constant": pred_constant,
        "actual_constant": actual_constant, "chance_recall": k / len(p),
        "prediction_cutoff_tie_count": pt, "actual_cutoff_tie_count": at,
        "tie_policy": "uniform cutoff inclusion weights; dot(predicted,actual)/k; flat vectors yield chance k/N",
    }


def _risk_table(table, assets):
    if not isinstance(table, dict) or set(table) != set(assets):
        raise ValueError("risk table must contain every declared asset exactly once")
    result = {}
    for asset in assets:
        row = table[asset]
        if not isinstance(row, dict) or not set(RISK_METRICS) <= set(row):
            raise ValueError(f"{asset}: both risk metrics required")
        values = {metric: _number(row[metric], f"{asset}/{metric}") for metric in RISK_METRICS}
        if not 0 <= values["max_drawdown"] <= 1 or values["volatility"] < 0:
            raise ValueError(f"{asset}: risk metrics out of bounds")
        result[asset] = values
    return result


def score_event(event, predictions, actual, assets):
    """Score each method independently while keeping every failed method visible.

    Include all method keys even on failure (value None). Aggregation may also
    receive an explicit expected method list to account for omitted methods.
    """
    if (not isinstance(assets, (list, tuple)) or not assets
            or any(not isinstance(asset, str) or not asset for asset in assets)
            or len(set(assets)) != len(assets)):
        raise ValueError("ordered unique declared assets required")
    if not isinstance(predictions, dict) or not predictions or any(not isinstance(m, str) or not m for m in predictions):
        raise ValueError("predictions must include named methods, with None for failures")
    if not isinstance(event.get("event_id"), str) or not event["event_id"]:
        raise ValueError("nonempty event_id required")
    eligible = event.get("eligible", True)
    if type(eligible) is not bool:
        raise ValueError("event eligible flag must be boolean")
    systemic = event.get("systemic_flag")
    if type(systemic) is not bool and not (not eligible and systemic is None):
        raise ValueError("eligible events require boolean systemic_flag")
    origins = event.get("origin_assets")
    if (not isinstance(origins, list) or not origins
            or any(not isinstance(asset, str) for asset in origins) or len(set(origins)) != len(origins)
            or any(asset not in assets for asset in origins)):
        raise ValueError("complete unique origin_assets must belong to declared assets")
    scored_assets = ([] if systemic is None else list(assets) if systemic
                     else [asset for asset in assets if asset not in origins])
    if eligible and len(scored_assets) < 3:
        raise ValueError("event scoring universe needs at least three assets for top3")
    result = {
        "event_id": event["event_id"], "timestamp": event.get("timestamp"),
        "systemic_flag": event["systemic_flag"], "origin_assets": list(origins),
        "eligible": eligible, "exclusion_reasons": list(event.get("exclusion_reasons", [])),
        "assets": list(assets), "scored_assets": scored_assets, "scored_asset_count": len(scored_assets),
        "actual_status": "not_scored", "methods": {}, "status": "ineligible" if not eligible else "failed",
    }
    if not eligible:
        result["methods"] = {method: {"status": "ineligible", "error": "Event lacks the frozen input/future availability."}
                             for method in predictions}
        return result
    try:
        actual = _risk_table(actual, assets)
    except (ValueError, TypeError) as exc:
        result.update(actual_status="invalid", error=str(exc))
        result["methods"] = {method: {"status": "failed", "error": "Actual risk unavailable: " + str(exc)} for method in predictions}
        return result
    result["actual_status"] = "valid"
    for method, table in predictions.items():
        try:
            prediction = _risk_table(table, assets)
            metrics = {}
            for metric in RISK_METRICS:
                p = [prediction[asset][metric] for asset in scored_assets]
                a = [actual[asset][metric] for asset in scored_assets]
                mae = mean(abs(x - y) for x, y in zip(p, a))
                metrics[metric] = {"mae": mae, "mae_bps": mae * 10000, **rank_scores(p, a)}
            result["methods"][method] = {"status": "scored", "metrics": metrics}
        except (ValueError, TypeError) as exc:
            result["methods"][method] = {"status": "failed", "error": str(exc)}
    count = sum(value["status"] == "scored" for value in result["methods"].values())
    result["status"] = "scored" if count == len(predictions) else "partial" if count else "failed"
    return result


def aggregate_events(results, methods=None):
    """Equal-weight events on a common complete set, with all catalog counts."""
    results = list(results)
    ids = [result["event_id"] for result in results]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate events cannot be selected or counted twice")
    methods = list(methods) if methods is not None else list(dict.fromkeys(m for r in results for m in r["methods"]))
    if len(set(methods)) != len(methods) or any(not isinstance(method, str) or not method for method in methods):
        raise ValueError("methods must be unique nonempty names")
    if results and not methods:
        raise ValueError("expected methods are required for nonempty results")
    groups = {}
    for group in ("all", "systemic", "non_systemic"):
        catalog = [r for r in results if group == "all" or r["systemic_flag"] == (group == "systemic")]
        eligible = [r for r in catalog if r["eligible"]]
        paired = [r for r in eligible if r["actual_status"] == "valid"
                  and all(r["methods"].get(m, {}).get("status") == "scored" for m in methods)]
        scores, status_counts = {}, {}
        for method in methods:
            status_counts[method] = dict(Counter(r["methods"].get(method, {}).get("status", "missing") for r in catalog))
            scores[method] = {}
            for metric in RISK_METRICS:
                entries = [r["methods"][method]["metrics"][metric] for r in paired]
                defined = [entry["spearman"] for entry in entries if entry["spearman"] is not None]
                scores[method][metric] = {
                    "event_count": len(entries), "mae": mean(e["mae"] for e in entries) if entries else None,
                    "mae_bps": mean(e["mae_bps"] for e in entries) if entries else None,
                    "spearman": mean(defined) if defined else None, "spearman_defined_events": len(defined),
                    "spearman_undefined_events": len(entries) - len(defined),
                    "spearman_undefined_reasons": dict(Counter(e["spearman_reason"] for e in entries if e["spearman"] is None)),
                    "top3_recall_expected": mean(e["top3_recall_expected"] for e in entries) if entries else None,
                }
        paired_ids = {r["event_id"] for r in paired}
        groups[group] = {
            "catalog_event_count": len(catalog), "eligible_event_count": len(eligible),
            "unclassified_event_count": sum(r["systemic_flag"] is None for r in catalog),
            "common_paired_event_count": len(paired), "common_paired_event_ids": [r["event_id"] for r in paired],
            "method_status_counts": status_counts, "scores": scores,
            "excluded_from_pairing": [{"event_id": r["event_id"], "eligible": r["eligible"], "actual_status": r["actual_status"],
                 "method_status": {m: r["methods"].get(m, {}).get("status", "missing") for m in methods}}
                for r in catalog if r["event_id"] not in paired_ids],
        }
    return {"method_names": methods, "groups": groups,
            "aggregation": "Per-event asset MAE first, then equal-weight events. Every method uses the identical complete paired event set within each group.",
            "spearman_aggregation": "Mean only over defined per-event Spearman values; defined/undefined coverage is reported separately for every method.",
            "disclosure": "n counts events, not assets or paths. Overlapping horizons remain dependent. Systemic labels are posthoc. Descriptive exploration, not causal or calibrated forecasting evidence."}
