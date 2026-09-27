"""Pure, tier-specific scoring for the frozen A-line protocol.

No file, model, or network access occurs here. ``score_window`` retains failed
methods on every scheduled origin. Its ``tiers[tier]['methods'][name]`` records
are the input to ``summarize``; baseline names are ``dynamic_naive`` and
``fixed_pair``. Rates use fractions (not percent), volatility uses decimal units.
Ranking uses competition rank <= 2, so ties can produce sets larger than two.
"""

from __future__ import annotations

import copy
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev

from forecast_metrics.engine import asset_metrics, freeze_forecast


TIERS = {
    "major": ("BTC", "ETH", "SOL", "BNB"),
    "small": ("XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC"),
}
ASSETS = tuple(asset for tier in TIERS.values() for asset in tier)
FIXED_SMALL_PAIR = ("AVAX", "LINK")
PROTOCOL_VERSION = "lora-a-v2.1"
MAJOR_TOLERANCE = 0.02
LOOKBACK = 256
HORIZON = 30
BOOTSTRAP_SEED = 20260926
BOOTSTRAP_REPLICATES = 2000
BASELINES = ("dynamic_naive", "fixed_pair")


def _fixed_pair(pair):
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(not isinstance(asset, str) or asset not in TIERS["small"] for asset in pair)
            or len(set(pair)) != 2):
        raise ValueError("fixed_pair must contain two distinct frozen small-tier assets")
    return tuple(sorted(pair))


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp requires a timezone")
    return result.astimezone(timezone.utc)


def _positive(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("prices must be finite positive numbers")
    return float(value)


def _error(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def _top_two(volatilities):
    return [symbol for symbol, value in volatilities.items()
            if 1 + sum(other > value for other in volatilities.values()) <= 2]


def _top_three_recall(predicted, actual):
    """Expected recall under independent uniform tie draws, as in B-line.

This observation-only metric does not change competition-rank top-two sets.
"""
    k = min(3, len(actual))

    def weights(values):
        cutoff = sorted(values.values(), reverse=True)[k - 1]
        above = sum(value > cutoff for value in values.values())
        tied = sum(value == cutoff for value in values.values())
        fraction = (k - above) / tied
        return {asset: 1.0 if value > cutoff else fraction if value == cutoff else 0.0
                for asset, value in values.items()}

    pw, aw = weights(predicted), weights(actual)
    return sum(pw[asset] * aw[asset] for asset in actual) / k


def _failed(error, actual_count=None):
    return {"status": "failed", "exact_set_hit": None, "hit_count": None,
            "selected_set": None, "selected_count": None,
            "actual_count": actual_count, "volatility_mae": None,
            "absolute_errors": None, "predicted_volatility": None,
            "max_drawdown_mae": None, "max_drawdown_absolute_errors": None,
            "predicted_max_drawdown": None, "volatility_top3_recall": None,
            "error": error}


def _score(actual, *, predicted=None, selected=None,
           actual_drawdown=None, predicted_drawdown=None):
    actual_set = _top_two(actual)
    chosen = list(selected) if selected is not None else _top_two(predicted)
    errors = ({asset: abs(predicted[asset] - actual[asset]) for asset in actual}
              if predicted is not None else None)
    drawdown_errors = ({asset: abs(predicted_drawdown[asset] - actual_drawdown[asset])
                       for asset in actual}
                      if predicted_drawdown is not None and actual_drawdown is not None else None)
    return {"status": "scored", "exact_set_hit": set(chosen) == set(actual_set),
            "hit_count": len(set(chosen) & set(actual_set)),
            "selected_set": chosen, "selected_count": len(chosen),
            "actual_count": len(actual_set),
            "volatility_mae": mean(errors.values()) if errors is not None else None,
            "absolute_errors": errors, "predicted_volatility": predicted,
            "max_drawdown_mae": mean(drawdown_errors.values()) if drawdown_errors is not None else None,
            "max_drawdown_absolute_errors": drawdown_errors,
            "predicted_max_drawdown": predicted_drawdown,
            "volatility_top3_recall": _top_three_recall(predicted, actual) if predicted is not None else None,
            "error": None}


def _validate_series(series):
    for name in ("close", "high", "low"):
        if not isinstance(series[name], list) or len(series[name]) != HORIZON:
            raise ValueError(f"{name} must contain exactly {HORIZON} future prices")
        for value in series[name]:
            _positive(value)
    if any(not lo <= close <= hi for lo, close, hi in
           zip(series["low"], series["close"], series["high"])):
        raise ValueError("low <= close <= high required")


def _history(window, anchor):
    if set(window["assets"]) != set(ASSETS) or len(window["assets"]) != len(ASSETS):
        raise ValueError("window must contain the frozen ten-asset universe")
    if window["interval_seconds"] != 60:
        raise ValueError("A-line requires one-minute candles")
    if set(window["histories"]) != set(ASSETS):
        raise ValueError("history assets mismatch")
    spots, naive = {}, {}
    for asset in ASSETS:
        history = window["histories"][asset]
        if len(history) != LOOKBACK:
            raise ValueError(f"{asset}: exactly 256 historical candles required")
        closes = [_positive(bar["close"]) for bar in history]
        for index, bar in enumerate(history):
            expected = anchor - timedelta(minutes=LOOKBACK - 1 - index)
            if _timestamp(bar["time"]) != expected:
                raise ValueError(f"{asset}: historical time grid mismatch")
        returns = [math.log(b) - math.log(a) for a, b in zip(closes, closes[1:])]
        spots[asset] = closes[-1]
        naive[asset] = pstdev(returns) * math.sqrt(HORIZON)
    return spots, naive


def _actual(window, future, anchor, spots):
    if "assets" in future:
        if future.get("window_id") != window["window_id"]:
            raise ValueError("truth/window identity mismatch")
        if _timestamp(future["as_of"]) != anchor:
            raise ValueError("truth/window anchor mismatch")
        expected = [anchor + timedelta(minutes=i) for i in range(1, HORIZON + 1)]
        if [_timestamp(t) for t in future["times"]] != expected:
            raise ValueError("truth/window future grid mismatch")
        future = future["assets"]
    if set(future) != set(ASSETS):
        raise ValueError("truth assets mismatch")
    actual, drawdown = {}, {}
    for asset in ASSETS:
        series = future[asset]
        _validate_series(series)
        metrics = asset_metrics(spots[asset], series["close"], series["high"], series["low"], 1)
        actual[asset] = metrics["volatility"]
        drawdown[asset] = metrics["max_drawdown"]
    return actual, drawdown


def _prediction(window, value, anchor, spots):
    if value is None:
        raise ValueError("prediction unavailable")
    # Accept either the forecast itself or the public adapter result wrapper.
    if "forecast" in value:
        value = value["forecast"]
    forecast = freeze_forecast(value).data
    for key in ("source", "quote_currency", "interval_seconds"):
        if forecast[key] != window[key]:
            raise ValueError(f"forecast/window {key} mismatch")
    if _timestamp(forecast["as_of"]) != anchor:
        raise ValueError("forecast/window anchor mismatch")
    if len(forecast["times"]) != HORIZON:
        raise ValueError("forecast horizon must be exactly 30")
    if forecast["spots"] != spots:
        raise ValueError("forecast/window spots mismatch")
    predicted, drawdown = {}, {}
    for asset in ASSETS:
        path_metrics = [asset_metrics(
            spots[asset], path["assets"][asset]["close"], path["assets"][asset]["high"],
            path["assets"][asset]["low"], 1) for path in forecast["paths"]]
        predicted[asset] = mean(item["volatility"] for item in path_metrics)
        drawdown[asset] = mean(item["max_drawdown"] for item in path_metrics)
    return predicted, drawdown, len(forecast["paths"])


def score_window(window, future, predictions, *, fixed_pair=FIXED_SMALL_PAIR):
    """Score a MarketWindow v1 and explicit future prices without reading data.

``future`` may be asset->OHLC lists or the harness truth wrapper with matching
identity/times. ``predictions`` maps model names to forecast dictionaries (or
adapter wrappers); None and invalid predictions remain failed scheduled rows.
Invalid origin identity raises, since it cannot be paired safely. All other
input/model failures are recorded. No model is dropped from the returned row.
``fixed_pair`` must be chosen and frozen from permitted training data by the
caller. No data-dependent choice happens here. The legacy AVAX/LINK default is
only API compatibility; v2.1 callers explicitly pass their January-fixed pair.
Passing None is an error rather than an implicit baseline-selection request.
"""
    fixed_pair = _fixed_pair(fixed_pair)
    window_id = window["window_id"]
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("nonempty window_id required")
    anchor = _timestamp(window["as_of"])
    if any(not isinstance(name, str) or not name or name in BASELINES
           for name in predictions):
        raise ValueError("model names must be nonempty and cannot shadow baselines")
    row = {"protocol_version": PROTOCOL_VERSION,
           "fixed_pair": list(fixed_pair), "window_id": window_id, "as_of": window["as_of"],
           "utc_day": anchor.date().isoformat(), "truth_status": "failed",
           "failures": [], "tiers": {}}
    try:
        spots, naive = _history(window, anchor)
        actual, actual_drawdown = _actual(window, future, anchor, spots)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        failure = _error(exc)
        row["failures"].append({"method": "truth_or_history", **failure})
        for tier in TIERS:
            names = ["dynamic_naive", *predictions]
            if tier == "small":
                names.append("fixed_pair")
            row["tiers"][tier] = {"actual_volatility": None, "actual_max_drawdown": None, "actual_set": None,
                                  "actual_count": None,
                                  "methods": {name: _failed(failure) for name in names}}
        return row
    row["truth_status"] = "scored"
    for tier, assets in TIERS.items():
        truth = {asset: actual[asset] for asset in assets}
        row["tiers"][tier] = {
            "actual_volatility": truth, "actual_set": _top_two(truth),
            "actual_max_drawdown": {asset: actual_drawdown[asset] for asset in assets},
            "actual_count": len(_top_two(truth)),
            "methods": {"dynamic_naive": _score(
                truth, predicted={asset: naive[asset] for asset in assets})}}
        if tier == "small":
            row["tiers"][tier]["methods"]["fixed_pair"] = _score(
                truth, selected=fixed_pair)
    for name, value in predictions.items():
        try:
            values, drawdown, path_count = _prediction(window, value, anchor, spots)
            for tier, assets in TIERS.items():
                item = _score({a: actual[a] for a in assets},
                              predicted={a: values[a] for a in assets},
                              actual_drawdown={a: actual_drawdown[a] for a in assets},
                              predicted_drawdown={a: drawdown[a] for a in assets})
                item["path_count"] = path_count
                row["tiers"][tier]["methods"][name] = item
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            failure = _error(exc)
            row["failures"].append({"method": name, **failure})
            for tier in TIERS:
                row["tiers"][tier]["methods"][name] = _failed(
                    failure, row["tiers"][tier]["actual_count"])
    return row


def merge_rows(*batches):
    """Merge separately cached model scores only on identical window evidence.

Returns the union of scheduled origins, never their intersection. A missing
model then remains missing in summarize; conflicting truth/baselines/models
raise instead of silently replacing cached evidence.
"""
    merged = {}
    for batch in batches:
        seen = set()
        for row in batch:
            key = row["window_id"]
            if key in seen:
                raise ValueError("duplicate window_id within score batch")
            seen.add(key)
            if key not in merged:
                merged[key] = copy.deepcopy(row)
                continue
            existing = merged[key]
            for field in ("as_of", "utc_day", "truth_status", "protocol_version", "fixed_pair"):
                if existing[field] != row[field]:
                    raise ValueError(f"cached window {field} mismatch")
            for tier in TIERS:
                left, right = existing["tiers"][tier], row["tiers"][tier]
                for field in ("actual_volatility", "actual_max_drawdown", "actual_set", "actual_count"):
                    if left[field] != right[field]:
                        raise ValueError(f"cached {tier} {field} mismatch")
                for name, item in right["methods"].items():
                    if name in left["methods"] and left["methods"][name] != item:
                        raise ValueError(f"conflicting cached method {name}")
                    left["methods"][name] = copy.deepcopy(item)
            for failure in row["failures"]:
                if failure not in existing["failures"]:
                    existing["failures"].append(copy.deepcopy(failure))
    return sorted(merged.values(), key=lambda row: (row["as_of"], row["window_id"]))


def _quantile(ordered, q):
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _paired_interval(day_values):
    """Sample UTC days with replacement, retaining every origin in each day."""
    if not day_values:
        return {"estimate": None, "ci95": None, "n_days": 0,
                "interval_status": "no_paired_windows"}
    groups = defaultdict(list)
    for day, value in day_values:
        groups[day].append(value)
    totals = [(math.fsum(groups[day]), len(groups[day])) for day in sorted(groups)]
    estimate = math.fsum(total for total, _ in totals) / sum(n for _, n in totals)
    if len(totals) < 2:
        return {"estimate": estimate, "ci95": None, "n_days": len(totals),
                "interval_status": "insufficient_day_clusters"}
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [totals[rng.randrange(len(totals))] for _ in totals]
        samples.append(math.fsum(total for total, _ in selected)
                       / sum(n for _, n in selected))
    samples.sort()
    return {"estimate": estimate,
            "ci95": [_quantile(samples, .025), _quantile(samples, .975)],
            "n_days": len(totals), "interval_status": "reported"}


def _item(row, tier, method):
    return row["tiers"][tier]["methods"].get(method)


def _method_summary(rows, tier, method):
    scored = [item for row in rows if (item := _item(row, tier, method))
              and item["status"] == "scored"]
    n = len(rows)
    maes = [item["volatility_mae"] for item in scored
            if item["volatility_mae"] is not None]
    errors = defaultdict(list)
    drawdown_errors = defaultdict(list)
    for item in scored:
        for asset, value in (item["absolute_errors"] or {}).items():
            errors[asset].append(value)
        for asset, value in (item["max_drawdown_absolute_errors"] or {}).items():
            drawdown_errors[asset].append(value)
    drawdown_maes = [item["max_drawdown_mae"] for item in scored if item["max_drawdown_mae"] is not None]
    recalls = [item["volatility_top3_recall"] for item in scored if item["volatility_top3_recall"] is not None]
    return {
        "n_scheduled": n, "n_scored": len(scored), "n_failed": n - len(scored),
        "n_missing": sum(_item(row, tier, method) is None for row in rows),
        "coverage": len(scored) / n if n else 0.0,
        "exact_set_hits": sum(item["exact_set_hit"] for item in scored),
        "exact_set_hit_rate": mean(item["exact_set_hit"] for item in scored) if scored else None,
        "mean_hit_count": mean(item["hit_count"] for item in scored) if scored else None,
        "volatility_mae": mean(maes) if maes else None,
        "n_mae_origins": len(maes),
        "n_mae_asset_origins": sum(len(values) for values in errors.values()),
        "n_mae_asset_origins_scheduled": n * len(TIERS[tier]) if method != "fixed_pair" else 0,
        "mae_sample_unit": "asset-origin; averaged equally across assets and origins within tier",
        "asset_volatility_mae": {asset: mean(values) for asset, values in errors.items()},
        "max_drawdown_mae": mean(drawdown_maes) if drawdown_maes else None,
        "asset_max_drawdown_mae": {asset: mean(values) for asset, values in drawdown_errors.items()},
        "n_max_drawdown_mae_origins": len(drawdown_maes),
        "n_max_drawdown_mae_asset_origins": sum(len(values) for values in drawdown_errors.values()),
        "volatility_top3_recall": mean(recalls) if recalls else None,
        "n_volatility_top3_recall_origins": len(recalls),
        "selected_count_histogram": dict(sorted(Counter(item["selected_count"] for item in scored).items())),
        "actual_count_histogram": dict(sorted(Counter(item["actual_count"] for item in scored).items())),
        "n_selected_over_two": sum(item["selected_count"] > 2 for item in scored),
        "n_actual_over_two": sum(item["actual_count"] > 2 for item in scored),
    }


def _paired_summary(rows, tier, candidate, reference):
    paired = [(row, left, right) for row in rows
              if (left := _item(row, tier, candidate)) and left["status"] == "scored"
              and (right := _item(row, tier, reference)) and right["status"] == "scored"]
    n = len(rows)
    result = {"n_scheduled": n, "n_paired": len(paired),
              "n_failed": n - len(paired),
              "complete_grid": n > 0 and len(paired) == n,
              "coverage": len(paired) / n if n else 0.0,
              "difference_direction": "candidate minus reference; positive favors candidate for hits, negative for MAE"}
    for field, label in (("exact_set_hit", "exact_set_hit_rate_difference"),
                         ("hit_count", "mean_hit_count_difference"),
                         ("volatility_mae", "volatility_mae_difference"),
                         ("max_drawdown_mae", "max_drawdown_mae_difference"),
                         ("volatility_top3_recall", "volatility_top3_recall_difference")):
        values = [(row["utc_day"], float(left[field]) - float(right[field]))
                  for row, left, right in paired
                  if left[field] is not None and right[field] is not None]
        result[label] = _paired_interval(values)
        result[label]["n_paired"] = len(values)
        result[label]["candidate_mean"] = (mean(float(left[field]) for _, left, right in paired
                                                if left[field] is not None and right[field] is not None)
                                            if values else None)
        result[label]["reference_mean"] = (mean(float(right[field]) for _, left, right in paired
                                                if left[field] is not None and right[field] is not None)
                                            if values else None)
    return result


def summarize(rows, model_names=None):
    """Return tier reports and paired comparisons, keeping the scheduled grid.

Selection metric: result['tiers']['small']['methods'][epoch]['volatility_mae'].
When model_names is given it also registers wholly missing models as failures.
It never changes the baseline universe. Intervals are descriptive percentile
day-cluster bootstrap intervals; passing uses frozen point-estimate rules.
"""
    rows = list(rows)
    seen_ids, seen_times = set(), set()
    fixed_pair = None
    for row in rows:
        if row.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("score protocol differs from v2.1; do not mix earlier experiment rows")
        pair = _fixed_pair(row.get("fixed_pair"))
        if fixed_pair is not None and pair != fixed_pair:
            raise ValueError("fixed_pair differs across scheduled windows")
        fixed_pair = pair
        fixed_score = _item(row, "small", "fixed_pair")
        if (fixed_score is not None and fixed_score["status"] == "scored"
                and _fixed_pair(fixed_score["selected_set"]) != pair):
            raise ValueError("fixed_pair score disagrees with frozen pair metadata")
        timestamp = _timestamp(row["as_of"])
        if row["window_id"] in seen_ids or timestamp in seen_times:
            raise ValueError("duplicate scheduled window identity or origin")
        if row["utc_day"] != timestamp.date().isoformat():
            raise ValueError("UTC day must match forecast origin")
        seen_ids.add(row["window_id"])
        seen_times.add(timestamp)
    models = set(model_names or ())
    for row in rows:
        for tier in TIERS:
            models.update(name for name in row["tiers"][tier]["methods"] if name not in BASELINES)
    if any(not isinstance(name, str) or not name or name in BASELINES for name in models):
        raise ValueError("invalid model name")
    report = {"protocol_version": PROTOCOL_VERSION, "n_scheduled": len(rows),
              "fixed_pair": list(fixed_pair) if fixed_pair is not None else None,
              "major_tolerance": MAJOR_TOLERANCE,
              "model_names": sorted(models), "tiers": {},
              "observation_addendum": {
                  "metrics": ["max_drawdown_mae", "volatility_mae", "volatility_top3_recall"],
                  "new_metrics_role": "max_drawdown_mae and volatility_top3_recall are log-only; selection and gates unchanged",
                  "drawdown_definition": "P0 and future close running-peak maximum drawdown; mean of pathwise scalars",
                  "top3_tie_policy": "uniform cutoff inclusion weights; dot(predicted,actual)/min(3,N); flat vectors yield chance min(3,N)/N",
                  "baseline_availability": "dynamic_naive has volatility ranking but no MDD path; fixed_pair has neither scalar MDD nor top3 ranking"},
              "sample_unit": "forecast origin; not assets or paths",
              "ranking_policy": "competition rank <=2; ties retained",
              "intervals": {"method": "paired UTC-origin-day cluster bootstrap",
                            "confidence": .95, "replicates": BOOTSTRAP_REPLICATES,
                            "seed": BOOTSTRAP_SEED},
              "interpretation": "Point-estimate acceptance only; no claim of statistical superiority or noninferiority.",
              "failure_policy": "Incomplete scheduled paired grid prevents acceptance."}
    for tier, assets in TIERS.items():
        baselines = ["dynamic_naive"] + (["fixed_pair"] if tier == "small" else [])
        names = baselines + sorted(models)
        references = baselines + (["original"] if "original" in models else [])
        report["tiers"][tier] = {
            "assets": list(assets), "n_scheduled": len(rows),
            "methods": {name: _method_summary(rows, tier, name) for name in names},
            "paired": {candidate: {reference: _paired_summary(rows, tier, candidate, reference)
                                   for reference in references if reference != candidate}
                       for candidate in sorted(models)}}
    return report


def gates(report, candidate, reference="original"):
    """Apply the frozen complete-grid point-estimate gates, with no new tuning."""
    checks, reasons = {}, []
    specifications = [("small", baseline, True)
                      for baseline in ("dynamic_naive", "fixed_pair", reference)]
    specifications.append(("major", "dynamic_naive", False))
    for tier, baseline, strict in specifications:
        key = f"{tier}_vs_{baseline}"
        paired = report["tiers"][tier].get("paired", {}).get(candidate, {}).get(baseline)
        if paired is None:
            checks[key] = {"passed": False, "reason": "missing paired comparison"}
            reasons.append(f"{key}: missing paired comparison")
            continue
        difference = paired["exact_set_hit_rate_difference"]["estimate"]
        complete = paired["complete_grid"] and paired["n_scheduled"] == report["n_scheduled"]
        threshold = 0.0 if strict else -MAJOR_TOLERANCE
        better = difference is not None and (difference > threshold if strict else difference >= threshold)
        passed = complete and better
        reason = ("passed" if passed else "incomplete paired grid" if not complete
                  else "point estimate does not meet frozen threshold")
        checks[key] = {"passed": passed, "reason": reason,
                       "n_scheduled": paired["n_scheduled"], "n_paired": paired["n_paired"],
                       "exact_set_hit_rate_difference": difference,
                       "threshold": threshold,
                       "rule": "strictly > 0" if strict else ">= -0.02 (minus 2 percentage points)"}
        if not passed:
            reasons.append(f"{key}: {reason}")
    return {"passed": all(check["passed"] for check in checks.values()),
            "candidate": candidate, "reference": reference, "checks": checks,
            "reasons": reasons,
            "interpretation": "Point-estimate engineering gate; intervals are reported, not additional pass thresholds."}
