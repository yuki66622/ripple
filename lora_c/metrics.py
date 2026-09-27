"""Pure C-line scoring; no filesystem, market-data, or model access.

Rows preserve every scheduled origin and failed method. MDD is the close-only
running-peak maximum drawdown including P0, computed per forecast path before
averaging. ``historical_30`` persists the MDD of the last 31 input closes; it has
no fabricated forecast path, volatility, or ranking. All rates use fractions.

Only small-tier MDD MAE selects a model. The per-epoch major guard is separate
from May's small-tier acceptance rule. Day-block intervals are descriptive.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from statistics import mean

from lora_a import metrics as _core  # Frozen, pure numerical/validation code only.


PROTOCOL_VERSION = "lora-c-v1"
TIERS = _core.TIERS
ASSETS = _core.ASSETS
LOOKBACK = _core.LOOKBACK
HORIZON = _core.HORIZON
BOOTSTRAP_SEED = _core.BOOTSTRAP_SEED
BOOTSTRAP_REPLICATES = _core.BOOTSTRAP_REPLICATES
BASELINES = ("dynamic_naive", "historical_30")
REQUIRED_ORIGINS = 300
MAJOR_DRAWDOWN_TOLERANCE = .2
_ERRORS = (ValueError, TypeError, KeyError, IndexError, OverflowError)
_FIELDS = {
    "exact_set_hit": "exact_set_hit_rate_difference",
    "hit_count": "mean_hit_count_difference",
    "volatility_mae": "volatility_mae_difference",
    "max_drawdown_mae": "max_drawdown_mae_difference",
    "volatility_top3_recall": "volatility_top3_recall_difference",
}


def _historical_drawdown(window):
    result = {}
    for asset in ASSETS:
        closes = [_core._positive(bar["close"])
                  for bar in window["histories"][asset][-31:]]
        if len(closes) != 31:
            raise ValueError("historical_30 requires 31 input closes")
        peak, drawdown = closes[0], 0.0
        for close in closes[1:]:
            peak = max(peak, close)
            drawdown = max(drawdown, (peak - close) / peak)
        result[asset] = drawdown
    return result


def _drawdown_score(actual, predicted, actual_count):
    errors = {asset: abs(predicted[asset] - actual[asset]) for asset in actual}
    item = _core._failed(None, actual_count)
    item.update(status="scored", max_drawdown_mae=mean(errors.values()),
                max_drawdown_absolute_errors=errors, predicted_max_drawdown=predicted,
                history_close_count=31)
    return item


def score_window(window, future, predictions):
    """Score a MarketWindow v1 with direct or wrapped truth and adapter forecasts.

``predictions`` maps model names to forecasts or None; baseline names are
reserved. Invalid identity raises. Invalid history/truth/predictions are kept
as failed rows. Historical baseline construction never depends on future data.
"""
    window_id = window["window_id"]
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("nonempty window_id required")
    anchor = _core._timestamp(window["as_of"])
    if any(not isinstance(name, str) or not name or name in BASELINES
           for name in predictions):
        raise ValueError("model names must be nonempty and cannot shadow baselines")
    row = {"protocol_version": PROTOCOL_VERSION, "window_id": window_id,
           "as_of": window["as_of"], "utc_day": anchor.date().isoformat(),
           "truth_status": "failed", "failures": [], "tiers": {}}
    try:
        spots, naive = _core._history(window, anchor)
        historical = _historical_drawdown(window)
        actual, actual_drawdown = _core._actual(window, future, anchor, spots)
    except _ERRORS as exc:
        failure = _core._error(exc)
        row["failures"].append({"method": "truth_or_history", **failure})
        for tier in TIERS:
            row["tiers"][tier] = {
                "actual_volatility": None, "actual_max_drawdown": None,
                "actual_set": None, "actual_count": None,
                "methods": {name: _core._failed(failure)
                            for name in (*BASELINES, *predictions)}}
        return row
    row["truth_status"] = "scored"
    for tier, assets in TIERS.items():
        truth = {asset: actual[asset] for asset in assets}
        drawdown = {asset: actual_drawdown[asset] for asset in assets}
        actual_set = _core._top_two(truth)
        row["tiers"][tier] = {
            "actual_volatility": truth, "actual_max_drawdown": drawdown,
            "actual_set": actual_set, "actual_count": len(actual_set),
            "methods": {
                "dynamic_naive": _core._score(
                    truth, predicted={asset: naive[asset] for asset in assets}),
                "historical_30": _drawdown_score(
                    drawdown, {asset: historical[asset] for asset in assets}, len(actual_set)),
            }}
    for name, value in predictions.items():
        try:
            volatility, drawdown, path_count = _core._prediction(window, value, anchor, spots)
            for tier, assets in TIERS.items():
                item = _core._score(
                    {a: actual[a] for a in assets},
                    predicted={a: volatility[a] for a in assets},
                    actual_drawdown={a: actual_drawdown[a] for a in assets},
                    predicted_drawdown={a: drawdown[a] for a in assets})
                item["path_count"] = path_count
                row["tiers"][tier]["methods"][name] = item
        except _ERRORS as exc:
            failure = _core._error(exc)
            row["failures"].append({"method": name, **failure})
            for tier in TIERS:
                row["tiers"][tier]["methods"][name] = _core._failed(
                    failure, row["tiers"][tier]["actual_count"])
    return row


def merge_rows(*batches):
    """Union cached methods on identical evidence, retaining unpaired origins."""
    merged = {}
    for batch in batches:
        seen = set()
        for row in batch:
            key = row["window_id"]
            if key in seen:
                raise ValueError("duplicate window_id within score batch")
            seen.add(key)
            if row.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("C scoring protocol required")
            if key not in merged:
                merged[key] = copy.deepcopy(row)
                continue
            existing = merged[key]
            for field in ("as_of", "utc_day", "truth_status", "protocol_version"):
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
    return sorted(merged.values(), key=lambda row: (_core._timestamp(row["as_of"]), row["window_id"]))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _item(row, tier, method):
    return row["tiers"][tier]["methods"].get(method)


def _metric(item, field, assets):
    if not item or item.get("status") != "scored":
        return None
    value = item.get(field)
    if field == "exact_set_hit":
        return float(value) if isinstance(value, bool) else None
    if not _finite(value) or value < 0:
        return None
    if field in ("volatility_mae", "max_drawdown_mae"):
        error_field = "absolute_errors" if field == "volatility_mae" else "max_drawdown_absolute_errors"
        errors = item.get(error_field)
        if (not isinstance(errors, dict) or set(errors) != set(assets)
                or any(not _finite(v) or v < 0 for v in errors.values())
                or not math.isclose(value, mean(errors.values()), rel_tol=1e-12, abs_tol=1e-15)):
            return None
    return float(value)


def _method_summary(rows, tier, method):
    assets, n = TIERS[tier], len(rows)
    scored = [item for row in rows if (item := _item(row, tier, method))
              and item.get("status") == "scored"]
    values = {field: [value for item in scored
                      if (value := _metric(item, field, assets)) is not None]
              for field in _FIELDS}
    ranking = [item for item in scored if _metric(item, "exact_set_hit", assets) is not None]
    result = {
        "n_scheduled": n, "n_scored": len(scored), "n_failed": n - len(scored),
        "n_missing": sum(_item(row, tier, method) is None for row in rows),
        "coverage": len(scored) / n if n else 0.0,
        "exact_set_hits": int(sum(values["exact_set_hit"])) if values["exact_set_hit"] else None,
        "exact_set_hit_rate": mean(values["exact_set_hit"]) if values["exact_set_hit"] else None,
        "mean_hit_count": mean(values["hit_count"]) if values["hit_count"] else None,
        "n_ranking_origins": len(values["exact_set_hit"]),
        "volatility_top3_recall": mean(values["volatility_top3_recall"]) if values["volatility_top3_recall"] else None,
        "n_volatility_top3_recall_origins": len(values["volatility_top3_recall"]),
        "selected_count_histogram": dict(sorted(Counter(item["selected_count"] for item in ranking).items())),
        "actual_count_histogram": dict(sorted(Counter(item["actual_count"] for item in scored
                                                     if item.get("actual_count") is not None).items())),
        "n_selected_over_two": sum(item["selected_count"] > 2 for item in ranking),
        "n_actual_over_two": sum(item.get("actual_count", 0) > 2 for item in scored
                                 if item.get("actual_count") is not None),
        "mae_sample_unit": "asset-origin; equal asset mean within origin, then equal origin mean",
        "metric_availability": {"volatility": method != "historical_30",
                                "ranking": method != "historical_30",
                                "max_drawdown": method != "dynamic_naive"},
    }
    for field, errors_field, prefix, asset_label, available in (
        ("volatility_mae", "absolute_errors", "n_mae", "asset_volatility_mae", method != "historical_30"),
        ("max_drawdown_mae", "max_drawdown_absolute_errors", "n_max_drawdown_mae", "asset_max_drawdown_mae", method != "dynamic_naive"),
    ):
        valid = [item for item in scored if _metric(item, field, assets) is not None]
        count = len(valid)
        result[field] = mean(values[field]) if values[field] else None
        result[f"{prefix}_origins"] = count
        result[f"{prefix}_origins_scheduled"] = n if available else 0
        result[f"{prefix}_asset_origins"] = count * len(assets)
        result[f"{prefix}_asset_origins_scheduled"] = n * len(assets) if available else 0
        result[f"{prefix}_coverage"] = count / n if n and available else None
        result[asset_label] = {asset: mean(item[errors_field][asset] for item in valid)
                               for asset in assets} if valid else {}
    return result


def _paired_summary(rows, tier, candidate, reference):
    assets, n = TIERS[tier], len(rows)
    paired = [(row, left, right) for row in rows
              if (left := _item(row, tier, candidate)) and left.get("status") == "scored"
              and (right := _item(row, tier, reference)) and right.get("status") == "scored"]
    result = {"n_scheduled": n, "n_paired": len(paired), "n_failed": n - len(paired),
              "complete_grid": n > 0 and len(paired) == n,
              "coverage": len(paired) / n if n else 0.0,
              "difference_direction": "candidate minus reference; negative favors candidate for MAE"}
    for field, label in _FIELDS.items():
        values = [(row["utc_day"], lv, rv) for row, left, right in paired
                  if (lv := _metric(left, field, assets)) is not None
                  and (rv := _metric(right, field, assets)) is not None]
        stats = _core._paired_interval([(day, lv - rv) for day, lv, rv in values])
        stats.update(n_paired=len(values), n_scheduled=n, n_unscored=n - len(values),
                     complete_grid=n > 0 and len(values) == n,
                     candidate_mean=mean(v[1] for v in values) if values else None,
                     reference_mean=mean(v[2] for v in values) if values else None)
        result[label] = stats
    return result


def summarize(rows, model_names=None):
    """Keep the scheduled grid; missing/null metric values get explicit coverage.

Selection reads tiers.small.methods[candidate].max_drawdown_mae, but the caller
must also pass selection_gate and retain every prior epoch's major guard.
model_names registers even completely missing models. No baseline is selected
or tuned here. This function accepts only C-protocol score rows.
"""
    rows = list(rows)
    seen_ids, seen_times = set(), set()
    models = set(model_names or ())
    for row in rows:
        if row.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("C scoring protocol required; do not mix experiment rows")
        timestamp = _core._timestamp(row["as_of"])
        if row["window_id"] in seen_ids or timestamp in seen_times:
            raise ValueError("duplicate scheduled window identity or origin")
        if row["utc_day"] != timestamp.date().isoformat():
            raise ValueError("UTC day must match forecast origin")
        seen_ids.add(row["window_id"])
        seen_times.add(timestamp)
        for tier in TIERS:
            models.update(name for name in row["tiers"][tier]["methods"] if name not in BASELINES)
    if any(not isinstance(name, str) or not name or name in BASELINES for name in models):
        raise ValueError("invalid model name")
    report = {
        "protocol_version": PROTOCOL_VERSION, "n_scheduled": len(rows),
        "required_origins": REQUIRED_ORIGINS, "model_names": sorted(models), "tiers": {},
        "selection_metric": "small.max_drawdown_mae",
        "sample_unit": "forecast origin; not assets or paths",
        "ranking_policy": "competition rank <=2; ties retained; log-only",
        "historical_30_definition": "MDD of last31 input closes (30 returns), persisted as a scalar; no future data",
        "drawdown_definition": "P0 and future close running-peak MDD; mean of pathwise scalars",
        "top3_tie_policy": "uniform cutoff inclusion weights; dot(predicted,actual)/min(3,N); log-only",
        "major_drawdown_tolerance": MAJOR_DRAWDOWN_TOLERANCE,
        "intervals": {"method": "paired UTC-origin-day cluster bootstrap", "confidence": .95,
                      "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "interpretation": "Frozen point-estimate gates; intervals are descriptive, not additional significance thresholds.",
        "failure_policy": "Retain all scheduled origins; incomplete metric-paired grid prevents its decision gate.",
    }
    names = [*BASELINES, *sorted(models)]
    references = [*BASELINES, *(["original"] if "original" in models else [])]
    for tier, assets in TIERS.items():
        report["tiers"][tier] = {
            "assets": list(assets), "n_scheduled": len(rows),
            "methods": {name: _method_summary(rows, tier, name) for name in names},
            "paired": {candidate: {reference: _paired_summary(rows, tier, candidate, reference)
                                   for reference in references if reference != candidate}
                       for candidate in sorted(models)},
        }
    return report


def _complete_mdd_pair(report, tier, candidate, reference):
    """Return finite paired means only for the complete frozen 300-origin grid."""
    result = {"passed": False, "reason": "incomplete or invalid paired MDD grid",
              "n_scheduled": report.get("n_scheduled"), "n_paired": 0,
              "candidate_mean": None, "reference_mean": None, "difference": None}
    if report.get("protocol_version") != PROTOCOL_VERSION or candidate in BASELINES or candidate == reference:
        result["reason"] = "invalid C protocol or candidate"
        return result
    tier_report = report.get("tiers", {}).get(tier, {})
    paired = tier_report.get("paired", {}).get(candidate, {}).get(reference, {})
    metric = paired.get("max_drawdown_mae_difference", {})
    result.update(n_paired=metric.get("n_paired", 0), candidate_mean=metric.get("candidate_mean"),
                  reference_mean=metric.get("reference_mean"), difference=metric.get("estimate"))
    if (report.get("n_scheduled") != REQUIRED_ORIGINS
            or tier_report.get("n_scheduled") != REQUIRED_ORIGINS
            or paired.get("n_scheduled") != REQUIRED_ORIGINS
            or paired.get("n_paired") != REQUIRED_ORIGINS or paired.get("complete_grid") is not True
            or metric.get("n_paired") != REQUIRED_ORIGINS or metric.get("complete_grid") is not True):
        return result
    for name, field in ((candidate, "candidate_mean"), (reference, "reference_mean")):
        item = tier_report.get("methods", {}).get(name, {})
        value = item.get("max_drawdown_mae")
        if (item.get("n_scheduled") != REQUIRED_ORIGINS or item.get("n_scored") != REQUIRED_ORIGINS
                or item.get("n_failed") != 0 or item.get("n_missing") != 0 or item.get("coverage") != 1
                or item.get("n_max_drawdown_mae_origins") != REQUIRED_ORIGINS
                or item.get("n_max_drawdown_mae_asset_origins") != REQUIRED_ORIGINS * len(TIERS[tier])
                or not _finite(value) or value < 0
                or not _finite(result[field]) or result[field] < 0
                or not math.isclose(value, result[field], rel_tol=1e-12, abs_tol=1e-15)):
            return result
    if (not _finite(result["difference"])
            or not math.isclose(result["difference"], result["candidate_mean"] - result["reference_mean"],
                                rel_tol=1e-12, abs_tol=1e-15)):
        return result
    result.update(passed=True, reason="complete paired MDD grid")
    return result


def major_drawdown_guard(report, candidate):
    """Per-epoch stop guard. Exactly 20% degradation passes; greater fails.

Incomplete evidence also prevents continuation but is not called observed
degradation. With a zero reference, relative_change is None in all cases.
"""
    result = _complete_mdd_pair(report, "major", candidate, "original")
    result.update(candidate=candidate, reference="original", relative_change=None,
                  threshold_relative=MAJOR_DRAWDOWN_TOLERANCE, triggered=False,
                  evidence_complete=result["passed"])
    if not result["passed"]:
        return result
    value, original = result["candidate_mean"], result["reference_mean"]
    if original == 0:
        triggered = value > 0
        reason = "positive candidate MDD MAE with zero original" if triggered else "both MDD MAEs zero"
    else:
        result["relative_change"] = (value - original) / original
        triggered = value > original * (1 + MAJOR_DRAWDOWN_TOLERANCE)
        reason = "major MDD MAE worsened by more than 20%" if triggered else "major MDD MAE within 20% tolerance"
    result.update(passed=not triggered, triggered=triggered, reason=reason)
    return result


def _strict_small_check(report, candidate, reference):
    result = _complete_mdd_pair(report, "small", candidate, reference)
    result["evidence_complete"] = result["passed"]
    result["rule"] = "candidate MDD MAE strictly lower than reference on complete300"
    if result["passed"]:
        result["passed"] = result["candidate_mean"] < result["reference_mean"]
        result["reason"] = "passed" if result["passed"] else "small MDD MAE did not strictly improve"
    return result


def _gate_result(candidate, checks):
    return {"candidate": candidate, "passed": all(v["passed"] for v in checks.values()),
            "checks": checks, "reasons": [f"{key}: {value['reason']}" for key, value in checks.items()
                                           if not value["passed"]],
            "interpretation": "Frozen point-estimate gate; CI, volatility MAE, and ranking do not affect the decision."}


def selection_gate(report, candidate):
    """April lock eligibility; runner separately retains all earlier major guards."""
    return _gate_result(candidate, {
        "small_vs_original": _strict_small_check(report, candidate, "original"),
        "major_drawdown_guard": major_drawdown_guard(report, candidate),
    })


def acceptance_gate(report, candidate):
    """May small-tier double gate. Major disclosure never adds an acceptance gate."""
    result = _gate_result(candidate, {
        f"small_vs_{reference}": _strict_small_check(report, candidate, reference)
        for reference in ("original", "historical_30")})
    result["major_drawdown_disclosure"] = major_drawdown_guard(report, candidate)
    result["major_drawdown_disclosure"]["role"] = "report only; not a May acceptance threshold"
    return result
