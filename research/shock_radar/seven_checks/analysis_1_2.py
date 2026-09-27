"""Frozen seven-checks analyses 1/2; pure run(bundle), no model/data loading.

Only the module entry point calls the shared, permission-scoped loader/writer.
Arithmetic uses the B-line cutoff tie policy. Shape is a retrospective label,
not a prediction, and model MAE always comes from the t0/30-minute cache.
"""
from __future__ import annotations

import math
from collections import Counter
from numbers import Real
from statistics import mean

from research.shock_radar.metrics import rank_scores

MONTHS = ("2026-01", "2026-02")
BASELINES = ("historical_30", "no_propagation", "btc_beta")
CLASSIFIED = ("v_reversal", "upward_false_breakout", "one_way_down", "other")
SHAPES = (*CLASSIFIED, "mixed_direction", "zero_initial_move", "incomplete_60m", "invalid_shape_input")


def _get_common():
    from . import common
    return common


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def overlap(actual_risk, assets):
    """Actual vol-versus-MDD Top3 on the FULL tier, without origin exclusions."""
    try:
        if len(assets) < 3:
            raise ValueError("at least three tier assets required")
        volatility = [actual_risk[asset]["volatility"] for asset in assets]
        drawdown = [actual_risk[asset]["max_drawdown"] for asset in assets]
        if (any(not _finite(v) or v < 0 for v in volatility)
                or any(not _finite(v) or not 0 <= v <= 1 for v in drawdown)):
            raise ValueError("finite nonnegative volatility and drawdown in [0,1] required")
        ranks = rank_scores(volatility, drawdown, k=3)
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "missing_or_invalid_risk", "reason": str(exc),
                "expected_intersection": None, "expected_recall": None,
                "volatility_cutoff_tied": None, "drawdown_cutoff_tied": None,
                "exact_overlap_eligible": False, "full_overlap": None}
    # Equal values do not make the Top3 ambiguous when their entire group fits.
    # A cutoff tie crosses k only when some, but not all, tied assets are needed.
    def crosses_cutoff(values):
        cutoff = sorted(values, reverse=True)[2]
        above = sum(value > cutoff for value in values)
        tied = sum(value == cutoff for value in values)
        return above < 3 < above + tied

    vol_tied = crosses_cutoff(volatility)
    mdd_tied = crosses_cutoff(drawdown)
    recall = ranks["top3_recall_expected"]
    eligible = not (vol_tied or mdd_tied)
    return {"status": "scored", "reason": None,
            "expected_intersection": 3 * recall, "expected_recall": recall,
            "volatility_cutoff_tied": vol_tied, "drawdown_cutoff_tied": mdd_tied,
            "volatility_cutoff_tie_count": ranks["prediction_cutoff_tie_count"],
            "drawdown_cutoff_tie_count": ranks["actual_cutoff_tie_count"],
            "exact_overlap_eligible": eligible, "full_overlap": recall == 1 if eligible else None}


def classify_log_path(y, direction):
    """Classify an explicit 61-point origin-composite log path by frozen order."""
    if len(y) != 61 or any(not _finite(value) for value in y):
        return {"shape": "invalid_shape_input", "reason": "61 finite log-price points required"}
    source_direction = direction
    if _finite(direction) and direction in (-1, 1):
        direction = "up" if direction == 1 else "down"
    if direction not in ("up", "down", "mixed"):
        return {"shape": "invalid_shape_input", "reason": "direction must be numeric +1/-1 or mixed"}
    y = [float(value) for value in y]
    s = abs(y[0])
    first_trough = y.index(min(y))
    variation = math.fsum(abs(b - a) for a, b in zip(y, y[1:]))
    efficiency = (y[0] - y[60]) / variation if variation > 0 else None
    result = {"shape": None, "reason": None, "direction": direction, "source_direction": source_direction,
              "initial_log_move": y[0], "initial_absolute_move": s,
              "final_log_move": y[60], "postshock_log_change": y[60] - y[0],
              "first_trough_minute": first_trough, "total_postshock_variation": variation,
              "downward_efficiency": efficiency, "log_path_0_60": y,
              "mixed_direction": direction == "mixed", "zero_initial_move": s == 0}
    if direction == "mixed":
        result.update(shape="mixed_direction", reason="mixed shock origins are reported separately")
    elif s == 0:
        result.update(shape="zero_initial_move", reason="initial composite log move equals zero")
    elif direction == "down" and first_trough <= 30 and y[60] >= -.2 * s:
        result["shape"] = "v_reversal"
    elif direction == "up" and y[60] <= .2 * s:
        result["shape"] = "upward_false_breakout"
    elif y[60] - y[0] <= -.5 * s and efficiency is not None and efficiency >= .5:
        result["shape"] = "one_way_down"
    else:
        result["shape"] = "other"
    return result


def classify_shape(closes, index, origin_assets, assets, direction):
    """Use every tied origin equally in log space; never substitute an asset.

Caller supplies the chronological table and t0 index. A missing reference or
any of the full 60 future minutes is retained as incomplete_60m. A table may
span the authorized January/February boundary, but never a missing time gap.
"""
    if (isinstance(index, bool) or not isinstance(index, int)
            or not isinstance(origin_assets, (list, tuple)) or not origin_assets
            or any(not isinstance(asset, str) for asset in origin_assets)
            or len(set(origin_assets)) != len(origin_assets)
            or any(asset not in assets for asset in origin_assets)):
        return {"shape": "invalid_shape_input", "reason": "valid t0 index and unique known origin assets required"}
    if index - 5 < 0 or index + 60 >= len(closes):
        return {"shape": "incomplete_60m", "reason": "t0-5 through t0+60 unavailable",
                "missing_reference": index - 5 < 0, "missing_future60": index + 60 >= len(closes)}
    columns = [list(assets).index(asset) for asset in origin_assets]
    try:
        references = [closes[index - 5][column] for column in columns]
        paths = [[closes[index + h][column] for h in range(61)] for column in columns]
        if any(not _finite(value) or value <= 0 for value in references):
            raise ValueError("positive finite origin reference close required")
        if any(not _finite(value) or value <= 0 for path in paths for value in path):
            raise ValueError("positive finite origin path close required")
        y = [mean(math.log(float(path[h]) / float(reference)) for path, reference in zip(paths, references))
             for h in range(61)]
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        return {"shape": "invalid_shape_input", "reason": str(exc)}
    result = classify_log_path(y, direction)
    result["origin_assets"] = list(origin_assets)
    result["origin_reference_closes"] = dict(zip(origin_assets, map(float, references)))
    result["origin_weight"] = "equal log-price weight across all tied origins"
    return result


def _shape_event(event, bundle, assets):
    # Combined authorized calendar makes January-to-February context usable;
    # event grouping still uses t0 month. No implicit same-month restriction.
    if "combined" in bundle and "global_index" in event:
        table, index = bundle["combined"]["closes"], event["global_index"]
    else:
        table, index = bundle["monthly"][event["month"]]["closes"], event["index"]
    return classify_shape(table, index, event.get("origin_assets"), assets, event.get("direction"))


def _mae(event, tier, method):
    record = event.get("metrics", {}).get(tier, {}).get(method)
    value = record.get("mdd_mae") if isinstance(record, dict) else None
    return float(value) if _finite(value) and value >= 0 else None


def _overlap_summary(events, rows, assets, common):
    intersections = [row["expected_intersection"] for row in rows]
    recalls = [row["expected_recall"] for row in rows]
    valid = [row for row in rows if row["status"] == "scored"]
    eligible = [row for row in valid if row["exact_overlap_eligible"]]
    return {"assets": list(assets), "asset_universe": "complete tier, including origins",
            "n_expected": len(events), "n_scored": len(valid), "n_missing": len(events) - len(valid),
            "expected_intersection": common.mean_ci(intersections, events, adjusted=False),
            "expected_recall": common.mean_ci(recalls, events, adjusted=False),
            "chance_intersection": 9 / len(assets), "chance_recall": 3 / len(assets),
            "full_overlap_count": sum(row["full_overlap"] for row in eligible),
            "full_overlap_eligible_count": len(eligible),
            "full_overlap_rate_among_eligible": mean(row["full_overlap"] for row in eligible) if eligible else None,
            "any_cutoff_tie_count": sum(row["volatility_cutoff_tied"] or row["drawdown_cutoff_tied"] for row in valid),
            "volatility_cutoff_tie_count": sum(row["volatility_cutoff_tied"] for row in valid),
            "drawdown_cutoff_tie_count": sum(row["drawdown_cutoff_tied"] for row in valid),
            "status_counts": dict(Counter(row["status"] for row in rows))}


def _shape_summary(events, tier, methods, model, assets, common):
    model_values = [_mae(event, tier, model) for event in events]
    summaries, comparisons = {}, {}
    for method in methods:
        values = [_mae(event, tier, method) for event in events]
        summaries[method] = common.mean_ci(values, events, adjusted=False)
        if method in BASELINES:
            diffs = [left - right if left is not None and right is not None else None
                     for left, right in zip(model_values, values)]
            paired = [(left, right) for left, right in zip(model_values, values)
                      if left is not None and right is not None]
            comparisons[method] = {**common.mean_ci(diffs, events, adjusted=False),
                                   "model_mean_on_pairs": mean(p[0] for p in paired) if paired else None,
                                   "baseline_mean_on_pairs": mean(p[1] for p in paired) if paired else None,
                                   "difference_direction": "model minus baseline; lower MDD MAE is better"}
    asset_counts = Counter(sum(asset in event.get("scored_assets", ()) for asset in assets) for event in events)
    return {"n_expected": len(events), "n_days": len({event["day"] for event in events}),
            "scored_tier_asset_count_distribution": dict(sorted(asset_counts.items())),
            "methods": summaries, "paired_model_minus_baseline": comparisons,
            "aggregation": "frozen scored_assets intersect tier; asset MAE first, equal event mean second"}


def run(bundle):
    """Return both analyses from validated, already-loaded inputs; no data IO."""
    common = _get_common()
    assets, tiers, methods = list(common.ASSETS), common.TIERS, list(common.METHODS)
    models = [method for method in methods if method not in BASELINES]
    if len(models) != 1 or not set(BASELINES).issubset(methods):
        raise ValueError("exactly one cached model and all three frozen baselines required")
    model = models[0]
    events = list(bundle["events"])
    if any(event.get("month") not in MONTHS for event in events):
        raise ValueError("analyses 1/2 accept only authorized January/February events")
    identifiers = [(event["month"], event["event_id"]) for event in events]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate event identities cannot be counted twice")
    shapes = [_shape_event(event, bundle, assets) for event in events]
    overlaps = {tier: [overlap(event.get("actual_risk"), members) for event in events]
                for tier, members in tiers.items()}
    result = {"analysis_1": {"groups": {}, "interpretation": "actual risk ranking alignment, not predictive skill",
                             "tie_policy": "B rank_scores uniform cutoff inclusion; ambiguous cutoff ties require partial inclusion, not a wholly included equal-value group; no Jaccard or A competition rank"},
              "analysis_2": {"groups": {}, "model": model, "shape_categories": list(SHAPES),
                             "interpretation": "60-minute retrospective shape; MAE evaluates cached t0 30-minute forecasts",
                             "directional_definition": "upward false breakout only; downward recovery is V reversal",
                             "point_estimate_order_is_not_expert_claim": True},
              "event_evidence": []}
    for month in (*MONTHS, "combined"):
        indices = [i for i, event in enumerate(events) if month == "combined" or event["month"] == month]
        subset = [events[i] for i in indices]
        result["analysis_1"]["groups"][month] = {"n_expected": len(subset), "tiers": {
            tier: _overlap_summary(subset, [overlaps[tier][i] for i in indices], members, common)
            for tier, members in tiers.items()}}
        counts = {shape: sum(shapes[i]["shape"] == shape for i in indices) for shape in SHAPES}
        shape_groups = {tier: {shape: _shape_summary(
            [events[i] for i in indices if shapes[i]["shape"] == shape], tier, methods, model, members, common)
            for shape in SHAPES} for tier, members in tiers.items()}
        result["analysis_2"]["groups"][month] = {
            "n_expected": len(subset), "shape_counts": counts,
            "n_classified": sum(counts[shape] for shape in CLASSIFIED),
            "n_unclassified": sum(counts[shape] for shape in SHAPES if shape not in CLASSIFIED),
            "tiers": shape_groups,
            "model_shape_point_estimate_order": {tier: sorted(
                [{"shape": shape, "estimate": group["methods"][model]["estimate"],
                  "n": group["methods"][model]["n"],
                  "stability_eligible": group["methods"][model]["stability_eligible"]}
                 for shape, group in shape_groups[tier].items()
                 if shape in CLASSIFIED and group["methods"][model]["estimate"] is not None],
                key=lambda item: (item["estimate"], CLASSIFIED.index(item["shape"]))) for tier in tiers}}
    for i, event in enumerate(events):
        result["event_evidence"].append({"event_id": event["event_id"], "month": event["month"],
                                         "timestamp": event["timestamp"], "day": event["day"],
                                         "origin_assets": list(event["origin_assets"]),
                                         "shape": shapes[i], "overlap": {tier: overlaps[tier][i] for tier in tiers}})
    return result


if __name__ == "__main__":
    common = _get_common()
    bundle = common.load()
    common.write_result("result_1_2.json", run(bundle), bundle)
