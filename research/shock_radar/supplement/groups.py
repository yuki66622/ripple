"""Pure frozen tier/strength/systemic cuts; no file, model, or network access.

grouped(rows) returns per_event plus summaries[axis][group]. Events are weighted
equally after their within-group asset errors are averaged. The three axes
overlap; their event counts must not be pooled as independent observations.
"""
from __future__ import annotations

from collections import Counter
from math import isfinite
from numbers import Real
from statistics import mean

from ..metrics import rank_scores, risk_metrics

ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
METHODS = ["kronos_base", "no_propagation", "btc_beta", "historical_30"]
TIERS = {"big": ASSETS[:4], "small": ASSETS[4:]}
GROUPS = {"tier": ["big", "small"], "strength": ["6_to_8", "ge_8"],
          "systemic": ["systemic", "non_systemic"]}


def _validate_row(row):
    if not isinstance(row, dict) or not isinstance(row.get("event_id"), str) or not row["event_id"]:
        raise ValueError("Each row needs a nonempty event_id")
    if row.get("assets") != ASSETS:
        raise ValueError("The frozen ordered ten-asset universe is required")
    if not isinstance(row.get("timestamp"), str) or not row["timestamp"]:
        raise ValueError("Each row needs its saved timestamp")
    systemic = row.get("systemic_flag")
    if type(systemic) is not bool:
        raise ValueError("systemic_flag must be a boolean posthoc label")
    magnitude = row.get("magnitude_sigma")
    if isinstance(magnitude, bool) or not isinstance(magnitude, Real) or not isfinite(magnitude) or magnitude < 6:
        raise ValueError("Frozen strength groups require finite magnitude_sigma >= 6")
    origins = row.get("origin_assets")
    if (not isinstance(origins, list) or not origins or len(set(origins)) != len(origins)
            or any(a not in ASSETS for a in origins)):
        raise ValueError("Complete distinct original origin_assets are required")
    expected = ASSETS if systemic else [a for a in ASSETS if a not in origins]
    if row.get("scored_assets") != expected:
        raise ValueError("scored_assets must retain the original scoring universe and order")
    saved = row.get("saved_score")
    if not isinstance(saved, dict) or saved.get("scored_assets") != expected:
        raise ValueError("saved_score.scored_assets must bind the original scoring universe")
    if not isinstance(row.get("spots"), dict) or set(row["spots"]) != set(ASSETS):
        raise ValueError("Every asset spot is required")
    paths = row.get("paths")
    if not isinstance(paths, dict) or set(paths) != set(METHODS):
        raise ValueError("All four frozen methods must exist; failures cannot be silently omitted")
    actual = row.get("actual")
    if not isinstance(actual, dict) or set(actual) != set(ASSETS):
        raise ValueError("Every asset actual path is required")
    for label, table in [("actual", actual), *paths.items()]:
        if not isinstance(table, dict) or set(table) != set(ASSETS):
            raise ValueError(f"{label}: every declared asset path is required")
        if any(not isinstance(path, (list, tuple)) or len(path) != 30 for path in table.values()):
            raise ValueError(f"{label}: exactly 30 future closes required")
    # Existing numerical engine rejects nonpositive/nonfinite prices and spots.
    actual_risk = {a: risk_metrics(row["spots"][a], actual[a]) for a in ASSETS}
    predicted = {m: {a: risk_metrics(row["spots"][a], paths[m][a]) for a in ASSETS} for m in METHODS}
    return actual_risk, predicted


def _event_group(assets, actual, predicted):
    count = len(assets)
    reason = "empty_scored_assets" if not count else "fewer_than_three_scored_assets" if count < 3 else None
    methods = {}
    for method in METHODS:
        mae = mean(abs(predicted[method][a]["max_drawdown"] - actual[a]["max_drawdown"])
                   for a in assets) if count else None
        ranking = (rank_scores([predicted[method][a]["volatility"] for a in assets],
                               [actual[a]["volatility"] for a in assets], k=3) if count >= 3 else None)
        methods[method] = {
            "max_drawdown_mae": mae, "max_drawdown_mae_bps": None if mae is None else mae * 10000,
            "volatility_top3_recall_expected": ranking["top3_recall_expected"] if ranking else None,
        }
    return {"scored_assets": list(assets), "asset_count": count,
            "mae_defined": bool(count), "mae_null_reason": None if count else "empty_scored_assets",
            "top3_defined": count >= 3, "top3_null_reason": reason,
            "chance_recall": 3 / count if count >= 3 else None, "methods": methods}


def _summarize(entries):
    mae_entries = [entry for entry in entries if entry["mae_defined"]]
    top_entries = [entry for entry in entries if entry["top3_defined"]]
    return {
        "event_count": len(entries), "mae_event_count": len(mae_entries),
        "mae_null_count": len(entries) - len(mae_entries),
        "top3_defined_count": len(top_entries), "top3_null_count": len(entries) - len(top_entries),
        "top3_null_reasons": dict(Counter(e["top3_null_reason"] for e in entries if not e["top3_defined"])),
        "chance_recall_mean": mean(e["chance_recall"] for e in top_entries) if top_entries else None,
        "asset_count_distribution": dict(Counter(str(e["asset_count"]) for e in entries)),
        "methods": {method: {
            "max_drawdown_mae": mean(e["methods"][method]["max_drawdown_mae"] for e in mae_entries) if mae_entries else None,
            "max_drawdown_mae_bps": mean(e["methods"][method]["max_drawdown_mae_bps"] for e in mae_entries) if mae_entries else None,
            "volatility_top3_recall_expected": mean(e["methods"][method]["volatility_top3_recall_expected"] for e in top_entries) if top_entries else None,
        } for method in METHODS},
    }


def grouped(rows):
    """Return independent group axes with immutable original scoring membership.

    summaries[axis][group] has event_count, mae_event_count/null_count,
    top3_defined_count/null_count/null_reasons, chance_recall_mean, and methods.
    Each methods entry has max_drawdown_mae, max_drawdown_mae_bps and
    volatility_top3_recall_expected. per_event exposes every denominator.
    Missing/malformed methods raise ValueError instead of reducing the cohort.
    """
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a sequence of frozen in-memory event records")
    collected = {axis: {group: [] for group in groups} for axis, groups in GROUPS.items()}
    per_event, seen = [], set()
    for row in rows:
        actual, predicted = _validate_row(row)
        event_id = row["event_id"]
        if event_id in seen:
            raise ValueError("Duplicate event_id cannot count twice")
        seen.add(event_id)
        scored = row["scored_assets"]
        strength = "6_to_8" if row["magnitude_sigma"] < 8 else "ge_8"
        systemic = "systemic" if row["systemic_flag"] else "non_systemic"
        event_groups = {"tier": {name: _event_group([a for a in scored if a in members], actual, predicted)
                                 for name, members in TIERS.items()},
                        "strength": {strength: _event_group(scored, actual, predicted)},
                        "systemic": {systemic: _event_group(scored, actual, predicted)}}
        for axis, groups in event_groups.items():
            for name, entry in groups.items():
                collected[axis][name].append(entry)
        per_event.append({"event_id": event_id, "timestamp": row["timestamp"],
                          "magnitude_sigma": float(row["magnitude_sigma"]),
                          "systemic_flag": row["systemic_flag"], "origin_assets": list(row["origin_assets"]),
                          "original_scored_assets": list(scored), "groups": event_groups})
    return {
        "event_count": len(rows), "methods": list(METHODS), "per_event": per_event,
        "summaries": {axis: {name: _summarize(entries) for name, entries in groups.items()}
                      for axis, groups in collected.items()},
        "definitions": {
            "tier_assets": {name: list(assets) for name, assets in TIERS.items()},
            "strength": {"6_to_8": "6 <= initial magnitude_sigma < 8", "ge_8": "initial magnitude_sigma >= 8"},
            "systemic": "Original posthoc true/false label; false can include simultaneous origins.",
            "scoring": "Tier intersects original scored_assets; other cuts retain that whole original universe.",
            "aggregation": "Within-event asset MAE then equal-weight events; four methods use identical sets.",
            "top3": "Original uniform cutoff-tie expected recall; k remains 3. N<3 is null, never smaller k.",
            "chance": "Mean 3/N among Top3-defined events; not a simulated baseline or full-set accuracy.",
            "overlap": "Axes overlap; their event counts are not additional independent observations.",
            "empty": "Empty groups and empty event-tier intersections are visible; estimates are null, not zero.",
        },
    }
