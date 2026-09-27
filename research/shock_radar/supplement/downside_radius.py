"""Pure arithmetic for frozen shock-radar paths; no I/O or model access."""

from math import isfinite, log, sqrt
from numbers import Real
from statistics import mean

from research.shock_radar.metrics import rank_scores, risk_metrics


METHODS = ("kronos_base", "no_propagation", "btc_beta", "historical_30")
RADIUS_METHODS = ("kronos_base", "historical_30")
METRICS = ("max_drawdown", "volatility")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name}: finite positive price required")


def _validate(rows):
    """Reject incomplete frozen evidence, never silently shrink its denominator."""
    rows = list(rows)
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each row must be an event dictionary")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise ValueError("unique nonempty event_id required")
        seen.add(event_id)
        assets, scored = row.get("assets"), row.get("scored_assets")
        for name, values in (("assets", assets), ("scored_assets", scored)):
            if (not isinstance(values, (list, tuple))
                    or any(not isinstance(a, str) or not a for a in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{event_id}/{name}: ordered unique asset names required")
        if not assets or not set(scored) <= set(assets):
            raise ValueError(f"{event_id}: scored assets must be a subset of declared assets")
        spots, actual, paths = row.get("spots"), row.get("actual"), row.get("paths")
        if not isinstance(spots, dict) or set(spots) != set(assets):
            raise ValueError(f"{event_id}: all declared spots required")
        if not isinstance(paths, dict) or set(paths) != set(METHODS):
            raise ValueError(f"{event_id}: exactly four frozen methods required")
        for asset, spot in spots.items():
            _positive(spot, f"{event_id}/{asset}/spot")
        for name, table in (("actual", actual), *paths.items()):
            if not isinstance(table, dict) or set(table) != set(assets):
                raise ValueError(f"{event_id}/{name}: complete declared asset paths required")
            for asset, closes in table.items():
                if not isinstance(closes, (list, tuple)) or len(closes) != 30:
                    raise ValueError(f"{event_id}/{name}/{asset}: exactly 30 closes required")
                for value in closes:
                    _positive(value, f"{event_id}/{name}/{asset}")
    return rows


def _downside_vol(spot, closes):
    prices = [spot, *closes]
    returns = [log(b) - log(a) for a, b in zip(prices, prices[1:])]
    # sqrt(mean(min(r, 0)^2)) * sqrt(H) = sqrt(sum(min(r, 0)^2)).
    return sqrt(sum(min(value, 0.) ** 2 for value in returns))


def _mae(predicted, actual):
    return mean(abs(p - a) for p, a in zip(predicted, actual)) if actual else None


def _average(values):
    defined = [value for value in values if value is not None]
    return mean(defined) if defined else None


def downside(rows):
    """Downside volatility over all 30 returns and the original scoring universe."""
    rows = _validate(rows)
    per_event = []
    for row in rows:
        assets = row["scored_assets"]
        actual = [_downside_vol(row["spots"][a], row["actual"][a]) for a in assets]
        record = {"event_id": row["event_id"], "timestamp": row.get("timestamp"),
                  "scored_assets": list(assets), "asset_count": len(assets),
                  "actual": dict(zip(assets, actual)), "methods": {}}
        for method in METHODS:
            predicted = [_downside_vol(row["spots"][a], row["paths"][method][a]) for a in assets]
            mae = _mae(predicted, actual)
            top = rank_scores(predicted, actual) if len(assets) >= 3 else None
            record["methods"][method] = {
                "predicted": dict(zip(assets, predicted)), "mae": mae,
                "mae_bps": mae * 10000 if mae is not None else None,
                "top3_recall_expected": top["top3_recall_expected"] if top else None,
                "top3_reason": None if top else "fewer_than_three_scored_assets",
                "chance_recall": 3 / len(assets) if top else None,
            }
        per_event.append(record)
    aggregate = {"event_count": len(rows), "methods": {}}
    for method in METHODS:
        entries = [record["methods"][method] for record in per_event]
        valid = sum(entry["top3_recall_expected"] is not None for entry in entries)
        mae_count = sum(entry["mae"] is not None for entry in entries)
        aggregate["methods"][method] = {
            "event_count": len(rows), "mae_event_count": mae_count,
            "mae_null_count": len(rows) - mae_count,
            "mae": _average([entry["mae"] for entry in entries]),
            "mae_bps": _average([entry["mae_bps"] for entry in entries]),
            "top3_recall_expected": _average([entry["top3_recall_expected"] for entry in entries]),
            "top3_valid_count": valid, "top3_null_count": len(rows) - valid,
            "top3_null_reasons": ({"fewer_than_three_scored_assets": len(rows) - valid}
                                  if valid < len(rows) else {}),
            "chance_recall": _average([entry["chance_recall"] for entry in entries]),
        }
    return {"per_event": per_event, "aggregate": aggregate,
            "definition": "sqrt(sum(min(one-minute log return,0)^2)) over all 30 returns; zero target, not mean-centered or negative-count normalized",
            "aggregation": "Asset MAE within original scored_assets, then equal-weight events; same event/asset sets for all four methods. Top3 uniform cutoff-tie expected recall."}


def _segment_risk(spot, closes, offset):
    boundary = spot if offset == 0 else closes[offset - 1]
    return risk_metrics(boundary, closes[offset:offset + 10])


def _prefix_radius(advantages):
    radius_minutes = 0
    for advantage in advantages:
        if advantage is not True:
            break
        radius_minutes += 10
    return radius_minutes or None


def radius(rows):
    """Three 10-return segments, each with its own path's boundary and reset peak."""
    rows = _validate(rows)
    per_event = []
    for row in rows:
        assets = row["scored_assets"]
        record = {"event_id": row["event_id"], "timestamp": row.get("timestamp"),
                  "scored_assets": list(assets), "asset_count": len(assets), "segments": []}
        for offset in (0, 10, 20):
            actual = {a: _segment_risk(row["spots"][a], row["actual"][a], offset) for a in assets}
            segment = {"start_minute": offset, "end_minute": offset + 10, "methods": {}}
            for method in RADIUS_METHODS:
                predicted = {a: _segment_risk(row["spots"][a], row["paths"][method][a], offset) for a in assets}
                segment["methods"][method] = {}
                for metric in METRICS:
                    mae = _mae([predicted[a][metric] for a in assets], [actual[a][metric] for a in assets])
                    segment["methods"][method][metric] = {
                        "mae": mae, "mae_bps": mae * 10000 if mae is not None else None,
                    }
            record["segments"].append(segment)
        per_event.append(record)
    segments = []
    for index, offset in enumerate((0, 10, 20)):
        segment = {"start_minute": offset, "end_minute": offset + 10,
                   "event_count": len(rows), "methods": {}, "advantages": {}}
        for method in RADIUS_METHODS:
            segment["methods"][method] = {}
            for metric in METRICS:
                entries = [record["segments"][index]["methods"][method][metric] for record in per_event]
                count = sum(entry["mae"] is not None for entry in entries)
                segment["methods"][method][metric] = {
                    "event_count": count, "null_event_count": len(rows) - count,
                    "mae": _average([entry["mae"] for entry in entries]),
                    "mae_bps": _average([entry["mae_bps"] for entry in entries]),
                }
        for metric in METRICS:
            model = segment["methods"]["kronos_base"][metric]["mae"]
            baseline = segment["methods"]["historical_30"][metric]["mae"]
            segment["advantages"][metric] = model < baseline if model is not None and baseline is not None else None
        values = list(segment["advantages"].values())
        segment["advantages"]["joint"] = all(values) if all(v is not None for v in values) else None
        segments.append(segment)
    radii = {metric: _prefix_radius([segment["advantages"][metric] for segment in segments])
             for metric in (*METRICS, "joint")}
    return {"per_event": per_event, "aggregate": {
                "event_count": len(rows), "segments": segments,
                "advantage_radius_minutes": radii,
                "joint_radius_statement": "未测得共同优势半径" if radii["joint"] is None else f"描述性共同优势半径 {radii['joint']} 分钟",
            },
            "definition": "Contiguous prefix from minute 0 with strictly smaller equal-event-weight mean MAE than frozen historical_30. Later isolated wins do not extend radius.",
            "boundary_policy": "Each predicted segment uses its own path C0/C10/C20; actual uses its own. Peak resets per segment; population log-return std times sqrt(10).",
            "disclosure": "Exploratory descriptive definition, not guaranteed future warning capability; events may overlap and are not independent."}
