"""Frozen January shock detector; pure arrays, no data/model/output access.

At t, r5[t] is compared to population std(r5[t-60:t]). Cooldown operates
continuously from t=65, before fixed-anchor episode merging. Systemic labels
are explicitly posthoc and cannot change event anchors or calibration counts.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np

K_GRID = (3, 4, 5, 6)
FIRST_INDEX = 65
COOLDOWN = 30
MERGE_MINUTES = 10
SELECTION_RULE = "eligible episodes in [20,60]; closest to 40; ties choose larger k; otherwise selected_k=null"


def _prepare(closes, times, assets):
    values = np.asarray(closes)
    if values.ndim != 2 or values.dtype.kind not in "fiu":
        raise ValueError("closes must be a numeric N-by-assets array")
    values = values.astype(np.float64, copy=False)
    names = list(assets)
    if (not names or any(not isinstance(s, str) or not s for s in names)
            or len(set(names)) != len(names) or values.shape[1] != len(names)):
        raise ValueError("assets must be unique nonempty names matching the close columns")
    stamps = list(times)
    if len(stamps) != len(values):
        raise ValueError("times must match the number of close rows")
    if not np.all(np.isfinite(values)) or not np.all(values > 0):
        raise ValueError("all closes must be finite and positive")
    parsed = []
    for stamp in stamps:
        if not isinstance(stamp, str):
            raise ValueError("times must be UTC candle-end strings")
        try:
            moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("invalid timestamp") from None
        if moment.utcoffset() != timedelta(0) or moment.second or moment.microsecond:
            raise ValueError("times must be exact UTC minutes")
        if parsed and moment - parsed[-1] != timedelta(minutes=1):
            raise ValueError("times must be ordered, unique and complete one-minute steps")
        parsed.append(moment)
    r5 = np.full(values.shape, np.nan, dtype=np.float64)
    sigma = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) > 5:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r5[5:] = np.log(values[5:] / values[:-5])
        if not np.all(np.isfinite(r5[5:])):
            raise ValueError("five-minute returns exceed finite numerical range")
    if len(values) > FIRST_INDEX:
        # The first view is r5[5:65], used only at t=65. The final view
        # would be for t=N, so discard it. No r5[t] enters its own sigma.
        views = np.lib.stride_tricks.sliding_window_view(r5[5:], 60, axis=0)
        sigma[FIRST_INDEX:] = np.std(views[:-1], axis=-1, ddof=0)
    return values, stamps, names, parsed, r5, sigma


def _event_id(k, trigger_origins):
    anchor = {"policy": "shock-radar-v1", "k": k, "origins": trigger_origins}
    encoded = json.dumps(anchor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "event_" + hashlib.sha256(encoded).hexdigest()


def _merge(triggers, times, parsed, k):
    """Fixed first-trigger anchor; later members never move/extend its boundary."""
    indices = [trigger["index"] for trigger in triggers]
    events, cursor = [], 0
    while cursor < len(triggers):
        index = triggers[cursor]["index"]
        last = bisect_right(indices, index + MERGE_MINUTES)
        members = triggers[cursor:last]
        origins = [trigger for trigger in members if trigger["index"] == index]
        origin_assets = [trigger["asset"] for trigger in origins]
        directions = {trigger["direction"] for trigger in origins}
        lower = bisect_left(indices, index - MERGE_MINUTES)
        upper = bisect_right(indices, index + MERGE_MINUTES)
        systemic_assets = {trigger["asset"] for trigger in triggers[lower:upper]}
        complete = index + MERGE_MINUTES < len(times)
        expected_time = (parsed[index] + timedelta(minutes=MERGE_MINUTES)).isoformat().replace("+00:00", "Z")
        excluded = []
        if index < 255:
            excluded.append("insufficient_history_256")
        if index + 30 >= len(times):
            excluded.append("incomplete_future_30")
        events.append({
            "event_id": _event_id(k, origins), "index": index, "timestamp": times[index],
            "origin_asset": origin_assets[0] if len(origin_assets) == 1 else None,
            "origin_assets": origin_assets, "direction": next(iter(directions)) if len(directions) == 1 else "mixed",
            "magnitude_sigma": max(trigger["magnitude_sigma"] for trigger in origins),
            "systemic_flag": len(systemic_assets) >= 3 if complete else None,
            "systemic_asset_count": len(systemic_assets),
            "systemic_confirmed_at": expected_time if complete else None,
            "systemic_confirmation_complete": complete, "systemic_confirmation_expected_at": expected_time,
            "eligible": not excluded, "exclusion_reasons": excluded, "member_triggers": members,
        })
        cursor = last
    return events


def _detect(prepared, k):
    if isinstance(k, bool) or k not in K_GRID:
        raise ValueError("k must be one of the frozen values 3, 4, 5, 6")
    k = int(k)
    values, times, assets, parsed, r5, sigma = prepared
    finite = np.isfinite(sigma)
    zero = finite & (sigma == 0)
    candidates = finite & (sigma > 0) & (np.abs(r5) > k * sigma)
    triggers, per_asset = [], {}
    for column, asset in enumerate(assets):
        raw = np.flatnonzero(candidates[:, column])
        last_kept, kept = -COOLDOWN, 0
        for index in raw:
            index = int(index)
            if index < last_kept + COOLDOWN:
                continue
            last_kept, kept = index, kept + 1
            magnitude = float(abs(r5[index, column]) / sigma[index, column])
            if not np.isfinite(magnitude):
                raise ValueError("magnitude_sigma exceeds finite numerical range")
            triggers.append({"index": index, "timestamp": times[index], "asset": asset,
                             "direction": 1 if r5[index, column] > 0 else -1, "magnitude_sigma": magnitude})
        per_asset[asset] = {"raw_candidate_count": len(raw), "retained_trigger_count": kept,
                            "suppressed_candidate_count": len(raw) - kept,
                            "zero_sigma_count": int(np.count_nonzero(zero[:, column]))}
    order = {asset: i for i, asset in enumerate(assets)}
    triggers.sort(key=lambda trigger: (trigger["index"], order[trigger["asset"]]))
    events = _merge(triggers, times, parsed, k)
    eligible = sum(event["eligible"] for event in events)
    stats = {"raw_candidate_count": int(np.count_nonzero(candidates)), "retained_trigger_count": len(triggers),
             "suppressed_candidate_count": int(np.count_nonzero(candidates)) - len(triggers),
             "zero_sigma_count": int(np.count_nonzero(zero)), "merged_event_count": len(events),
             "eligible_event_count": eligible, "excluded_event_count": len(events) - eligible,
             "per_asset": per_asset}
    return {"k": k, "events": events, "triggers": triggers, "stats": stats,
            "definitions": {"r5": "log(C[t]/C[t-5])", "sigma": "population std(r5[t-60:t]); excludes r5[t]",
                            "first_computable_index": FIRST_INDEX, "cooldown": "[last_kept,last_kept+30)",
                            "merge": "fixed first trigger t0; consume through t0+10 inclusive; no chaining",
                            "systemic": "posthoc >=3 distinct retained-trigger assets in [t0-10,t0+10]; never a trigger or input",
                            "zero_sigma": "undefined, excluded from candidates; counts are asset-minute observations"}}


def detect_events(closes, times, assets, k):
    """Detect without reading files, mutating inputs, using models or randomness."""
    return _detect(_prepare(closes, times, assets), k)


def calibrate(closes, times, assets):
    """Count the frozen four thresholds; never cap/subsample or expand the grid."""
    prepared = _prepare(closes, times, assets)
    catalogs = {str(k): _detect(prepared, k) for k in K_GRID}
    grid = [{"k": k, **catalogs[str(k)]["stats"]} for k in K_GRID]
    feasible = [row for row in grid if 20 <= row["eligible_event_count"] <= 60]
    selected = min(feasible, key=lambda row: (abs(row["eligible_event_count"] - 40), -row["k"])) if feasible else None
    return {"selected_k": selected["k"] if selected else None, "grid": grid,
            "catalogs": catalogs, "selection_rule": SELECTION_RULE}
