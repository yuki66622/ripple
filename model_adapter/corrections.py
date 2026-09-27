"""User-approved candle containment and independently checked audit records.

Prices stay sampled: only high/low expand to include all four original prices.
Open and close are copied exactly. Missing, nonfinite or nonpositive prices are
never filled, clipped, interpolated or resampled.
"""
from __future__ import annotations

import copy
import math

POLICY = "containment-expand-v1"
OHLC = ("open", "high", "low", "close")
OHLCVA = (*OHLC, "volume", "amount")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def structure_issues(raw_paths, horizon, expected_assets=None, expected_count=None):
    """Inspect all raw structures before correction, without modifying them."""
    issues = []
    if not isinstance(raw_paths, list) or not raw_paths:
        return ["raw_paths must be a nonempty list"]
    if expected_count is not None and len(raw_paths) != expected_count:
        issues.append("wrong path count")
    seen = set()
    common_assets = set(expected_assets) if expected_assets is not None else None
    for path_index, path in enumerate(raw_paths):
        if not isinstance(path, dict):
            issues.append(f"path/{path_index}: expected an object")
            continue
        path_id = path.get("path_id")
        if not isinstance(path_id, str) or not path_id or path_id in seen:
            issues.append(f"path/{path_index}: missing or duplicate path_id")
        else:
            seen.add(path_id)
        assets = path.get("assets")
        if not isinstance(assets, dict) or not assets:
            issues.append(f"path/{path_index}: missing assets")
            continue
        if common_assets is None:
            common_assets = set(assets)
        if set(assets) != common_assets:
            issues.append(f"{path_id}: wrong asset set")
        for asset, values in assets.items():
            prefix = f"{path_id}/{asset}"
            if not isinstance(asset, str) or not asset or not isinstance(values, dict):
                issues.append(f"{prefix}: invalid asset structure")
                continue
            if set(values) != set(OHLCVA) or any(not isinstance(values.get(key), list) or len(values[key]) != horizon for key in OHLCVA):
                issues.append(f"{prefix}: wrong output shape or missing OHLCVA field")
                continue
            for i in range(horizon):
                row = {key: values[key][i] for key in OHLCVA}
                if not all(_finite_number(value) for value in row.values()):
                    issues.append(f"{prefix}/{i}: non-finite output")
                    continue
                if min(row[key] for key in OHLC) <= 0:
                    issues.append(f"{prefix}/{i}: nonpositive price")
                # Output policy allows finite negative unused volume fields.
                # Their values remain unchanged and are audited separately.
    return issues


def _record(path_id, asset, timestamp, original):
    corrected = dict(original)
    corrected["high"] = max(original.values())
    corrected["low"] = min(original.values())
    high_bps = abs(corrected["high"] - original["high"]) / original["close"] * 10000
    low_bps = abs(corrected["low"] - original["low"]) / original["close"] * 10000
    return {
        "path_id": path_id, "asset": asset, "time": timestamp,
        "original": dict(original), "corrected": corrected,
        "high_adjustment_bps": high_bps, "low_adjustment_bps": low_bps,
        "max_adjustment_bps": max(high_bps, low_bps),
        "was_corrected": corrected["high"] != original["high"] or corrected["low"] != original["low"],
    }


def _summary(records):
    count = sum(record["was_corrected"] for record in records)
    total = len(records)
    return {"policy": POLICY, "bps_denominator": "original_close",
            "total_candles": total, "corrected_candles": count,
            "correction_rate": count / total, "correction_rate_pct": count / total * 100,
            "max_adjustment_bps": max(record["max_adjustment_bps"] for record in records),
            "records": records}


def correct_paths(raw_paths, times, expected_assets=None, expected_count=None):
    """Return copied OHLCVA paths plus all audit records; no input mutation."""
    if not isinstance(times, list) or not times or len(set(times)) != len(times):
        raise ValueError("nonempty unique forecast times are required")
    issues = structure_issues(raw_paths, len(times), expected_assets, expected_count)
    if issues:
        raise ValueError("; ".join(issues))
    corrected_paths = copy.deepcopy(raw_paths)
    records = []
    for path, corrected_path in zip(raw_paths, corrected_paths):
        for asset, values in path["assets"].items():
            for index, timestamp in enumerate(times):
                original = {key: values[key][index] for key in OHLC}
                record = _record(path["path_id"], asset, timestamp, original)
                records.append(record)
                for key in ("high", "low"):
                    corrected_path["assets"][asset][key][index] = record["corrected"][key]
    return corrected_paths, _summary(records)


def validate_corrections(forecast, raw_paths=None):
    """Raise ValueError for incomplete, altered or inconsistent audit metadata.

Single-argument validation checks every original-to-corrected operation and its
actual published prices. Supplying raw_paths additionally binds every original
price to the untouched model output. This function never mutates either input.
"""
    from forecast_metrics.engine import freeze_forecast
    if not isinstance(forecast, dict):
        raise ValueError("forecast must be an object")
    # Shared frozen metrics validation includes full grid, shape and bound checks.
    freeze_forecast(forecast)
    audit = forecast.get("ohlc_corrections")
    required = {"policy", "bps_denominator", "total_candles", "corrected_candles",
                "correction_rate", "correction_rate_pct", "max_adjustment_bps", "records"}
    if not isinstance(audit, dict) or set(audit) != required:
        raise ValueError("complete correction audit is required")
    if audit["policy"] != POLICY or audit["bps_denominator"] != "original_close":
        raise ValueError("unexpected correction policy or bps denominator")
    records = audit["records"]
    if not isinstance(records, list):
        raise ValueError("correction records must be a list")
    times = forecast["times"]
    published = {(p["path_id"], asset, stamp): {key: values[key][index] for key in ("high", "low", "close")}
                 for p in forecast["paths"] for asset, values in p["assets"].items()
                 for index, stamp in enumerate(times)}
    originals = None
    if raw_paths is not None:
        issues = structure_issues(raw_paths, len(times), forecast["spots"], len(forecast["paths"]))
        if issues:
            raise ValueError("raw paths: " + "; ".join(issues))
        originals = {(p["path_id"], asset, stamp): {key: values[key][index] for key in OHLC}
                     for p in raw_paths for asset, values in p["assets"].items()
                     for index, stamp in enumerate(times)}
        if set(originals) != set(published):
            raise ValueError("raw path/time identities do not match forecast")
    seen = set()
    expected_records = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("correction record must be an object")
        try:
            key = (record["path_id"], record["asset"], record["time"])
            if any(not isinstance(part, str) for part in key) or key not in published or key in seen:
                raise ValueError("unknown or duplicate correction record identity")
            original = record["original"]
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete correction record") from exc
        if not isinstance(original, dict) or set(original) != set(OHLC) or not all(_finite_number(v) and v > 0 for v in original.values()):
            raise ValueError("original correction prices must be complete finite positive OHLC")
        expected = _record(*key, original)
        if type(record.get("was_corrected")) is not bool:
            raise ValueError("was_corrected must be boolean")
        if not isinstance(record.get("corrected"), dict) or set(record["corrected"]) != set(OHLC) or not all(_finite_number(v) and v > 0 for v in record["corrected"].values()):
            raise ValueError("corrected OHLC must be complete finite positive prices")
        for field in ("high_adjustment_bps", "low_adjustment_bps", "max_adjustment_bps"):
            if not _finite_number(record.get(field)) or record[field] < 0:
                raise ValueError("correction bps must be finite nonnegative numbers")
        if record != expected:
            raise ValueError("correction record differs from the authorized exact operation")
        if any(published[key][field] != expected["corrected"][field] for field in ("high", "low", "close")):
            raise ValueError("published prices do not match correction audit")
        if originals is not None and original != originals[key]:
            raise ValueError("correction original differs from raw model output")
        seen.add(key)
        expected_records.append(expected)
    if seen != set(published):
        raise ValueError("correction audit must cover every output candle, including unchanged bars")
    expected_summary = _summary(expected_records)
    for field in ("total_candles", "corrected_candles"):
        if type(audit[field]) is not int:
            raise ValueError("correction counts must be integers")
    for field in ("correction_rate", "correction_rate_pct", "max_adjustment_bps"):
        if not _finite_number(audit[field]):
            raise ValueError("correction summary must contain finite numbers")
    if audit != expected_summary:
        raise ValueError("correction summary differs from complete record totals")
