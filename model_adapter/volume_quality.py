"""Audit unused predicted volume fields without changing sampled values."""
from __future__ import annotations

from .corrections import _finite_number, structure_issues, validate_corrections

VOLUME_POLICY = "unused-volume-audit-v1"


def _summary(records):
    total = len(records)
    invalid = sum(not record["volume_valid"] for record in records)
    return {"policy": VOLUME_POLICY,
            "status": "volume_forecast_unavailable" if invalid else "valid",
            "total_candles": total, "volume_invalid_count": invalid,
            "volume_invalid_rate": invalid / total, "records": records}


def build_volume_quality(raw_paths, times):
    """Return a record for every candle, with untouched volume/amount values."""
    if not isinstance(times, list) or not times or not all(isinstance(t, str) for t in times) or len(set(times)) != len(times):
        raise ValueError("nonempty unique string forecast times are required")
    issues = structure_issues(raw_paths, len(times))
    if issues:
        raise ValueError("raw output: " + "; ".join(issues))
    records = [{"path_id": path["path_id"], "asset": asset, "time": stamp,
                "volume": values["volume"][index], "amount": values["amount"][index],
                "volume_valid": values["volume"][index] >= 0 and values["amount"][index] >= 0}
               for path in raw_paths for asset, values in path["assets"].items()
               for index, stamp in enumerate(times)]
    return _summary(records)


def validate_forecast_output(forecast, raw_paths=None):
    """Require exact OHLC and volume audits; optionally bind all raw outputs."""
    validate_corrections(forecast, raw_paths)
    audit = forecast.get("volume_quality")
    fields = {"policy", "status", "total_candles", "volume_invalid_count", "volume_invalid_rate", "records"}
    if not isinstance(audit, dict) or set(audit) != fields:
        raise ValueError("complete volume quality audit is required")
    if audit["policy"] != VOLUME_POLICY:
        raise ValueError("unexpected volume quality policy")
    times = forecast["times"]
    expected_ids = {(path["path_id"], asset, stamp)
                    for path in forecast["paths"] for asset in path["assets"] for stamp in times}
    records = audit["records"]
    if not isinstance(records, list):
        raise ValueError("volume quality records must be a list")
    originals = None
    if raw_paths is not None:
        # validate_corrections already validated raw structure and identities.
        originals = {(record["path_id"], record["asset"], record["time"]): record
                     for record in build_volume_quality(raw_paths, times)["records"]}
    seen = set()
    for record in records:
        record_fields = {"path_id", "asset", "time", "volume", "amount", "volume_valid"}
        if not isinstance(record, dict) or set(record) != record_fields:
            raise ValueError("complete volume quality record required")
        key = (record["path_id"], record["asset"], record["time"])
        if any(not isinstance(x, str) for x in key) or key not in expected_ids or key in seen:
            raise ValueError("unknown or duplicate volume quality identity")
        if not _finite_number(record["volume"]) or not _finite_number(record["amount"]):
            raise ValueError("volume quality must preserve finite numerical values")
        if type(record["volume_valid"]) is not bool or record["volume_valid"] != (record["volume"] >= 0 and record["amount"] >= 0):
            raise ValueError("volume_valid disagrees with untouched volume and amount")
        if originals is not None and record != originals[key]:
            raise ValueError("volume quality record differs from raw model output")
        seen.add(key)
    if seen != expected_ids:
        raise ValueError("volume audit must cover every output candle")
    if type(audit["total_candles"]) is not int or type(audit["volume_invalid_count"]) is not int or not _finite_number(audit["volume_invalid_rate"]):
        raise ValueError("volume summary requires integer counts and a finite rate")
    if audit != _summary(records):
        raise ValueError("volume quality summary disagrees with complete records")
