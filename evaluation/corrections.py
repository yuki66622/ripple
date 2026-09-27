"""Structural correction evidence, kept separate from forecast accuracy."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations

QUALITY_FIELDS = ("policy", "total_candles", "corrected_candles", "correction_rate",
                  "correction_rate_pct", "max_adjustment_bps")
COMPARABLE_IDENTITY_FIELDS = (
    "tokenizer_revision", "adapter_revision", "sampling", "output_policy", "volume_policy",
    "backend", "torch_version", "sktime_version", "model_timestamp", "output_timestamp",
)
DISCLOSURE = (
    "Correction rate measures structural validity under containment-expand-v1; "
    "fewer corrections do not establish improved forecast accuracy. Model "
    "comparisons require identical inputs, prediction configuration, tokenizer, adapter, "
    "sampling, output policies, backend/library versions and timestamp semantics."
)


def extract_quality(forecast):
    """Copy only scalar audit evidence; this is NOT the publication validator."""
    audit = forecast.get("ohlc_corrections")
    if audit is None:
        return None
    if not isinstance(audit, dict):
        raise ValueError("ohlc_corrections must be an object")
    quality = {key: audit[key] for key in QUALITY_FIELDS}
    if quality["policy"] != "containment-expand-v1":
        raise ValueError("unsupported correction policy")
    for key in ("total_candles", "corrected_candles"):
        value = quality[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid correction audit {key}")
    if not 0 <= quality["corrected_candles"] <= quality["total_candles"] or not quality["total_candles"]:
        raise ValueError("invalid correction audit candle denominator")
    for key in ("correction_rate", "correction_rate_pct", "max_adjustment_bps"):
        value = quality[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid correction audit {key}")
    # Scalar consistency also matters for persisted artifacts. Full record/raw
    # verification remains owned by model_adapter.validate_corrections.
    rate = quality["corrected_candles"] / quality["total_candles"]
    if (not math.isclose(quality["correction_rate"], rate, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(quality["correction_rate_pct"], rate * 100, rel_tol=1e-12, abs_tol=1e-10)):
        raise ValueError("correction rate does not match candle counts")
    return quality


def validate_for_publication(forecast, raw_paths=None):
    """Always call the model-owned authoritative validator; never fail open."""
    from model_adapter import validate_forecast_output
    validate_forecast_output(forecast, raw_paths)


def _workload_key(artifact):
    """Legacy identity for descriptive deduplication, never a fair-pair gate."""
    value = {"window_id": artifact.get("window_id"), "predict_config": artifact.get("predict_config")}
    if not value["window_id"] or not isinstance(value["predict_config"], dict):
        return None
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def comparison_context(artifact):
    """Return complete weight-comparison controls, or None when not established.

    Only checkpoint revision and user-facing model label may differ. In
    particular, equal seeds on different backends are not equal sampling runs.
    Missing legacy metadata cannot silently establish fair comparability.
    """
    result = artifact.get("result") or {}
    if not isinstance(result, dict):
        return None
    runtime = artifact.get("runtime") or result.get("runtime") or {}
    if not isinstance(runtime, dict):
        return None
    identity = runtime.get("identity")
    config = artifact.get("predict_config")
    if (not isinstance(identity, dict) or not isinstance(config, dict)
            or set(config) != {"lookback", "horizon", "path_count", "seed"}
            or not isinstance(artifact.get("window_id"), str) or not artifact["window_id"]):
        return None
    for name, low, high in (("lookback", 2, 512), ("horizon", 2, 128),
                            ("path_count", 1, 16), ("seed", 0, 2**32 - 1)):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            return None
    for name in (*COMPARABLE_IDENTITY_FIELDS, "model_revision"):
        if name == "sampling":
            continue
        if not isinstance(identity.get(name), str) or not identity[name]:
            return None
    sampling = identity.get("sampling")
    if not isinstance(sampling, dict) or not {"T", "top_k", "top_p", "sample_count", "verbose"} <= set(sampling):
        return None
    for name in ("T", "top_p"):
        value = sampling[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            return None
    if sampling["top_p"] > 1 or not isinstance(sampling["verbose"], bool):
        return None
    for name, minimum in (("top_k", 0), ("sample_count", 1)):
        value = sampling[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            return None
    forecast = result.get("forecast") or {}
    if not isinstance(forecast, dict):
        return None
    if forecast.get("model_revision") not in (None, identity["model_revision"]):
        return None
    quality = artifact.get("correction_quality")
    if isinstance(quality, dict) and quality.get("policy") != identity["output_policy"]:
        return None
    # Two stored runtime copies must not disagree about the controls.
    nested_runtime = result.get("runtime") or {}
    if not isinstance(nested_runtime, dict):
        return None
    nested_identity = nested_runtime.get("identity")
    if nested_identity is not None and nested_identity != identity:
        return None
    context = {"window_id": artifact["window_id"], "predict_config": config,
               "identity": {name: identity[name] for name in COMPARABLE_IDENTITY_FIELDS}}
    try:
        return json.loads(json.dumps(context, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError):
        return None


def comparison_key(artifact):
    """Task-independent fair-pair key; model weight hashes intentionally differ."""
    value = comparison_context(artifact)
    if value is None:
        return None
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _weighted(artifacts):
    qualities = [artifact["correction_quality"] for artifact in artifacts]
    total = sum(q["total_candles"] for q in qualities)
    corrected = sum(q["corrected_candles"] for q in qualities)
    rate = corrected / total if total else None
    return {
        "forecast_count": len(qualities), "total_candles": total, "corrected_candles": corrected,
        "correction_rate": rate, "correction_rate_pct": None if rate is None else rate * 100,
        "max_adjustment_bps": max((q["max_adjustment_bps"] for q in qualities), default=None),
    }


def summarize_quality(artifacts):
    """Model-specific weighted evidence plus comparisons on matched workloads.

    Invalid/missing audits remain in explicit denominators but cannot contribute
    percentages. Equivalent repeated audits count once; conflicting repeated
    workloads remain visible and are excluded rather than selecting an attempt.
    """
    grouped = {}
    for artifact in artifacts:
        prediction = artifact.get("result", {}).get("forecast", {})
        runtime = artifact.get("runtime", artifact.get("result", {}).get("runtime", {}))
        model = artifact.get("model_spec", {}).get("model_label", "unknown")
        revision = prediction.get("model_revision") or runtime.get("identity", {}).get("model_revision") or "unavailable"
        quality = artifact.get("correction_quality")
        policy = quality.get("policy", "unavailable") if isinstance(quality, dict) else "unavailable"
        key = (model, revision, policy)
        grouped.setdefault(key, []).append(artifact)
    reports, valid_by_group = [], {}
    for (model, revision, policy), group in sorted(grouped.items()):
        valid, invalid, missing, unvalidated = [], 0, 0, 0
        workloads, conflicts, duplicate_count = {}, [], 0
        for index, artifact in enumerate(group):
            workload = _workload_key(artifact)
            workloads.setdefault(workload or f"unkeyed:{index}", []).append(artifact)
        for artifact in group:
            status = artifact.get("correction_quality_status", "missing")
            quality = artifact.get("correction_quality")
            if quality is None:
                if status == "invalid":
                    invalid += 1
                else:
                    missing += 1
                continue
            if status != "validated":
                if status == "invalid":
                    invalid += 1
                else:
                    unvalidated += 1
                continue
            try:
                extract_quality({"ohlc_corrections": quality})
            except (KeyError, TypeError, ValueError):
                invalid += 1
                continue
        for workload, attempts in workloads.items():
            duplicate_count += len(attempts) - 1
            first = attempts[0]
            signature = (first.get("correction_quality_status", "missing"), first.get("correction_quality"),
                         comparison_context(first))
            if any((a.get("correction_quality_status", "missing"), a.get("correction_quality"),
                    comparison_context(a)) != signature
                   for a in attempts[1:]):
                conflicts.append({"workload": workload, "window_id": first.get("window_id"),
                                  "predict_config": first.get("predict_config"),
                                  "attempts": [{"task_id": a.get("task_id"), "forecast_id": a.get("forecast_id"),
                                                "artifact_path": a.get("artifact_path"),
                                                "correction_quality_status": a.get("correction_quality_status", "missing"),
                                                "correction_quality": a.get("correction_quality")} for a in attempts]})
                continue
            if signature[0] == "validated" and signature[1] is not None:
                try:
                    extract_quality({"ohlc_corrections": signature[1]})
                except (KeyError, TypeError, ValueError):
                    continue
                valid.append(first)
        group_id = hashlib.sha256(json.dumps([model, revision, policy]).encode()).hexdigest()
        reports.append({"group_id": group_id, "model": model, "model_revision": revision, "policy": policy,
                        **_weighted(valid), "attempted_forecasts": len(group),
                        "missing_audit_count": missing, "invalid_audit_count": invalid,
                        "unvalidated_audit_count": unvalidated,
                        "distinct_workloads": len(workloads), "duplicate_records": duplicate_count,
                        "conflicting_duplicate_workloads": len(conflicts), "conflicts": conflicts,
                        "comparable_forecasts": sum(comparison_key(a) is not None for a in valid),
                        "incomparable_context_forecasts": sum(comparison_key(a) is None for a in valid)})
        valid_by_group[group_id] = {comparison_key(a): a for a in valid if comparison_key(a) is not None}
    matched = []
    for a, b in combinations(reports, 2):
        if a["policy"] != b["policy"] or a["policy"] == "unavailable":
            continue
        left, right = valid_by_group[a["group_id"]], valid_by_group[b["group_id"]]
        common = sorted(set(left) & set(right))
        matched.append({
            "model_a": a["model"], "revision_a": a["model_revision"],
            "model_b": b["model"], "revision_b": b["model_revision"],
            "policy": a["policy"], "matched_task_count": len(common),
            "unmatched_a_count": a["forecast_count"] - len(common),
            "unmatched_b_count": b["forecast_count"] - len(common),
            "a": _weighted([left[key] for key in common]),
            "b": _weighted([right[key] for key in common]),
            "matching": "identical window_id, full predict_config and complete runtime controls; only model revision/label may differ",
            "required_identity_fields": list(COMPARABLE_IDENTITY_FIELDS),
            "status": "matched_descriptive_only" if common else "no_comparable_workloads",
        })
    return {"by_model": reports, "matched_comparisons": matched,
            "aggregation": "sum(corrected_candles)/sum(total_candles); never mean of task percentages",
            "duplicate_policy": "Equivalent repeated scalar audits count once per model/window/config; conflicting repeats excluded with all attempts retained",
            "disclosure": DISCLOSURE}
