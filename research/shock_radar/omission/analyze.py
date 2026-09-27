"""Audit historical-volatility Top3 omissions using frozen risk scalars only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CONTRACT_SHA256 = "fb865f8cd5599871764bff1ca2d70856454769128012ac1fdfa3f7a519367ea2"
ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC")
METRICS = ("volatility", "max_drawdown")
INPUTS = {
    "2026-01": (ROOT / "research/shock_radar/artifacts/january-v1/inference-original-v1", 84),
    "2026-02": (ROOT / "research/shock_radar/closure/artifacts/february-v1/inference-original-v1", 56),
}
OUTPUT_NAMES = ("events.csv", "missed_assets.csv", "summary.json", "manifest.json")


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def finite_risk(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid nonnegative finite risk: {field}")
    if field.endswith("max_drawdown") and value > 1:
        raise ValueError(f"Drawdown outside [0,1]: {field}")
    return value


def validate_record(data, month, filename=None):
    event = data["event"]
    eid, timestamp = event["event_id"], event["timestamp"]
    if not isinstance(eid, str) or not eid.startswith("event_"):
        raise ValueError("Invalid event ID")
    if filename is not None and filename != eid + ".json":
        raise ValueError("Event ID/filename mismatch")
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0 or timestamp[:7] != month:
        raise ValueError("Event time must be UTC within declared month")
    history = data["predicted_risk"]["historical_30"]
    actual = data["actual_risk"]
    if not isinstance(history, dict) or set(history) != set(ASSETS):
        raise ValueError("Historical risk table must contain all ten assets including origins")
    if not isinstance(actual, dict) or set(actual) != set(ASSETS):
        raise ValueError("Actual risk table must contain all ten assets including origins")
    return {
        "event_id": eid, "timestamp": timestamp, "month": month,
        "historical_volatility": {a: finite_risk(history[a]["volatility"], f"{a}.historical.volatility") for a in ASSETS},
        "actual": {metric: {a: finite_risk(actual[a][metric], f"{a}.actual.{metric}") for a in ASSETS} for metric in METRICS},
    }


def analyze_event(record):
    # Python stable sort matches the product JS stable sort over this same fixed list.
    ranked = sorted(ASSETS, key=lambda a: -record["historical_volatility"][a])
    chosen = ranked[:3]
    rest = [a for a in ASSETS if a not in chosen]
    result = {k: record[k] for k in ("event_id", "timestamp", "month")}
    result.update(selected_assets=chosen, unselected_assets=rest)
    details = []
    all_missed = set()
    for metric in METRICS:
        actual = record["actual"][metric]
        weakest = min(actual[a] for a in chosen)
        missed = [a for a in rest if actual[a] > weakest]
        excesses = [(actual[a]-weakest)*10000 for a in missed]
        result[metric] = {
            "weakest_selected_actual": weakest, "missed_assets": missed,
            "missed_asset_count": len(missed), "has_omission": bool(missed),
            "max_excess_bps": max(excesses) if excesses else None,
        }
        for asset, excess in zip(missed, excesses):
            details.append({**{k: record[k] for k in ("event_id", "timestamp", "month")},
                            "metric": metric, "asset": asset, "selected_assets": chosen,
                            "weakest_selected_actual": weakest, "actual": actual[asset], "excess_bps": excess})
        all_missed.update(missed)
    result.update(
        either_metric_has_omission=any(result[m]["has_omission"] for m in METRICS),
        both_metrics_have_omission=all(result[m]["has_omission"] for m in METRICS),
        union_missed_assets=[a for a in ASSETS if a in all_missed],
        union_missed_asset_count=len(all_missed),
    )
    return result, details


def summarize(events, missed):
    n = len(events)
    result = {"event_count": n}
    for metric in METRICS:
        omitted_events = sum(e[metric]["has_omission"] for e in events)
        values = [r["excess_bps"] for r in missed if r["metric"] == metric]
        omitted_pairs = sum(e[metric]["missed_asset_count"] for e in events)
        if omitted_pairs != len(values):
            raise ValueError("Missed asset-event count differs from amplitude denominator")
        result[metric] = {
            "events_with_omission": omitted_events,
            "event_denominator": n,
            "event_omission_rate": omitted_events/n if n else None,
            "missed_asset_event_pairs": omitted_pairs,
            "mean_missed_assets_per_event": omitted_pairs/n if n else None,
            "excess_bps_denominator": len(values),
            "median_excess_bps": median(values) if values else None,
            "max_excess_bps": max(values) if values else None,
        }
    for label, field in (("either_metric", "either_metric_has_omission"), ("both_metrics", "both_metrics_have_omission")):
        count = sum(e[field] for e in events)
        result[label] = {"events_with_omission": count, "event_denominator": n,
                         "event_omission_rate": count/n if n else None}
    union = sum(e["union_missed_asset_count"] for e in events)
    result["union_missed_asset_event_pairs"] = union
    result["mean_union_missed_assets_per_event"] = union/n if n else None
    return result


def csv_event(row):
    result = {k: row[k] for k in ("event_id", "timestamp", "month")}
    for k in ("selected_assets", "unselected_assets", "union_missed_assets"):
        result[k] = "|".join(row[k])
    for metric in METRICS:
        for key, value in row[metric].items():
            result[f"{metric}_{key}"] = "|".join(value) if isinstance(value, list) else value
    for k in ("either_metric_has_omission", "both_metrics_have_omission", "union_missed_asset_count"):
        result[k] = row[k]
    return result


def write_csv(name, rows, fields):
    with (OUT/name).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, value):
    with (OUT/name).open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main():
    if any((OUT/name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("Preserve previous audit: output file already exists")
    if sha256((OUT/"CONTRACT.md").read_bytes()) != CONTRACT_SHA256:
        raise ValueError("Frozen contract hash changed; do not compute results")
    before, errors, records, found = {}, [], [], {}
    for month, (folder, expected) in INPUTS.items():
        files = sorted(folder.glob("event_*.json"))
        found[month] = len(files)
        if len(files) != expected:
            errors.append({"month": month, "error": "unexpected_file_count", "expected": expected, "found": len(files)})
        for path in files:
            if path.resolve().parent != folder.resolve():
                errors.append({"month": month, "source": path.name, "error": "Input resolves outside allowed folder"})
                continue
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            before[relative] = sha256(content)
            try:
                records.append(validate_record(json.loads(content), month, path.name))
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                errors.append({"month": month, "source": relative, "error": str(exc)})
    if len({r["event_id"] for r in records}) != len(records):
        errors.append({"error": "duplicate_event_id"})
    after = {relative: sha256((ROOT/relative).read_bytes()) for relative in before}
    if after != before:
        errors.append({"error": "input_hash_changed_during_read"})
    contract_unchanged = sha256((OUT/"CONTRACT.md").read_bytes()) == CONTRACT_SHA256
    if not contract_unchanged:
        errors.append({"error": "contract_changed_during_read"})
    population = {"expected_by_month": {m: v[1] for m,v in INPUTS.items()}, "found_by_month": found,
                  "expected_events": 140, "validated_events": len(records), "invalid_or_missing": errors}
    manifest = {"schema_version": 1, "started_or_checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract_sha256": CONTRACT_SHA256, "input_sha256_before": before, "input_sha256_after": after,
                "input_hashes_unchanged": before == after, "contract_hash_unchanged": contract_unchanged,
                "model_calls": 0, "network_calls": 0, "raw_market_reads": 0,
                "code_sha256": sha256(Path(__file__).read_bytes())}
    if errors or len(records) != 140:
        summary = {"status": "invalid_input_rates_not_computed", "contract_sha256": CONTRACT_SHA256,
                   "population": population, "groups": None}
        write_json("summary.json", summary)
        write_json("manifest.json", manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    records.sort(key=lambda e: (e["timestamp"], e["event_id"]))
    events, missed = [], []
    for record in records:
        row, details = analyze_event(record)
        events.append(row)
        missed.extend(details)
    groups = {month: summarize([e for e in events if e["month"] == month], [r for r in missed if r["month"] == month]) for month in INPUTS}
    groups["combined"] = summarize(events, missed)
    summary = {"schema_version": 1, "status": "complete", "contract_sha256": CONTRACT_SHA256,
               "population": population, "assets": list(ASSETS), "groups": groups,
               "selection": "Top3 by frozen historical_30 volatility descending; exact ties preserve fixed asset order; all10 include origins",
               "comparison": "Same historical-volatility Top3 for both actual metrics; any unselected actual value strictly above min(selected actual) is an omission",
               "units": "Risk scalars are fractions; excess_bps=(unselected_actual-min_selected_actual)*10000",
               "amplitude_denominator": "Missed asset-event pairs, separately for volatility and max_drawdown; no misses => median/max null",
               "aggregation": "Every complete event has equal weight; zero-omission events retained; combined counts sum both months",
               "limitations": "Relative screening omission in a frozen shock catalog, not an absolute loss alarm or Kronos score. Event windows may be correlated. No inference of causal/independent samples or safe exclusion of seven assets.",
               "publication_status": "requires_user_confirmation_before_pitch"}
    # Verify all frozen bytes again after deterministic analysis, before publishing.
    final_hashes = {relative: sha256((ROOT/relative).read_bytes()) for relative in before}
    if final_hashes != before or sha256((OUT/"CONTRACT.md").read_bytes()) != CONTRACT_SHA256:
        raise ValueError("Source changed during analysis; results not published")
    event_rows = [csv_event(e) for e in events]
    write_csv("events.csv", event_rows, list(event_rows[0]))
    detail_rows = [{**r, "selected_assets": "|".join(r["selected_assets"])} for r in missed]
    detail_fields = ["event_id", "timestamp", "month", "metric", "asset", "selected_assets", "weakest_selected_actual", "actual", "excess_bps"]
    write_csv("missed_assets.csv", detail_rows, detail_fields)
    write_json("summary.json", summary)
    manifest.update(input_sha256_after=final_hashes, output_sha256={name: sha256((OUT/name).read_bytes()) for name in OUTPUT_NAMES if name != "manifest.json"})
    write_json("manifest.json", manifest)
    print(json.dumps({"status": summary["status"], "population": population, "groups": groups}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
