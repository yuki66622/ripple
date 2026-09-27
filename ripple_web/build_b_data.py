"""Export the frozen B-line explorer without touching A-line or March data.

Only explorer.json is written. No model, market-data, network or old builder
imports. Existing presentation datasets remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / "research/shock_radar"
WEB = Path(__file__).resolve().parent
ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
CLASS_FIELDS = ("id", "ordinal", "name", "size", "origins", "origin_count", "up", "down", "median_followers")


class FrozenReader:
    def __init__(self):
        self.sources = {}

    def load(self, path):
        path = Path(path).resolve()
        rel = path.relative_to(ROOT).as_posix()
        # Explicitly restrict the exporter to delivered B artifacts and the
        # existing presentation class names. New callers cannot read A inputs.
        permitted = (
            rel.startswith("research/shock_radar/closure/artifacts/january-graph-v1/")
            or rel.startswith("research/shock_radar/closure/artifacts/demo-multipath-v2/")
            or rel.startswith("research/shock_radar/artifacts/january-v1/inference-original-v1/event_")
            or rel in {
                "ripple_web/data/nearest.json",
                "research/shock_radar/deep_checks/coverage.json",
                "research/shock_radar/deep_checks/permutation.json",
                "research/shock_radar/deep_checks/clustering.json",
                "research/shock_radar/deep_checks/failures.json",
            }
        )
        if not permitted:
            raise ValueError(f"Input outside frozen B scope: {rel}")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if rel in self.sources and self.sources[rel] != digest:
            raise ValueError(f"Frozen input changed: {rel}")
        self.sources[rel] = digest
        return json.loads(content)

    def verify(self):
        for relative, expected in self.sources.items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen input changed: {relative}")


def validate_slots(slots):
    if set(slots) != set(ASSETS):
        raise ValueError("Fingerprint must have exactly the ten frozen assets")
    for asset in ASSETS:
        slot = slots[asset]
        if type(slot["present"]) is not bool:
            raise ValueError("Presence must be boolean")
        if not slot["present"]:
            if any(slot[k] is not None for k in ("direction", "magnitude_sigma", "delay_minutes")):
                raise ValueError("Absent fingerprint slots must keep null fields")
        else:
            if type(slot["direction"]) is not int or slot["direction"] not in (-1, 1):
                raise ValueError("Invalid trigger direction")
            for key in ("magnitude_sigma", "delay_minutes"):
                if isinstance(slot[key], bool) or not isinstance(slot[key], (int, float)) or not math.isfinite(slot[key]):
                    raise ValueError("Nonfinite fingerprint coordinate")
            if slot["magnitude_sigma"] < 6 or not 0 <= slot["delay_minutes"] <= 30:
                raise ValueError("Fingerprint outside frozen k6/30-minute domain")
    return slots


def event_presentation(row):
    full = row["full_historical_event"]["event"]
    actual = row["saved_risk"]["actual_risk"]
    if set(actual) != set(ASSETS) or row["event_id"] != full["event_id"]:
        raise ValueError("Frozen event identity/assets mismatch")
    if row["fingerprint"]["assets"] != ASSETS:
        raise ValueError("Frozen fingerprint order mismatch")
    slots = validate_slots(row["fingerprint"]["slots"])
    for asset in ASSETS:
        value = actual[asset]["max_drawdown"]
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("Invalid saved actual drawdown")
    # Stable original ten-asset ordering resolves ties; no claim of probabilistic ranking.
    ranked = sorted(ASSETS, key=lambda a: -actual[a]["max_drawdown"])
    return {"event_id": row["event_id"], "timestamp": row["timestamp"],
            "origin_assets": row["origin_assets"], "direction": full["direction"],
            "magnitude_sigma": full["magnitude_sigma"], "class_id": row["cluster_id"],
            "slots": slots, "actual_max_mdd": actual[ranked[0]]["max_drawdown"],
            "actual_top_mdd": ranked[:3]}


def build():
    reader = FrozenReader()
    graphdir = SR / "closure/artifacts/january-graph-v1"
    index = reader.load(graphdir / "index.json")
    nearest = reader.load(graphdir / "demo-nearest.json")
    old = reader.load(WEB / "data/nearest.json")
    events = sorted((event_presentation(r) for r in index["events"]), key=lambda r: (r["timestamp"], r["event_id"]))
    if index["assets"] != ASSETS or len(events) != 84 or len({e["event_id"] for e in events}) != 84:
        raise ValueError("Expected the fixed84-event ten-asset library")
    classes = [{field: row[field] for field in CLASS_FIELDS} for row in old["classes"]]
    if sorted(c["size"] for c in classes) != [1, 5, 8, 70]:
        raise ValueError("Existing presentation classes differ from frozen84 partition")
    actual_sizes = {c["id"]: sum(e["class_id"] == c["id"] for e in events) for c in classes}
    if any(actual_sizes[c["id"]] != c["size"] for c in classes):
        raise ValueError("Class membership mismatch")
    query = nearest["query"]
    demo = {"event_id": query["exclude_event_id"], "observed_minutes": query["observed_minutes"],
            "expected_matches": [{"event_id": r["event_id"], "distance": r["distance"]} for r in nearest["results"]]}
    if len(demo["expected_matches"]) != 3 or demo["event_id"] not in {e["event_id"] for e in events}:
        raise ValueError("Frozen demo query/result shape invalid")
    uncertainty = reader.load(SR / "closure/artifacts/demo-multipath-v2/uncertainty.json")
    if uncertainty["status"] != "complete" or len(uncertainty["events"]) != 3:
        raise ValueError("Expected completed three-event sampling batch")
    replays = {}
    for item in uncertainty["events"]:
        eid = item["event_id"]
        original = reader.load(SR / f"artifacts/january-v1/inference-original-v1/{eid}.json")
        if item["original_forecast_id"] != original["forecast_id"]:
            raise ValueError("Original prediction identity mismatch")
        quantiles = {asset: {key: item["asset_mdd_quantiles"][asset][key]
                            for key in ("p05", "p50", "p95", "n_valid")} for asset in ASSETS}
        for q in quantiles.values():
            if q["n_valid"] != 10 or not 0 <= q["p05"] <= q["p50"] <= q["p95"] < 1:
                raise ValueError("Invalid frozen MDD quantiles")
        replays[eid] = {"forecast_id": item["forecast_id"], "original_forecast_id": item["original_forecast_id"],
                        "mdd": quantiles, "scored_assets": original["score"]["scored_assets"]}
    coverage = reader.load(SR / "deep_checks/coverage.json")
    permutation = reader.load(SR / "deep_checks/permutation.json")
    clustering = reader.load(SR / "deep_checks/clustering.json")
    failures = reader.load(SR / "deep_checks/failures.json")
    evidence = {
        "coverage": {key: coverage[key] for key in ("pooled", "by_tier", "by_event")},
        "permutation": {"real": permutation["real"], "tests": permutation["tests"],
                        "null_diagnostics": permutation["null_diagnostics"], "replicates": permutation["replicate_count"]},
        "clustering": clustering["summary"],
        "failures": {key: failures[key] for key in ("population", "full", "top10")},
        "scored_asset_counts": {row["event_id"]: row["scored_asset_count"] for row in failures["top10_rows"]},
        # Existing evaluation.json rows omit event_id; retain a direct join for
        # both January and February rows without changing that published file.
        "scored_asset_counts_by_timestamp": {row["timestamp"]: row["scored_asset_count"] for row in failures["top10_rows"]},
    }
    if len(evidence["scored_asset_counts"]) != 10 or len(evidence["scored_asset_counts_by_timestamp"]) != 10:
        raise ValueError("Expected ten failure denominators")
    reader.verify()
    result = {"schema_version": 1, "assets": ASSETS, "events": events, "classes": classes,
              "demo": demo, "replays": replays, "evidence": evidence,
              "sources": [{"path": p, "sha256": h} for p, h in sorted(reader.sources.items())]}
    # Public data must not leak local paths through nested diagnostic metadata.
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if str(ROOT) in serialized or "/Users/" in serialized or "file://" in serialized:
        raise ValueError("Absolute local path in public output")
    return result


def main():
    result = build()
    output = WEB / "data/explorer.json"
    output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": "ripple_web/data/explorer.json", "events": len(result["events"]),
                      "classes": len(result["classes"]), "replays": len(result["replays"]),
                      "source_count": len(result["sources"]), "source_hashes_unchanged": True,
                      "model_calls": 0, "raw_market_data_reads": 0}))


if __name__ == "__main__":
    main()
