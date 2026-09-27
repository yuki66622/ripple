"""Read-only independent verification of saved three-event sample quantiles."""
from pathlib import Path
import json
import math

from ..io import ROOT
from ..supplement.common import frozen_inputs
OUT = Path(__file__).resolve().parent / "artifacts/demo-multipath-v2"
BASE = ROOT / "research/shock_radar/artifacts/january-v1"


def drawdown(prices):
    peak = prices[0]
    worst = 0.
    for price in prices[1:]:
        assert math.isfinite(price) and price > 0
        peak = max(peak, price)
        worst = max(worst, 1. - price / peak)
    return worst


def quantile(values, q):
    ordered = sorted(values)
    pos = q * (len(ordered)-1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi]-ordered[lo]) * (pos-lo)


def verify():
    summary = json.loads((OUT / "uncertainty.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert summary["frozen_originals_unchanged"] and frozen_inputs() == manifest["frozen_january_sha256"]
    assert [e["event_id"] for e in summary["events"]] == [e["event_id"] for e in manifest["choices"]]
    calls, scalar_checks, matching_path0, complete, failures = 0, 0, 0, 0, []
    for event in summary["events"]:
        r = json.loads((OUT / (event["event_id"] + ".json")).read_text())
        if r["status"] != "complete":
            failures.append(event["event_id"])
            assert all(v["p05"] is None and v["p50"] is None and v["p95"] is None for v in event["asset_mdd_quantiles"].values())
            continue
        complete += 1
        old = json.loads((BASE / "inference-original-v1" / (event["event_id"] + ".json")).read_text())
        assert r["original_forecast_id"] == old["forecast_id"] and r["forecast_id"] != old["forecast_id"]
        assert r["runtime"]["identity"] == summary["model_identity"] == manifest["model_identity"]
        runtime_calls = r["runtime"]["calls"]
        calls += len(runtime_calls)
        assert len(runtime_calls) == 100
        assert len({(c["asset"], c["path_index"]) for c in runtime_calls}) == 100
        for asset, intervals in event["asset_mdd_quantiles"].items():
            spot = r["forecast"]["spots"][asset]
            dds = []
            for path, raw in zip(r["forecast"]["paths"], r["raw_paths"]):
                assert path["assets"][asset]["close"] == raw["assets"][asset]["close"]
                dds.append(drawdown([spot] + path["assets"][asset]["close"]))
            assert len(dds) == intervals["n_valid"] == 10
            for a, b in zip(dds, intervals["per_path_mdd"]):
                assert math.isclose(a, b, abs_tol=1e-13)
                scalar_checks += 1
            for label, q in [("p05", .05), ("p50", .5), ("p95", .95)]:
                assert math.isclose(quantile(dds, q), intervals[label], abs_tol=1e-13)
                scalar_checks += 1
            assert intervals["p05"] <= intervals["p50"] <= intervals["p95"]
            matches = r["forecast"]["paths"][0]["assets"][asset]["close"] == old["method_close_paths"]["kronos_base"][asset]
            assert matches == intervals["path_zero_close_equals_original"]
            matching_path0 += int(matches)
    assert calls <= summary["recorded_asset_path_calls"]
    if not failures:
        assert calls == summary["recorded_asset_path_calls"] == 300
    return {"complete_events": complete, "failed_events": failures, "verified_completed_asset_path_calls": calls,
            "independent_mdd_quantile_checks": scalar_checks, "matching_original_path_zero_assets": matching_path0,
            "frozen_originals_unchanged": True, "verification_model_calls": 0}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False))
