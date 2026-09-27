"""Explicitly authorized ten-path sampling for three existing demo events only."""
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from model_adapter import KronosAdapter, PredictionValidationError, validate_forecast_output
from forecast_metrics.engine import freeze_forecast
from ..io import ROOT, digest, load_january, market_window, write_json
from ..metrics import risk_metrics
from ..supplement.common import frozen_inputs

HERE = Path(__file__).resolve().parent
BASE = ROOT / "research/shock_radar/artifacts/january-v1"
OUT = HERE / "artifacts/demo-multipath-v2"
CONFIG = {"lookback": 256, "horizon": 30, "path_count": 10, "seed": 20260926}
MODEL = ROOT / "scenario-lab/models/Kronos-base"
TOKENIZER = ROOT / "scenario-lab/models/Kronos-Tokenizer-base"


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe(v) for v in value]
    return value


def percentiles(spot, paths):
    """MDD per path first; no average price curves and no partial quantiles."""
    values = []
    for path in paths:
        try:
            values.append(risk_metrics(spot, path)["max_drawdown"])
        except (ValueError, TypeError):
            values.append(None)
    valid = [v for v in values if v is not None]
    q = np.quantile(valid, [.05, .50, .95], method="linear").tolist() if len(paths) == len(valid) == 10 else [None] * 3
    return {"p05": q[0], "p50": q[1], "p95": q[2], "n_expected": 10,
            "n_valid": len(valid), "per_path_mdd": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if OUT.exists():
        raise FileExistsError("No overwrite or implicit resampling of an existing demo batch")
    before = frozen_inputs()
    old = json.loads((BASE / "inference-original-v1/run_manifest.json").read_text())
    selection = json.loads((BASE / "selection-approved-k6.json").read_text())
    choices = selection["replays_selected_before_inference"]
    assert len(choices) == 3 and len({c["event_id"] for c in choices}) == 3
    data = load_january()
    assert data["provenance"] == old["provenance"]
    adapter = KronosAdapter(model_path=MODEL, tokenizer_path=TOKENIZER, device=args.device)
    identity = adapter.identity()
    assert identity == old["model_identity"], "Original model/runtime identity changed; do not silently compare"
    OUT.mkdir(parents=True)
    code_paths = [Path(__file__), HERE / "CONTRACT.md", ROOT / "model_adapter/adapter.py", ROOT / "research/shock_radar/metrics.py"]
    codes = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in code_paths}
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "choices": choices,
                "config": CONFIG, "model_identity": identity, "provenance": data["provenance"],
                "frozen_january_sha256": before, "code_sha256": codes,
                "expected_asset_path_calls": 300,
                "quantile_method": "linear over ten individual per-path MDD values, per asset",
                "interpretation": "Empirical model sampling dispersion, NOT calibrated coverage or joint scenarios"}
    write_json(OUT / "manifest.json", manifest)
    samples_dir = OUT / "source_at_run"
    samples_dir.mkdir()
    for p in code_paths:
        with (samples_dir / p.name).open("xb") as f:
            f.write(p.read_bytes())
    summaries, started, actual_calls = [], time.perf_counter(), 0
    for ordinal, choice in enumerate(choices, 1):
        original = json.loads((BASE / "inference-original-v1" / (choice["event_id"] + ".json")).read_text())
        event = original["event"]
        window = market_window(data, event["index"])
        assert window["window_id"] == original["window_id"]
        record = {"event": event, "choice": choice, "window_id": window["window_id"],
                  "original_forecast_id": original["forecast_id"], "forecast_id": None,
                  "forecast": None, "runtime": None, "raw_paths": None, "status": "failed"}
        start = time.perf_counter()
        try:
            output = adapter.predict(window, CONFIG, "B-closure-ten-paths-" + event["event_id"])
            record.update(output)
            validate_forecast_output(output["forecast"], output["raw_paths"])
            assert len(output["forecast"]["paths"]) == 10
            record["forecast_id"] = freeze_forecast(output["forecast"]).identity
            record["status"] = "complete"
        except PredictionValidationError as exc:
            record.update(raw_paths=safe(exc.raw_paths), runtime=exc.runtime,
                          error={"type": type(exc).__name__, "issues": exc.issues})
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        asset_summaries = {}
        for asset in data["assets"]:
            if record["status"] == "complete":
                paths = [p["assets"][asset]["close"] for p in record["forecast"]["paths"]]
                vals = percentiles(window["histories"][asset][-1]["close"], paths)
                vals["path_zero_close_equals_original"] = paths[0] == original["method_close_paths"]["kronos_base"][asset]
            else:
                vals = {"p05": None, "p50": None, "p95": None, "n_expected": 10,
                        "n_valid": 0, "per_path_mdd": [], "reason": "event output unavailable; no resampling"}
            vals["forecast_id"] = record["forecast_id"]
            asset_summaries[asset] = vals
        calls = len((record.get("runtime") or {}).get("calls", []))
        actual_calls += calls
        record.update(elapsed_seconds=time.perf_counter()-start, asset_path_calls_recorded=calls,
                      asset_mdd_quantiles=asset_summaries)
        write_json(OUT / (event["event_id"] + ".json"), safe(record))
        summaries.append({"event_id": event["event_id"], "timestamp": event["timestamp"],
                          "choice": choice, "status": record["status"], "forecast_id": record["forecast_id"],
                          "original_forecast_id": record["original_forecast_id"],
                          "asset_mdd_quantiles": asset_summaries,
                          "asset_path_calls_recorded": calls, "elapsed_seconds": record["elapsed_seconds"]})
        print(json.dumps({"completed": ordinal, "total": 3, "status": record["status"],
                          "asset_path_calls": calls, "seconds": record["elapsed_seconds"]}), flush=True)
    assert frozen_inputs() == before, "Existing January evidence changed"
    assert {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in code_paths} == codes
    ending = KronosAdapter(model_path=MODEL, tokenizer_path=TOKENIZER, device=args.device).identity()
    assert ending == identity, "Original model changed during sampling"
    payload = {"schema_version": 1, "events": summaries, "event_count": 3,
               "status": "complete" if all(s["status"] == "complete" for s in summaries) else "partial",
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "model_identity": identity, "config": CONFIG, "expected_asset_path_calls": 300,
               "recorded_asset_path_calls": actual_calls, "end_to_end_seconds": time.perf_counter()-started,
               "quantile_method": "numpy method=linear", "quantile_units": "fractional max drawdown, multiply by100 for percent",
               "frozen_originals_unchanged": True, "model_identity_unchanged": True,
               "disclosure": "Ten samples per asset; p05/p50/p95 are uncalibrated model sampling dispersion, not95% confidence or coverage; no joint-portfolio use"}
    write_json(OUT / "uncertainty.json", payload)
    print(json.dumps({"status": payload["status"], "asset_path_calls": actual_calls,
                      "seconds": payload["end_to_end_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
