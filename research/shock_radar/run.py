"""Run every approved January event with original Kronos, preserving failures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from forecast_metrics.engine import freeze_forecast
from model_adapter import KronosAdapter, PredictionValidationError, validate_forecast_output

from .baselines import predict_baselines
from .calibration import OUTPUT
from .io import ROOT, digest, future_closes, load_january, market_window, write_json
from .metrics import aggregate_events, risk_metrics, score_event

METHODS = ["kronos_base", "no_propagation", "btc_beta", "historical_30"]
MODEL = ROOT / "scenario-lab/models/Kronos-base"
TOKENIZER = ROOT / "scenario-lab/models/Kronos-Tokenizer-base"


def _failure_safe(value):
    """Keep every failed raw value explicitly, with tagged nonfinite numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, dict):
        return {key: _failure_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_failure_safe(item) for item in value]
    return value


def _risk_table(spots, paths):
    return {a: risk_metrics(spots[a], closes) for a, closes in paths.items()}


def load_selection():
    selection = json.loads((OUTPUT / "selection-approved-k6.json").read_text())
    calibration = json.loads((OUTPUT / "calibration.json").read_text())
    raw = (OUTPUT / "catalog-k6.json").read_bytes()
    if digest(raw) != selection["catalog_sha256"] or selection["selected_k"] != 6:
        raise ValueError("approved selection/catalog mismatch")
    if selection.get("authorization") != "用 k=6 的全部 84 次，接受样本数超出原范围":
        raise ValueError("explicit 84-event amendment is required")
    events = [e for e in json.loads(raw)["events"] if e["eligible"]]
    if len(events) != 84 or [e["event_id"] for e in events] != selection["event_ids"]:
        raise ValueError("must run all and only the 84 eligible approved events")
    return selection, events, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    selection, events, calibration = load_selection()
    out = OUTPUT / "inference-original-v1"
    if out.exists():
        raise FileExistsError("inference artifacts already exist; no implicit retry/resampling")
    data = load_january()
    if data["provenance"] != calibration["provenance"]:
        raise ValueError("data changed after calibration")
    adapter = KronosAdapter(model_path=MODEL, tokenizer_path=TOKENIZER, device=args.device)
    identity = adapter.identity()  # Fail clearly before the batch if backend unavailable.
    out.mkdir()
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "event_count": len(events),
                "selection": selection, "model_identity": identity, "model_path": str(MODEL),
                "tokenizer_path": str(TOKENIZER), "device": args.device, "provenance": data["provenance"],
                "method_names": METHODS, "predict_config": selection["predict_config"],
                "replay_selection_policy": json.loads((OUTPUT / "replay-selection-policy.json").read_text()),
                "code_sha256": {name: digest((ROOT / "research/shock_radar" / name).read_bytes())
                                  for name in ("io.py", "baselines.py", "metrics.py", "run.py")}}
    write_json(out / "run_manifest.json", manifest)
    # Preserve the exact executable sources even if later work changes modules.
    sources = out / "code_at_run"
    sources.mkdir()
    for name, expected_hash in manifest["code_sha256"].items():
        raw = (ROOT / "research/shock_radar" / name).read_bytes()
        if digest(raw) != expected_hash:
            raise ValueError("research code changed while freezing run")
        with (sources / name).open("xb") as stream:
            stream.write(raw)
    scores, timings = [], []
    started = time.perf_counter()
    for n, event in enumerate(events, 1):
        event_started = time.perf_counter()
        window = market_window(data, event["index"])
        spots = {a: window["histories"][a][-1]["close"] for a in data["assets"]}
        baseline = predict_baselines(window, event["origin_assets"])
        predictions = {method: None for method in METHODS}
        paths = dict(baseline["paths"])
        for method, values in baseline["paths"].items():
            predictions[method] = _risk_table(spots, values)
        artifact = {"event": event, "window_id": window["window_id"], "forecast_id": None,
                    "forecast": None, "raw_paths": None, "runtime": None,
                    "baseline_metadata": baseline["metadata"], "errors": dict(baseline["errors"])}
        try:
            output = adapter.predict(window, selection["predict_config"], "shock-k6-v1-" + event["event_id"])
            # Preserve model output even if a downstream validation step fails.
            artifact.update(output)
            validate_forecast_output(output["forecast"], output["raw_paths"])
            frozen = freeze_forecast(output["forecast"])
            # The frozen protocol uses exactly one path; no averaging prices.
            assert len(output["forecast"]["paths"]) == 1
            paths["kronos_base"] = {a: series["close"] for a, series in output["forecast"]["paths"][0]["assets"].items()}
            predictions["kronos_base"] = _risk_table(spots, paths["kronos_base"])
            artifact.update(output, forecast_id=frozen.identity)
        except PredictionValidationError as exc:
            artifact.update(raw_paths=_failure_safe(exc.raw_paths), runtime=exc.runtime,
                            raw_paths_encoding="nonfinite floats explicitly tagged when present")
            artifact["errors"]["kronos_base"] = {"type": type(exc).__name__, "issues": exc.issues}
        except Exception as exc:
            artifact["errors"]["kronos_base"] = {"type": type(exc).__name__, "message": str(exc)}
        # Truth is loaded separately AFTER forecasting; never given to the model/baselines.
        actual_paths = future_closes(data, event["index"])
        actual = _risk_table(spots, actual_paths)
        score = score_event(event, predictions, actual, data["assets"])
        artifact.update(predicted_risk=predictions, actual_risk=actual,
                        method_close_paths=paths, score=score)
        elapsed = time.perf_counter() - event_started
        artifact["end_to_end_seconds"] = elapsed
        write_json(out / (event["event_id"] + ".json"), artifact)
        scores.append(score)
        timings.append(elapsed)
        print(json.dumps({"completed": n, "total": len(events), "timestamp": event["timestamp"],
                          "kronos_status": score["methods"]["kronos_base"]["status"],
                          "seconds": round(elapsed, 3)}, ensure_ascii=False), flush=True)
    # New adapter object hashes source files anew; it does not load/run a model.
    ending_identity = KronosAdapter(model_path=MODEL, tokenizer_path=TOKENIZER, device=args.device).identity()
    stable = ending_identity == identity
    report = aggregate_events(scores, methods=METHODS)
    report.update({"status": "complete" if stable else "invalid_model_changed_during_run",
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(), "event_count": len(events),
                   "model_identity_unchanged": stable, "model_identity": identity,
                   "model_calls": len(events), "end_to_end_seconds": time.perf_counter()-started,
                   "per_event_seconds": timings, "predict_config": selection["predict_config"]})
    write_json(out / "summary.json", report)
    print(json.dumps({"status": report["status"], "events": len(events),
                      "paired": report["groups"]["all"]["common_paired_event_count"],
                      "seconds": report["end_to_end_seconds"]}), flush=True)
    return 0 if stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
