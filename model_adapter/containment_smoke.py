"""Same-window, same-seed real acceptance for authorized containment policy."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from data_pipeline import load_window
from forecast_metrics.engine import freeze_forecast, make_holdings, recalculate, evaluate_alerts
from model_adapter import KronosAdapter, validate_corrections


def main():
    old = json.loads(Path("model_adapter/evidence/benchmark-mps-256-30.json").read_text())
    window = load_window(as_of=old["as_of"])
    if window["window_id"] != old["window_id"]:
        raise RuntimeError("input window changed; stop rather than silently compare different inputs")
    result = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "window_id": window["window_id"], "as_of": window["as_of"],
              "source": window["source"], "policy": "containment-expand-v1", "runs": []}
    adapter = KronosAdapter(device="mps")
    target = Path("model_adapter/evidence/containment-expand-v1-mps-256-30.json")
    for count in (1, 2):
        config = {"lookback": 256, "horizon": 30, "path_count": count, "seed": 20260926}
        started = time.perf_counter()
        prediction = adapter.predict(window, config, f"containment-smoke-{count}")
        validate_corrections(prediction["forecast"], prediction["raw_paths"])
        frozen = freeze_forecast(prediction["forecast"])
        holdings = make_holdings(frozen)
        metrics = recalculate(frozen, holdings)
        alerts = evaluate_alerts(metrics)
        old_raw = next(run["raw_paths"] for run in old["runs"] if run["runtime"]["path_count"] == count)
        unchanged = prediction["raw_paths"] == old_raw
        record = {"config": config, "status": "valid", **prediction,
                  "forecast_id": frozen.identity, "metrics": metrics, "alerts": alerts,
                  "wall_ms": (time.perf_counter() - started) * 1000,
                  "raw_identical_to_pre_correction_evidence": unchanged,
                  "freeze_and_metrics_passed": True}
        result["runs"].append(record)
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        audit = prediction["forecast"]["ohlc_corrections"]
        print(json.dumps({"paths": count, "status": "valid", "load_ms": prediction["runtime"]["load_ms"],
                          "inference_ms": prediction["runtime"]["inference_ms"],
                          "correction_ms": prediction["runtime"]["correction_ms"],
                          "corrected_candles": audit["corrected_candles"],
                          "total_candles": audit["total_candles"],
                          "correction_rate_pct": audit["correction_rate_pct"],
                          "max_adjustment_bps": audit["max_adjustment_bps"],
                          "raw_identical_to_previous": unchanged}), flush=True)


if __name__ == "__main__":
    main()
