"""Predeclared three-window acceptance probe, including rejected forecasts."""
import json
from pathlib import Path

from data_pipeline import load_window
from forecast_metrics.engine import freeze_forecast, make_holdings, recalculate, evaluate_alerts
from model_adapter import KronosAdapter, PredictionValidationError

ORIGINS = ["2025-01-01T12:00:00Z", "2025-01-15T12:00:00Z", "2025-01-31T12:00:00Z"]


def main():
    adapter = KronosAdapter()
    records = {"purpose": "bounded integration acceptance, not forecast quality evaluation",
               "planned_origins": ORIGINS, "runs": []}
    target = Path("model_adapter/evidence/three-window-smoke-mps.json")
    for i, origin in enumerate(ORIGINS):
        window = load_window(as_of=origin)
        record = {"as_of": origin, "window_id": window["window_id"]}
        try:
            prediction = adapter.predict(window, {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926}, f"three-window-smoke-{i}")
            forecast = freeze_forecast(prediction["forecast"])
            metrics = recalculate(forecast, make_holdings(forecast))
            alerts = evaluate_alerts(metrics)
            record.update(status="valid", **prediction, metrics=metrics, alerts=alerts, forecast_id=forecast.identity)
        except PredictionValidationError as exc:
            record.update(status="rejected", raw_paths=exc.raw_paths, runtime=exc.runtime, issues=exc.issues)
        records["runs"].append(record)
        target.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"as_of": origin, "status": record["status"], "inference_ms": record["runtime"]["inference_ms"], "issues": len(record.get("issues", []))}), flush=True)


if __name__ == "__main__":
    main()
