"""Apply the approved field policy to saved raw output, without inference."""
import hashlib
import json
from pathlib import Path

from forecast_metrics.engine import freeze_forecast, make_holdings, recalculate, evaluate_alerts
from model_adapter import validate_forecast_output
from model_adapter.adapter import ADAPTER_REVISION, assemble_forecast


def main():
    source = Path("demo_app/runtime/artifacts/6e7405b69a5070d5c6bea83e3e5e50b4cf16ba64799d883a094e412450e90d24.json")
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != source.stem:
        raise ValueError("original artifact checksum mismatch")
    saved = json.loads(source_bytes)
    raw = saved["rejected_output"]["raw_paths"]
    original_raw = json.dumps(raw, sort_keys=True, allow_nan=False)
    original_runtime = saved["rejected_output"]["runtime"]
    forecast = assemble_forecast(
        saved["window"], saved["config"],
        "saved-output-reprocess:" + saved["job_id"] + ":unused-volume-audit-v1",
        raw, model_revision=original_runtime["identity"]["model_revision"],
    )
    validate_forecast_output(forecast, raw)
    frozen = freeze_forecast(forecast)
    metrics = recalculate(frozen, make_holdings(frozen))
    alerts = evaluate_alerts(metrics)
    unchanged_raw = json.dumps(raw, sort_keys=True, allow_nan=False) == original_raw
    unchanged_source = source.read_bytes() == source_bytes
    unchanged_closes = all(p["assets"][asset]["close"] == raw_path["assets"][asset]["close"]
                           for p, raw_path in zip(forecast["paths"], raw)
                           for asset in p["assets"])
    if not (unchanged_raw and unchanged_source and unchanged_closes):
        raise RuntimeError("read-only reprocessing changed raw evidence or close values")
    artifact = {
        "execution": "saved_raw_reprocessing_no_inference",
        "new_model_calls": 0, "source_artifact_sha256": digest,
        "source_artifact": str(source), "postprocess_adapter_revision": ADAPTER_REVISION,
        "window": saved["window"], "config": saved["config"],
        "forecast": forecast, "forecast_id": frozen.identity, "raw_paths": raw,
        "metrics": metrics, "alerts": alerts,
        "original_runtime_not_a_new_measurement": original_runtime,
        "checks": {"raw_unchanged": unchanged_raw, "source_file_unchanged": unchanged_source,
                   "close_values_unchanged": unchanged_closes, "full_validation_passed": True,
                   "freeze_and_price_metrics_passed": True},
    }
    target = Path("model_adapter/evidence/unused-volume-audit-v1-saved-kraken.json")
    content = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        if target.read_text() != content:
            raise ValueError("existing reprocessing evidence differs; do not overwrite")
    else:
        with target.open("x") as stream:
            stream.write(content)
    print(json.dumps({"evidence": str(target), "new_model_calls": 0,
                      "volume_quality": {k: v for k, v in forecast["volume_quality"].items() if k != "records"},
                      "ohlc_corrections": {k: v for k, v in forecast["ohlc_corrections"].items() if k != "records"},
                      "checks": artifact["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
