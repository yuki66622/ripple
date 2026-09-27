"""Run: python3 -m forecast_metrics.demo --output forecast_metrics/demo_report.json"""

import argparse
import json
from pathlib import Path

from .engine import PAIRING, evaluate_alerts, freeze_forecast, make_holdings, recalculate, report_payload


def fixture():
    """Fictional arithmetic fixture, not fetched or model-generated prices."""
    return {
        "model_revision": "fictional-hand-check-v1", "source": "synthetic_arithmetic_fixture",
        "prediction_run_id": "fictional-demo-run-1", "quote_currency": "USD",
        "pairing": PAIRING, "as_of": "2026-09-26T12:00:00Z", "interval_seconds": 60,
        "times": ["2026-09-26T12:01:00Z", "2026-09-26T12:02:00Z", "2026-09-26T12:03:00Z"],
        "spots": {"BTC": 100, "ETH": 50, "SOL": 10},
        "paths": [{"path_id": "example-1", "assets": {
            "BTC": {"close": [101, 98, 102], "high": [102, 100, 104], "low": [99, 97, 98]},
            "ETH": {"close": [50, 51, 52], "high": [51, 52, 53], "low": [49, 50, 51]},
            "SOL": {"close": [10, 9, 10], "high": [10.1, 9.5, 10.2], "low": [9.8, 8.9, 9]},
        }}],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    forecast = freeze_forecast(fixture())
    holdings = make_holdings(forecast)
    metrics = recalculate(forecast, holdings)
    report = report_payload(metrics, evaluate_alerts(metrics))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps({"source": "虚构算术示例", "forecast_id": forecast.identity,
                      "quantities": holdings.data["quantities"],
                      "BTC": metrics["paths"][0]["assets"]["BTC"],
                      "portfolio": metrics["paths"][0]["portfolio"],
                      "alerts": report["alerts"]["assets"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
