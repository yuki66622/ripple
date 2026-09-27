"""Explicit synthetic ledger/forecast fixtures; no real model inference."""

import copy
import tempfile
import unittest
from unittest.mock import patch

from data_pipeline import load_window
from evaluation.runner import build_manifest, run_task
from evaluation.scoring import score_forecast
from evaluation.test_evaluation import FixtureAdapter, fixture_forecast
from evaluation.volume_quality import extract_volume_quality, summarize_volume_quality
from forecast_metrics.engine import freeze_forecast
from model_adapter import build_volume_quality


def ledger_fixture(index, *, invalid=0, total=90, error=.01, status="scored", model="TEST_ONLY_model", seed=7):
    quality = {"policy": "unused-volume-audit-v1", "status": "volume_forecast_unavailable" if invalid else "valid",
               "total_candles": total, "volume_invalid_count": invalid, "volume_invalid_rate": invalid / total}
    origin = f"2025-01-01T{index:02d}:00:00Z"
    rows = [{"as_of": origin, "symbol": asset, "forecast_id": f"TEST_ONLY_forecast_{index}",
             "model": model, "metric": metric, "predicted": 0, "actual": error,
             "absolute_error": error, "status": "scored", "regime": "unclassified"}
            for asset in ("BTC", "ETH", "SOL") for metric in ("terminal_return", "volatility")] if status == "scored" else []
    return {
        "task_id": f"TEST_ONLY_task_{index}", "window_id": f"TEST_ONLY_window_{index}", "as_of": origin,
        "forecast_id": f"TEST_ONLY_forecast_{index}", "status": status,
        "assets": ["BTC", "ETH", "SOL"],
        "profile_id": "TEST_ONLY_profile", "source": "TEST_ONLY_source", "quote_currency": "USDT",
        "model_spec": {"model_label": model}, "predict_config": {"lookback": 256, "horizon": 30, "path_count": 1, "seed": seed},
        "result": {"forecast": {"model_revision": "TEST_ONLY_revision_" + model}},
        "volume_quality": quality, "volume_quality_status": "validated",
        "evaluation": {"status": status, "rows": rows, "model": model},
    }


class NegativeFixtureAdapter(FixtureAdapter):
    def predict(self, *args):
        result = super().predict(*args)
        series = result["raw_paths"][0]["assets"]["BTC"]
        series["volume"][0] = -1.0
        series["amount"][0] = -2.0  # Both negative still count this candle once.
        result["forecast"]["volume_quality"] = build_volume_quality(result["raw_paths"], result["forecast"]["times"])
        return result


class VolumeEvidenceTests(unittest.TestCase):
    def test_same_candle_both_negative_counts_once_and_price_can_score(self):
        manifest = build_manifest(limit=1, horizon=2, model_label="TEST_ONLY_negative_volume")
        with tempfile.TemporaryDirectory() as directory:
            result = run_task(manifest["tasks"][0], directory, adapter=NegativeFixtureAdapter())
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["volume_quality_status"], "validated")
        self.assertEqual(result["volume_quality"]["volume_invalid_count"], 1)
        self.assertEqual(result["volume_quality"]["volume_invalid_rate"], 1 / 6)
        self.assertEqual(result["correction_quality"]["correction_rate"], 0)
        self.assertEqual(result["result"]["raw_paths"][0]["assets"]["BTC"]["volume"][0], -1)

    def test_missing_new_volume_audit_fails_scoring(self):
        window = load_window()
        forecast = fixture_forecast(window)
        forecast.pop("volume_quality")
        frozen = freeze_forecast(forecast)
        result = score_forecast(window, frozen.data, frozen.identity)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["volume_quality_status"], "missing")
        self.assertEqual(result["correction_quality_status"], "unvalidated")

    def test_pending_retains_volume_evidence(self):
        window = load_window()
        result = NegativeFixtureAdapter().predict(window, {"horizon": 2, "path_count": 1}, "TEST_ONLY")
        frozen = freeze_forecast(result["forecast"])
        with patch("evaluation.scoring.load_truth", return_value=None):
            scored = score_forecast(window, frozen.data, frozen.identity)
        self.assertEqual(scored["status"], "pending_truth")
        self.assertEqual(scored["volume_quality_status"], "validated")
        self.assertEqual(scored["volume_quality"]["volume_invalid_count"], 1)

    def test_scalar_invalid_denominator_rejected(self):
        value = ledger_fixture(1)["volume_quality"]
        value["volume_invalid_rate"] = .8
        with self.assertRaises(ValueError):
            extract_volume_quality({"volume_quality": value})

    def test_rate_groups_use_existing_normalized_errors_not_other_rows(self):
        data = [ledger_fixture(1, invalid=0, error=.01), ledger_fixture(2, invalid=45, error=.05)]
        # This large raw-price error is deliberately irrelevant to the association.
        data[0]["evaluation"]["rows"].append({"model": "TEST_ONLY_model", "metric": "predicted_high", "symbol": "BTC",
                                               "absolute_error": 100000, "status": "scored"})
        result = summarize_volume_quality(data)
        group = result["by_group"][0]
        self.assertEqual(group["coverage"]["eligible_price_error_windows"], 2)
        self.assertAlmostEqual(group["low_vs_high"]["comparison"]["right_minus_left_mean_absolute_error"]["terminal_return"], .04)
        self.assertEqual(group["low_vs_high"]["comparison"]["status"], "insufficient_evidence")
        self.assertFalse(result["automatic_alarm"])

    def test_same_profile_model_config_grouping_and_threshold(self):
        data = [ledger_fixture(1, invalid=18), ledger_fixture(2, invalid=30),
                ledger_fixture(3, invalid=45, model="TEST_ONLY_other"), ledger_fixture(4, seed=8)]
        default = summarize_volume_quality(data)
        self.assertEqual(len(default["by_group"]), 3)
        group = next(g for g in default["by_group"] if g["coverage"]["input_records"] == 2)
        self.assertEqual(group["low_vs_high"]["high"]["window_count"], 1)
        changed = summarize_volume_quality(data[:2], high_rate_threshold=.5)["by_group"][0]
        self.assertEqual(changed["low_vs_high"]["high"]["window_count"], 0)

    def test_exact_reruns_deduplicate_without_increasing_sample_count(self):
        first = ledger_fixture(1, invalid=30)
        repeated = copy.deepcopy(first)
        repeated["forecast_id"] = "TEST_ONLY_different_run"
        repeated["task_id"] = "TEST_ONLY_different_task"
        summary = summarize_volume_quality([first, repeated])["by_group"][0]
        self.assertEqual(summary["coverage"]["distinct_windows"], 1)
        self.assertEqual(summary["coverage"]["duplicate_records"], 1)
        self.assertEqual(summary["coverage"]["eligible_price_error_windows"], 1)
        self.assertEqual(summary["volume_summary"]["total_candles"], 90)

    def test_conflicting_reruns_do_not_choose_better_score_or_throw(self):
        first = ledger_fixture(1, invalid=30, error=.01)
        repeated = ledger_fixture(1, invalid=30, error=.5)
        summary = summarize_volume_quality([first, repeated])["by_group"][0]
        self.assertEqual(summary["coverage"]["conflicting_duplicate_windows"], 1)
        self.assertEqual(summary["coverage"]["eligible_price_error_windows"], 0)
        self.assertEqual(len(summary["observations"][0]["attempts"]), 2)

    def test_pending_failed_and_incomplete_coverage_remains_visible(self):
        data = [ledger_fixture(1, invalid=30), ledger_fixture(2, invalid=30, status="pending_truth"),
                ledger_fixture(3, status="failed"), ledger_fixture(4, status="incomplete_truth")]
        data[2]["volume_quality"] = None
        data[2]["volume_quality_status"] = "missing"
        group = summarize_volume_quality(data)["by_group"][0]
        self.assertEqual(group["coverage"]["distinct_window_status_counts"],
                         {"scored": 1, "pending_truth": 1, "failed": 1, "incomplete_truth": 1})
        self.assertEqual(group["coverage"]["eligible_price_error_windows"], 1)
        self.assertEqual(group["coverage"]["volume_audit_status_counts"]["missing"], 1)


if __name__ == "__main__":
    unittest.main()
