"""No Kronos inference: exact arithmetic, real history, sktime baselines/folds."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_pipeline import DataError, load_truth, load_window
from forecast_metrics.engine import PAIRING, freeze_forecast
from evaluation.scoring import score_forecast
from evaluation.runner import build_manifest, run_task, save_manifest, summarize
from evaluation.corrections import extract_quality, summarize_quality


def fixture_forecast(window, horizon=2, path_factors=None, observed=None):
    from datetime import datetime, timedelta
    anchor = datetime.fromisoformat(window["as_of"].replace("Z", "+00:00"))
    spots = {s: window["histories"][s][-1]["close"] for s in window["assets"]}
    paths = []
    for i, factors in enumerate(path_factors or [[1] * horizon]):
        assets = {}
        for s, spot in spots.items():
            if observed:
                assets[s] = {key: observed["assets"][s][key] for key in ("close", "high", "low")}
            else:
                closes = [spot * factor for factor in factors]
                assets[s] = {"close": closes, "high": closes, "low": closes}
        paths.append({"path_id": "test-path-" + str(i), "assets": assets})
    forecast = {
        "model_revision": "TEST_ONLY_numerical_fixture", "prediction_run_id": "TEST_ONLY",
        "source": window["source"], "quote_currency": window["quote_currency"], "pairing": PAIRING,
        "as_of": window["as_of"], "interval_seconds": 60, "spots": spots, "paths": paths,
        "times": [(anchor + timedelta(minutes=i)).isoformat().replace("+00:00", "Z") for i in range(1, horizon + 1)],
    }
    add_fixture_audit(forecast)
    return forecast


def add_fixture_audit(forecast):
    """Explicit numerical fixture: unchanged candles with full audit and raw OHLCVA."""
    raw_paths, records = [], []
    for path in forecast["paths"]:
        raw = {"path_id": path["path_id"], "assets": {}}
        for symbol, series in path["assets"].items():
            raw["assets"][symbol] = {
                **copy.deepcopy(series), "open": list(series["close"]),
                "volume": [1.0] * len(series["close"]), "amount": list(series["close"]),
            }
            for i, timestamp in enumerate(forecast["times"]):
                candle = {key: raw["assets"][symbol][key][i] for key in ("open", "high", "low", "close")}
                records.append({
                    "path_id": path["path_id"], "asset": symbol, "time": timestamp,
                    "original": dict(candle), "corrected": dict(candle),
                    "high_adjustment_bps": 0.0, "low_adjustment_bps": 0.0,
                    "max_adjustment_bps": 0.0, "was_corrected": False,
                })
        raw_paths.append(raw)
    forecast["ohlc_corrections"] = {
        "policy": "containment-expand-v1", "bps_denominator": "original_close",
        "total_candles": len(records), "corrected_candles": 0,
        "correction_rate": 0.0, "correction_rate_pct": 0.0,
        "max_adjustment_bps": 0.0, "records": records,
    }
    from model_adapter import build_volume_quality
    forecast["volume_quality"] = build_volume_quality(raw_paths, forecast["times"])
    return raw_paths


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = load_window()

    def test_oracle_fixture_exact_zero_and_real_naive(self):
        truth = load_truth(self.window)
        frozen = freeze_forecast(fixture_forecast(self.window, 30, observed=truth))
        result = score_forecast(self.window, frozen.data, frozen.identity)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(len(result["rows"]), 27)
        for row in result["rows"]:
            self.assertEqual(set(row), {"as_of", "symbol", "forecast_id", "model", "metric", "predicted", "actual", "absolute_error", "status", "regime"})
            if row["model"] == "Kronos-base":
                self.assertEqual(row["absolute_error"], 0)
            elif row["model"] == "naive-last":
                self.assertEqual(row["predicted"], 0)
            self.assertEqual(row["regime"], "unclassified")
        self.assertEqual(result["quality_status"], "insufficient_evidence")
        self.assertEqual(result["correction_quality_status"], "validated")
        self.assertEqual(result["volume_quality_status"], "validated")

    def test_metrics_before_path_aggregation(self):
        frozen = freeze_forecast(fixture_forecast(self.window, path_factors=[[1.1, 1], [.9, 1]]))
        result = score_forecast(self.window, frozen.data, frozen.identity)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["path_count"], 2)
        vol = next(row["predicted"] for row in result["rows"] if row["model"] == "Kronos-base" and row["symbol"] == "BTC" and row["metric"] == "volatility")
        # Averaging prices first would yield a flat [spot,spot] forecast and 0 volatility.
        self.assertGreater(vol, .1)
        paths = result["prediction_metrics"]["paths"]
        self.assertAlmostEqual(vol, sum(path["assets"]["BTC"]["volatility"] for path in paths) / 2)

    def test_pending_truth_and_incomplete_truth_are_distinct(self):
        frozen = freeze_forecast(fixture_forecast(self.window))
        with patch("evaluation.scoring.load_truth", return_value=None):
            result = score_forecast(self.window, frozen.data, frozen.identity)
        self.assertEqual(result["status"], "pending_truth")
        self.assertEqual(result["pending_count"], 1)
        self.assertEqual(result["sample_count"], 0)
        self.assertTrue(all(row["actual"] is None for row in result["rows"]))
        with patch("evaluation.scoring.load_truth", side_effect=DataError("missing candle")):
            result = score_forecast(self.window, frozen.data, frozen.identity)
        self.assertEqual(result["status"], "incomplete_truth")
        self.assertEqual(result["incomplete_count"], 1)

    def test_wrong_forecast_identity_fails(self):
        frozen = freeze_forecast(fixture_forecast(self.window))
        result = score_forecast(self.window, frozen.data, "wrong-id")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["sample_count"], 0)

    def test_wrong_truth_grid_cannot_score(self):
        frozen = freeze_forecast(fixture_forecast(self.window))
        truth = load_truth(self.window, horizon=2)
        truth["times"].pop()
        with patch("evaluation.scoring.load_truth", return_value=truth):
            result = score_forecast(self.window, frozen.data, frozen.identity)
        self.assertEqual(result["status"], "incomplete_truth")

    def test_historical_baseline_uses_past_only(self):
        frozen = freeze_forecast(fixture_forecast(self.window))
        first = score_forecast(self.window, frozen.data, frozen.identity)
        truth = load_truth(self.window, horizon=2)
        for series in truth["assets"].values():
            for key in ("close", "high", "low"):
                series[key] = [value * 2 for value in series[key]]
        with patch("evaluation.scoring.load_truth", return_value=truth):
            second = score_forecast(self.window, frozen.data, frozen.identity)
        get_baseline = lambda value: [row["predicted"] for row in value["rows"] if row["model"] == "historical-volatility"]
        self.assertEqual(get_baseline(first), get_baseline(second))
        self.assertNotEqual(first["rows"][0]["actual"], second["rows"][0]["actual"])


class FixtureAdapter:
    def predict(self, window, config, prediction_run_id):
        forecast = fixture_forecast(window, config["horizon"],
                                    path_factors=[[1] * config["horizon"]] * config["path_count"])
        forecast["prediction_run_id"] = prediction_run_id
        raw_paths = add_fixture_audit(forecast)
        return {"forecast": forecast, "runtime": {"test_fixture": True}, "raw_paths": raw_paths}


class RejectedFixture(Exception):
    def __init__(self):
        self.raw_paths = [{"invalid": float("nan")}]
        self.runtime = {"test_fixture": True}
        self.issues = ["test nonfinite value"]
        super().__init__("TEST_ONLY rejected sample")


class FailingAdapter:
    def predict(self, *args):
        raise RejectedFixture()


class MissingPathAdapter(FixtureAdapter):
    def predict(self, *args):
        result = super().predict(*args)
        result["forecast"]["paths"].pop()
        return result


class MissingAuditAdapter(FixtureAdapter):
    def predict(self, *args):
        result = super().predict(*args)
        result["forecast"].pop("ohlc_corrections")
        return result


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_manifest(limit=2, stride=30, path_count=2, model_label="TEST_ONLY_fixture")

    def test_sktime_folds_identical_and_no_future_payload(self):
        manifest = self.manifest
        self.assertEqual(manifest["available_windows"], 1479)
        self.assertEqual(manifest["selected_count"], 2)
        self.assertEqual([task["origin_index"] for task in manifest["tasks"]], [255, 285])
        for task in manifest["tasks"]:
            self.assertNotIn("truth", task)
            self.assertNotIn("truth", task["window"])
            self.assertEqual(len(task["window"]["histories"]["BTC"]), 256)
            self.assertEqual(task["window"]["histories"]["BTC"][-1]["time"], task["window"]["as_of"])
        second = build_manifest(limit=2, stride=30, path_count=2, model_label="TEST_ONLY_fixture")
        self.assertEqual(manifest, second)

    def test_failed_raw_path_and_denominator_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            failed = run_task(self.manifest["tasks"][0], directory, adapter=FailingAdapter())
            passed = run_task(self.manifest["tasks"][1], directory, adapter=FixtureAdapter())
            saved = json.loads(Path(failed["artifact_path"]).read_text())
            self.assertEqual(saved["raw_paths"][0]["invalid"], "NaN")
            self.assertIn("raw_nonfinite_encoding", saved)
            summary = summarize([failed, passed], requested_count=3)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["not_run_count"], 1)
        self.assertEqual(summary["requested_count"], 3)

    def test_remote_inference_only_and_repeat_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            first = run_task(self.manifest["tasks"][0], directory, adapter=FixtureAdapter(), score=False)
            second = run_task(self.manifest["tasks"][0], directory, adapter=FixtureAdapter(), score=False)
            self.assertEqual(first["status"], "predicted")
            self.assertNotIn("evaluation", first)
            self.assertNotEqual(first["artifact_path"], second["artifact_path"])
            self.assertTrue(Path(first["artifact_path"]).is_file())
            self.assertEqual(first["forecast_id"], second["forecast_id"])
            with self.assertRaisesRegex(ValueError, "duplicate task"):
                summarize([first, second])

    def test_remote_result_must_include_all_requested_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_task(self.manifest["tasks"][0], directory, adapter=MissingPathAdapter(), score=False)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["stage"], "validate_forecast_output")
            self.assertEqual(result["correction_quality_status"], "unvalidated")
            self.assertIn("result", result)

    def test_new_results_cannot_publish_without_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_task(self.manifest["tasks"][0], directory, adapter=MissingAuditAdapter(), score=False)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["stage"], "validate_forecast_output")
            self.assertEqual(result["correction_quality_status"], "missing")

    def test_changed_task_identity_rejected_before_model(self):
        task = copy.deepcopy(self.manifest["tasks"][0])
        task["predict_config"]["seed"] += 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "identity"):
                run_task(task, directory, adapter=FixtureAdapter())

    def test_manifest_cannot_silently_replace_different_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = save_manifest(self.manifest, directory)
            self.assertEqual(save_manifest(self.manifest, directory), first)
            changed = copy.deepcopy(self.manifest)
            changed["stride"] = 10
            with self.assertRaises(FileExistsError):
                save_manifest(changed, directory)


class CorrectionQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = load_window()

    def audited(self):
        forecast = fixture_forecast(self.window)
        add_fixture_audit(forecast)
        return freeze_forecast(forecast)

    def test_pending_and_failed_keep_validated_structural_evidence(self):
        frozen = self.audited()
        with patch("evaluation.scoring.load_truth", return_value=None):
            pending = score_forecast(self.window, frozen.data, frozen.identity)
        self.assertEqual(pending["status"], "pending_truth")
        self.assertEqual(pending["correction_quality_status"], "validated")
        self.assertEqual(pending["correction_quality"], {
            "policy": "containment-expand-v1", "total_candles": 6,
            "corrected_candles": 0, "correction_rate": 0.0,
            "correction_rate_pct": 0.0, "max_adjustment_bps": 0.0,
        })
        failed = score_forecast(self.window, frozen.data, "wrong identity")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["correction_quality"], pending["correction_quality"])
        self.assertEqual(failed["quality_status"], "insufficient_evidence")

    def test_corrupt_metadata_is_not_a_valid_quality_signal(self):
        forecast = self.audited().data
        forecast["ohlc_corrections"]["correction_rate"] = .5
        frozen = freeze_forecast(forecast)
        result = score_forecast(self.window, frozen.data, frozen.identity)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["correction_quality_status"], "invalid")
        self.assertIsNone(result["correction_quality"])
        with self.assertRaises(ValueError):
            extract_quality(forecast)

    def test_record_validator_failure_preserves_untrusted_summary(self):
        forecast = self.audited().data
        forecast["ohlc_corrections"]["records"][0]["corrected"]["close"] *= 2
        frozen = freeze_forecast(forecast)
        result = score_forecast(self.window, frozen.data, frozen.identity)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["correction_quality_status"], "unvalidated")
        self.assertEqual(result["correction_quality"]["total_candles"], 6)

    @staticmethod
    def artifact(model, revision, task_id, window_id, paths, corrected):
        # Explicit scalar ledger fixture; publication tests above use full raw audits.
        total = paths * 3 * 2
        return {
            "task_id": task_id, "window_id": window_id, "model_spec": {"model_label": model},
            "predict_config": {"lookback": 256, "horizon": 2, "path_count": paths, "seed": 7},
            "result": {"forecast": {"model_revision": revision}}, "status": "scored",
            "artifact_path": "TEST_ONLY_scalar_fixture", "correction_quality_status": "validated",
            "correction_quality": {"policy": "containment-expand-v1", "total_candles": total,
                "corrected_candles": corrected, "correction_rate": corrected / total,
                "correction_rate_pct": corrected / total * 100,
                "max_adjustment_bps": 4 if corrected else 0},
        }

    def test_weighted_candle_denominators_not_average_percentages(self):
        values = [self.artifact("base", "r1", "task1", "window1", 1, 6),
                  self.artifact("base", "r1", "task2", "window2", 2, 0)]
        result = summarize(values)["correction_quality"]["by_model"][0]
        self.assertEqual(result["total_candles"], 18)
        self.assertEqual(result["corrected_candles"], 6)
        self.assertAlmostEqual(result["correction_rate"], 1 / 3)
        self.assertNotEqual(result["correction_rate"], .5)

    def test_same_window_rerun_counts_corrections_once_without_throwing(self):
        first = self.artifact("base", "r1", "task1", "window1", 1, 2)
        repeated = copy.deepcopy(first)
        repeated["task_id"] = "task2"
        repeated["forecast_id"] = "TEST_ONLY_new_run"
        result = summarize_quality([first, repeated])["by_model"][0]
        self.assertEqual(result["attempted_forecasts"], 2)
        self.assertEqual(result["forecast_count"], 1)
        self.assertEqual(result["duplicate_records"], 1)
        self.assertEqual(result["total_candles"], 6)

    def test_conflicting_correction_reruns_are_retained_not_selected(self):
        first = self.artifact("base", "r1", "task1", "window1", 1, 2)
        repeated = self.artifact("base", "r1", "task2", "window1", 1, 0)
        result = summarize_quality([first, repeated])["by_model"][0]
        self.assertEqual(result["forecast_count"], 0)
        self.assertEqual(result["conflicting_duplicate_workloads"], 1)
        self.assertIsNone(result["correction_rate"])
        self.assertEqual(len(result["conflicts"][0]["attempts"]), 2)

    def test_models_separate_and_comparison_uses_matched_settings_only(self):
        values = [self.artifact("base", "r1", "task1", "window1", 1, 6),
                  self.artifact("base", "r1", "task2", "window2", 2, 0),
                  self.artifact("tuned", "r2", "task3", "window1", 1, 3),
                  self.artifact("tuned", "r2", "task4", "window2", 1, 0)]
        # A fair weight comparison requires explicit matching runtime controls;
        # legacy scalar-only ledgers remain descriptive, not comparable.
        from evaluation.test_comparison_compatibility import controlled_artifact
        for value in values:
            value["result"]["runtime"] = controlled_artifact(
                revision=value["result"]["forecast"]["model_revision"])["result"]["runtime"]
        result = summarize_quality(values)
        self.assertEqual(len(result["by_model"]), 2)
        matched = result["matched_comparisons"][0]
        self.assertEqual(matched["matched_task_count"], 1)
        self.assertEqual(matched["a"]["total_candles"], 6)
        self.assertEqual(matched["b"]["total_candles"], 6)
        self.assertEqual(matched["a"]["correction_rate"], 1)
        self.assertEqual(matched["b"]["correction_rate"], .5)

    def test_invalid_and_missing_audits_are_not_counted_as_zero_corrections(self):
        missing = self.artifact("base", "r1", "task1", "window1", 1, 0)
        missing["correction_quality"] = None
        missing["correction_quality_status"] = "missing"
        invalid = self.artifact("base", "r1", "task2", "window2", 1, 0)
        invalid["correction_quality_status"] = "invalid"
        result = summarize_quality([missing, invalid])
        self.assertTrue(all(group["total_candles"] == 0 for group in result["by_model"]))
        self.assertTrue(all(group["correction_rate"] is None for group in result["by_model"]))
        self.assertEqual(sum(group["missing_audit_count"] for group in result["by_model"]), 1)
        self.assertEqual(sum(group["invalid_audit_count"] for group in result["by_model"]), 1)


if __name__ == "__main__":
    unittest.main()
