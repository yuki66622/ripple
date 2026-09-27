"""Synthetic arithmetic/accounting fixtures only; never run model or network."""

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.comparison import ASSETS, _canonical, compare_batch, compare_records, competition_top, main
from forecast_metrics.engine import ASSET_METRICS


IDENTITY = {"model_revision": "TEST_ONLY_base", "tokenizer_revision": "TEST_ONLY_tokenizer",
            "adapter_revision": "TEST_ONLY_adapter", "sampling": {"top_p": .9, "T": 1, "sample_count": 1},
            "output_policy": "containment-expand-v1", "volume_policy": "unused-volume-audit-v1",
            "backend": "TEST_ONLY_cpu", "torch_version": "TEST_ONLY", "sktime_version": "TEST_ONLY",
            "model_timestamp": "UTC candle start", "output_timestamp": "UTC candle end"}


def fixtures(days=(1,), paths=1):
    tasks, records = [], []
    for day in days:
        origin = f"2025-01-{day:02d}T12:00:00Z"
        config = {"lookback": 256, "horizon": 30, "path_count": paths, "seed": 20260926}
        window = {"window_id": f"TEST_ONLY_window_{day}", "as_of": origin, "assets": list(ASSETS),
                  "profile_id": "TEST_ONLY_Jan", "source": "TEST_ONLY_source", "quote_currency": "USDT", "interval_seconds": 60}
        task = {"window": window, "predict_config": config, "model_spec": {"model_label": "Kronos-base"}}
        task["task_id"] = "task_" + hashlib.sha256(_canonical(task).encode()).hexdigest()
        tasks.append(task)
        forecast_id = f"TEST_ONLY_forecast_{day}"
        times = [(datetime(2025, 1, day, 12, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat().replace("+00:00", "Z") for i in range(31)]
        prediction = {"forecast_id": forecast_id, "as_of": origin, "sample_count": paths, "times": times,
                      "paths": [], "summary": {"assets": {}}}
        truth = {"forecast_id": "TEST_ONLY_truth", "as_of": origin, "sample_count": 1, "times": times,
                 "paths": [], "summary": {"assets": {}}}
        pvalues, tvalues, rows = {}, {}, []
        for asset in ASSETS:
            pvalues[asset] = {metric: .01 for metric in ASSET_METRICS}
            tvalues[asset] = {metric: .02 for metric in ASSET_METRICS}
            prediction["summary"]["assets"][asset] = {m: {"mean": v} for m, v in pvalues[asset].items()}
            truth["summary"]["assets"][asset] = {m: {"mean": v} for m, v in tvalues[asset].items()}
            for metric in ASSET_METRICS:
                rows.append({"as_of": origin, "symbol": asset, "forecast_id": forecast_id, "model": "Kronos-base",
                             "metric": metric, "predicted": .01, "actual": .02, "absolute_error": .01, "status": "scored"})
            for model, metric in (("naive-last", "terminal_return"), ("historical-volatility", "volatility")):
                rows.append({"as_of": origin, "symbol": asset, "forecast_id": forecast_id, "model": model,
                             "metric": metric, "predicted": 0, "actual": .02, "absolute_error": .02, "status": "scored"})
        prediction["paths"] = [{"path_id": f"TEST_ONLY_{i}", "assets": copy.deepcopy(pvalues)} for i in range(paths)]
        truth["paths"] = [{"path_id": "observed", "assets": tvalues}]
        evaluation = {"status": "scored", "sample_count": 1, "forecast_id": forecast_id,
                      "as_of": origin, "window_id": window["window_id"], "model": "Kronos-base", "path_count": paths,
                      "prediction_metrics": prediction, "truth_metrics": truth, "rows": rows}
        artifact = {**{key: window[key] for key in ("as_of", "window_id", "profile_id", "source", "quote_currency")},
                    "task_id": task["task_id"], "predict_config": config, "model_spec": task["model_spec"],
                    "status": "scored", "forecast_id": forecast_id, "correction_quality_status": "validated",
                    "volume_quality_status": "validated", "evaluation": evaluation,
                    "result": {"forecast": {"model_revision": IDENTITY["model_revision"]}, "runtime": {"identity": copy.deepcopy(IDENTITY)}}}
        records.append({"path": f"TEST_ONLY/{day}.json", "artifact": artifact})
    return {"tasks": tasks, "selected_count": len(tasks), "task_ids": [t["task_id"] for t in tasks],
            "experiment_id": "TEST_ONLY_experiment"}, records


def change_metric(artifact, symbol, metric, *, predicted, actual, baseline=None):
    evaluation = artifact["evaluation"]
    for name, value in (("prediction_metrics", predicted), ("truth_metrics", actual)):
        metrics = evaluation[name]
        for path in metrics["paths"]:
            path["assets"][symbol][metric] = value
        metrics["summary"]["assets"][symbol][metric]["mean"] = value
    for row in evaluation["rows"]:
        if row["symbol"] == symbol and row["metric"] == metric:
            if row["model"] == "Kronos-base":
                row["predicted"] = predicted
            elif baseline is not None:
                row["predicted"] = baseline
            row["actual"] = actual
            row["absolute_error"] = abs(row["predicted"] - actual)


def compare(manifest, records, **kwargs):
    return compare_records(manifest, records, expected_identity=IDENTITY, bootstrap_replicates=100, **kwargs)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        # Explicit synthetic metric ledgers have no raw market provenance.
        # Isolate that expensive gate only here; EvidenceGateTests exercise it
        # without mocks using synthetic full forecasts and real local truth.
        gate = patch("evaluation.comparison._verify_numerical_evidence", return_value=None)
        gate.start()
        self.addCleanup(gate.stop)

    def test_paired_mae_skill_and_equal_asset_weight(self):
        manifest, records = fixtures((1, 8))
        result = compare(manifest, records)
        self.assertEqual(result["status"], "complete")
        stat = result["overall"]["errors"]["terminal_return"]["by_asset"]["equal_weight_assets"]
        self.assertEqual((stat["model_mae"], stat["baseline_mae"], stat["skill"]), (.01, .02, .5))
        self.assertEqual((stat["win"], stat["tie"], stat["loss"]), (2, 0, 0))
        self.assertEqual(stat["window_count"], 2)  # Never six asset/windows counted as six origins.

    def test_equal_weight_skill_is_ratio_of_maes_not_mean_asset_skills(self):
        manifest, records = fixtures()
        for symbol, actual, pred in (("BTC", .01, .005), ("ETH", .02, .04), ("SOL", .03, -.03)):
            change_metric(records[0]["artifact"], symbol, "terminal_return", predicted=pred, actual=actual, baseline=0)
        stats = compare(manifest, records)["overall"]["errors"]["terminal_return"]["by_asset"]
        self.assertAlmostEqual(stats["equal_weight_assets"]["skill"], 1 - ((.005 + .02 + .06) / 3) / .02)
        self.assertNotAlmostEqual(stats["equal_weight_assets"]["skill"], sum(stats[s]["skill"] for s in ASSETS) / 3)

    def test_complete_manifest_retains_failed_pending_missing_and_duplicate(self):
        manifest, records = fixtures((1, 2, 3, 4, 5))
        records[1]["artifact"].update(status="failed", error={"message": "TEST_ONLY_failure"})
        records[2]["artifact"]["status"] = "pending_truth"
        records.pop(3)  # Missing day four.
        records.append(copy.deepcopy(records[-1]))  # Duplicate day five: neither selected.
        result = compare(manifest, records)
        self.assertEqual(result["coverage"]["status_counts"], {"scored": 1, "failed": 1, "pending_truth": 1, "missing": 1, "duplicate": 1})
        self.assertEqual(result["coverage"]["paired_coverage"], .2)
        self.assertEqual(len(result["tasks"]), 5)
        self.assertEqual(len(result["tasks"][-1]["attempts"]), 2)

    def test_all_failed_has_no_skill_or_success_claim(self):
        manifest, records = fixtures()
        records[0]["artifact"]["status"] = "failed"
        result = compare(manifest, records)
        self.assertEqual(result["status"], "no_comparable_results")
        self.assertIsNone(result["overall"]["errors"]["volatility"]["by_asset"]["BTC"]["skill"])

    def test_incomplete_duplicate_mismatched_and_nonfinite_rows_rejected(self):
        for mutation in ("missing", "duplicate", "actual", "nan"):
            manifest, records = fixtures()
            rows = records[0]["artifact"]["evaluation"]["rows"]
            if mutation == "missing":
                rows.pop()
            elif mutation == "duplicate":
                rows.append(copy.deepcopy(rows[0]))
            elif mutation == "actual":
                rows[-1]["actual"] = .1
                rows[-1]["absolute_error"] = abs(rows[-1]["predicted"] - .1)
            else:
                rows[0]["predicted"] = float("nan")
            with self.subTest(mutation=mutation):
                self.assertEqual(compare(manifest, records)["coverage"]["status_counts"], {"invalid_scored": 1})

    def test_changed_sampling_backend_or_revision_does_not_mix(self):
        for field, value in (("sampling", {"top_p": .5}), ("backend", "TEST_ONLY_other"), ("model_revision", "other")):
            manifest, records = fixtures()
            records[0]["artifact"]["result"]["runtime"]["identity"][field] = value
            self.assertEqual(compare(manifest, records)["coverage"]["status_counts"], {"invalid_scored": 1})

    def test_wrong_result_shape_is_accounted_not_fatal(self):
        manifest, records = fixtures()
        records[0]["artifact"]["result"] = []
        self.assertEqual(compare(manifest, records)["coverage"]["status_counts"], {"invalid_scored": 1})

    def test_competition_ties_keep_all_assets_and_exact_sets(self):
        self.assertEqual(competition_top({"BTC": 3, "ETH": 2, "SOL": 2}), list(ASSETS))
        manifest, records = fixtures()
        # All three model/truth vols tie. Constant ETH+SOL misses the full set.
        ranking = compare(manifest, records)["overall"]["top2_ranking"]
        self.assertEqual(ranking["strategies"]["Kronos"]["mean_selected_assets"], 3)
        self.assertEqual(ranking["strategies"]["Kronos"]["exact_set_hit_rate"], 1)
        self.assertEqual(ranking["strategies"]["constant-ETH-SOL"]["exact_set_hit_rate"], 0)
        self.assertEqual(ranking["paired_exact_set_comparisons"]["constant-ETH-SOL"]["win"], 1)

    def test_drawdown_strict_threshold_confusion_and_zero_positive_recall(self):
        manifest, records = fixtures()
        result = compare(manifest, records)
        self.assertIsNone(result["overall"]["drawdown_3pct"]["by_asset"]["all_assets"]["recall"])
        artifact = records[0]["artifact"]
        change_metric(artifact, "BTC", "max_drawdown", predicted=.04, actual=.05)
        change_metric(artifact, "ETH", "max_drawdown", predicted=.04, actual=.03)
        change_metric(artifact, "SOL", "max_drawdown", predicted=.03, actual=.04)
        stats = compare(manifest, records)["overall"]["drawdown_3pct"]["by_asset"]["all_assets"]
        self.assertEqual((stats["tp"], stats["fp"], stats["fn"], stats["actual_positives"]), (1, 1, 1, 2))
        self.assertEqual(stats["recall"], .5)

    def test_mean_pathwise_metric_is_verified_before_sorting(self):
        manifest, records = fixtures(paths=2)
        metrics = records[0]["artifact"]["evaluation"]["prediction_metrics"]
        metrics["paths"][0]["assets"]["BTC"]["volatility"] = .02
        metrics["paths"][1]["assets"]["BTC"]["volatility"] = 0
        self.assertEqual(compare(manifest, records)["coverage"]["paired_windows"], 1)
        metrics["summary"]["assets"]["BTC"]["volatility"]["mean"] = 0
        self.assertEqual(compare(manifest, records)["coverage"]["status_counts"], {"invalid_scored": 1})

    def test_zero_baseline_mae_is_undefined_not_infinite(self):
        manifest, records = fixtures()
        for asset in ASSETS:
            change_metric(records[0]["artifact"], asset, "terminal_return", predicted=.01, actual=0, baseline=0)
        stat = compare(manifest, records)["overall"]["errors"]["terminal_return"]["by_asset"]["equal_weight_assets"]
        self.assertIsNone(stat["skill"])
        self.assertEqual(stat["skill_status"], "undefined_zero_baseline_mae")

    def test_calendar_segments_and_day_bootstrap_are_deterministic(self):
        manifest, records = fixtures((1, 8, 15, 22, 31))
        first, second = compare(manifest, records), compare(manifest, records)
        self.assertEqual(first, second)
        self.assertEqual([s["requested_windows"] for s in first["calendar_segments"].values()], [1, 1, 1, 2])
        interval = first["overall"]["errors"]["terminal_return"]["by_asset"]["BTC"]["skill_interval"]
        self.assertEqual(interval["day_count"], 5)
        self.assertEqual(interval["skill_percentile_95"], [.5, .5])
        self.assertTrue(interval["exploratory"])

    def test_manifest_duplicate_and_modified_hash_rejected(self):
        manifest, records = fixtures()
        manifest["tasks"][0]["predict_config"]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "hash"):
            compare(manifest, records)
        manifest, records = fixtures()
        manifest["tasks"].append(copy.deepcopy(manifest["tasks"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare(manifest, records)

    def test_file_reader_corruption_missing_summary_and_unknown_files_visible(self):
        manifest, records = fixtures((1, 8))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "tasks").mkdir()
            (folder / "summary.json").write_text(json.dumps({"experiment_id": manifest["experiment_id"], "expected_identity": IDENTITY}))
            first = records[0]["artifact"]
            (folder / "tasks" / (first["task_id"] + ".json")).write_text(json.dumps(first))
            (folder / "tasks" / (records[1]["artifact"]["task_id"] + ".json")).write_text("broken")
            (folder / "tasks" / "unknown.json").write_text("{}")
            result = compare_batch(manifest, folder, bootstrap_replicates=0)
            self.assertEqual(result["coverage"]["status_counts"], {"scored": 1, "invalid_artifact": 1})
            self.assertEqual(result["coverage"]["unassigned_record_count"], 1)
            (folder / "summary.json").unlink()
            result = compare_batch(manifest, folder, bootstrap_replicates=0)
            self.assertEqual(result["coverage"]["paired_windows"], 0)
            self.assertIn("FileNotFoundError", result["batch_summary_error"])
            (folder / "summary.json").write_text("[]")
            result = compare_batch(manifest, folder, bootstrap_replicates=0)
            self.assertEqual(result["coverage"]["paired_windows"], 0)
            self.assertIn("must be an object", result["batch_summary_error"])

    def test_cli_writes_comparison_and_refuses_overwrite(self):
        manifest, records = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "tasks").mkdir()
            (folder / "manifest.json").write_text(json.dumps(manifest))
            (folder / "summary.json").write_text(json.dumps({"experiment_id": manifest["experiment_id"], "expected_identity": IDENTITY}))
            artifact = records[0]["artifact"]
            (folder / "tasks" / (artifact["task_id"] + ".json")).write_text(json.dumps(artifact))
            args = ["--manifest", str(folder / "manifest.json"), "--batch-dir", str(folder),
                    "--out", str(folder / "comparison.json"), "--bootstrap-replicates", "0"]
            self.assertEqual(main(args), 0)
            with self.assertRaises(FileExistsError):
                main(args)


class EvidenceGateTests(unittest.TestCase):
    """Full synthetic forecast/raw audits against actual local historical truth."""

    @classmethod
    def setUpClass(cls):
        from data_pipeline import load_window
        from evaluation.test_evaluation import FixtureAdapter
        from evaluation.scoring import score_forecast
        from forecast_metrics.engine import freeze_forecast
        window = load_window()
        config = {"lookback": 256, "horizon": 2, "path_count": 1, "seed": 20260926}
        task = {"window": window, "predict_config": config, "model_spec": {"model_label": "Kronos-base"}}
        task["task_id"] = "task_" + hashlib.sha256(_canonical(task).encode()).hexdigest()
        result = FixtureAdapter().predict(window, config, "TEST_ONLY_comparison_gate")
        cls.identity = {**copy.deepcopy(IDENTITY), "model_revision": result["forecast"]["model_revision"]}
        result["runtime"]["identity"] = cls.identity
        frozen = freeze_forecast(result["forecast"])
        evaluation = score_forecast(window, frozen.data, frozen.identity)
        artifact = {**{key: window[key] for key in ("as_of", "window_id", "profile_id", "source", "quote_currency")},
                    "task_id": task["task_id"], "predict_config": config, "model_spec": task["model_spec"],
                    "status": "scored", "forecast_id": frozen.identity, "correction_quality_status": "validated",
                    "volume_quality_status": "validated", "evaluation": evaluation, "result": result}
        cls.manifest = {"tasks": [task], "selected_count": 1, "task_ids": [task["task_id"]], "experiment_id": "TEST_ONLY"}
        cls.records = [{"path": "TEST_ONLY_full_forecast.json", "artifact": artifact}]

    def check(self, records):
        return compare_records(self.manifest, records, expected_identity=self.identity, bootstrap_replicates=0)

    def test_untouched_full_evidence_passes_without_model(self):
        self.assertEqual(self.check(self.records)["coverage"]["paired_windows"], 1)

    def test_raw_forecast_audit_baseline_and_truth_tampering_rejected(self):
        for target in ("close", "volume_audit", "historical_baseline", "truth"):
            records = copy.deepcopy(self.records)
            artifact = records[0]["artifact"]
            if target == "close":
                artifact["result"]["forecast"]["paths"][0]["assets"]["BTC"]["close"][0] *= 1.00001
            elif target == "volume_audit":
                record = artifact["result"]["forecast"]["volume_quality"]["records"][0]
                record["volume_valid"] = not record["volume_valid"]
            elif target == "historical_baseline":
                row = next(r for r in artifact["evaluation"]["rows"] if r["model"] == "historical-volatility")
                row.update(predicted=row["actual"], absolute_error=0)
            else:
                # Coherent mutation of rows + stored metric means/paths still
                # cannot override independent local truth during re-scoring.
                change_metric(artifact, "BTC", "terminal_return", predicted=0, actual=.2, baseline=0)
            with self.subTest(target=target):
                self.assertEqual(self.check(records)["coverage"]["status_counts"], {"invalid_scored": 1})


if __name__ == "__main__":
    unittest.main()
