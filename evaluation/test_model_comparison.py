"""Explicit synthetic report fixtures, never synthetic trained checkpoints.

No forecasting model or training runs. The CLI test mocks only the already
tested batch verifier; root acceptance uses the same genuine base batch twice.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.model_comparison import ASSETS, METRICS, compare_reports, compare_model_batches, main
from evaluation.corrections import COMPARABLE_IDENTITY_FIELDS
from evaluation.test_comparison_compatibility import controlled_artifact


def synthetic_report(label="TEST_ONLY_left", days=(1, 2), error=.125):
    revision = label + "_not_trained_revision"
    identity = controlled_artifact(label=label, revision=revision)["result"]["runtime"]["identity"]
    tasks = []
    for day in days:
        pairs = {asset: {metric: {"model": .25 + error, "actual": .25,
                                  "model_error": abs(error), "baseline": 0., "baseline_error": .25}
                         for metric in METRICS} for asset in ASSETS}
        tasks.append({"task_id": f"{label}_task_{day}", "window_id": f"TEST_ONLY_input_{day}",
                      "as_of": f"2025-01-{day:02d}T12:00:00Z", "status": "scored",
                      "model_revision": revision, "forecast_id": f"{label}_forecast_{day}",
                      "artifact_paths": [f"TEST_ONLY/{label}/{day}.json"], "errors": [], "pairs": pairs})
    return {"schema_version": 1, "status": "complete", "experiment_id": label + "_synthetic",
            "assets": list(ASSETS),
            "model": label, "model_revisions": [revision], "expected_identity": identity,
            "predict_config": {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926},
            "profile_id": "TEST_ONLY_profile", "source": "TEST_ONLY_synthetic_source", "quote_currency": "USDT",
            "coverage": {"requested_windows": len(days), "paired_windows": len(days),
                         "status_counts": {"scored": len(days)}, "unassigned_record_count": 0},
            "tasks": tasks, "unassigned_records": []}


def stats(result, metric="terminal_return", asset="equal_weight_assets"):
    return result["errors"][metric]["by_asset"][asset]


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.left = synthetic_report()
        self.right = synthetic_report("TEST_ONLY_right", error=.0625)

    def test_lower_error_right_model_has_positive_skill(self):
        result = compare_reports(self.left, self.right)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["same_model_revision"])
        self.assertFalse(result["self_comparison"])
        self.assertEqual(result["coverage"]["matched_windows"], 2)
        for metric in METRICS:
            for asset in (*ASSETS, "equal_weight_assets"):
                value = stats(result, metric, asset)
                self.assertEqual((value["left_mae"], value["right_mae"], value["right_vs_left_skill"]), (.125, .0625, .5))
                self.assertEqual((value["win"], value["tie"], value["loss"]), (2, 0, 0))

    def test_higher_error_right_model_has_negative_skill(self):
        result = compare_reports(self.right, self.left)
        self.assertEqual(stats(result)["right_vs_left_skill"], -1)
        self.assertEqual((stats(result)["win"], stats(result)["loss"]), (0, 2))

    def test_identical_saved_result_has_zero_improvement(self):
        result = compare_reports(self.left, copy.deepcopy(self.left))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["same_model_revision"])
        self.assertTrue(result["self_comparison"])
        self.assertEqual(stats(result)["right_vs_left_skill"], 0)
        self.assertEqual(stats(result)["right_minus_left_mae"], 0)
        self.assertEqual((stats(result)["win"], stats(result)["tie"], stats(result)["loss"]), (0, 2, 0))

    def test_both_perfect_have_zero_delta_but_undefined_ratio(self):
        left = synthetic_report(error=0)
        result = compare_reports(left, copy.deepcopy(left))
        self.assertEqual(stats(result)["right_minus_left_mae"], 0)
        self.assertIsNone(stats(result)["right_vs_left_skill"])
        self.assertEqual(stats(result)["skill_status"], "undefined_zero_left_mae")
        self.assertEqual(stats(result)["tie"], 2)

    def test_equal_asset_skill_is_ratio_of_maes_not_mean_skills(self):
        for report, values in ((self.left, (.125, .25, .5)), (self.right, (.0625, .25, .25))):
            for task in report["tasks"]:
                for asset, error in zip(ASSETS, values):
                    pair = task["pairs"][asset]["terminal_return"]
                    pair.update(model=.25 + error, model_error=error)
        result = compare_reports(self.left, self.right)
        value = stats(result)
        self.assertAlmostEqual(value["right_vs_left_skill"], 1 - (.0625 + .25 + .25) / (.125 + .25 + .5))
        self.assertNotAlmostEqual(value["right_vs_left_skill"], sum(stats(result, asset=s)["right_vs_left_skill"] for s in ASSETS) / 3)
        self.assertEqual(value["window_count"], 2)

    def test_naive_baseline_error_is_never_used(self):
        before = compare_reports(self.left, self.right)["errors"]
        for task in self.right["tasks"]:
            for asset in ASSETS:
                for metric in METRICS:
                    task["pairs"][asset][metric]["baseline_error"] = 10000.
        self.assertEqual(compare_reports(self.left, self.right)["errors"], before)

    def test_every_runtime_control_difference_is_incompatible(self):
        for key in COMPARABLE_IDENTITY_FIELDS:
            with self.subTest(key=key):
                right = copy.deepcopy(self.right)
                identity = right["expected_identity"]
                if key == "sampling":
                    identity[key]["top_p"] = .1
                else:
                    identity[key] += "_DIFFERENT"
                result = compare_reports(self.left, right)
                self.assertEqual(result["coverage"]["status_counts"], {"incompatible_controls": 2})
                self.assertIsNone(stats(result)["right_vs_left_skill"])

    def test_missing_identity_fails_closed_without_losing_denominators(self):
        self.right["expected_identity"].pop("tokenizer_revision")
        result = compare_reports(self.left, self.right)
        self.assertEqual(result["coverage"]["matched_windows"], 0)
        self.assertEqual((result["coverage"]["left_requested_windows"], result["coverage"]["right_requested_windows"]), (2, 2))
        self.assertEqual(result["status"], "no_comparable_results")

    def test_seed_or_other_predict_config_difference_is_incompatible(self):
        for key in self.right["predict_config"]:
            with self.subTest(key=key):
                right = copy.deepcopy(self.right)
                right["predict_config"][key] += 1
                result = compare_reports(self.left, right)
                self.assertEqual(result["coverage"]["status_counts"], {"incompatible_config": 2})

    def test_source_or_input_mismatch_is_not_matched_by_timestamp(self):
        right = copy.deepcopy(self.right)
        right["source"] = "TEST_ONLY_other_exchange"
        self.assertEqual(compare_reports(self.left, right)["coverage"]["status_counts"], {"incompatible_source": 2})
        self.right["tasks"][0]["window_id"] = "TEST_ONLY_revised_candles"
        result = compare_reports(self.left, self.right)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["coverage"]["status_counts"], {"incompatible_input": 1, "matched": 1})

    def test_different_actual_target_never_compares_errors(self):
        pair = self.right["tasks"][0]["pairs"]["BTC"]["volatility"]
        pair["actual"] = .5
        pair["model_error"] = abs(pair["model"] - pair["actual"])
        result = compare_reports(self.left, self.right)
        self.assertEqual(result["coverage"]["status_counts"], {"incompatible_target_or_error": 1, "matched": 1})
        self.assertIn("target mismatch", result["tasks"][0]["reasons"][0])

    def test_corrupt_model_error_is_not_accepted(self):
        self.right["tasks"][0]["pairs"]["ETH"]["terminal_return"]["model_error"] = 0
        result = compare_reports(self.left, self.right)
        self.assertEqual(result["coverage"]["incompatible_windows"], 1)

    def test_failed_missing_and_pending_counterparts_are_retained(self):
        left = synthetic_report(days=(1, 2, 3, 4))
        right = synthetic_report("TEST_ONLY_right", days=(1, 2, 3, 4))
        for task, status in zip(right["tasks"][1:], ("failed", "missing", "pending_truth")):
            task.update(status=status, errors=["TEST_ONLY retained reason"])
            task.pop("pairs")
        right["status"] = "partial"
        right["coverage"].update(paired_windows=1, status_counts={"scored": 1, "failed": 1, "missing": 1, "pending_truth": 1})
        result = compare_reports(left, right)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["coverage"]["matched_windows"], 1)
        self.assertEqual(result["coverage"]["unavailable_counterpart_windows"], 3)
        self.assertEqual(result["right"]["coverage"], right["coverage"])
        self.assertEqual([r["right"]["status"] for r in result["tasks"]], ["scored", "failed", "missing", "pending_truth"])
        self.assertEqual(stats(result)["window_count"], 1)

    def test_all_failed_retains_both_denominators_and_no_performance(self):
        for report in (self.left, self.right):
            for task in report["tasks"]:
                task.update(status="failed", errors=["TEST_ONLY retained model failure"])
                task.pop("pairs")
            report["status"] = "no_comparable_results"
            report["coverage"].update(paired_windows=0, status_counts={"failed": 2})
            report["model_revisions"] = []
        result = compare_reports(self.left, self.right)
        self.assertEqual(result["status"], "no_comparable_results")
        self.assertEqual(result["coverage"]["left_requested_windows"], 2)
        self.assertEqual(result["coverage"]["right_requested_windows"], 2)
        self.assertEqual(result["coverage"]["unavailable_counterpart_windows"], 2)
        self.assertEqual(result["left"]["coverage"]["status_counts"], {"failed": 2})
        self.assertEqual(result["right"]["coverage"]["status_counts"], {"failed": 2})
        self.assertIsNone(stats(result)["right_vs_left_skill"])

    def test_different_manifest_coverage_is_not_called_complete(self):
        left = synthetic_report(days=(1, 2))
        right = synthetic_report("TEST_ONLY_right", days=(2, 3))
        result = compare_reports(left, right)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["coverage"]["status_counts"], {"only_left": 1, "matched": 1, "only_right": 1})
        self.assertEqual(result["coverage"]["union_origins"], 3)
        self.assertEqual(result["coverage"]["matched_fraction_of_left"], .5)

    def test_unassigned_records_prevent_complete_claim(self):
        self.right["status"] = "partial"
        self.right["coverage"]["unassigned_record_count"] = 1
        self.right["unassigned_records"] = [{"path": "TEST_ONLY_unknown", "reason": "unknown task"}]
        result = compare_reports(self.left, self.right)
        self.assertEqual(result["coverage"]["matched_windows"], 2)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["right"]["unassigned_records"], self.right["unassigned_records"])

    def test_duplicate_origin_mixed_revisions_and_bad_denominators_rejected(self):
        for mutation in ("duplicate", "mixed", "denominator"):
            right = copy.deepcopy(self.right)
            if mutation == "duplicate":
                right["tasks"][1]["as_of"] = right["tasks"][0]["as_of"]
            elif mutation == "mixed":
                right["model_revisions"].append("TEST_ONLY_another_revision")
            else:
                right["coverage"]["requested_windows"] = 50
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare_reports(self.left, right)

    def test_model_batches_always_verifies_both_sides(self):
        with patch("evaluation.model_comparison.compare_batch", side_effect=[self.left, self.right]) as verify:
            result = compare_model_batches({"TEST_ONLY": "left"}, "left-dir", {"TEST_ONLY": "right"}, "right-dir")
        self.assertEqual(verify.call_count, 2)
        self.assertTrue(all(call.kwargs == {"bootstrap_replicates": 0} for call in verify.call_args_list))
        self.assertEqual(result["status"], "complete")

    def test_cli_keeps_output_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for side in ("left", "right"):
                (root / (side + ".json")).write_text(json.dumps({"TEST_ONLY_manifest": side}))
            argv = ["--left-manifest", str(root / "left.json"), "--left-batch-dir", "TEST_ONLY_left",
                    "--right-manifest", str(root / "right.json"), "--right-batch-dir", "TEST_ONLY_right",
                    "--out", str(root / "result.json")]
            with patch("evaluation.model_comparison.compare_batch", side_effect=[self.left, self.right]):
                self.assertEqual(main(argv), 0)
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["coverage"]["matched_windows"], 2)
            with patch("evaluation.model_comparison.compare_batch", side_effect=[self.left, self.right]):
                with self.assertRaises(FileExistsError):
                    main(argv)


if __name__ == "__main__":
    unittest.main()
