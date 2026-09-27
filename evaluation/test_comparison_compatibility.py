"""Pure scalar ledger fixtures; no model, market download or checkpoint loading."""

import copy
import unittest

from evaluation.corrections import COMPARABLE_IDENTITY_FIELDS, comparison_key, summarize_quality


def controlled_artifact(label="base", revision="TEST_ONLY_base", task="task-base"):
    return {
        "task_id": task, "window_id": "TEST_ONLY_same_content_hash", "status": "scored",
        "artifact_path": "TEST_ONLY_in_memory", "model_spec": {"model_label": label},
        "predict_config": {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926},
        "result": {"forecast": {"model_revision": revision}, "runtime": {"identity": {
            "model_revision": revision, "tokenizer_revision": "TEST_ONLY_frozen_tokenizer",
            "adapter_revision": "sktime-kronos-field-quality-v3",
            "sampling": {"T": 1.0, "top_k": 0, "top_p": .9, "sample_count": 1, "verbose": False},
            "output_policy": "containment-expand-v1", "volume_policy": "unused-volume-audit-v1",
            "backend": "mps", "torch_version": "2.14.0", "sktime_version": "1.2.0",
            "model_timestamp": "UTC candle start", "output_timestamp": "UTC candle end",
        }}},
        "correction_quality_status": "validated",
        "correction_quality": {"policy": "containment-expand-v1", "total_candles": 90,
            "corrected_candles": 9, "correction_rate": .1, "correction_rate_pct": 10.,
            "max_adjustment_bps": 2.},
    }


class ComparisonCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.base = controlled_artifact()
        self.tuned = controlled_artifact("tuned", "TEST_ONLY_tuned", "task-tuned")

    def pair(self, left=None, right=None):
        return summarize_quality([left or self.base, right or self.tuned])["matched_comparisons"][0]

    def assert_unmatched(self):
        pair = self.pair()
        self.assertEqual(pair["matched_task_count"], 0)
        self.assertEqual(pair["status"], "no_comparable_workloads")
        self.assertEqual(pair["unmatched_a_count"], 1)
        self.assertEqual(pair["unmatched_b_count"], 1)
        self.assertIsNone(pair["a"]["correction_rate"])
        self.assertIsNone(pair["b"]["correction_rate"])

    def test_only_weight_revision_and_label_may_differ(self):
        self.assertEqual(comparison_key(self.base), comparison_key(self.tuned))
        pair = self.pair()
        self.assertEqual(pair["matched_task_count"], 1)
        self.assertEqual(pair["status"], "matched_descriptive_only")
        self.assertEqual(pair["unmatched_a_count"], 0)
        self.assertEqual(pair["a"]["total_candles"], 90)

    def test_top_p_counterexample_is_rejected(self):
        self.tuned["result"]["runtime"]["identity"]["sampling"]["top_p"] = .1
        self.assert_unmatched()

    def test_every_runtime_control_must_match(self):
        for field in COMPARABLE_IDENTITY_FIELDS:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.tuned)
                identity = changed["result"]["runtime"]["identity"]
                if field == "sampling":
                    identity[field]["T"] = .5
                else:
                    identity[field] += "_DIFFERENT"
                self.assertEqual(self.pair(right=changed)["matched_task_count"], 0)

    def test_missing_each_control_never_assumes_a_match(self):
        for field in (*COMPARABLE_IDENTITY_FIELDS, "model_revision"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.tuned)
                del changed["result"]["runtime"]["identity"][field]
                self.assertIsNone(comparison_key(changed))
                self.assertEqual(self.pair(right=changed)["matched_task_count"], 0)

    def test_legacy_missing_identity_keeps_descriptive_counts(self):
        for artifact in (self.base, self.tuned):
            del artifact["result"]["runtime"]
        report = summarize_quality([self.base, self.tuned])
        self.assertEqual(sum(g["total_candles"] for g in report["by_model"]), 180)
        self.assertTrue(all(g["incomparable_context_forecasts"] == 1 for g in report["by_model"]))
        self.assert_unmatched()

    def test_missing_or_changed_configuration_is_not_comparable(self):
        for field in self.tuned["predict_config"]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.tuned)
                changed["predict_config"][field] += 1
                self.assertEqual(self.pair(right=changed)["matched_task_count"], 0)
                del changed["predict_config"][field]
                self.assertIsNone(comparison_key(changed))

    def test_different_input_hash_does_not_pair(self):
        self.tuned["window_id"] = "TEST_ONLY_changed_input"
        self.assert_unmatched()

    def test_nonfinite_or_incomplete_sampling_fails_closed(self):
        for bad in (float("nan"), float("inf"), False, 0):
            with self.subTest(value=bad):
                changed = copy.deepcopy(self.tuned)
                changed["result"]["runtime"]["identity"]["sampling"]["T"] = bad
                self.assertIsNone(comparison_key(changed))
        del self.tuned["result"]["runtime"]["identity"]["sampling"]["sample_count"]
        self.assert_unmatched()

    def test_model_revision_must_agree_with_forecast(self):
        self.tuned["result"]["runtime"]["identity"]["model_revision"] = "TEST_ONLY_wrong"
        self.assert_unmatched()

    def test_conflicting_runtime_copies_cannot_pair(self):
        self.tuned["runtime"] = copy.deepcopy(self.tuned["result"]["runtime"])
        self.tuned["runtime"]["identity"]["backend"] = "cpu"
        self.assert_unmatched()

    def test_same_window_controls_conflict_is_not_first_attempt_wins(self):
        repeated = copy.deepcopy(self.base)
        repeated["task_id"] = "task-repeat"
        repeated["result"]["runtime"]["identity"]["sampling"]["top_p"] = .1
        report = summarize_quality([self.base, repeated, self.tuned])
        base = next(g for g in report["by_model"] if g["model"] == "base")
        self.assertEqual(base["forecast_count"], 0)
        self.assertEqual(base["conflicting_duplicate_workloads"], 1)
        self.assertEqual(len(base["conflicts"][0]["attempts"]), 2)
        self.assertEqual(report["matched_comparisons"][0]["matched_task_count"], 0)

    def test_extra_sampling_controls_are_not_ignored(self):
        self.tuned["result"]["runtime"]["identity"]["sampling"]["new_setting"] = True
        self.assert_unmatched()

    def test_malformed_context_is_not_accepted(self):
        for bad in ([], "invalid", 7):
            changed = copy.deepcopy(self.tuned)
            changed["result"]["runtime"] = bad
            self.assertIsNone(comparison_key(changed))


if __name__ == "__main__":
    unittest.main()
