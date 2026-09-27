"""Observation warnings never mutate the report used for selection/gates."""
from copy import deepcopy
import unittest

from lora_a.reporting import observation_metrics, interval_text


class ObservationTests(unittest.TestCase):
    def report(self, baseline=.1):
        row = {"n_scored": 300, "n_scheduled": 300, "n_failed": 0,
               "volatility_mae": .004, "volatility_top3_recall": .6}
        methods = {"original": {**row, "max_drawdown_mae": baseline},
                   "epoch_01": {**row, "max_drawdown_mae": .12}}
        pair = {"epoch_01": {"original": {"max_drawdown_mae_difference": {
            "candidate_mean": .12, "reference_mean": baseline, "n_paired": 300}}}}
        return {"tiers": {tier: {"methods": deepcopy(methods), "paired": deepcopy(pair)} for tier in ("major", "small")}}

    def test_twenty_percent_warning_is_log_only(self):
        report = self.report()
        before = deepcopy(report)
        result = observation_metrics(report, "epoch_01")
        self.assertEqual(result["additional_metrics_decision_use"], "none")
        self.assertTrue(result["tiers"]["small"]["drawdown_worse_by_20pct_or_more"])
        self.assertEqual(report, before)

    def test_zero_baseline_does_not_invent_relative_change(self):
        result = observation_metrics(self.report(0), "epoch_01")["tiers"]["small"]
        self.assertIsNone(result["relative_drawdown_mae_change"])
        self.assertEqual(result["drawdown_mae_absolute_change"], .12)

    def test_hit_difference_interval_uses_percentage_points(self):
        self.assertEqual(interval_text({"ci95": [-.02, .03]}, True), "[-2.000pp, +3.000pp]")


if __name__ == "__main__":
    unittest.main()
