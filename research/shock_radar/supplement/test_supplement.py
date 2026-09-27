"""Small counterexamples for the new deterministic definitions."""
from copy import deepcopy
from math import exp, sqrt
import unittest

from .grouped_direction import directions, by_trigger, METHODS
from .downside_radius import _downside_vol, _segment_risk, _prefix_radius


class SupplementTests(unittest.TestCase):
    def test_downside_zero_target_and_full_denominator(self):
        # One negative return among 30: no centering or negative-only division.
        prices = [100 * exp(-.02)] * 30
        self.assertAlmostEqual(_downside_vol(100, prices), .02)
        self.assertEqual(_downside_vol(100, [101.] * 30), 0.)
        self.assertAlmostEqual(_downside_vol(100, [100 * exp(-.01 * (i+1)) for i in range(30)]), .01 * sqrt(30))

    def test_segment_uses_its_own_boundary_and_resets_peak(self):
        closes = [200.] * 10 + [100.] * 10 + [150.] * 10
        self.assertEqual(_segment_risk(100., closes, 0)["max_drawdown"], 0.)
        self.assertEqual(_segment_risk(100., closes, 10)["max_drawdown"], .5)
        self.assertEqual(_segment_risk(100., closes, 20)["max_drawdown"], 0.)

    def test_later_isolated_advantages_do_not_invent_radius(self):
        self.assertIsNone(_prefix_radius([False, True, True]))
        self.assertEqual(_prefix_radius([True, False, True]), 10)
        self.assertEqual(_prefix_radius([True, True, True]), 30)
        self.assertIsNone(_prefix_radius([None, True, True]))

    def test_direction_flat_predictions_not_dropped(self):
        row = {"event_id": "e", "direction": 1, "assets": ["A", "B", "C"],
               "spots": {a: 100. for a in "ABC"},
               "actual": {a: [end] * 30 for a, end in zip("ABC", [101., 99., 100.])},
               "paths": {m: {a: [100.] * 30 for a in "ABC"} for m in METHODS}}
        out = directions([row])["aggregate"]
        self.assertEqual(out["all_pairs"], 3)
        self.assertEqual(out["true_flat_pairs"], 1)
        self.assertEqual(out["common_pairs"], 2)
        self.assertEqual(out["methods"]["kronos_base"]["common_binary_accuracy"], 0.)
        self.assertEqual(out["methods"]["kronos_base"]["all_three_sign_correct"], 1)
        self.assertEqual(out["methods"]["always_continue"]["common_binary_accuracy"], .5)
        row["direction"] = "mixed"
        out = directions([row])["aggregate"]
        self.assertEqual(out["common_pairs"], 0)
        self.assertIsNone(out["methods"]["kronos_base"]["common_binary_accuracy"])
        self.assertEqual(out["methods"]["kronos_base"]["all_nonzero_truth_pairs"], 2)

    def test_direction_group_contributions_recover_fixed_comparison(self):
        rows = []
        for i, direction in enumerate([1, -1, -1]):
            methods = {m: {"metrics": {"volatility": {"top3_recall_expected": .5, "mae_bps": 2.},
                                       "max_drawdown": {"mae_bps": 3. if m == "kronos_base" else 6. + i}}}
                       for m in METHODS}
            rows.append({"event_id": str(i), "direction": direction, "saved_score": {"methods": methods}})
        out = by_trigger(rows)
        self.assertEqual(out["groups"]["mixed_or_unknown"]["event_count"], 0)
        self.assertIsNone(out["groups"]["mixed_or_unknown"]["methods"]["kronos_base"]["max_drawdown_mae_bps"])
        self.assertAlmostEqual(out["mdd_advantage_decomposition"]["btc_beta"]["total_advantage_bps"], 4.)


if __name__ == "__main__":
    unittest.main()
