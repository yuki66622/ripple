"""Synthetic confusion tables only: no historical files, model, or network."""
import math
import unittest

from .thresholds import THRESHOLDS, evaluate_thresholds


class ThresholdTests(unittest.TestCase):
    def test_all_four_confusion_cells_and_total(self):
        row = evaluate_thresholds([.04, .04, .01, .01], [.04, .01, .04, .01])["rows"][2]
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["tn"]), (1, 1, 1, 1))
        self.assertEqual((row["precision"], row["recall"], row["f1"]), (.5, .5, .5))
        self.assertEqual(row["asset_pair_count"], 4)

    def test_threshold_is_strict_and_never_rounded(self):
        result = evaluate_thresholds([.03, math.nextafter(.03, 1)], [.03, .03])
        row = result["rows"][2]
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["tn"]), (0, 1, 0, 1))
        self.assertEqual(row["precision"], 0)
        self.assertIsNone(row["recall"])
        self.assertEqual(row["recall_reason"], "no_actual_positives")
        self.assertEqual(row["f1"], 0)

    def test_no_predicted_positive_preserves_false_negatives_and_zero_f1(self):
        row = evaluate_thresholds([0, 0], [.05, 0])["rows"][0]
        self.assertIsNone(row["precision"])
        self.assertEqual(row["precision_reason"], "no_predicted_positives")
        self.assertEqual((row["recall"], row["f1"], row["fn"]), (0, 0, 1))

    def test_all_negative_and_empty_are_null_not_fake_perfect(self):
        for predicted, actual in (([0, 0], [0, 0]), ([], [])):
            result = evaluate_thresholds(predicted, actual)
            self.assertIsNone(result["recommended_threshold"])
            for row in result["rows"]:
                for metric in ("precision", "recall", "f1"):
                    self.assertIsNone(row[metric])
                    self.assertIsNotNone(row[metric + "_reason"])
                self.assertEqual(sum(row[k] for k in ("tp", "fp", "fn", "tn")), len(actual))

    def test_recommendation_needs_both_positive_counts_at_least_ten(self):
        for predicted, actual in (([.1]*9, [.1]*9), ([.1]*10, [.1]*9+[0]), ([.1]*9+[0], [.1]*10)):
            result = evaluate_thresholds(predicted, actual)
            self.assertIsNone(result["recommended_threshold"])
            self.assertTrue(all(not r["recommendation_eligible"] for r in result["rows"]))
        result = evaluate_thresholds([.1]*10, [.1]*10)
        self.assertEqual(result["recommended_threshold"], .01)

    def test_highest_f1_then_lower_threshold(self):
        # 1% adds ten false positives, while 2% and 3% both classify perfectly.
        result = evaluate_thresholds([.04]*10+[.015]*10, [.04]*10+[0]*10)
        self.assertEqual(result["recommended_threshold"], .02)
        self.assertEqual([r["threshold"] for r in result["rows"]], list(THRESHOLDS))
        self.assertEqual(result["existing_product_threshold"], .03)
        self.assertFalse(result["product_rule_changed"])

    def test_unsupported_perfect_cell_does_not_win_recommendation(self):
        result = evaluate_thresholds([.06]*9+[.015]*11, [.06]*9+[.015]*11)
        self.assertEqual(result["recommended_threshold"], .01)
        self.assertEqual(result["rows"][-1]["f1"], 1)
        self.assertFalse(result["rows"][-1]["recommendation_eligible"])

    def test_invalid_or_mismatched_values_are_not_dropped(self):
        for bad in (-.01, 1.01, float("nan"), float("inf"), True, None, "0.1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluate_thresholds([bad], [.1])
            with self.subTest(actual=bad), self.assertRaises(ValueError):
                evaluate_thresholds([.1], [bad])
        with self.assertRaises(ValueError):
            evaluate_thresholds([.1], [])

    def test_each_pair_counted_once_at_every_threshold(self):
        predicted = [.0, .01, .015, .02, .025, .03, .04, .05, .06, .1]*84
        actual = list(reversed(predicted))
        result = evaluate_thresholds(predicted, actual)
        self.assertEqual(result["asset_pair_count"], 840)
        for row in result["rows"]:
            self.assertEqual(sum(row[key] for key in ("tp", "fp", "fn", "tn")), 840)
            self.assertEqual(row["predicted_positives"], sum(v>row["threshold"] for v in predicted))
            self.assertEqual(row["actual_positives"], sum(v>row["threshold"] for v in actual))


if __name__ == "__main__":
    unittest.main()
