"""Synthetic close paths and risk tables only; no historical data or models."""

import copy
import json
from math import sqrt
import unittest

from forecast_metrics.engine import asset_metrics
from .metrics import aggregate_events, rank_scores, risk_metrics, score_event

ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
METHODS = ["kronos_base", "no_propagation", "btc_beta", "historical_30"]


def event(identifier="SYNTHETIC_event", *, systemic=False, origins=None, eligible=True):
    return {"event_id": identifier, "timestamp": "SYNTHETIC_no_data_timestamp",
            "origin_assets": origins or ["BTC"], "systemic_flag": systemic,
            "eligible": eligible, "exclusion_reasons": [] if eligible else ["SYNTHETIC_edge"]}


def table(error=0.):
    return {asset: {"max_drawdown": (i + 1) * .001 + error,
                    "volatility": (i + 1) * .002 + error} for i, asset in enumerate(ASSETS)}


class RiskMetricTests(unittest.TestCase):
    def test_close_risk_matches_existing_financial_engine(self):
        paths = [[100., 100.], [98., 101., 99., 102.], [110., 100., 80., 90.],
                 [100. * (1 + ((i % 5) - 2) * .005) for i in range(30)]]
        for closes in paths:
            with self.subTest(closes=closes):
                expected = asset_metrics(100., closes, closes, closes, 1.)
                actual = risk_metrics(100., closes)
                self.assertEqual(actual["max_drawdown"], expected["max_drawdown"])
                self.assertEqual(actual["volatility"], expected["volatility"])

    def test_drawdown_includes_current_spot(self):
        values = risk_metrics(100., [80., 90.])
        self.assertAlmostEqual(values["max_drawdown"], .2)
        self.assertGreater(values["volatility"], 0.)

    def test_flat_path_zero_risk_and_input_unchanged(self):
        closes = [100.] * 30
        before = list(closes)
        self.assertEqual(risk_metrics(100., closes), {"max_drawdown": 0., "volatility": 0.})
        self.assertEqual(closes, before)

    def test_invalid_prices_or_too_short_paths_rejected(self):
        for spot, closes in ((0, [1, 1]), (-1, [1, 1]), (100, [0, 100]), (100, [100]),
                             (True, [100, 100]), (100, [float("nan"), 100]), (100, [100, float("inf")])):
            with self.subTest(spot=spot, closes=closes), self.assertRaises(ValueError):
                risk_metrics(spot, closes)


class RankingTests(unittest.TestCase):
    def test_spearman_monotone_and_reversed(self):
        self.assertEqual(rank_scores([1, 2, 3, 4], [1, 2, 3, 4])["spearman"], 1.)
        self.assertEqual(rank_scores([4, 3, 2, 1], [1, 2, 3, 4])["spearman"], -1.)

    def test_spearman_uses_average_tie_ranks(self):
        result = rank_scores([1, 1, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(result["spearman"], sqrt(.9))

    def test_cutoff_ties_have_fractional_expected_recall(self):
        result = rank_scores([5, 4, 4, 4, 1], [5, 4, 3, 2, 1])
        # Predicted weights 1,2/3,2/3,2/3,0; actual weights 1,1,1,0,0.
        self.assertAlmostEqual(result["top3_recall_expected"], 7 / 9)
        self.assertEqual(result["prediction_cutoff_tie_count"], 3)

    def test_flat_predictions_are_chance_not_perfect(self):
        result = rank_scores([0] * 10, list(range(10)))
        self.assertIsNone(result["spearman"])
        self.assertEqual(result["spearman_reason"], "constant_prediction")
        self.assertAlmostEqual(result["top3_recall_expected"], .3)
        self.assertEqual(result["chance_recall"], .3)
        self.assertTrue(result["predicted_constant"])

    def test_constant_truth_and_both_constant_are_undefined(self):
        result = rank_scores([1, 2, 3, 4], [0] * 4)
        self.assertIsNone(result["spearman"])
        self.assertEqual(result["spearman_reason"], "constant_actual")
        self.assertAlmostEqual(result["top3_recall_expected"], .75)
        both = rank_scores([0] * 10, [0] * 10)
        self.assertEqual(both["spearman_reason"], "constant_prediction_and_actual")
        self.assertAlmostEqual(both["top3_recall_expected"], .3)

    def test_joint_permutation_cannot_change_tie_result(self):
        p, a = [5, 4, 4, 4, 1], [5, 4, 3, 2, 1]
        expected = rank_scores(p, a)
        order = [3, 4, 2, 1, 0]
        self.assertEqual(rank_scores([p[i] for i in order], [a[i] for i in order]), expected)

    def test_no_overlap_and_invalid_shapes(self):
        self.assertEqual(rank_scores([6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6])["top3_recall_expected"], 0.)
        for p, a, k in (([1, 2], [1, 2], 3), ([1, 2, 3], [1, 2], 2),
                        ([1, 2, 3], [1, 2, 3], True), ([1, float("nan"), 3], [1, 2, 3], 3)):
            with self.subTest(p=p, a=a, k=k), self.assertRaises(ValueError):
                rank_scores(p, a, k)


class EventScoringTests(unittest.TestCase):
    def test_non_systemic_excludes_every_simultaneous_origin(self):
        actual, prediction = table(), table()
        for asset in ("BTC", "ETH"):
            prediction[asset] = {"max_drawdown": .8, "volatility": .8}
        result = score_event(event(origins=["BTC", "ETH"]), {"kronos_base": prediction}, actual, ASSETS)
        self.assertEqual(result["scored_assets"], ASSETS[2:])
        self.assertEqual(result["scored_asset_count"], 8)
        self.assertEqual(result["methods"]["kronos_base"]["metrics"]["max_drawdown"]["mae"], 0.)

    def test_systemic_keeps_all_declared_assets(self):
        actual, prediction = table(), table()
        prediction["BTC"]["max_drawdown"] += .1
        result = score_event(event(systemic=True), {"kronos_base": prediction}, actual, ASSETS)
        self.assertEqual(result["scored_assets"], ASSETS)
        metric = result["methods"]["kronos_base"]["metrics"]["max_drawdown"]
        self.assertAlmostEqual(metric["mae"], .01)
        self.assertAlmostEqual(metric["mae_bps"], 100.)

    def test_each_metric_scored_separately_and_failure_retained(self):
        prediction = table()
        for values in prediction.values():
            values["max_drawdown"] += .02
            values["volatility"] += .03
        result = score_event(event(), {"kronos_base": prediction, "btc_beta": None}, table(), ASSETS)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["methods"]["btc_beta"]["status"], "failed")
        self.assertAlmostEqual(result["methods"]["kronos_base"]["metrics"]["max_drawdown"]["mae"], .02)
        self.assertAlmostEqual(result["methods"]["kronos_base"]["metrics"]["volatility"]["mae"], .03)
        json.dumps(result, allow_nan=False)

    def test_incomplete_actual_or_prediction_never_silently_scores(self):
        incomplete = table()
        incomplete.pop("LTC")
        result = score_event(event(), {"kronos_base": table()}, incomplete, ASSETS)
        self.assertEqual(result["actual_status"], "invalid")
        self.assertEqual(result["methods"]["kronos_base"]["status"], "failed")
        result = score_event(event(), {"kronos_base": incomplete}, table(), ASSETS)
        self.assertEqual(result["actual_status"], "valid")
        self.assertEqual(result["methods"]["kronos_base"]["status"], "failed")

    def test_ineligible_event_is_retained_without_using_truth(self):
        result = score_event(event(eligible=False), {"kronos_base": None}, None, ASSETS)
        self.assertEqual(result["status"], "ineligible")
        self.assertEqual(result["actual_status"], "not_scored")
        self.assertEqual(result["exclusion_reasons"], ["SYNTHETIC_edge"])

    def test_unconfirmed_edge_keeps_catalog_count_without_invented_classification(self):
        edge = event(eligible=False)
        edge["systemic_flag"] = None
        result = score_event(edge, {"kronos_base": None}, None, ASSETS)
        self.assertEqual(result["status"], "ineligible")
        self.assertIsNone(result["systemic_flag"])
        self.assertEqual(result["scored_assets"], [])
        groups = aggregate_events([result])["groups"]
        self.assertEqual(groups["all"]["catalog_event_count"], 1)
        self.assertEqual(groups["all"]["unclassified_event_count"], 1)
        self.assertEqual(groups["systemic"]["catalog_event_count"], 0)
        self.assertEqual(groups["non_systemic"]["catalog_event_count"], 0)
        edge["eligible"] = True
        with self.assertRaisesRegex(ValueError, "systemic_flag"):
            score_event(edge, {"kronos_base": table()}, table(), ASSETS)


class EventAggregationTests(unittest.TestCase):
    def test_equal_weight_event_means_use_identical_complete_pairs(self):
        first = score_event(event("one", systemic=True), {"a": table(.01), "b": table(.02)}, table(), ASSETS)
        second = score_event(event("two"), {"a": table(.03), "b": table(.04)}, table(), ASSETS)
        failed = score_event(event("three"), {"a": table(.5), "b": None}, table(), ASSETS)
        edge = score_event(event("four", eligible=False), {"a": None, "b": None}, None, ASSETS)
        result = aggregate_events([first, second, failed, edge], methods=["a", "b"])
        overall = result["groups"]["all"]
        self.assertEqual(overall["catalog_event_count"], 4)
        self.assertEqual(overall["eligible_event_count"], 3)
        self.assertEqual(overall["common_paired_event_count"], 2)
        self.assertEqual(overall["common_paired_event_ids"], ["one", "two"])
        self.assertAlmostEqual(overall["scores"]["a"]["max_drawdown"]["mae"], .02)
        self.assertAlmostEqual(overall["scores"]["b"]["max_drawdown"]["mae"], .03)
        self.assertEqual(overall["method_status_counts"]["b"], {"scored": 2, "failed": 1, "ineligible": 1})
        self.assertEqual(result["groups"]["systemic"]["common_paired_event_count"], 1)
        self.assertEqual(result["groups"]["non_systemic"]["common_paired_event_count"], 1)

    def test_flat_method_keeps_spearman_coverage_not_zero(self):
        flat = {asset: {"max_drawdown": 0., "volatility": 0.} for asset in ASSETS}
        scored = score_event(event(systemic=True), {"flat": flat, "good": table()}, table(), ASSETS)
        scores = aggregate_events([scored])["groups"]["all"]["scores"]
        flat_summary = scores["flat"]["volatility"]
        self.assertIsNone(flat_summary["spearman"])
        self.assertEqual(flat_summary["spearman_defined_events"], 0)
        self.assertEqual(flat_summary["spearman_undefined_events"], 1)
        self.assertEqual(flat_summary["spearman_undefined_reasons"], {"constant_prediction": 1})
        self.assertAlmostEqual(flat_summary["top3_recall_expected"], .3)
        self.assertEqual(scores["good"]["volatility"]["spearman_defined_events"], 1)

    def test_all_failed_and_omitted_expected_method_preserve_denominator(self):
        failed = score_event(event(), {method: None for method in METHODS}, table(), ASSETS)
        aggregate = aggregate_events([failed], methods=METHODS)["groups"]["all"]
        self.assertEqual(aggregate["catalog_event_count"], 1)
        self.assertEqual(aggregate["common_paired_event_count"], 0)
        for method in METHODS:
            self.assertIsNone(aggregate["scores"][method]["volatility"]["mae"])
            self.assertEqual(aggregate["method_status_counts"][method], {"failed": 1})
        passed = score_event(event(), {"kronos_base": table()}, table(), ASSETS)
        aggregate = aggregate_events([passed], methods=METHODS)["groups"]["all"]
        self.assertEqual(aggregate["method_status_counts"]["historical_30"], {"missing": 1})
        self.assertEqual(aggregate["common_paired_event_count"], 0)

    def test_duplicate_events_cannot_inflate_sample_count(self):
        scored = score_event(event(), {"a": table()}, table(), ASSETS)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_events([scored, copy.deepcopy(scored)])

    def test_empty_aggregation_is_serializable_without_fake_zero_scores(self):
        result = aggregate_events([], methods=METHODS)
        self.assertEqual(result["groups"]["all"]["catalog_event_count"], 0)
        self.assertIsNone(result["groups"]["all"]["scores"]["kronos_base"]["max_drawdown"]["mae"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
