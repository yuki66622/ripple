"""Synthetic-only checks; this module never opens project market data."""

import copy
import json
import math
import unittest
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev

from forecast_metrics.engine import PAIRING, asset_metrics
from lora_a.metrics import ASSETS, TIERS, gates, merge_rows, score_window, summarize


def _iso(value):
    return value.isoformat().replace("+00:00", "Z")


def _series(amplitude):
    closes = [100 * math.exp(amplitude if i % 2 == 0 else 0) for i in range(30)]
    return {"close": closes, "high": closes[:], "low": closes[:]}


def fixture(day=1, minute=0):
    anchor = datetime(2026, 2, day, 12, minute, tzinfo=timezone.utc)
    # Major benchmark is already correct; small naive/fixed pair is wrong.
    naive = {asset: .001 for asset in ASSETS}
    naive.update(BTC=.02, ETH=.015, AVAX=.02, LINK=.015)
    actual = {asset: .001 for asset in ASSETS}
    actual.update(BTC=.02, ETH=.015, XRP=.025, ADA=.018)
    histories = {}
    for asset in ASSETS:
        histories[asset] = [{
            "time": _iso(anchor - timedelta(minutes=255 - i)),
            "close": 100 * math.exp(naive[asset] if i % 2 == 0 else 0),
        } for i in range(256)]
    window = {"window_id": f"synthetic-{day}-{minute}", "as_of": _iso(anchor),
              "assets": list(ASSETS), "histories": histories,
              "interval_seconds": 60, "source": "synthetic-test-only",
              "quote_currency": "USDT"}
    future = {asset: _series(actual[asset]) for asset in ASSETS}

    def forecast(amplitudes, name):
        return {"model_revision": name, "source": window["source"],
                "quote_currency": "USDT", "prediction_run_id": f"synthetic-{name}-{day}-{minute}",
                "pairing": PAIRING, "as_of": window["as_of"], "interval_seconds": 60,
                "times": [_iso(anchor + timedelta(minutes=i)) for i in range(1, 31)],
                "spots": {asset: 100.0 for asset in ASSETS},
                "paths": [{"path_id": "synthetic-path", "assets": {
                    asset: _series(amplitudes[asset]) for asset in ASSETS}}]}

    return window, future, forecast(actual, "selected"), forecast(naive, "original")


class MetricsTests(unittest.TestCase):
    def test_success_has_tier_specific_gates_and_frozen_fixed_pair(self):
        rows = []
        for day in (1, 2):
            window, future, selected, original = fixture(day)
            row = score_window(window, future, {"selected": selected, "original": original})
            fixed = row["tiers"]["small"]["methods"]["fixed_pair"]
            self.assertEqual(fixed["selected_set"], ["AVAX", "LINK"])
            self.assertIsNone(fixed["volatility_mae"])
            self.assertIsNone(fixed["absolute_errors"])
            rows.append(row)
        report = summarize(rows)
        small = report["tiers"]["small"]
        self.assertEqual(small["methods"]["selected"]["exact_set_hit_rate"], 1)
        self.assertEqual(small["methods"]["selected"]["mean_hit_count"], 2)
        self.assertEqual(small["methods"]["selected"]["volatility_mae"], 0)
        self.assertEqual(small["methods"]["selected"]["n_mae_asset_origins"], 12)
        self.assertEqual(small["methods"]["fixed_pair"]["n_mae_asset_origins"], 0)
        self.assertEqual(small["methods"]["selected"]["max_drawdown_mae"], 0)
        self.assertEqual(small["methods"]["selected"]["n_max_drawdown_mae_asset_origins"], 12)
        self.assertIsNone(small["methods"]["dynamic_naive"]["max_drawdown_mae"])
        self.assertIsNone(small["methods"]["fixed_pair"]["max_drawdown_mae"])
        self.assertIsNone(small["methods"]["fixed_pair"]["volatility_top3_recall"])
        self.assertEqual(small["methods"]["original"]["exact_set_hit_rate"], 0)
        self.assertNotIn("fixed_pair", report["tiers"]["major"]["methods"])
        self.assertTrue(gates(report, "selected")["passed"])
        self.assertEqual(small["paired"]["selected"]["original"]["n_paired"], 2)
        self.assertIsNone(small["paired"]["selected"]["fixed_pair"]["volatility_mae_difference"]["estimate"])
        self.assertEqual(small["paired"]["selected"]["original"]["max_drawdown_mae_difference"]["n_paired"], 2)
        self.assertEqual(small["paired"]["selected"]["original"]["volatility_top3_recall_difference"]["n_paired"], 2)
        self.assertIsNone(small["paired"]["selected"]["dynamic_naive"]["max_drawdown_mae_difference"]["estimate"])
        json.dumps(report, allow_nan=False)

    def test_explicit_january_pair_changes_only_fixed_baseline_score(self):
        window, future, selected, original = fixture()
        predictions = {"selected": selected, "original": original}
        legacy = score_window(window, future, predictions, fixed_pair=("AVAX", "LINK"))
        alternate = score_window(window, future, predictions, fixed_pair=("XRP", "ADA"))
        self.assertEqual(alternate["fixed_pair"], ["ADA", "XRP"])
        self.assertFalse(legacy["tiers"]["small"]["methods"]["fixed_pair"]["exact_set_hit"])
        self.assertTrue(alternate["tiers"]["small"]["methods"]["fixed_pair"]["exact_set_hit"])
        self.assertEqual(legacy["tiers"]["major"], alternate["tiers"]["major"])
        for name in ("dynamic_naive", "original", "selected"):
            self.assertEqual(legacy["tiers"]["small"]["methods"][name],
                             alternate["tiers"]["small"]["methods"][name])
        # Changing future labels cannot cause the frozen pair to be reselected.
        future["DOGE"], future["LTC"] = _series(.2), _series(.15)
        changed = score_window(window, future, predictions, fixed_pair=("XRP", "ADA"))
        self.assertEqual(changed["tiers"]["small"]["methods"]["fixed_pair"]["selected_set"], ["ADA", "XRP"])
        self.assertFalse(changed["tiers"]["small"]["methods"]["fixed_pair"]["exact_set_hit"])
        self.assertEqual(summarize([alternate])["fixed_pair"], ["ADA", "XRP"])

    def test_fixed_pair_must_be_two_distinct_small_assets_and_cannot_vary(self):
        window, future, selected, _ = fixture()
        for pair in (None, (), ("XRP",), ("XRP", "XRP"), ("BTC", "XRP"), ("XRP", "ADA", "LTC"), "XRP,ADA"):
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, "fixed_pair"):
                score_window(window, future, {"selected": selected}, fixed_pair=pair)
        first = score_window(window, future, {}, fixed_pair=("XRP", "ADA"))
        window2, future2, _, _ = fixture(2)
        second = score_window(window2, future2, {}, fixed_pair=("AVAX", "LINK"))
        with self.assertRaisesRegex(ValueError, "fixed_pair differs"):
            summarize([first, second])
        conflicting_cache = score_window(window, future, {}, fixed_pair=("AVAX", "LINK"))
        with self.assertRaisesRegex(ValueError, "fixed_pair mismatch"):
            merge_rows([first], [conflicting_cache])
        altered = copy.deepcopy(first)
        altered["tiers"]["small"]["methods"]["fixed_pair"]["selected_set"] = ["AVAX", "LINK"]
        with self.assertRaisesRegex(ValueError, "metadata"):
            summarize([altered])

    def test_major_minus_two_percentage_points_boundary_is_inclusive(self):
        window, future, selected, original = fixture()
        good = score_window(window, future, {"selected": selected, "original": original})
        selected["paths"][0]["assets"]["SOL"] = _series(.03)
        bad = score_window(window, future, {"selected": selected, "original": original})
        rows = []
        anchor = datetime(2026, 2, 1, 12, tzinfo=timezone.utc)
        for index in range(100):
            row = copy.deepcopy(bad if index < 2 else good)
            timestamp = anchor + timedelta(minutes=index * 30)
            row.update(window_id=f"synthetic-gate-{index}", as_of=_iso(timestamp),
                       utc_day=timestamp.date().isoformat())
            rows.append(row)
        result = gates(summarize(rows), "selected")
        self.assertTrue(result["passed"])
        major = result["checks"]["major_vs_dynamic_naive"]
        self.assertEqual(major["exact_set_hit_rate_difference"], -.02)
        self.assertEqual(major["threshold"], -.02)
        rows[2]["tiers"]["major"] = copy.deepcopy(bad["tiers"]["major"])
        result = gates(summarize(rows), "selected")
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"]["major_vs_dynamic_naive"]["exact_set_hit_rate_difference"], -.03)

    def test_naive_uses_255_returns_scaled_to_forecast_horizon(self):
        window, future, selected, _ = fixture()
        row = score_window(window, future, {"selected": selected})
        closes = [bar["close"] for bar in window["histories"]["BTC"]]
        returns = [math.log(b) - math.log(a) for a, b in zip(closes, closes[1:])]
        expected = pstdev(returns) * math.sqrt(30)
        self.assertEqual(len(returns), 255)
        actual = row["tiers"]["major"]["methods"]["dynamic_naive"]["predicted_volatility"]["BTC"]
        self.assertAlmostEqual(actual, expected)
        self.assertNotAlmostEqual(actual, pstdev(returns) * math.sqrt(255))

    def test_prediction_averages_pathwise_volatility_not_prices(self):
        window, future, selected, _ = fixture()
        selected["paths"] = []
        for sign in (1, -1):
            prices = [100 + sign * (10 if i % 2 == 0 else 0) for i in range(30)]
            selected["paths"].append({"path_id": str(sign), "assets": {
                asset: {"close": prices[:], "high": prices[:], "low": prices[:]}
                for asset in ASSETS}})
        row = score_window(window, future, {"selected": selected})
        predicted = row["tiers"]["major"]["methods"]["selected"]["predicted_volatility"]["BTC"]
        expected = mean(asset_metrics(100, p["assets"]["BTC"]["close"],
                                     p["assets"]["BTC"]["high"],
                                     p["assets"]["BTC"]["low"], 1)["volatility"]
                        for p in selected["paths"])
        self.assertAlmostEqual(predicted, expected)
        self.assertGreater(predicted, 0)
        self.assertEqual(mean(p["assets"]["BTC"]["close"][0] for p in selected["paths"]), 100)
        predicted_mdd = row["tiers"]["major"]["methods"]["selected"]["predicted_max_drawdown"]["BTC"]
        self.assertAlmostEqual(predicted_mdd, mean([10 / 110, .1]))
        self.assertGreater(predicted_mdd, 0)  # Averaging the prices would give zero.

    def test_observed_and_predicted_drawdown_include_p0_and_ignore_intrabar_extremes(self):
        window, future, selected, _ = fixture()
        for asset in ASSETS:
            future[asset] = {"close": [80.0] * 30, "high": [130.0] * 30, "low": [60.0] * 30}
            selected["paths"][0]["assets"][asset] = {
                "close": [90.0] * 30, "high": [140.0] * 30, "low": [50.0] * 30}
        row = score_window(window, future, {"selected": selected})
        for tier in TIERS:
            self.assertAlmostEqual(row["tiers"][tier]["actual_max_drawdown"][TIERS[tier][0]], .2)
            item = row["tiers"][tier]["methods"]["selected"]
            self.assertAlmostEqual(item["predicted_max_drawdown"][TIERS[tier][0]], .1)
            self.assertAlmostEqual(item["max_drawdown_mae"], .1)
            self.assertIsNone(row["tiers"][tier]["methods"]["dynamic_naive"]["max_drawdown_mae"])

    def test_top3_recall_uses_fractional_cutoff_ties_and_chance_when_flat(self):
        window, future, selected, _ = fixture()
        for asset, amplitude in zip(TIERS["small"], (.04, .03, .02, .02, .02, .01)):
            future[asset] = _series(amplitude)
            selected["paths"][0]["assets"][asset] = _series(amplitude)
        row = score_window(window, future, {"selected": selected})
        self.assertAlmostEqual(row["tiers"]["small"]["methods"]["selected"]["volatility_top3_recall"], 7 / 9)
        self.assertTrue(row["tiers"]["small"]["methods"]["selected"]["exact_set_hit"])
        for asset in ASSETS:
            selected["paths"][0]["assets"][asset] = _series(0)
        one_flat = score_window(window, future, {"selected": selected})
        for tier in TIERS:
            self.assertAlmostEqual(one_flat["tiers"][tier]["methods"]["selected"]["volatility_top3_recall"], 3 / len(TIERS[tier]))
        for asset in ASSETS:
            future[asset] = _series(0)
        both_flat = score_window(window, future, {"selected": selected})
        for tier in TIERS:
            item = both_flat["tiers"][tier]["methods"]["selected"]
            self.assertAlmostEqual(item["volatility_top3_recall"], 3 / len(TIERS[tier]))
            self.assertTrue(item["exact_set_hit"])  # Main competition-top2 stays unchanged.

    def test_observation_addendum_cannot_change_frozen_gates(self):
        window, future, selected, original = fixture()
        report = summarize([score_window(window, future, {"selected": selected, "original": original})])
        before = gates(report, "selected")
        self.assertTrue(before["passed"])
        for tier in TIERS:
            item = report["tiers"][tier]["methods"]["selected"]
            item["max_drawdown_mae"] = 1000
            item["volatility_top3_recall"] = 0
            for paired in report["tiers"][tier]["paired"]["selected"].values():
                paired["max_drawdown_mae_difference"]["estimate"] = 1000
                paired["volatility_top3_recall_difference"]["estimate"] = -1
        self.assertEqual(gates(report, "selected"), before)

    def test_ties_retain_larger_sets_and_disclose_counts(self):
        window, future, selected, _ = fixture()
        for asset in ASSETS:
            future[asset] = _series(0)
            selected["paths"][0]["assets"][asset] = _series(0)
        row = score_window(window, future, {"selected": selected})
        self.assertEqual(row["tiers"]["small"]["methods"]["selected"]["hit_count"], 6)
        self.assertEqual(row["tiers"]["major"]["methods"]["selected"]["selected_count"], 4)
        result = summarize([row])["tiers"]["small"]["methods"]["selected"]
        self.assertEqual(result["n_selected_over_two"], 1)
        self.assertEqual(result["n_actual_over_two"], 1)
        self.assertEqual(result["selected_count_histogram"], {6: 1})
        self.assertFalse(row["tiers"]["small"]["methods"]["fixed_pair"]["exact_set_hit"])

    def test_missing_prediction_stays_in_grid_and_prevents_acceptance(self):
        rows = []
        for day in (1, 2):
            window, future, selected, original = fixture(day)
            rows.append(score_window(window, future, {
                "selected": selected if day == 1 else None, "original": original}))
        report = summarize(rows, model_names=["selected", "original", "wholly_missing"])
        selected = report["tiers"]["small"]["methods"]["selected"]
        self.assertEqual((selected["n_scheduled"], selected["n_scored"], selected["n_failed"]), (2, 1, 1))
        self.assertEqual(selected["coverage"], .5)
        missing = report["tiers"]["small"]["methods"]["wholly_missing"]
        self.assertEqual(missing["n_failed"], 2)
        self.assertEqual(missing["n_missing"], 2)
        self.assertFalse(gates(report, "selected")["passed"])

    def test_pairing_uses_common_origins_not_separate_model_averages(self):
        rows = []
        for day in (1, 2, 3):
            window, future, selected, original = fixture(day)
            # selected is correct only day1; original available only day2/3,
            # and correct only day3. Paired selected-original = -1/2, whereas
            # subtracting their standalone rates would yield 1/3 - 1/2.
            rows.append(score_window(window, future, {
                "selected": selected if day == 1 else original,
                "original": None if day == 1 else selected if day == 3 else original}))
        report = summarize(rows)
        pair = report["tiers"]["small"]["paired"]["selected"]["original"]
        self.assertEqual(pair["n_paired"], 2)
        self.assertEqual(pair["exact_set_hit_rate_difference"]["estimate"], -.5)
        self.assertEqual(pair["exact_set_hit_rate_difference"]["candidate_mean"], 0)
        self.assertEqual(pair["exact_set_hit_rate_difference"]["reference_mean"], .5)
        self.assertFalse(pair["complete_grid"])

    def test_bootstrap_reproducible_and_counts_origins_not_assets(self):
        rows = []
        for day, minute in ((1, 0), (1, 30), (2, 0)):
            window, future, selected, original = fixture(day, minute)
            rows.append(score_window(window, future, {
                "selected": selected if day == 1 else original, "original": original}))
        first = summarize(rows)
        second = summarize(list(reversed(rows)))
        self.assertEqual(first, second)
        pair = first["tiers"]["small"]["paired"]["selected"]["original"]
        self.assertEqual(pair["n_paired"], 3)
        self.assertEqual(pair["exact_set_hit_rate_difference"]["n_days"], 2)
        self.assertAlmostEqual(pair["exact_set_hit_rate_difference"]["estimate"], 2 / 3)
        self.assertEqual(pair["exact_set_hit_rate_difference"]["ci95"], [0, 1])

    def test_one_day_has_no_spurious_confidence_interval(self):
        window, future, selected, original = fixture()
        report = summarize([score_window(window, future, {"selected": selected, "original": original})])
        interval = report["tiers"]["small"]["paired"]["selected"]["original"]["exact_set_hit_rate_difference"]
        self.assertIsNone(interval["ci95"])
        self.assertEqual(interval["interval_status"], "insufficient_day_clusters")

    def test_equal_original_does_not_pass_strict_small_gate(self):
        window, future, selected, _ = fixture()
        report = summarize([score_window(window, future, {"selected": selected, "original": selected})])
        result = gates(report, "selected")
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["small_vs_original"]["passed"])
        self.assertTrue(result["checks"]["major_vs_dynamic_naive"]["passed"])

    def test_truth_wrapper_checks_grid_and_identity_and_preserves_failures(self):
        window, future, selected, _ = fixture()
        wrapper = {"window_id": window["window_id"], "as_of": window["as_of"],
                   "times": selected["times"], "assets": future}
        result = score_window(window, wrapper, {"selected": {"forecast": selected}})
        self.assertEqual(result["truth_status"], "scored")
        wrapper["times"] = wrapper["times"][:-1]
        result = score_window(window, wrapper, {"selected": selected})
        self.assertEqual(result["truth_status"], "failed")
        self.assertEqual(summarize([result])["tiers"]["small"]["methods"]["fixed_pair"]["n_failed"], 1)

    def test_forecast_mismatch_is_model_failure_not_silent_pairing(self):
        for field, value in (("source", "other-source"), ("as_of", "2026-02-01T13:00:00Z")):
            window, future, selected, _ = fixture()
            selected[field] = value
            row = score_window(window, future, {"selected": selected})
            self.assertEqual(row["truth_status"], "scored")
            self.assertEqual(row["tiers"]["small"]["methods"]["selected"]["status"], "failed")

    def test_cache_merge_requires_identical_truth_and_keeps_union(self):
        window, future, selected, original = fixture(1)
        selected_row = score_window(window, future, {"selected": selected})
        original_row = score_window(window, future, {"original": original})
        window2, future2, selected2, _ = fixture(2)
        only_selected = score_window(window2, future2, {"selected": selected2})
        combined = merge_rows([selected_row, only_selected], [original_row])
        report = summarize(combined)
        self.assertEqual(report["n_scheduled"], 2)
        self.assertEqual(report["tiers"]["small"]["methods"]["original"]["n_missing"], 1)
        self.assertFalse(gates(report, "selected")["passed"])
        altered = copy.deepcopy(original_row)
        altered["tiers"]["small"]["actual_volatility"]["XRP"] += .01
        with self.assertRaisesRegex(ValueError, "mismatch"):
            merge_rows([selected_row], [altered])

    def test_duplicate_origin_or_bad_day_is_rejected(self):
        window, future, selected, _ = fixture()
        row = score_window(window, future, {"selected": selected})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([row, row])
        different_id = copy.deepcopy(row)
        different_id["window_id"] = "other-id-same-origin"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([row, different_id])
        row["utc_day"] = "2026-02-02"
        with self.assertRaisesRegex(ValueError, "UTC day"):
            summarize([row])

    def test_empty_grid_cannot_pass(self):
        report = summarize([], model_names=["selected", "original"])
        self.assertFalse(gates(report, "selected")["passed"])


if __name__ == "__main__":
    unittest.main()
