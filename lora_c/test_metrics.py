"""Synthetic C-line checks. No files, market datasets, or models are opened."""

import copy
import json
import math
import random
import unittest
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev

from forecast_metrics.engine import PAIRING, asset_metrics
from lora_c.metrics import (
    ASSETS, TIERS, PROTOCOL_VERSION, acceptance_gate, major_drawdown_guard,
    merge_rows, score_window, selection_gate, summarize,
)


def _iso(value):
    return value.isoformat().replace("+00:00", "Z")


def _series(closes):
    return {"close": list(closes), "high": list(closes), "low": list(closes)}


def fixture():
    anchor = datetime(2026, 4, 1, 12, tzinfo=timezone.utc)
    histories = {}
    for asset in ASSETS:
        closes = [100.0] * 256
        closes[-31] = 125.0  # Historical baseline is exactly 20%.
        histories[asset] = [{"time": _iso(anchor - timedelta(minutes=255 - i)), "close": close}
                            for i, close in enumerate(closes)]
    window = {"window_id": "synthetic-c-only", "as_of": _iso(anchor),
              "assets": list(ASSETS), "histories": histories,
              "interval_seconds": 60, "source": "synthetic-c-test-only",
              "quote_currency": "USDT"}
    future = {asset: _series([70.0] * 30) for asset in ASSETS}

    def forecast(name, close):
        return {"model_revision": name, "source": window["source"], "quote_currency": "USDT",
                "prediction_run_id": f"synthetic-{name}", "pairing": PAIRING,
                "as_of": window["as_of"], "interval_seconds": 60,
                "times": [_iso(anchor + timedelta(minutes=i)) for i in range(1, 31)],
                "spots": {asset: 100.0 for asset in ASSETS},
                "paths": [{"path_id": "synthetic-path", "assets": {
                    asset: _series([close] * 30) for asset in ASSETS}}]}

    # Truth MDD .30, selected .28, original .25, historical .20.
    return window, future, forecast("epoch_01", 72.0), forecast("original", 75.0)


def _row():
    window, future, selected, original = fixture()
    return score_window(window, future, {"epoch_01": selected, "original": original})


def _grid(row, count=300):
    result = []
    anchor = datetime(2026, 4, 1, 12, tzinfo=timezone.utc)
    for i in range(count):
        item = copy.deepcopy(row)
        timestamp = anchor + timedelta(minutes=i * 120)
        item.update(window_id=f"synthetic-c-origin-{i}", as_of=_iso(timestamp),
                    utc_day=timestamp.date().isoformat())
        result.append(item)
    return result


def _set_report_mae(report, tier, method, value):
    """Change one synthetic mean consistently to isolate gate boundary logic."""
    t = report["tiers"][tier]
    t["methods"][method]["max_drawdown_mae"] = value
    for candidate, references in t["paired"].items():
        for reference, comparison in references.items():
            metric = comparison["max_drawdown_mae_difference"]
            if candidate == method:
                metric["candidate_mean"] = value
            if reference == method:
                metric["reference_mean"] = value
            if metric["candidate_mean"] is not None and metric["reference_mean"] is not None:
                metric["estimate"] = metric["candidate_mean"] - metric["reference_mean"]


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = _row()
        cls.rows = _grid(cls.row)
        cls.report = summarize(cls.rows)

    def test_complete_report_and_both_gates(self):
        report = self.report
        self.assertEqual(report["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(report["selection_metric"], "small.max_drawdown_mae")
        self.assertEqual(report["n_scheduled"], 300)
        for tier, assets in TIERS.items():
            methods = report["tiers"][tier]["methods"]
            for name in ("original", "epoch_01", "historical_30", "dynamic_naive"):
                self.assertEqual(methods[name]["n_scored"], 300)
                self.assertEqual(methods[name]["n_failed"], 0)
                self.assertEqual(methods[name]["coverage"], 1)
            self.assertAlmostEqual(methods["epoch_01"]["max_drawdown_mae"], .02)
            self.assertAlmostEqual(methods["original"]["max_drawdown_mae"], .05)
            self.assertAlmostEqual(methods["historical_30"]["max_drawdown_mae"], .10)
            self.assertEqual(methods["epoch_01"]["n_max_drawdown_mae_asset_origins"], 300 * len(assets))
            self.assertIsNone(methods["dynamic_naive"]["max_drawdown_mae"])
            self.assertIsNone(methods["historical_30"]["volatility_mae"])
            self.assertIsNone(methods["historical_30"]["exact_set_hit_rate"])
            self.assertIsNone(methods["historical_30"]["volatility_top3_recall"])
            self.assertEqual(methods["historical_30"]["n_mae_asset_origins_scheduled"], 0)
            paired = report["tiers"][tier]["paired"]["epoch_01"]["original"]
            self.assertEqual(paired["max_drawdown_mae_difference"]["n_paired"], 300)
            self.assertTrue(paired["max_drawdown_mae_difference"]["complete_grid"])
        self.assertTrue(selection_gate(report, "epoch_01")["passed"])
        self.assertTrue(acceptance_gate(report, "epoch_01")["passed"])
        json.dumps(report, allow_nan=False)

    def test_drawdown_includes_p0_and_uses_closes_not_intrabar_extremes(self):
        window, future, selected, _ = fixture()
        for asset in ASSETS:
            future[asset] = {"close": [80.0] * 30, "high": [200.0] * 30, "low": [10.0] * 30}
            selected["paths"][0]["assets"][asset] = {
                "close": [90.0] * 30, "high": [300.0] * 30, "low": [1.0] * 30}
        row = score_window(window, future, {"epoch_01": selected})
        for tier in TIERS:
            data = row["tiers"][tier]
            self.assertTrue(all(value == .2 for value in data["actual_max_drawdown"].values()))
            item = data["methods"]["epoch_01"]
            self.assertTrue(all(value == .1 for value in item["predicted_max_drawdown"].values()))
            self.assertAlmostEqual(item["max_drawdown_mae"], .1)

    def test_historical_baseline_uses_exactly_last31_and_never_future(self):
        window, future, _, _ = fixture()
        for asset in ASSETS:
            window["histories"][asset][0]["close"] = 10000.0  # Must be ignored.
            window["histories"][asset][-32]["close"] = 1000.0  # Just outside baseline.
        first = score_window(window, future, {})
        for asset in ASSETS:
            future[asset] = _series([99.0] * 30)
        second = score_window(window, future, {})
        for tier in TIERS:
            a = first["tiers"][tier]["methods"]["historical_30"]
            b = second["tiers"][tier]["methods"]["historical_30"]
            self.assertEqual(a["history_close_count"], 31)
            self.assertEqual(a["predicted_max_drawdown"], b["predicted_max_drawdown"])
            self.assertTrue(all(value == .2 for value in a["predicted_max_drawdown"].values()))
            self.assertNotEqual(a["max_drawdown_mae"], b["max_drawdown_mae"])
        for asset in ASSETS:
            window["histories"][asset][-31]["close"] = 100.0
        changed = score_window(window, future, {})
        self.assertTrue(all(value == 0 for value in changed["tiers"]["small"]["methods"]
                            ["historical_30"]["predicted_max_drawdown"].values()))

    def test_historical_running_peak_is_not_endpoint_return(self):
        window, future, _, _ = fixture()
        for asset in ASSETS:
            for bar, price in zip(window["histories"][asset][-31:], [100., 200., 50.] + [100.] * 28):
                bar["close"] = price
        row = score_window(window, future, {})
        self.assertEqual(row["tiers"]["small"]["methods"]["historical_30"]
                         ["predicted_max_drawdown"]["XRP"], .75)

    def test_prediction_averages_pathwise_mdd_and_volatility(self):
        window, future, selected, _ = fixture()
        selected["paths"] = []
        for sign in (1, -1):
            closes = [100 + sign * (10 if i % 2 == 0 else 0) for i in range(30)]
            selected["paths"].append({"path_id": str(sign), "assets": {
                asset: _series(closes) for asset in ASSETS}})
        row = score_window(window, future, {"epoch_01": selected})
        item = row["tiers"]["major"]["methods"]["epoch_01"]
        self.assertAlmostEqual(item["predicted_max_drawdown"]["BTC"], mean([10 / 110, .1]))
        expected = mean(asset_metrics(100, p["assets"]["BTC"]["close"],
                                     p["assets"]["BTC"]["high"], p["assets"]["BTC"]["low"], 1)
                        ["volatility"] for p in selected["paths"])
        self.assertAlmostEqual(item["predicted_volatility"]["BTC"], expected)
        self.assertGreater(expected, 0)
        self.assertEqual(mean(p["assets"]["BTC"]["close"][0] for p in selected["paths"]), 100)

    def test_dynamic_naive_uses_255_returns_without_mdd_path(self):
        window, future, _, _ = fixture()
        row = score_window(window, future, {})
        closes = [bar["close"] for bar in window["histories"]["BTC"]]
        returns = [math.log(b) - math.log(a) for a, b in zip(closes, closes[1:])]
        naive = row["tiers"]["major"]["methods"]["dynamic_naive"]
        self.assertEqual(len(returns), 255)
        self.assertAlmostEqual(naive["predicted_volatility"]["BTC"], pstdev(returns) * math.sqrt(30))
        self.assertIsNone(naive["predicted_max_drawdown"])
        self.assertIsNone(naive["max_drawdown_mae"])

    def test_ties_retain_competition_sets_and_top3_chance(self):
        for tier, assets in TIERS.items():
            item = self.row["tiers"][tier]["methods"]["epoch_01"]
            self.assertTrue(item["exact_set_hit"])
            self.assertEqual(item["selected_count"], len(assets))
            self.assertEqual(item["actual_count"], len(assets))
            self.assertAlmostEqual(item["volatility_top3_recall"], 3 / len(assets))
            summary = self.report["tiers"][tier]["methods"]["epoch_01"]
            self.assertEqual(summary["n_selected_over_two"], 300)
            self.assertEqual(summary["selected_count_histogram"], {len(assets): 300})

    def test_missing_prediction_keeps_full_grid_and_prevents_gates(self):
        window, future, _, original = fixture()
        missing = score_window(window, future, {"epoch_01": None, "original": original})
        rows = copy.deepcopy(self.rows)
        rows[0]["tiers"] = missing["tiers"]
        report = summarize(rows)
        for tier in TIERS:
            candidate = report["tiers"][tier]["methods"]["epoch_01"]
            self.assertEqual((candidate["n_scheduled"], candidate["n_scored"], candidate["n_failed"]), (300, 299, 1))
            paired = report["tiers"][tier]["paired"]["epoch_01"]["original"]
            self.assertEqual(paired["max_drawdown_mae_difference"]["n_paired"], 299)
            self.assertFalse(paired["complete_grid"])
        self.assertFalse(selection_gate(report, "epoch_01")["passed"])
        self.assertFalse(acceptance_gate(report, "epoch_01")["passed"])

    def test_null_or_incomplete_asset_metric_cannot_pass_as_scored(self):
        for defect in ("null", "nonfinite", "missing_asset"):
            with self.subTest(defect=defect):
                rows = copy.deepcopy(self.rows)
                item = rows[0]["tiers"]["small"]["methods"]["epoch_01"]
                if defect == "missing_asset":
                    del item["max_drawdown_absolute_errors"]["LTC"]
                else:
                    item["max_drawdown_mae"] = None if defect == "null" else float("nan")
                report = summarize(rows)
                method = report["tiers"]["small"]["methods"]["epoch_01"]
                self.assertEqual(method["n_scored"], 300)
                self.assertEqual(method["n_max_drawdown_mae_origins"], 299)
                self.assertEqual(method["n_max_drawdown_mae_asset_origins"], 299 * 6)
                self.assertFalse(selection_gate(report, "epoch_01")["passed"])
                self.assertFalse(acceptance_gate(report, "epoch_01")["passed"])
                json.dumps(report, allow_nan=False)

    def test_failed_truth_fails_every_method_without_dropping_origin(self):
        window, future, selected, original = fixture()
        future["LTC"]["close"][0] = None
        row = score_window(window, future, {"epoch_01": selected, "original": original})
        self.assertEqual(row["truth_status"], "failed")
        for tier in TIERS:
            self.assertTrue(all(item["status"] == "failed" for item in row["tiers"][tier]["methods"].values()))
        report = summarize([row], model_names=["epoch_02"])
        self.assertEqual(report["tiers"]["small"]["methods"]["epoch_02"]["n_missing"], 1)
        self.assertFalse(selection_gate(report, "epoch_02")["passed"])

    def test_strict_small_threshold_and_historical_only_in_acceptance(self):
        report = copy.deepcopy(self.report)
        original = report["tiers"]["small"]["methods"]["original"]["max_drawdown_mae"]
        _set_report_mae(report, "small", "epoch_01", original)
        self.assertFalse(selection_gate(report, "epoch_01")["passed"])
        self.assertTrue(selection_gate(report, "epoch_01")["checks"]["small_vs_original"]["evidence_complete"])
        self.assertFalse(acceptance_gate(report, "epoch_01")["passed"])
        report = copy.deepcopy(self.report)
        _set_report_mae(report, "small", "historical_30", .01)
        self.assertTrue(selection_gate(report, "epoch_01")["passed"])
        self.assertFalse(acceptance_gate(report, "epoch_01")["passed"])
        self.assertEqual(list(acceptance_gate(report, "epoch_01")["checks"]),
                         ["small_vs_original", "small_vs_historical_30"])

    def test_major_guard_exact20_boundary_and_above(self):
        report = copy.deepcopy(self.report)
        _set_report_mae(report, "major", "original", .1)
        _set_report_mae(report, "major", "epoch_01", .12)
        result = major_drawdown_guard(report, "epoch_01")
        self.assertTrue(result["passed"])
        self.assertFalse(result["triggered"])
        self.assertAlmostEqual(result["relative_change"], .2)
        _set_report_mae(report, "major", "epoch_01", math.nextafter(.12, math.inf))
        self.assertFalse(major_drawdown_guard(report, "epoch_01")["passed"])
        self.assertFalse(selection_gate(report, "epoch_01")["passed"])
        acceptance = acceptance_gate(report, "epoch_01")
        self.assertTrue(acceptance["passed"])
        self.assertTrue(acceptance["major_drawdown_disclosure"]["triggered"])

    def test_major_zero_reference_and_missing_evidence(self):
        report = copy.deepcopy(self.report)
        _set_report_mae(report, "major", "original", 0.)
        _set_report_mae(report, "major", "epoch_01", 0.)
        result = major_drawdown_guard(report, "epoch_01")
        self.assertTrue(result["passed"])
        self.assertIsNone(result["relative_change"])
        _set_report_mae(report, "major", "epoch_01", 1e-20)
        result = major_drawdown_guard(report, "epoch_01")
        self.assertFalse(result["passed"])
        self.assertTrue(result["triggered"])
        self.assertIsNone(result["relative_change"])
        report["tiers"]["major"]["methods"]["epoch_01"]["n_max_drawdown_mae_origins"] = 299
        result = major_drawdown_guard(report, "epoch_01")
        self.assertFalse(result["passed"])
        self.assertFalse(result["evidence_complete"])
        self.assertFalse(result["triggered"])

    def test_volatility_ranking_and_intervals_cannot_change_decisions(self):
        report = copy.deepcopy(self.report)
        before = (selection_gate(report, "epoch_01"), acceptance_gate(report, "epoch_01"))
        for tier in TIERS:
            item = report["tiers"][tier]["methods"]["epoch_01"]
            item.update(volatility_mae=99., exact_set_hit_rate=0., volatility_top3_recall=0.)
            for pair in report["tiers"][tier]["paired"]["epoch_01"].values():
                pair["max_drawdown_mae_difference"]["ci95"] = [-100., 100.]
                pair["exact_set_hit_rate_difference"]["estimate"] = -1.
        self.assertEqual(before, (selection_gate(report, "epoch_01"), acceptance_gate(report, "epoch_01")))

    def test_required300_and_no_silent_successful_subset(self):
        report = summarize(self.rows[:299])
        self.assertFalse(selection_gate(report, "epoch_01")["passed"])
        self.assertFalse(selection_gate(report, "epoch_01")["checks"]["small_vs_original"]["evidence_complete"])
        self.assertFalse(acceptance_gate(report, "epoch_01")["passed"])
        self.assertFalse(major_drawdown_guard(report, "epoch_01")["passed"])
        self.assertFalse(selection_gate(summarize([]), "epoch_01")["passed"])

    def test_pairing_uses_same_origin_and_bootstraps_days_deterministically(self):
        rows = _grid(self.row, 4)
        # Three origins share one day, the fourth is on the next. Difference
        # means must weight origins, while bootstrap resamples whole UTC days.
        differences = (.1, .1, .1, -.1)
        for index, (row, difference) in enumerate(zip(rows, differences)):
            when = datetime(2026, 4, 1 + (index == 3), 12, index, tzinfo=timezone.utc)
            row.update(as_of=_iso(when), utc_day=when.date().isoformat())
            for tier, assets in TIERS.items():
                for name, error in (("epoch_01", .2 + difference), ("original", .2)):
                    item = row["tiers"][tier]["methods"][name]
                    item["max_drawdown_mae"] = error
                    item["max_drawdown_absolute_errors"] = {asset: error for asset in assets}
        first, second = summarize(rows), summarize(list(reversed(rows)))
        a = first["tiers"]["small"]["paired"]["epoch_01"]["original"]["max_drawdown_mae_difference"]
        b = second["tiers"]["small"]["paired"]["epoch_01"]["original"]["max_drawdown_mae_difference"]
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["estimate"], .05)
        self.assertEqual(a["n_days"], 2)
        rng = random.Random(20260926)
        groups = [(3 * (.3 - .2), 3), (.1 - .2, 1)]
        samples = []
        for _ in range(2000):
            blocks = [groups[rng.randrange(2)] for _ in range(2)]
            samples.append(math.fsum(x[0] for x in blocks) / sum(x[1] for x in blocks))
        samples.sort()
        for actual, expected in zip(a["ci95"], [samples[49] + .975 * (samples[50] - samples[49]),
                                               samples[1949] + .025 * (samples[1950] - samples[1949])]):
            self.assertAlmostEqual(actual, expected)
        one_day = summarize(rows[:1])["tiers"]["small"]["paired"]["epoch_01"]["original"]
        self.assertIsNone(one_day["max_drawdown_mae_difference"]["ci95"])

    def test_merge_cached_models_keeps_union_and_rejects_conflicts(self):
        window, future, selected, original = fixture()
        original_row = score_window(window, future, {"original": original})
        selected_row = score_window(window, future, {"epoch_01": selected})
        merged = merge_rows([original_row], [selected_row])
        self.assertEqual(merged, [self.row])
        second = _grid(selected_row, 2)[1]
        union = merge_rows([original_row], [selected_row, second])
        summary = summarize(union)
        self.assertEqual(summary["n_scheduled"], 2)
        self.assertEqual(summary["tiers"]["small"]["methods"]["original"]["n_missing"], 1)
        selected_row["tiers"]["small"]["actual_max_drawdown"]["XRP"] = .123
        with self.assertRaisesRegex(ValueError, "actual_max_drawdown mismatch"):
            merge_rows([original_row], [selected_row])

    def test_invalid_identity_protocol_and_reserved_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([self.row, self.row])
        earlier = copy.deepcopy(self.row)
        earlier["protocol_version"] = "lora-a-v2.1"
        with self.assertRaisesRegex(ValueError, "C scoring protocol"):
            summarize([earlier])
        wrong_day = copy.deepcopy(self.row)
        wrong_day["utc_day"] = "2026-04-02"
        with self.assertRaisesRegex(ValueError, "UTC day"):
            summarize([wrong_day])
        window, future, selected, _ = fixture()
        for name in ("historical_30", "dynamic_naive", ""):
            with self.assertRaises(ValueError):
                score_window(window, future, {name: selected})


if __name__ == "__main__":
    unittest.main()
