import copy
import json
import math
import unittest
from dataclasses import FrozenInstanceError

from .demo import fixture
from .engine import evaluate_alerts, freeze_forecast, make_holdings, recalculate, report_payload


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.source = fixture()
        self.forecast = freeze_forecast(self.source)
        self.holdings = make_holdings(self.forecast)
        self.metrics = recalculate(self.forecast, self.holdings)

    def test_hand_calculation_and_new_volatility_definition(self):
        row = self.metrics["paths"][0]["assets"]["BTC"]
        self.assertAlmostEqual(row["terminal_return"], .02)
        self.assertAlmostEqual(row["pnl"], 100)  # 50 BTC units in demo portfolio.
        self.assertAlmostEqual(row["max_drawdown"], 3 / 101)
        returns = [math.log(101 / 100), math.log(98 / 101), math.log(102 / 98)]
        center = sum(returns) / 3
        expected = math.sqrt(sum((r - center) ** 2 for r in returns))
        self.assertAlmostEqual(row["volatility"], expected)
        self.assertNotAlmostEqual(row["volatility"], math.sqrt(sum(r * r for r in returns)))
        self.assertEqual(row["predicted_high"], 104)
        self.assertEqual(row["predicted_low"], 97)
        self.assertAlmostEqual(row["amplitude"], .07)

    def test_constant_log_growth_has_zero_volatility(self):
        series = self.source["paths"][0]["assets"]["BTC"]
        series.update(close=[200, 400, 800], high=[200, 400, 800], low=[200, 400, 800])
        metrics = recalculate(freeze_forecast(self.source), self.holdings)
        self.assertAlmostEqual(metrics["paths"][0]["assets"]["BTC"]["volatility"], 0, places=13)

    def test_portfolio_paths_and_contributions(self):
        self.assertEqual(self.holdings.data["quantities"], {"BTC": 50, "ETH": 60, "SOL": 200})
        row = self.metrics["paths"][0]["portfolio"]
        self.assertEqual(row["value_path"], [10000, 10050, 9760, 10220])
        self.assertEqual(row["pnl_contributions"], {"BTC": 100, "ETH": 120, "SOL": 0})
        self.assertEqual(row["pnl"], sum(row["pnl_contributions"].values()))
        self.assertAlmostEqual(row["terminal_return"], .022)
        self.assertAlmostEqual(row["max_drawdown"], 290 / 10050)

    def test_zero_total_pnl_does_not_require_contribution_percentages(self):
        series = self.source["paths"][0]["assets"]["ETH"]
        series.update(close=[50, 50, 50], high=[50, 50, 50], low=[50, 50, 50])
        series = self.source["paths"][0]["assets"]["BTC"]
        series.update(close=[100, 100, 100], high=[100, 100, 100], low=[100, 100, 100])
        result = recalculate(freeze_forecast(self.source), self.holdings)
        self.assertEqual(result["paths"][0]["portfolio"]["pnl"], 0)
        json.dumps(result, allow_nan=False)

    def test_multi_path_metrics_before_aggregation_and_quantiles(self):
        data = self.source
        data["times"] = data["times"][:2]
        data["spots"] = {"BTC": 100}
        data["paths"] = [{"path_id": name, "assets": {"BTC": {
            "close": prices, "high": prices, "low": prices}}}
                         for name, prices in (("up", [110, 100]), ("down", [90, 100]))]
        forecast = freeze_forecast(data)
        metrics = recalculate(forecast, make_holdings(forecast, 100, {"BTC": 1}))
        summary = metrics["summary"]["assets"]["BTC"]
        self.assertGreater(summary["volatility"]["mean"], 0)
        self.assertGreater(summary["max_drawdown"]["mean"], 0)
        self.assertAlmostEqual(summary["predicted_high"]["p05"], 100.5)
        self.assertAlmostEqual(summary["predicted_high"]["p50"], 105)
        self.assertAlmostEqual(summary["predicted_high"]["p95"], 109.5)
        data["paths"] = [{"path_id": "averaged", "assets": {"BTC": {
            "close": [100, 100], "high": [100, 100], "low": [100, 100]}}}]
        averaged = freeze_forecast(data)
        bad = recalculate(averaged, make_holdings(averaged, 100, {"BTC": 1}))
        self.assertEqual(bad["summary"]["assets"]["BTC"]["volatility"]["mean"], 0)

    def test_forecast_is_immutable_and_repeatable(self):
        original_id = self.forecast.identity
        self.source["spots"]["BTC"] = 123
        exported = self.forecast.data
        exported["spots"]["BTC"] = 456
        self.assertEqual(self.forecast.data["spots"]["BTC"], 100)
        self.assertEqual(self.forecast.identity, original_id)
        self.assertEqual(recalculate(self.forecast, self.holdings), self.metrics)
        self.assertEqual(freeze_forecast(fixture()).identity, original_id)
        with self.assertRaises(FrozenInstanceError):
            self.forecast.identity = "changed"

    def test_quantity_change_keeps_forecast_id(self):
        doubled = recalculate(self.forecast, make_holdings(self.forecast, 20000))
        self.assertEqual(doubled["forecast_id"], self.metrics["forecast_id"])
        self.assertNotEqual(doubled["metrics_id"], self.metrics["metrics_id"])
        self.assertEqual(doubled["paths"][0]["portfolio"]["pnl"], 440)
        self.assertEqual(doubled["paths"][0]["assets"]["BTC"]["volatility"],
                         self.metrics["paths"][0]["assets"]["BTC"]["volatility"])

    def test_refresh_keeps_quantities_and_uses_new_spot_basis(self):
        later = fixture()
        later["as_of"] = "2026-09-26T12:01:00Z"
        later["times"] = ["2026-09-26T12:02:00Z", "2026-09-26T12:03:00Z", "2026-09-26T12:04:00Z"]
        later["spots"]["BTC"] = 102
        result = recalculate(freeze_forecast(later), self.holdings)
        self.assertNotEqual(result["forecast_id"], self.metrics["forecast_id"])
        self.assertEqual(result["holdings"], self.holdings.data)
        self.assertEqual(result["current_value"], 10100)
        self.assertEqual(result["paths"][0]["portfolio"]["pnl"], 120)
        self.assertAlmostEqual(result["paths"][0]["portfolio"]["terminal_return"], 120 / 10100)

    def test_threshold_only_reevaluation_and_report_anchor(self):
        before = copy.deepcopy(self.metrics)
        default = evaluate_alerts(self.metrics)
        muted = evaluate_alerts(self.metrics, 1, 0)
        self.assertEqual(before, self.metrics)
        self.assertEqual(default["forecast_id"], muted["forecast_id"])
        self.assertTrue(default["assets"]["SOL"]["needs_review"])
        self.assertFalse(any(r["needs_review"] for r in muted["assets"].values()))
        report = report_payload(self.metrics, default)
        self.assertEqual(report["forecast_id"], self.forecast.identity)
        other = recalculate(self.forecast, make_holdings(self.forecast, 20000))
        with self.assertRaises(ValueError):
            report_payload(other, default)

    def test_strict_threshold_and_competition_ties(self):
        metrics = copy.deepcopy(self.metrics)
        for symbol, value in (("BTC", .1), ("ETH", .1), ("SOL", .05)):
            metrics["summary"]["assets"][symbol]["volatility"]["mean"] = value
            metrics["summary"]["assets"][symbol]["max_drawdown"]["mean"] = .03
        alerts = evaluate_alerts(metrics)
        self.assertEqual(alerts["assets"]["SOL"]["volatility_rank"], 3)
        self.assertFalse(alerts["assets"]["SOL"]["needs_review"])
        metrics["summary"]["assets"]["SOL"]["volatility"]["mean"] = .1
        self.assertTrue(all(r["needs_review"] for r in evaluate_alerts(metrics)["assets"].values()))

    def test_reject_invalid_price_grid_or_pairing(self):
        edits = [lambda d: d["spots"].update(BTC=0),
                 lambda d: d["spots"].update(BTC=True),
                 lambda d: d["spots"].update(BTC=float("nan")),
                 lambda d: d.update(as_of="2026-09-26T12:00:00"),
                 lambda d: d["times"].__setitem__(1, d["times"][0]),
                 lambda d: d.update(pairing="unknown"),
                 lambda d: d.update(paths=[]),
                 lambda d: d["paths"][0]["assets"]["BTC"]["low"].__setitem__(0, 999),
                 lambda d: d["paths"][0]["assets"].pop("SOL"),
                 lambda d: d["paths"].append(copy.deepcopy(d["paths"][0])),
                 lambda d: d.update(times=d["times"][:1])]
        for edit in edits:
            data = fixture()
            edit(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                freeze_forecast(data)

    def test_reject_invalid_holdings_and_alert_policy(self):
        for kwargs in ({"capital": 0}, {"weights": {"BTC": .4, "ETH": .3, "SOL": .2}},
                       {"weights": {"BTC": 1, "ETH": .1, "SOL": -.1}}):
            with self.assertRaises(ValueError):
                make_holdings(self.forecast, **kwargs)
        for kwargs in ({"drawdown_threshold": -1}, {"top_k": True}, {"top_k": -1}):
            with self.assertRaises(ValueError):
                evaluate_alerts(self.metrics, **kwargs)


if __name__ == "__main__":
    unittest.main()
